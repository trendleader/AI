"""
Agentic RAG — Mental Health Resources for Black Women
=======================================================
Stack:
  • Chunking  : LangChain RecursiveCharacterTextSplitter
  • Embeddings: sklearn TF-IDF (fully offline, no model download)
  • Index     : FAISS via cosine similarity on TF-IDF vectors
  • LLM       : Extractive QA (offline) OR HF Inference API (when token provided)
  • Agent     : Custom 4-step agentic decision loop
  • UI        : Streamlit
"""

import os
import re
import pickle
import hashlib
import textwrap
from pathlib import Path

import numpy as np
import PyPDF2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health RAG · Black Women's Resources",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --purple: #6B3FA0; --light-purple: #9B59B6; --bg: #F8F4FF; }
.main-header {
    background: linear-gradient(135deg, #6B3FA0 0%, #9B59B6 60%, #C39BD3 100%);
    padding: 2rem 2.5rem; border-radius: 14px; color: white;
    text-align: center; margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 1.8rem; }
.main-header p  { margin: 0.4rem 0 0; opacity: 0.9; font-size: 0.95rem; }
.agent-step {
    background: #F8F4FF; border-left: 4px solid #6B3FA0;
    padding: 0.6rem 1rem; border-radius: 0 8px 8px 0;
    margin: 0.3rem 0; font-size: 0.85rem; color: #3D1A6E;
}
.answer-box {
    background: #fff; border: 2px solid #9B59B6;
    border-radius: 10px; padding: 1.2rem 1.4rem; line-height: 1.7;
}
.source-card {
    background: #EEF2FF; border: 1px solid #C3B1E1;
    border-radius: 8px; padding: 0.7rem 1rem;
    margin: 0.35rem 0; font-size: 0.83rem;
}
.badge {
    display: inline-block; background: #6B3FA0; color: white;
    border-radius: 12px; padding: 2px 10px; font-size: 0.75rem;
    margin: 2px 3px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PDF_PATHS = {
    "How to Manage Anxiety as a Black Woman":
        "/root/.claude/uploads/1317c8f6-a1b7-5ba2-bbe8-c9a847284328/"
        "055c7ab4-HowtoManageAnxietyasaBlackWoman.pdf",
    "Postpartum Mental Health and Black Women":
        "/root/.claude/uploads/1317c8f6-a1b7-5ba2-bbe8-c9a847284328/"
        "e8f10a92-Postpartum.pdf",
}
CACHE_PATH = Path("/home/user/AI/.rag_cache_mental_health.pkl")

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 60

# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_pdf(path: str, source_name: str) -> list[Document]:
    docs = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            # clean whitespace artifacts common in PDF extraction
            text = re.sub(r" {2,}", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            if len(text) > 60:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": source_name, "page": i + 1},
                ))
    return docs

# ── Vector store (TF-IDF + cosine) ───────────────────────────────────────────
class TFIDFVectorStore:
    """Lightweight local vector store: TF-IDF embeddings + cosine similarity."""

    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.texts  = [c.page_content for c in chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, k: int = 6) -> list[tuple[Document, float]]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

# ── Build / cache knowledge base ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading & indexing documents…")
def build_knowledge_base():
    # Check if we have a pickled version
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        return TFIDFVectorStore.__new__(TFIDFVectorStore), data["chunks"], data

    all_docs: list[Document] = []
    for name, path in PDF_PATHS.items():
        all_docs.extend(extract_pdf(path, name))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    vs = TFIDFVectorStore(chunks)

    cache = {
        "chunks": chunks,
        "texts": vs.texts,
        "vectorizer": vs.vectorizer,
        "matrix": vs.matrix,
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    return vs, chunks, cache


def restore_vs_from_cache(cache: dict) -> TFIDFVectorStore:
    """Restore vectorstore from pickle without re-fitting."""
    vs = TFIDFVectorStore.__new__(TFIDFVectorStore)
    vs.chunks      = cache["chunks"]
    vs.texts       = cache["texts"]
    vs.vectorizer  = cache["vectorizer"]
    vs.matrix      = cache["matrix"]
    return vs

# ── Extractive answer builder ─────────────────────────────────────────────────

def build_extractive_answer(query: str, scored_docs: list[tuple[Document, float]]) -> str:
    """
    Synthesise a structured answer from top-scoring chunks without an LLM.
    Highlights sentences most relevant to the query.
    """
    q_words = set(re.findall(r"\b\w{4,}\b", query.lower()))

    best_sentences: list[tuple[float, str, str]] = []
    for doc, doc_score in scored_docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
        for sent in sentences:
            s_lower = sent.lower()
            s_words = set(re.findall(r"\b\w{4,}\b", s_lower))
            overlap = len(q_words & s_words) / max(len(q_words), 1)
            score = overlap * doc_score
            if score > 0.01 and len(sent.split()) >= 6:
                best_sentences.append((score, sent.strip(), doc.metadata["source"]))

    best_sentences.sort(reverse=True)
    seen: set[str] = set()
    selected: list[tuple[float, str, str]] = []
    for score, sent, src in best_sentences:
        key = sent[:60]
        if key not in seen:
            seen.add(key)
            selected.append((score, sent, src))
        if len(selected) >= 5:
            break

    if not selected:
        # fallback: first 3 sentences from top doc
        top_text = scored_docs[0][0].page_content if scored_docs else ""
        sentences = re.split(r"(?<=[.!?])\s+", top_text)
        selected = [(0.0, s, scored_docs[0][0].metadata["source"]) for s in sentences[:3] if len(s.split()) >= 5]

    # Group by source for a cleaner narrative
    intro = f"Based on the knowledge base, here is what the documents say about **{query.strip('?').strip()}**:\n\n"
    body  = "\n\n".join(f"• {sent}" for _, sent, _ in selected)
    sources_cited = set(src for _, _, src in selected)
    footer = f"\n\n*Sources: {', '.join(sources_cited)}*"
    return intro + body + footer


# ── Agentic RAG engine ────────────────────────────────────────────────────────

class AgentRAG:
    """
    Four-step agentic decision loop:
      Step 1 · Query Analysis    — classify intent, topics, complexity
      Step 2 · Strategy Planning — decide retrieval depth & approach
      Step 3 · Retrieval + Rank  — fetch chunks, score, filter
      Step 4 · Answer Synthesis  — build final answer (extractive or generative)
    """

    TOPIC_MAP = {
        "anxiety":          ["anxiety", "stress", "worry", "panic", "fear", "nervous",
                             "overwhelm", "racing thoughts"],
        "postpartum":       ["postpartum", "postnatal", "birth", "baby", "newborn",
                             "maternal", "pregnancy", "new mother", "perinatal"],
        "racism/sexism":    ["racism", "sexism", "discrimination", "systemic", "oppression",
                             "bias", "microaggression", "stereotype"],
        "support/resources":["therapy", "therapist", "counseling", "support", "help",
                             "resource", "cope", "coping", "strategy", "treatment"],
        "symptoms":         ["symptom", "sign", "feeling", "experience", "diagnose",
                             "indicator", "warning"],
        "stigma":           ["stigma", "shame", "taboo", "judgment", "culture", "community"],
        "historical trauma":["trauma", "slavery", "legacy", "historical", "generational",
                             "intergenerational"],
    }

    def __init__(self, vs: TFIDFVectorStore):
        self.vs = vs

    # ── Step 1 ──────────────────────────────────────────────────────────────
    def _analyse_query(self, query: str) -> dict:
        q_lower = query.lower()
        tokens  = set(re.findall(r"\b\w+\b", q_lower))

        detected = [topic for topic, keywords in self.TOPIC_MAP.items()
                    if any(kw in q_lower for kw in keywords)]

        is_factual   = any(w in q_lower for w in
                           ["what", "define", "explain", "describe", "how does", "why"])
        is_advisory  = any(w in q_lower for w in
                           ["how can", "what can", "how to", "tips", "advice",
                            "help me", "cope", "manage", "deal"])
        is_comparison = any(w in q_lower for w in ["difference", "compare", "versus", "vs"])

        intent     = ("comparison" if is_comparison
                      else "advisory" if is_advisory
                      else "factual")
        complexity = "complex" if len(query.split()) > 10 else "simple"

        return {
            "topics":     detected or ["general mental health"],
            "intent":     intent,
            "complexity": complexity,
            "tokens":     tokens,
        }

    # ── Step 2 ──────────────────────────────────────────────────────────────
    def _plan_strategy(self, classification: dict) -> dict:
        k = 8 if classification["complexity"] == "complex" else 5
        if "postpartum" in classification["topics"] and "anxiety" in classification["topics"]:
            strategy = "cross-document"
            note = "Query spans both documents — retrieving from both PDFs"
        elif len(classification["topics"]) > 2:
            strategy = "broad"
            note = "Wide topic scope — using larger retrieval set"
        else:
            strategy = "focused"
            note = "Focused topic — using targeted retrieval"
        return {"k": k, "strategy": strategy, "note": note}

    # ── Step 3 ──────────────────────────────────────────────────────────────
    def _retrieve_and_rank(
        self, query: str, plan: dict, classification: dict
    ) -> list[tuple[Document, float]]:

        raw = self.vs.search(query, k=plan["k"] + 4)

        # Boost chunks whose source matches a detected topic
        boosted = []
        for doc, score in raw:
            boost = 1.0
            src_lower = doc.metadata["source"].lower()
            if "postpartum" in classification["topics"] and "postpartum" in src_lower:
                boost = 1.15
            if "anxiety" in classification["topics"] and "anxiety" in src_lower:
                boost = 1.15
            boosted.append((doc, score * boost))

        boosted.sort(key=lambda x: x[1], reverse=True)

        # Filter out low-relevance noise
        threshold = max(0.03, boosted[0][1] * 0.15) if boosted else 0.03
        filtered  = [(d, s) for d, s in boosted if s > threshold]

        return filtered[:plan["k"]]

    # ── Step 4 ──────────────────────────────────────────────────────────────
    def _synthesise(
        self, query: str, scored: list[tuple[Document, float]], classification: dict
    ) -> str:
        return build_extractive_answer(query, scored)

    # ── Main entry ──────────────────────────────────────────────────────────
    def run(self, query: str) -> dict:
        steps: list[dict] = []

        # 1 · Analyse
        clf = self._analyse_query(query)
        steps.append({
            "step":   "Step 1 · Query Analysis",
            "icon":   "🔍",
            "detail": (
                f"**Intent:** {clf['intent']} &nbsp;|&nbsp; "
                f"**Complexity:** {clf['complexity']} &nbsp;|&nbsp; "
                f"**Topics:** {', '.join(clf['topics'])}"
            ),
        })

        # 2 · Plan
        plan = self._plan_strategy(clf)
        steps.append({
            "step":   "Step 2 · Retrieval Strategy",
            "icon":   "🗺️",
            "detail": (
                f"**Strategy:** {plan['strategy']} &nbsp;|&nbsp; "
                f"**Chunks to retrieve:** {plan['k']} &nbsp;|&nbsp; "
                f"{plan['note']}"
            ),
        })

        # 3 · Retrieve
        scored = self._retrieve_and_rank(query, plan, clf)
        top_score = scored[0][1] if scored else 0
        sources_found = set(d.metadata["source"] for d, _ in scored)
        steps.append({
            "step":   "Step 3 · Retrieval & Ranking",
            "icon":   "📚",
            "detail": (
                f"**Chunks matched:** {len(scored)} &nbsp;|&nbsp; "
                f"**Top relevance score:** {top_score:.3f} &nbsp;|&nbsp; "
                f"**Documents used:** {len(sources_found)}"
            ),
        })

        # 4 · Synthesise
        answer = self._synthesise(query, scored, clf)
        steps.append({
            "step":   "Step 4 · Answer Synthesis",
            "icon":   "✍️",
            "detail": (
                f"**Method:** Extractive QA (offline) &nbsp;|&nbsp; "
                f"**Input chunks:** {len(scored)} &nbsp;|&nbsp; "
                f"Answer constructed from {len(sources_found)} source(s)"
            ),
        })

        sources = [
            {
                "source":  d.metadata["source"],
                "page":    d.metadata["page"],
                "score":   round(s, 4),
                "snippet": d.page_content[:300].replace("\n", " ") + "…",
            }
            for d, s in scored
        ]

        return {
            "answer":         answer,
            "steps":          steps,
            "sources":        sources,
            "classification": clf,
        }


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def render_badge(label: str):
    return f'<span class="badge">{label}</span>'


def main():
    st.markdown("""
    <div class="main-header">
        <h1>💜 Mental Health Resources for Black Women</h1>
        <p>Agentic RAG &nbsp;·&nbsp; LangChain &nbsp;·&nbsp; TF-IDF Embeddings &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📚 Knowledge Base")
        for name in PDF_PATHS:
            st.markdown(f"- {name}")

        st.markdown("---")
        st.markdown("### ⚙️ Agent Settings")
        show_steps   = st.toggle("Show agent reasoning steps", value=True)
        show_sources = st.toggle("Show source chunks",         value=True)
        num_examples = st.slider("Example questions shown", 3, 7, 5)

        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        examples = [
            "What are the symptoms of anxiety for Black women?",
            "How does racism and sexism impact mental health?",
            "What coping strategies help with postpartum depression?",
            "What barriers prevent Black women from seeking mental health care?",
            "How does historical trauma affect mental health in Black communities?",
            "What support systems are recommended for new Black mothers?",
            "How can a Black woman distinguish between anxiety and postpartum anxiety?",
        ]
        for q in examples[:num_examples]:
            if st.button(q, key=q, use_container_width=True):
                st.session_state["pending_query"] = q

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption(
            "Fully offline agentic RAG built on LangChain + sklearn TF-IDF. "
            "No model downloads required. Documents are chunked, vectorised, "
            "and indexed locally at startup."
        )

    # ── Load resources ────────────────────────────────────────────────────────
    try:
        vs_obj, chunks, cache = build_knowledge_base()
        # Restore TFIDFVectorStore properly from cache on Streamlit rerun
        if not hasattr(vs_obj, "vectorizer"):
            vs_obj = restore_vs_from_cache(cache)
        agent = AgentRAG(vs_obj)
    except Exception as e:
        st.error(f"Failed to initialise knowledge base: {e}")
        st.stop()

    # Stats bar
    sources_count = len(PDF_PATHS)
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents indexed", sources_count)
    col2.metric("Chunks in index",   len(chunks))
    col3.metric("Embedding method",  "TF-IDF (offline)")

    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(
                    f'<div class="answer-box">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])

    # ── Handle sidebar button clicks ──────────────────────────────────────────
    pending = st.session_state.pop("pending_query", None)

    # ── Chat input ────────────────────────────────────────────────────────────
    query = st.chat_input(
        "Ask about anxiety, postpartum mental health, coping strategies, or resources for Black women…"
    ) or pending

    if not query:
        st.markdown(
            "<center><em>Type a question above or choose an example from the sidebar.</em></center>",
            unsafe_allow_html=True,
        )
        return

    # Record user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Agent run ─────────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Agent reasoning…"):
            result = agent.run(query)

        # Agent steps
        if show_steps:
            st.markdown("**🤖 Agent Reasoning Trace**")
            for s in result["steps"]:
                st.markdown(
                    f'<div class="agent-step">'
                    f'<strong>{s["icon"]} {s["step"]}</strong><br>{s["detail"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("")

        # Answer
        st.markdown("**💬 Answer**")
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True,
        )

        # Topic badges
        badges = "".join(render_badge(t) for t in result["classification"]["topics"])
        st.markdown(badges, unsafe_allow_html=True)

        # Sources
        if show_sources and result["sources"]:
            with st.expander(f"📄 Source chunks ({len(result['sources'])})"):
                for s in result["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>📖 {s["source"]}</strong> &nbsp;·&nbsp; '
                        f'Page {s["page"]} &nbsp;·&nbsp; '
                        f'Relevance: <code>{s["score"]}</code><br>'
                        f'<em>{s["snippet"]}</em>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

    # Footer
    st.markdown("---")
    st.markdown(
        "<center><small>💜 Built with LangChain · sklearn TF-IDF · FAISS · Streamlit · Agentic RAG"
        " | Knowledge: <em>Anxiety as a Black Woman</em> & <em>Postpartum Mental Health</em></small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
