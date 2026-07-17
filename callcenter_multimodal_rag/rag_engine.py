"""Cached resources (embeddings, vector store, Whisper) and Claude-backed
retrieval/answering for the call-center RAG prototype."""
import anthropic
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_callcenter_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPTS = {
    "qa": (
        "You are an assistant helping a live call-center agent answer customer "
        "questions accurately and quickly. Answer only from the provided context. "
        "If the context does not contain the answer, say so plainly instead of "
        "guessing — never invent a policy or number. Cite sources inline using "
        "bracketed numbers like [1] that match the numbered context blocks. Keep "
        "answers concise and scannable: the agent is reading this while on a live "
        "call."
    ),
    "response": (
        "You are helping a live call-center agent respond to a customer. Using "
        "only the provided context, draft a short, empathetic response the agent "
        "can read or paraphrase to the customer. Flag any policy caveats, required "
        "disclosures, or missing information the agent should double-check. Cite "
        "sources inline using bracketed numbers like [1]."
    ),
}

SUMMARY_SYSTEM_PROMPT = (
    "You are summarizing a call-center recording transcript for a supervisor. "
    "Produce: a one-paragraph summary, the customer's overall sentiment, any "
    "commitments or promises made to the customer, and a short bulleted list of "
    "follow-up action items. Base everything strictly on the transcript provided."
)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner="Opening vector store…")
def get_vector_store(_embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)


@st.cache_resource(show_spinner="Loading Whisper model (first run downloads weights)…")
def get_whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def collection_size(vector_store: Chroma, media_type: str | None = None) -> int:
    try:
        if media_type is None:
            return vector_store._collection.count()
        return len(vector_store.get(where={"media_type": media_type})["ids"])
    except Exception:
        return 0


def list_audio_sources(vector_store: Chroma) -> list[str]:
    try:
        metas = vector_store.get(where={"media_type": "audio"})["metadatas"]
    except Exception:
        return []
    return sorted({m["source"] for m in metas})


def get_call_transcript(vector_store: Chroma, source: str) -> str:
    """Reassemble the full transcript for one ingested call recording, ordered
    by timestamp, from its stored chunks."""
    result = vector_store.get(
        where={"$and": [{"media_type": "audio"}, {"source": source}]},
    )
    pairs = sorted(
        zip(result["metadatas"], result["documents"]), key=lambda p: p[0].get("start", 0)
    )
    return "\n".join(text for _meta, text in pairs)


def retrieve(vector_store: Chroma, query: str, k: int, media_types: list[str]):
    filt = {"media_type": {"$in": media_types}} if len(media_types) == 1 else None
    return vector_store.similarity_search(query, k=k, filter=filt)


def build_context_block(docs) -> list[str]:
    """Return one human-readable, numbered citation label + text per doc."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        if meta.get("media_type") == "audio":
            label = f"[{i}] 🎧 {meta.get('source')} @ {meta.get('timestamp')}"
        else:
            page = meta.get("page")
            label = f"[{i}] 📄 {meta.get('source')}" + (
                f", page {page + 1}" if page is not None else ""
            )
        blocks.append((label, doc.page_content))
    return blocks


def make_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def stream_answer(
    client: anthropic.Anthropic,
    model: str,
    mode: str,
    context_blocks: list[tuple],
    question: str,
    deep_reasoning: bool,
):
    context_text = "\n\n".join(f"{label}\n{text}" for label, text in context_blocks)
    user_content = (
        f"Context:\n{context_text}\n\nQuestion: {question}" if context_text else question
    )

    kwargs = {}
    if deep_reasoning:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": "high"}

    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPTS[mode],
        messages=[{"role": "user", "content": user_content}],
        **kwargs,
    ) as stream:
        yield from stream.text_stream


def stream_call_summary(client: anthropic.Anthropic, model: str, transcript: str):
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Transcript:\n{transcript}"}],
    ) as stream:
        yield from stream.text_stream
