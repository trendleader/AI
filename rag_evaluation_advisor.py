"""
RAG Evaluation Advisor - Streamlit Prototype
Install: pip install streamlit anthropic pdfplumber python-docx pandas
"""

import streamlit as st
import anthropic
import io
import os

# Optional imports - handle gracefully
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    import pandas as pd
    CSV_SUPPORT = True
except ImportError:
    CSV_SUPPORT = False


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    content = uploaded_file.read()

    if name.endswith(".pdf"):
        if not PDF_SUPPORT:
            return "[PDF support unavailable — install pdfplumber]"
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)

    if name.endswith(".docx"):
        if not DOCX_SUPPORT:
            return "[DOCX support unavailable — install python-docx]"
        doc = DocxDocument(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)

    if name.endswith(".csv"):
        if not CSV_SUPPORT:
            return content.decode("utf-8", errors="replace")
        df = pd.read_csv(io.BytesIO(content))
        return df.to_string(index=False)

    # Plain text, markdown, code files, etc.
    return content.decode("utf-8", errors="replace")


def build_analysis_prompt(use_case: str, file_summaries: list[dict]) -> str:
    files_block = ""
    for i, f in enumerate(file_summaries, 1):
        preview = f["text"][:3000]
        files_block += f"\n\n--- File {i}: {f['name']} ({f['size_kb']:.1f} KB) ---\n{preview}"
        if len(f["text"]) > 3000:
            files_block += f"\n[...truncated, total chars: {len(f['text'])}]"

    return f"""You are an expert in Retrieval-Augmented Generation (RAG) system design.

The user wants to build a RAG system for the following use case:
{use_case}

They have provided the following documents/artifacts for ingestion:{files_block}

Based on this content and use case, provide a detailed, tailored analysis and recommendations covering:

## 1. Document Analysis
- What types of content are present (structured, unstructured, code, tables, etc.)
- Content characteristics relevant to chunking (average section length, headers, formatting)
- Estimated corpus size and complexity

## 2. Chunking Strategy
- Recommended chunking method (fixed-size, recursive character, semantic/sentence-based, markdown-aware, etc.)
- Recommended chunk size (in tokens or characters) with justification
- Recommended overlap size
- Any special considerations (e.g., preserve tables, code blocks, headers)

## 3. Embedding Strategy
- Recommended embedding model(s) with specific model names
- Justification based on content type, language, domain
- Dimensionality considerations
- Whether to use a single model or multiple (e.g., sparse + dense hybrid)

## 4. Vector Store Strategy
- Recommended vector database(s) (Chroma, Pinecone, Weaviate, FAISS, Qdrant, Milvus, pgvector, etc.)
- Justification (scale, filtering needs, hosting, latency)
- Index type recommendation (HNSW, IVF, flat, etc.)
- Metadata fields worth storing for filtering

## 5. Retrieval Configuration
- Recommended top-k value
- Whether to use hybrid search (BM25 + vector)
- Re-ranking recommendations
- Query expansion or HyDE considerations

## 6. Additional Recommendations
- Any preprocessing steps (OCR, table extraction, code parsing)
- Context window management strategy
- Evaluation metrics to track

Be specific and opinionated. Provide concrete model names, parameter values, and library suggestions."""


def stream_recommendations(prompt: str):
    client = anthropic.Anthropic()
    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=4000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Evaluation Advisor",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Evaluation Advisor")
st.caption("Upload your documents and get tailored recommendations for chunking, embedding, and vector store strategy.")

# Sidebar — use case description
with st.sidebar:
    st.header("About Your Use Case")
    use_case = st.text_area(
        "Describe what you want your RAG system to do",
        placeholder=(
            "e.g., 'A customer support chatbot that answers questions about our product "
            "documentation and troubleshooting guides. Users will ask natural language questions "
            "and expect precise, grounded answers.'"
        ),
        height=160,
    )

    st.markdown("---")
    st.markdown("**Supported file types**")
    st.markdown("PDF, DOCX, TXT, MD, CSV, PY, JS, TS, JSON, YAML, and other text formats")
    st.markdown("---")
    st.markdown("**Requirements**")
    st.code("pip install streamlit anthropic\npdfplumber python-docx pandas", language="bash")

    if not PDF_SUPPORT:
        st.warning("pdfplumber not installed — PDF files won't be parsed.")
    if not DOCX_SUPPORT:
        st.warning("python-docx not installed — DOCX files won't be parsed.")

# Main area — file upload
st.subheader("Upload Documents & Artifacts")
uploaded_files = st.file_uploader(
    "Drop files here or click to browse",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "md", "csv", "py", "js", "ts", "jsx", "tsx",
          "json", "yaml", "yml", "html", "xml", "rst"],
)

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) uploaded**")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, f in enumerate(uploaded_files):
        with cols[i % 4]:
            st.metric(f.name, f"{f.size / 1024:.1f} KB")

# Analyze button
st.markdown("---")
analyze_btn = st.button(
    "Analyze & Get Recommendations",
    type="primary",
    disabled=(not uploaded_files or not use_case.strip()),
    help="Upload at least one file and describe your use case to proceed.",
)

if not use_case.strip() and uploaded_files:
    st.info("Please describe your use case in the sidebar before analyzing.")

if analyze_btn:
    with st.spinner("Extracting text from files..."):
        file_summaries = []
        errors = []
        for f in uploaded_files:
            try:
                text = extract_text(f)
                file_summaries.append({
                    "name": f.name,
                    "size_kb": f.size / 1024,
                    "text": text,
                })
            except Exception as e:
                errors.append(f"{f.name}: {e}")

    if errors:
        for err in errors:
            st.error(f"Extraction error — {err}")

    if file_summaries:
        prompt = build_analysis_prompt(use_case, file_summaries)

        st.subheader("RAG Strategy Recommendations")
        output_container = st.empty()
        full_text = ""

        try:
            for chunk in stream_recommendations(prompt):
                full_text += chunk
                output_container.markdown(full_text)
        except anthropic.APIConnectionError:
            st.error("Could not connect to Anthropic API. Check your ANTHROPIC_API_KEY environment variable.")
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Set ANTHROPIC_API_KEY in your environment.")
        except Exception as e:
            st.error(f"API error: {e}")

        if full_text:
            st.download_button(
                "Download Recommendations (Markdown)",
                data=full_text,
                file_name="rag_recommendations.md",
                mime="text/markdown",
            )
