"""Multimodal RAG prototype for call-center agents.

Ingests PDFs, Word docs, text files, and call recordings (audio, transcribed
locally with faster-whisper) into a shared vector store, then answers agent
questions or drafts customer responses grounded in that knowledge — with
inline citations back to the source document/page or call timestamp.
"""
import os

import streamlit as st
from dotenv import load_dotenv

from ingestion import (
    SUPPORTED_AUDIO_EXTS,
    SUPPORTED_DOC_EXTS,
    chunk_documents,
    load_text_document,
    transcribe_audio,
)
from rag_engine import (
    build_context_block,
    collection_size,
    get_call_transcript,
    get_embeddings,
    get_vector_store,
    get_whisper_model,
    list_audio_sources,
    make_client,
    retrieve,
    stream_answer,
    stream_call_summary,
)

load_dotenv()

MODEL_OPTIONS = ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Anthropic API Key",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Reads ANTHROPIC_API_KEY from the environment if left blank.",
        )
        model = st.selectbox("Model", MODEL_OPTIONS, index=0)
        deep_reasoning = st.toggle(
            "Deep reasoning mode",
            value=False,
            help="Enables adaptive thinking + high effort. Slower, but better for "
            "tricky policy questions. Leave off for fast live-call responses.",
        )

        st.divider()
        st.subheader("Retrieval Settings")
        top_k = st.slider("Chunks to retrieve (k)", 1, 10, 4)
        chunk_size = st.slider("Chunk size (documents)", 200, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, 50)

        st.divider()
        st.subheader("📄 Knowledge Base Documents")
        st.caption("PDF, Word, or text policy docs, FAQs, product manuals.")
        doc_files = st.file_uploader(
            "Upload documents",
            type=[e.lstrip(".") for e in SUPPORTED_DOC_EXTS],
            accept_multiple_files=True,
            key="doc_uploader",
        )
        if st.button("Ingest Documents", disabled=not doc_files):
            _ingest_documents(doc_files, chunk_size, chunk_overlap)

        st.divider()
        st.subheader("🎧 Call Recordings")
        st.caption("Transcribed locally with Whisper — audio is discarded after ingestion.")
        whisper_size = st.selectbox("Whisper model size", ["tiny", "base", "small"], index=1)
        audio_files = st.file_uploader(
            "Upload call recordings",
            type=[e.lstrip(".") for e in SUPPORTED_AUDIO_EXTS],
            accept_multiple_files=True,
            key="audio_uploader",
        )
        if st.button("Transcribe & Ingest", disabled=not audio_files):
            _ingest_audio(audio_files, whisper_size)

        st.divider()
        st.subheader("📚 Knowledge Base Stats")
        embeddings = get_embeddings()
        vector_store = get_vector_store(embeddings)
        col1, col2 = st.columns(2)
        col1.metric("Document chunks", collection_size(vector_store, "document"))
        col2.metric("Call chunks", collection_size(vector_store, "audio"))

        if st.button("🗑️ Clear knowledge base", type="secondary"):
            vector_store.delete_collection()
            get_vector_store.clear()
            st.session_state.pop("ingested_sources", None)
            st.success("Knowledge base cleared.")
            st.rerun()

        return api_key, model, deep_reasoning, top_k, vector_store


def _ingest_documents(doc_files, chunk_size, chunk_overlap):
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    ingested = st.session_state.setdefault("ingested_sources", set())
    new_chunks = 0
    for f in doc_files:
        if f.name in ingested:
            continue
        with st.spinner(f"Ingesting {f.name}…"):
            docs = load_text_document(f)
            chunks = chunk_documents(docs, chunk_size, chunk_overlap)
            if chunks:
                vector_store.add_documents(chunks)
                new_chunks += len(chunks)
        ingested.add(f.name)
    st.success(f"Added {new_chunks} chunks.") if new_chunks else st.info("No new documents.")


def _ingest_audio(audio_files, whisper_size):
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    whisper_model = get_whisper_model(whisper_size)
    ingested = st.session_state.setdefault("ingested_sources", set())
    new_chunks = 0
    for f in audio_files:
        if f.name in ingested:
            continue
        with st.spinner(f"Transcribing {f.name}…"):
            docs = transcribe_audio(f, whisper_model)
            if docs:
                vector_store.add_documents(docs)
                new_chunks += len(docs)
        ingested.add(f.name)
    st.success(f"Transcribed and added {new_chunks} chunks.") if new_chunks else st.info(
        "No new recordings."
    )


def render_chat_tab(api_key, model, deep_reasoning, top_k, vector_store):
    mode_label = st.radio(
        "Answer mode",
        ["Answer question (Q&A)", "Draft customer response"],
        horizontal=True,
    )
    mode = "qa" if mode_label.startswith("Answer") else "response"

    col1, col2 = st.columns(2)
    search_docs = col1.checkbox("Search knowledge base documents", value=True)
    search_audio = col2.checkbox("Search call transcripts", value=True)
    media_types = [t for t, on in [("document", search_docs), ("audio", search_audio)] if on]

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for label, _text in msg["sources"]:
                        st.markdown(label)

    query = st.chat_input("Ask a question or paste the customer's question…")
    if not query:
        return

    if not api_key:
        st.error("Enter your Anthropic API key in the sidebar.")
        st.stop()
    if not media_types:
        st.warning("Select at least one source to search.")
        st.stop()
    if collection_size(vector_store) == 0:
        st.warning("Knowledge base is empty. Ingest documents or call recordings first.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        docs = retrieve(vector_store, query, top_k, media_types)
        context_blocks = build_context_block(docs)
        client = make_client(api_key)
        answer = st.write_stream(
            stream_answer(client, model, mode, context_blocks, query, deep_reasoning)
        )
        if context_blocks:
            with st.expander("Sources"):
                for label, _text in context_blocks:
                    st.markdown(label)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": context_blocks}
    )


def render_summary_tab(api_key, model, vector_store):
    st.caption("Summarize an ingested call recording for a supervisor handoff or QA review.")
    sources = list_audio_sources(vector_store)
    if not sources:
        st.info("No call recordings ingested yet.")
        return

    selected = st.selectbox("Call recording", sources)
    if st.button("Summarize call"):
        if not api_key:
            st.error("Enter your Anthropic API key in the sidebar.")
            st.stop()
        transcript = get_call_transcript(vector_store, selected)
        client = make_client(api_key)
        with st.spinner("Summarizing…"):
            st.write_stream(stream_call_summary(client, model, transcript))
        with st.expander("Full transcript"):
            st.text(transcript)


def main():
    st.set_page_config(page_title="Call Center RAG Assistant", page_icon="🎧", layout="wide")
    st.title("🎧 Call Center Multimodal RAG Assistant")
    st.caption(
        "Ground answers in your policy docs, manuals, and past call recordings — "
        "with citations agents can trust on a live call."
    )

    api_key, model, deep_reasoning, top_k, vector_store = render_sidebar()

    chat_tab, summary_tab = st.tabs(["💬 Agent Assist", "📞 Call Summarizer"])
    with chat_tab:
        render_chat_tab(api_key, model, deep_reasoning, top_k, vector_store)
    with summary_tab:
        render_summary_tab(api_key, model, vector_store)


if __name__ == "__main__":
    main()
