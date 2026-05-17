import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
PERSIST_DIR = "./chroma_rag_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_REPO = "mistralai/Mistral-7B-Instruct-v0.2"
SUPPORTED_TYPES = {"pdf": ".pdf", "txt": ".txt", "docx": ".docx"}

# ── Cached resources (survive reruns, rebuilt only when params change) ────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner="Opening vector store…")
def get_vector_db(_embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)


# ── Document helpers ──────────────────────────────────────────────────────────
def _load_uploaded_file(uploaded_file) -> list:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        if ext == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
        elif ext == ".txt":
            docs = TextLoader(tmp_path, encoding="utf-8").load()
        elif ext in (".docx", ".doc"):
            docs = UnstructuredWordDocumentLoader(tmp_path).load()
        else:
            docs = []
        # Tag each chunk with the original filename so sources are readable
        for d in docs:
            d.metadata["source"] = uploaded_file.name
    finally:
        os.unlink(tmp_path)
    return docs


def ingest_file(vector_db: Chroma, uploaded_file, chunk_size: int, chunk_overlap: int) -> int:
    docs = _load_uploaded_file(uploaded_file)
    if not docs:
        return 0
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    vector_db.add_documents(chunks)
    return len(chunks)


def collection_size(vector_db: Chroma) -> int:
    try:
        return vector_db._collection.count()
    except Exception:
        return 0


# ── QA chain builder ──────────────────────────────────────────────────────────
def build_qa_chain(
    vector_db: Chroma,
    hf_token: str,
    temperature: float,
    max_new_tokens: int,
    top_k: int,
) -> RetrievalQA:
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO,
        task="text-generation",
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=hf_token,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )


# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="RAG Chat", page_icon="🔍", layout="wide")
    st.title("🔍 RAG Document Chat")
    st.caption("Upload documents, then ask questions about them.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        hf_token = st.text_input(
            "HuggingFace API Token",
            value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
            type="password",
            help="Required to call the Mistral endpoint.",
        )

        st.divider()
        st.subheader("Retrieval Settings")
        top_k = st.slider("Chunks to retrieve (k)", 1, 10, 3)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        max_new_tokens = st.slider("Max new tokens", 64, 1024, 512, 64)

        st.divider()
        st.subheader("Chunking Settings")
        chunk_size = st.slider("Chunk size", 200, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)

        st.divider()
        st.subheader("📄 Add Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or DOCX files",
            type=["pdf", "txt", "docx", "doc"],
            accept_multiple_files=True,
        )

        if st.button("Ingest Documents", disabled=not uploaded_files):
            embeddings = get_embeddings()
            vector_db = get_vector_db(embeddings)
            already_ingested = st.session_state.get("ingested_files", set())
            new_chunks = 0
            for f in uploaded_files:
                if f.name not in already_ingested:
                    with st.spinner(f"Ingesting {f.name}…"):
                        n = ingest_file(vector_db, f, chunk_size, chunk_overlap)
                    new_chunks += n
                    already_ingested.add(f.name)
            st.session_state["ingested_files"] = already_ingested
            # Invalidate the cached QA chain so it uses the updated DB
            st.session_state.pop("qa_chain", None)
            if new_chunks:
                st.success(f"Added {new_chunks} chunks.")
            else:
                st.info("No new documents to ingest.")

        st.divider()
        st.subheader("📚 Indexed Documents")
        embeddings = get_embeddings()
        vector_db = get_vector_db(embeddings)
        count = collection_size(vector_db)
        st.metric("Total chunks in store", count)
        ingested = st.session_state.get("ingested_files", set())
        if ingested:
            for name in sorted(ingested):
                st.write(f"• {name}")

        if st.button("🗑️ Clear vector store", type="secondary"):
            vector_db.delete_collection()
            st.session_state.pop("qa_chain", None)
            st.session_state.pop("ingested_files", None)
            get_vector_db.clear()
            st.success("Vector store cleared.")
            st.rerun()

        if st.button("🧹 Clear chat history"):
            st.session_state["messages"] = []
            st.rerun()

    # ── Main chat area ────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(src)

    query = st.chat_input("Ask a question about your documents…")

    if query:
        if not hf_token:
            st.error("Please enter your HuggingFace API token in the sidebar.")
            st.stop()

        if collection_size(vector_db) == 0:
            st.warning("No documents indexed yet. Upload and ingest files first.")
            st.stop()

        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Build or reuse the QA chain (rebuilt when sidebar settings change)
        chain_key = (hf_token, temperature, max_new_tokens, top_k)
        if st.session_state.get("qa_chain_key") != chain_key or "qa_chain" not in st.session_state:
            with st.spinner("Initialising LLM…"):
                st.session_state["qa_chain"] = build_qa_chain(
                    vector_db, hf_token, temperature, max_new_tokens, top_k
                )
            st.session_state["qa_chain_key"] = chain_key

        qa_chain = st.session_state["qa_chain"]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = qa_chain.invoke({"query": query})

            answer = result["result"]
            source_docs = result.get("source_documents", [])

            st.markdown(answer)

            source_lines = []
            seen = set()
            for doc in source_docs:
                src = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "")
                label = f"**{src}**" + (f" — page {page + 1}" if page != "" else "")
                if label not in seen:
                    source_lines.append(f"- {label}")
                    seen.add(label)

            if source_lines:
                with st.expander("Sources"):
                    for line in source_lines:
                        st.markdown(line)

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": source_lines}
        )


if __name__ == "__main__":
    main()
