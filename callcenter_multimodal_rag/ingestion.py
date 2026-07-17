"""Document and audio loading/chunking for the call-center RAG pipeline.

Pure processing functions — model/resource loading (embeddings, Whisper)
lives in rag_engine.py and is passed in here.
"""
import os
import tempfile
from datetime import timedelta

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_DOC_EXTS = (".pdf", ".txt", ".docx", ".doc")
SUPPORTED_AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".ogg", ".flac")


def _format_timestamp(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def load_text_document(uploaded_file) -> list[Document]:
    """Load a PDF, Word, or plain-text file into LangChain Documents."""
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
            raise ValueError(f"Unsupported document type: {ext}")
    finally:
        os.unlink(tmp_path)

    for doc in docs:
        doc.metadata["source"] = uploaded_file.name
        doc.metadata["media_type"] = "document"
    return docs


def _make_audio_doc(source: str, text: str, start: float, end: float) -> Document:
    return Document(
        page_content=text.strip(),
        metadata={
            "source": source,
            "media_type": "audio",
            "start": start,
            "end": end,
            "timestamp": f"{_format_timestamp(start)}–{_format_timestamp(end)}",
        },
    )


def transcribe_audio(uploaded_file, whisper_model, max_chars: int = 1000) -> list[Document]:
    """Transcribe a call recording with faster-whisper and merge consecutive
    segments into timestamp-tagged chunks of roughly `max_chars` characters.

    The audio file itself is written to a temp path only for the duration of
    transcription and then deleted — only the transcript text is retained.
    """
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        segments, _info = whisper_model.transcribe(tmp_path, beam_size=5)
        segments = list(segments)
    finally:
        os.unlink(tmp_path)

    docs = []
    buf_text, buf_start, buf_end = "", None, None
    for seg in segments:
        if buf_start is None:
            buf_start = seg.start
        buf_text += seg.text
        buf_end = seg.end
        if len(buf_text) >= max_chars:
            docs.append(_make_audio_doc(uploaded_file.name, buf_text, buf_start, buf_end))
            buf_text, buf_start, buf_end = "", None, None
    if buf_text.strip():
        docs.append(_make_audio_doc(uploaded_file.name, buf_text, buf_start, buf_end))
    return docs


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split text documents into overlapping chunks. Audio documents are
    already segment-sized (see transcribe_audio) and pass through unchanged
    so their start/end timestamps stay meaningful."""
    text_docs = [d for d in docs if d.metadata.get("media_type") != "audio"]
    audio_docs = [d for d in docs if d.metadata.get("media_type") == "audio"]
    if not text_docs:
        return audio_docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(text_docs) + audio_docs
