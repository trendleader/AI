# Call Center Multimodal RAG Assistant (Prototype)

A Streamlit prototype that lets a call-center team ground Claude's answers in
their own knowledge base: PDFs, Word docs, plain-text policy files, **and**
past call recordings (audio), all searchable from one chat interface.

## What it does

- **Ingests PDFs, `.docx`/`.doc`, and `.txt`** files as chunked, embedded
  knowledge-base documents.
- **Ingests call recordings** (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`) by
  transcribing them locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  and embedding the transcript in timestamp-tagged chunks. The audio file
  itself is deleted right after transcription — only the transcript text is
  stored.
- **Agent Assist chat** — ask a question or paste a customer's question and
  get an answer grounded only in the retrieved context, with inline `[1]`
  citations back to the source document/page or call timestamp. A mode
  toggle switches between a direct Q&A answer and a drafted customer-facing
  response.
- **Call Summarizer** — pick any ingested call recording and get a summary,
  customer sentiment, commitments made, and follow-up action items for a
  supervisor/QA handoff.
- **Deep reasoning toggle** — turns on Claude's adaptive thinking + high
  effort for gnarly policy questions; leave it off for fast live-call
  turnaround.

## Architecture

```
ingestion.py   Pure document/audio → LangChain Document loading + chunking
rag_engine.py  Cached models (embeddings, Whisper, Chroma) + retrieval + Claude calls
app.py         Streamlit UI wiring it together
```

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via
  `langchain-huggingface`.
- **Vector store**: Chroma, persisted to `./chroma_callcenter_db`.
- **LLM**: Claude via the official `anthropic` Python SDK, called directly
  (no LangChain LLM wrapper) so streaming, thinking, and effort controls are
  first-class. Defaults to `claude-opus-4-8`; `claude-sonnet-5` and
  `claude-haiku-4-5` are selectable in the sidebar for cost/latency tuning.

## Setup

```bash
cd callcenter_multimodal_rag
pip install -r requirements.txt
# ffmpeg is required by faster-whisper; poppler-utils improves PDF text extraction
# On Debian/Ubuntu: sudo apt-get install ffmpeg poppler-utils
cp .env.example .env   # then fill in ANTHROPIC_API_KEY (or paste it in the sidebar)
streamlit run app.py
```

First run downloads the embedding model and the selected Whisper model
weights — subsequent runs are cached.

## Notes / limitations (prototype scope)

- Whisper runs on CPU (`int8` quantized) — fine for prototyping; use a GPU
  build of `faster-whisper` for production call volumes.
- The vector store is a local, unauthenticated Chroma directory — add
  per-tenant isolation and access control before handling real customer data.
- No PII redaction is performed on transcripts or documents before embedding;
  add a redaction step if call recordings may contain sensitive data.
