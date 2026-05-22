# CLAUDE.md — AI Tools Repository

## Repository Overview

This repository is a collection of AI-powered Python applications, primarily Streamlit web apps, focused on career development, technical interview preparation, certification study, document processing, and general productivity tools. Most apps use OpenAI or Anthropic Claude as the LLM backend.

## Development Environment

- **Runtime:** Python 3.11 (devcontainer image: `mcr.microsoft.com/devcontainers/python:1-3.11-bullseye`)
- **UI Framework:** Streamlit (port 8501)
- **IDE:** VS Code with `ms-python.python` and `ms-python.vscode-pylance` extensions
- **Container setup:** On attach, devcontainer installs `packages.txt` (apt) and `requirements.txt` (pip), then auto-starts `resumejobanalyzing.py`

### Running Apps

```bash
# Any Streamlit app
streamlit run <app_file.py> --server.enableCORS false --server.enableXsrfProtection false

# Install all dependencies
pip install -r requirements.txt
sudo apt install -y $(cat packages.txt)
```

### System Dependencies (packages.txt)

```
tesseract-ocr
tesseract-ocr-eng
poppler-utils
```

## File Catalog

### Career & Resume Tools

| File | Purpose | AI Model |
|------|---------|----------|
| `careerint.py` | Full interview prep suite with avatar interviewers and multi-module coaching | Claude (Anthropic) |
| `careers.py` | Career development and job-matching assistant | Claude (Anthropic) |
| `resume_iq.py` | Resume analysis with interview simulation and avatar UI | Claude (Anthropic) |
| `careercoach26v2.py` | Resume coaching, ATS scoring, and technical interview simulation | OpenAI |
| `careercoach2026.py` | Earlier version of careercoach26v2 | OpenAI |
| `enhancedresumecoach.py` | Enhanced resume coaching with detailed feedback | OpenAI |
| `cleanedresumeanalyzer.py` | Clean resume analysis and ATS matching | OpenAI |
| `resume_matcher_app.py` | Resume-to-job-description matching tool | OpenAI |
| `resumejobanalyzing.py` | Resume and job ad side-by-side analysis | OpenAI |
| `resume_job_analysis.py` | Earlier iteration of resume/job analysis | OpenAI |
| `resumejob.py` | Basic resume/job comparison tool | OpenAI |
| `resume_ats_tool.py` | ATS compliance checker | OpenAI |
| `linkedin_optimizer.py` | LinkedIn profile optimization tool | OpenAI |
| `jobs.py` | Job posting scraper and resume tailoring agent | OpenAI |
| `apply.py` | Resume tailoring and job application assistant with DOCX/PDF export | OpenAI |
| `work.py` | General career assistant | OpenAI |
| `careerint.py` | Interview coaching suite (largest file, ~7000 lines, most up to date) | Claude |

### Technical Interview & Exam Prep

| File | Purpose | AI Model |
|------|---------|----------|
| `technical_interview.py` | LeetCode-style coding interview prep with SQLite question DB | OpenAI |
| `technicalinterview.py` | Alternative coding interview prep | OpenAI |
| `interview.py` | General interview preparation | OpenAI |
| `interview_prep_app.py` | Interview prep with structured question banks | OpenAI |
| `careerint.py` | Includes behavioral, technical, and hiring manager interview modules | Claude |
| `dax_interview_prep.py` | DAX (Power BI) coding interview prep | OpenAI |
| `excel_skills.py` | Excel proficiency tasks with openpyxl file upload evaluation | OpenAI |
| `techtest.py` | Technical testing tool | OpenAI |
| `technical_test.py` | Alternative technical test implementation | OpenAI |
| `technicaltest.py` | Another technical test iteration | OpenAI |
| `scenarios.py` | Data-only: scenario definitions for technical writing exercises | None |
| `writing.py` | Technical writing coach using scenarios/analyzer/ai_feedback modules | OpenAI |

### Certification Study Apps

| File | Purpose | AI Model |
|------|---------|----------|
| `pl300_bootcamp.py` | Microsoft PL-300 (Power BI) exam prep with full quiz simulation | OpenAI |
| `pl300_bootcamp_app.py` | Alternative PL-300 app | OpenAI |
| `aws_cert_app.py` | AWS certification study app | OpenAI |
| `aws_streamlit.py` | AWS topics Streamlit interface | OpenAI |
| `full_aws_test.py` | Full AWS mock exam | OpenAI |
| `claude_architect_study_guide.py` | Claude Certified Architect exam prep app | Anthropic |
| `streamlit_certification_app.py` | Generic certification study platform | OpenAI |
| `fixed_exam_prep_system.py` | Fixed exam prep with session tracking | OpenAI |
| `windows_robust_exam_prep.py` | Windows-compatible exam prep | OpenAI |
| `fixed_test_presenter.py` | Test presentation utility | OpenAI |
| `pythondev.py` | Python developer skills coach | OpenAI |

### Document & Business Tools

| File | Purpose | AI Model |
|------|---------|----------|
| `brd.py` | Business Requirements Document generator | OpenAI |
| `brd2.py` | BRD generator v2 | OpenAI |
| `brd3.py` | BRD generator v3 (current) | OpenAI |
| `brd_generator.py` | BRD utility functions | OpenAI |
| `business_analyst_app.py` | Full business analyst toolkit | OpenAI |
| `legal_research_assistant_prof.py` | Legal research with document upload and chat | OpenAI + HuggingFace |
| `legal_research_assistant_dual_validated.py` | Dual-validated legal research assistant | OpenAI + HuggingFace |
| `paper_humanizer.py` | AI-text humanizer/paraphraser for academic papers | OpenAI |
| `receipts.py` | Receipt OCR and expense categorization | None (pytesseract) |
| `analyzer.py` | Rule-based technical writing analyzer (library, not standalone app) | None |
| `ai_feedback.py` | Shared AI feedback module used by `writing.py` | OpenAI |

### RAG & AI Infrastructure

| File | Purpose | AI Model |
|------|---------|----------|
| `rag_pipeline_app.py` | RAG document QA — upload PDF/TXT/DOCX, query with LLM | Mistral-7B (HuggingFace) |
| `voice_assistant.py` | CLI voice assistant with multi-personality support | OpenAI / HuggingFace |
| `voice_assistant_streamlit.py` | Streamlit UI for voice assistant using gTTS + Web Speech API | OpenAI / HuggingFace |
| `avatar_component.py` | Shared utility — generates HTML for animated interviewer avatars | None |
| `chainlit_sqlchatbot.py` | Chainlit-based SQL chatbot | OpenAI |

### Finance & Analytics

| File | Purpose | AI Model |
|------|---------|----------|
| `loan_approval_analyzer.py` | Loan DTI calculator and approval estimator | OpenAI (via API) |
| `fraud_clean.py` | Pharmacy claims fraud detection (ML + OpenAI interpretation) | scikit-learn + OpenAI |
| `lottery_predictor_multi.py` | Multi-game lottery prediction (Powerball, Mega Millions, Pick 6) | OpenAI |
| `powerball_predictor_enhanced.py` | Enhanced Powerball predictor | OpenAI |

### Miscellaneous

| File | Purpose | AI Model |
|------|---------|----------|
| `word_to_audiobook.py` | Word document to audiobook converter | TTS |
| `python-career-chatbot.py` | Python career Q&A chatbot | OpenAI |
| `freelancer_marketplace_python.py` | Freelancer marketplace platform | OpenAI |
| `mental_health_ai_plat.py` | Mental health support platform | OpenAI |
| `Gherkin.py` / `Gherkin2.py` / `gherkine.py` | Gherkin BDD scenario generator | OpenAI |
| `questions_db.py` | Questions database for interview apps | None |
| `complete_streamlit_app.py` | General complete Streamlit template | OpenAI |
| `streamlit_app_complete_final.py` | Final version of complete app template | OpenAI |
| `sql_challenges_app_enhanced.py` | SQL challenge practice app | OpenAI |
| `modified_imports (1).py` | Import cleanup utility | None |

### Jupyter Notebooks

| File | Purpose |
|------|---------|
| `Classifier.ipynb` | ML classification experiments |
| `Training2.ipynb` | Model training experiments |
| `Train1.ipynb` | Training notebook v1 |
| `LangChain_Basics.ipynb` | LangChain tutorial notebook |
| `Create_Agent_from_scratch.ipynb` | Agent-from-scratch tutorial |

## Shared Utility Modules

These are imported by multiple apps — do not delete or refactor without checking dependents:

- **`avatar_component.py`** — `get_avatar_html(persona, message, is_speaking)` returns self-contained HTML for animated interviewer avatars. Used by `careerint.py`, `resume_iq.py`, and others.
- **`ai_feedback.py`** — `get_ai_feedback(user_text, scenario, api_key)` returns `AIFeedback` dataclass. Used by `writing.py`.
- **`scenarios.py`** — Pure-data module defining `WritingScenario` and `ScenarioRequirement` dataclasses. Used by `writing.py` and `ai_feedback.py`.
- **`analyzer.py`** — Rule-based `WritingAnalyzer` class with `IssueSeverity` enum. Used by `writing.py`.
- **`questions_db.py`** — Static question database for interview/exam apps.

## Common Code Patterns

### Streamlit App Structure

All Streamlit apps follow this structure:

```python
import streamlit as st

# 1. Page config — MUST be first Streamlit call
st.set_page_config(page_title="...", page_icon="...", layout="wide")

# 2. Custom CSS injection
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)

# 3. Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. API key retrieval — env vars preferred, sidebar fallback
api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key", type="password")

# 5. Main UI / navigation
# 6. Business logic / AI calls
```

### Session State Conventions

```python
# Always guard initialization
if "key" not in st.session_state:
    st.session_state.key = default_value

# Chat history pattern
st.session_state.messages.append({"role": "user", "content": prompt})
st.session_state.messages.append({"role": "assistant", "content": response})
```

### Anthropic (Claude) API Pattern

```python
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-sonnet-4-5",  # or claude-opus-4-5, claude-haiku-4-5
    max_tokens=2048,
    messages=[{"role": "user", "content": prompt}]
)
text = response.content[0].text
```

### OpenAI API Pattern

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)
text = response.choices[0].message.content
```

### RAG Pattern (LangChain + Chroma)

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

@st.cache_resource(show_spinner="Loading model…")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector DB persists to ./chroma_rag_db
vector_db = Chroma(persist_directory="./chroma_rag_db", embedding_function=embeddings)
```

### Custom CSS / Theming

Apps use a dark theme with CSS variables injected via `st.markdown`:

```css
:root {
    --bg:      #0a0b0f;
    --surface: #13151c;
    --border:  #1e2130;
    --accent:  #6c63ff;
    --accent2: #00d4aa;
    --text:    #e8eaf0;
    --muted:   #6b7280;
}
```

Google Fonts (`Syne` for headings, `DM Sans` for body) are imported in CSS.

### Document Processing

```python
# PDF
import pdfplumber  # preferred for text extraction
from PyPDF2 import PdfReader  # alternative

# DOCX
from docx import Document  # python-docx

# OCR
import pytesseract
from PIL import Image
```

### Voice / Audio (voice_assistant_streamlit.py pattern)

Audio is injected as base64 data URIs to avoid Streamlit's media storage GC issues:

```python
from gtts import gTTS
import base64

def speak(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts.save(f.name)
        audio_b64 = base64.b64encode(open(f.name, "rb").read()).decode()
    st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}"></audio>',
                unsafe_allow_html=True)
```

## Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | `careerint.py`, `careers.py`, `resume_iq.py`, `claude_architect_study_guide.py` | Anthropic Claude API access |
| `OPENAI_API_KEY` | Most other apps | OpenAI GPT API access |
| `HF_API_KEY` or `HUGGINGFACEHUB_API_TOKEN` | `rag_pipeline_app.py`, `voice_assistant.py`, `legal_research_assistant_prof.py` | HuggingFace Inference API |

Use `.env` files with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
```

## Git Workflow

- **Main branch:** `main`
- **Feature branches:** `claude/<feature-name>-<hash>` pattern (e.g., `claude/rag-pipeline-streamlit-ui-yCUi4`)
- Development branch for current work: `claude/claude-md-docs-7bNXZ`
- Commits are pushed directly; PRs are created via GitHub

Recent significant additions (see git log):
- Voice assistant with Streamlit UI and audio fix
- RAG pipeline with LangChain + Chroma + Mistral-7B
- Claude Architect study guide app
- `careerint.py` — most actively developed file (187KB)

## Key Conventions

1. **No test files exist** — test manually by running the Streamlit app
2. **No type annotations** in most files — add when modifying
3. **Error handling:** API failures show `st.error()` messages; no silent failures
4. **File naming:** snake_case for Python files; similar tools exist in multiple versions (brd.py, brd2.py, brd3.py) — the highest-numbered version is current
5. **Model selection:** Default to `claude-sonnet-4-6` for Claude apps, `gpt-4o` for OpenAI apps
6. **Caching:** Use `@st.cache_resource` for expensive model loads; `@st.cache_data` for data fetching
7. **Port:** All Streamlit apps run on port 8501 (auto-forwarded in devcontainer)

## Adding a New App

1. Create `<app_name>.py` in the root directory
2. Follow the Streamlit app structure above
3. Add any new pip dependencies to `requirements.txt`
4. Add any new apt packages to `packages.txt`
5. Run: `streamlit run <app_name>.py --server.enableCORS false --server.enableXsrfProtection false`
