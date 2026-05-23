import streamlit as st
import anthropic
import random
import json
import os
from datetime import datetime

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Azure AI-102 Study Guide",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --azure-blue: #0078D4;
    --azure-dark: #004578;
    --azure-light: #E1F0FF;
    --correct: #107C10;
    --incorrect: #D83B01;
}
.hero-banner {
    background: linear-gradient(135deg, #0078D4 0%, #004578 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero-banner h1 { font-size: 2.4rem; margin: 0 0 .4rem; }
.hero-banner p  { font-size: 1.1rem; opacity: .9; margin: 0; }

.domain-card {
    background: #f8f9fa;
    border-left: 5px solid #0078D4;
    padding: 1rem 1.2rem;
    border-radius: 6px;
    margin-bottom: .8rem;
}
.domain-card h4 { color: #004578; margin: 0 0 .3rem; }
.domain-card p  { color: #555; margin: 0; font-size: .92rem; }

.service-chip {
    display: inline-block;
    background: #E1F0FF;
    color: #004578;
    border-radius: 20px;
    padding: .25rem .75rem;
    margin: .2rem;
    font-size: .85rem;
    font-weight: 600;
}
.correct-badge   { color: #107C10; font-weight: 700; }
.incorrect-badge { color: #D83B01; font-weight: 700; }
.score-box {
    background: linear-gradient(135deg, #0078D4, #004578);
    color: white;
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
}
.score-box h2 { font-size: 3rem; margin: 0; }
.key-concept {
    background: #fff8e1;
    border-left: 4px solid #F7630C;
    padding: .7rem 1rem;
    border-radius: 4px;
    margin-bottom: .5rem;
    font-size: .93rem;
}
.tip-box {
    background: #e8f5e9;
    border-left: 4px solid #107C10;
    padding: .7rem 1rem;
    border-radius: 4px;
    margin-bottom: .5rem;
    font-size: .93rem;
}
</style>
""", unsafe_allow_html=True)

# ── Exam Domain Data ─────────────────────────────────────────────────────────
DOMAINS = {
    "Plan & Manage Azure AI Solutions (15–20%)": {
        "icon": "🏗️",
        "topics": [
            "Select appropriate Azure AI services for requirements",
            "Plan and configure security for Azure AI services",
            "Create and manage Azure AI service resources",
            "Monitor Azure AI services with Azure Monitor & alerts",
            "Implement responsible AI principles (fairness, reliability, safety)",
            "Manage AI service keys and endpoints",
            "Configure virtual networks and private endpoints",
            "Implement content filtering and usage policies",
        ],
        "services": ["Azure AI Services", "Azure Key Vault", "Azure Monitor", "Azure Policy",
                     "Managed Identity", "Private Endpoint", "Azure RBAC"],
    },
    "Implement Computer Vision Solutions (15–20%)": {
        "icon": "👁️",
        "topics": [
            "Analyze images with Azure AI Vision (Image Analysis 4.0)",
            "Extract text using OCR / Read API",
            "Detect and analyze faces with Face API",
            "Train custom image classification models (Custom Vision)",
            "Train custom object detection models",
            "Process video with Azure Video Indexer",
            "Segment images with Image Segmentation",
            "Perform spatial analysis with Azure AI Vision",
        ],
        "services": ["Azure AI Vision", "Custom Vision", "Face API", "Azure Video Indexer",
                     "Azure AI Studio"],
    },
    "Implement NLP Solutions (30–35%)": {
        "icon": "💬",
        "topics": [
            "Analyze text with Azure AI Language (sentiment, entities, key phrases)",
            "Build and deploy conversational language understanding (CLU) models",
            "Create custom named entity recognition (NER) models",
            "Build and deploy Azure AI Bot Service bots",
            "Implement question-answering with custom question answering",
            "Translate text and documents with Azure AI Translator",
            "Implement speech-to-text and text-to-speech",
            "Implement speech translation and custom speech models",
            "Classify text with custom text classification",
            "Summarize text and extract information",
        ],
        "services": ["Azure AI Language", "Azure AI Translator", "Azure AI Speech",
                     "Azure Bot Service", "CLU", "Custom NER", "QnA / Custom Q&A"],
    },
    "Implement Knowledge Mining & Document Intelligence (10–15%)": {
        "icon": "🔍",
        "topics": [
            "Implement Azure AI Search indexes and indexers",
            "Enrich search with built-in and custom skillsets",
            "Create knowledge stores from enriched content",
            "Implement semantic search and vector search",
            "Extract data from documents with Azure AI Document Intelligence",
            "Build custom document extraction models",
            "Use prebuilt models (invoices, receipts, IDs, tax forms)",
        ],
        "services": ["Azure AI Search", "Azure AI Document Intelligence",
                     "Cognitive Skillsets", "Knowledge Store", "Semantic Ranker"],
    },
    "Implement Generative AI Solutions (10–15%)": {
        "icon": "✨",
        "topics": [
            "Deploy and use Azure OpenAI models (GPT-4, DALL-E, Whisper, Embeddings)",
            "Apply prompt engineering techniques (zero-shot, few-shot, chain-of-thought)",
            "Build Retrieval-Augmented Generation (RAG) solutions",
            "Implement function calling and tool use",
            "Use Azure AI Studio for model catalog and fine-tuning",
            "Implement content safety with Azure AI Content Safety",
            "Build AI agents and orchestrate multi-step workflows",
            "Apply responsible AI practices in generative AI",
        ],
        "services": ["Azure OpenAI Service", "Azure AI Studio", "Azure AI Content Safety",
                     "GPT-4o", "DALL-E 3", "Embeddings", "Semantic Kernel", "Prompt Flow"],
    },
}

# ── Practice Questions ────────────────────────────────────────────────────────
QUESTIONS = [
    # Domain 1
    {
        "domain": "Plan & Manage Azure AI Solutions (15–20%)",
        "question": "A developer needs to call Azure AI services from an Azure VM without storing credentials in code. What is the BEST approach?",
        "options": ["Store the API key in an environment variable on the VM",
                    "Use a system-assigned managed identity and grant it Cognitive Services User role",
                    "Hard-code the subscription key in the application configuration",
                    "Pass the API key via a query parameter in the request URL"],
        "answer": 1,
        "explanation": "Managed identities eliminate the need to manage credentials. Assigning the VM a system-assigned managed identity and granting it the 'Cognitive Services User' role lets the app obtain tokens via IMDS without any stored secret.",
    },
    {
        "domain": "Plan & Manage Azure AI Solutions (15–20%)",
        "question": "You want to ensure that your Azure AI Language service is only accessible from your corporate VNet and not the public internet. What should you configure?",
        "options": ["A shared access signature (SAS) token",
                    "A Private Endpoint and disable public network access",
                    "IP allow-list with 0.0.0.0/0",
                    "Azure AD Conditional Access policy"],
        "answer": 1,
        "explanation": "A Private Endpoint creates a private IP within your VNet for the service. Combined with disabling public network access, all traffic stays inside the VNet.",
    },
    {
        "domain": "Plan & Manage Azure AI Solutions (15–20%)",
        "question": "Which Azure AI Responsible AI principle focuses on ensuring AI systems behave as intended across all demographic groups without producing disparate outcomes?",
        "options": ["Reliability & Safety", "Privacy & Security", "Fairness", "Inclusiveness"],
        "answer": 2,
        "explanation": "Fairness requires that AI systems treat all people equitably and don't produce discriminatory outcomes based on race, gender, age, or other characteristics.",
    },
    # Domain 2 – Computer Vision
    {
        "domain": "Implement Computer Vision Solutions (15–20%)",
        "question": "You need to extract printed and handwritten text from scanned documents at scale. Which Azure AI Vision feature should you use?",
        "options": ["Image Analysis – tagging", "Read API (OCR)", "Face API", "Custom Vision classification"],
        "answer": 1,
        "explanation": "The Read API (now part of Azure AI Vision Image Analysis 4.0 as 'Read') is optimized for high-accuracy OCR on both printed and handwritten text in documents and images.",
    },
    {
        "domain": "Implement Computer Vision Solutions (15–20%)",
        "question": "A retail company wants to detect defective products on an assembly line using computer vision. They have a labeled dataset of 500 images per class. Which service is MOST appropriate?",
        "options": ["Azure AI Vision Image Analysis – dense captioning",
                    "Custom Vision – Object Detection",
                    "Face API – verification",
                    "Azure Video Indexer"],
        "answer": 1,
        "explanation": "Custom Vision Object Detection lets you train a model on your own labeled images to detect specific objects (defects). It's designed for this domain-specific training scenario.",
    },
    {
        "domain": "Implement Computer Vision Solutions (15–20%)",
        "question": "Which Azure AI Vision feature can identify individual faces in an image AND verify whether two face images belong to the same person?",
        "options": ["Image Analysis – people detection", "Face API – Detect + Verify",
                    "Custom Vision – classification", "Video Indexer – face detection"],
        "answer": 1,
        "explanation": "Face API provides Detect (locate and analyze faces) and Verify (compare two faces for identity match) operations specifically designed for facial recognition tasks.",
    },
    # Domain 3 – NLP
    {
        "domain": "Implement NLP Solutions (30–35%)",
        "question": "A company needs to build a chatbot that understands user intent from natural language and maps it to predefined actions. Which Azure AI Language feature should they use?",
        "options": ["Sentiment Analysis", "Conversational Language Understanding (CLU)",
                    "Key Phrase Extraction", "Custom Named Entity Recognition"],
        "answer": 1,
        "explanation": "CLU (Conversational Language Understanding) is the successor to LUIS. It allows you to train a model to identify intents and extract entities from user utterances.",
    },
    {
        "domain": "Implement NLP Solutions (30–35%)",
        "question": "You need to build a FAQ bot grounded in your company's existing knowledge base documents. The fastest path to production is:",
        "options": ["Train a BERT model from scratch",
                    "Use Azure AI Language Custom Question Answering",
                    "Implement CLU with intents for every FAQ item",
                    "Fine-tune GPT-4 on FAQ pairs"],
        "answer": 1,
        "explanation": "Custom Question Answering (successor to QnA Maker) lets you upload documents or FAQs and automatically extracts question-answer pairs, then exposes them via a REST endpoint with minimal code.",
    },
    {
        "domain": "Implement NLP Solutions (30–35%)",
        "question": "Your application must transcribe real-time call center audio and then detect whether the caller is satisfied or frustrated. Which TWO services should you combine?",
        "options": ["Azure AI Speech (Speech-to-Text) + Azure AI Language (Sentiment Analysis)",
                    "Azure Video Indexer + Custom Vision",
                    "Azure AI Translator + Face API",
                    "Azure OpenAI Whisper + Custom NER"],
        "answer": 0,
        "explanation": "Azure AI Speech converts audio to text in real time; Azure AI Language Sentiment Analysis then scores the transcribed text for positive/negative/neutral/mixed sentiment.",
    },
    {
        "domain": "Implement NLP Solutions (30–35%)",
        "question": "A global e-commerce platform wants to translate product descriptions from English into 50 languages automatically. Which service handles this at enterprise scale?",
        "options": ["Azure AI Language – Key Phrase Extraction",
                    "Azure AI Translator (Document Translation)",
                    "Azure OpenAI GPT-4 completion",
                    "Azure AI Speech – Speech Translation"],
        "answer": 1,
        "explanation": "Azure AI Translator's Document Translation feature handles bulk, asynchronous translation of documents into multiple languages and supports 100+ languages at enterprise scale.",
    },
    # Domain 4 – Knowledge Mining
    {
        "domain": "Implement Knowledge Mining & Document Intelligence (10–15%)",
        "question": "Which Azure AI Search concept allows you to enrich documents during indexing by running AI skills such as OCR, entity extraction, and sentiment analysis?",
        "options": ["Semantic ranker", "Skillset (enrichment pipeline)", "Knowledge store", "Suggesters"],
        "answer": 1,
        "explanation": "A Skillset defines a pipeline of built-in or custom AI skills that process each document during indexing, adding enriched fields (entities, sentiment, translated text, etc.) to the search index.",
    },
    {
        "domain": "Implement Knowledge Mining & Document Intelligence (10–15%)",
        "question": "You need to extract structured data (vendor, total amount, line items) from thousands of PDF invoices. Which prebuilt model in Azure AI Document Intelligence should you use?",
        "options": ["Business card model", "Invoice model", "Receipt model", "General document model"],
        "answer": 1,
        "explanation": "The prebuilt Invoice model in Azure AI Document Intelligence is trained to extract invoice-specific fields: vendor name, invoice number, due date, line items, subtotals, and totals.",
    },
    {
        "domain": "Implement Knowledge Mining & Document Intelligence (10–15%)",
        "question": "Azure AI Search Semantic Ranker improves search results by:",
        "options": ["Running BM25 full-text scoring only",
                    "Re-ranking results using language models to understand query intent",
                    "Applying custom scoring profiles based on field weights",
                    "Indexing vector embeddings for nearest-neighbor search"],
        "answer": 1,
        "explanation": "Semantic Ranker applies deep language models to re-rank BM25 results by understanding the semantic meaning of the query and documents, returning more contextually relevant results.",
    },
    # Domain 5 – Generative AI
    {
        "domain": "Implement Generative AI Solutions (10–15%)",
        "question": "You are building a RAG (Retrieval-Augmented Generation) solution with Azure OpenAI and Azure AI Search. What is the PRIMARY purpose of adding a retrieval step?",
        "options": ["To reduce the token cost of embedding generation",
                    "To ground the LLM response in authoritative documents, reducing hallucination",
                    "To replace the need for a system prompt",
                    "To increase the temperature of the model output"],
        "answer": 1,
        "explanation": "RAG retrieves relevant documents from a search index and includes them in the LLM context. This grounds responses in factual data, significantly reducing hallucinations.",
    },
    {
        "domain": "Implement Generative AI Solutions (10–15%)",
        "question": "Which prompting technique provides the LLM with several input-output examples before presenting the actual task?",
        "options": ["Zero-shot prompting", "Few-shot prompting",
                    "Chain-of-thought prompting", "System message prompting"],
        "answer": 1,
        "explanation": "Few-shot prompting includes 2–10 example input/output pairs in the prompt so the model learns the desired format and reasoning pattern from demonstrations.",
    },
    {
        "domain": "Implement Generative AI Solutions (10–15%)",
        "question": "An application uses Azure OpenAI GPT-4o and needs the model to call an external weather API when users ask about weather. Which feature enables this?",
        "options": ["Embeddings API", "Function calling (tool use)",
                    "Assistants API with file search", "DALL-E image generation"],
        "answer": 1,
        "explanation": "Function calling allows you to describe external functions/APIs to the model. The model outputs a structured JSON call when it decides a function is needed, and your code executes the actual API call.",
    },
    {
        "domain": "Implement Generative AI Solutions (10–15%)",
        "question": "Which Azure service provides a unified portal for discovering foundation models, experimenting with prompts, evaluating models, and deploying AI applications?",
        "options": ["Azure Machine Learning Studio", "Azure AI Studio",
                    "Azure Databricks", "Power Platform AI Builder"],
        "answer": 1,
        "explanation": "Azure AI Studio is Microsoft's unified platform for generative AI development: model catalog, prompt flow, evaluation, fine-tuning, and deployment — all in one portal.",
    },
    {
        "domain": "Plan & Manage Azure AI Solutions (15–20%)",
        "question": "You want to prevent Azure OpenAI from generating harmful content in your application. Which service provides built-in content moderation for generative AI outputs?",
        "options": ["Azure Policy", "Azure AI Content Safety",
                    "Microsoft Defender for Cloud", "Azure Sentinel"],
        "answer": 1,
        "explanation": "Azure AI Content Safety analyzes text and images for harmful content (hate, violence, sexual, self-harm) and can block or flag model outputs before they reach users.",
    },
    {
        "domain": "Implement Computer Vision Solutions (15–20%)",
        "question": "Azure AI Vision Image Analysis 4.0 introduces which NEW capability not present in version 3.2?",
        "options": ["OCR for printed text", "Dense captions with natural language descriptions for regions",
                    "Color palette detection", "Celebrity recognition"],
        "answer": 1,
        "explanation": "Image Analysis 4.0 introduced dense captions, which generates natural-language captions for up to 10 distinct regions of an image, providing richer scene understanding.",
    },
    {
        "domain": "Implement NLP Solutions (30–35%)",
        "question": "You need to identify and redact sensitive PII such as social security numbers and email addresses from text. Which Azure AI Language feature handles this?",
        "options": ["Key phrase extraction", "PII detection and redaction",
                    "Sentiment analysis", "Language detection"],
        "answer": 1,
        "explanation": "The PII (Personally Identifiable Information) detection feature in Azure AI Language identifies sensitive entities and can return redacted versions of the text with PII replaced.",
    },
    {
        "domain": "Implement Knowledge Mining & Document Intelligence (10–15%)",
        "question": "What does a Knowledge Store in Azure AI Search allow you to do?",
        "options": ["Cache search queries for faster response",
                    "Persist AI-enriched data to Azure Storage for downstream analytics",
                    "Store encrypted search index backups",
                    "Sync search results to Azure SQL Database"],
        "answer": 1,
        "explanation": "A Knowledge Store saves enriched content (from skillset processing) as JSON objects, tables, or file projections in Azure Storage, making it available for Power BI, ML, or other analytics tools.",
    },
]

# ── Key Concept Cheat Sheets ─────────────────────────────────────────────────
KEY_CONCEPTS = {
    "Azure AI Service Endpoints": [
        "All Azure AI services expose a REST API at: `https://<resource-name>.cognitiveservices.azure.com/`",
        "Authenticate with: `Ocp-Apim-Subscription-Key` header or Bearer token (managed identity)",
        "Multi-service resource supports all AI services with one endpoint + key",
        "Single-service resources offer custom domain names and independent quotas",
    ],
    "Responsible AI Principles": [
        "**Fairness** – AI must not discriminate based on protected characteristics",
        "**Reliability & Safety** – AI must perform consistently and fail gracefully",
        "**Privacy & Security** – AI must protect personal data",
        "**Inclusiveness** – AI must empower all people including those with disabilities",
        "**Transparency** – People must understand how AI makes decisions",
        "**Accountability** – Humans must be responsible for AI systems",
    ],
    "Azure OpenAI Key Terms": [
        "**Deployment** – A named instance of a model (e.g., gpt-4o-deployment)",
        "**Prompt** – Input text sent to the model",
        "**Completion** – The model's generated response",
        "**Temperature (0-2)** – Controls randomness (0=deterministic, 2=creative)",
        "**Top-p** – Nucleus sampling; alternative to temperature",
        "**Max tokens** – Maximum length of the response",
        "**System message** – Sets the model's persona and instructions",
        "**Few-shot examples** – Example turns in conversation history",
    ],
    "Azure AI Search Architecture": [
        "**Index** – Schema defining searchable fields and their types",
        "**Indexer** – Crawler that pulls data from a data source into the index",
        "**Data Source** – Blob Storage, SQL, Cosmos DB, etc.",
        "**Skillset** – Pipeline of AI enrichments (OCR, NER, translation, etc.)",
        "**Knowledge Store** – Persists enriched data to Azure Storage",
        "**Semantic Ranker** – Language-model re-ranking (requires Standard tier+)",
        "**Vector Search** – Nearest-neighbor search on embedding vectors",
    ],
    "CLU vs QnA vs Custom NER": [
        "**CLU (Conversational Language Understanding)** – Maps utterances to intents + extracts entities. Use for chatbot commands.",
        "**Custom Question Answering** – Returns best answer from a knowledge base. Use for FAQ bots.",
        "**Custom NER** – Identifies domain-specific entities (e.g., contract clauses, drug names) in free text.",
        "**Custom Text Classification** – Assigns text to one or more custom categories.",
    ],
    "Document Intelligence Models": [
        "**Prebuilt Invoice** – Vendor, invoice #, line items, totals",
        "**Prebuilt Receipt** – Merchant, items, totals, date",
        "**Prebuilt ID Document** – Passports, driver's licenses",
        "**Prebuilt Business Card** – Name, title, company, contact info",
        "**Prebuilt W-2 / Tax** – US tax form fields",
        "**General Document** – Layout, key-value pairs, tables (any document)",
        "**Custom Extraction** – Train on your own labeled forms",
        "**Custom Classification** – Classify document type before routing",
    ],
}

# ── Session State Init ────────────────────────────────────────────────────────
defaults = {
    "api_key": "",
    "quiz_active": False,
    "quiz_questions": [],
    "quiz_index": 0,
    "quiz_score": 0,
    "quiz_answers": [],
    "quiz_domain_filter": "All Domains",
    "quiz_completed": False,
    "study_progress": {},
    "ai_response": "",
    "ai_loading": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_client():
    key = st.session_state.api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


def stream_explanation(prompt: str) -> str:
    client = get_client()
    if not client:
        return "⚠️ Please enter your Anthropic API key in the sidebar to get AI explanations."
    full = ""
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        system=(
            "You are an expert Azure AI Engineer and Microsoft Certified Trainer. "
            "Give concise, exam-focused answers. Use bullet points where helpful. "
            "Highlight Azure service names in bold."
        ),
    ) as stream:
        for text in stream.text_stream:
            full += text
    return full


def start_quiz(questions: list):
    st.session_state.quiz_questions = questions
    st.session_state.quiz_index = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_answers = []
    st.session_state.quiz_active = True
    st.session_state.quiz_completed = False


def reset_quiz():
    st.session_state.quiz_active = False
    st.session_state.quiz_completed = False
    st.session_state.quiz_questions = []
    st.session_state.quiz_answers = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Key")
    key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-ant-...",
        help="Used for AI-powered explanations. Never stored beyond this session.",
    )
    if key_input:
        st.session_state.api_key = key_input
        st.success("Key saved for this session.")

    st.divider()
    st.markdown("## 📊 Exam Overview")
    st.markdown("""
| Domain | Weight |
|--------|--------|
| Plan & Manage AI | 15–20% |
| Computer Vision | 15–20% |
| NLP | 30–35% |
| Knowledge Mining | 10–15% |
| Generative AI | 10–15% |
""")
    st.divider()
    st.markdown("### 📌 Quick Stats")
    answered = len([a for a in st.session_state.quiz_answers if a is not None])
    correct = st.session_state.quiz_score
    st.metric("Questions Answered", answered)
    st.metric("Correct Answers", correct)
    if answered:
        st.metric("Running Accuracy", f"{correct/answered*100:.0f}%")

# ── Main Nav ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🤖 Azure AI Engineer Associate</h1>
  <p>AI-102 Exam Study Guide · 2026 Edition</p>
</div>
""", unsafe_allow_html=True)

tab_home, tab_domains, tab_services, tab_concepts, tab_quiz, tab_ask = st.tabs([
    "🏠 Home",
    "📚 Domains",
    "☁️ Services",
    "💡 Key Concepts",
    "🎯 Practice Quiz",
    "🤖 Ask AI",
])

# ── TAB: HOME ────────────────────────────────────────────────────────────────
with tab_home:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Questions", len(QUESTIONS))
    col2.metric("Exam Domains", len(DOMAINS))
    col3.metric("Passing Score", "700 / 1000")

    st.markdown("---")
    st.subheader("About the AI-102 Exam")
    st.markdown("""
The **Microsoft Azure AI Engineer Associate (AI-102)** certification validates expertise in designing,
building, and deploying AI solutions using Azure AI services. Candidates should have 1+ year of
experience building AI/ML solutions on Azure.

**Exam format:**
- 40–60 questions (multiple choice, case studies, drag-and-drop)
- 180 minutes
- Passing score: 700/1000
- Proctored in-person or online via Pearson VUE / Certiport

**Prerequisite knowledge:**
- Azure fundamentals (or AZ-900)
- Proficiency in Python or C#
- REST API concepts
""")

    st.subheader("📅 Study Plan Recommendation")
    plan = {
        "Week 1": "Azure AI Services overview, security, responsible AI, monitoring",
        "Week 2": "Computer Vision – Image Analysis, Custom Vision, Face API, Video Indexer",
        "Week 3": "NLP – Language service, CLU, QnA, Translator, Speech",
        "Week 4": "Knowledge Mining – AI Search, skillsets, Document Intelligence",
        "Week 5": "Generative AI – Azure OpenAI, prompt engineering, RAG, AI Studio",
        "Week 6": "Full practice tests, review weak domains, hands-on labs",
    }
    for week, focus in plan.items():
        st.markdown(f"""
<div class="domain-card">
<h4>{week}</h4>
<p>{focus}</p>
</div>
""", unsafe_allow_html=True)

    st.subheader("🔗 Official Resources")
    st.markdown("""
- [AI-102 Exam Page](https://learn.microsoft.com/en-us/certifications/exams/ai-102/)
- [Microsoft Learn – AI-102 Learning Path](https://learn.microsoft.com/en-us/training/courses/ai-102t00)
- [Azure AI Services Documentation](https://learn.microsoft.com/en-us/azure/ai-services/)
- [Azure AI Studio](https://ai.azure.com)
- [Free Practice Assessments on Microsoft Learn](https://learn.microsoft.com/en-us/certifications/practice-assessments-for-microsoft-certifications)
""")

# ── TAB: DOMAINS ─────────────────────────────────────────────────────────────
with tab_domains:
    st.subheader("Exam Domains & Learning Objectives")
    selected_domain = st.selectbox(
        "Select a domain to explore",
        list(DOMAINS.keys()),
    )
    domain = DOMAINS[selected_domain]
    st.markdown(f"### {domain['icon']} {selected_domain}")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("**Learning Objectives:**")
        for t in domain["topics"]:
            st.markdown(f"- {t}")

    with col_right:
        st.markdown("**Key Azure Services:**")
        chips = " ".join(
            f'<span class="service-chip">{s}</span>' for s in domain["services"]
        )
        st.markdown(chips, unsafe_allow_html=True)

    st.divider()

    # Domain progress (questions answered per domain)
    st.markdown("#### Domain Questions Available")
    domain_q = [q for q in QUESTIONS if q["domain"] == selected_domain]
    st.info(f"**{len(domain_q)}** practice questions available for this domain.")
    if st.button(f"Start {domain['icon']} Domain Quiz", key="domain_quiz_btn"):
        shuffled = random.sample(domain_q, len(domain_q))
        start_quiz(shuffled)
        st.session_state.quiz_domain_filter = selected_domain
        st.rerun()

# ── TAB: SERVICES ────────────────────────────────────────────────────────────
with tab_services:
    st.subheader("☁️ Azure AI Services Reference")

    services_data = {
        "Azure AI Vision": {
            "category": "Computer Vision",
            "description": "Analyze images and video. Features: image captioning, object detection, OCR (Read API), smart crop, background removal, dense captions.",
            "endpoint": "cognitiveservices.azure.com/vision/v3.2/",
            "tiers": "Free (S0) | Standard (S1)",
            "exam_tip": "Image Analysis 4.0 uses a single /analyze endpoint with visual features parameter. Read API is async for large docs.",
        },
        "Azure Custom Vision": {
            "category": "Computer Vision",
            "description": "Train custom image classifiers and object detectors with your own labeled images. Exports to ONNX, CoreML, TensorFlow.",
            "endpoint": "customvision.ai",
            "tiers": "Free (F0) | Standard (S0)",
            "exam_tip": "Needs minimum 15 images per tag for classification, 15 bounding boxes per tag for detection.",
        },
        "Azure Face API": {
            "category": "Computer Vision",
            "description": "Detect, analyze, and identify human faces. Operations: Detect, Find Similar, Group, Identify, Verify.",
            "endpoint": "cognitiveservices.azure.com/face/v1.0/",
            "tiers": "Free | Standard",
            "exam_tip": "Limited Access — requires approval for identification/verification features due to responsible AI policies.",
        },
        "Azure AI Language": {
            "category": "NLP",
            "description": "Unified NLP service. Features: sentiment, key phrases, NER, PII detection, language detection, CLU, custom NER, custom classification, QnA, summarization.",
            "endpoint": "cognitiveservices.azure.com/language/:analyze-text",
            "tiers": "Free (F0) | Standard (S)",
            "exam_tip": "Successor to Text Analytics + LUIS + QnA Maker. CLU replaces LUIS. Custom Q&A replaces QnA Maker.",
        },
        "Azure AI Translator": {
            "category": "NLP",
            "description": "Neural machine translation. Text translation, document translation, dictionary lookup, transliteration. 100+ languages.",
            "endpoint": "api.cognitive.microsofttranslator.com",
            "tiers": "Free (F0) | Standard (S1)",
            "exam_tip": "Document Translation is async and uses Azure Blob Storage as source/target. Supports glossaries for custom terms.",
        },
        "Azure AI Speech": {
            "category": "NLP",
            "description": "STT (batch + real-time), TTS (neural voices), speech translation, speaker recognition, custom speech, custom voice.",
            "endpoint": "cognitiveservices.azure.com/speech/",
            "tiers": "Free (F0) | Standard (S0)",
            "exam_tip": "Use Speech SDK for real-time; Batch Transcription REST API for large audio files. Custom Speech improves accuracy for domain-specific terms.",
        },
        "Azure AI Search": {
            "category": "Knowledge Mining",
            "description": "Full-text search with AI enrichment. Components: indexes, indexers, skillsets, knowledge store, semantic ranker, vector search.",
            "endpoint": "<search-name>.search.windows.net",
            "tiers": "Free | Basic | Standard (S1/S2/S3) | Storage Optimized",
            "exam_tip": "Semantic ranker requires Standard tier or higher. Vector search uses HNSW algorithm on embedding fields.",
        },
        "Azure AI Document Intelligence": {
            "category": "Knowledge Mining",
            "description": "Extract structured data from documents. Prebuilt models + custom extraction + custom classification.",
            "endpoint": "cognitiveservices.azure.com/formrecognizer/",
            "tiers": "Free (F0) | Standard (S0)",
            "exam_tip": "Formerly Form Recognizer. Use analyze-result polling pattern (202 + operation-location header) for async operations.",
        },
        "Azure OpenAI Service": {
            "category": "Generative AI",
            "description": "Access OpenAI GPT-4o, GPT-4, DALL-E 3, Whisper, Embeddings via Azure. Enterprise security, private networking, content filtering.",
            "endpoint": "<resource>.openai.azure.com/openai/deployments/<deploy>/",
            "tiers": "Standard | Provisioned Throughput (PTU)",
            "exam_tip": "Must create a DEPLOYMENT to use a model. Use `Azure OpenAI On Your Data` feature for RAG without custom code.",
        },
        "Azure AI Studio": {
            "category": "Generative AI",
            "description": "Unified portal: model catalog, prompt flow, evaluation, fine-tuning, AI hub projects, connections to AI services.",
            "endpoint": "ai.azure.com",
            "tiers": "Consumption-based",
            "exam_tip": "Prompt Flow is the visual orchestration tool for LLM chains. Evaluation component measures groundedness, relevance, coherence.",
        },
        "Azure AI Content Safety": {
            "category": "Generative AI",
            "description": "Detect harmful content in text and images: hate, violence, sexual, self-harm. Severity scores 0–6. Supports custom blocklists.",
            "endpoint": "cognitiveservices.azure.com/contentsafety/",
            "tiers": "Free (F0) | Standard (S0)",
            "exam_tip": "Integrates directly with Azure OpenAI deployments as a content filter policy. Also available as standalone API.",
        },
        "Azure Bot Service": {
            "category": "NLP",
            "description": "Host and manage conversational AI bots. Integrates with Teams, Slack, Webchat. Works with CLU and QnA backends.",
            "endpoint": "azure.microsoft.com/services/bot-services/",
            "tiers": "Free (F0) | Standard (S1)",
            "exam_tip": "Use Bot Framework SDK (Python/C#/JS) to build bots; Bot Service provides channels and hosting.",
        },
    }

    categories = sorted(set(s["category"] for s in services_data.values()))
    selected_cat = st.radio("Filter by category", ["All"] + categories, horizontal=True)

    filtered = {
        k: v for k, v in services_data.items()
        if selected_cat == "All" or v["category"] == selected_cat
    }

    for name, info in filtered.items():
        with st.expander(f"**{name}** – {info['category']}"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Endpoint pattern:** `{info['endpoint']}`")
            st.markdown(f"**Pricing tiers:** {info['tiers']}")
            st.markdown(f"""
<div class="tip-box">
📝 <strong>Exam Tip:</strong> {info['exam_tip']}
</div>
""", unsafe_allow_html=True)

# ── TAB: KEY CONCEPTS ────────────────────────────────────────────────────────
with tab_concepts:
    st.subheader("💡 Key Concepts & Cheat Sheets")
    for topic, points in KEY_CONCEPTS.items():
        with st.expander(f"**{topic}**"):
            for point in points:
                st.markdown(f"""
<div class="key-concept">{point}</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("🔄 Service Comparison: CLU vs QnA vs Custom NER")
    comp_data = {
        "Feature": ["Primary use", "Training data", "Output", "Replaces"],
        "CLU": ["Intent + entity extraction from utterances", "Labeled utterances with intents", "Intent label + entity spans", "LUIS"],
        "Custom Q&A": ["FAQ / knowledge base answering", "Q&A pairs from docs/URLs", "Best matching answer + confidence", "QnA Maker"],
        "Custom NER": ["Entity extraction from free text", "Labeled text with entity spans", "Entity spans + types", "Custom LUIS entity models"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(comp_data).set_index("Feature"), use_container_width=True)

    st.divider()
    st.subheader("📐 Azure OpenAI API Call Structure")
    st.code("""
import openai

client = openai.AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com/",
    api_key="<your-key>",
    api_version="2024-02-01"
)

response = client.chat.completions.create(
    model="<deployment-name>",        # NOT the model name — the DEPLOYMENT
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain Azure AI Search."}
    ],
    temperature=0.7,
    max_tokens=800,
)
print(response.choices[0].message.content)
""", language="python")

    st.subheader("🔍 Azure AI Search — Index Schema Example")
    st.code("""
{
  "name": "my-index",
  "fields": [
    {"name": "id",       "type": "Edm.String",  "key": true,  "filterable": true},
    {"name": "title",    "type": "Edm.String",  "searchable": true, "analyzer": "en.microsoft"},
    {"name": "content",  "type": "Edm.String",  "searchable": true},
    {"name": "category", "type": "Edm.String",  "filterable": true, "facetable": true},
    {"name": "embedding","type": "Collection(Edm.Single)", "dimensions": 1536,
     "vectorSearchProfile": "my-vector-profile"}
  ],
  "vectorSearch": {
    "algorithms": [{"name": "my-hnsw", "kind": "hnsw"}],
    "profiles":   [{"name": "my-vector-profile", "algorithm": "my-hnsw"}]
  }
}
""", language="json")

# ── TAB: PRACTICE QUIZ ───────────────────────────────────────────────────────
with tab_quiz:
    st.subheader("🎯 Practice Quiz")

    if not st.session_state.quiz_active and not st.session_state.quiz_completed:
        st.markdown("Configure your quiz session:")
        col_a, col_b = st.columns(2)
        with col_a:
            domain_filter = st.selectbox(
                "Domain",
                ["All Domains"] + list(DOMAINS.keys()),
                key="domain_select",
            )
        with col_b:
            num_q = st.slider("Number of questions", 5, len(QUESTIONS), 10)

        if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
            pool = QUESTIONS if domain_filter == "All Domains" else [
                q for q in QUESTIONS if q["domain"] == domain_filter
            ]
            if len(pool) < num_q:
                st.warning(f"Only {len(pool)} questions available for this domain. Using all of them.")
                num_q = len(pool)
            selected_q = random.sample(pool, num_q)
            start_quiz(selected_q)
            st.rerun()

    elif st.session_state.quiz_active and not st.session_state.quiz_completed:
        idx = st.session_state.quiz_index
        questions = st.session_state.quiz_questions
        total = len(questions)

        progress = idx / total
        st.progress(progress, text=f"Question {idx + 1} of {total}")

        q = questions[idx]
        st.markdown(f"**Domain:** `{q['domain']}`")
        st.markdown(f"### Q{idx+1}. {q['question']}")

        answer_key = f"q_{idx}"
        user_choice = st.radio(
            "Select your answer:",
            q["options"],
            key=answer_key,
            index=None,
        )

        col_sub, col_skip = st.columns([2, 1])
        submitted = col_sub.button("Submit Answer", type="primary", disabled=user_choice is None)
        skipped = col_skip.button("Skip →")

        if submitted and user_choice is not None:
            chosen_idx = q["options"].index(user_choice)
            correct = chosen_idx == q["answer"]
            if correct:
                st.session_state.quiz_score += 1
                st.markdown('<p class="correct-badge">✅ Correct!</p>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<p class="incorrect-badge">❌ Incorrect. Correct answer: <em>{q["options"][q["answer"]]}</em></p>',
                    unsafe_allow_html=True,
                )
            st.info(f"**Explanation:** {q['explanation']}")
            st.session_state.quiz_answers.append({"q": idx, "correct": correct, "chosen": chosen_idx})

            if idx + 1 >= total:
                st.session_state.quiz_completed = True
                st.session_state.quiz_active = False
            else:
                if st.button("Next Question →"):
                    st.session_state.quiz_index += 1
                    st.rerun()

        elif skipped:
            st.session_state.quiz_answers.append({"q": idx, "correct": False, "chosen": None})
            if idx + 1 >= total:
                st.session_state.quiz_completed = True
                st.session_state.quiz_active = False
            else:
                st.session_state.quiz_index += 1
            st.rerun()

    elif st.session_state.quiz_completed:
        score = st.session_state.quiz_score
        total = len(st.session_state.quiz_questions)
        pct = score / total * 100 if total else 0

        st.markdown(f"""
<div class="score-box">
  <h2>{score}/{total}</h2>
  <p style="font-size:1.4rem">Score: {pct:.0f}%</p>
  <p>{"🎉 Excellent!" if pct >= 80 else "📚 Keep studying!" if pct >= 60 else "⚠️ Review needed"}</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Question Review")
        for i, ans in enumerate(st.session_state.quiz_answers):
            q = st.session_state.quiz_questions[ans["q"]]
            icon = "✅" if ans["correct"] else "❌"
            with st.expander(f"{icon} Q{i+1}: {q['question'][:80]}..."):
                st.markdown(f"**Your answer:** {q['options'][ans['chosen']] if ans['chosen'] is not None else 'Skipped'}")
                st.markdown(f"**Correct answer:** {q['options'][q['answer']]}")
                st.markdown(f"**Explanation:** {q['explanation']}")

        if st.button("🔄 New Quiz", type="primary", use_container_width=True):
            reset_quiz()
            st.rerun()

# ── TAB: ASK AI ──────────────────────────────────────────────────────────────
with tab_ask:
    st.subheader("🤖 Ask the AI Study Tutor")
    st.markdown("Get instant AI-powered explanations, comparisons, and study help.")

    if not (st.session_state.api_key or os.getenv("ANTHROPIC_API_KEY")):
        st.warning("Enter your Anthropic API key in the sidebar to use this feature.")

    quick_prompts = [
        "Explain the difference between CLU and QnA in Azure AI Language",
        "When should I use Azure AI Search vs Azure AI Document Intelligence?",
        "How does RAG work with Azure OpenAI and Azure AI Search?",
        "What are the key responsible AI principles I need to know for AI-102?",
        "Explain Azure AI Vision Image Analysis 4.0 key features",
        "How do I secure Azure AI services with managed identity?",
        "Compare Azure OpenAI temperature vs top-p parameters",
        "What is a skillset in Azure AI Search and how does it work?",
    ]

    st.markdown("**Quick prompts:**")
    cols = st.columns(2)
    for i, p in enumerate(quick_prompts):
        if cols[i % 2].button(p, key=f"qp_{i}", use_container_width=True):
            st.session_state.ai_prompt = p

    st.divider()
    user_q = st.text_area(
        "Or type your own question:",
        value=st.session_state.get("ai_prompt", ""),
        height=100,
        placeholder="e.g. What is the difference between Azure AI Vision and Custom Vision?",
    )

    if st.button("Ask AI Tutor", type="primary", disabled=not user_q.strip()):
        with st.spinner("Generating explanation…"):
            context = (
                "You are helping a student prepare for the Microsoft Azure AI Engineer Associate (AI-102) exam. "
                f"Student question: {user_q}"
            )
            response = stream_explanation(context)
            st.session_state.ai_response = response
            st.session_state.ai_prompt = ""

    if st.session_state.ai_response:
        st.markdown("---")
        st.markdown("### AI Tutor Response")
        st.markdown(st.session_state.ai_response)
        if st.button("Clear Response"):
            st.session_state.ai_response = ""
            st.rerun()
