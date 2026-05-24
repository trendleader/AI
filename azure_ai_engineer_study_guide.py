import streamlit as st
import anthropic
import random
import os
from datetime import datetime

import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Azure AI Exam Study Guide (AI-102 & AI-103)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --azure-blue: #0078D4;
    --azure-dark: #004578;
    --azure-103: #7719AA;
    --azure-103-light: #F0E6F8;
}
.hero-banner {
    background: linear-gradient(135deg, #0078D4 0%, #004578 60%, #7719AA 100%);
    color: white; padding: 2rem 2.5rem; border-radius: 12px;
    margin-bottom: 1.5rem; text-align: center;
}
.hero-banner h1 { font-size: 2.2rem; margin: 0 0 .4rem; }
.hero-banner p  { font-size: 1.05rem; opacity: .9; margin: 0; }

.badge-102 {
    display:inline-block; background:#0078D4; color:white;
    border-radius:12px; padding:.15rem .65rem; font-size:.75rem; font-weight:700;
    margin-right:.3rem; vertical-align:middle;
}
.badge-103 {
    display:inline-block; background:#7719AA; color:white;
    border-radius:12px; padding:.15rem .65rem; font-size:.75rem; font-weight:700;
    margin-right:.3rem; vertical-align:middle;
}
.badge-both {
    display:inline-block; background:#107C10; color:white;
    border-radius:12px; padding:.15rem .65rem; font-size:.75rem; font-weight:700;
    margin-right:.3rem; vertical-align:middle;
}

.domain-card {
    background:#f8f9fa; border-left:5px solid #0078D4;
    padding:1rem 1.2rem; border-radius:6px; margin-bottom:.8rem;
}
.domain-card-103 {
    background:#f8f0ff; border-left:5px solid #7719AA;
    padding:1rem 1.2rem; border-radius:6px; margin-bottom:.8rem;
}
.domain-card h4, .domain-card-103 h4 { color:#004578; margin:0 0 .3rem; }
.domain-card-103 h4 { color:#7719AA; }
.domain-card p, .domain-card-103 p { color:#555; margin:0; font-size:.92rem; }

.service-chip {
    display:inline-block; background:#E1F0FF; color:#004578;
    border-radius:20px; padding:.25rem .75rem; margin:.2rem;
    font-size:.85rem; font-weight:600;
}
.service-chip-103 {
    display:inline-block; background:#F0E6F8; color:#7719AA;
    border-radius:20px; padding:.25rem .75rem; margin:.2rem;
    font-size:.85rem; font-weight:600;
}
.correct-badge   { color:#107C10; font-weight:700; }
.incorrect-badge { color:#D83B01; font-weight:700; }

.score-box {
    background:linear-gradient(135deg,#0078D4,#7719AA);
    color:white; border-radius:10px; padding:1.5rem; text-align:center;
}
.score-box h2 { font-size:3rem; margin:0; }

.key-concept {
    background:#fff8e1; border-left:4px solid #F7630C;
    padding:.7rem 1rem; border-radius:4px; margin-bottom:.5rem; font-size:.93rem;
}
.key-concept-103 {
    background:#f8f0ff; border-left:4px solid #7719AA;
    padding:.7rem 1rem; border-radius:4px; margin-bottom:.5rem; font-size:.93rem;
}
.tip-box {
    background:#e8f5e9; border-left:4px solid #107C10;
    padding:.7rem 1rem; border-radius:4px; margin-bottom:.5rem; font-size:.93rem;
}
.retirement-banner {
    background:#FFF4CE; border:2px solid #F7630C;
    border-radius:8px; padding:.8rem 1.2rem; margin-bottom:1rem;
}
.new-banner {
    background:#EDF8ED; border:2px solid #107C10;
    border-radius:8px; padding:.8rem 1.2rem; margin-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)

# ── AI-102 Domain Data ────────────────────────────────────────────────────────
DOMAINS_102 = {
    "Plan & Manage Azure AI Solutions (15–20%)": {
        "icon": "🏗️", "exam": "AI-102",
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
        "services": ["Azure AI Services", "Azure Key Vault", "Azure Monitor",
                     "Azure Policy", "Managed Identity", "Private Endpoint", "Azure RBAC"],
    },
    "Implement Computer Vision Solutions (15–20%)": {
        "icon": "👁️", "exam": "AI-102",
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
        "services": ["Azure AI Vision", "Custom Vision", "Face API",
                     "Azure Video Indexer", "Azure AI Studio"],
    },
    "Implement NLP Solutions (30–35%)": {
        "icon": "💬", "exam": "AI-102",
        "topics": [
            "Analyze text with Azure AI Language (sentiment, entities, key phrases)",
            "Build and deploy Conversational Language Understanding (CLU) models",
            "Create custom Named Entity Recognition (NER) models",
            "Build and deploy Azure AI Bot Service bots",
            "Implement question-answering with Custom Question Answering",
            "Translate text and documents with Azure AI Translator",
            "Implement speech-to-text and text-to-speech",
            "Implement speech translation and custom speech models",
            "Classify text with custom text classification",
            "Summarize text and extract information",
        ],
        "services": ["Azure AI Language", "Azure AI Translator", "Azure AI Speech",
                     "Azure Bot Service", "CLU", "Custom NER", "Custom Q&A"],
    },
    "Implement Knowledge Mining & Document Intelligence (10–15%)": {
        "icon": "🔍", "exam": "AI-102",
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
        "icon": "✨", "exam": "AI-102",
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

# ── AI-103 Domain Data ────────────────────────────────────────────────────────
DOMAINS_103 = {
    "Plan & Manage an Azure AI Solution (25–30%)": {
        "icon": "🏗️", "exam": "AI-103",
        "topics": [
            "Select and deploy models from Azure AI Foundry model catalog",
            "Configure Azure AI Foundry hubs, projects, and connections",
            "Implement security with managed identity, RBAC, and Key Vault",
            "Configure private networking and private endpoints for AI Foundry",
            "Monitor AI solutions with Azure Monitor, metrics, and diagnostic logs",
            "Implement Responsible AI with Azure AI Content Safety and evaluation",
            "Apply governance with Azure Policy and Microsoft Purview",
            "Configure content filtering policies for Azure OpenAI deployments",
            "Manage model deployments: Standard, Global Standard, and PTU",
            "Set up CI/CD pipelines for AI solution deployment",
        ],
        "services": ["Azure AI Foundry", "Azure Key Vault", "Azure Monitor",
                     "Azure Policy", "Managed Identity", "Azure RBAC",
                     "Azure AI Content Safety", "Microsoft Purview"],
    },
    "Implement Generative AI & Agentic Solutions (30–35%)": {
        "icon": "🤖", "exam": "AI-103",
        "topics": [
            "Build chat applications with azure.ai.projects SDK",
            "Implement RAG with hybrid Azure AI Search (keyword + vector)",
            "Design and orchestrate Prompt Flow DAG pipelines",
            "Evaluate LLM responses (groundedness, relevance, coherence, fluency)",
            "Fine-tune GPT-4o on domain-specific datasets in Azure AI Foundry",
            "Build agents with Foundry Agent Service and built-in tools",
            "Use agent tools: code_interpreter, file_search, bing_grounding, azure_ai_search, openapi_v3",
            "Orchestrate multi-agent workflows with Microsoft Agent Framework (Semantic Kernel + AutoGen)",
            "Implement Magentic-One for complex multi-agent coordination",
            "Apply Provisioned Throughput Units (PTU) for low-latency workloads",
            "Use Microsoft.Extensions.AI for cross-framework AI integration",
        ],
        "services": ["Azure OpenAI Service", "Azure AI Foundry", "Foundry Agent Service",
                     "Semantic Kernel", "AutoGen", "Magentic-One", "Prompt Flow",
                     "azure.ai.projects SDK", "Microsoft.Extensions.AI"],
    },
    "Implement Computer Vision Solutions (10–15%)": {
        "icon": "👁️", "exam": "AI-103",
        "topics": [
            "Analyze images with Azure AI Vision Image Analysis 4.0",
            "Extract text with the Read API (OCR) for documents and images",
            "Generate image descriptions with dense captioning",
            "Perform background removal and smart cropping",
            "Process video content with Azure Video Indexer",
            "Generate images with DALL-E 3 in Azure OpenAI",
            "Build multimodal applications combining vision and language models",
            "Use GPT-4o vision capabilities for image understanding",
        ],
        "services": ["Azure AI Vision", "Azure Video Indexer", "Azure OpenAI (DALL-E 3)",
                     "GPT-4o Vision", "Azure AI Foundry"],
    },
    "Implement Text Analysis Solutions (10–15%)": {
        "icon": "💬", "exam": "AI-103",
        "topics": [
            "Perform sentiment analysis, NER, key phrase extraction with Azure AI Language",
            "Detect and redact PII from text",
            "Summarize documents with extractive and abstractive summarization",
            "Classify text with custom text classification models",
            "Use LLMs for text analysis tasks via Azure OpenAI (LLM-first approach)",
            "Implement speech-to-text and text-to-speech with Azure AI Speech",
            "Build speech-enabled AI agents with real-time transcription",
            "Translate text and documents with Azure AI Translator",
        ],
        "services": ["Azure AI Language", "Azure AI Speech", "Azure AI Translator",
                     "Azure OpenAI Service", "Azure AI Foundry"],
    },
    "Implement Information Extraction Solutions (10–15%)": {
        "icon": "🔍", "exam": "AI-103",
        "topics": [
            "Build Azure AI Search indexes with hybrid search (BM25 + vector)",
            "Configure semantic ranker for intent-aware result ordering",
            "Implement vector search with embedding models and HNSW algorithm",
            "Build AI enrichment pipelines with skillsets for RAG data prep",
            "Extract structured data from documents with Azure AI Document Intelligence",
            "Use prebuilt models: Invoice, Receipt, ID, W-2, Business Card",
            "Train custom extraction and classification models",
            "Build information extraction pipelines for RAG grounding",
        ],
        "services": ["Azure AI Search", "Azure AI Document Intelligence",
                     "Cognitive Skillsets", "Semantic Ranker", "Vector Search", "HNSW"],
    },
}

# ── Practice Questions: AI-102 ────────────────────────────────────────────────
QUESTIONS_102 = [
    {
        "exam": "AI-102",
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
        "exam": "AI-102",
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
        "exam": "AI-102",
        "domain": "Plan & Manage Azure AI Solutions (15–20%)",
        "question": "Which Azure AI Responsible AI principle focuses on ensuring AI systems do not produce disparate outcomes across demographic groups?",
        "options": ["Reliability & Safety", "Privacy & Security", "Fairness", "Inclusiveness"],
        "answer": 2,
        "explanation": "Fairness requires that AI systems treat all people equitably and don't produce discriminatory outcomes based on race, gender, age, or other characteristics.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement Computer Vision Solutions (15–20%)",
        "question": "You need to extract printed and handwritten text from scanned documents at scale. Which Azure AI Vision feature should you use?",
        "options": ["Image Analysis – tagging", "Read API (OCR)", "Face API", "Custom Vision classification"],
        "answer": 1,
        "explanation": "The Read API (now part of Azure AI Vision Image Analysis 4.0) is optimized for high-accuracy OCR on both printed and handwritten text in documents and images.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement Computer Vision Solutions (15–20%)",
        "question": "A retail company wants to detect defective products on an assembly line. They have 500 labeled images per defect class. Which service is MOST appropriate?",
        "options": ["Azure AI Vision Image Analysis – dense captioning",
                    "Custom Vision – Object Detection",
                    "Face API – verification",
                    "Azure Video Indexer"],
        "answer": 1,
        "explanation": "Custom Vision Object Detection lets you train a model on your own labeled images to detect specific objects. It's designed for domain-specific training scenarios.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement NLP Solutions (30–35%)",
        "question": "A company needs to build a chatbot that maps user utterances to predefined intents and extracts entities. Which Azure AI Language feature should they use?",
        "options": ["Sentiment Analysis", "Conversational Language Understanding (CLU)",
                    "Key Phrase Extraction", "Custom Named Entity Recognition"],
        "answer": 1,
        "explanation": "CLU (Conversational Language Understanding) is the successor to LUIS. It trains a model to identify intents and extract entities from user utterances.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement NLP Solutions (30–35%)",
        "question": "You need to build a FAQ bot from existing company documents with minimal development effort. The best approach is:",
        "options": ["Train a BERT model from scratch",
                    "Use Azure AI Language Custom Question Answering",
                    "Implement CLU with intents for every FAQ item",
                    "Fine-tune GPT-4 on FAQ pairs"],
        "answer": 1,
        "explanation": "Custom Question Answering (successor to QnA Maker) lets you upload documents or FAQs and automatically extracts Q&A pairs, then exposes them via REST — minimal code required.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement NLP Solutions (30–35%)",
        "question": "Your application must transcribe real-time call center audio, then detect caller sentiment. Which TWO services should you combine?",
        "options": ["Azure AI Speech (STT) + Azure AI Language (Sentiment Analysis)",
                    "Azure Video Indexer + Custom Vision",
                    "Azure AI Translator + Face API",
                    "Azure OpenAI Whisper + Custom NER"],
        "answer": 0,
        "explanation": "Azure AI Speech converts audio to text in real time; Azure AI Language Sentiment Analysis then scores the transcribed text for positive/negative/neutral/mixed sentiment.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement Knowledge Mining & Document Intelligence (10–15%)",
        "question": "Which Azure AI Search concept enriches documents during indexing with OCR, entity extraction, and sentiment analysis?",
        "options": ["Semantic ranker", "Skillset (enrichment pipeline)", "Knowledge store", "Suggesters"],
        "answer": 1,
        "explanation": "A Skillset defines a pipeline of built-in or custom AI skills that process each document during indexing, adding enriched fields to the search index.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement Knowledge Mining & Document Intelligence (10–15%)",
        "question": "You need to extract structured data (vendor, total, line items) from thousands of PDF invoices. Which Azure AI Document Intelligence model applies?",
        "options": ["Business card model", "Invoice model", "Receipt model", "General document model"],
        "answer": 1,
        "explanation": "The prebuilt Invoice model in Azure AI Document Intelligence extracts invoice-specific fields: vendor name, invoice number, due date, line items, subtotals, and totals.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement Knowledge Mining & Document Intelligence (10–15%)",
        "question": "What does a Knowledge Store in Azure AI Search allow you to do?",
        "options": ["Cache search queries for faster response",
                    "Persist AI-enriched data to Azure Storage for downstream analytics",
                    "Store encrypted search index backups",
                    "Sync search results to Azure SQL Database"],
        "answer": 1,
        "explanation": "A Knowledge Store saves enriched content as JSON objects, tables, or file projections in Azure Storage, making it available for Power BI, ML, or other analytics.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement Generative AI Solutions (10–15%)",
        "question": "You are building a RAG solution with Azure OpenAI and Azure AI Search. What is the PRIMARY purpose of the retrieval step?",
        "options": ["To reduce the token cost of embedding generation",
                    "To ground the LLM response in authoritative documents, reducing hallucination",
                    "To replace the need for a system prompt",
                    "To increase the temperature of the model output"],
        "answer": 1,
        "explanation": "RAG retrieves relevant documents from a search index and includes them in the LLM context. This grounds responses in factual data, significantly reducing hallucinations.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement Generative AI Solutions (10–15%)",
        "question": "An application uses Azure OpenAI GPT-4o and needs the model to call an external weather API. Which feature enables this?",
        "options": ["Embeddings API", "Function calling (tool use)",
                    "Assistants API with file search", "DALL-E image generation"],
        "answer": 1,
        "explanation": "Function calling lets you describe external functions to the model. The model outputs a structured JSON call when it decides a function is needed; your code executes the actual API call.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement Generative AI Solutions (10–15%)",
        "question": "Which Azure service provided a unified portal for discovering foundation models, experimenting with prompts, and deploying AI apps (as of AI-102 scope)?",
        "options": ["Azure Machine Learning Studio", "Azure AI Studio",
                    "Azure Databricks", "Power Platform AI Builder"],
        "answer": 1,
        "explanation": "Azure AI Studio (now rebranded as Azure AI Foundry) is Microsoft's unified platform for generative AI development: model catalog, prompt flow, evaluation, and deployment.",
    },
    {
        "exam": "AI-102",
        "domain": "Plan & Manage Azure AI Solutions (15–20%)",
        "question": "Which service provides built-in content moderation for Azure OpenAI outputs, detecting hate, violence, and self-harm?",
        "options": ["Azure Policy", "Azure AI Content Safety",
                    "Microsoft Defender for Cloud", "Azure Sentinel"],
        "answer": 1,
        "explanation": "Azure AI Content Safety analyzes text and images for harmful content and can block or flag model outputs before they reach users.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement NLP Solutions (30–35%)",
        "question": "You need to identify and redact PII (social security numbers, email addresses) from text. Which Azure AI Language feature handles this?",
        "options": ["Key phrase extraction", "PII detection and redaction",
                    "Sentiment analysis", "Language detection"],
        "answer": 1,
        "explanation": "The PII detection feature in Azure AI Language identifies sensitive entities and returns redacted versions of the text with PII replaced.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement Computer Vision Solutions (15–20%)",
        "question": "Azure AI Vision Image Analysis 4.0 introduced which NEW capability not in version 3.2?",
        "options": ["OCR for printed text",
                    "Dense captions with natural-language descriptions for image regions",
                    "Color palette detection", "Celebrity recognition"],
        "answer": 1,
        "explanation": "Image Analysis 4.0 introduced dense captions, generating natural-language captions for up to 10 distinct regions of an image.",
    },
    {
        "exam": "AI-102",
        "domain": "Implement Knowledge Mining & Document Intelligence (10–15%)",
        "question": "Azure AI Search Semantic Ranker improves results by:",
        "options": ["Running BM25 full-text scoring only",
                    "Re-ranking results using language models to understand query intent",
                    "Applying custom scoring profiles based on field weights",
                    "Indexing vector embeddings for nearest-neighbor search"],
        "answer": 1,
        "explanation": "Semantic Ranker uses deep language models to re-rank BM25 results by understanding the semantic meaning of the query and documents.",
    },
]

# ── Practice Questions: AI-103 ────────────────────────────────────────────────
QUESTIONS_103 = [
    {
        "exam": "AI-103",
        "domain": "Plan & Manage an Azure AI Solution (25–30%)",
        "question": "In Azure AI Foundry, what is the relationship between a Hub and a Project?",
        "options": [
            "A Hub is a billing container; Projects are independent deployments",
            "A Hub provides shared infrastructure (compute, connections, security); Projects are workspaces for specific AI apps within a Hub",
            "A Project contains multiple Hubs for different regions",
            "A Hub is only used for model fine-tuning; Projects handle inference",
        ],
        "answer": 1,
        "explanation": "In Azure AI Foundry, a Hub provides shared resources (Azure OpenAI connections, compute, storage, Key Vault) and governance. Projects inherit Hub resources and serve as the working environment for building specific AI applications.",
    },
    {
        "exam": "AI-103",
        "domain": "Plan & Manage an Azure AI Solution (25–30%)",
        "question": "You need to serve 10 million tokens per minute to your production GPT-4o application with guaranteed latency. Which deployment type should you use?",
        "options": [
            "Standard (pay-per-token)",
            "Global Standard",
            "Provisioned Throughput Units (PTU)",
            "Serverless API",
        ],
        "answer": 2,
        "explanation": "PTU (Provisioned Throughput Units) provide dedicated compute capacity for guaranteed throughput and low, consistent latency — essential for high-volume production workloads.",
    },
    {
        "exam": "AI-103",
        "domain": "Plan & Manage an Azure AI Solution (25–30%)",
        "question": "Which Responsible AI evaluation metric measures whether an AI response is supported by the provided source documents (not hallucinated)?",
        "options": ["Coherence", "Fluency", "Groundedness", "Relevance"],
        "answer": 2,
        "explanation": "Groundedness measures whether a model's response is factually supported by the retrieved context/source documents. A grounded response makes no claims beyond what the sources support.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Generative AI & Agentic Solutions (30–35%)",
        "question": "You are building an AI agent that needs to execute Python code to analyze uploaded CSV files and return results. Which Foundry Agent Service built-in tool should you enable?",
        "options": ["bing_grounding", "file_search", "code_interpreter", "openapi_v3"],
        "answer": 2,
        "explanation": "The code_interpreter tool allows the agent to write and execute Python code in a sandboxed environment, enabling data analysis, visualization, and computation on uploaded files.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Generative AI & Agentic Solutions (30–35%)",
        "question": "Your agent must search the web for real-time information to answer user questions. Which Foundry Agent Service tool provides this capability?",
        "options": ["file_search", "azure_ai_search", "code_interpreter", "bing_grounding"],
        "answer": 3,
        "explanation": "The bing_grounding tool connects the agent to Bing Search, allowing it to retrieve real-time web search results to ground its responses in current information.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Generative AI & Agentic Solutions (30–35%)",
        "question": "You are implementing a multi-agent system where Agent A researches topics and Agent B writes reports based on Agent A's output. Which Microsoft framework is designed for this multi-agent orchestration pattern?",
        "options": [
            "Azure Logic Apps",
            "Microsoft Agent Framework (Semantic Kernel + AutoGen / Magentic-One)",
            "Azure Durable Functions",
            "Power Automate AI Builder",
        ],
        "answer": 1,
        "explanation": "Microsoft Agent Framework combines Semantic Kernel and AutoGen (formalized as Magentic-One) to enable multi-agent orchestration where specialized agents collaborate, delegate, and pass context to each other.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Generative AI & Agentic Solutions (30–35%)",
        "question": "In Prompt Flow, what is a DAG (Directed Acyclic Graph) used for?",
        "options": [
            "Defining the database schema for storing prompt results",
            "Orchestrating a sequence of LLM calls, tools, and functions with defined dependencies",
            "Training a custom language model",
            "Managing Azure AI Foundry Hub access permissions",
        ],
        "answer": 1,
        "explanation": "In Prompt Flow, a DAG defines the orchestration of nodes (LLM calls, Python functions, prompts, tools) and their data flow dependencies, enabling complex LLM application pipelines.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Generative AI & Agentic Solutions (30–35%)",
        "question": "You want to improve GPT-4o accuracy on your company's proprietary terminology. After RAG and prompt engineering, performance is still insufficient. What should you try next?",
        "options": [
            "Switch to a larger model (GPT-4o vs GPT-4o-mini)",
            "Fine-tune GPT-4o in Azure AI Foundry on domain-specific examples",
            "Increase the temperature parameter",
            "Add more documents to the knowledge store",
        ],
        "answer": 1,
        "explanation": "Fine-tuning adapts the model's weights on your domain-specific data, teaching it your terminology and response style. It's the right next step when RAG and prompt engineering are insufficient.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Generative AI & Agentic Solutions (30–35%)",
        "question": "Which SDK is the primary client library for building AI applications in Azure AI Foundry (AI-103 scope)?",
        "options": [
            "azure-cognitiveservices-language",
            "azure.ai.projects",
            "openai (PyPI)",
            "azure-search-documents",
        ],
        "answer": 1,
        "explanation": "The azure.ai.projects SDK is the unified client library for Azure AI Foundry, providing access to agents, model deployments, connections, evaluations, and other Foundry resources from Python or .NET.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Generative AI & Agentic Solutions (30–35%)",
        "question": "Hybrid search in Azure AI Search combines which two retrieval methods for optimal RAG performance?",
        "options": [
            "BM25 keyword search + semantic ranker re-ranking",
            "BM25 keyword search + vector (ANN) search with RRF fusion",
            "SQL full-text search + BERT embeddings",
            "Semantic ranker + custom scoring profiles",
        ],
        "answer": 1,
        "explanation": "Hybrid search combines BM25 (lexical) and vector (approximate nearest-neighbor) search results using Reciprocal Rank Fusion (RRF) to get the best of both: exact keyword matches and semantic similarity.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Computer Vision Solutions (10–15%)",
        "question": "A developer wants to add image understanding to their chat application using GPT-4o. What is the AI-103 recommended approach?",
        "options": [
            "Train a Custom Vision model and call it before GPT-4o",
            "Use GPT-4o's native vision capability by passing base64-encoded images in the message content",
            "Use Azure AI Vision for description, then pass text to GPT-4o",
            "Use Azure Video Indexer to process still images",
        ],
        "answer": 1,
        "explanation": "GPT-4o natively supports multimodal input. Passing base64-encoded images directly in the messages array gives the model full visual understanding without a separate vision service call.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Computer Vision Solutions (10–15%)",
        "question": "Your RAG application needs to generate images based on user descriptions. Which Azure OpenAI model should you deploy?",
        "options": ["Whisper", "GPT-4o-mini", "DALL-E 3", "text-embedding-3-large"],
        "answer": 2,
        "explanation": "DALL-E 3 is Azure OpenAI's image generation model. It creates high-quality images from text descriptions and is available as a deployment in Azure AI Foundry.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Text Analysis Solutions (10–15%)",
        "question": "For an AI-103 application, when should you prefer Azure AI Language's extractive summarization over using GPT-4o for summarization?",
        "options": [
            "When you need abstractive summaries with reasoning",
            "When cost and latency are critical and outputs must cite exact source sentences",
            "When the documents contain images to summarize",
            "When you need to translate the summary to another language",
        ],
        "answer": 1,
        "explanation": "Extractive summarization selects verbatim sentences from the source — it's deterministic, lower cost, lower latency, and provides exact citations. Use GPT-4o when you need abstractive or reasoned summaries.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Text Analysis Solutions (10–15%)",
        "question": "You are building a real-time speech-enabled AI assistant that transcribes audio AND sends the transcript to GPT-4o. Which Azure AI Speech feature enables real-time audio capture and transcription?",
        "options": [
            "Batch transcription REST API",
            "Real-time speech-to-text with Speech SDK",
            "Audio Content Safety",
            "Text-to-speech neural voices",
        ],
        "answer": 1,
        "explanation": "The Speech SDK's real-time speech-to-text provides continuous audio recognition with low latency, ideal for feeding live transcripts to an LLM for AI assistant applications.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Information Extraction Solutions (10–15%)",
        "question": "You need to prepare a large document corpus for RAG indexing. Documents include PDFs with tables and scanned images. What is the correct processing pipeline?",
        "options": [
            "Azure AI Translator → Azure AI Search",
            "Azure AI Document Intelligence → chunk text → embed → Azure AI Search",
            "Azure Blob Storage → Azure AI Language → Azure AI Search",
            "GPT-4o Vision → Azure Cosmos DB",
        ],
        "answer": 1,
        "explanation": "Azure AI Document Intelligence extracts text, tables, and structure from PDFs and scanned documents; text is then chunked, embedded (text-embedding-3-large), and indexed in Azure AI Search for RAG retrieval.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Information Extraction Solutions (10–15%)",
        "question": "In Azure AI Search, what algorithm does vector search use for approximate nearest-neighbor (ANN) retrieval?",
        "options": ["K-means clustering", "HNSW (Hierarchical Navigable Small World)",
                    "BM25 inverted index", "TF-IDF cosine similarity"],
        "answer": 1,
        "explanation": "Azure AI Search uses the HNSW (Hierarchical Navigable Small World) algorithm for vector search, providing high-recall approximate nearest-neighbor retrieval with configurable precision/speed tradeoffs.",
    },
    {
        "exam": "AI-103",
        "domain": "Plan & Manage an Azure AI Solution (25–30%)",
        "question": "Your organization requires all Azure AI Foundry traffic to stay within the corporate network. What should you configure?",
        "options": [
            "Enable public network access with IP allowlist",
            "Configure a Private Endpoint for the AI Foundry Hub and disable public access",
            "Use SAS tokens for authentication",
            "Configure Azure Front Door as a reverse proxy",
        ],
        "answer": 1,
        "explanation": "Configuring a Private Endpoint for the Azure AI Foundry Hub creates a private IP within your VNet. Disabling public network access ensures all traffic (model API calls, SDK, portal) routes through the private endpoint.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Generative AI & Agentic Solutions (30–35%)",
        "question": "Which tool in Foundry Agent Service would you use to expose your existing company REST API as an agent capability?",
        "options": ["code_interpreter", "file_search", "bing_grounding", "openapi_v3"],
        "answer": 3,
        "explanation": "The openapi_v3 tool allows you to provide an OpenAPI specification, enabling the agent to call your existing REST APIs as tools — without writing custom integration code.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Information Extraction Solutions (10–15%)",
        "question": "After indexing documents into Azure AI Search, search results are good for exact keywords but miss semantically related results. What should you add?",
        "options": [
            "Increase the number of replicas",
            "Add vector fields with embeddings and enable hybrid search",
            "Add more synonyms to the index analyzer",
            "Switch from Standard to Basic tier",
        ],
        "answer": 1,
        "explanation": "Adding vector fields (embeddings from text-embedding-3-large) and enabling hybrid search (BM25 + vector) captures semantic similarity that keyword search misses, improving recall for related concepts.",
    },
    {
        "exam": "AI-103",
        "domain": "Implement Text Analysis Solutions (10–15%)",
        "question": "Microsoft.Extensions.AI (MEAI) in the AI-103 context is used to:",
        "options": [
            "Provide a billing framework for Azure AI services",
            "Offer a vendor-neutral abstraction layer for AI services in .NET, enabling portability across providers",
            "Replace the Azure OpenAI SDK for Python",
            "Manage Azure AI Foundry Hub authentication",
        ],
        "answer": 1,
        "explanation": "Microsoft.Extensions.AI provides a unified set of .NET abstractions (IChatClient, IEmbeddingGenerator, etc.) that work across OpenAI, Azure OpenAI, Ollama, and other providers — enabling portable AI application code.",
    },
]

# ── All combined ──────────────────────────────────────────────────────────────
ALL_DOMAINS   = {**DOMAINS_102, **DOMAINS_103}
ALL_QUESTIONS = QUESTIONS_102 + QUESTIONS_103

# ── Key Concepts ──────────────────────────────────────────────────────────────
KEY_CONCEPTS_102 = {
    "Azure AI Service Endpoints (AI-102)": {
        "exam": "AI-102",
        "points": [
            "All Azure AI services expose a REST API at: `https://<resource-name>.cognitiveservices.azure.com/`",
            "Authenticate with: `Ocp-Apim-Subscription-Key` header or Bearer token (managed identity)",
            "Multi-service resource supports all AI services with one endpoint + key",
            "Single-service resources offer custom domain names and independent quotas",
        ],
    },
    "Responsible AI Principles (Both Exams)": {
        "exam": "BOTH",
        "points": [
            "**Fairness** – AI must not discriminate based on protected characteristics",
            "**Reliability & Safety** – AI must perform consistently and fail gracefully",
            "**Privacy & Security** – AI must protect personal data",
            "**Inclusiveness** – AI must empower all people including those with disabilities",
            "**Transparency** – People must understand how AI makes decisions",
            "**Accountability** – Humans must be responsible for AI systems",
        ],
    },
    "Azure OpenAI Key Terms (Both Exams)": {
        "exam": "BOTH",
        "points": [
            "**Deployment** – A named instance of a model (e.g., gpt-4o-deployment) — use deployment name, not model name",
            "**Temperature (0–2)** – Controls randomness; 0 = deterministic, 2 = creative",
            "**Top-p** – Nucleus sampling; alternative to temperature",
            "**System message** – Sets the model's persona and instructions",
            "**Few-shot examples** – Example turns provided in conversation history",
            "**Function calling** – Model returns structured JSON to trigger external tool/API",
        ],
    },
    "Azure AI Search Architecture (Both Exams)": {
        "exam": "BOTH",
        "points": [
            "**Index** – Schema defining searchable fields and their types",
            "**Indexer** – Crawler that pulls data from a data source into the index",
            "**Skillset** – Pipeline of AI enrichments (OCR, NER, translation, embeddings)",
            "**Knowledge Store** – Persists enriched data to Azure Storage",
            "**Semantic Ranker** – Language-model re-ranking (Standard tier+)",
            "**Vector Search** – HNSW nearest-neighbor search on embedding fields",
            "**Hybrid Search** – BM25 + vector with RRF fusion (AI-103 emphasis)",
        ],
    },
    "CLU vs QnA vs Custom NER (AI-102)": {
        "exam": "AI-102",
        "points": [
            "**CLU** – Maps utterances to intents + extracts entities. Replaces LUIS. Use for chatbot command understanding.",
            "**Custom Q&A** – Returns best answer from knowledge base docs. Replaces QnA Maker. Use for FAQ bots.",
            "**Custom NER** – Identifies domain-specific entities in free text (contracts, medical records).",
            "**Custom Text Classification** – Assigns text to one or more custom categories.",
        ],
    },
    "Document Intelligence Models (Both Exams)": {
        "exam": "BOTH",
        "points": [
            "**Prebuilt Invoice** – Vendor, invoice #, line items, totals",
            "**Prebuilt Receipt** – Merchant, items, totals, date",
            "**Prebuilt ID Document** – Passports, driver's licenses",
            "**Prebuilt Business Card** – Name, title, company, contact info",
            "**Prebuilt W-2 / Tax** – US tax form fields",
            "**General Document** – Layout, key-value pairs, tables (any document)",
            "**Custom Extraction** – Train on your own labeled forms",
            "**Custom Classification** – Route document type before extraction",
        ],
    },
}

KEY_CONCEPTS_103 = {
    "Azure AI Foundry Architecture (AI-103)": {
        "exam": "AI-103",
        "points": [
            "**Hub** – Shared infrastructure: Azure OpenAI connections, compute, storage, Key Vault, governance",
            "**Project** – Working environment for a specific AI app; inherits Hub resources",
            "**Connections** – Configured links to Azure OpenAI, AI Search, storage, external APIs",
            "**Model Catalog** – Browse and deploy 1,700+ models (OpenAI, Meta, Mistral, Phi, etc.)",
            "**Prompt Flow** – Visual DAG orchestration tool for LLM pipelines",
            "**Evaluation** – Automated LLM quality metrics: groundedness, relevance, coherence, fluency",
        ],
    },
    "Foundry Agent Service Tools (AI-103)": {
        "exam": "AI-103",
        "points": [
            "**code_interpreter** – Execute Python in sandbox; analyze files, generate plots",
            "**file_search** – Vector search over uploaded documents (PDFs, docs, etc.)",
            "**bing_grounding** – Real-time web search via Bing for current information",
            "**azure_ai_search** – Query your Azure AI Search index as a grounding tool",
            "**openapi_v3** – Call any REST API described by an OpenAPI spec",
            "**computer_use_preview** – Control desktop/browser UI (experimental)",
        ],
    },
    "Microsoft Agent Framework (AI-103)": {
        "exam": "AI-103",
        "points": [
            "**Semantic Kernel** – Plugin-based orchestration framework (Python, C#, Java)",
            "**AutoGen** – Multi-agent conversation framework with role-based agents",
            "**Magentic-One** – Microsoft's production-grade multi-agent system (Orchestrator + specialized agents)",
            "**Microsoft.Extensions.AI (MEAI)** – .NET abstraction layer for AI providers (portable code)",
            "**Agent patterns**: ReAct (reason + act), Orchestrator-Worker, Critic-Refinement",
        ],
    },
    "RAG Pipeline Architecture (AI-103)": {
        "exam": "AI-103",
        "points": [
            "**Step 1 – Ingest**: Extract text with Azure AI Document Intelligence",
            "**Step 2 – Chunk**: Split documents into 512–1024 token chunks with overlap",
            "**Step 3 – Embed**: Generate embeddings with text-embedding-3-large (1536 dims)",
            "**Step 4 – Index**: Store chunks + embeddings in Azure AI Search (hybrid index)",
            "**Step 5 – Retrieve**: Hybrid search (BM25 + vector) + semantic ranker",
            "**Step 6 – Generate**: Pass retrieved chunks as context to GPT-4o",
            "**Evaluate**: Measure groundedness, relevance, coherence with Prompt Flow evaluation",
        ],
    },
    "PTU vs Standard Deployments (AI-103)": {
        "exam": "AI-103",
        "points": [
            "**Standard** – Pay per 1K tokens; variable latency; best for dev/test or unpredictable traffic",
            "**Global Standard** – Routes to lowest-latency Azure region; same pricing as Standard",
            "**PTU (Provisioned Throughput Units)** – Reserved compute; guaranteed throughput and latency SLA; best for production",
            "**Serverless API** – Marketplace models billed per token; no deployment needed",
            "PTU is measured in tokens-per-minute (TPM); size based on peak load requirements",
        ],
    },
    "Prompt Flow Evaluation Metrics (AI-103)": {
        "exam": "AI-103",
        "points": [
            "**Groundedness** – Is the response supported by the retrieved context? (anti-hallucination)",
            "**Relevance** – Does the response address the user's question?",
            "**Coherence** – Is the response logically structured and consistent?",
            "**Fluency** – Is the language grammatically correct and natural?",
            "**Similarity** – How close is the response to a ground-truth reference?",
            "All metrics scored 1–5 by an LLM-as-judge or by human raters",
        ],
    },
}

# ── Services Data ─────────────────────────────────────────────────────────────
SERVICES_DATA = {
    "Azure AI Vision": {
        "category": "Computer Vision", "exams": ["AI-102", "AI-103"],
        "description": "Analyze images and video. Features: image captioning, dense captions, object detection, OCR (Read API), smart crop, background removal.",
        "endpoint": "cognitiveservices.azure.com/vision/v3.2/",
        "tiers": "Free (S0) | Standard (S1)",
        "exam_tip": "Image Analysis 4.0 uses a single /analyze endpoint with visual features. AI-103 prefers GPT-4o vision for richer multimodal understanding.",
    },
    "Azure Custom Vision": {
        "category": "Computer Vision", "exams": ["AI-102"],
        "description": "Train custom image classifiers and object detectors. Exports to ONNX, CoreML, TensorFlow.",
        "endpoint": "customvision.ai",
        "tiers": "Free (F0) | Standard (S0)",
        "exam_tip": "AI-102 only. Needs minimum 15 images per tag for classification, 15 bounding boxes per tag for detection.",
    },
    "Azure Face API": {
        "category": "Computer Vision", "exams": ["AI-102"],
        "description": "Detect, analyze, and identify human faces. Operations: Detect, Find Similar, Group, Identify, Verify.",
        "endpoint": "cognitiveservices.azure.com/face/v1.0/",
        "tiers": "Free | Standard",
        "exam_tip": "AI-102 only. Limited Access — requires approval for identification/verification features due to responsible AI policies.",
    },
    "Azure AI Language": {
        "category": "NLP / Text Analysis", "exams": ["AI-102", "AI-103"],
        "description": "Unified NLP service: sentiment, key phrases, NER, PII detection, language detection, CLU, custom NER, Custom Q&A, summarization.",
        "endpoint": "cognitiveservices.azure.com/language/:analyze-text",
        "tiers": "Free (F0) | Standard (S)",
        "exam_tip": "Successor to Text Analytics + LUIS + QnA Maker. AI-103 also accepts GPT-4o for LLM-first text analysis tasks.",
    },
    "Azure AI Translator": {
        "category": "NLP / Text Analysis", "exams": ["AI-102", "AI-103"],
        "description": "Neural machine translation. Text translation, document translation, dictionary lookup, transliteration. 100+ languages.",
        "endpoint": "api.cognitive.microsofttranslator.com",
        "tiers": "Free (F0) | Standard (S1)",
        "exam_tip": "Document Translation is async and uses Azure Blob Storage as source/target. Supports glossaries for custom terminology.",
    },
    "Azure AI Speech": {
        "category": "NLP / Text Analysis", "exams": ["AI-102", "AI-103"],
        "description": "STT (batch + real-time), TTS (neural voices), speech translation, speaker recognition, custom speech, custom voice.",
        "endpoint": "cognitiveservices.azure.com/speech/",
        "tiers": "Free (F0) | Standard (S0)",
        "exam_tip": "AI-103: Use Speech SDK real-time STT to feed live transcripts to Foundry agents. Batch Transcription REST API for large audio files.",
    },
    "Azure AI Search": {
        "category": "Information Extraction", "exams": ["AI-102", "AI-103"],
        "description": "Full-text + vector search. Components: indexes, indexers, skillsets, knowledge store, semantic ranker, hybrid search.",
        "endpoint": "<search-name>.search.windows.net",
        "tiers": "Free | Basic | Standard (S1/S2/S3)",
        "exam_tip": "AI-103 emphasis: hybrid search (BM25+vector) with RRF fusion. Semantic ranker requires Standard tier+. HNSW for ANN vector search.",
    },
    "Azure AI Document Intelligence": {
        "category": "Information Extraction", "exams": ["AI-102", "AI-103"],
        "description": "Extract structured data from documents. Prebuilt models (invoices, receipts, IDs) + custom extraction + classification.",
        "endpoint": "cognitiveservices.azure.com/formrecognizer/",
        "tiers": "Free (F0) | Standard (S0)",
        "exam_tip": "Key RAG role (AI-103): extracts text, tables, and layout from PDFs/images for the ingest step of a RAG pipeline.",
    },
    "Azure OpenAI Service": {
        "category": "Generative AI", "exams": ["AI-102", "AI-103"],
        "description": "GPT-4o, GPT-4, DALL-E 3, Whisper, Embeddings via Azure. Enterprise security, private networking, content filtering.",
        "endpoint": "<resource>.openai.azure.com/openai/deployments/<deploy>/",
        "tiers": "Standard | Global Standard | PTU",
        "exam_tip": "AI-103: Must use DEPLOYMENT name (not model name). PTU for guaranteed latency SLA. azure.ai.projects SDK wraps OpenAI client in Foundry.",
    },
    "Azure AI Foundry": {
        "category": "Generative AI", "exams": ["AI-103"],
        "description": "Unified platform (successor to Azure AI Studio): model catalog, Prompt Flow, evaluations, fine-tuning, Foundry Agent Service, AI hub + projects.",
        "endpoint": "ai.azure.com / <hub>.api.azureml.ms",
        "tiers": "Consumption-based",
        "exam_tip": "AI-103 primary portal. Know Hub vs Project distinction. azure.ai.projects SDK is the main programmatic interface.",
    },
    "Foundry Agent Service": {
        "category": "Generative AI", "exams": ["AI-103"],
        "description": "Managed service for building AI agents. Built-in tools: code_interpreter, file_search, bing_grounding, azure_ai_search, openapi_v3, computer_use_preview.",
        "endpoint": "via azure.ai.projects SDK",
        "tiers": "Consumption-based (uses underlying model deployment tokens)",
        "exam_tip": "Agents run asynchronously via Threads and Runs pattern. Poll run status until completed/failed. Each tool has specific setup requirements.",
    },
    "Azure AI Content Safety": {
        "category": "Generative AI", "exams": ["AI-102", "AI-103"],
        "description": "Detect harmful content: hate, violence, sexual, self-harm. Severity 0–6. Custom blocklists. Integrates as Azure OpenAI content filter.",
        "endpoint": "cognitiveservices.azure.com/contentsafety/",
        "tiers": "Free (F0) | Standard (S0)",
        "exam_tip": "AI-103: Configure content filter policies per Azure OpenAI deployment in AI Foundry. Groundedness detection also available as a feature.",
    },
    "Azure Bot Service": {
        "category": "NLP / Text Analysis", "exams": ["AI-102"],
        "description": "Host and manage conversational AI bots. Integrates with Teams, Slack, Webchat. Works with CLU and QnA backends.",
        "endpoint": "azure.microsoft.com/services/bot-services/",
        "tiers": "Free (F0) | Standard (S1)",
        "exam_tip": "AI-102 only. Use Bot Framework SDK (Python/C#/JS) to build; Bot Service handles channels. AI-103 favors Foundry Agent Service for new builds.",
    },
    "Semantic Kernel": {
        "category": "Generative AI", "exams": ["AI-103"],
        "description": "Open-source SDK for building AI agents and LLM applications. Plugin system, memory, planners, multi-agent support (Python, C#, Java).",
        "endpoint": "github.com/microsoft/semantic-kernel",
        "tiers": "Open source (compute costs from underlying AI services)",
        "exam_tip": "Core of Microsoft Agent Framework. Plugins wrap tools/APIs. KernelFunction decorator exposes functions to the LLM for auto-invocation.",
    },
}

# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "api_key": "",
    "exam_mode": "Both (AI-102 + AI-103)",
    "quiz_active": False,
    "quiz_questions": [],
    "quiz_index": 0,
    "quiz_score": 0,
    "quiz_answers": [],
    "quiz_completed": False,
    "ai_response": "",
    "ai_prompt": "",
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


def stream_explanation(prompt: str, system_msg: str) -> str:
    client = get_client()
    if not client:
        return "⚠️ Please enter your Anthropic API key in the sidebar to get AI explanations."
    full = ""
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
        system=system_msg,
    ) as stream:
        for text in stream.text_stream:
            full += text
    return full


def get_active_questions():
    mode = st.session_state.exam_mode
    if mode == "AI-102 Only":
        return QUESTIONS_102
    if mode == "AI-103 Only":
        return QUESTIONS_103
    return ALL_QUESTIONS


def get_active_domains():
    mode = st.session_state.exam_mode
    if mode == "AI-102 Only":
        return DOMAINS_102
    if mode == "AI-103 Only":
        return DOMAINS_103
    return ALL_DOMAINS


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


def exam_badge(exam: str) -> str:
    if exam == "AI-102":
        return '<span class="badge-102">AI-102</span>'
    if exam == "AI-103":
        return '<span class="badge-103">AI-103</span>'
    return '<span class="badge-both">AI-102 + AI-103</span>'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Exam Mode")
    exam_mode = st.radio(
        "Study for:",
        ["Both (AI-102 + AI-103)", "AI-102 Only", "AI-103 Only"],
        index=["Both (AI-102 + AI-103)", "AI-102 Only", "AI-103 Only"].index(
            st.session_state.exam_mode
        ),
    )
    if exam_mode != st.session_state.exam_mode:
        st.session_state.exam_mode = exam_mode
        reset_quiz()
        st.rerun()

    st.divider()
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
        st.success("Key saved.")

    st.divider()
    mode = st.session_state.exam_mode
    if mode in ("Both (AI-102 + AI-103)", "AI-102 Only"):
        st.markdown("### 📊 AI-102 Domain Weights")
        st.markdown("""
| Domain | Weight |
|--------|--------|
| Plan & Manage AI | 15–20% |
| Computer Vision | 15–20% |
| NLP | 30–35% |
| Knowledge Mining | 10–15% |
| Generative AI | 10–15% |
""")
    if mode in ("Both (AI-102 + AI-103)", "AI-103 Only"):
        st.markdown("### 📊 AI-103 Domain Weights")
        st.markdown("""
| Domain | Weight |
|--------|--------|
| Plan & Manage AI | 25–30% |
| Gen AI & Agents | 30–35% |
| Computer Vision | 10–15% |
| Text Analysis | 10–15% |
| Info Extraction | 10–15% |
""")

    st.divider()
    st.markdown("### 📌 Quiz Stats")
    answered = len([a for a in st.session_state.quiz_answers if a is not None])
    correct = st.session_state.quiz_score
    st.metric("Questions Answered", answered)
    st.metric("Correct Answers", correct)
    if answered:
        st.metric("Running Accuracy", f"{correct / answered * 100:.0f}%")

# ── Hero Banner ───────────────────────────────────────────────────────────────
mode = st.session_state.exam_mode
if mode == "AI-102 Only":
    subtitle = "AI-102 Exam Study Guide · Retiring June 30, 2026"
elif mode == "AI-103 Only":
    subtitle = "AI-103 Exam Study Guide · Azure AI Apps & Agents Developer Associate"
else:
    subtitle = "AI-102 & AI-103 Combined Study Guide · 2026 Edition"

st.markdown(f"""
<div class="hero-banner">
  <h1>🤖 Azure AI Engineer Study Guide</h1>
  <p>{subtitle}</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_home, tab_compare, tab_domains, tab_services, tab_concepts, tab_quiz, tab_ask = st.tabs([
    "🏠 Home",
    "🔄 AI-102 vs AI-103",
    "📚 Domains",
    "☁️ Services",
    "💡 Key Concepts",
    "🎯 Practice Quiz",
    "🤖 Ask AI",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: HOME
# ═══════════════════════════════════════════════════════════════════════════════
with tab_home:
    # Retirement / new exam alerts
    st.markdown("""
<div class="retirement-banner">
⚠️ <strong>AI-102 Retirement Notice:</strong> The Azure AI Engineer Associate (AI-102) exam retires on
<strong>June 30, 2026</strong>. If you haven't taken it yet, consider preparing for AI-103 instead.
</div>
<div class="new-banner">
✅ <strong>AI-103 is LIVE (Beta):</strong> "Developing AI Apps and Agents on Azure" launched April 21, 2026.
GA expected June 2026. This is the direct successor to AI-102.
</div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    active_q = get_active_questions()
    active_d = get_active_domains()
    c1.metric("Practice Questions", len(active_q))
    c2.metric("Exam Domains", len(active_d))
    c3.metric("Passing Score", "700 / 1000")
    c4.metric("Questions on Exam", "40–60")

    st.markdown("---")

    col_102, col_103 = st.columns(2)
    with col_102:
        st.markdown("### 🔵 AI-102 — Azure AI Engineer Associate")
        st.markdown("""
**Status:** Retiring June 30, 2026

**Focus:** Broad coverage of Azure AI services — Custom Vision, Face API, Bot Service, NLP, Knowledge Mining, plus foundational Generative AI.

**Format:**
- 40–60 questions · 180 minutes
- Multiple choice, case studies, drag-and-drop

**Prerequisite skills:**
- Azure fundamentals
- Python or C# proficiency
- REST API concepts
        """)

    with col_103:
        st.markdown("### 🟣 AI-103 — Azure AI Apps & Agents Developer")
        st.markdown("""
**Status:** Beta (GA June 2026)

**Focus:** Deep-dive into Generative AI, agentic workflows, Microsoft Agent Framework, Azure AI Foundry, and RAG-based information extraction.

**Format:**
- 40–60 questions · 120 minutes *(shorter!)*
- Scenario-driven, hands-on developer focus

**Prerequisite skills:**
- Azure fundamentals
- Python or C# proficiency
- Familiarity with LLMs and REST APIs
        """)

    st.markdown("---")
    mode = st.session_state.exam_mode
    if mode in ("Both (AI-102 + AI-103)", "AI-102 Only"):
        st.subheader("📅 AI-102 Study Plan (6 Weeks)")
        plan_102 = {
            "Week 1": "Azure AI Services overview, security, responsible AI, monitoring",
            "Week 2": "Computer Vision – Image Analysis 4.0, Custom Vision, Face API, Video Indexer",
            "Week 3": "NLP – Language service, CLU, Custom Q&A, Translator, Speech, Bot Service",
            "Week 4": "Knowledge Mining – AI Search, skillsets, Document Intelligence",
            "Week 5": "Generative AI – Azure OpenAI, prompt engineering, RAG, AI Studio",
            "Week 6": "Full practice tests, review weak domains, hands-on labs",
        }
        for week, focus in plan_102.items():
            st.markdown(f'<div class="domain-card"><h4>{week}</h4><p>{focus}</p></div>',
                        unsafe_allow_html=True)

    if mode in ("Both (AI-102 + AI-103)", "AI-103 Only"):
        st.subheader("📅 AI-103 Study Plan (6 Weeks)")
        plan_103 = {
            "Week 1": "Azure AI Foundry – Hubs, Projects, connections, security, PTU vs Standard",
            "Week 2": "Generative AI deep-dive – azure.ai.projects SDK, Prompt Flow DAGs, evaluation metrics",
            "Week 3": "Agentic solutions – Foundry Agent Service, built-in tools, multi-agent frameworks",
            "Week 4": "Information Extraction – Hybrid search, RAG pipeline, Document Intelligence for ingest",
            "Week 5": "Computer Vision + Text Analysis – GPT-4o vision, DALL-E 3, Speech SDK integration",
            "Week 6": "Full practice tests, evaluate RAG pipelines end-to-end, hands-on Foundry labs",
        }
        for week, focus in plan_103.items():
            st.markdown(f'<div class="domain-card-103"><h4>{week}</h4><p>{focus}</p></div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔗 Official Resources")
    st.markdown("""
- [AI-102 Exam Page](https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-engineer/) — Retiring June 30, 2026
- [AI-103 Exam Page](https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-apps-and-agents-developer-associate/) — Beta / New
- [AI-103 Study Guide (Microsoft Learn)](https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/ai-103)
- [Azure AI Foundry Portal](https://ai.azure.com)
- [azure.ai.projects SDK Docs](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/sdk-overview)
- [Microsoft Agent Framework Announcement](https://visualstudiomagazine.com/articles/2025/10/01/semantic-kernel-autogen--open-source-microsoft-agent-framework.aspx)
- [Free Practice Assessments on Microsoft Learn](https://learn.microsoft.com/en-us/certifications/practice-assessments-for-microsoft-certifications)
""")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: AI-102 vs AI-103 COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("🔄 AI-102 vs AI-103 — Side-by-Side Comparison")

    comp = {
        "Attribute": [
            "Exam number", "Certification title", "Status", "Exam duration",
            "Primary focus", "Portal/Platform", "Main SDK",
            "Agent framework", "Computer Vision approach",
            "NLP approach", "Information Extraction",
            "Generative AI weight", "Retirement/GA",
        ],
        "AI-102": [
            "AI-102", "Azure AI Engineer Associate", "Retiring June 30, 2026", "180 minutes",
            "Broad Azure AI service coverage", "Azure AI Studio (now Foundry)", "azure-cognitiveservices-* SDKs",
            "Basic agents via Assistants API", "Custom Vision + Face API + AI Vision",
            "CLU + Custom Q&A + Bot Service + Translator + Speech", "Azure AI Search + Document Intelligence (separate topics)",
            "10–15% of exam", "Retiring June 30, 2026",
        ],
        "AI-103": [
            "AI-103", "Azure AI Apps & Agents Developer Associate", "Beta (GA June 2026)", "120 minutes",
            "Generative AI, agentic workflows, RAG at depth", "Azure AI Foundry", "azure.ai.projects SDK",
            "Foundry Agent Service + Semantic Kernel + AutoGen + Magentic-One",
            "GPT-4o Vision + DALL-E 3 + AI Vision (lighter coverage of Custom Vision/Face)",
            "Azure AI Language + Azure AI Speech + Azure AI Translator (LLM-first approach)",
            "Unified as 'Information Extraction': hybrid AI Search + Document Intelligence for RAG",
            "30–35% of exam (largest domain)", "GA expected June 2026",
        ],
    }
    df_comp = pd.DataFrame(comp).set_index("Attribute")
    st.dataframe(df_comp, use_container_width=True)

    st.divider()
    st.subheader("📌 What's New in AI-103 vs AI-102")

    col_new, col_gone = st.columns(2)
    with col_new:
        st.markdown("#### ✅ New / Expanded in AI-103")
        new_items = [
            ("Azure AI Foundry Hub + Project model", "AI-103"),
            ("azure.ai.projects SDK", "AI-103"),
            ("Foundry Agent Service + all built-in tools", "AI-103"),
            ("Microsoft Agent Framework (Semantic Kernel + AutoGen)", "AI-103"),
            ("Magentic-One multi-agent orchestration", "AI-103"),
            ("Microsoft.Extensions.AI (.NET abstraction)", "AI-103"),
            ("PTU deployments — throughput guarantees", "AI-103"),
            ("Prompt Flow DAG evaluation (groundedness, relevance…)", "AI-103"),
            ("GPT-4o fine-tuning in Foundry", "AI-103"),
            ("Hybrid search (BM25 + vector) with RRF", "AI-103"),
            ("RAG pipeline architecture as a first-class topic", "AI-103"),
            ("GPT-4o vision for multimodal chat applications", "AI-103"),
        ]
        for item, badge in new_items:
            st.markdown(f'<span class="badge-103">{badge}</span> {item}', unsafe_allow_html=True)

    with col_gone:
        st.markdown("#### ⬇️ Reduced / Removed in AI-103")
        reduced = [
            "Azure Custom Vision training (lighter or absent)",
            "Face API (identification/verification)",
            "Azure Bot Service (replaced by Foundry Agent Service)",
            "CLU / Custom Q&A as primary chatbot pattern",
            "AI-102-style Knowledge Mining as separate domain",
            "Azure Video Indexer (less emphasis)",
            "Multi-step OCR pipeline (replaced by Doc Intelligence for RAG)",
        ]
        for item in reduced:
            st.markdown(f"- {item}")

    st.divider()
    st.subheader("🎯 Which Exam Should I Take?")
    st.markdown("""
| Scenario | Recommendation |
|----------|---------------|
| Taking exam before June 30, 2026 | Either works — AI-102 material is still valid |
| Starting prep today for long-term career | **AI-103** — it's the future of the cert |
| Already have AI-102 scheduled | Stick with AI-102; AI-103 complements rather than cancels it |
| Focus is agentic AI / LLM development | **AI-103** — much deeper coverage |
| Focus is NLP, Custom Vision, Bot Service | **AI-102** — broader coverage of those services |
| Want to take BOTH | AI-103 subsumes most of AI-102's generative AI + search topics |
""")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: DOMAINS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_domains:
    st.subheader("📚 Exam Domains & Learning Objectives")
    active_domains = get_active_domains()

    selected_domain = st.selectbox("Select a domain to explore", list(active_domains.keys()))
    domain = active_domains[selected_domain]
    exam_lbl = domain["exam"]

    st.markdown(
        f"### {domain['icon']} {exam_badge(exam_lbl)} {selected_domain}",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("**Learning Objectives:**")
        for t in domain["topics"]:
            st.markdown(f"- {t}")

    with col_right:
        st.markdown("**Key Azure Services:**")
        chip_class = "service-chip-103" if exam_lbl == "AI-103" else "service-chip"
        chips = " ".join(f'<span class="{chip_class}">{s}</span>' for s in domain["services"])
        st.markdown(chips, unsafe_allow_html=True)

    st.divider()
    active_q = get_active_questions()
    domain_q = [q for q in active_q if q["domain"] == selected_domain]
    st.info(f"**{len(domain_q)}** practice questions available for this domain.")
    if domain_q and st.button(f"Start {domain['icon']} Domain Quiz", key="domain_quiz_btn"):
        start_quiz(random.sample(domain_q, len(domain_q)))
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: SERVICES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_services:
    st.subheader("☁️ Azure AI Services Reference")

    mode = st.session_state.exam_mode
    categories = sorted(set(s["category"] for s in SERVICES_DATA.values()))
    selected_cat = st.radio("Filter by category", ["All"] + categories, horizontal=True)

    exam_filter = st.radio("Filter by exam", ["All Exams", "AI-102 Only", "AI-103 Only",
                                               "Both Exams"], horizontal=True)

    for name, info in SERVICES_DATA.items():
        if selected_cat != "All" and info["category"] != selected_cat:
            continue
        if exam_filter == "AI-102 Only" and "AI-102" not in info["exams"]:
            continue
        if exam_filter == "AI-103 Only" and "AI-103" not in info["exams"]:
            continue
        if exam_filter == "Both Exams" and set(info["exams"]) != {"AI-102", "AI-103"}:
            continue

        badges = " ".join(exam_badge(e) for e in info["exams"])
        with st.expander(f"**{name}** – {info['category']}"):
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Endpoint pattern:** `{info['endpoint']}`")
            st.markdown(f"**Pricing tiers:** {info['tiers']}")
            st.markdown(f'<div class="tip-box">📝 <strong>Exam Tip:</strong> {info["exam_tip"]}</div>',
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: KEY CONCEPTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_concepts:
    st.subheader("💡 Key Concepts & Cheat Sheets")
    mode = st.session_state.exam_mode
    all_concepts = {**KEY_CONCEPTS_102, **KEY_CONCEPTS_103}

    for topic, data in all_concepts.items():
        exam_lbl = data["exam"]
        if mode == "AI-102 Only" and exam_lbl == "AI-103":
            continue
        if mode == "AI-103 Only" and exam_lbl == "AI-102":
            continue
        css_class = "key-concept-103" if exam_lbl == "AI-103" else "key-concept"
        with st.expander(f"{exam_badge(exam_lbl)} **{topic}**", expanded=False):
            st.markdown("", unsafe_allow_html=True)
            for point in data["points"]:
                st.markdown(f'<div class="{css_class}">{point}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("📐 azure.ai.projects SDK — Agent Example (AI-103)")
    st.code("""
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

client = AIProjectClient(
    endpoint="https://<hub-name>.api.azureml.ms",
    credential=DefaultAzureCredential(),
)

# Create an agent with built-in tools
agent = client.agents.create_agent(
    model="gpt-4o",                        # Deployment name
    name="research-agent",
    instructions="You are a research assistant. Use tools to answer questions.",
    tools=[{"type": "bing_grounding"},      # Real-time web search
           {"type": "code_interpreter"}],   # Execute Python
)

# Create a thread and send a message
thread = client.agents.create_thread()
client.agents.create_message(thread.id, role="user", content="What is today's MSFT stock price?")

# Run the agent
run = client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
messages = client.agents.list_messages(thread_id=thread.id)
print(messages.data[0].content[0].text.value)
""", language="python")

    st.subheader("📐 Azure OpenAI API — Both Exams")
    st.code("""
import openai

client = openai.AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com/",
    api_key="<your-key>",           # Or use DefaultAzureCredential
    api_version="2024-12-01-preview"
)

# Chat completion (text)
response = client.chat.completions.create(
    model="<deployment-name>",      # DEPLOYMENT name, NOT model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain Azure AI Foundry."}
    ],
    temperature=0.7,
    max_tokens=800,
)

# Multimodal (AI-103): pass image in content array
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": [
        {"type": "text",      "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<b64>"}},
    ]}],
)
""", language="python")

    st.subheader("🔍 Azure AI Search — Hybrid Index Schema (AI-103)")
    st.code("""
{
  "name": "rag-index",
  "fields": [
    {"name": "id",        "type": "Edm.String",             "key": true},
    {"name": "content",   "type": "Edm.String",             "searchable": true},
    {"name": "source",    "type": "Edm.String",             "filterable": true},
    {"name": "embedding", "type": "Collection(Edm.Single)", "dimensions": 1536,
     "vectorSearchProfile": "hnsw-profile"}
  ],
  "vectorSearch": {
    "algorithms": [{"name": "hnsw-algo", "kind": "hnsw",
                    "parameters": {"m": 4, "efConstruction": 400}}],
    "profiles":   [{"name": "hnsw-profile", "algorithm": "hnsw-algo"}]
  },
  "semantic": {
    "configurations": [{"name": "semantic-config",
                         "prioritizedFields": {"contentFields": [{"fieldName": "content"}]}}]
  }
}
""", language="json")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: PRACTICE QUIZ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_quiz:
    st.subheader("🎯 Practice Quiz")
    active_q = get_active_questions()
    active_domains = get_active_domains()

    if not st.session_state.quiz_active and not st.session_state.quiz_completed:
        st.markdown("Configure your quiz session:")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            exam_q_filter = st.selectbox(
                "Exam filter",
                ["All"] + (["AI-102"] if st.session_state.exam_mode != "AI-103 Only" else []) +
                           (["AI-103"] if st.session_state.exam_mode != "AI-102 Only" else []),
            )
        with col_b:
            domain_filter = st.selectbox(
                "Domain",
                ["All Domains"] + list(active_domains.keys()),
                key="domain_select",
            )
        with col_c:
            max_q = len(active_q)
            num_q = st.slider("Number of questions", 5, max(5, max_q), min(10, max_q))

        if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
            pool = active_q
            if exam_q_filter != "All":
                pool = [q for q in pool if q.get("exam") == exam_q_filter]
            if domain_filter != "All Domains":
                pool = [q for q in pool if q["domain"] == domain_filter]
            if len(pool) < num_q:
                st.warning(f"Only {len(pool)} questions available. Using all.")
                num_q = len(pool)
            if pool:
                start_quiz(random.sample(pool, num_q))
                st.rerun()
            else:
                st.error("No questions match your filters.")

    elif st.session_state.quiz_active and not st.session_state.quiz_completed:
        idx = st.session_state.quiz_index
        questions = st.session_state.quiz_questions
        total = len(questions)

        st.progress(idx / total, text=f"Question {idx + 1} of {total}")

        q = questions[idx]
        st.markdown(
            f"{exam_badge(q.get('exam', 'AI-102'))} **Domain:** `{q['domain']}`",
            unsafe_allow_html=True,
        )
        st.markdown(f"### Q{idx + 1}. {q['question']}")

        user_choice = st.radio("Select your answer:", q["options"], key=f"q_{idx}", index=None)

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
                    f'<p class="incorrect-badge">❌ Incorrect. Correct answer: '
                    f'<em>{q["options"][q["answer"]]}</em></p>',
                    unsafe_allow_html=True,
                )
            st.info(f"**Explanation:** {q['explanation']}")
            st.session_state.quiz_answers.append({"q": idx, "correct": correct, "chosen": chosen_idx})

            if idx + 1 >= total:
                st.session_state.quiz_completed = True
                st.session_state.quiz_active = False
                st.rerun()
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
  <p>{"🎉 Excellent! Exam-ready!" if pct >= 80 else "📚 Keep studying!" if pct >= 60 else "⚠️ Review needed"}</p>
</div>
""", unsafe_allow_html=True)

        # Per-exam breakdown if mixed quiz
        answers_taken = st.session_state.quiz_answers
        q_list = st.session_state.quiz_questions
        if st.session_state.exam_mode == "Both (AI-102 + AI-103)":
            st.markdown("---")
            st.subheader("📊 Score Breakdown by Exam")
            breakdown = {"AI-102": {"correct": 0, "total": 0}, "AI-103": {"correct": 0, "total": 0}}
            for ans in answers_taken:
                q = q_list[ans["q"]]
                ex = q.get("exam", "AI-102")
                if ex in breakdown:
                    breakdown[ex]["total"] += 1
                    if ans["correct"]:
                        breakdown[ex]["correct"] += 1
            c1, c2 = st.columns(2)
            for col, (ex, data) in zip([c1, c2], breakdown.items()):
                t = data["total"]
                c = data["correct"]
                col.metric(
                    f"{ex} Score",
                    f"{c}/{t}" if t else "N/A",
                    f"{c/t*100:.0f}%" if t else "",
                )

        st.markdown("---")
        st.subheader("Question Review")
        for i, ans in enumerate(answers_taken):
            q = q_list[ans["q"]]
            icon = "✅" if ans["correct"] else "❌"
            ex = q.get("exam", "AI-102")
            with st.expander(f"{icon} Q{i+1} [{ex}]: {q['question'][:75]}..."):
                st.markdown(f"**Your answer:** {q['options'][ans['chosen']] if ans['chosen'] is not None else 'Skipped'}")
                st.markdown(f"**Correct answer:** {q['options'][q['answer']]}")
                st.markdown(f"**Explanation:** {q['explanation']}")

        if st.button("🔄 New Quiz", type="primary", use_container_width=True):
            reset_quiz()
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: ASK AI
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.subheader("🤖 Ask the AI Study Tutor")
    st.markdown("Get instant AI-powered explanations for AI-102, AI-103, or comparison questions.")

    if not (st.session_state.api_key or os.getenv("ANTHROPIC_API_KEY")):
        st.warning("Enter your Anthropic API key in the sidebar to use this feature.")

    mode = st.session_state.exam_mode
    quick_prompts = [
        # Shared / comparison
        "What are the key differences between AI-102 and AI-103?",
        "Should I take AI-102 or AI-103 in 2026?",
        "How does RAG work with Azure OpenAI and Azure AI Search?",
        "What are the six Microsoft Responsible AI principles?",
        # AI-102 specific
        "Explain the difference between CLU and Custom Question Answering",
        "How do I secure Azure AI services with managed identity?",
        "What is a skillset in Azure AI Search and how does it work?",
        # AI-103 specific
        "Explain Azure AI Foundry Hubs vs Projects",
        "What are all the built-in tools in Foundry Agent Service?",
        "How does Microsoft Agent Framework combine Semantic Kernel and AutoGen?",
        "Walk me through a RAG pipeline architecture for AI-103",
        "What is the difference between PTU and Standard Azure OpenAI deployments?",
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
        placeholder="e.g. What is the Foundry Agent Service code_interpreter tool and when should I use it?",
    )

    if st.button("Ask AI Tutor", type="primary", disabled=not user_q.strip()):
        with st.spinner("Generating explanation…"):
            sys_msg = (
                "You are an expert Microsoft Azure AI Engineer and Microsoft Certified Trainer "
                "specializing in both the AI-102 (Azure AI Engineer Associate, retiring June 2026) "
                "and the AI-103 (Azure AI Apps and Agents Developer Associate, beta April 2026) exams. "
                "Give concise, exam-focused answers. Use bullet points where helpful. "
                "Bold Azure service and framework names. When relevant, note which exam (AI-102 vs AI-103) "
                "a topic applies to."
            )
            context = f"Student is studying for: {mode}\n\nQuestion: {user_q}"
            response = stream_explanation(context, sys_msg)
            st.session_state.ai_response = response
            st.session_state.ai_prompt = ""

    if st.session_state.ai_response:
        st.markdown("---")
        st.markdown("### AI Tutor Response")
        st.markdown(st.session_state.ai_response)
        if st.button("Clear Response"):
            st.session_state.ai_response = ""
            st.rerun()
