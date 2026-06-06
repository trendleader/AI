"""
AWS Data Engineering Agentic RAG - Streamlit Application
Implements Corrective RAG with LangChain + LangGraph decision loop:
  Query Router → Retriever → Relevance Grader → Generator →
  Hallucination Grader → Answer Grader → (re-route if needed)
"""

import os
import json
import time
import streamlit as st
from typing import TypedDict, List, Literal, Optional

# ── LangChain / LangGraph ──────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# ── Model backends ─────────────────────────────────────────────────────────
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ── Streamlit page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AWS Data Engineering - Agentic RAG",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2.2rem;color:#FF9900;font-weight:700;text-align:center;margin-bottom:0.5rem;}
    .sub-header  {font-size:1rem;color:#888;text-align:center;margin-bottom:2rem;}
    .agent-step  {background:#1e1e2e;border-left:4px solid #FF9900;padding:0.6rem 1rem;
                  border-radius:6px;margin:4px 0;font-size:0.85rem;color:#cdd6f4;}
    .agent-step.done {border-left-color:#a6e3a1;}
    .agent-step.warn {border-left-color:#f38ba8;}
    .answer-box  {background:#1a1a2e;border:1px solid #FF9900;border-radius:10px;
                  padding:1.2rem;margin-top:1rem;}
    .source-chip {display:inline-block;background:#2a2a3e;color:#89b4fa;
                  border-radius:12px;padding:2px 10px;font-size:0.75rem;margin:2px;}
    .metric-card {background:#1e1e2e;border-radius:8px;padding:0.8rem;
                  text-align:center;border:1px solid #313244;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  AWS KNOWLEDGE BASE  (embedded – no external file needed)
# ═══════════════════════════════════════════════════════════════════════════

AWS_KNOWLEDGE = """
# AWS Data Engineering Reference Guide

## Domain 1: Data Ingestion & Transformation

### Streaming Data Ingestion
Real-time data arrives continuously in never-ending streams (sensor data, social media feeds, application logs, stock quotes). Data is processed as it arrives, enabling real-time analytics.

**Amazon Kinesis Data Streams**: Managed service for real-time data streams. Each shard handles 1 MB/sec or 1,000 records/sec writes; 2 MB/sec reads. Retention up to 365 days.
**Amazon Kinesis Data Firehose**: Fully managed delivery service. Loads data to S3, Redshift, OpenSearch, Splunk. Supports compression, encryption, Lambda transformation.
**Amazon Kinesis Data Analytics**: SQL or Apache Flink for real-time stream processing. Auto-scales, serverless.
**Amazon MSK (Managed Streaming for Apache Kafka)**: Highly scalable platform for real-time data feeds. Compatible with Kafka APIs.
**DynamoDB Streams**: Captures all changes to DynamoDB tables in real-time (ordered, 24-hour retention).
**AWS Glue Streaming ETL**: Serverless streaming data integration from Kinesis and Kafka.

### Batch Data Ingestion
Data delivered in large chunks at specific intervals (CSV files, log files, database backups). Processed at predefined times.

**Amazon S3**: Scalable object storage for storing large datasets. 11 nines durability. Storage classes: Standard, Intelligent-Tiering, Standard-IA, One Zone-IA, Glacier Instant Retrieval, Glacier Flexible Retrieval, Glacier Deep Archive.
**AWS Glue**: Serverless ETL service. Glue Data Catalog stores metadata. Glue Crawlers auto-discover schemas. Glue Jobs run PySpark/Python/Scala. Glue DataBrew for visual data preparation.
**Amazon EMR**: Managed Hadoop/Spark framework. Choose instance types: memory-optimized (r-series) for Spark, compute-optimized (c-series) for CPU tasks. Supports Hive, Pig, HBase, Presto, Flink.
**AWS DMS (Database Migration Service)**: Migrates data between databases with minimal downtime. Supports homogeneous and heterogeneous migrations. Continuous replication via CDC.
**Amazon AppFlow**: Fully managed integration service for SaaS applications (Salesforce, SAP, ServiceNow).
**AWS Transfer Family**: SFTP, FTP, FTPS managed file transfer into S3 or EFS.
**AWS Snow Family**: Snowcone (8TB), Snowball Edge (80TB), Snowmobile (100PB) for offline data transfer.

### Data Transformation
**AWS Lambda**: Serverless compute. Max timeout 15 minutes. Up to 10 GB memory. Higher memory = more CPU. Provisioned concurrency reduces cold starts. Good for small transformations.
**Amazon EMR**: Large-scale distributed processing with Hadoop/Spark/Hive.
**AWS Glue**: Managed ETL. DynamicFrames extend Spark DataFrames. Bookmarks track processed data to avoid reprocessing.
**Amazon Redshift**: COPY command loads data efficiently. UNLOAD exports to S3. Spectrum queries S3 directly.

---

## Domain 2: Data Store Management

### Database Selection Guide
**Amazon RDS**: Managed relational DB (MySQL, PostgreSQL, Oracle, SQL Server, MariaDB, Aurora). Multi-AZ for HA. Read Replicas for scaling reads. Automated backups.
**Amazon Aurora**: MySQL/PostgreSQL-compatible. 5x faster than MySQL. 3x faster than PostgreSQL. Serverless v2 auto-scales. Global Database for multi-region.
**Amazon DynamoDB**: NoSQL key-value/document DB. Single-digit millisecond latency at any scale. Partition key distributes data evenly. Global Secondary Indexes (GSI) for alternate query patterns. Local Secondary Indexes (LSI) on same partition key. On-demand or provisioned capacity. DynamoDB Accelerator (DAX) provides in-memory cache.
**Amazon Redshift**: Columnar data warehouse. Massively Parallel Processing (MPP). Distribution keys minimize data movement. Sort keys optimize query performance. VACUUM reclaims space. Concurrency Scaling handles burst. RA3 nodes separate storage/compute.
**Amazon Athena**: Serverless SQL queries on S3. Pay per query ($5/TB scanned). Supports Parquet, ORC (columnar formats reduce cost). Partition pruning speeds queries.
**Amazon OpenSearch Service**: Search and analytics. Real-time log analytics, full-text search.
**Amazon ElastiCache**: In-memory cache. Redis (persistence, pub/sub, Lua scripting). Memcached (simple, multi-threaded).
**Amazon DocumentDB**: MongoDB-compatible managed document database.
**Amazon Keyspaces**: Managed Apache Cassandra.
**Amazon Neptune**: Graph database. Supports Gremlin and SPARQL.
**Amazon Timestream**: Purpose-built time series database for IoT and operational data.
**Amazon QLDB**: Immutable, cryptographically verifiable ledger database.

### S3 Data Lake Architecture
**S3 Lifecycle Policies**: Automate transitions between storage classes. Define ID, Prefix, Transitions, Expiration rules. Optimize costs.
**S3 Intelligent-Tiering**: Automatically moves objects between access tiers. No retrieval fees. Monthly monitoring fee per object.
**AWS Lake Formation**: Centralized data lake governance. Fine-grained column-level security. Data Catalog integration. Tag-based access control (LF-Tags).
**S3 Object Lock**: WORM (Write Once Read Many). Compliance mode (no one can delete) vs Governance mode (admins can delete).
**S3 Replication**: CRR (Cross-Region Replication) for disaster recovery and compliance. SRR (Same-Region Replication) for log aggregation.

---

## Domain 3: Data Operations & Support

### Workflow Orchestration
**Amazon MWAA (Managed Workflows for Apache Airflow)**: Complex DAG-based orchestration. Visual UI. Python-based DAGs. Good for dependencies between tasks.
**AWS Step Functions**: Visual serverless workflow coordination. Standard Workflows (long-running, durable) vs Express Workflows (high-volume, short-duration). 200+ AWS service integrations.
**AWS Glue Workflows**: Visual pipeline for Glue Crawlers, Jobs, Triggers. Native integration with Glue components.
**Amazon EventBridge**: Event-driven architecture. Schedule rules (cron). Event patterns match and route events. 90+ AWS service integrations.
**AWS Lambda**: Event-driven triggers from S3, DynamoDB Streams, Kinesis, SQS, SNS.

### Data Quality & Monitoring
**AWS Glue Data Quality**: Define rules using DQDL (Data Quality Definition Language). Evaluate data quality in Glue jobs. Rules: Completeness, Uniqueness, Referential Integrity, Accuracy.
**Amazon CloudWatch**: Metrics, logs, alarms, dashboards. Log Insights for querying logs. Contributor Insights for identifying high-impact contributors. Evidently for A/B testing.
**AWS CloudTrail**: API audit trail. All AWS API calls logged. S3 data events. Integration with EventBridge.
**Amazon DataZone**: Data governance and cataloging across organization. Business data catalog. Data subscriptions.
**AWS Config**: Continuous compliance monitoring. Configuration history. Compliance rules. Remediation actions.

---

## Domain 4: Data Security & Governance

### Encryption & Access Control
**AWS KMS (Key Management Service)**: Managed encryption keys. CMKs (Customer Managed Keys) vs AWS Managed Keys. Key rotation. Envelope encryption. Integrates with most AWS services.
**AWS IAM**: Identity and Access Management. Policies (Identity-based, Resource-based, Permission boundaries). Least privilege principle. Service Control Policies (SCPs) in Organizations.
**AWS Secrets Manager**: Store, rotate, manage secrets (DB credentials, API keys). Automatic rotation via Lambda. Cross-account access.
**AWS Macie**: ML-powered sensitive data discovery in S3. PII detection. Findings to Security Hub.
**Amazon VPC**: Network isolation. Private subnets for databases. VPC Endpoints (Interface and Gateway) for private AWS service access. PrivateLink for service sharing.

### Data Governance
**AWS Glue Data Catalog**: Central metadata repository. Tables, databases, partitions, schemas. Used by Athena, EMR, Redshift Spectrum.
**AWS Lake Formation**: Data lake governance. Row and column-level security. Data sharing across accounts. Governed Tables with ACID transactions.
**Amazon DataZone**: Business-friendly data portal. Data discovery, governance, collaboration across teams.

---

## Domain 5: Data Analysis & Visualization

### Analytics Services
**Amazon Redshift**: Complex SQL analytics on structured data. Materialized views. Federated queries to RDS/Aurora. ML integration (CREATE MODEL).
**Amazon Athena**: Ad-hoc SQL on S3. Federated queries via connectors. Prepared statements. CTAS (Create Table As Select). Workgroups for cost control.
**Amazon EMR**: Big data processing. Spark MLlib for ML. Hive for SQL. Presto for interactive queries. Jupyter notebooks via EMR Studio.
**Amazon QuickSight**: BI and visualization. SPICE (in-memory engine). ML Insights (anomaly detection, forecasting). Embedded analytics. Q (NLP queries).
**Amazon SageMaker**: Complete ML platform. Data Wrangler (visual feature engineering). Feature Store. Training jobs (built-in algorithms + custom containers). Hyperparameter Tuning. Model Registry. Endpoints (real-time + batch). Model Monitor (drift detection). Clarify (bias/explainability). Pipelines (MLOps).

### Key Architectural Patterns
**Lambda Architecture**: Batch layer (accuracy) + Speed layer (real-time) + Serving layer (merged view).
**Kappa Architecture**: Single streaming pipeline replaces both layers. Reprocess by replaying from Kafka/Kinesis.
**Data Lakehouse**: Combines data lake flexibility with data warehouse performance. AWS Lake Formation + Redshift Spectrum + Athena.
**Medallion Architecture**: Bronze (raw) → Silver (cleaned/validated) → Gold (business-ready aggregates).
**Event-Driven Architecture**: Events trigger processing. Kinesis → Lambda → DynamoDB. Decoupled, scalable.

---

## Performance Optimization Cheat Sheet

**Redshift**: Distribution keys (EVEN, KEY, ALL) minimize data movement. Sort keys (COMPOUND, INTERLEAVED) speed range queries. VACUUM removes deleted rows. ANALYZE updates statistics. WLM (Workload Management) queues prioritize queries. Concurrency Scaling auto-adds capacity.
**DynamoDB**: Avoid hot partitions with high-cardinality partition keys. Use exponential backoff for retries. Batch operations reduce API calls. DynamoDB Streams + Lambda for triggers. TTL removes expired items automatically.
**S3**: Use Parquet/ORC columnar formats for analytics (70-85% less data scanned in Athena). Partition data by date/region for partition pruning. S3 Select filters data server-side. Multipart upload for files >100MB. Transfer Acceleration for global uploads.
**Kinesis**: Add shards to increase throughput. Batch records to maximize shard utilization. Enhanced fan-out for low-latency consumers (2MB/sec per consumer per shard). Use partition keys that distribute load evenly.
**Lambda**: Optimize memory (more memory = more CPU). Keep functions warm with provisioned concurrency. Use /tmp (512MB-10GB) for temporary storage. Minimize cold starts with smaller deployment packages.
**EMR**: Right-size instance types. Use Spot Instances for non-critical workloads (60-90% savings). Enable dynamic resource allocation in Spark. Use S3 as persistent storage instead of HDFS. EMR Serverless eliminates cluster management.

---

## Common Exam Scenarios

**Q: Real-time fraud detection at scale?**
A: Kinesis Data Streams → Kinesis Data Analytics (Flink) → Lambda → DynamoDB → SNS alerts. SageMaker endpoint for ML scoring.

**Q: Cost-effective data lake for ad-hoc queries?**
A: S3 (Parquet, partitioned) + Athena + Glue Data Catalog. Use Glue Crawlers to auto-discover schema.

**Q: ETL pipeline with dependencies?**
A: AWS Glue Workflows or Step Functions. Glue for transformation, MWAA for complex multi-system orchestration.

**Q: Migrate on-premises Oracle to cloud?**
A: AWS DMS (heterogeneous migration) + Schema Conversion Tool (SCT). Target: Aurora PostgreSQL or RDS.

**Q: Secure multi-tenant data lake?**
A: Lake Formation with tag-based access control (LF-Tags). Column and row-level security. Service Control Policies.

**Q: Global, low-latency reads for application?**
A: DynamoDB Global Tables (multi-region active-active). ElastiCache Redis in front for sub-millisecond reads.

**Q: Process 10TB daily batch job efficiently?**
A: EMR with Spark (Spot Instances). Write output to S3 in Parquet. Glue Crawler updates catalog. Athena for queries.

**Q: Reduce Redshift query time?**
A: Check distribution keys (avoid ALL for large tables). Add sort keys matching WHERE/JOIN columns. Update table statistics with ANALYZE. Use materialized views. Enable result caching.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH STATE
# ═══════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    steps: List[str]
    route: str
    relevance: str
    hallucination: str
    answer_grade: str
    iterations: int


# ═══════════════════════════════════════════════════════════════════════════
#  VECTOR STORE  (built once per session, cached in st.session_state)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Building vector store from AWS knowledge base…")
def build_vectorstore(backend: str, openai_key: str = "") -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    docs = splitter.create_documents(
        [AWS_KNOWLEDGE],
        metadatas=[{"source": "AWS Data Engineering Reference Guide"}],
    )

    if backend == "OpenAI" and openai_key:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

    return Chroma.from_documents(docs, embeddings, collection_name="aws_de_rag")


# ═══════════════════════════════════════════════════════════════════════════
#  LLM FACTORY
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading language model…")
def get_llm(backend: str, model_name: str, openai_key: str = "", temperature: float = 0.0):
    if backend == "OpenAI" and openai_key:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=openai_key,
        )
    # Llama / HuggingFace fallback
    if HF_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=max(temperature, 0.01),
        )
        return HuggingFacePipeline(pipeline=pipe)
    raise ValueError("No valid backend available.")


# ═══════════════════════════════════════════════════════════════════════════
#  AGENTIC RAG  – LangGraph nodes
# ═══════════════════════════════════════════════════════════════════════════

def build_agentic_graph(vectorstore: Chroma, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ── Prompt templates ───────────────────────────────────────────────────

    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert AWS data engineering assistant. "
         "Classify the user's question.\n"
         "Reply with exactly one word:\n"
         "  'vectorstore' – if the question is about AWS services, data engineering, architecture, "
         "concepts, best practices, certifications, or anything in the knowledge base.\n"
         "  'direct' – if it is a simple greeting, off-topic, or something the LLM can answer "
         "without any AWS context.\n"
         "No other words."),
        ("human", "Question: {question}"),
    ])

    grader_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are grading whether a retrieved document is relevant to the question.\n"
         "Answer 'yes' if the document contains information useful to answer the question.\n"
         "Answer 'no' otherwise. One word only."),
        ("human", "Question: {question}\n\nDocument:\n{document}"),
    ])

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert AWS Data Engineering assistant. "
         "Answer the question using ONLY the provided context. "
         "Be thorough, structured, and include relevant AWS service names. "
         "If the context does not contain enough information, say so honestly."),
        ("human",
         "Context:\n{context}\n\n"
         "Question: {question}\n\n"
         "Provide a detailed, well-structured answer:"),
    ])

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are checking whether an AI answer is grounded in the provided documents.\n"
         "Answer 'yes' if the answer is fully supported by the context.\n"
         "Answer 'no' if the answer contains unsupported claims. One word only."),
        ("human", "Documents:\n{documents}\n\nAnswer:\n{generation}"),
    ])

    answer_grade_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Does this answer fully address the question?\n"
         "Answer 'yes' if it resolves the question. 'no' otherwise. One word only."),
        ("human", "Question: {question}\n\nAnswer: {generation}"),
    ])

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are rewriting a question to improve retrieval from a vector database "
         "about AWS data engineering. Make it more specific and keyword-rich."),
        ("human", "Original question: {question}\n\nRewritten question:"),
    ])

    direct_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful AWS Data Engineering expert. "
         "Answer concisely and accurately."),
        ("human", "{question}"),
    ])

    parser = StrOutputParser()

    # ── Node functions ─────────────────────────────────────────────────────

    def route_question(state: GraphState) -> GraphState:
        chain = router_prompt | llm | parser
        result = chain.invoke({"question": state["question"]}).strip().lower()
        route = "vectorstore" if "vectorstore" in result else "direct"
        return {**state, "route": route,
                "steps": state["steps"] + [f"🔀 Router → **{route}**"]}

    def retrieve(state: GraphState) -> GraphState:
        docs = retriever.invoke(state["question"])
        return {**state, "documents": docs,
                "steps": state["steps"] + [f"🔍 Retrieved **{len(docs)}** document chunks"]}

    def grade_documents(state: GraphState) -> GraphState:
        chain = grader_prompt | llm | parser
        relevant = []
        for doc in state["documents"]:
            score = chain.invoke({
                "question": state["question"],
                "document": doc.page_content,
            }).strip().lower()
            if "yes" in score:
                relevant.append(doc)
        relevance = "relevant" if relevant else "not_relevant"
        return {**state,
                "documents": relevant,
                "relevance": relevance,
                "steps": state["steps"] + [
                    f"✅ **{len(relevant)}** relevant docs" if relevant
                    else "⚠️ No relevant docs – will rewrite query"
                ]}

    def rewrite_query(state: GraphState) -> GraphState:
        chain = rewrite_prompt | llm | parser
        new_q = chain.invoke({"question": state["question"]}).strip()
        return {**state,
                "question": new_q,
                "iterations": state["iterations"] + 1,
                "steps": state["steps"] + [f"✏️ Rewritten query: *{new_q}*"]}

    def generate(state: GraphState) -> GraphState:
        context = "\n\n---\n\n".join(d.page_content for d in state["documents"])
        chain = rag_prompt | llm | parser
        answer = chain.invoke({"context": context, "question": state["question"]})
        return {**state, "generation": answer,
                "steps": state["steps"] + ["💬 Generated RAG answer"]}

    def direct_answer(state: GraphState) -> GraphState:
        chain = direct_prompt | llm | parser
        answer = chain.invoke({"question": state["question"]})
        return {**state, "generation": answer,
                "steps": state["steps"] + ["💬 Direct LLM answer (no retrieval needed)"]}

    def grade_hallucination(state: GraphState) -> GraphState:
        docs_text = "\n\n".join(d.page_content for d in state["documents"])
        chain = hallucination_prompt | llm | parser
        result = chain.invoke({
            "documents": docs_text,
            "generation": state["generation"],
        }).strip().lower()
        grade = "grounded" if "yes" in result else "hallucination"
        return {**state, "hallucination": grade,
                "steps": state["steps"] + [
                    "✅ Answer is grounded in sources" if grade == "grounded"
                    else "⚠️ Possible hallucination – regenerating"
                ]}

    def grade_answer(state: GraphState) -> GraphState:
        chain = answer_grade_prompt | llm | parser
        result = chain.invoke({
            "question": state["question"],
            "generation": state["generation"],
        }).strip().lower()
        grade = "useful" if "yes" in result else "not_useful"
        return {**state, "answer_grade": grade,
                "steps": state["steps"] + [
                    "✅ Answer fully addresses the question" if grade == "useful"
                    else "⚠️ Answer incomplete – retrying"
                ]}

    # ── Edge conditions ────────────────────────────────────────────────────

    def decide_after_route(state: GraphState) -> Literal["retrieve", "direct_answer"]:
        return "retrieve" if state["route"] == "vectorstore" else "direct_answer"

    def decide_after_grading(state: GraphState) -> Literal["generate", "rewrite_query"]:
        if state["relevance"] == "relevant":
            return "generate"
        if state["iterations"] >= 2:
            return "generate"  # give up rewriting after 2 attempts
        return "rewrite_query"

    def decide_after_hallucination(state: GraphState) -> Literal["grade_answer", "generate"]:
        if state["hallucination"] == "grounded":
            return "grade_answer"
        if state["iterations"] >= 2:
            return "grade_answer"
        return "generate"

    def decide_after_answer_grade(state: GraphState) -> Literal[END, "rewrite_query"]:
        if state["answer_grade"] == "useful":
            return END
        if state["iterations"] >= 2:
            return END
        return "rewrite_query"

    # ── Build graph ────────────────────────────────────────────────────────
    graph = StateGraph(GraphState)

    graph.add_node("route_question",     route_question)
    graph.add_node("retrieve",           retrieve)
    graph.add_node("grade_documents",    grade_documents)
    graph.add_node("rewrite_query",      rewrite_query)
    graph.add_node("generate",           generate)
    graph.add_node("direct_answer",      direct_answer)
    graph.add_node("grade_hallucination", grade_hallucination)
    graph.add_node("grade_answer",       grade_answer)

    graph.set_entry_point("route_question")

    graph.add_conditional_edges("route_question", decide_after_route, {
        "retrieve":     "retrieve",
        "direct_answer": "direct_answer",
    })
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", decide_after_grading, {
        "generate":     "generate",
        "rewrite_query": "rewrite_query",
    })
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "grade_hallucination")
    graph.add_conditional_edges("grade_hallucination", decide_after_hallucination, {
        "grade_answer": "grade_answer",
        "generate":     "generate",
    })
    graph.add_conditional_edges("grade_answer", decide_after_answer_grade, {
        END:             END,
        "rewrite_query": "rewrite_query",
    })
    graph.add_edge("direct_answer", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown('<div class="main-header">☁️ AWS Data Engineering – Agentic RAG</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">LangChain · LangGraph · Corrective RAG · ChromaDB</div>',
                unsafe_allow_html=True)

    # ── Sidebar config ──────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg",
                 width=120)
        st.markdown("## ⚙️ Configuration")

        backend = st.selectbox("Model Backend", ["OpenAI", "Llama (HuggingFace)"])

        openai_key = ""
        model_name = ""

        if backend == "OpenAI":
            openai_key = st.text_input("OpenAI API Key", type="password",
                                       value=os.getenv("OPENAI_API_KEY", ""))
            model_name = st.selectbox("Model", [
                "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"
            ])
        else:
            model_name = st.selectbox("Llama Model", [
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ])
            st.info("Llama models download on first use (~2-7 GB). Ensure HF_TOKEN env var is set.")

        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)

        st.markdown("---")
        st.markdown("### 🔬 Agentic RAG Pipeline")
        st.markdown("""
**Decision nodes:**
1. 🔀 **Query Router** – vectorstore vs direct LLM
2. 🔍 **Retriever** – ChromaDB semantic search
3. ✅ **Relevance Grader** – filter irrelevant docs
4. 💬 **Generator** – RAG answer from context
5. 🛡️ **Hallucination Grader** – grounded in sources?
6. ⭐ **Answer Grader** – fully answers question?
7. ✏️ **Query Rewriter** – improve & retry (max 2x)
        """)

        st.markdown("---")
        st.markdown("### 💡 Sample Questions")
        sample_qs = [
            "What is the difference between Kinesis Data Streams and Kinesis Firehose?",
            "How do I optimize Redshift query performance?",
            "When should I use DynamoDB vs RDS vs Redshift?",
            "Explain the Medallion architecture pattern",
            "How does AWS Glue differ from EMR for ETL?",
            "What are S3 storage classes and when to use each?",
            "Design a real-time fraud detection pipeline on AWS",
            "How does Lake Formation handle data governance?",
        ]
        for q in sample_qs:
            if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
                st.session_state["prefill"] = q

        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    # ── Session state init ──────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "prefill" not in st.session_state:
        st.session_state["prefill"] = ""

    # ── Build/load vectorstore & LLM ────────────────────────────────────────
    ready = False
    if backend == "OpenAI" and not openai_key:
        st.warning("⚠️ Enter your OpenAI API key in the sidebar to get started.")
    else:
        try:
            with st.spinner("Initialising vector store & LLM…"):
                vectorstore = build_vectorstore(backend, openai_key)
                llm = get_llm(backend, model_name, openai_key, temperature)
                graph = build_agentic_graph(vectorstore, llm)
            ready = True
        except Exception as e:
            st.error(f"Initialisation error: {e}")

    # ── Metrics row ─────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><b>{len(st.session_state["chat_history"])}</b><br>Questions Asked</div>',
                    unsafe_allow_html=True)
    with col2:
        kb_chunks = len(AWS_KNOWLEDGE.split("\n## ")) * 4
        st.markdown(f'<div class="metric-card"><b>{kb_chunks}+</b><br>Knowledge Chunks</div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><b>7</b><br>Agent Decision Nodes</div>',
                    unsafe_allow_html=True)
    with col4:
        status = "✅ Ready" if ready else "⏳ Waiting"
        st.markdown(f'<div class="metric-card"><b>{status}</b><br>System Status</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Chat interface ──────────────────────────────────────────────────────
    # Render history
    for entry in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant", avatar="☁️"):
            st.markdown(f'<div class="answer-box">{entry["answer"]}</div>',
                        unsafe_allow_html=True)
            if entry.get("steps"):
                with st.expander("🔍 Agent decision trace", expanded=False):
                    for step in entry["steps"]:
                        css = "done" if ("✅" in step or "💬" in step) else ("warn" if "⚠️" in step else "")
                        st.markdown(f'<div class="agent-step {css}">{step}</div>',
                                    unsafe_allow_html=True)
            if entry.get("sources"):
                st.markdown("**Sources:** " + " ".join(
                    f'<span class="source-chip">{s}</span>'
                    for s in entry["sources"]
                ), unsafe_allow_html=True)

    # Input box
    prefill = st.session_state.pop("prefill", "")
    question = st.chat_input("Ask anything about AWS Data Engineering…") or prefill

    if question and ready:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant", avatar="☁️"):
            agent_placeholder = st.empty()
            answer_placeholder = st.empty()

            # Stream agent steps live
            steps_so_far: List[str] = []

            def render_steps(steps):
                html = "".join(
                    f'<div class="agent-step">{s}</div>' for s in steps
                )
                agent_placeholder.markdown(
                    f"**🤖 Agent reasoning…**\n{html}", unsafe_allow_html=True
                )

            initial_state = GraphState(
                question=question,
                generation="",
                documents=[],
                steps=[],
                route="",
                relevance="",
                hallucination="",
                answer_grade="",
                iterations=0,
            )

            final_state = None
            try:
                for chunk in graph.stream(initial_state):
                    for node_name, node_state in chunk.items():
                        steps_so_far = node_state.get("steps", steps_so_far)
                        render_steps(steps_so_far)
                        time.sleep(0.05)
                    final_state = node_state

            except Exception as e:
                st.error(f"Agent error: {e}")
                final_state = {"generation": f"Error: {e}", "steps": [], "documents": []}

            agent_placeholder.empty()

            answer = final_state.get("generation", "No answer generated.")
            sources = list({
                d.metadata.get("source", "AWS Reference")
                for d in final_state.get("documents", [])
            })

            answer_placeholder.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("🔍 Agent decision trace", expanded=True):
                for step in steps_so_far:
                    css = "done" if ("✅" in step or "💬" in step) else ("warn" if "⚠️" in step else "")
                    st.markdown(f'<div class="agent-step {css}">{step}</div>',
                                unsafe_allow_html=True)

            if sources:
                st.markdown("**Sources:** " + " ".join(
                    f'<span class="source-chip">{s}</span>' for s in sources
                ), unsafe_allow_html=True)

            st.session_state["chat_history"].append({
                "question": question,
                "answer": answer,
                "steps": steps_so_far,
                "sources": sources,
            })

    elif question and not ready:
        st.warning("Please configure your API key in the sidebar first.")

    # ── Knowledge base explorer ─────────────────────────────────────────────
    with st.expander("📚 Browse AWS Knowledge Base", expanded=False):
        sections = [s.strip() for s in AWS_KNOWLEDGE.split("\n## ") if s.strip()]
        tabs = st.tabs([s.split("\n")[0][:35] for s in sections])
        for tab, section in zip(tabs, sections):
            with tab:
                st.markdown(section)


if __name__ == "__main__":
    main()
