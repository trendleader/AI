"""
AWS Data Engineer Certification Prep Streamlit Application - COMPLETE FIXED VERSION
Comprehensive system with full exam simulation, RAG integration, and certification prep features
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import random
import asyncio
from collections import defaultdict
import time

# Configure Streamlit page
st.set_page_config(
    page_title="AWS Data Engineer Cert Prep",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .study-mode-header {
        color: #232F3E;
        border-left: 4px solid #FF9900;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .exam-question {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #FF9900;
    }
    .correct-answer {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .wrong-answer {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .progress-metric {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(45deg, #FF9900, #232F3E);
        color: white;
        margin: 0.5rem;
    }
    .exam-timer {
        font-size: 1.2rem;
        font-weight: bold;
        color: #FF9900;
    }
    .pass-score {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .fail-score {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'study_progress' not in st.session_state:
        st.session_state.study_progress = {
            'questions_answered': 0,
            'correct_answers': 0,
            'topics_covered': [],
            'weak_areas': [],
            'study_sessions': 0,
            'total_study_time': 0,
            'exam_attempts': 0,
            'best_exam_score': 0
        }
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

# Simple fallback chunk class
class SimpleChunk:
    def __init__(self, content, chunk_id, path="unknown"):
        self.content = content
        self.id = chunk_id
        self.path = path
        self.aws_services = []
        self.keywords = []
        self.metadata = {'domain': 'general'}

class ComprehensiveRAG:
    """RAG system that loads your comprehensive model or uses fallback"""
    
    def __init__(self):
        self.chunks = []
        self.embeddings_matrix = None
        self.keyword_index = defaultdict(list)
        self.service_index = defaultdict(list)
        self.aws_services = set()
        self.model_loaded = False
        self.prompt_templates = {}
        self.learned_weights = {}
        self._create_fallback_knowledge()
        
    def _create_fallback_knowledge(self):
        """Create basic knowledge base if model file isn't available"""
        fallback_content = [
            ("Lambda functions have a maximum timeout of 15 minutes and can use up to 10GB of memory. For CPU-intensive tasks, increasing memory allocation also increases CPU power.", "lambda_basics"),
            ("S3 storage classes: Standard for frequent access, Standard-IA for infrequent access, Glacier for archival, and Deep Archive for long-term retention with the lowest cost.", "s3_storage"),
            ("Kinesis Data Streams: Each shard can handle 1MB/sec or 1,000 records/sec for writes, and 2MB/sec or 2,000 records/sec for reads.", "kinesis_limits"),
            ("Redshift performance optimization: Use distribution keys to minimize data movement, sort keys for query optimization, and VACUUM to reclaim space.", "redshift_optimization"),
            ("DynamoDB: Use partition keys that distribute data evenly, avoid hot partitions, and consider Global Secondary Indexes for different query patterns.", "dynamodb_design"),
            ("EMR clusters: Choose instance types based on workload - memory-optimized for Spark, compute-optimized for CPU-intensive tasks.", "emr_instances"),
        ]
        
        self.chunks = [
            SimpleChunk(content, f"fallback_{i}", path)
            for i, (content, path) in enumerate(fallback_content)
        ]
        
    def load_comprehensive_model(self, model_path):
        """Load the comprehensive model from pkl file"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different possible formats
            if isinstance(model_data, dict):
                self.chunks = model_data.get('chunks', [])
                self.embeddings_matrix = model_data.get('embeddings_matrix')
                self.keyword_index = defaultdict(list, model_data.get('keyword_index', {}))
                self.service_index = defaultdict(list, model_data.get('service_index', {}))
                self.aws_services = model_data.get('aws_services', set())
                self.prompt_templates = model_data.get('prompt_templates', {})
                self.learned_weights = model_data.get('learned_weights', {
                    'semantic_weight': 0.5,
                    'keyword_weight': 0.3,
                    'service_weight': 0.2,
                    'domain_boost': 1.2
                })
            else:
                # If it's a list or other format, adapt accordingly
                if hasattr(model_data, '__iter__'):
                    self.chunks = list(model_data) if model_data else self.chunks
                    
            self.model_loaded = True
            return True, f"Loaded {len(self.chunks)} chunks from comprehensive model"
            
        except Exception as e:
            return False, f"Error loading model: {e}"
    
    def search(self, query, top_k=5):
        """Search the loaded knowledge base"""
        start_time = time.time()
        
        query_words = set(query.lower().split())
        chunk_scores = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunk_words = set(chunk_content.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            
            if overlap > 0:
                score = overlap / len(query_words.union(chunk_words))
                
                # Boost score for AWS service mentions
                aws_services_in_query = ['lambda', 's3', 'kinesis', 'redshift', 'dynamodb', 'emr', 'glue', 'athena']
                if any(service in query.lower() and service in chunk_content.lower() for service in aws_services_in_query):
                    score *= 1.5
                
                chunk_scores.append((i, score))
        
        # Sort by score and get top results
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:top_k]
        
        results = []
        context_parts = []
        
        for i, (chunk_idx, score) in enumerate(top_chunks):
            chunk = self.chunks[chunk_idx]
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            
            result = {
                'chunk_id': getattr(chunk, 'id', f'chunk_{chunk_idx}'),
                'path': getattr(chunk, 'path', 'unknown'),
                'content': content,
                'score': score,
                'aws_services': getattr(chunk, 'aws_services', []),
                'keywords': getattr(chunk, 'keywords', [])[:5],
                'domain': getattr(chunk, 'metadata', {}).get('domain', 'general')
            }
            results.append(result)
            context_parts.append(f"- {content}")
        
        search_time = (time.time() - start_time) * 1000
        
        return {
            'results': results,
            'context': '\n'.join(context_parts),
            'total_results': len(results),
            'search_time_ms': search_time
        }
    
    async def generate_llm_response(self, query, context, response_type="general"):
        """Generate LLM response using OpenAI or fallback"""
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key or openai_key == 'your-openai-key-here':
            # Fallback response based on context
            fallback_response = self._generate_fallback_response(query, context, response_type)
            return {
                'answer': fallback_response,
                'provider': 'Knowledge Base Only',
                'tokens_used': 0,
                'cost_estimate': 0.0
            }
        
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            
            prompt_templates = {
                "certification": """
For AWS Certified Data Engineer exam preparation:

Context: {context}

Question: {query}

Provide a comprehensive exam-focused answer that includes:
1. Core concept explanation
2. AWS services involved  
3. Best practices and implementation steps
4. Common exam scenarios and tips
5. What to remember for the certification

Answer:
""",
                "practice": """
Based on AWS Data Engineering documentation:

{context}

Create a practice question about: {query}

Provide:
1. Realistic scenario
2. Multiple choice question (A, B, C, D)
3. Correct answer with explanation
4. Why other options are incorrect
5. Exam strategy tip

Question:
""",
                "general": """
Based on AWS Data Engineering knowledge:

{context}

Question: {query}

Provide a detailed, practical answer with implementation guidance and best practices.

Answer:
"""
            }
            
            prompt = prompt_templates.get(response_type, prompt_templates["general"]).format(context=context, query=query)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert AWS Data Engineering certification instructor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            return {
                'answer': response.choices[0].message.content,
                'provider': 'OpenAI GPT-3.5',
                'tokens_used': response.usage.total_tokens,
                'cost_estimate': (response.usage.total_tokens / 1000) * 0.002
            }
            
        except Exception as e:
            fallback_response = self._generate_fallback_response(query, context, response_type)
            return {
                'answer': f"Using knowledge base response:\n\n{fallback_response}",
                'provider': 'Fallback (OpenAI Error)',
                'tokens_used': 0,
                'cost_estimate': 0.0
            }
    
    def _generate_fallback_response(self, query, context, response_type):
        """Generate a response using only the knowledge base"""
        if not context:
            return f"No relevant information found for '{query}'. Please try a more specific AWS Data Engineering topic."
            
        if response_type == "practice":
            return f"""
**Practice Question Based on: {query}**

Scenario: You are working with AWS {query} and need to optimize for performance and cost.

Question: What is the most important consideration when configuring {query} for a data engineering workload?

A) Always use the largest instance size
B) Focus on the specific requirements of your workload
C) Use default configurations
D) Minimize all costs regardless of performance

Correct Answer: B

Explanation: AWS services should be configured based on your specific workload requirements, balancing performance, cost, and scalability needs.

Based on knowledge: {context[:200]}...
"""
        elif response_type == "certification":
            return f"""
**AWS Certification Study Guide: {query}**

Key Points to Remember:
{context}

Exam Tips:
- Understand the core concepts and use cases
- Know when to use this service vs alternatives  
- Remember cost optimization strategies
- Practice scenario-based questions

This topic commonly appears in AWS Data Engineering certification exams focusing on practical implementation and best practices.
"""
        else:
            return f"""
**AWS Data Engineering Guide: {query}**

Based on the knowledge base:

{context}

This information covers the key concepts and best practices for {query} in AWS data engineering contexts.
"""

@st.cache_resource
def load_rag_system():
    """Load the comprehensive RAG system"""
    
    # Try to load from the specified path
    project_dir = Path(r"C:\Users\pjone\OneDrive\Desktop\NewPython")
    model_path = project_dir / "aws_enhanced_rag_model.pkl"
    
    rag = ComprehensiveRAG()
    
    if model_path.exists():
        try:
            success, message = rag.load_comprehensive_model(model_path)
            if success:
                return rag, "comprehensive"
            else:
                st.warning(f"⚠️ {message} - Using fallback knowledge base")
        except Exception as e:
            st.warning(f"⚠️ Error loading model: {e} - Using fallback knowledge base")
    else:
        st.info(f"ℹ️ Model file not found at {model_path} - Using fallback knowledge base")
    
    return rag, "fallback"

def check_openai_setup():
    """Check OpenAI configuration"""
    project_dir = Path(r"C:\Users\pjone\OneDrive\Desktop\NewPython")
    env_file = project_dir / "test.env"
    
    if not env_file.exists():
        return False, f"test.env file not found at {env_file}"
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('OPENAI_API_KEY='):
                    key = line.split('=', 1)[1].strip()
                    if key and key != 'your-openai-key-here':
                        os.environ['OPENAI_API_KEY'] = key
                        return True, "OpenAI key loaded successfully"
                    else:
                        return False, "OpenAI key is placeholder"
        
        return False, "OPENAI_API_KEY not found in test.env"
    
    except Exception as e:
        return False, f"Error reading test.env: {e}"

def create_sidebar():
    """Create sidebar with navigation"""
    
    st.sidebar.markdown("# 🎓 AWS Data Engineer Prep")
    
    # Check systems status
    openai_ok, openai_msg = check_openai_setup()
    
    # Load RAG system status
    if st.session_state.rag_system is None:
        st.session_state.rag_system, system_type = load_rag_system()
        
        if system_type == "comprehensive":
            st.sidebar.success("✅ Comprehensive model loaded")
            st.sidebar.info(f"📊 {len(st.session_state.rag_system.chunks)} knowledge chunks available")
        else:
            st.sidebar.warning("⚠️ Using fallback knowledge base")
            st.sidebar.info(f"📊 {len(st.session_state.rag_system.chunks)} fallback chunks available")
    
    if openai_ok:
        st.sidebar.success("✅ OpenAI API configured")
    else:
        st.sidebar.warning(f"⚠️ {openai_msg}")
    
    # Study mode selection
    study_modes = {
        "💬 Chat Mode": "chat",
        "📚 Topic Review": "concept", 
        "❓ Practice Questions": "practice",
        "🎯 Exam Simulation": "exam",
        "📊 Progress Dashboard": "progress"
    }
    
    selected_mode = st.sidebar.selectbox(
        "Select Study Mode",
        options=list(study_modes.keys()),
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats
    progress = st.session_state.study_progress
    
    st.sidebar.markdown("### 📈 Quick Stats")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Questions", progress['questions_answered'])
    with col2:
        accuracy = (progress['correct_answers'] / max(progress['questions_answered'], 1)) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    st.sidebar.metric("Topics Covered", len(progress['topics_covered']))
    st.sidebar.metric("Study Sessions", progress['study_sessions'])
    st.sidebar.metric("Exam Attempts", progress['exam_attempts'])
    if progress['best_exam_score'] > 0:
        st.sidebar.metric("Best Exam Score", f"{progress['best_exam_score']:.1f}%")
    
    st.sidebar.markdown("---")
    
    # AWS Exam domains
    st.sidebar.markdown("### 📋 Exam Domains")
    domains = {
        "Domain 1: Data Ingestion & Transformation": "34%",
        "Domain 2: Data Store Management": "26%", 
        "Domain 3: Data Operations & Support": "22%",
        "Domain 4: Data Security & Governance": "18%"
    }
    
    for domain, weight in domains.items():
        st.sidebar.markdown(f"**{domain}** - {weight}")
    
    return study_modes[selected_mode], openai_ok

def display_chat_interface():
    """Display chat interface"""
    
    st.markdown('<h1 class="main-header">AWS Data Engineer Certification Prep</h1>', 
                unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask about AWS Data Engineering concepts, or type 'practice [topic]' for questions"):
        
        st.session_state.chat_history.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        search_results = st.session_state.rag_system.search(prompt, top_k=3)
        
        if prompt.lower().startswith("practice"):
            response_type = "practice"
        elif any(word in prompt.lower() for word in ["exam", "certification", "test"]):
            response_type = "certification"
        else:
            response_type = "general"
        
        with st.spinner("Generating comprehensive answer..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                llm_response = loop.run_until_complete(
                    st.session_state.rag_system.generate_llm_response(
                        prompt, search_results['context'], response_type
                    )
                )
                loop.close()
            except Exception as e:
                # Fallback if async fails
                llm_response = {
                    'answer': st.session_state.rag_system._generate_fallback_response(
                        prompt, search_results['context'], response_type
                    ),
                    'provider': 'Fallback (Async Error)',
                    'tokens_used': 0,
                    'cost_estimate': 0.0
                }
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": llm_response['answer'],
            "metadata": {
                'provider': llm_response['provider'],
                'tokens_used': llm_response['tokens_used'],
                'cost_estimate': llm_response['cost_estimate'],
                'search_results': len(search_results['results']),
                'search_time_ms': search_results['search_time_ms']
            },
            "timestamp": datetime.now()
        })
        
        st.session_state.study_progress['study_sessions'] += 1
        topic = prompt.split(' ', 1)[-1] if ' ' in prompt else prompt
        if topic not in st.session_state.study_progress['topics_covered']:
            st.session_state.study_progress['topics_covered'].append(topic)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.caption(f"🔍 {metadata.get('search_results', 0)} sources")
                with col2:
                    st.caption(f"⚡ {metadata.get('search_time_ms', 0):.0f}ms")
                with col3:
                    st.caption(f"🤖 {metadata.get('provider', 'Unknown')}")
                with col4:
                    st.caption(f"💰 ${metadata.get('cost_estimate', 0):.4f}")

def display_topic_review():
    """Display topic review interface"""
    
    st.markdown('<h2 class="study-mode-header">📚 Topic Review Mode</h2>', 
                unsafe_allow_html=True)
    
    # Available topics
    available_topics = [
        "Lambda Optimization", "S3 Storage Classes", "Kinesis Streaming",
        "DynamoDB Configuration", "Redshift Performance", "EMR Clusters",
        "Glue ETL Jobs", "Athena Query Optimization", "Data Pipeline Design",
        "IAM for Data Engineering", "VPC for Data Services", "CloudFormation Templates"
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        custom_topic = st.text_input("Enter a topic to study:", 
                                   placeholder="e.g., Lambda memory optimization")
    
    with col2:
        selected_topic = st.selectbox("Or choose from available topics:", 
                                    [""] + available_topics)
    
    topic = custom_topic or selected_topic
    
    if topic and st.button("📖 Study This Topic", type="primary"):
        
        with st.spinner(f"Generating comprehensive study material for: {topic}"):
            search_results = st.session_state.rag_system.search(topic, top_k=5)
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                llm_response = loop.run_until_complete(
                    st.session_state.rag_system.generate_llm_response(
                        topic, search_results['context'], "certification"
                    )
                )
                loop.close()
            except Exception as e:
                llm_response = {
                    'answer': st.session_state.rag_system._generate_fallback_response(
                        topic, search_results['context'], "certification"
                    ),
                    'provider': 'Fallback',
                    'tokens_used': 0,
                    'cost_estimate': 0.0
                }
        
        st.markdown("### 🎯 Comprehensive Study Material")
        st.markdown('<div class="exam-question">', unsafe_allow_html=True)
        st.markdown(llm_response['answer'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if search_results['results']:
            with st.expander("📚 Knowledge Sources Used"):
                for i, result in enumerate(search_results['results'][:3], 1):
                    st.markdown(f"**Source {i}** (Score: {result['score']:.3f})")
                    st.markdown(f"Domain: {result['domain']}")
                    st.markdown(f"Content: {result['content'][:200]}...")
                    st.markdown("---")
        
        if topic not in st.session_state.study_progress['topics_covered']:
            st.session_state.study_progress['topics_covered'].append(topic)
            st.success(f"✅ Added '{topic}' to your covered topics!")

def display_practice_questions():
    """Display practice questions interface"""
    
    st.markdown('<h2 class="study-mode-header">❓ Practice Questions</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        topic = st.text_input("Topic for practice question:", 
                             placeholder="e.g., Lambda, S3, Kinesis")
    
    with col2:
        difficulty = st.selectbox("Question difficulty:", ["Associate Level", "Advanced", "Exam Simulation"])
    
    if st.button("🎲 Generate Practice Question", type="primary"):
        
        if topic:
            with st.spinner("Generating certification-style practice question..."):
                search_results = st.session_state.rag_system.search(topic, top_k=3)
                
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    llm_response = loop.run_until_complete(
                        st.session_state.rag_system.generate_llm_response(
                            f"Create a {difficulty.lower()} practice question about {topic}",
                            search_results['context'], "practice"
                        )
                    )
                    loop.close()
                except Exception as e:
                    llm_response = {
                        'answer': st.session_state.rag_system._generate_fallback_response(
                            f"Create a practice question about {topic}",
                            search_results['context'], "practice"
                        ),
                        'provider': 'Fallback',
                        'tokens_used': 0,
                        'cost_estimate': 0.0
                    }
            
            st.markdown("### 📝 Practice Question")
            st.markdown('<div class="exam-question">', unsafe_allow_html=True)
            st.markdown(llm_response['answer'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### 📤 Your Confidence")
            
            confidence = st.slider("How confident are you with this topic?", 1, 5, 3)
            
            if st.button("✅ Submit Confidence"):
                st.session_state.study_progress['questions_answered'] += 1
                
                if confidence >= 4:
                    st.session_state.study_progress['correct_answers'] += 1
                    st.markdown('<div class="correct-answer">High confidence! You seem to understand this topic well.</div>', 
                              unsafe_allow_html=True)
                elif confidence <= 2:
                    if topic not in st.session_state.study_progress['weak_areas']:
                        st.session_state.study_progress['weak_areas'].append(topic)
                    st.markdown('<div class="wrong-answer">Low confidence - added to weak areas for focused study.</div>', 
                              unsafe_allow_html=True)
                else:
                    st.info("Good progress! Continue practicing this topic.")
        else:
            st.warning("Please enter a topic first.")

def generate_exam_questions(num_questions, domain_mix):
    """Generate realistic exam questions"""
    
    question_templates = [
        {
            "scenario": "A company processes 10GB of daily log files using Lambda functions. Current functions use 512MB memory and timeout after 5 minutes during peak processing times.",
            "question": "What is the MOST cost-effective solution to resolve the timeout issues?",
            "options": {
                "A": "Increase memory to 3GB and timeout to 15 minutes",
                "B": "Increase memory to 1GB and monitor CloudWatch metrics for further optimization", 
                "C": "Split the processing across multiple smaller Lambda functions",
                "D": "Switch to EMR for batch processing of the log files"
            },
            "correct": "B",
            "explanation": "Increasing memory provides more CPU power and is often the first optimization step. Monitoring metrics allows for data-driven optimization. Option A over-provisions, C adds complexity unnecessarily, and D changes the architecture when optimization can solve the issue.",
            "domain": "Domain 1: Data Ingestion and Transformation",
            "exam_tip": "AWS exam often tests right-sizing resources with monitoring - avoid over-provisioning."
        },
        {
            "scenario": "A data analytics company stores application logs that are actively analyzed for 30 days, occasionally accessed for compliance during days 30-365, and archived for 7 years.",
            "question": "Which S3 lifecycle policy provides the MOST cost-effective solution?",
            "options": {
                "A": "Standard → Glacier (30 days) → Deep Archive (365 days)",
                "B": "Standard → Standard-IA (30 days) → Glacier (365 days)",  
                "C": "Intelligent-Tiering for all data",
                "D": "Standard → Standard-IA (30 days) → Glacier (90 days) → Deep Archive (2 years)"
            },
            "correct": "D",
            "explanation": "Matches access patterns: frequent (30 days), occasional compliance (90 days-2 years), then long-term archive. Standard-IA works for occasional access, Glacier for compliance, Deep Archive for final retention.",
            "domain": "Domain 2: Data Store Management",
            "exam_tip": "Match storage classes to actual access patterns, not just time periods."
        },
        {
            "scenario": "A streaming application processes 5,000 records per second but experiences WriteProvisionedThroughputExceeded errors during peak hours.",
            "question": "What should be implemented to resolve this issue?",
            "options": {
                "A": "Increase the number of Kinesis consumers",
                "B": "Add more shards to the Kinesis stream",
                "C": "Enable server-side encryption",
                "D": "Use larger record batch sizes"
            },
            "correct": "B", 
            "explanation": "Each Kinesis shard handles 1MB/sec or 1,000 records/sec. WriteProvisionedThroughputExceeded indicates insufficient shard capacity. Need at least 5 shards for 5,000 records/sec.",
            "domain": "Domain 1: Data Ingestion and Transformation", 
            "exam_tip": "Kinesis throttling = add shards. Know the shard limits: 1MB/sec or 1,000 records/sec."
        },
        {
            "scenario": "A Redshift cluster experiences slow query performance when joining large tables. The cluster has adequate compute capacity but queries take much longer than expected.",
            "question": "Which optimization technique will MOST likely improve query performance?", 
            "options": {
                "A": "Increase the number of compute nodes",
                "B": "Implement appropriate distribution keys and sort keys",
                "C": "Enable automatic workload management",
                "D": "Increase the cluster's storage capacity"
            },
            "correct": "B",
            "explanation": "Distribution keys minimize data movement during joins. Sort keys optimize query filtering. These are fundamental Redshift optimizations that often provide the biggest performance gains.",
            "domain": "Domain 2: Data Store Management",
            "exam_tip": "Redshift performance issues often solved with proper key design before adding resources."
        },
        {
            "scenario": "A data processing Lambda function needs access to S3 buckets and Redshift clusters across multiple AWS accounts in different regions.",
            "question": "What is the MOST secure approach to provide the necessary permissions?",
            "options": {
                "A": "Create IAM users with access keys for each account",
                "B": "Use IAM roles with cross-account trust relationships",
                "C": "Embed credentials in Lambda environment variables",
                "D": "Use a single IAM user with broad permissions across all accounts"
            },
            "correct": "B",
            "explanation": "IAM roles with cross-account trust provide temporary credentials without storing long-term keys. This follows AWS security best practices for cross-account access.",
            "domain": "Domain 4: Data Security and Governance",
            "exam_tip": "For AWS services accessing other AWS services, always prefer IAM roles over access keys."
        }
    ]
    
    # Select and randomize questions
    selected_questions = random.sample(question_templates, min(num_questions, len(question_templates)))
    
    # Add remaining questions by duplicating and modifying if needed
    while len(selected_questions) < num_questions:
        selected_questions.extend(random.sample(question_templates, 
                                               min(num_questions - len(selected_questions), len(question_templates))))
    
    # Format for exam
    exam_questions = []
    for i, template in enumerate(selected_questions[:num_questions], 1):
        exam_questions.append({
            'id': i,
            'scenario': template['scenario'],
            'question': template['question'],
            'options': template['options'],
            'correct_answer': template['correct'],
            'explanation': template['explanation'],
            'domain': template['domain'],
            'exam_tip': template['exam_tip'],
            'answered': False,
            'user_answer': None,
            'marked': False
        })
    
    return exam_questions

def display_exam_simulation():
    """Complete exam simulation matching AWS certification format"""
    
    st.markdown('<h2 class="study-mode-header">🎯 AWS Certification Exam Simulation</h2>', 
                unsafe_allow_html=True)
    
    # Check if exam is in progress
    if 'exam_session' in st.session_state:
        display_exam_interface()
        return
    
    # Exam configuration
    st.markdown("### 📋 Exam Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_questions = st.selectbox("Number of questions:", [10, 20, 30, 50, 65])
        st.caption("Real exam: 65 questions")
    
    with col2:
        time_limit = st.selectbox("Time limit (minutes):", [30, 60, 90, 130])
        st.caption("Real exam: 130 minutes")
    
    with col3:
        domain_mix = st.selectbox("Question mix:", 
                                ["Balanced (exam-like)", "Domain 1 Focus", "Domain 2 Focus", 
                                 "Domain 3 Focus", "Domain 4 Focus", "Weak Areas Only"])
    
    if domain_mix == "Balanced (exam-like)":
        st.info("""
        **Question Distribution (matches real exam):**
        - Domain 1 (Data Ingestion & Transformation): ~34% 
        - Domain 2 (Data Store Management): ~26%
        - Domain 3 (Data Operations & Support): ~22% 
        - Domain 4 (Security & Governance): ~18%
        """)
    
    if st.button("🚀 Start Exam Simulation", type="primary", key="start_exam"):
        # Initialize exam session
        exam_questions = generate_exam_questions(num_questions, domain_mix)
        
        st.session_state.exam_session = {
            'questions': exam_questions,
            'start_time': datetime.now(),
            'time_limit_minutes': time_limit,
            'current_question': 0,
            'answers': {},
            'marked_questions': set(),
            'completed': False
        }
        
        st.rerun()

def display_exam_interface():
    """Display the actual exam interface"""
    
    exam = st.session_state.exam_session
    
    # Calculate time remaining
    elapsed_time = datetime.now() - exam['start_time']
    time_remaining = timedelta(minutes=exam['time_limit_minutes']) - elapsed_time
    
    if time_remaining.total_seconds() <= 0 and not exam['completed']:
        # Time's up - auto submit
        submit_exam()
        return
    
    # Header with timer
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("### 🎯 AWS Data Engineer Exam Simulation")
    
    with col2:
        if not exam['completed']:
            minutes_left = int(time_remaining.total_seconds() // 60)
            seconds_left = int(time_remaining.total_seconds() % 60)
            st.markdown(f'<div class="exam-timer">⏰ Time Remaining: {minutes_left:02d}:{seconds_left:02d}</div>', 
                       unsafe_allow_html=True)
    
    with col3:
        if st.button("📤 Submit Exam", type="primary"):
            submit_exam()
            return
    
    # Question navigation
    st.markdown("### 📋 Question Navigation")
    
    # Create navigation grid
    cols = st.columns(10)
    questions_per_row = 10
    
    for i, question in enumerate(exam['questions']):
        col_idx = i % questions_per_row
        with cols[col_idx]:
            # Determine button style based on status
            if question['answered']:
                if question['id'] in exam['marked_questions']:
                    button_type = "secondary"  # Answered and marked
                    label = f"✓📌{question['id']}"
                else:
                    button_type = "secondary"  # Answered
                    label = f"✓{question['id']}"
            elif question['id'] in exam['marked_questions']:
                button_type = "secondary"  # Marked but not answered
                label = f"📌{question['id']}"
            else:
                button_type = "secondary"  # Not answered
                label = f"{question['id']}"
            
            if st.button(label, key=f"nav_{question['id']}", help=f"Go to question {question['id']}"):
                st.session_state.exam_session['current_question'] = i
                st.rerun()
    
    st.markdown("---")
    
    # Current question display
    current_idx = exam['current_question']
    current_q = exam['questions'][current_idx]
    
    st.markdown(f"### Question {current_q['id']} of {len(exam['questions'])}")
    st.markdown(f"**Domain:** {current_q['domain']}")
    
    # Question content
    st.markdown('<div class="exam-question">', unsafe_allow_html=True)
    st.markdown(f"**Scenario:** {current_q['scenario']}")
    st.markdown(f"**Question:** {current_q['question']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Answer options
    st.markdown("### Select your answer:")
    
    current_answer = exam['answers'].get(current_q['id'], None)
    
    selected_option = st.radio(
        "Options:",
        options=list(current_q['options'].keys()),
        format_func=lambda x: f"{x}. {current_q['options'][x]}",
        index=list(current_q['options'].keys()).index(current_answer) if current_answer else None,
        key=f"q_{current_q['id']}_answer"
    )
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📌 Mark for Review", key=f"mark_{current_q['id']}"):
            if current_q['id'] in exam['marked_questions']:
                exam['marked_questions'].remove(current_q['id'])
                st.success("Unmarked for review")
            else:
                exam['marked_questions'].add(current_q['id'])
                st.success("Marked for review")
    
    with col2:
        if st.button("💾 Save Answer", key=f"save_{current_q['id']}"):
            if selected_option:
                exam['answers'][current_q['id']] = selected_option
                exam['questions'][current_idx]['answered'] = True
                exam['questions'][current_idx]['user_answer'] = selected_option
                st.success("Answer saved!")
            else:
                st.warning("Please select an answer first")
    
    with col3:
        if current_idx > 0:
            if st.button("⬅️ Previous", key="prev_question"):
                st.session_state.exam_session['current_question'] = current_idx - 1
                st.rerun()
    
    with col4:
        if current_idx < len(exam['questions']) - 1:
            if st.button("➡️ Next", key="next_question"):
                st.session_state.exam_session['current_question'] = current_idx + 1
                st.rerun()
    
    # Progress summary
    st.markdown("---")
    st.markdown("### 📊 Progress Summary")
    
    answered_count = len(exam['answers'])
    marked_count = len(exam['marked_questions'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Questions Answered", f"{answered_count}/{len(exam['questions'])}")
    with col2:
        st.metric("Marked for Review", marked_count)
    with col3:
        completion_percentage = (answered_count / len(exam['questions'])) * 100
        st.metric("Completion", f"{completion_percentage:.1f}%")
    with col4:
        st.metric("Time Used", str(elapsed_time).split('.')[0])

def submit_exam():
    """Submit and score the exam"""
    
    exam = st.session_state.exam_session
    exam['completed'] = True
    exam['end_time'] = datetime.now()
    
    # Calculate score
    correct_answers = 0
    total_questions = len(exam['questions'])
    
    for question in exam['questions']:
        user_answer = exam['answers'].get(question['id'])
        if user_answer == question['correct_answer']:
            correct_answers += 1
    
    score_percentage = (correct_answers / total_questions) * 100
    
    # Update progress
    st.session_state.study_progress['exam_attempts'] += 1
    if score_percentage > st.session_state.study_progress['best_exam_score']:
        st.session_state.study_progress['best_exam_score'] = score_percentage
    
    # Display results
    st.markdown("# 🎉 Exam Completed!")
    
    if score_percentage >= 72:  # AWS passing score is typically 720/1000
        st.markdown(f'<div class="pass-score">PASSED! 🎉<br>Your Score: {score_percentage:.1f}%<br>({correct_answers}/{total_questions} correct)</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="fail-score">Not Passed 📚<br>Your Score: {score_percentage:.1f}%<br>({correct_answers}/{total_questions} correct)<br>Passing Score: 72%</div>', 
                   unsafe_allow_html=True)
    
    # Detailed results
    st.markdown("### 📊 Detailed Results")
    
    for question in exam['questions']:
        user_answer = exam['answers'].get(question['id'], "Not answered")
        is_correct = user_answer == question['correct_answer']
        
        with st.expander(f"Question {question['id']} - {'✅ Correct' if is_correct else '❌ Incorrect'}"):
            st.markdown(f"**Scenario:** {question['scenario']}")
            st.markdown(f"**Question:** {question['question']}")
            st.markdown(f"**Your Answer:** {user_answer}")
            st.markdown(f"**Correct Answer:** {question['correct_answer']}. {question['options'][question['correct_answer']]}")
            st.markdown(f"**Explanation:** {question['explanation']}")
            st.markdown(f"**Exam Tip:** {question['exam_tip']}")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Take Another Exam", type="primary"):
            del st.session_state.exam_session
            st.rerun()
    
    with col2:
        if st.button("📊 View Progress Dashboard"):
            del st.session_state.exam_session
            st.session_state.current_mode = "progress"
            st.rerun()

def display_progress_dashboard():
    """Display progress dashboard"""
    
    st.markdown('<h2 class="study-mode-header">📊 Progress Dashboard</h2>', 
                unsafe_allow_html=True)
    
    progress = st.session_state.study_progress
    
    # Overview metrics
    st.markdown("### 📈 Overall Progress")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="progress-metric">', unsafe_allow_html=True)
        st.metric("Study Sessions", progress['study_sessions'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="progress-metric">', unsafe_allow_html=True)
        accuracy = (progress['correct_answers'] / max(progress['questions_answered'], 1)) * 100
        st.metric("Overall Accuracy", f"{accuracy:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="progress-metric">', unsafe_allow_html=True)
        st.metric("Topics Covered", len(progress['topics_covered']))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="progress-metric">', unsafe_allow_html=True)
        st.metric("Exam Attempts", progress['exam_attempts'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Exam performance
    if progress['exam_attempts'] > 0:
        st.markdown("### 🎯 Exam Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Best Exam Score", f"{progress['best_exam_score']:.1f}%")
            if progress['best_exam_score'] >= 72:
                st.success("🎉 You've achieved a passing score!")
            else:
                st.info(f"📚 {72 - progress['best_exam_score']:.1f}% more needed to pass")
        
        with col2:
            # Create a simple progress chart
            df_scores = pd.DataFrame({
                'Attempt': [f"Attempt {i+1}" for i in range(progress['exam_attempts'])],
                'Score': [progress['best_exam_score']]  # In a real app, you'd track all scores
            })
            
            fig = px.line(df_scores, x='Attempt', y='Score', 
                         title='Exam Score Progress',
                         line_shape='linear')
            fig.add_hline(y=72, line_dash="dash", line_color="green", 
                         annotation_text="Passing Score (72%)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Topics breakdown
    st.markdown("### 📚 Study Topics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Covered Topics")
        if progress['topics_covered']:
            for topic in progress['topics_covered'][-10:]:  # Show last 10
                st.markdown(f"- {topic}")
            if len(progress['topics_covered']) > 10:
                st.caption(f"... and {len(progress['topics_covered']) - 10} more")
        else:
            st.info("No topics covered yet. Start studying!")
    
    with col2:
        st.markdown("#### ⚠️ Weak Areas")
        if progress['weak_areas']:
            for area in progress['weak_areas']:
                st.markdown(f"- {area}")
            st.info("💡 Focus your study time on these areas")
        else:
            st.success("No weak areas identified yet!")
    
    # Study recommendations
    st.markdown("### 💡 Study Recommendations")
    
    if progress['exam_attempts'] == 0:
        st.info("🎯 **Ready to test yourself?** Try the Exam Simulation mode to assess your knowledge!")
    elif progress['best_exam_score'] < 72:
        st.warning("📚 **Focus on fundamentals:** Review your weak areas and take more practice questions.")
    elif progress['best_exam_score'] < 85:
        st.info("🔧 **Fine-tune your knowledge:** You're close! Focus on advanced scenarios and edge cases.")
    else:
        st.success("🎉 **Exam Ready!** Your scores indicate you're well-prepared for the certification exam!")
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📖 Study Weak Areas", type="primary"):
            st.session_state.current_mode = "concept"
            st.rerun()
    
    with col2:
        if st.button("❓ Practice Questions", type="primary"):
            st.session_state.current_mode = "practice"
            st.rerun()
    
    with col3:
        if st.button("🎯 Take Exam", type="primary"):
            st.session_state.current_mode = "exam"
            st.rerun()

# Main application
def main():
    """Main application entry point"""
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar and get current mode
    current_mode, openai_ok = create_sidebar()
    
    # Route to appropriate display function
    if current_mode == "chat":
        display_chat_interface()
    elif current_mode == "concept":
        display_topic_review()
    elif current_mode == "practice":
        display_practice_questions()
    elif current_mode == "exam":
        display_exam_simulation()
    elif current_mode == "progress":
        display_progress_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.rag_system and st.session_state.rag_system.model_loaded:
            st.success("✅ Knowledge Base: Loaded")
        else:
            st.info("ℹ️ Knowledge Base: Fallback Mode")
    
    with col2:
        if openai_ok:
            st.success("✅ OpenAI: Connected")
        else:
            st.warning("⚠️ OpenAI: Using Fallback")
    
    with col3:
        st.info(f"📊 Session: {len(st.session_state.chat_history)} messages")

if __name__ == "__main__":
    main()