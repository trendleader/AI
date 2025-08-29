"""
AWS Data Engineer Certification Prep Streamlit Application
A complete chatbot interface for AWS certification preparation
"""

import streamlit as st
import asyncio
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="AWS Data Engineer Cert Prep",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'study_progress' not in st.session_state:
        st.session_state.study_progress = {
            'questions_answered': 0,
            'correct_answers': 0,
            'topics_covered': [],
            'weak_areas': [],
            'study_sessions': 0,
            'total_study_time': 0
        }
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

@st.cache_resource
def load_rag_system():
    """Load the RAG system (cached for performance)"""
    try:
        # Import your systems
        from env_loader_integration import setup_api_keys
        from aws_certification_prep_rag import AWSCertificationPrepRAG
        
        # Setup API keys
        services = setup_api_keys()
        
        if not services.get('openai'):
            st.error("OpenAI API key not found. Please add it to your test.env file.")
            return None
        
        # Initialize certification prep system
        cert_prep = AWSCertificationPrepRAG()
        
        # Try to load existing knowledge base
        try:
            cert_prep.rag_system.load_enhanced_model("aws_enhanced_rag_model.pkl")
        except FileNotFoundError:
            st.warning("Knowledge base not found. Please build it first using the setup scripts.")
        
        return cert_prep
    
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None

def create_sidebar():
    """Create the sidebar with navigation and progress"""
    
    st.sidebar.markdown("# 🎓 AWS Data Engineer Prep")
    
    # Study mode selection
    study_modes = {
        "💬 Chat Mode": "chat",
        "📚 Topic Review": "concept", 
        "❓ Practice Questions": "practice",
        "🎯 Exam Simulation": "exam",
        "📊 Progress Dashboard": "progress",
        "🎪 Weak Areas Focus": "weak_areas"
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
    
    return study_modes[selected_mode]

async def get_rag_response(question: str, mode: str) -> Dict[str, Any]:
    """Get response from RAG system"""
    
    if not st.session_state.rag_system:
        return {"error": "RAG system not initialized"}
    
    try:
        from aws_certification_prep_rag import StudyMode
        
        # Map modes to study modes
        mode_map = {
            "chat": StudyMode.CONCEPT_REVIEW,
            "concept": StudyMode.CONCEPT_REVIEW,
            "practice": StudyMode.PRACTICE_QUESTIONS,
            "exam": StudyMode.EXAM_SIMULATION
        }
        
        study_mode = mode_map.get(mode, StudyMode.CONCEPT_REVIEW)
        
        # Get response
        response = await st.session_state.rag_system.study_session(question, study_mode)
        
        return response
        
    except Exception as e:
        return {"error": f"Error getting response: {e}"}

def display_chat_interface():
    """Display the main chat interface"""
    
    st.markdown('<h1 class="main-header">AWS Data Engineer Certification Prep</h1>', 
                unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me about AWS Data Engineering concepts, or type 'practice [topic]' for questions"):
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        # Determine response mode based on input
        mode = "chat"
        if prompt.lower().startswith("practice"):
            mode = "practice"
        elif prompt.lower().startswith("exam"):
            mode = "exam"
        
        # Show thinking spinner
        with st.spinner("Thinking..."):
            # Get response (simulate async call)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(get_rag_response(prompt, mode))
            loop.close()
        
        if "error" not in response:
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.get('answer', 'No response generated'),
                "mode": mode,
                "metadata": response.get('llm_metadata', {}),
                "timestamp": datetime.now()
            })
            
            # Update progress
            st.session_state.study_progress['study_sessions'] += 1
            topic = prompt.split(' ', 1)[-1] if ' ' in prompt else prompt
            if topic not in st.session_state.study_progress['topics_covered']:
                st.session_state.study_progress['topics_covered'].append(topic)
        
        else:
            st.error(response['error'])
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                if metadata:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"⏱️ {metadata.get('response_time_ms', 0):.0f}ms")
                    with col2:
                        st.caption(f"💰 ${metadata.get('cost_estimate', 0):.4f}")
                    with col3:
                        st.caption(f"🔤 {metadata.get('tokens_used', 0)} tokens")

def display_topic_review():
    """Display topic review interface"""
    
    st.markdown('<h2 class="study-mode-header">📚 Topic Review Mode</h2>', 
                unsafe_allow_html=True)
    
    # Topic selection
    common_topics = [
        "Lambda optimization for data processing",
        "S3 lifecycle policies and storage classes", 
        "Kinesis vs MSK for streaming data",
        "DynamoDB vs RDS selection criteria",
        "AWS Glue ETL job optimization",
        "Redshift performance tuning",
        "Data pipeline monitoring with CloudWatch",
        "IAM roles and policies for data services",
        "Data encryption and security best practices",
        "EMR cluster configuration and optimization"
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        custom_topic = st.text_input("Enter a topic to study:", 
                                   placeholder="e.g., Lambda memory optimization")
    
    with col2:
        selected_topic = st.selectbox("Or choose from common topics:", 
                                    [""] + common_topics)
    
    topic = custom_topic or selected_topic
    
    if topic and st.button("📖 Study This Topic", type="primary"):
        
        with st.spinner(f"Preparing comprehensive study material for: {topic}"):
            # Get detailed explanation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(get_rag_response(topic, "concept"))
            loop.close()
        
        if "error" not in response:
            st.markdown("### 🎯 Study Material")
            
            # Display the study content in an attractive format
            st.markdown('<div class="exam-question">', unsafe_allow_html=True)
            st.markdown(response['answer'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add to progress
            if topic not in st.session_state.study_progress['topics_covered']:
                st.session_state.study_progress['topics_covered'].append(topic)
                st.success(f"✅ Added '{topic}' to your covered topics!")
        
        else:
            st.error(response['error'])

def display_practice_questions():
    """Display practice questions interface"""
    
    st.markdown('<h2 class="study-mode-header">❓ Practice Questions</h2>', 
                unsafe_allow_html=True)
    
    # Question generation
    domains = [
        "Data Ingestion and Transformation",
        "Data Store Management", 
        "Data Operations and Support",
        "Data Security and Governance"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_domain = st.selectbox("Select domain:", ["All Domains"] + domains)
    
    with col2:
        topic = st.text_input("Specific topic (optional):", 
                             placeholder="e.g., Lambda, S3, Kinesis")
    
    if st.button("🎲 Generate Practice Question", type="primary"):
        
        question_topic = topic or selected_domain
        
        with st.spinner("Generating practice question..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop) 
            response = loop.run_until_complete(get_rag_response(question_topic, "practice"))
            loop.close()
        
        if "error" not in response:
            # Store current question for answer checking
            st.session_state.current_question = {
                'content': response['answer'],
                'topic': question_topic,
                'generated_at': datetime.now()
            }
            
            # Display question
            st.markdown("### 📝 Practice Question")
            st.markdown('<div class="exam-question">', unsafe_allow_html=True)
            st.markdown(response['answer'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Answer submission
            st.markdown("### 📤 Submit Your Answer")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_answer = st.text_area("Your answer explanation:", 
                                         placeholder="Explain your reasoning...")
            
            with col2:
                st.markdown("**Confidence Level**")
                confidence = st.slider("How confident are you?", 1, 5, 3)
            
            if st.button("✅ Submit Answer"):
                st.session_state.study_progress['questions_answered'] += 1
                
                # Simple confidence-based feedback
                if confidence >= 4:
                    st.session_state.study_progress['correct_answers'] += 1
                    st.markdown('<div class="correct-answer">Great job! High confidence usually indicates good understanding.</div>', 
                              unsafe_allow_html=True)
                elif confidence <= 2:
                    # Add to weak areas
                    if question_topic not in st.session_state.study_progress['weak_areas']:
                        st.session_state.study_progress['weak_areas'].append(question_topic)
                    
                    st.markdown('<div class="wrong-answer">This topic needs more study. Added to your weak areas.</div>', 
                              unsafe_allow_html=True)
                else:
                    st.info("Good effort! Keep practicing this topic.")
        
        else:
            st.error(response['error'])

def display_exam_simulation():
    """Display exam simulation interface"""
    
    st.markdown('<h2 class="study-mode-header">🎯 Exam Simulation</h2>', 
                unsafe_allow_html=True)
    
    st.info("💡 This mode generates full exam-style multiple choice questions with detailed explanations.")
    
    # Exam configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_questions = st.selectbox("Number of questions:", [5, 10, 15, 20, 25])
    
    with col2:
        time_limit = st.selectbox("Time limit (minutes):", [15, 30, 45, 60, 90])
    
    with col3:
        domain_focus = st.selectbox("Domain focus:", 
                                  ["Mixed", "Data Ingestion", "Data Store Management", 
                                   "Data Operations", "Data Security"])
    
    if st.button("🚀 Start Exam Simulation", type="primary"):
        
        st.markdown("### ⏰ Exam Started!")
        
        # Create sample exam questions (in real implementation, generate from RAG)
        sample_questions = [
            {
                'question': "A company processes 10TB of log data daily using Lambda functions. The functions frequently timeout after 5 minutes. What is the BEST solution?",
                'options': {
                    'A': 'Increase Lambda timeout to 15 minutes',
                    'B': 'Switch to EMR for large-scale processing',
                    'C': 'Use smaller batch sizes with multiple Lambda invocations', 
                    'D': 'Increase Lambda memory allocation'
                },
                'correct': 'B',
                'explanation': 'Lambda has a 15-minute maximum timeout. For 10TB daily processing, EMR is more appropriate for large-scale batch processing.',
                'domain': 'Data Ingestion and Transformation'
            }
        ]
        
        # Display questions
        for i, q in enumerate(sample_questions[:num_questions], 1):
            st.markdown(f"### Question {i}")
            st.markdown('<div class="exam-question">', unsafe_allow_html=True)
            
            st.markdown(f"**{q['question']}**")
            st.markdown("")
            
            # Display options as radio buttons
            answer = st.radio(f"Select your answer for Question {i}:", 
                            list(q['options'].values()),
                            key=f"q{i}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("📊 Submit Exam"):
            # Calculate score (simplified)
            score = 85  # Mock score
            
            st.markdown("### 🎉 Exam Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Score", f"{score}%", delta="15%" if score > 70 else "-5%")
            
            with col2:
                st.metric("Questions Correct", f"{int(score/100 * num_questions)}/{num_questions}")
            
            with col3:
                status = "PASS ✅" if score >= 72 else "NEEDS IMPROVEMENT ⚠️"
                st.metric("Status", status)
            
            # Update progress
            st.session_state.study_progress['questions_answered'] += num_questions
            st.session_state.study_progress['correct_answers'] += int(score/100 * num_questions)

def display_progress_dashboard():
    """Display comprehensive progress dashboard"""
    
    st.markdown('<h2 class="study-mode-header">📊 Progress Dashboard</h2>', 
                unsafe_allow_html=True)
    
    progress = st.session_state.study_progress
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="progress-metric">', unsafe_allow_html=True)
        st.metric("Questions Answered", progress['questions_answered'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        accuracy = (progress['correct_answers'] / max(progress['questions_answered'], 1)) * 100
        st.markdown('<div class="progress-metric">', unsafe_allow_html=True)
        st.metric("Accuracy", f"{accuracy:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="progress-metric">', unsafe_allow_html=True)
        st.metric("Topics Covered", len(progress['topics_covered']))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="progress-metric">', unsafe_allow_html=True)
        st.metric("Study Sessions", progress['study_sessions'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Progress charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Study Progress Over Time")
        
        # Mock data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        cumulative_questions = [i + random.randint(0, 3) for i in range(len(dates))]
        
        fig = px.line(x=dates, y=cumulative_questions, 
                     title="Cumulative Questions Answered",
                     labels={'x': 'Date', 'y': 'Questions'})
        fig.update_traces(line_color='#FF9900')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Domain Coverage")
        
        domains = ["Data Ingestion", "Data Store Mgmt", "Data Operations", "Security & Governance"]
        coverage = [85, 72, 68, 45]  # Mock data
        
        fig = px.bar(x=domains, y=coverage,
                    title="Coverage by Domain (%)",
                    color=coverage,
                    color_continuous_scale=['red', 'yellow', 'green'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Weak areas analysis
    if progress['weak_areas']:
        st.markdown("### 🎪 Areas Needing Focus")
        
        for area in progress['weak_areas']:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.warning(f"📚 {area}")
            with col2:
                if st.button(f"Study {area}", key=f"study_{area}"):
                    st.session_state.study_topic = area
                    st.experimental_rerun()
    
    # Study recommendations
    st.markdown("### 💡 Study Recommendations")
    
    if accuracy < 70:
        st.error("🚨 Focus on fundamental concepts before attempting practice exams")
    elif accuracy < 80:
        st.warning("⚠️ Good progress! Increase practice question frequency")
    else:
        st.success("🎉 You're ready for exam simulation mode!")

def display_weak_areas_focus():
    """Display weak areas focus interface"""
    
    st.markdown('<h2 class="study-mode-header">🎪 Weak Areas Focus</h2>', 
                unsafe_allow_html=True)
    
    progress = st.session_state.study_progress
    
    if not progress['weak_areas']:
        st.info("🎉 No weak areas identified yet! Keep practicing to get personalized recommendations.")
        
        # Suggest starting areas
        st.markdown("### 📚 Suggested Starting Topics")
        suggested = [
            "AWS Lambda configuration and optimization",
            "S3 storage classes and lifecycle policies",
            "Data pipeline orchestration with Step Functions",
            "IAM policies for data services"
        ]
        
        for topic in suggested:
            if st.button(f"📖 Study: {topic}", key=f"suggest_{topic}"):
                # Redirect to topic review
                pass
    
    else:
        st.markdown("### 🎯 Your Identified Weak Areas")
        
        for i, area in enumerate(progress['weak_areas'], 1):
            
            with st.expander(f"{i}. {area}", expanded=i==1):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Focus Area:** {area}")
                    
                    # Mock confidence score
                    confidence = 65 + i * 5
                    st.progress(confidence / 100)
                    st.caption(f"Current confidence: {confidence}%")
                
                with col2:
                    if st.button(f"📚 Study Now", key=f"weak_{i}"):
                        with st.spinner(f"Loading study material for {area}..."):
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            response = loop.run_until_complete(get_rag_response(area, "concept"))
                            loop.close()
                        
                        if "error" not in response:
                            st.markdown("#### 📖 Focused Study Material")
                            st.markdown('<div class="exam-question">', unsafe_allow_html=True)
                            st.markdown(response['answer'])
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            if st.button("✅ Mark as Improved", key=f"improve_{i}"):
                                progress['weak_areas'].remove(area)
                                st.success(f"Removed {area} from weak areas!")
                                st.experimental_rerun()

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Load RAG system
    if not st.session_state.rag_system:
        with st.spinner("Loading AWS Data Engineer knowledge base..."):
            st.session_state.rag_system = load_rag_system()
    
    # Create sidebar and get selected mode
    selected_mode = create_sidebar()
    
    # Display appropriate interface based on selected mode
    if selected_mode == "chat":
        display_chat_interface()
    elif selected_mode == "concept":
        display_topic_review()
    elif selected_mode == "practice":
        display_practice_questions()
    elif selected_mode == "exam":
        display_exam_simulation()
    elif selected_mode == "progress":
        display_progress_dashboard()
    elif selected_mode == "weak_areas":
        display_weak_areas_focus()
    
    # Footer
    st.markdown("---")
    st.markdown("🎓 **AWS Certified Data Engineer - Associate Prep System** | Built with Streamlit + RAG + LLM")

if __name__ == "__main__":
    # Add import path setup
    import sys
    from pathlib import Path
    PROJECT_DIR = Path(r"C:\Users\pjone\OneDrive\Desktop\NewPython")
    sys.path.append(str(PROJECT_DIR))
    
    main()
