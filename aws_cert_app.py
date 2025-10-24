import streamlit as st
import random
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from questions_db import QUESTIONS_DATABASE, TOPICS

# Try to load from .env file if it exists, but don't require it
try:
    load_dotenv()
except:
    pass

# Initialize OpenAI client
def get_openai_client():
    """Initialize OpenAI client with API key"""
    # First check if API key is in session state (from Streamlit input)
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        api_key = st.session_state.openai_api_key
    else:
        # Fall back to environment variable
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    # Initialize with explicit settings to avoid proxy issues
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize OpenAI client: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="AWS Certification Exam Prep",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        color: #FF9900;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 1em;
    }
    .mode-button {
        padding: 1em;
        border-radius: 0.5em;
        margin: 0.5em;
    }
    .correct-answer {
        color: green;
        font-weight: bold;
    }
    .incorrect-answer {
        color: red;
        font-weight: bold;
    }
    .score-card {
        background-color: #f0f0f0;
        padding: 1.5em;
        border-radius: 0.5em;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'study_progress' not in st.session_state:
    st.session_state.study_progress = {}
if 'exam_answers' not in st.session_state:
    st.session_state.exam_answers = {}
if 'exam_started' not in st.session_state:
    st.session_state.exam_started = False
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'exam_questions' not in st.session_state:
    st.session_state.exam_questions = []

def get_openai_topic_details(topic: str, certification: str) -> str:
    """Get detailed information about a topic using OpenAI"""
    try:
        client = get_openai_client()
        
        prompt = f"""
        Provide a comprehensive study guide for AWS {certification} exam on the topic: {topic}
        
        Include:
        1. Key Concepts
        2. Important AWS Services Related
        3. Common Use Cases
        4. Best Practices
        5. Exam Tips
        
        Keep the response concise but informative (3-4 paragraphs).
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert AWS Solutions Architect helping candidates prepare for AWS certifications."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error fetching details: {str(e)}. Please ensure your OpenAI API key is valid and you have available credits."

def display_home():
    """Display the home/welcome page"""
    st.markdown('<h1 class="main-title">☁️ AWS Certification Exam Prep</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Show API key status
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        st.success("✓ OpenAI API Key Ready - Topic Explorer is available")
    elif os.getenv("OPENAI_API_KEY"):
        st.info("✓ OpenAI API Key loaded from .env - Topic Explorer is available")
    else:
        st.warning("⚠ No OpenAI API Key - Enter it in the sidebar to use Topic Explorer. Study and Exam modes work without it.")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Study Mode")
        st.write("Learn at your own pace with detailed explanations for each question.")
        if st.button("Start Study Mode", key="study_btn"):
            st.session_state.mode = "study"
            st.rerun()
    
    with col2:
        st.markdown("### 🎯 Exam Mode")
        st.write("Take a full practice exam with timed questions and scoring.")
        if st.button("Start Exam Mode", key="exam_btn"):
            st.session_state.mode = "exam"
            st.rerun()
    
    with col3:
        st.markdown("### 🔍 Topic Explorer")
        st.write("Search for topics and get AI-powered study materials.")
        if st.button("Open Topic Explorer", key="topic_btn"):
            st.session_state.mode = "topic_explorer"
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📋 Available Certifications")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**AWS Certified Machine Learning Engineer - Associate**")
        st.write(f"Total Questions: {len([q for q in QUESTIONS_DATABASE if q.get('certification') == 'ML Engineer'])}")
    with col2:
        st.info("**AWS Certified Data Engineer - Associate**")
        st.write(f"Total Questions: {len([q for q in QUESTIONS_DATABASE if q.get('certification') == 'Data Engineer'])}")

def display_study_mode():
    """Display study mode interface"""
    st.markdown("### 📚 Study Mode")
    st.markdown("Review questions with detailed explanations. Take your time to understand each concept.")
    st.markdown("---")
    
    # Certification filter
    certification = st.radio(
        "Select Certification:",
        ["Machine Learning Engineer", "Data Engineer"],
        horizontal=True
    )
    
    cert_key = "ML Engineer" if certification == "Machine Learning Engineer" else "Data Engineer"
    filtered_questions = [q for q in QUESTIONS_DATABASE if q.get('certification') == cert_key]
    
    if not filtered_questions:
        st.warning("No questions found for this certification.")
        return
    
    # Domain filter
    domains = sorted(set(q.get('domain', 'Unknown') for q in filtered_questions))
    selected_domain = st.selectbox("Filter by Domain:", ["All"] + domains)
    
    if selected_domain != "All":
        filtered_questions = [q for q in filtered_questions if q.get('domain') == selected_domain]
    
    # Question selector
    question_num = st.number_input(
        "Select Question Number:",
        min_value=1,
        max_value=len(filtered_questions),
        value=1
    )
    
    if filtered_questions:
        question = filtered_questions[question_num - 1]
        
        st.markdown("---")
        st.markdown(f"**Domain:** {question.get('domain', 'N/A')}")
        st.markdown(f"**Question {question_num} of {len(filtered_questions)}**")
        st.markdown("---")
        
        # Display question
        st.markdown(f"### {question['question']}")
        
        # Display options
        st.markdown("**Options:**")
        for i, option in enumerate(question['options']):
            st.write(f"{chr(65 + i)}. {option}")
        
        # Show answer and explanation
        with st.expander("📖 View Answer & Explanation", expanded=False):
            correct_answer = question['options'][question['correct_answer']]
            st.markdown(f"**Correct Answer:** {chr(65 + question['correct_answer'])}. {correct_answer}")
            st.markdown(f"**Explanation:**\n{question['explanation']}")
            if 'related_topic' in question:
                st.markdown(f"**Related Topic:** {question['related_topic']}")
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous Question"):
                st.session_state.current_question_index = max(0, st.session_state.current_question_index - 1)
                st.rerun()
        with col2:
            if st.button("Next Question →"):
                st.session_state.current_question_index = min(len(filtered_questions) - 1, st.session_state.current_question_index + 1)
                st.rerun()

def display_exam_mode():
    """Display exam mode interface"""
    st.markdown("### 🎯 Exam Mode")
    st.markdown("Take a practice exam. Select your answers and get your score at the end.")
    st.markdown("---")
    
    if not st.session_state.exam_started:
        col1, col2 = st.columns(2)
        with col1:
            num_questions = st.slider("Number of Questions:", 5, 50, 10)
        with col2:
            certification = st.radio(
                "Select Certification:",
                ["Machine Learning Engineer", "Data Engineer"],
                horizontal=True
            )
        
        if st.button("Start Exam", key="start_exam"):
            cert_key = "ML Engineer" if certification == "Machine Learning Engineer" else "Data Engineer"
            filtered_questions = [q for q in QUESTIONS_DATABASE if q.get('certification') == cert_key]
            st.session_state.exam_questions = random.sample(filtered_questions, min(num_questions, len(filtered_questions)))
            st.session_state.exam_started = True
            st.session_state.exam_answers = {}
            st.session_state.current_question_index = 0
            st.rerun()
    else:
        if len(st.session_state.exam_questions) == 0:
            st.error("No questions available for this exam.")
            if st.button("Back to Home"):
                st.session_state.mode = None
                st.session_state.exam_started = False
                st.rerun()
            return
        
        current_q_idx = st.session_state.current_question_index
        question = st.session_state.exam_questions[current_q_idx]
        
        # Progress bar
        progress = (current_q_idx + 1) / len(st.session_state.exam_questions)
        st.progress(progress)
        st.markdown(f"**Question {current_q_idx + 1} of {len(st.session_state.exam_questions)}**")
        st.markdown("---")
        
        # Display question
        st.markdown(f"### {question['question']}")
        
        # Display options as radio buttons
        selected_answer = st.radio(
            "Select your answer:",
            options=[f"{chr(65 + i)}. {option}" for i, option in enumerate(question['options'])],
            key=f"q_{current_q_idx}"
        )
        
        # Convert answer letter (A, B, C, D) to index (0, 1, 2, 3)
        answer_letter = selected_answer[0]  # Get first character (A, B, C, or D)
        st.session_state.exam_answers[current_q_idx] = ord(answer_letter) - ord('A')
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if current_q_idx > 0 and st.button("← Previous"):
                st.session_state.current_question_index -= 1
                st.rerun()
        with col2:
            if st.button("Submit Exam"):
                display_exam_results()
                return
        with col3:
            if current_q_idx < len(st.session_state.exam_questions) - 1 and st.button("Next →"):
                st.session_state.current_question_index += 1
                st.rerun()

def display_exam_results():
    """Display exam results"""
    st.markdown("---")
    st.markdown("### 📊 Exam Results")
    
    correct = 0
    total = len(st.session_state.exam_questions)
    
    results_data = []
    
    for idx, question in enumerate(st.session_state.exam_questions):
        user_answer = st.session_state.exam_answers.get(idx)
        is_correct = user_answer == question['correct_answer']
        if is_correct:
            correct += 1
        
        results_data.append({
            'question_num': idx + 1,
            'question': question['question'],
            'user_answer': question['options'][user_answer] if user_answer is not None else "Not answered",
            'correct_answer': question['options'][question['correct_answer']],
            'is_correct': is_correct
        })
    
    percentage = (correct / total) * 100
    
    # Display score card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correct Answers", f"{correct}/{total}")
    with col2:
        st.metric("Percentage", f"{percentage:.1f}%")
    with col3:
        status = "✅ PASS" if percentage >= 70 else "❌ Review"
        st.metric("Status", status)
    
    st.markdown("---")
    
    # Detailed results
    st.markdown("### Detailed Results")
    for result in results_data:
        with st.expander(f"Q{result['question_num']}: {result['question'][:60]}..."):
            st.markdown(f"**Your Answer:** {result['user_answer']}")
            if result['is_correct']:
                st.markdown(f"<p class='correct-answer'>✓ Correct!</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='incorrect-answer'>✗ Incorrect</p>", unsafe_allow_html=True)
                st.markdown(f"**Correct Answer:** {result['correct_answer']}")
    
    st.markdown("---")
    if st.button("Return to Home"):
        st.session_state.mode = None
        st.session_state.exam_started = False
        st.session_state.exam_answers = {}
        st.rerun()

def display_topic_explorer():
    """Display topic explorer with OpenAI integration"""
    st.markdown("### 🔍 Topic Explorer")
    st.markdown("Search for AWS topics and get detailed study materials powered by AI.")
    st.markdown("---")
    
    certification = st.radio(
        "Select Certification:",
        ["Machine Learning Engineer", "Data Engineer"],
        horizontal=True
    )
    
    # Get available topics
    cert_key = "ML Engineer" if certification == "Machine Learning Engineer" else "Data Engineer"
    available_topics = sorted(set(q.get('domain', 'General') for q in QUESTIONS_DATABASE if q.get('certification') == cert_key))
    
    # Topic search
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_topic = st.selectbox("Select a Topic:", available_topics, key="topic_select")
    with col2:
        search_button = st.button("📚 Get Details")
    
    # Custom topic search
    st.markdown("**Or search for a custom topic:**")
    custom_topic = st.text_input("Enter any AWS topic (e.g., 'Amazon S3', 'Lambda', 'DynamoDB'):", key="custom_topic")
    
    if search_button or st.button("Search Custom Topic"):
        topic_to_search = custom_topic if custom_topic else selected_topic
        
        if topic_to_search:
            st.markdown("---")
            st.markdown(f"### 📖 {topic_to_search}")
            
            with st.spinner("🔄 Fetching AI-powered study materials..."):
                details = get_openai_topic_details(topic_to_search, cert_key)
            
            st.markdown(details)
            
            # Show related questions
            st.markdown("---")
            st.markdown("### Related Practice Questions")
            related_questions = [
                q for q in QUESTIONS_DATABASE 
                if q.get('certification') == cert_key and topic_to_search.lower() in q.get('question', '').lower()
            ]
            
            if related_questions:
                for i, q in enumerate(related_questions[:5], 1):
                    with st.expander(f"Q{i}: {q['question'][:70]}..."):
                        st.write(f"**Question:** {q['question']}")
                        st.write("**Options:**")
                        for j, opt in enumerate(q['options']):
                            st.write(f"{chr(65 + j)}. {opt}")
                        with st.expander("Show Answer"):
                            st.markdown(f"**Answer:** {chr(65 + q['correct_answer'])}. {q['options'][q['correct_answer']]}")
                            st.markdown(f"**Explanation:** {q['explanation']}")
            else:
                st.info("No practice questions found for this topic in the database. Try searching another topic.")

# Main app layout
st.sidebar.markdown("### Navigation")

# Add API Key input in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 OpenAI API Key")
api_key_input = st.sidebar.text_input(
    "Enter your OpenAI API key:",
    value=st.session_state.get('openai_api_key', ''),
    type='password',
    help="Your API key is only stored in this session and is never saved. Get one at https://platform.openai.com/api-keys"
)

if api_key_input:
    st.session_state.openai_api_key = api_key_input
    st.sidebar.success("✓ API key saved for this session")
elif os.getenv("OPENAI_API_KEY"):
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")
    st.sidebar.info("ℹ Using API key from .env file")

st.sidebar.markdown("---")
if st.session_state.mode is None:
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.mode = None
    st.sidebar.markdown("---")
    display_home()
elif st.session_state.mode == "study":
    if st.sidebar.button("🏠 Back to Home", use_container_width=True):
        st.session_state.mode = None
        st.rerun()
    st.sidebar.markdown("---")
    display_study_mode()
elif st.session_state.mode == "exam":
    if st.sidebar.button("🏠 Back to Home", use_container_width=True):
        st.session_state.mode = None
        st.session_state.exam_started = False
        st.rerun()
    st.sidebar.markdown("---")
    display_exam_mode()
elif st.session_state.mode == "topic_explorer":
    if st.sidebar.button("🏠 Back to Home", use_container_width=True):
        st.session_state.mode = None
        st.rerun()
    st.sidebar.markdown("---")
    display_topic_explorer()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
AWS Certification Exam Prep | Data Engineer & ML Engineer Associate<br>
Powered by Streamlit + OpenAI | Last Updated: 2024
</div>
""", unsafe_allow_html=True)
