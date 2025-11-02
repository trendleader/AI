import streamlit as st
from openai import OpenAI
import json
from datetime import datetime
from typing import Optional

# Page Configuration
st.set_page_config(
    page_title="PL 300 Bootcamp Prep",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1em;
        padding: 10px 20px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "quiz_total" not in st.session_state:
    st.session_state.quiz_total = 0
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = []
if "study_history" not in st.session_state:
    st.session_state.study_history = []

# PL 300 Topics
PL300_TOPICS = {
    "Data Modeling": [
        "Star schema design",
        "Fact and dimension tables",
        "Cardinality and relationships",
        "Role-playing dimensions",
        "Snowflake vs Star schema"
    ],
    "Power Query": [
        "Data transformation techniques",
        "Query folding optimization",
        "Custom columns and conditional logic",
        "Merge and append queries",
        "Error handling in Power Query"
    ],
    "DAX": [
        "CALCULATE function",
        "Context transition",
        "FILTER vs CALCULATETABLE",
        "Time intelligence functions",
        "SUMX and other iterators"
    ],
    "Visualizations": [
        "Report design best practices",
        "Interactive filtering",
        "Bookmarks and drills",
        "Slicers and filtering panes",
        "Mobile layout design"
    ],
    "Data Analysis": [
        "Aggregations and hierarchies",
        "Implicit vs explicit measures",
        "Row-level security (RLS)",
        "Composite models",
        "Performance optimization"
    ]
}

def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key by making a simple API call"""
    try:
        client = OpenAI(api_key=api_key)
        # Make a minimal test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception as e:
        st.error(f"Invalid API key: {str(e)}")
        return False

def generate_quiz_question(topic: str, subtopic: str, api_key: str) -> dict:
    """Generate a quiz question using OpenAI"""
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""Generate a multiple-choice question for a Power BI PL 300 certification exam.
Topic: {topic}
Subtopic: {subtopic}

Format your response as JSON with this exact structure:
{{
    "question": "The question text",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "A",
    "explanation": "A brief explanation of why this is correct"
}}

IMPORTANT: The "correct_answer" field MUST be a single character: A, B, C, or D (not the full option text).

Only return the JSON, no other text."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        question_data = json.loads(response_text)
        return question_data
    except Exception as e:
        st.error(f"Error generating question: {str(e)}")
        return None

def generate_concept_explanation(topic: str, subtopic: str, api_key: str) -> str:
    """Generate a detailed concept explanation using OpenAI"""
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""Provide a comprehensive but concise explanation for someone studying for the Power BI PL 300 certification exam.

Topic: {topic}
Subtopic: {subtopic}

Include:
1. Clear definition
2. Real-world use case
3. Key points to remember
4. Common mistakes to avoid

Keep the explanation to 2-3 paragraphs, practical and focused on exam preparation."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")
        return None

def generate_practice_test(topic: str, api_key: str, num_questions: int = 5) -> list:
    """Generate a complete practice test"""
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""Generate {num_questions} multiple-choice questions for a Power BI PL 300 certification exam focused on the topic: {topic}

Format your response as a JSON array where each element has this structure:
{{
    "question": "The question text",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "A",
    "explanation": "A brief explanation"
}}

IMPORTANT: The "correct_answer" field MUST be a single character: A, B, C, or D (not the full option text).

Only return the JSON array, no other text."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        questions = json.loads(response_text)
        return questions
    except Exception as e:
        st.error(f"Error generating practice test: {str(e)}")
        return []

# Main App Layout
st.title("📊 Power BI PL 300 Bootcamp Prep")
st.markdown("Master the Microsoft Power BI Data Analyst certification with AI-powered practice")

# Sidebar - API Key Setup
with st.sidebar:
    st.header("⚙️ Setup")
    
    if not st.session_state.api_key_entered:
        st.warning("Enter your OpenAI API key to get started")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-..."
        )
        
        if api_key:
            if st.button("✓ Verify API Key", use_container_width=True):
                with st.spinner("Validating API key..."):
                    if validate_openai_key(api_key):
                        st.session_state.api_key = api_key
                        st.session_state.api_key_entered = True
                        st.success("API key validated!")
                        st.rerun()
                    else:
                        st.error("API key validation failed")
    else:
        st.success("✓ API Key Connected")
        if st.button("🔄 Change API Key", use_container_width=True):
            st.session_state.api_key_entered = False
            st.rerun()
    
    st.divider()
    
    # Stats
    if st.session_state.quiz_total > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score", f"{st.session_state.quiz_score}/{st.session_state.quiz_total}")
        with col2:
            percentage = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
            st.metric("Accuracy", f"{percentage:.1f}%")

# Main Content - Only show if API key is entered
if st.session_state.api_key_entered:
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Practice Quiz", "📚 Study Concepts", "📝 Practice Test", "📈 Progress"])
    
    # Tab 1: Practice Quiz
    with tab1:
        st.header("Practice Quiz")
        st.write("Test your knowledge on specific Power BI topics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_topic = st.selectbox(
                "Select Topic",
                list(PL300_TOPICS.keys()),
                key="topic_select"
            )
        
        with col2:
            selected_subtopic = st.selectbox(
                "Select Subtopic",
                PL300_TOPICS[selected_topic],
                key="subtopic_select"
            )
        
        if st.button("🎲 Generate Question", use_container_width=True, key="gen_question"):
            with st.spinner("Generating question..."):
                question_data = generate_quiz_question(selected_topic, selected_subtopic, st.session_state.api_key)
                
                if question_data:
                    st.session_state.current_question = question_data
                    st.session_state.question_answered = False
        
        if "current_question" in st.session_state:
            q = st.session_state.current_question
            
            st.subheader(q["question"])
            
            options = q["options"]
            option_keys = ["A", "B", "C", "D"]
            
            user_answer = st.radio(
                "Select your answer:",
                options,
                key=f"answer_{datetime.now().timestamp()}"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✓ Submit Answer", use_container_width=True):
                    selected_index = options.index(user_answer)
                    
                    # Handle both single letter (A, B, C, D) and full option text
                    correct_answer = q["correct_answer"].strip()
                    if len(correct_answer) == 1 and correct_answer in "ABCD":
                        correct_index = ord(correct_answer) - ord("A")
                    else:
                        # If it's the full option text, find which option it matches
                        correct_index = None
                        for opt_idx, option in enumerate(options):
                            if option.lower() in correct_answer.lower() or correct_answer.lower() in option.lower():
                                correct_index = opt_idx
                                break
                        if correct_index is None:
                            correct_index = 0
                    
                    if selected_index == correct_index:
                        st.success("✓ Correct!")
                        st.session_state.quiz_score += 1
                    else:
                        st.error("✗ Incorrect")
                    
                    st.session_state.quiz_total += 1
                    st.session_state.question_answered = True
            
            with col2:
                if st.button("💡 Show Explanation", use_container_width=True):
                    st.info(q["explanation"])
    
    # Tab 2: Study Concepts
    with tab2:
        st.header("Study Concepts")
        st.write("Deep dive into Power BI concepts with AI-generated explanations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            study_topic = st.selectbox(
                "Select Topic",
                list(PL300_TOPICS.keys()),
                key="study_topic_select"
            )
        
        with col2:
            study_subtopic = st.selectbox(
                "Select Subtopic",
                PL300_TOPICS[study_topic],
                key="study_subtopic_select"
            )
        
        if st.button("📖 Generate Explanation", use_container_width=True):
            with st.spinner("Generating explanation..."):
                explanation = generate_concept_explanation(study_topic, study_subtopic, st.session_state.api_key)
                
                if explanation:
                    st.session_state.current_explanation = explanation
                    st.session_state.study_history.append({
                        "topic": study_topic,
                        "subtopic": study_subtopic,
                        "timestamp": datetime.now().isoformat()
                    })
        
        if "current_explanation" in st.session_state:
            st.markdown(st.session_state.current_explanation)
            st.divider()
            
            if st.checkbox("Save to Study Notes"):
                st.success("✓ Saved!")
    
    # Tab 3: Practice Test
    with tab3:
        st.header("Practice Test")
        st.write("Full practice tests covering complete topics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_topic = st.selectbox(
                "Select Topic for Test",
                list(PL300_TOPICS.keys()),
                key="test_topic_select"
            )
        
        with col2:
            num_questions = st.slider("Number of Questions", 3, 15, 5)
        
        with col3:
            if st.button("🚀 Start Test", use_container_width=True):
                with st.spinner("Generating practice test..."):
                    questions = generate_practice_test(test_topic, st.session_state.api_key, num_questions)
                    
                    if questions:
                        st.session_state.practice_test_questions = questions
                        st.session_state.test_started = True
                        st.session_state.test_answers = [None] * len(questions)
                        st.session_state.current_test_index = 0
        
        if st.session_state.get("test_started", False):
            questions = st.session_state.practice_test_questions
            current_idx = st.session_state.current_test_index
            
            # Progress bar
            progress = (current_idx + 1) / len(questions)
            st.progress(progress)
            st.caption(f"Question {current_idx + 1} of {len(questions)}")
            
            if current_idx < len(questions):
                q = questions[current_idx]
                st.subheader(q["question"])
                
                user_answer = st.radio(
                    "Select your answer:",
                    q["options"],
                    key=f"test_answer_{current_idx}"
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if current_idx > 0:
                        if st.button("⬅ Previous"):
                            st.session_state.current_test_index -= 1
                            st.rerun()
                
                with col2:
                    selected_idx = q["options"].index(user_answer)
                    st.session_state.test_answers[current_idx] = selected_idx
                    
                    if current_idx < len(questions) - 1:
                        if st.button("Next ➡"):
                            st.session_state.current_test_index += 1
                            st.rerun()
                
                with col3:
                    if current_idx == len(questions) - 1:
                        if st.button("✓ Submit Test", use_container_width=True):
                            st.session_state.test_completed = True
            
            # Show results if test completed
            if st.session_state.get("test_completed", False):
                st.divider()
                st.subheader("📊 Test Results")
                
                correct = 0
                for idx, question in enumerate(questions):
                    # Handle both single letter (A, B, C, D) and full option text
                    correct_answer = question["correct_answer"].strip()
                    if len(correct_answer) == 1 and correct_answer in "ABCD":
                        correct_idx = ord(correct_answer) - ord("A")
                    else:
                        # If it's the full option text, find which option it matches
                        correct_idx = None
                        for opt_idx, option in enumerate(question["options"]):
                            if option.lower() in correct_answer.lower() or correct_answer.lower() in option.lower():
                                correct_idx = opt_idx
                                break
                        if correct_idx is None:
                            correct_idx = 0  # Default to first option if no match
                    
                    if st.session_state.test_answers[idx] == correct_idx:
                        correct += 1
                
                score_percentage = (correct / len(questions)) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", f"{correct}/{len(questions)}")
                with col2:
                    st.metric("Percentage", f"{score_percentage:.1f}%")
                with col3:
                    if score_percentage >= 80:
                        st.metric("Status", "✓ Pass")
                    else:
                        st.metric("Status", "✗ Needs Work")
                
                st.divider()
                st.subheader("Review Answers")
                
                for idx, question in enumerate(questions):
                    # Handle both single letter (A, B, C, D) and full option text
                    correct_answer = question["correct_answer"].strip()
                    if len(correct_answer) == 1 and correct_answer in "ABCD":
                        correct_idx = ord(correct_answer) - ord("A")
                    else:
                        # If it's the full option text, find which option it matches
                        correct_idx = None
                        for opt_idx, option in enumerate(question["options"]):
                            if option.lower() in correct_answer.lower() or correct_answer.lower() in option.lower():
                                correct_idx = opt_idx
                                break
                        if correct_idx is None:
                            correct_idx = 0
                    
                    user_idx = st.session_state.test_answers[idx]
                    
                    with st.expander(f"Question {idx + 1}: {question['question'][:50]}..."):
                        st.write(f"**Your Answer:** {question['options'][user_idx]}")
                        st.write(f"**Correct Answer:** {question['options'][correct_idx]}")
                        st.info(question['explanation'])
                
                if st.button("🔄 Take Another Test"):
                    st.session_state.test_started = False
                    st.session_state.test_completed = False
                    st.rerun()
    
    # Tab 4: Progress
    with tab4:
        st.header("Your Progress")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Questions Attempted", st.session_state.quiz_total)
        
        with col2:
            st.metric("Correct Answers", st.session_state.quiz_score)
        
        with col3:
            if st.session_state.quiz_total > 0:
                accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
                st.metric("Overall Accuracy", f"{accuracy:.1f}%")
            else:
                st.metric("Overall Accuracy", "N/A")
        
        st.divider()
        
        st.subheader("Study History")
        
        if st.session_state.study_history:
            study_df = st.session_state.study_history
            for item in study_df[-10:]:  # Show last 10
                st.write(f"📚 {item['topic']} → {item['subtopic']}")
        else:
            st.info("No study history yet. Start studying to track your progress!")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Progress", use_container_width=True):
                st.session_state.quiz_score = 0
                st.session_state.quiz_total = 0
                st.session_state.study_history = []
                st.success("Progress reset!")
                st.rerun()
        
        with col2:
            if st.button("Export Progress", use_container_width=True):
                progress_data = {
                    "total_questions": st.session_state.quiz_total,
                    "correct_answers": st.session_state.quiz_score,
                    "accuracy": (st.session_state.quiz_score / max(st.session_state.quiz_total, 1)) * 100,
                    "study_sessions": len(st.session_state.study_history),
                    "exported_at": datetime.now().isoformat()
                }
                st.json(progress_data)

else:
    st.info("👈 Enter your OpenAI API key in the sidebar to get started!")
    st.markdown("""
    ## About This App
    
    This Power BI PL 300 Bootcamp Prep app helps you prepare for the Microsoft Power BI Data Analyst certification exam.
    
    ### Features
    - **Practice Quiz**: Generate unlimited practice questions on specific topics
    - **Study Concepts**: Get AI-generated explanations of key Power BI concepts
    - **Practice Tests**: Take full-length practice tests to assess your readiness
    - **Progress Tracking**: Monitor your performance and improvement over time
    
    ### Getting Started
    1. Get your [OpenAI API key](https://platform.openai.com/account/api-keys)
    2. Enter it in the sidebar
    3. Start practicing!
    
    ### Topics Covered
    - Data Modeling
    - Power Query
    - DAX (Data Analysis Expressions)
    - Visualizations
    - Data Analysis & Optimization
    """)
