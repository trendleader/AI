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
if "full_exam_started" not in st.session_state:
    st.session_state.full_exam_started = False
if "full_exam_completed" not in st.session_state:
    st.session_state.full_exam_completed = False
if "full_exam_questions" not in st.session_state:
    st.session_state.full_exam_questions = []
if "full_exam_answers" not in st.session_state:
    st.session_state.full_exam_answers = []
if "full_exam_index" not in st.session_state:
    st.session_state.full_exam_index = 0
if "full_exam_timed" not in st.session_state:
    st.session_state.full_exam_timed = False
if "exam_start_time" not in st.session_state:
    st.session_state.exam_start_time = None

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

def generate_full_exam(api_key: str, num_questions: int = 65) -> list:
    """Generate a full exam with questions distributed across all topics"""
    try:
        client = OpenAI(api_key=api_key)
        questions_per_topic = num_questions // len(PL300_TOPICS)
        all_questions = []
        
        for topic in PL300_TOPICS.keys():
            prompt = f"""Generate {questions_per_topic} multiple-choice questions for a Power BI PL 300 certification exam focused on the topic: {topic}

Format your response as a JSON array where each element has this structure:
{{
    "question": "The question text",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "A",
    "explanation": "A brief explanation",
    "topic": "{topic}"
}}

IMPORTANT: The "correct_answer" field MUST be a single character: A, B, C, or D (not the full option text).

Only return the JSON array, no other text."""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            topic_questions = json.loads(response_text)
            all_questions.extend(topic_questions)
        
        return all_questions
    except Exception as e:
        st.error(f"Error generating exam: {str(e)}")
        return []
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Practice Quiz", "📚 Study Concepts", "📝 Practice Test", "🏆 Full Exam", "📈 Progress"])
    
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
                
                # Determine status and emoji
                if score_percentage >= 90:
                    status = "🌟 Excellent!"
                elif score_percentage >= 80:
                    status = "✓ Pass"
                elif score_percentage >= 70:
                    status = "⚠ Borderline"
                else:
                    status = "✗ Needs Work"
                
                # Display main score prominently
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.metric("Score", f"{correct}/{len(questions)}")
                
                with col2:
                    # Large percentage display with custom styling
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background-color: rgba(70, 130, 180, 0.2); border-radius: 10px; border: 2px solid rgba(70, 130, 180, 0.5);">
                        <h1 style="margin: 0; font-size: 72px; font-weight: bold;">{score_percentage:.0f}%</h1>
                        <p style="margin: 10px 0 0 0; font-size: 18px; font-weight: 500;">{status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.metric("Status", status)
                
                # Additional stats
                st.divider()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Correct", correct)
                
                with col2:
                    st.metric("Incorrect", len(questions) - correct)
                
                with col3:
                    accuracy_text = f"{correct}/{len(questions)}"
                    st.metric("Results", accuracy_text)
                
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
    
    # Tab 4: Full Exam Simulation
    with tab4:
        st.header("🏆 Full PL-300 Exam Simulation")
        st.write("Complete 65-question simulated exam matching the actual PL-300 certification test")
        
        if not st.session_state.get("full_exam_started", False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **Exam Details:**
                - 65 Questions
                - All PL-300 Topics
                - ~120 Minutes (Recommended)
                - Passing Score: 70%
                """)
            
            with col2:
                st.warning("""
                **Important:**
                - You can navigate back and forth
                - Review all answers before submitting
                - Each question counts equally
                - Detailed results after completion
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Start Untimed Exam", use_container_width=True, key="start_untimed"):
                    with st.spinner("Generating 65-question exam..."):
                        questions = generate_full_exam(st.session_state.api_key, 65)
                        
                        if questions:
                            st.session_state.full_exam_questions = questions
                            st.session_state.full_exam_answers = [None] * len(questions)
                            st.session_state.full_exam_index = 0
                            st.session_state.full_exam_started = True
                            st.session_state.full_exam_completed = False
                            st.session_state.full_exam_timed = False
                            st.rerun()
            
            with col2:
                if st.button("⏱ Start Timed Exam (120 min)", use_container_width=True, key="start_timed"):
                    with st.spinner("Generating 65-question exam..."):
                        questions = generate_full_exam(st.session_state.api_key, 65)
                        
                        if questions:
                            st.session_state.full_exam_questions = questions
                            st.session_state.full_exam_answers = [None] * len(questions)
                            st.session_state.full_exam_index = 0
                            st.session_state.full_exam_started = True
                            st.session_state.full_exam_completed = False
                            st.session_state.full_exam_timed = True
                            st.session_state.exam_start_time = datetime.now()
                            st.rerun()
        
        elif st.session_state.get("full_exam_started", False) and not st.session_state.get("full_exam_completed", False):
            questions = st.session_state.full_exam_questions
            current_idx = st.session_state.full_exam_index
            
            # Calculate time remaining if timed
            if st.session_state.full_exam_timed:
                elapsed = (datetime.now() - st.session_state.exam_start_time).total_seconds()
                remaining_seconds = max(0, 120 * 60 - elapsed)
                remaining_minutes = int(remaining_seconds // 60)
                remaining_secs = int(remaining_seconds % 60)
                
                if remaining_seconds <= 0:
                    st.session_state.full_exam_completed = True
                    st.error("⏰ Time's up! Exam submitted automatically.")
                    st.rerun()
                
                # Time indicator
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if remaining_minutes < 5:
                        st.error(f"⏱ Time Remaining: {remaining_minutes}:{remaining_secs:02d}")
                    elif remaining_minutes < 15:
                        st.warning(f"⏱ Time Remaining: {remaining_minutes}:{remaining_secs:02d}")
                    else:
                        st.info(f"⏱ Time Remaining: {remaining_minutes}:{remaining_secs:02d}")
            
            # Progress bar
            progress = (current_idx + 1) / len(questions)
            st.progress(progress)
            st.caption(f"Question {current_idx + 1} of {len(questions)}")
            
            # Question display
            q = questions[current_idx]
            st.subheader(q["question"])
            st.caption(f"Topic: {q.get('topic', 'General')}")
            
            # Answer options
            user_answer = st.radio(
                "Select your answer:",
                q["options"],
                key=f"exam_answer_{current_idx}"
            )
            
            # Navigation
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if current_idx > 0:
                    if st.button("⬅ Previous", use_container_width=True):
                        st.session_state.full_exam_index -= 1
                        st.rerun()
            
            with col2:
                selected_idx = q["options"].index(user_answer)
                st.session_state.full_exam_answers[current_idx] = selected_idx
                st.caption(f"Question {current_idx + 1} saved")
            
            with col3:
                if current_idx < len(questions) - 1:
                    if st.button("Next ➡", use_container_width=True):
                        st.session_state.full_exam_index += 1
                        st.rerun()
            
            with col4:
                if current_idx == len(questions) - 1:
                    if st.button("✓ Submit Exam", use_container_width=True, key="submit_exam"):
                        st.session_state.full_exam_completed = True
                        st.rerun()
            
            # Question overview
            st.divider()
            st.subheader("Question Overview")
            
            with st.expander("View all questions status"):
                cols = st.columns(13)
                for idx in range(len(questions)):
                    with cols[idx % 13]:
                        if st.session_state.full_exam_answers[idx] is not None:
                            st.button(f"{idx + 1}", key=f"nav_q_{idx}", disabled=False)
                        else:
                            st.button(f"{idx + 1}", key=f"nav_q_{idx}_empty", disabled=True)
        
        elif st.session_state.get("full_exam_completed", False):
            questions = st.session_state.full_exam_questions
            
            # Calculate results by topic
            correct_by_topic = {}
            total_by_topic = {}
            
            for topic in PL300_TOPICS.keys():
                correct_by_topic[topic] = 0
                total_by_topic[topic] = 0
            
            correct_total = 0
            
            for idx, question in enumerate(questions):
                topic = question.get("topic", "General")
                total_by_topic[topic] = total_by_topic.get(topic, 0) + 1
                
                # Get correct answer
                correct_answer = question["correct_answer"].strip()
                if len(correct_answer) == 1 and correct_answer in "ABCD":
                    correct_idx = ord(correct_answer) - ord("A")
                else:
                    correct_idx = None
                    for opt_idx, option in enumerate(question["options"]):
                        if option.lower() in correct_answer.lower() or correct_answer.lower() in option.lower():
                            correct_idx = opt_idx
                            break
                    if correct_idx is None:
                        correct_idx = 0
                
                if st.session_state.full_exam_answers[idx] == correct_idx:
                    correct_by_topic[topic] = correct_by_topic.get(topic, 0) + 1
                    correct_total += 1
            
            score_percentage = (correct_total / len(questions)) * 100
            
            # Determine overall status
            if score_percentage >= 70:
                overall_status = "🎉 PASSED"
                overall_color = "green"
            else:
                overall_status = "❌ FAILED"
                overall_color = "red"
            
            # Main results
            st.divider()
            st.subheader("📊 EXAM RESULTS")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.metric("Final Score", f"{correct_total}/65")
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 25px; background-color: rgba(70, 130, 180, 0.2); border-radius: 10px; border: 3px solid rgba(70, 130, 180, 0.5);">
                    <h1 style="margin: 0; font-size: 80px; font-weight: bold;">{score_percentage:.1f}%</h1>
                    <p style="margin: 15px 0 0 0; font-size: 22px; font-weight: bold;">{overall_status}</p>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">Passing Score: 70%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if score_percentage >= 70:
                    st.success("✓ PASS")
                else:
                    st.error("✗ FAIL")
            
            # Detailed breakdown
            st.divider()
            st.subheader("📈 Performance by Topic")
            
            topic_cols = st.columns(len(PL300_TOPICS))
            
            for col, topic in enumerate(PL300_TOPICS.keys()):
                with topic_cols[col]:
                    correct = correct_by_topic.get(topic, 0)
                    total = total_by_topic.get(topic, 0)
                    if total > 0:
                        topic_percentage = (correct / total) * 100
                        st.metric(topic, f"{correct}/{total}", f"{topic_percentage:.0f}%")
                    else:
                        st.metric(topic, "0/0", "N/A")
            
            # Question review
            st.divider()
            st.subheader("Review Your Answers")
            
            for idx, question in enumerate(questions):
                correct_answer = question["correct_answer"].strip()
                if len(correct_answer) == 1 and correct_answer in "ABCD":
                    correct_idx = ord(correct_answer) - ord("A")
                else:
                    correct_idx = None
                    for opt_idx, option in enumerate(question["options"]):
                        if option.lower() in correct_answer.lower() or correct_answer.lower() in option.lower():
                            correct_idx = opt_idx
                            break
                    if correct_idx is None:
                        correct_idx = 0
                
                user_idx = st.session_state.full_exam_answers[idx]
                is_correct = user_idx == correct_idx
                
                status_icon = "✓" if is_correct else "✗"
                
                with st.expander(f"{status_icon} Question {idx + 1}: {question['question'][:60]}..."):
                    st.write(f"**Topic:** {question.get('topic', 'General')}")
                    st.write(f"**Your Answer:** {question['options'][user_idx]}")
                    st.write(f"**Correct Answer:** {question['options'][correct_idx]}")
                    
                    if is_correct:
                        st.success("✓ Correct")
                    else:
                        st.error("✗ Incorrect")
                    
                    st.info(f"**Explanation:** {question['explanation']}")
            
            # Final buttons
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Retake Full Exam", use_container_width=True):
                    st.session_state.full_exam_started = False
                    st.session_state.full_exam_completed = False
                    st.rerun()
            
            with col2:
                if st.button("📋 Export Results", use_container_width=True):
                    export_data = {
                        "exam_type": "Full PL-300 Simulation",
                        "total_questions": 65,
                        "correct_answers": correct_total,
                        "score_percentage": round(score_percentage, 1),
                        "status": "PASSED" if score_percentage >= 70 else "FAILED",
                        "passing_score": 70,
                        "timestamp": datetime.now().isoformat(),
                        "topic_breakdown": {
                            topic: {
                                "correct": correct_by_topic.get(topic, 0),
                                "total": total_by_topic.get(topic, 0),
                                "percentage": round((correct_by_topic.get(topic, 0) / max(total_by_topic.get(topic, 1), 1)) * 100, 1)
                            }
                            for topic in PL300_TOPICS.keys()
                        }
                    }
                    st.json(export_data)
    
    # Tab 5: Progress
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
