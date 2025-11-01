"""
Python Developer Interview Prep App
Uses Streamlit and OpenAI API for interactive interview preparation
"""

import streamlit as st
from openai import OpenAI
import os

# Page configuration
st.set_page_config(
    page_title="Python Dev Interview Prep",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐍 Python Developer Interview Prep")
st.markdown("Master Python interviews with interactive challenges and real-time AI feedback")

# ============================================================================
# SIDEBAR - API KEY SETUP
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Your API key is never stored or logged"
    )
    
    st.markdown("---")
    
    difficulty = st.radio(
        "Select Difficulty Level",
        ["Beginner", "Intermediate", "Advanced"],
        help="This affects the complexity of questions and challenges"
    )
    
    st.markdown("---")
    st.markdown(
        "**⚠️ Cost Notice:** You will be charged for OpenAI API usage based on tokens consumed."
    )

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "client" not in st.session_state:
    st.session_state.client = None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "current_challenge" not in st.session_state:
    st.session_state.current_challenge = None

# ============================================================================
# VALIDATE API KEY
# ============================================================================
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to get started.")
    st.stop()

# Initialize OpenAI client with explicit configuration
if st.session_state.client is None:
    try:
        st.session_state.client = OpenAI(
            api_key=api_key,
            organization=None
        )
        # Test the connection
        st.session_state.client.models.list()
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
        st.info("Make sure you have the latest OpenAI library installed: `pip install --upgrade openai`")
        st.stop()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def call_openai(messages: list, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    """Call OpenAI API and return response"""
    try:
        response = st.session_state.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1500,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ OpenAI API Error: {str(e)}")
        st.info("Troubleshooting: Check your API key validity and account balance at https://platform.openai.com/account/billing/overview")
        return None

def generate_interview_question(difficulty: str) -> str:
    """Generate a Python interview question based on difficulty"""
    prompt = f"""Generate a single Python interview question at the {difficulty} level.
    
    Requirements:
    - Focus on practical Python concepts
    - Be specific and clear
    - Expect 2-4 sentence answer
    - Don't include the answer
    
    Format: Just the question, nothing else."""
    
    response = call_openai([{"role": "user", "content": prompt}])
    return response

def evaluate_answer(question: str, answer: str, difficulty: str) -> str:
    """Evaluate a user's interview answer"""
    prompt = f"""You are an expert Python interviewer. Evaluate this answer:

Question: {question}

User's Answer: {answer}

Difficulty Level: {difficulty}

Provide feedback on:
1. Correctness (is it accurate?)
2. Completeness (did they cover key points?)
3. Code examples (if applicable, are they well-written?)
4. Areas for improvement

Be constructive and specific. Keep feedback to 3-4 sentences."""
    
    response = call_openai([{"role": "user", "content": prompt}])
    return response

def generate_coding_challenge(difficulty: str) -> dict:
    """Generate a coding challenge"""
    prompt = f"""Create a Python coding challenge at the {difficulty} level.

Format your response as:
TITLE: [Challenge Title]
DESCRIPTION: [What the user needs to implement]
CONSTRAINTS: [Any constraints or requirements]
EXAMPLE_INPUT: [Example input/usage]
EXAMPLE_OUTPUT: [Expected output]"""
    
    response = call_openai([{"role": "user", "content": prompt}])
    return {"challenge": response}

def evaluate_code(challenge: str, code: str, difficulty: str) -> str:
    """Evaluate submitted code"""
    prompt = f"""You are a Python code reviewer. Evaluate this code solution:

Challenge: {challenge}

Submitted Code:
```python
{code}
```

Difficulty Level: {difficulty}

Evaluate on:
1. Correctness - does it solve the challenge?
2. Efficiency - is it optimal?
3. Code quality - readability, best practices
4. Edge cases - does it handle them?

Provide actionable feedback in 3-4 sentences."""
    
    response = call_openai([{"role": "user", "content": prompt}])
    return response

# ============================================================================
# MAIN APP CONTENT
# ============================================================================

# Mode selection
col1, col2 = st.columns(2)

with col1:
    mode = st.radio(
        "Select Mode",
        ["📝 Interview Questions", "💻 Coding Challenges"],
        horizontal=False
    )

# ============================================================================
# MODE 1: INTERVIEW QUESTIONS
# ============================================================================
if mode == "📝 Interview Questions":
    st.header("Interview Question Practice")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🎲 Generate New Question", use_container_width=True):
            with st.spinner("Generating question..."):
                question = generate_interview_question(difficulty)
                st.session_state.current_question = question
                st.session_state.conversation_history = []
    
    # Display current question
    if "current_question" in st.session_state:
        st.markdown("### Question")
        st.info(st.session_state.current_question)
        
        # User answer input
        user_answer = st.text_area(
            "Your Answer:",
            placeholder="Type your answer here...",
            height=150,
            key="interview_answer"
        )
        
        if st.button("📤 Submit Answer", use_container_width=True):
            if user_answer.strip():
                with st.spinner("Evaluating your answer..."):
                    feedback = evaluate_answer(
                        st.session_state.current_question,
                        user_answer,
                        difficulty
                    )
                    
                    if feedback:
                        st.markdown("### 📊 Feedback")
                        st.success(feedback)
            else:
                st.warning("Please enter an answer before submitting.")
    else:
        st.info("Click 'Generate New Question' to start practicing!")

# ============================================================================
# MODE 2: CODING CHALLENGES
# ============================================================================
elif mode == "💻 Coding Challenges":
    st.header("Interactive Coding Challenges")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🎲 Get New Challenge", use_container_width=True):
            with st.spinner("Generating challenge..."):
                challenge_data = generate_coding_challenge(difficulty)
                st.session_state.current_challenge = challenge_data["challenge"]
    
    # Display current challenge
    if st.session_state.current_challenge:
        st.markdown("### Challenge")
        st.info(st.session_state.current_challenge)
        
        # Code input
        code_input = st.text_area(
            "Your Solution:",
            placeholder="Write your Python code here...",
            height=250,
            key="code_solution"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Submit Solution", use_container_width=True):
                if code_input.strip():
                    with st.spinner("Evaluating your code..."):
                        feedback = evaluate_code(
                            st.session_state.current_challenge,
                            code_input,
                            difficulty
                        )
                        
                        if feedback:
                            st.markdown("### 📊 Code Review")
                            st.success(feedback)
                else:
                    st.warning("Please enter code before submitting.")
        
        with col2:
            if st.button("💡 Get a Hint", use_container_width=True):
                with st.spinner("Generating hint..."):
                    hint_prompt = f"Provide a brief hint (1-2 sentences) for this Python challenge:\n\n{st.session_state.current_challenge}\n\nDon't give away the solution, just point them in the right direction."
                    hint = call_openai([{"role": "user", "content": hint_prompt}])
                    if hint:
                        st.info(f"💭 Hint: {hint}")
    else:
        st.info("Click 'Get New Challenge' to start coding!")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "💡 **Tips:** Start with Beginner mode to warm up. "
    "Take your time to think through problems. "
    "Review the feedback carefully to identify learning gaps."
)
