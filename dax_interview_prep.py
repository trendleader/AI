import streamlit as st
from openai import OpenAI
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="DAX Interview Prep",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .challenge-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .question-card {
        border-left: 4px solid #0066cc;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f0f7ff;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "client" not in st.session_state:
    st.session_state.client = None
if "progress" not in st.session_state:
    st.session_state.progress = {"questions_completed": 0, "challenges_completed": 0}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# DAX Interview Questions Database
DAX_QUESTIONS = [
    {
        "id": 1,
        "category": "Basics",
        "question": "What is DAX and what are its primary use cases?",
        "difficulty": "Easy"
    },
    {
        "id": 2,
        "category": "Basics",
        "question": "Explain the difference between implicit and explicit measures in DAX.",
        "difficulty": "Easy"
    },
    {
        "id": 3,
        "category": "Functions",
        "question": "What is the difference between SUMX and SUM functions, and when would you use each?",
        "difficulty": "Medium"
    },
    {
        "id": 4,
        "category": "Functions",
        "question": "Explain the concept of filter context and row context in DAX.",
        "difficulty": "Medium"
    },
    {
        "id": 5,
        "category": "Advanced",
        "question": "How would you implement a dynamic ranking measure in DAX that updates based on filters?",
        "difficulty": "Hard"
    },
    {
        "id": 6,
        "category": "Advanced",
        "question": "Describe how you would use CALCULATE to override filter context and provide an example.",
        "difficulty": "Hard"
    },
    {
        "id": 7,
        "category": "Performance",
        "question": "What are some best practices for optimizing DAX query performance?",
        "difficulty": "Hard"
    },
    {
        "id": 8,
        "category": "Relationships",
        "question": "How do relationships affect DAX calculations, and when might you use TREATAS or CROSSJOIN?",
        "difficulty": "Medium"
    }
]

# DAX Coding Challenges
DAX_CHALLENGES = [
    {
        "id": 1,
        "title": "Basic Total Sales Measure",
        "difficulty": "Easy",
        "description": "Create a measure that calculates the total sales from a Sales table. The table has columns: 'SalesAmount' and 'Date'.",
        "starter_code": "Total Sales = ",
        "solution_hint": "Use SUMX or SUM function on SalesAmount column",
        "test_scenario": "Should sum all values in SalesAmount column"
    },
    {
        "id": 2,
        "title": "Year-to-Date (YTD) Calculation",
        "difficulty": "Medium",
        "description": "Create a YTD sales measure that calculates sales from the beginning of the year to the current date.",
        "starter_code": "YTD Sales = ",
        "solution_hint": "Use CALCULATE with DATESYTD function to filter dates",
        "test_scenario": "Should return cumulative sales from Jan 1 to current date in context"
    },
    {
        "id": 3,
        "title": "Previous Year Comparison",
        "difficulty": "Medium",
        "description": "Create a measure that returns the sales amount from the same period in the previous year.",
        "starter_code": "Prior Year Sales = ",
        "solution_hint": "Use CALCULATE with DATEADD to shift dates by -1 year",
        "test_scenario": "Should return sales from the equivalent period last year"
    },
    {
        "id": 4,
        "title": "Percentage of Total",
        "difficulty": "Medium",
        "description": "Create a measure that calculates what percentage of total sales each product represents.",
        "starter_code": "% of Total = ",
        "solution_hint": "Divide current sales by total sales using DIVIDE function",
        "test_scenario": "Should return a percentage value between 0 and 1"
    },
    {
        "id": 5,
        "title": "Dynamic Ranking",
        "difficulty": "Hard",
        "description": "Create a measure that ranks products by sales amount dynamically based on applied filters.",
        "starter_code": "Product Rank = ",
        "solution_hint": "Use RANKX function with appropriate scope and filter context",
        "test_scenario": "Should return rank number that updates based on product selection"
    },
    {
        "id": 6,
        "title": "Moving Average",
        "difficulty": "Hard",
        "description": "Create a measure that calculates a 3-month moving average of sales.",
        "starter_code": "3-Month Moving Avg = ",
        "solution_hint": "Use CALCULATE with DATESBETWEEN to create rolling window",
        "test_scenario": "Should return average of current and previous 2 months"
    }
]

def setup_openai_client(api_key):
    """Initialize OpenAI client with provided API key"""
    try:
        client = OpenAI(api_key=api_key)
        st.session_state.client = client
        st.session_state.api_key = api_key
        return True
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return False

def get_ai_feedback(prompt, system_message="You are an expert DAX instructor helping prepare candidates for interviews."):
    """Get feedback from OpenAI"""
    if not st.session_state.client:
        return "Please configure your OpenAI API key first."
    
    try:
        response = st.session_state.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting feedback: {str(e)}"

def render_sidebar():
    """Render the sidebar with API key input and navigation"""
    st.sidebar.title("🎯 DAX Interview Prep")
    st.sidebar.divider()
    
    # API Key Input
    st.sidebar.subheader("OpenAI Configuration")
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="Your API key is never stored or transmitted beyond this session"
    )
    
    if api_key and api_key != st.session_state.api_key:
        if setup_openai_client(api_key):
            st.sidebar.success("✅ API Key configured successfully!")
        else:
            st.sidebar.error("❌ Failed to configure API key")
    
    st.sidebar.divider()
    
    # Progress Tracker
    st.sidebar.subheader("📈 Your Progress")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Questions", st.session_state.progress["questions_completed"])
    with col2:
        st.metric("Challenges", st.session_state.progress["challenges_completed"])
    
    st.sidebar.divider()
    
    # Navigation
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio(
        "Select a section:",
        ["Home", "Interview Questions", "Coding Challenges", "Q&A Chat"]
    )
    
    return page

def render_home():
    """Render the home page"""
    st.title("📊 DAX Interview Preparation Platform")
    
    st.markdown("""
    Welcome to your personalized DAX interview prep tool! This platform is designed to help you master
    DAX concepts and prepare for technical interviews.
    
    **Features:**
    - 📝 **Interview Questions**: Practice answering common DAX questions with AI-powered feedback
    - 💻 **Coding Challenges**: Write and test DAX code with real-world scenarios
    - 💬 **Q&A Chat**: Ask specific questions about DAX concepts and get detailed explanations
    - 📈 **Progress Tracking**: Monitor your preparation journey
    
    **Getting Started:**
    1. Enter your OpenAI API key in the sidebar (required for AI feedback)
    2. Choose a section from the navigation menu
    3. Start practicing!
    
    **Tips for Success:**
    - Start with easy questions and progress to harder ones
    - Use the AI feedback to understand where to improve
    - Review DAX documentation alongside your practice
    - Focus on understanding concepts, not just memorizing
    """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📚 Learn")
        st.write("Master DAX fundamentals through guided interview questions")
    
    with col2:
        st.subheader("✍️ Practice")
        st.write("Write real DAX code and receive instant feedback")
    
    with col3:
        st.subheader("🚀 Improve")
        st.write("Track your progress and identify areas for growth")

def render_interview_questions():
    """Render the interview questions section"""
    st.title("📝 Interview Questions")
    
    if not st.session_state.client:
        st.warning("⚠️ Please configure your OpenAI API key in the sidebar to use this feature.")
        return
    
    # Filter by difficulty
    col1, col2 = st.columns(2)
    with col1:
        difficulty_filter = st.selectbox("Filter by difficulty:", ["All", "Easy", "Medium", "Hard"])
    with col2:
        category_filter = st.selectbox("Filter by category:", ["All"] + list(set([q["category"] for q in DAX_QUESTIONS])))
    
    # Apply filters
    filtered_questions = DAX_QUESTIONS
    if difficulty_filter != "All":
        filtered_questions = [q for q in filtered_questions if q["difficulty"] == difficulty_filter]
    if category_filter != "All":
        filtered_questions = [q for q in filtered_questions if q["category"] == category_filter]
    
    st.divider()
    
    for q in filtered_questions:
        with st.container():
            st.markdown(f"### Q{q['id']}: {q['question']}")
            st.caption(f"Category: {q['category']} | Difficulty: {q['difficulty']}")
            
            # Tabs for answer input and AI feedback
            tab1, tab2 = st.tabs(["Your Answer", "AI Feedback"])
            
            with tab1:
                user_answer = st.text_area(
                    "Your answer:",
                    key=f"q_{q['id']}_answer",
                    height=150,
                    placeholder="Type your answer here..."
                )
                
                if st.button(f"Get Feedback on Q{q['id']}", key=f"q_{q['id']}_btn"):
                    if user_answer:
                        with st.spinner("Getting AI feedback..."):
                            feedback_prompt = f"""
                            Interview Question: {q['question']}
                            
                            Candidate's Answer: {user_answer}
                            
                            Please provide:
                            1. Assessment of the answer (what was good, what was missing)
                            2. Key points they should have mentioned
                            3. Suggestions for improvement
                            4. A score out of 10
                            """
                            feedback = get_ai_feedback(feedback_prompt)
                            st.session_state.progress["questions_completed"] += 1
                            st.rerun()
                    else:
                        st.warning("Please enter an answer first.")
            
            with tab2:
                answer_key = f"q_{q['id']}_answer"
                if answer_key in st.session_state and st.session_state[answer_key]:
                    if st.button(f"Show Feedback for Q{q['id']}", key=f"q_{q['id']}_feedback_btn"):
                        with st.spinner("Generating feedback..."):
                            feedback_prompt = f"""
                            Interview Question: {q['question']}
                            
                            Candidate's Answer: {st.session_state[answer_key]}
                            
                            Please provide detailed feedback:
                            1. Strengths of the answer
                            2. Areas for improvement
                            3. Key concepts to emphasize
                            4. Rate this answer 1-10
                            """
                            feedback = get_ai_feedback(feedback_prompt)
                            st.write(feedback)
            
            st.divider()

def render_coding_challenges():
    """Render the coding challenges section"""
    st.title("💻 DAX Coding Challenges")
    
    if not st.session_state.client:
        st.warning("⚠️ Please configure your OpenAI API key in the sidebar to use this feature.")
        return
    
    # Select challenge
    challenge_titles = [f"[{c['difficulty']}] {c['title']}" for c in DAX_CHALLENGES]
    selected_idx = st.selectbox("Select a challenge:", range(len(DAX_CHALLENGES)), 
                                format_func=lambda x: challenge_titles[x])
    
    challenge = DAX_CHALLENGES[selected_idx]
    
    st.divider()
    
    # Challenge description
    st.subheader(challenge["title"])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Difficulty", challenge["difficulty"])
    with col2:
        st.metric("Challenge ID", challenge["id"])
    
    st.write(challenge["description"])
    st.info(f"**Hint:** {challenge['solution_hint']}")
    st.write(f"**Test Scenario:** {challenge['test_scenario']}")
    
    st.divider()
    
    # Code editor
    st.subheader("Write Your DAX Code")
    user_code = st.text_area(
        "DAX Code:",
        value=challenge["starter_code"],
        height=200,
        key=f"challenge_{challenge['id']}_code"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💡 Get Hint", key=f"challenge_{challenge['id']}_hint"):
            st.info(f"Hint: {challenge['solution_hint']}")
    
    with col2:
        if st.button("✅ Submit Solution", key=f"challenge_{challenge['id']}_submit"):
            if user_code and user_code != challenge["starter_code"]:
                with st.spinner("Evaluating your code..."):
                    evaluation_prompt = f"""
                    DAX Challenge: {challenge['title']}
                    
                    Challenge Description: {challenge['description']}
                    
                    Expected Scenario: {challenge['test_scenario']}
                    
                    User's DAX Code:
                    {user_code}
                    
                    Please evaluate this DAX code and provide:
                    1. Is the syntax correct?
                    2. Does it solve the challenge?
                    3. Are there any performance issues?
                    4. Suggestions for improvement
                    5. Rate the solution 1-10
                    6. If correct, provide a brief explanation of why it works
                    """
                    evaluation = get_ai_feedback(evaluation_prompt)
                    st.success("Submission evaluated!")
                    st.write(evaluation)
                    st.session_state.progress["challenges_completed"] += 1
            else:
                st.warning("Please write some code before submitting.")
    
    # Show reference solution
    if st.checkbox("Show Reference Solution", key=f"challenge_{challenge['id']}_ref"):
        st.subheader("Reference Solution")
        with st.spinner("Generating reference solution..."):
            ref_prompt = f"""
            Create an optimal DAX solution for this challenge:
            {challenge['description']}
            
            Provide:
            1. The DAX code
            2. Explanation of how it works
            3. Why this approach is optimal
            """
            reference = get_ai_feedback(ref_prompt)
            st.code(reference, language="dax")

def render_qa_chat():
    """Render the Q&A chat section"""
    st.title("💬 DAX Q&A Chat")
    
    if not st.session_state.client:
        st.warning("⚠️ Please configure your OpenAI API key in the sidebar to use this feature.")
        return
    
    st.write("Ask any DAX-related questions and get detailed answers from an expert DAX instructor.")
    
    # Display chat history
    st.subheader("Conversation")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Chat input
    user_input = st.text_input("Ask a question about DAX:", key="chat_input")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("Thinking..."):
            # Build the conversation for context
            messages = [{"role": "system", "content": "You are an expert DAX instructor. Provide clear, detailed explanations of DAX concepts, functions, and best practices."}]
            for msg in st.session_state.chat_history:
                messages.append(msg)
            
            try:
                response = st.session_state.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1500
                )
                assistant_message = response.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def main():
    """Main application function"""
    page = render_sidebar()
    
    if page == "Home":
        render_home()
    elif page == "Interview Questions":
        render_interview_questions()
    elif page == "Coding Challenges":
        render_coding_challenges()
    elif page == "Q&A Chat":
        render_qa_chat()

if __name__ == "__main__":
    main()
