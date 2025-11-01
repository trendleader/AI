import streamlit as st
from openai import OpenAI
import json

# Set up page configuration
st.set_page_config(
    page_title="AI/ML Interview Prep",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .feedback-box {
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .positive {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .suggestion {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .area {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False
if "client" not in st.session_state:
    st.session_state.client = None
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []
if "coding_history" not in st.session_state:
    st.session_state.coding_history = []


def initialize_openai_client(api_key):
    """Initialize OpenAI client with provided API key"""
    try:
        client = OpenAI(api_key=api_key)
        # Test the API key with a simple call
        client.models.list()
        return True, client
    except Exception as e:
        return False, str(e)


def evaluate_interview_answer(client, question, answer):
    """Use OpenAI to evaluate interview answer"""
    prompt = f"""You are an expert AI/ML interviewer. Evaluate the following answer to an interview question.

Question: {question}

Answer: {answer}

Provide your evaluation in JSON format with the following structure:
{{
    "score": <number from 1-10>,
    "strengths": "<list key strengths of the answer>",
    "areas_for_improvement": "<specific areas to improve>",
    "suggested_answer_points": "<key points that should have been covered>",
    "feedback": "<constructive feedback>"
}}

Return only valid JSON, no additional text."""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the JSON response
        response_text = response.content[0].text
        evaluation = json.loads(response_text)
        return evaluation
    except Exception as e:
        return {"error": str(e)}


def evaluate_code(client, problem, code_solution):
    """Use OpenAI to evaluate coding solution"""
    prompt = f"""You are an expert AI/ML engineer reviewing code in a technical interview.

Problem: {problem}

Submitted Code:
```python
{code_solution}
```

Evaluate this code solution and provide feedback in JSON format:
{{
    "score": <number from 1-10>,
    "correctness": "<Does the code solve the problem correctly?>",
    "code_quality": "<Comment on code quality, readability, style>",
    "efficiency": "<Any efficiency concerns? Big O analysis if relevant>",
    "best_practices": "<Are ML/data science best practices followed?>",
    "improvements": "<Specific suggestions for improvement>",
    "strengths": "<What the code does well>"
}}

Return only valid JSON, no additional text."""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1200,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response.content[0].text
        evaluation = json.loads(response_text)
        return evaluation
    except Exception as e:
        return {"error": str(e)}


# Sidebar for API key setup and navigation
st.sidebar.title("🔧 Setup & Navigation")

with st.sidebar.expander("🔑 OpenAI API Key", expanded=not st.session_state.api_key_set):
    api_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        key="api_key_input"
    )
    
    if st.button("Connect API Key", type="primary"):
        if api_key_input:
            success, result = initialize_openai_client(api_key_input)
            if success:
                st.session_state.api_key_set = True
                st.session_state.client = result
                st.success("✅ API Key connected successfully!")
                st.rerun()
            else:
                st.error(f"❌ Failed to connect: {result}")
        else:
            st.warning("Please enter an API key")

if st.session_state.api_key_set:
    st.sidebar.success("🟢 API Connected")
else:
    st.sidebar.error("🔴 Please connect your API key to continue")

# Main navigation
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Mode:",
    ["📖 Interview Questions", "💻 Coding Challenges", "📊 Progress Dashboard"]
)

# Main content area
st.title("🚀 AI/ML Interview Prep Assistant")

if not st.session_state.api_key_set:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to get started.")
    st.info("Your API key is only used for API calls and is not stored.")
else:
    
    if page == "📖 Interview Questions":
        st.header("Interview Question Practice")
        
        # Comprehensive FAANG GenAI/ML interview questions (2025)
        sample_questions = [
            # GenAI & LLM Specific (2025 Focus)
            "How would you evaluate a large language model in production? What metrics would you track?",
            "Explain how you would deploy a 20B parameter model with a 100ms latency requirement.",
            "What is Retrieval Augmented Generation (RAG) and how does it improve LLM performance?",
            "How do you detect and mitigate hallucinations in large language models?",
            "Explain the difference between fine-tuning and prompt engineering. When would you use each?",
            "What is RLHF (Reinforcement Learning from Human Feedback) and why is it important for LLMs?",
            "How would you approach building a production recommendation system using an LLM?",
            
            # Core ML Fundamentals
            "Explain the difference between supervised and unsupervised learning with examples.",
            "What is the bias-variance tradeoff and how does it affect model performance?",
            "Explain what overfitting is and techniques to prevent it.",
            "What is cross-validation and why is it important?",
            
            # FAANG Specific (ML System Design)
            "Given a dataset with purchase history, how would you build a propensity score model?",
            "How would you build a fraud detection model without labels? Walk through your approach.",
            "Describe your approach to building an image classification system for production at scale.",
            "How would you identify meaningful customer segmentation? What methods would you use?",
            
            # Advanced Topics
            "What are the key differences between discriminative and generative models?",
            "Explain how transformers work. Why are they better than RNNs for NLP?",
            "How do you handle imbalanced labels in classification? Discuss multiple approaches.",
            "What is the curse of dimensionality? How do you handle high-dimensional data?",
            
            # Data & Feature Engineering
            "Walk me through your feature engineering process. How do you identify important features?",
            "How do you handle missing data? When would you use different imputation strategies?",
            "Explain feature scaling/normalization. Why is it important and when would you skip it?",
            "How do you deal with outliers in your dataset? What's your approach?",
            
            # Model Evaluation & Deployment
            "When should you use accuracy vs precision vs recall vs F1? Give business examples.",
            "How do you approach model versioning and A/B testing in production?",
            "Explain the concept of data drift and model drift. How would you monitor for it?",
            "What considerations would you make when deploying a model to production?",
            
            # System Design Focus
            "Design a machine learning system for real-time fraud detection at scale.",
            "How would you design a recommendation system for an e-commerce platform?",
            "Design a system to detect anomalies in time-series data at a tech company.",
            
            # Ethics & Responsible AI
            "How do you detect and mitigate bias in machine learning models?",
            "What are the ethical considerations when building AI systems? Give examples.",
            "How would you explain a black-box model's predictions to business stakeholders?"
        ]
        
        st.subheader("Select or Enter a Question")
        question_choice = st.selectbox(
            "Choose a question:",
            ["Custom question"] + sample_questions,
            key="question_select"
        )
        
        if question_choice == "Custom question":
            question = st.text_area(
                "Enter your own interview question:",
                placeholder="E.g., Explain what a neural network is...",
                key="custom_question"
            )
        else:
            question = question_choice
        
        if question:
            st.markdown("---")
            answer = st.text_area(
                "Your Answer:",
                placeholder="Type your answer here...",
                height=200,
                key="interview_answer"
            )
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button("🔍 Get Feedback", type="primary"):
                    if answer.strip():
                        with st.spinner("Evaluating your answer..."):
                            evaluation = evaluate_interview_answer(
                                st.session_state.client,
                                question,
                                answer
                            )
                        
                        if "error" not in evaluation:
                            st.session_state.interview_history.append({
                                "question": question,
                                "answer": answer,
                                "evaluation": evaluation
                            })
                            st.rerun()
                        else:
                            st.error(f"Error: {evaluation['error']}")
                    else:
                        st.warning("Please provide an answer")
        
        # Display feedback if available
        if st.session_state.interview_history:
            st.subheader("Feedback")
            latest = st.session_state.interview_history[-1]
            eval_data = latest["evaluation"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Score", f"{eval_data.get('score', 'N/A')}/10")
            
            with col2:
                st.subheader("Question")
                st.write(latest["question"])
            
            with col3:
                st.subheader("Your Answer")
                st.write(latest["answer"])
            
            st.markdown("---")
            
            st.markdown('<div class="feedback-box positive">', unsafe_allow_html=True)
            st.subheader("💪 Strengths")
            st.write(eval_data.get("strengths", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feedback-box suggestion">', unsafe_allow_html=True)
            st.subheader("🎯 Areas for Improvement")
            st.write(eval_data.get("areas_for_improvement", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
            st.subheader("📚 Key Points to Cover")
            st.write(eval_data.get("suggested_answer_points", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
            st.subheader("💬 General Feedback")
            st.write(eval_data.get("feedback", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("Clear and Try Another Question"):
                st.session_state.interview_history = []
                st.rerun()
    
    elif page == "💻 Coding Challenges":
        st.header("Coding Challenge Practice")
        
        sample_challenges = {
            # ML-Specific Challenges
            "Feature Scaling": "Implement Min-Max normalization for a list of numbers. Scale values to the range [0, 1]. Input: [1, 2, 5, 10, 20], Output: [0, 0.05, 0.2, 0.45, 1.0]",
            "Accuracy Score": "Implement a function to calculate accuracy, precision, recall, and F1-score given true labels and predictions.",
            "Train-Test Split": "Implement a function to split a dataset into training and testing sets with a given ratio and optional random seed.",
            "One-Hot Encoding": "Implement one-hot encoding for categorical variables. Convert categorical features into binary vectors.",
            "Confusion Matrix": "Implement a function to compute a confusion matrix given true labels and predictions.",
            
            # Algorithm & Data Structure Challenges
            "Binary Search": "Implement binary search on a sorted array. Return the index of target, or -1 if not found.",
            "Linear Search": "Implement linear search and return the index of target element, or -1 if not found.",
            "Remove Duplicates": "Remove duplicates from a list while preserving order.",
            "Matrix Transpose": "Write a function that transposes a matrix (list of lists).",
            
            # ML Algorithms from Scratch
            "k-Means Clustering": "Implement k-means clustering algorithm. Return cluster assignments for given data points.",
            "Linear Regression": "Implement simple linear regression using gradient descent. Return slope and intercept.",
            "Cosine Similarity": "Calculate cosine similarity between two vectors.",
            "Data Standardization": "Implement z-score normalization: (x - mean) / std_dev",
        }
        
        st.subheader("Select a Coding Challenge")
        challenge_name = st.selectbox(
            "Choose a challenge:",
            list(sample_challenges.keys()),
            key="coding_challenge_select"
        )
        
        challenge_description = sample_challenges[challenge_name]
        
        st.markdown(f"**Challenge:** {challenge_name}")
        st.info(challenge_description)
        
        code_input = st.text_area(
            "Write your solution:",
            placeholder="def solve():\n    pass",
            height=250,
            key="code_input"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("📝 Review Code", type="primary"):
                if code_input.strip():
                    with st.spinner("Analyzing your code..."):
                        evaluation = evaluate_code(
                            st.session_state.client,
                            challenge_description,
                            code_input
                        )
                    
                    if "error" not in evaluation:
                        st.session_state.coding_history.append({
                            "challenge": challenge_name,
                            "code": code_input,
                            "evaluation": evaluation
                        })
                        st.rerun()
                    else:
                        st.error(f"Error: {evaluation['error']}")
                else:
                    st.warning("Please submit code to review")
        
        # Display code review feedback
        if st.session_state.coding_history:
            st.subheader("Code Review Feedback")
            latest = st.session_state.coding_history[-1]
            eval_data = latest["evaluation"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Score", f"{eval_data.get('score', 'N/A')}/10")
            
            with col2:
                st.metric("Challenge", latest["challenge"])
            
            with col3:
                st.subheader("Your Code")
                st.code(latest["code"], language="python")
            
            st.markdown("---")
            
            st.markdown('<div class="feedback-box positive">', unsafe_allow_html=True)
            st.subheader("✅ Strengths")
            st.write(eval_data.get("strengths", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feedback-box suggestion">', unsafe_allow_html=True)
            st.subheader("⚙️ Correctness")
            st.write(eval_data.get("correctness", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feedback-box suggestion">', unsafe_allow_html=True)
            st.subheader("🎨 Code Quality")
            st.write(eval_data.get("code_quality", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feedback-box suggestion">', unsafe_allow_html=True)
            st.subheader("⚡ Efficiency & Big O")
            st.write(eval_data.get("efficiency", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
            st.subheader("📋 Best Practices")
            st.write(eval_data.get("best_practices", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
            st.subheader("💡 Improvements")
            st.write(eval_data.get("improvements", ""))
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("Clear and Try Another Challenge"):
                st.session_state.coding_history = []
                st.rerun()
    
    elif page == "📊 Progress Dashboard":
        st.header("Your Progress")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Interview Questions Attempted", len(st.session_state.interview_history))
        
        with col2:
            st.metric("Coding Challenges Attempted", len(st.session_state.coding_history))
        
        if st.session_state.interview_history:
            st.subheader("Interview Question History")
            for i, attempt in enumerate(st.session_state.interview_history, 1):
                with st.expander(f"Attempt {i}: {attempt['question'][:50]}..."):
                    st.write("**Question:**", attempt["question"])
                    st.write("**Your Answer:**", attempt["answer"])
                    st.write("**Score:**", attempt["evaluation"].get("score", "N/A"), "/ 10")
        
        if st.session_state.coding_history:
            st.subheader("Coding Challenge History")
            for i, attempt in enumerate(st.session_state.coding_history, 1):
                with st.expander(f"Attempt {i}: {attempt['challenge']}"):
                    st.write("**Challenge:**", attempt["challenge"])
                    st.code(attempt["code"], language="python")
                    st.write("**Score:**", attempt["evaluation"].get("score", "N/A"), "/ 10")
        
        if not st.session_state.interview_history and not st.session_state.coding_history:
            st.info("No attempts yet. Start with interview questions or coding challenges!")
