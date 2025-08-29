import streamlit as st
import openai
from openai import OpenAI
import PyPDF2
import docx
import io
import re
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple

# Configure page
st.set_page_config(
    page_title="AI Career Coach",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = "upload"
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'fit_score' not in st.session_state:
    st.session_state.fit_score = 0
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}
if 'interview_scores' not in st.session_state:
    st.session_state.interview_scores = {}

class AIAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def generate_response(self, prompt: str, system_message: str = "") -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

class ResumeJobAnalyzer(AIAgent):
    def evaluate_fit(self, resume: str, job_desc: str) -> Dict:
        system_message = """You are an expert HR analyst and career coach. 
        Evaluate how well a candidate's resume matches a job description.
        Provide a detailed analysis with a score from 1-10 and specific recommendations."""
        
        prompt = f"""
        Analyze the following resume against the job description and provide:
        1. Overall fit score (1-10)
        2. Detailed breakdown of strengths
        3. Areas for improvement
        4. Specific recommendations
        5. Key missing skills or qualifications
        
        Resume:
        {resume}
        
        Job Description:
        {job_desc}
        
        Please format your response as JSON with the following structure:
        {{
            "overall_score": <number 1-10>,
            "recommendation": "<apply/consider_applying/needs_improvement/pursue_other_opportunities>",
            "strengths": ["list of strengths"],
            "weaknesses": ["list of weaknesses"],
            "missing_skills": ["list of missing skills"],
            "improvement_suggestions": ["list of suggestions"],
            "detailed_analysis": "comprehensive analysis text"
        }}
        """
        
        response = self.generate_response(prompt, system_message)
        try:
            return json.loads(response)
        except:
            return {"error": "Failed to parse evaluation"}

class ResumeImprovementAgent(AIAgent):
    def suggest_improvements(self, resume: str, job_desc: str, evaluation: Dict) -> str:
        system_message = """You are a professional resume writer and career coach.
        Help candidates improve their resumes based on job requirements and evaluation feedback."""
        
        prompt = f"""
        Based on the following resume, job description, and evaluation feedback,
        provide specific suggestions to improve the resume:
        
        Resume: {resume}
        Job Description: {job_desc}
        Evaluation: {evaluation}
        
        Provide:
        1. Specific sections to modify
        2. Keywords to include
        3. Experience/skills to highlight
        4. Format improvements
        5. A revised version of key sections
        """
        
        return self.generate_response(prompt, system_message)

class InterviewPrepAgent(AIAgent):
    def generate_behavioral_questions(self, job_desc: str) -> List[str]:
        system_message = """You are an interview preparation expert.
        Generate relevant behavioral interview questions based on the job description."""
        
        prompt = f"""
        Based on this job description, generate 5 relevant behavioral interview questions
        that would likely be asked for this position:
        
        {job_desc}
        
        Format as a JSON list of questions.
        """
        
        response = self.generate_response(prompt, system_message)
        try:
            return json.loads(response)
        except:
            return ["Tell me about a time you faced a challenge at work."]

    def generate_technical_questions(self, job_desc: str, tech_type: str) -> List[str]:
        system_message = f"""You are a technical interview expert specializing in {tech_type}.
        Generate relevant technical questions and coding scenarios."""
        
        prompt = f"""
        Based on this job description, generate 3 {tech_type} technical questions
        or coding scenarios that would be appropriate for this role:
        
        {job_desc}
        
        Format as a JSON list of questions/scenarios.
        """
        
        response = self.generate_response(prompt, system_message)
        try:
            return json.loads(response)
        except:
            return [f"Write a basic {tech_type} function to solve a common problem."]

    def evaluate_interview_response(self, question: str, answer: str, job_context: str) -> Dict:
        system_message = """You are an interview evaluator. Assess responses fairly and provide constructive feedback."""
        
        prompt = f"""
        Evaluate this interview response:
        
        Question: {question}
        Answer: {answer}
        Job Context: {job_context}
        
        Provide evaluation in JSON format:
        {{
            "score": <1-10>,
            "strengths": ["list of strengths"],
            "improvements": ["list of improvements"],
            "feedback": "detailed constructive feedback"
        }}
        """
        
        response = self.generate_response(prompt, system_message)
        try:
            return json.loads(response)
        except:
            return {"score": 5, "feedback": "Unable to evaluate response"}

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded PDF or DOCX file."""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    else:
        return uploaded_file.read().decode('utf-8')

def create_score_gauge(score: int, title: str):
    """Create a gauge chart for scores."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 5},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgray"},
                {'range': [3, 6], 'color': "yellow"},
                {'range': [6, 8], 'color': "lightgreen"},
                {'range': [8, 10], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 7
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    st.title("🚀 AI Career Coach & Interview Prep")
    st.markdown("*Powered by OpenAI GPT-4 for comprehensive career guidance*")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("🔑 Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Enter your OpenAI API key to enable AI functionality"
        )
        if api_key:
            st.session_state.openai_api_key = api_key
        
        st.header("🎯 Navigation")
        stages = ["📤 Upload Documents", "📊 Resume Analysis", "✏️ Resume Improvement", "🎤 Interview Prep"]
        current_stage = st.radio("Select Stage:", stages)
    
    if not st.session_state.openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        st.info("You can get an API key from https://platform.openai.com/api-keys")
        return
    
    # Initialize AI agents
    analyzer = ResumeJobAnalyzer(st.session_state.openai_api_key)
    resume_agent = ResumeImprovementAgent(st.session_state.openai_api_key)
    interview_agent = InterviewPrepAgent(st.session_state.openai_api_key)
    
    # Stage 1: Document Upload
    if "📤 Upload Documents" in current_stage:
        st.header("📤 Upload Your Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Resume Upload")
            resume_file = st.file_uploader(
                "Upload your resume",
                type=['pdf', 'docx', 'txt'],
                help="Supported formats: PDF, DOCX, TXT"
            )
            
            if resume_file:
                with st.spinner("Extracting text from resume..."):
                    st.session_state.resume_text = extract_text_from_file(resume_file)
                st.success("✅ Resume uploaded successfully!")
                with st.expander("Preview Resume Text"):
                    st.text_area("Resume Content", st.session_state.resume_text, height=200, disabled=True)
        
        with col2:
            st.subheader("💼 Job Description")
            job_desc = st.text_area(
                "Paste the job description here",
                height=300,
                placeholder="Copy and paste the job description you're interested in..."
            )
            
            if job_desc:
                st.session_state.job_description = job_desc
                st.success("✅ Job description added!")
        
        if st.session_state.resume_text and st.session_state.job_description:
            if st.button("🔍 Analyze Fit", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your resume against the job description..."):
                    evaluation = analyzer.evaluate_fit(
                        st.session_state.resume_text, 
                        st.session_state.job_description
                    )
                    st.session_state.evaluation_results = evaluation
                    if 'overall_score' in evaluation:
                        st.session_state.fit_score = evaluation['overall_score']
                st.success("Analysis complete! Check the Resume Analysis tab.")
    
    # Stage 2: Resume Analysis
    elif "📊 Resume Analysis" in current_stage:
        st.header("📊 Resume-Job Fit Analysis")
        
        if st.session_state.evaluation_results:
            eval_data = st.session_state.evaluation_results
            
            # Score display
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.plotly_chart(
                    create_score_gauge(eval_data.get('overall_score', 0), "Overall Fit Score"),
                    use_container_width=True
                )
            
            with col2:
                score = eval_data.get('overall_score', 0)
                if score >= 8:
                    st.success("🎉 Excellent Fit!")
                    st.markdown("**Recommendation: Apply immediately!**")
                elif score >= 6:
                    st.info("👍 Good Fit")
                    st.markdown("**Recommendation: Apply with confidence**")
                elif score >= 4:
                    st.warning("⚠️ Moderate Fit")
                    st.markdown("**Recommendation: Improve resume first**")
                else:
                    st.error("❌ Poor Fit")
                    st.markdown("**Recommendation: Consider other opportunities**")
            
            with col3:
                st.metric("Fit Score", f"{score}/10", delta=score-5)
            
            # Detailed Analysis
            st.subheader("📋 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Strengths")
                for strength in eval_data.get('strengths', []):
                    st.markdown(f"• {strength}")
                
                st.markdown("### 🎯 Missing Skills")
                for skill in eval_data.get('missing_skills', []):
                    st.markdown(f"• {skill}")
            
            with col2:
                st.markdown("### 🔧 Areas for Improvement")
                for weakness in eval_data.get('weaknesses', []):
                    st.markdown(f"• {weakness}")
                
                st.markdown("### 💡 Suggestions")
                for suggestion in eval_data.get('improvement_suggestions', []):
                    st.markdown(f"• {suggestion}")
            
            # Full analysis
            with st.expander("📖 Complete Analysis"):
                st.markdown(eval_data.get('detailed_analysis', 'No detailed analysis available'))
        
        else:
            st.info("📤 Please upload your documents first to see the analysis.")
    
    # Stage 3: Resume Improvement
    elif "✏️ Resume Improvement" in current_stage:
        st.header("✏️ Resume Improvement Agent")
        
        if st.session_state.fit_score < 6:
            st.warning("🔧 Your resume could benefit from improvements. Let our AI agent help!")
            
            if st.button("🤖 Get Resume Improvement Suggestions", type="primary"):
                with st.spinner("AI Resume Agent is analyzing your resume..."):
                    improvements = resume_agent.suggest_improvements(
                        st.session_state.resume_text,
                        st.session_state.job_description,
                        st.session_state.evaluation_results
                    )
                    
                    st.subheader("📝 AI Resume Improvement Suggestions")
                    st.markdown(improvements)
        
        elif st.session_state.fit_score >= 6:
            st.success("🎉 Great! Your resume shows a good fit. Ready for interview prep?")
            
            if st.button("🎯 Proceed to Interview Preparation", type="primary"):
                st.session_state.current_stage = "interview"
                st.rerun()
        
        else:
            st.info("📊 Please complete the resume analysis first.")
    
    # Stage 4: Interview Preparation
    elif "🎤 Interview Prep" in current_stage:
        st.header("🎤 Interview Preparation Hub")
        
        tab1, tab2, tab3 = st.tabs(["🗣️ Behavioral Interview", "🐍 Python Coding", "🗄️ SQL Challenges"])
        
        with tab1:
            st.subheader("Behavioral Interview Preparation")
            
            if st.button("Generate Behavioral Questions"):
                with st.spinner("Generating behavioral questions..."):
                    questions = interview_agent.generate_behavioral_questions(st.session_state.job_description)
                    st.session_state.behavioral_questions = questions
            
            if 'behavioral_questions' in st.session_state:
                for i, question in enumerate(st.session_state.behavioral_questions):
                    st.markdown(f"**Question {i+1}:** {question}")
                    
                    answer = st.text_area(
                        f"Your answer to question {i+1}:",
                        key=f"behavioral_answer_{i}",
                        height=100
                    )
                    
                    if answer and st.button(f"Evaluate Answer {i+1}", key=f"eval_behavioral_{i}"):
                        with st.spinner("Evaluating your response..."):
                            evaluation = interview_agent.evaluate_interview_response(
                                question, answer, st.session_state.job_description
                            )
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.metric("Score", f"{evaluation.get('score', 0)}/10")
                            with col2:
                                st.markdown("**Feedback:**")
                                st.markdown(evaluation.get('feedback', 'No feedback available'))
        
        with tab2:
            st.subheader("Python Coding Interview")
            
            if st.button("Generate Python Challenges"):
                with st.spinner("Generating Python coding questions..."):
                    questions = interview_agent.generate_technical_questions(
                        st.session_state.job_description, "Python"
                    )
                    st.session_state.python_questions = questions
            
            if 'python_questions' in st.session_state:
                for i, question in enumerate(st.session_state.python_questions):
                    st.markdown(f"**Challenge {i+1}:** {question}")
                    
                    code_answer = st.text_area(
                        f"Your Python solution for challenge {i+1}:",
                        key=f"python_answer_{i}",
                        height=150,
                        placeholder="def solution():\n    # Your code here\n    pass"
                    )
                    
                    if code_answer and st.button(f"Evaluate Solution {i+1}", key=f"eval_python_{i}"):
                        with st.spinner("Evaluating your code..."):
                            evaluation = interview_agent.evaluate_interview_response(
                                question, code_answer, st.session_state.job_description
                            )
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.metric("Score", f"{evaluation.get('score', 0)}/10")
                            with col2:
                                st.markdown("**Code Review:**")
                                st.markdown(evaluation.get('feedback', 'No feedback available'))
        
        with tab3:
            st.subheader("SQL Interview Preparation")
            
            if st.button("Generate SQL Challenges"):
                with st.spinner("Generating SQL questions..."):
                    questions = interview_agent.generate_technical_questions(
                        st.session_state.job_description, "SQL"
                    )
                    st.session_state.sql_questions = questions
            
            if 'sql_questions' in st.session_state:
                for i, question in enumerate(st.session_state.sql_questions):
                    st.markdown(f"**SQL Challenge {i+1}:** {question}")
                    
                    sql_answer = st.text_area(
                        f"Your SQL query for challenge {i+1}:",
                        key=f"sql_answer_{i}",
                        height=120,
                        placeholder="SELECT \n  column1,\n  column2\nFROM table_name\nWHERE condition;"
                    )
                    
                    if sql_answer and st.button(f"Evaluate Query {i+1}", key=f"eval_sql_{i}"):
                        with st.spinner("Evaluating your SQL..."):
                            evaluation = interview_agent.evaluate_interview_response(
                                question, sql_answer, st.session_state.job_description
                            )
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.metric("Score", f"{evaluation.get('score', 0)}/10")
                            with col2:
                                st.markdown("**Query Review:**")
                                st.markdown(evaluation.get('feedback', 'No feedback available'))
    
    # Progress tracking sidebar
    with st.sidebar:
        st.header("📈 Your Progress")
        
        progress_data = {
            "Resume Upload": "✅" if st.session_state.resume_text else "⏳",
            "Job Description": "✅" if st.session_state.job_description else "⏳",
            "Fit Analysis": "✅" if st.session_state.evaluation_results else "⏳",
        }
        
        for task, status in progress_data.items():
            st.markdown(f"{status} {task}")
        
        if st.session_state.fit_score > 0:
            st.markdown(f"**Current Fit Score: {st.session_state.fit_score}/10**")
        
        st.markdown("---")
        st.markdown("### 🤖 Available AI Agents")
        st.markdown("• **Resume Analyzer**: Evaluates job fit")
        st.markdown("• **Resume Improver**: Suggests enhancements")
        st.markdown("• **Interview Coach**: Preps for interviews")
        st.markdown("• **Code Reviewer**: Evaluates technical responses")

if __name__ == "__main__":
    main()

# Installation requirements (create requirements.txt):
"""
streamlit>=1.28.0
openai>=1.0.0
PyPDF2>=3.0.0
python-docx>=0.8.11
plotly>=5.15.0
"""