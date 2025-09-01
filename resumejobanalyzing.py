import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import PyPDF2
from docx import Document
import numpy as np
import plotly.graph_objects as go
import json
import random
import time

# Download required NLTK data
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords') 
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

# Initialize NLTK data
download_nltk_data()

# Lightweight CSS
def load_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
    }
    
    .score-excellent { background: #10b981; color: white; }
    .score-good { background: #f59e0b; color: white; }
    .score-moderate { background: #ef4444; color: white; }
    
    .entity-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 0.25rem;
    }
    
    .entity-person { background: rgba(59, 130, 246, 0.2); color: #1d4ed8; }
    .entity-email { background: rgba(16, 185, 129, 0.2); color: #047857; }
    .entity-org { background: rgba(245, 158, 11, 0.2); color: #92400e; }
    
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

class CodingEvaluator:
    """Evaluate coding responses for SQL and Python questions"""
    
    def __init__(self):
        self.sql_keywords = ['select', 'from', 'where', 'join', 'group by', 'order by']
        self.python_concepts = ['def', 'class', 'import', 'for', 'while', 'if', 'try', 'except']
    
    def evaluate_sql_query(self, sql_code):
        """Evaluate SQL query response"""
        score_components = {
            'syntax': 0,
            'logic': 0,
            'efficiency': 0,
            'completeness': 0
        }
        
        sql_lower = sql_code.lower().strip()
        
        # Syntax check
        if any(keyword in sql_lower for keyword in ['select', 'from']):
            score_components['syntax'] = 0.8
        if ';' in sql_code:
            score_components['syntax'] = min(1.0, score_components['syntax'] + 0.2)
        
        # Logic check
        if 'where' in sql_lower:
            score_components['logic'] += 0.4
        if any(join in sql_lower for join in ['join', 'inner join', 'left join']):
            score_components['logic'] += 0.4
        if 'group by' in sql_lower:
            score_components['logic'] += 0.2
        
        # Efficiency
        if 'limit' in sql_lower:
            score_components['efficiency'] += 0.5
        if not 'select *' in sql_lower:
            score_components['efficiency'] += 0.5
        
        # Completeness
        word_count = len(sql_code.split())
        if word_count >= 8:
            score_components['completeness'] = min(1.0, word_count / 15)
        
        overall_score = sum(score_components.values()) / len(score_components) * 10
        
        return {
            'score': round(overall_score, 1),
            'components': {k: round(v * 10, 1) for k, v in score_components.items()},
            'feedback': self._generate_sql_feedback(score_components)
        }
    
    def evaluate_python_code(self, python_code):
        """Evaluate Python code response"""
        score_components = {
            'syntax': 0,
            'logic': 0,
            'pythonic': 0,
            'completeness': 0
        }
        
        # Syntax check
        try:
            compile(python_code, '<string>', 'exec')
            score_components['syntax'] = 1.0
        except SyntaxError:
            score_components['syntax'] = 0.3
        
        code_lower = python_code.lower()
        
        # Logic check
        if 'def ' in code_lower:
            score_components['logic'] += 0.4
        if 'return' in code_lower:
            score_components['logic'] += 0.3
        if any(control in code_lower for control in ['if', 'for', 'while']):
            score_components['logic'] += 0.3
        
        # Pythonic style
        if any(pythonic in code_lower for pythonic in ['enumerate', 'zip', 'list comprehension']):
            score_components['pythonic'] += 0.4
        if 'try:' in code_lower and 'except' in code_lower:
            score_components['pythonic'] += 0.3
        if not any(bad in code_lower for bad in ['global ', 'exec(']):
            score_components['pythonic'] += 0.3
        
        # Completeness
        line_count = len([line for line in python_code.split('\n') if line.strip()])
        if line_count >= 3:
            score_components['completeness'] = min(1.0, line_count / 10)
        
        overall_score = sum(score_components.values()) / len(score_components) * 10
        
        return {
            'score': round(overall_score, 1),
            'components': {k: round(v * 10, 1) for k, v in score_components.items()},
            'feedback': self._generate_python_feedback(score_components)
        }
    
    def _generate_sql_feedback(self, components):
        feedback = []
        if components['syntax'] < 0.7:
            feedback.append("Check SQL syntax - ensure proper SELECT/FROM structure")
        if components['logic'] < 0.5:
            feedback.append("Consider adding WHERE clauses, JOINs, or GROUP BY")
        if components['efficiency'] < 0.5:
            feedback.append("Avoid SELECT *, use LIMIT for large datasets")
        if components['completeness'] < 0.5:
            feedback.append("Expand your query to fully address the requirements")
        if not feedback:
            feedback.append("Good SQL structure!")
        return feedback
    
    def _generate_python_feedback(self, components):
        feedback = []
        if components['syntax'] < 0.7:
            feedback.append("Check Python syntax and indentation")
        if components['logic'] < 0.5:
            feedback.append("Add functions, control flow, and proper returns")
        if components['pythonic'] < 0.5:
            feedback.append("Use more Pythonic patterns and error handling")
        if components['completeness'] < 0.5:
            feedback.append("Provide a more complete solution")
        if not feedback:
            feedback.append("Great Python code!")
        return feedback

class ResumeJobAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js'],
            'data_science': ['pandas', 'numpy', 'sql', 'tableau', 'power bi'],
            'cloud': ['aws', 'azure', 'docker', 'kubernetes'],
            'databases': ['mysql', 'postgresql', 'mongodb'],
            'tools': ['git', 'jira', 'confluence']
        }
    
    def extract_text_from_pdf(self, file):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_skills(self, text):
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skill_categories.items():
            found_skills[category] = []
            for skill in skills:
                if skill in text_lower:
                    found_skills[category].append(skill)
        
        return found_skills
    
    def analyze_fit(self, resume_text, job_text=""):
        resume_skills = self.extract_skills(resume_text)
        word_count = len(resume_text.split())
        
        # Basic scoring
        base_score = 5.0
        if any(resume_skills.values()):
            base_score += 2.0
        if word_count > 200:
            base_score += 1.0
        if word_count > 500:
            base_score += 1.0
        if '@' in resume_text:
            base_score += 1.0
            
        if job_text:
            job_skills = self.extract_skills(job_text)
            # Simple job matching
            matches = 0
            total_job_skills = sum(len(skills) for skills in job_skills.values())
            
            if total_job_skills > 0:
                for category in job_skills:
                    job_category_skills = set(job_skills[category])
                    resume_category_skills = set(resume_skills.get(category, []))
                    matches += len(job_category_skills.intersection(resume_category_skills))
                
                match_rate = matches / total_job_skills
                base_score = (base_score * 0.6) + (match_rate * 10 * 0.4)
        else:
            job_skills = {}
        
        scaled_score = max(1, min(10, base_score))
        
        return {
            'score': round(scaled_score, 1),
            'resume_skills': resume_skills,
            'job_skills': job_skills,
            'recommendation': self.get_recommendation(scaled_score)
        }
    
    def get_recommendation(self, score):
        if score >= 8:
            return "🟢 Excellent fit! Please apply - you meet most requirements."
        elif score >= 6:
            return "🟡 Good fit! Consider applying - you have many relevant qualifications."
        elif score >= 4:
            return "🟠 Moderate fit. Consider highlighting transferable skills."
        else:
            return "🔴 Focus on skill development for better alignment."

class InterviewCoach:
    """AI Interview Coach with coding challenges"""
    
    def __init__(self):
        self.coding_evaluator = CodingEvaluator()
    
    def generate_coding_questions(self, skills):
        """Generate coding questions based on detected skills"""
        questions = {
            'sql_questions': [],
            'python_questions': []
        }
        
        # SQL Questions
        if any(skill in skills for skill in ['sql', 'mysql', 'postgresql', 'database']):
            questions['sql_questions'] = [
                "Write a SQL query to find the top 5 customers by total purchase amount.",
                "How would you find duplicate records in a customer table?",
                "Write a query to calculate the average order value per month.",
                "Create a query that joins users and orders tables to show user activity.",
                "Write a query to find customers who haven't made a purchase in the last 30 days."
            ]
        
        # Python Questions  
        if any(skill in skills for skill in ['python', 'programming']):
            questions['python_questions'] = [
                "Write a function to reverse a string without using built-in reverse methods.",
                "Create a function to find the intersection of two lists.",
                "Write a function to check if a string is a palindrome.",
                "Implement a simple calculator function that takes two numbers and an operator.",
                "Write a function to count the frequency of each character in a string."
            ]
        
        return questions
    
    def evaluate_coding_response(self, question, code, language):
        """Evaluate coding response"""
        if language == 'sql':
            return self.coding_evaluator.evaluate_sql_query(code)
        elif language == 'python':
            return self.coding_evaluator.evaluate_python_code(code)
        else:
            return {'score': 0, 'components': {}, 'feedback': ['Unknown language']}

def display_coding_results(evaluation, language, code):
    """Display coding evaluation results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score_class = "score-excellent" if evaluation['score'] >= 8 else "score-good" if evaluation['score'] >= 6 else "score-moderate"
        st.markdown(f'<div class="score-badge {score_class}">{evaluation["score"]}/10</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Language", language.upper())
    
    with col3:
        lines = len([line for line in code.split('\n') if line.strip()])
        st.metric("Lines of Code", lines)
    
    # Show code
    st.markdown("#### 📝 Your Code")
    st.code(code, language=language)
    
    # Component scores
    if evaluation['components']:
        st.markdown("#### 📊 Detailed Scoring")
        comp_cols = st.columns(len(evaluation['components']))
        for idx, (component, score) in enumerate(evaluation['components'].items()):
            with comp_cols[idx]:
                st.metric(component.title(), f"{score}/10")
    
    # Feedback
    st.markdown("#### 💡 Feedback")
    for feedback in evaluation['feedback']:
        st.write(f"• {feedback}")
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js'],
            'data_science': ['pandas', 'numpy', 'sql', 'tableau', 'power bi'],
            'cloud': ['aws', 'azure', 'docker', 'kubernetes'],
            'databases': ['mysql', 'postgresql', 'mongodb'],
            'tools': ['git', 'jira', 'confluence']
        }
    
    def extract_text_from_pdf(self, file):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_skills(self, text):
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skill_categories.items():
            found_skills[category] = []
            for skill in skills:
                if skill in text_lower:
                    found_skills[category].append(skill)
        
        return found_skills
    
    def analyze_fit(self, resume_text, job_text=""):
        resume_skills = self.extract_skills(resume_text)
        word_count = len(resume_text.split())
        
        # Basic scoring
        base_score = 5.0
        if any(resume_skills.values()):
            base_score += 2.0
        if word_count > 200:
            base_score += 1.0
        if word_count > 500:
            base_score += 1.0
        if '@' in resume_text:
            base_score += 1.0
            
        if job_text:
            job_skills = self.extract_skills(job_text)
            # Simple job matching
            matches = 0
            total_job_skills = sum(len(skills) for skills in job_skills.values())
            
            if total_job_skills > 0:
                for category in job_skills:
                    job_category_skills = set(job_skills[category])
                    resume_category_skills = set(resume_skills.get(category, []))
                    matches += len(job_category_skills.intersection(resume_category_skills))
                
                match_rate = matches / total_job_skills
                base_score = (base_score * 0.6) + (match_rate * 10 * 0.4)
        else:
            job_skills = {}
        
        scaled_score = max(1, min(10, base_score))
        
        return {
            'score': round(scaled_score, 1),
            'resume_skills': resume_skills,
            'job_skills': job_skills,
            'recommendation': self.get_recommendation(scaled_score)
        }
    
    def get_recommendation(self, score):
        if score >= 8:
            return "🟢 Excellent fit! Please apply - you meet most requirements."
        elif score >= 6:
            return "🟡 Good fit! Consider applying - you have many relevant qualifications."
        elif score >= 4:
            return "🟠 Moderate fit. Consider highlighting transferable skills."
        else:
            return "🔴 Focus on skill development for better alignment."

def simulate_ner(text):
    entities = []
    
    # Extract emails
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    for email in emails:
        entities.append({'entity_group': 'EMAIL', 'word': email, 'confidence': 0.95})
    
    # Extract phone numbers
    phones = re.findall(r'(\+?1?[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', text)
    for phone in phones:
        if phone:
            entities.append({'entity_group': 'PHONE', 'word': phone, 'confidence': 0.90})
    
    # Extract organizations
    org_keywords = ['Inc', 'LLC', 'Corp', 'Company', 'University', 'College']
    for keyword in org_keywords:
        pattern = rf'([A-Z][a-zA-Z\s]*{keyword})'
        matches = re.findall(pattern, text)
        for match in matches:
            entities.append({'entity_group': 'ORG', 'word': match.strip(), 'confidence': 0.80})
    
    return entities

def create_metric_card(title, value, subtitle=""):
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{value}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">{title}</div>
            {f'<div style="font-size: 0.8rem; opacity: 0.7;">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)

def display_entities(entities):
    if not entities:
        return
    
    st.markdown("#### 🔍 Extracted Information")
    entity_html = ""
    for entity in entities:
        entity_type = entity['entity_group'].lower()
        confidence = int(entity['confidence'] * 100)
        css_class = f"entity-{entity_type}" if entity_type in ['person', 'email', 'org'] else "entity-badge"
        
        entity_html += f"""
            <span class="entity-badge {css_class}">
                {entity['entity_group']}: {entity['word']} ({confidence}%)
            </span>
        """
    
    st.markdown(entity_html, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AI Resume & Interview Coach",
        page_icon="🤖",
        layout="wide"
    )
    
    # Load CSS
    load_css()
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 style="color: #1f2937; font-size: 2.5rem; margin-bottom: 0.5rem;">
                🤖 AI Resume & Interview Coach
            </h1>
            <p style="color: #6b7280; font-size: 1.1rem;">
                Powered by Advanced NLP & Machine Learning
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    analyzer = ResumeJobAnalyzer()
    
    # Initialize session state
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload Resume", "📊 Resume Analysis", "🎤 Interview Prep"])
    
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("## Upload Your Resume")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            resume_file = st.file_uploader(
                "Choose your resume file",
                type=['pdf', 'txt'],
                help="Upload a PDF or TXT file"
            )
            
            if resume_file:
                if resume_file.type == "application/pdf":
                    resume_text = analyzer.extract_text_from_pdf(resume_file)
                else:
                    resume_text = str(resume_file.read(), "utf-8")
                
                st.session_state.resume_text = resume_text
                st.success(f"✅ File uploaded: {resume_file.name}")
        
        with col2:
            st.markdown("### Or paste your resume text:")
            resume_text_input = st.text_area(
                "Resume content",
                height=200,
                placeholder="Paste your resume content here..."
            )
            
            if resume_text_input:
                st.session_state.resume_text = resume_text_input
        
        if st.session_state.resume_text:
            if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
                st.success("✅ Ready for analysis! Go to 'Resume Analysis' tab.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if st.session_state.resume_text:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📝 Resume Content")
                preview_text = st.session_state.resume_text[:300] + "..." if len(st.session_state.resume_text) > 300 else st.session_state.resume_text
                st.text_area("Preview", value=preview_text, height=150, disabled=True)
                
                job_description = st.text_area(
                    "Job Description (Optional)",
                    height=150,
                    placeholder="Paste job description for comparison..."
                )
                
                if st.button("🔬 Analyze with AI", type="primary", use_container_width=True):
                    with st.spinner("🤖 Analyzing..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Perform analysis
                        results = analyzer.analyze_fit(st.session_state.resume_text, job_description)
                        entities = simulate_ner(st.session_state.resume_text)
                        
                        st.session_state.analysis_results = results
                        st.session_state.entities = entities
                        
                        progress_bar.empty()
                        st.success("✅ Analysis complete!")
            
            with col2:
                st.markdown("### 📊 AI Analysis Results")
                
                if st.session_state.analysis_results:
                    results = st.session_state.analysis_results
                    
                    # Score display
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        create_metric_card("Overall Score", f"{results['score']}/10")
                    with col_b:
                        skill_count = sum(len(skills) for skills in results['resume_skills'].values())
                        create_metric_card("Skills Found", skill_count)
                    with col_c:
                        word_count = len(st.session_state.resume_text.split())
                        create_metric_card("Word Count", word_count)
                    
                    # Recommendation
                    st.markdown("### 💡 Recommendation")
                    st.info(results['recommendation'])
                    
                    # Entities
                    if 'entities' in st.session_state:
                        display_entities(st.session_state.entities)
                    
                    # Skills
                    if results['resume_skills']:
                        st.markdown("### 🛠️ Skills Found")
                        for category, skills in results['resume_skills'].items():
                            if skills:
                                st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(skills)}")
                else:
                    st.info("👈 Click 'Analyze with AI' to see results")
        else:
            st.warning("⚠️ Please upload your resume first.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if st.session_state.analysis_results and st.session_state.analysis_results['score'] >= 6:
            st.markdown("### 🎤 Interview Practice")
            
            # Initialize coaches
            interview_coach = InterviewCoach()
            
            # Create sub-tabs for different question types
            practice_tabs = st.tabs(["🤝 Behavioral", "⚙️ Technical", "💻 Coding Challenges"])
            
            with practice_tabs[0]:  # Behavioral Questions
                st.markdown("#### Behavioral Questions")
                behavioral_questions = [
                    "Tell me about yourself and your background.",
                    "Why are you interested in this position?",
                    "What are your greatest strengths?",
                    "Describe a challenging situation you faced and how you handled it.",
                    "Where do you see yourself in 5 years?",
                    "Tell me about a time you worked in a team.",
                    "Describe a project you're particularly proud of."
                ]
                
                selected_behavioral = st.selectbox("Select a behavioral question:", behavioral_questions)
                st.markdown(f"**Question:** {selected_behavioral}")
                
                behavioral_answer = st.text_area(
                    "Your Answer (Use STAR method - Situation, Task, Action, Result):",
                    height=150,
                    placeholder="Describe the Situation, Task you needed to accomplish, Actions you took, and Results achieved...",
                    key="behavioral_answer"
                )
                
                if st.button("📝 Evaluate Behavioral Response", use_container_width=True):
                    if behavioral_answer:
                        # STAR method scoring
                        word_count = len(behavioral_answer.split())
                        star_elements = sum([
                            1 for keyword in ['situation', 'task', 'action', 'result', 'when', 'what', 'how', 'outcome']
                            if keyword in behavioral_answer.lower()
                        ])
                        
                        score = min(10, max(3, (word_count / 25) * (star_elements / 2) * 10))
                        
                        score_class = "score-excellent" if score >= 8 else "score-good" if score >= 6 else "score-moderate"
                        st.markdown(f'<div class="score-badge {score_class}">{score:.1f}/10</div>', unsafe_allow_html=True)
                        
                        if score >= 8:
                            st.success("Excellent response! Good use of STAR method with specific details.")
                        elif score >= 6:
                            st.warning("Good response. Try to include more specific examples using STAR format.")
                        else:
                            st.error("Expand your answer using STAR: Situation, Task, Action, Result.")
                    else:
                        st.error("Please provide an answer first.")
            
            with practice_tabs[1]:  # Technical Questions
                st.markdown("#### Technical Questions")
                technical_questions = [
                    "How would you approach debugging a performance issue in an application?",
                    "Explain the difference between synchronous and asynchronous programming.",
                    "How do you ensure code quality in your projects?",
                    "Describe your experience with version control systems.",
                    "How would you design a simple web application architecture?",
                    "What's your approach to testing software?",
                    "How do you stay updated with new technologies?"
                ]
                
                selected_technical = st.selectbox("Select a technical question:", technical_questions)
                st.markdown(f"**Question:** {selected_technical}")
                
                technical_answer = st.text_area(
                    "Your Technical Answer:",
                    height=150,
                    placeholder="Provide a detailed technical explanation with examples...",
                    key="technical_answer"
                )
                
                if st.button("🔍 Evaluate Technical Response", use_container_width=True):
                    if technical_answer:
                        # Technical scoring
                        word_count = len(technical_answer.split())
                        tech_keywords = sum([
                            1 for keyword in ['algorithm', 'data', 'structure', 'performance', 'testing', 'debugging', 'architecture']
                            if keyword in technical_answer.lower()
                        ])
                        
                        score = min(10, max(3, (word_count / 20) * (tech_keywords / 2) * 10))
                        
                        score_class = "score-excellent" if score >= 8 else "score-good" if score >= 6 else "score-moderate"
                        st.markdown(f'<div class="score-badge {score_class}">{score:.1f}/10</div>', unsafe_allow_html=True)
                        
                        if score >= 8:
                            st.success("Excellent technical response! Good depth and technical terminology.")
                        elif score >= 6:
                            st.warning("Good technical knowledge. Consider adding more specific examples.")
                        else:
                            st.error("Expand with more technical details and concrete examples.")
                    else:
                        st.error("Please provide an answer first.")
            
            with practice_tabs[2]:  # Coding Challenges
                st.markdown("#### 💻 Coding Challenges")
                
                # Get skills for relevant questions
                skills = []
                if st.session_state.analysis_results:
                    for skill_list in st.session_state.analysis_results['resume_skills'].values():
                        skills.extend(skill_list)
                
                # Generate coding questions
                coding_questions = interview_coach.generate_coding_questions(skills)
                
                if coding_questions['sql_questions'] or coding_questions['python_questions']:
                    
                    # SQL Section
                    if coding_questions['sql_questions']:
                        st.markdown("##### 🗃️ SQL Challenges")
                        selected_sql = st.selectbox("Select SQL Challenge:", coding_questions['sql_questions'], key="sql_select")
                        
                        st.markdown(f"**SQL Challenge:** {selected_sql}")
                        
                        sql_code = st.text_area(
                            "Write your SQL solution:",
                            height=120,
                            placeholder="-- Write your SQL query here\nSELECT column1, column2\nFROM table_name\nWHERE condition;",
                            key="sql_code_input"
                        )
                        
                        if st.button("🔍 Evaluate SQL Code", key="eval_sql"):
                            if sql_code.strip():
                                evaluation = interview_coach.evaluate_coding_response(selected_sql, sql_code, 'sql')
                                st.markdown("---")
                                display_coding_results(evaluation, 'sql', sql_code)
                            else:
                                st.error("Please write some SQL code first.")
                        
                        st.markdown("---")
                    
                    # Python Section
                    if coding_questions['python_questions']:
                        st.markdown("##### 🐍 Python Challenges")
                        selected_python = st.selectbox("Select Python Challenge:", coding_questions['python_questions'], key="python_select")
                        
                        st.markdown(f"**Python Challenge:** {selected_python}")
                        
                        python_code = st.text_area(
                            "Write your Python solution:",
                            height=120,
                            placeholder="# Write your Python code here\ndef solution():\n    # Your implementation here\n    return result",
                            key="python_code_input"
                        )
                        
                        if st.button("🔍 Evaluate Python Code", key="eval_python"):
                            if python_code.strip():
                                evaluation = interview_coach.evaluate_coding_response(selected_python, python_code, 'python')
                                st.markdown("---")
                                display_coding_results(evaluation, 'python', python_code)
                            else:
                                st.error("Please write some Python code first.")
                    
                    # Coding Tips
                    st.markdown("##### 💡 Coding Interview Tips")
                    st.info("""
                    **SQL Tips:**
                    • Always use proper SELECT, FROM, WHERE structure
                    • Consider using JOINs when working with multiple tables
                    • Use LIMIT for large datasets
                    • Avoid SELECT * in production queries
                    
                    **Python Tips:**
                    • Write clean, readable code with good variable names
                    • Use functions to organize your logic
                    • Include error handling with try/except
                    • Consider edge cases in your solution
                    • Use Pythonic patterns when possible
                    """)
                
                else:
                    st.info("🎯 Upload a resume with programming/database skills to see coding challenges!")
                    st.markdown("**Available when you have skills in:**")
                    st.write("• SQL, MySQL, PostgreSQL (for SQL challenges)")
                    st.write("• Python, Programming (for Python challenges)")
        
        elif st.session_state.analysis_results:
            st.info("📈 Your resume score is below 6. Focus on improving your resume first.")
        else:
            st.warning("⚠️ Please complete resume analysis first.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            🤖 AI-powered resume analysis and interview coaching
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()