import streamlit as st
import pandas as pd
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

# Optional sklearn imports with fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("Scikit-learn not available. Using simplified text matching.")

# Download required NLTK data
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords') 
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            st.error(f"Error downloading NLTK data: {e}")

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
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            st.warning(f"NLTK initialization warning: {e}")
            self.lemmatizer = None
            self.stop_words = set()
        
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'scala'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask'],
            'data_science': ['pandas', 'numpy', 'sql', 'tableau', 'power bi', 'matplotlib', 'seaborn', 'scikit-learn'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite'],
            'tools': ['git', 'jira', 'confluence', 'jenkins', 'gitlab'],
            'frameworks': ['spring', 'hibernate', 'tensorflow', 'pytorch', 'opencv']
        }
    
    def extract_text_from_pdf(self, file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file):
        """Extract text from DOCX file"""
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def extract_skills(self, text):
        """Extract skills from text"""
        if not text:
            return {}
            
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skill_categories.items():
            found_skills[category] = []
            for skill in skills:
                # Use word boundaries for better matching
                pattern = rf'\b{re.escape(skill)}\b'
                if re.search(pattern, text_lower):
                    found_skills[category].append(skill)
        
        return found_skills
    
    def calculate_similarity_score(self, resume_text, job_text):
        """Calculate similarity between resume and job description"""
        if not SKLEARN_AVAILABLE or not job_text:
            return self.calculate_simple_similarity(resume_text, job_text)
        
        try:
            # Use TF-IDF for better matching
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            st.warning(f"Advanced similarity calculation failed: {e}. Using simple matching.")
            return self.calculate_simple_similarity(resume_text, job_text)
    
    def calculate_simple_similarity(self, resume_text, job_text):
        """Simple keyword-based similarity calculation"""
        if not job_text:
            return 0.5  # Default moderate score
        
        resume_words = set(resume_text.lower().split())
        job_words = set(job_text.lower().split())
        
        if not job_words:
            return 0.5
        
        intersection = resume_words.intersection(job_words)
        return len(intersection) / len(job_words)
    
    def analyze_fit(self, resume_text, job_text=""):
        """Analyze resume-job fit"""
        if not resume_text.strip():
            return {
                'score': 0,
                'resume_skills': {},
                'job_skills': {},
                'recommendation': "Please provide resume content to analyze."
            }
        
        resume_skills = self.extract_skills(resume_text)
        word_count = len(resume_text.split())
        
        # Basic scoring components
        base_score = 3.0  # Starting point
        
        # Content quality scoring
        if word_count > 100:
            base_score += 1.0
        if word_count > 300:
            base_score += 1.0
        if word_count > 500:
            base_score += 0.5
        
        # Skills scoring
        total_skills = sum(len(skills) for skills in resume_skills.values())
        if total_skills > 0:
            base_score += min(2.0, total_skills * 0.2)
        
        # Contact info
        if '@' in resume_text:
            base_score += 0.5
        
        # Phone number
        if re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', resume_text):
            base_score += 0.5
        
        # Job matching if job description provided
        if job_text.strip():
            job_skills = self.extract_skills(job_text)
            
            # Calculate skill overlap
            skill_matches = 0
            total_job_skills = sum(len(skills) for skills in job_skills.values())
            
            if total_job_skills > 0:
                for category in job_skills:
                    job_category_skills = set(job_skills[category])
                    resume_category_skills = set(resume_skills.get(category, []))
                    skill_matches += len(job_category_skills.intersection(resume_category_skills))
                
                skill_match_rate = skill_matches / total_job_skills
                
                # Text similarity
                text_similarity = self.calculate_similarity_score(resume_text, job_text)
                
                # Combined matching score
                combined_match = (skill_match_rate * 0.6) + (text_similarity * 0.4)
                base_score = (base_score * 0.4) + (combined_match * 10 * 0.6)
            else:
                job_skills = {}
        else:
            job_skills = {}
        
        # Ensure score is between 1 and 10
        final_score = max(1.0, min(10.0, base_score))
        
        return {
            'score': round(final_score, 1),
            'resume_skills': resume_skills,
            'job_skills': job_skills,
            'recommendation': self.get_recommendation(final_score),
            'word_count': word_count,
            'total_skills': sum(len(skills) for skills in resume_skills.values())
        }
    
    def get_recommendation(self, score):
        """Get recommendation based on score"""
        if score >= 8.5:
            return "🟢 Excellent fit! You're highly qualified - definitely apply!"
        elif score >= 7.0:
            return "🟡 Strong fit! You meet most requirements - apply with confidence."
        elif score >= 5.5:
            return "🟠 Good potential fit. Highlight your transferable skills."
        elif score >= 4.0:
            return "🟠 Moderate fit. Consider tailoring your resume to better match."
        else:
            return "🔴 Focus on developing relevant skills and experience first."

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
        
        # Check for database/SQL skills
        sql_skills = ['sql', 'mysql', 'postgresql', 'database', 'sqlite', 'oracle']
        if any(skill.lower() in ' '.join(skills).lower() for skill in sql_skills):
            questions['sql_questions'] = [
                "Write a SQL query to find the top 5 customers by total purchase amount.",
                "How would you find duplicate records in a customer table?",
                "Write a query to calculate the average order value per month.",
                "Create a query that joins users and orders tables to show user activity.",
                "Write a query to find customers who haven't made a purchase in the last 30 days.",
                "How would you optimize a slow-running query?",
                "Write a query to find the second highest salary in an employees table."
            ]
        
        # Check for Python skills
        python_skills = ['python', 'programming', 'coding', 'software', 'development']
        if any(skill.lower() in ' '.join(skills).lower() for skill in python_skills):
            questions['python_questions'] = [
                "Write a function to reverse a string without using built-in reverse methods.",
                "Create a function to find the intersection of two lists.",
                "Write a function to check if a string is a palindrome.",
                "Implement a simple calculator function that takes two numbers and an operator.",
                "Write a function to count the frequency of each character in a string.",
                "Create a function to find the largest element in a nested list.",
                "Write a function to remove duplicates from a list while preserving order."
            ]
        
        return questions
    
    def evaluate_coding_response(self, question, code, language):
        """Evaluate coding response"""
        if not code.strip():
            return {'score': 0, 'components': {}, 'feedback': ['No code provided']}
        
        if language.lower() == 'sql':
            return self.coding_evaluator.evaluate_sql_query(code)
        elif language.lower() == 'python':
            return self.coding_evaluator.evaluate_python_code(code)
        else:
            return {'score': 0, 'components': {}, 'feedback': ['Unknown language']}

def simulate_ner(text):
    """Simulate named entity recognition"""
    entities = []
    
    # Extract emails
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    emails = re.findall(email_pattern, text)
    for email in emails:
        entities.append({'entity_group': 'EMAIL', 'word': email, 'confidence': 0.95})
    
    # Extract phone numbers
    phone_pattern = r'(\+?1?[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
    phones = re.findall(phone_pattern, text)
    for phone in phones:
        if phone and len(re.sub(r'[^\d]', '', phone)) >= 10:
            entities.append({'entity_group': 'PHONE', 'word': phone, 'confidence': 0.90})
    
    # Extract organizations
    org_keywords = ['Inc', 'LLC', 'Corp', 'Company', 'University', 'College', 'Institute', 'Technologies', 'Systems']
    for keyword in org_keywords:
        pattern = rf'([A-Z][a-zA-Z\s&]*{keyword})'
        matches = re.findall(pattern, text)
        for match in matches:
            clean_match = match.strip()
            if len(clean_match) > 3:  # Avoid very short matches
                entities.append({'entity_group': 'ORG', 'word': clean_match, 'confidence': 0.80})
    
    # Extract potential names (capitalized words)
    name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    names = re.findall(name_pattern, text)
    for name in names[:3]:  # Limit to first 3 potential names
        if not any(keyword in name for keyword in org_keywords):
            entities.append({'entity_group': 'PERSON', 'word': name, 'confidence': 0.75})
    
    return entities

def create_metric_card(title, value, subtitle=""):
    """Create a metric card display"""
    st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{value}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">{title}</div>
            {f'<div style="font-size: 0.8rem; opacity: 0.7;">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)

def display_entities(entities):
    """Display extracted entities"""
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

def display_skills_visualization(skills_data):
    """Create skills visualization"""
    if not any(skills_data.values()):
        return
    
    categories = []
    counts = []
    
    for category, skills in skills_data.items():
        if skills:
            categories.append(category.replace('_', ' ').title())
            counts.append(len(skills))
    
    if categories:
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'][:len(categories)]
            )
        ])
        
        fig.update_layout(
            title="Skills by Category",
            xaxis_title="Skill Categories",
            yaxis_title="Number of Skills",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    st.set_page_config(
        page_title="AI Resume & Interview Coach",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
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
    if 'entities' not in st.session_state:
        st.session_state.entities = []
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload Resume", "📊 Resume Analysis", "🎤 Interview Prep"])
    
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("## Upload Your Resume")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### File Upload")
            resume_file = st.file_uploader(
                "Choose your resume file",
                type=['pdf', 'txt', 'docx'],
                help="Upload a PDF, DOCX, or TXT file"
            )
            
            if resume_file:
                try:
                    if resume_file.type == "application/pdf":
                        resume_text = analyzer.extract_text_from_pdf(resume_file)
                    elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        resume_text = analyzer.extract_text_from_docx(resume_file)
                    else:
                        resume_text = str(resume_file.read(), "utf-8")
                    
                    if resume_text.strip():
                        st.session_state.resume_text = resume_text
                        st.success(f"✅ File uploaded: {resume_file.name}")
                        st.info(f"Extracted {len(resume_text.split())} words")
                    else:
                        st.error("No text could be extracted from the file.")
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        with col2:
            st.markdown("### Or Paste Resume Text")
            resume_text_input = st.text_area(
                "Resume content",
                height=200,
                placeholder="Paste your resume content here..."
            )
            
            if resume_text_input.strip():
                st.session_state.resume_text = resume_text_input
        
        if st.session_state.resume_text:
            st.markdown("### Preview")
            preview_text = st.session_state.resume_text[:500] + "..." if len(st.session_state.resume_text) > 500 else st.session_state.resume_text
            st.text_area("Resume Preview", value=preview_text, height=100, disabled=True)
            
            if st.button("🚀 Ready for Analysis", type="primary", use_container_width=True):
                st.success("✅ Ready for analysis! Go to 'Resume Analysis' tab.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if st.session_state.resume_text:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🎯 Job Matching (Optional)")
                job_description = st.text_area(
                    "Job Description",
                    height=200,
                    placeholder="Paste job description here for targeted analysis...",
                    help="Provide a job description to get personalized matching analysis"
                )
                
                if st.button("🔬 Analyze with AI", type="primary", use_container_width=True):
                    with st.spinner("🤖 Analyzing your resume..."):
                        # Progress simulation
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.02)
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
                    
                    # Main metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        create_metric_card("Overall Score", f"{results['score']}/10")
                    with col_b:
                        create_metric_card("Skills Found", results.get('total_skills', 0))
                    with col_c:
                        create_metric_card("Word Count", results.get('word_count', 0))
                    
                    # Recommendation
                    st.markdown("### 💡 AI Recommendation")
                    recommendation = results['recommendation']
                    if results['score'] >= 7:
                        st.success(recommendation)
                    elif results['score'] >= 5:
                        st.warning(recommendation)
                    else:
                        st.error(recommendation)
                    
                    # Skills breakdown
                    if results['resume_skills']:
                        st.markdown("### 🛠️ Skills Analysis")
                        
                        # Skills visualization
                        display_skills_visualization(results['resume_skills'])
                        
                        # Skills details
                        skills_found = False
                        for category, skills in results['resume_skills'].items():
                            if skills:
                                skills_found = True
                                st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(skills)}")
                        
                        if not skills_found:
                            st.info("No specific technical skills detected. Consider adding more technical keywords to your resume.")
                    
                    # Job matching results
                    if results['job_skills']:
                        st.markdown("### 🎯 Job Matching Analysis")
                        
                        # Show job skills
                        st.markdown("**Required Skills Found in Job Description:**")
                        for category, skills in results['job_skills'].items():
                            if skills:
                                st.write(f"• **{category.replace('_', ' ').title()}:** {', '.join(skills)}")
                        
                        # Show matches
                        matches_found = False
                        st.markdown("**Skills You Have That Match:**")
                        for category in results['job_skills']:
                            job_skills_set = set(results['job_skills'][category])
                            resume_skills_set = set(results['resume_skills'].get(category, []))
                            matches = job_skills_set.intersection(resume_skills_set)
                            if matches:
                                matches_found = True
                                st.write(f"✅ **{category.replace('_', ' ').title()}:** {', '.join(matches)}")
                        
                        if not matches_found:
                            st.warning("No direct skill matches found. Consider highlighting transferable skills.")
                    
                    # Entities
                    if st.session_state.entities:
                        display_entities(st.session_state.entities)
                    
                else:
                    st.info("👈 Click 'Analyze with AI' to see results")
        else:
            st.warning("⚠️ Please upload your resume first in the 'Upload Resume' tab.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if st.session_state.analysis_results:
            st.markdown("### 🎤 Interview Practice")
            
            # Initialize coach
            interview_coach = InterviewCoach()
            
            # Get skills for question generation
            all_skills = []
            if st.session_state.analysis_results['resume_skills']:
                for skill_list in st.session_state.analysis_results['resume_skills'].values():
                    all_skills.extend(skill_list)
            
            # Create sub-tabs for different question types
            practice_tabs = st.tabs(["🤝 Behavioral", "⚙️ Technical", "💻 Coding Challenges"])
            
            with practice_tabs[0]:  # Behavioral Questions
                st.markdown("#### Behavioral Interview Questions")
                behavioral_questions = [
                    "Tell me about yourself and your background.",
                    "Why are you interested in this position?",
                    "What are your greatest strengths and weaknesses?",
                    "Describe a challenging situation you faced and how you handled it.",
                    "Where do you see yourself in 5 years?",
                    "Tell me about a time you worked in a team.",
                    "Describe a project you're particularly proud of.",
                    "How do you handle stress and pressure?",
                    "Tell me about a time you had to learn something new quickly.",
                    "Describe a situation where you had to deal with conflict."
                ]
                
                selected_behavioral = st.selectbox("Select a behavioral question:", behavioral_questions)
                st.markdown(f"**Question:** {selected_behavioral}")
                
                st.info("💡 **Tip:** Use the STAR method - Situation, Task, Action, Result")
                
                behavioral_answer = st.text_area(
                    "Your Answer:",
                    height=150,
                    placeholder="Use STAR method:\n• Situation: Describe the context\n• Task: Explain what needed to be done\n• Action: Detail what you did\n• Result: Share the outcome",
                    key="behavioral_answer"
                )
                
                if st.button("📝 Evaluate Behavioral Response", use_container_width=True):
                    if behavioral_answer.strip():
                        # STAR method evaluation
                        word_count = len(behavioral_answer.split())
                        
                        # Check for STAR elements
                        star_indicators = {
                            'situation': ['situation', 'when', 'at', 'during', 'while working'],
                            'task': ['task', 'needed to', 'had to', 'responsible for', 'goal was'],
                            'action': ['i', 'did', 'implemented', 'created', 'developed', 'led'],
                            'result': ['result', 'outcome', 'achieved', 'improved', 'increased', 'reduced']
                        }
                        
                        star_score = 0
                        answer_lower = behavioral_answer.lower()
                        
                        for element, indicators in star_indicators.items():
                            if any(indicator in answer_lower for indicator in indicators):
                                star_score += 0.25
                        
                        # Length and detail scoring
                        length_score = min(1.0, word_count / 100)
                        
                        # Overall score
                        total_score = ((star_score * 0.6) + (length_score * 0.4)) * 10
                        final_score = max(1, min(10, total_score))
                        
                        # Display results
                        col_score, col_words, col_star = st.columns(3)
                        with col_score:
                            score_class = "score-excellent" if final_score >= 8 else "score-good" if final_score >= 6 else "score-moderate"
                            st.markdown(f'<div class="score-badge {score_class}">{final_score:.1f}/10</div>', unsafe_allow_html=True)
                        with col_words:
                            st.metric("Words", word_count)
                        with col_star:
                            st.metric("STAR Elements", f"{int(star_score * 4)}/4")
                        
                        # Feedback
                        if final_score >= 8:
                            st.success("Excellent response! Good use of STAR method with specific details.")
                        elif final_score >= 6:
                            st.warning("Good response. Try to include more specific examples and outcomes.")
                        else:
                            st.error("Expand your answer using STAR format with more concrete details.")
                    else:
                        st.error("Please provide an answer first.")
            
            with practice_tabs[1]:  # Technical Questions
                st.markdown("#### Technical Interview Questions")
                technical_questions = [
                    "How would you approach debugging a performance issue in an application?",
                    "Explain the difference between synchronous and asynchronous programming.",
                    "How do you ensure code quality in your projects?",
                    "Describe your experience with version control systems like Git.",
                    "How would you design a simple web application architecture?",
                    "What's your approach to testing software?",
                    "How do you stay updated with new technologies?",
                    "Explain the concept of database normalization.",
                    "What are the trade-offs between different data structures?",
                    "How would you handle security in a web application?"
                ]
                
                selected_technical = st.selectbox("Select a technical question:", technical_questions)
                st.markdown(f"**Question:** {selected_technical}")
                
                technical_answer = st.text_area(
                    "Your Technical Answer:",
                    height=150,
                    placeholder="Provide a detailed technical explanation with examples, methodologies, and specific technologies...",
                    key="technical_answer"
                )
                
                if st.button("🔧 Evaluate Technical Response", use_container_width=True):
                    if technical_answer.strip():
                        # Technical evaluation
                        word_count = len(technical_answer.split())
                        answer_lower = technical_answer.lower()
                        
                        # Technical depth indicators
                        tech_terms = ['algorithm', 'data structure', 'performance', 'optimization', 'testing', 
                                    'debugging', 'architecture', 'scalability', 'security', 'database', 
                                    'framework', 'api', 'protocol', 'methodology']
                        
                        tech_score = sum(1 for term in tech_terms if term in answer_lower) / len(tech_terms)
                        
                        # Specific examples
                        example_indicators = ['example', 'instance', 'like', 'such as', 'for example', 'e.g.']
                        has_examples = any(indicator in answer_lower for indicator in example_indicators)
                        
                        # Length and structure
                        length_score = min(1.0, word_count / 80)
                        
                        # Final calculation
                        final_score = ((tech_score * 0.4) + (length_score * 0.3) + (0.3 if has_examples else 0)) * 10
                        final_score = max(1, min(10, final_score))
                        
                        # Display results
                        col_score, col_words, col_terms = st.columns(3)
                        with col_score:
                            score_class = "score-excellent" if final_score >= 8 else "score-good" if final_score >= 6 else "score-moderate"
                            st.markdown(f'<div class="score-badge {score_class}">{final_score:.1f}/10</div>', unsafe_allow_html=True)
                        with col_words:
                            st.metric("Words", word_count)
                        with col_terms:
                            tech_count = sum(1 for term in tech_terms if term in answer_lower)
                            st.metric("Tech Terms", tech_count)
                        
                        # Feedback
                        if final_score >= 8:
                            st.success("Excellent technical response! Good depth and specific examples.")
                        elif final_score >= 6:
                            st.warning("Good technical knowledge. Consider adding more specific examples or methodologies.")
                        else:
                            st.error("Expand with more technical details, examples, and demonstrate deeper understanding.")
                    else:
                        st.error("Please provide an answer first.")
            
            with practice_tabs[2]:  # Coding Challenges
                st.markdown("#### 💻 Coding Challenges")
                
                # Generate coding questions based on skills
                coding_questions = interview_coach.generate_coding_questions(all_skills)
                
                if coding_questions['sql_questions'] or coding_questions['python_questions']:
                    
                    # SQL Section
                    if coding_questions['sql_questions']:
                        st.markdown("##### 🗃️ SQL Challenges")
                        selected_sql = st.selectbox("Select SQL Challenge:", coding_questions['sql_questions'], key="sql_select")
                        
                        st.markdown(f"**SQL Challenge:** {selected_sql}")
                        
                        sql_code = st.text_area(
                            "Write your SQL solution:",
                            height=120,
                            placeholder="-- Write your SQL query here\nSELECT column1, column2\nFROM table_name\nWHERE condition\nORDER BY column1;",
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
                            placeholder="# Write your Python code here\ndef solution():\n    # Your implementation here\n    return result\n\n# Test your function\n# result = solution()\n# print(result)",
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
                    st.markdown("---")
                    st.markdown("##### 💡 Coding Interview Tips")
                    
                    tips_col1, tips_col2 = st.columns(2)
                    
                    with tips_col1:
                        if coding_questions['sql_questions']:
                            st.markdown("**SQL Best Practices:**")
                            st.markdown("""
                            • Use proper SELECT, FROM, WHERE structure
                            • Consider JOINs for multiple tables
                            • Use LIMIT for large datasets
                            • Avoid SELECT * in production
                            • Add proper aliases for readability
                            • Use aggregate functions when needed
                            """)
                    
                    with tips_col2:
                        if coding_questions['python_questions']:
                            st.markdown("**Python Best Practices:**")
                            st.markdown("""
                            • Write clean, readable code
                            • Use descriptive variable names
                            • Include error handling
                            • Consider edge cases
                            • Use Pythonic patterns
                            • Add comments for complex logic
                            """)
                
                else:
                    st.info("🎯 Upload a resume with programming/database skills to see coding challenges!")
                    st.markdown("**Coding challenges available when you have skills in:**")
                    st.markdown("• **SQL:** MySQL, PostgreSQL, SQL Server, etc.")
                    st.markdown("• **Python:** Programming, software development, etc.")
                    
                    if all_skills:
                        st.markdown("**Your detected skills:**")
                        st.write(", ".join(all_skills))
        
        else:
            st.warning("⚠️ Please complete resume analysis first in the 'Resume Analysis' tab.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar with tips
    with st.sidebar:
        st.markdown("### 📋 Quick Tips")
        st.markdown("""
        **Resume Tips:**
        • Include relevant technical skills
        • Use action verbs and quantify achievements
        • Keep format clean and readable
        • Tailor content to job descriptions
        
        **Interview Tips:**
        • Practice the STAR method
        • Prepare specific examples
        • Research the company
        • Ask thoughtful questions
        """)
        
        if SKLEARN_AVAILABLE:
            st.success("✅ Advanced text analysis enabled")
        else:
            st.warning("⚠️ Using simplified analysis mode")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            🤖 AI-powered resume analysis and interview coaching
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()