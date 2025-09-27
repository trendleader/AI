# MUST BE FIRST - Set page config before any other Streamlit commands
import streamlit as st
st.set_page_config(
    page_title="Resume & Interview Coach Pro",
    page_icon="🤖",
    layout="wide"
)

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
import plotly.express as px
import json
import random
import time
import requests
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional
import hashlib
import os

# Handle dotenv import gracefully
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Handle OpenAI import gracefully
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.error("OpenAI package not available. Please install: pip install openai")

# Initialize OpenAI client
openai_client = None
if OPENAI_AVAILABLE:
    # Try to get API key from multiple sources
    api_key = None
    
    # First try Streamlit secrets (recommended for deployment)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("✅ OpenAI API connected via Streamlit secrets!")
    except:
        pass
    
    # Then try environment variables if dotenv is available
    if not api_key and DOTENV_AVAILABLE:
        # Only try local env file if it exists (for local development)
        env_path = "test.env"  # Remove hard-coded path
        if os.path.exists(env_path):
            load_dotenv(env_path)
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                st.success("✅ OpenAI API connected via environment file!")
    
    # Finally try direct environment variable
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            st.success("✅ OpenAI API connected via environment variable!")
    
    if api_key:
        try:
            openai.api_key = api_key
            openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"OpenAI connection error: {str(e)}")
    else:
        st.warning("⚠️ OpenAI API key not found. Some AI features will use fallback methods.")
else:
    st.error("OpenAI package not installed. AI features will use fallback methods.")

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

# Enhanced CSS with AI agent styling
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
    
    .ai-agent-card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .agent-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .agent-active { background: #10b981; color: white; }
    .agent-idle { background: #6b7280; color: white; }
    .agent-working { background: #f59e0b; color: white; animation: pulse 2s infinite; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .job-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4f46e5;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .cover-letter-preview {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 2rem;
        border-radius: 8px;
        font-family: 'Times New Roman', serif;
        line-height: 1.6;
        margin: 1rem 0;
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
    
    .recommendation-card {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .interview-question-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .coding-challenge-card {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .fit-analysis-card {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

class AIAgent:
    """Base AI Agent class"""
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
        self.status = "idle"
        self.last_activity = None
    
    def set_status(self, status: str):
        self.status = status
        self.last_activity = datetime.now()
    
    def get_status_badge(self):
        status_class = f"agent-{self.status}"
        return f'<span class="agent-status {status_class}">{self.status.upper()}</span>'

class InterviewAgent(AIAgent):
    """Agent for conducting interview practice sessions"""
    
    def __init__(self):
        super().__init__("Interview Coach", "Behavioral & Technical Interview Practice")
        self.behavioral_questions = [
            "Tell me about a time when you had to overcome a significant challenge at work.",
            "Describe a situation where you had to work with a difficult team member.",
            "Give me an example of when you had to learn a new technology quickly.",
            "Tell me about a time when you disagreed with your manager's decision.",
            "Describe a project where you had to meet a tight deadline.",
            "Tell me about a time when you failed and what you learned from it.",
            "Describe a situation where you had to take initiative.",
            "Tell me about a time when you had to explain a complex technical concept to a non-technical person.",
            "Give me an example of when you had to prioritize multiple tasks.",
            "Describe a time when you received constructive criticism."
        ]
        
        self.sql_challenges = [
            {
                "difficulty": "Easy",
                "question": "Write a SQL query to find all employees with a salary greater than $50,000.",
                "schema": "employees(id, name, salary, department)",
                "expected_solution": "SELECT * FROM employees WHERE salary > 50000;"
            },
            {
                "difficulty": "Medium",
                "question": "Write a SQL query to find the second highest salary from the employees table.",
                "schema": "employees(id, name, salary, department)",
                "expected_solution": "SELECT MAX(salary) FROM employees WHERE salary < (SELECT MAX(salary) FROM employees);"
            },
            {
                "difficulty": "Hard",
                "question": "Write a SQL query to find employees whose salary is above the average salary of their department.",
                "schema": "employees(id, name, salary, department)",
                "expected_solution": "SELECT e1.* FROM employees e1 WHERE e1.salary > (SELECT AVG(e2.salary) FROM employees e2 WHERE e2.department = e1.department);"
            }
        ]
        
        self.python_challenges = [
            {
                "difficulty": "Easy",
                "question": "Write a Python function to reverse a string.",
                "expected_solution": "def reverse_string(s):\n    return s[::-1]"
            },
            {
                "difficulty": "Medium",
                "question": "Write a Python function to find the first non-repeating character in a string.",
                "expected_solution": "def first_non_repeating(s):\n    char_count = {}\n    for char in s:\n        char_count[char] = char_count.get(char, 0) + 1\n    for char in s:\n        if char_count[char] == 1:\n            return char\n    return None"
            },
            {
                "difficulty": "Hard",
                "question": "Write a Python function to implement a LRU (Least Recently Used) cache.",
                "expected_solution": "class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = {}\n        self.order = []\n    \n    def get(self, key):\n        if key in self.cache:\n            self.order.remove(key)\n            self.order.append(key)\n            return self.cache[key]\n        return -1\n    \n    def put(self, key, value):\n        if key in self.cache:\n            self.order.remove(key)\n        elif len(self.cache) >= self.capacity:\n            oldest = self.order.pop(0)\n            del self.cache[oldest]\n        self.cache[key] = value\n        self.order.append(key)"
            }
        ]
    
    def generate_behavioral_question(self) -> str:
        """Get a random behavioral interview question"""
        return random.choice(self.behavioral_questions)
    
    def get_coding_challenge(self, challenge_type: str, difficulty: str) -> Dict:
        """Get a coding challenge based on type and difficulty"""
        if challenge_type.lower() == "sql":
            challenges = [c for c in self.sql_challenges if c["difficulty"] == difficulty]
        elif challenge_type.lower() == "python":
            challenges = [c for c in self.python_challenges if c["difficulty"] == difficulty]
        else:
            return {}
        
        return random.choice(challenges) if challenges else {}
    
    def evaluate_behavioral_answer(self, question: str, answer: str) -> Dict:
        """Use OpenAI to evaluate behavioral interview answers"""
        self.set_status("working")
        
        if not openai_client:
            return self._generate_fallback_evaluation(answer)
        
        try:
            prompt = f"""
            As an expert interview coach, evaluate this behavioral interview answer using the STAR method (Situation, Task, Action, Result).

            QUESTION: {question}

            CANDIDATE'S ANSWER: {answer}

            Provide detailed feedback in JSON format:

            {{
                "overall_score": 0-10,
                "star_analysis": {{
                    "situation": {{
                        "present": true/false,
                        "clarity": 0-10,
                        "feedback": "specific feedback"
                    }},
                    "task": {{
                        "present": true/false,
                        "clarity": 0-10,
                        "feedback": "specific feedback"
                    }},
                    "action": {{
                        "present": true/false,
                        "clarity": 0-10,
                        "feedback": "specific feedback"
                    }},
                    "result": {{
                        "present": true/false,
                        "clarity": 0-10,
                        "feedback": "specific feedback"
                    }}
                }},
                "strengths": ["strength1", "strength2"],
                "improvements": ["improvement1", "improvement2"],
                "sample_improved_answer": "example of how to improve the answer"
            }}
            
            Focus on constructive feedback and actionable suggestions.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert interview coach with extensive experience in behavioral interviewing techniques."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            evaluation_text = response.choices[0].message.content
            
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', evaluation_text, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group(1))
            else:
                evaluation = json.loads(evaluation_text)
            
            self.set_status("active")
            return evaluation
            
        except Exception as e:
            st.warning(f"AI evaluation failed: {str(e)}")
            return self._generate_fallback_evaluation(answer)
    
    def _generate_fallback_evaluation(self, answer: str) -> Dict:
        """Fallback evaluation without OpenAI"""
        word_count = len(answer.split())
        
        return {
            "overall_score": 7.0 if word_count > 50 else 5.0,
            "star_analysis": {
                "situation": {"present": True, "clarity": 7, "feedback": "Good context provided"},
                "task": {"present": True, "clarity": 6, "feedback": "Task could be clearer"},
                "action": {"present": True, "clarity": 7, "feedback": "Actions described well"},
                "result": {"present": True, "clarity": 6, "feedback": "Results mentioned but could be more specific"}
            },
            "strengths": ["Good structure", "Relevant example"],
            "improvements": ["Add more specific metrics", "Enhance result description"],
            "sample_improved_answer": "Consider adding specific metrics and quantifiable results to strengthen your answer."
        }

class JobAlignmentAgent(AIAgent):
    """Agent for analyzing job-resume alignment and fit scoring"""
    
    def __init__(self):
        super().__init__("Job Alignment", "Resume-Job Description Fit Analysis")
    
    def analyze_job_fit(self, resume_text: str, job_description: str) -> Dict:
        """Analyze how well resume aligns with job description"""
        self.set_status("working")
        
        if not openai_client:
            return self._generate_fallback_analysis(resume_text, job_description)
        
        try:
            prompt = f"""
            As an expert recruiter and hiring manager, analyze the fit between this resume and job description.

            RESUME:
            {resume_text[:2000]}

            JOB DESCRIPTION:
            {job_description[:2000]}

            Provide comprehensive analysis in JSON format:

            {{
                "overall_fit_score": 0-100,
                "category_scores": {{
                    "skills_match": 0-100,
                    "experience_match": 0-100,
                    "education_match": 0-100,
                    "responsibility_match": 0-100
                }},
                "strengths": [
                    {{
                        "area": "area_name",
                        "details": "specific details",
                        "impact": "High/Medium/Low"
                    }}
                ],
                "gaps": [
                    {{
                        "area": "area_name",
                        "missing_skill": "specific skill",
                        "importance": "High/Medium/Low",
                        "suggestion": "how to address"
                    }}
                ],
                "matched_keywords": ["keyword1", "keyword2"],
                "missing_keywords": ["keyword1", "keyword2"],
                "recommendation": {{
                    "should_apply": true/false,
                    "confidence_level": "High/Medium/Low",
                    "reasoning": "detailed reasoning",
                    "application_strategy": "specific strategy"
                }},
                "salary_fit": {{
                    "estimated_range": "range if mentioned",
                    "candidate_level": "entry/mid/senior",
                    "market_competitiveness": "assessment"
                }}
            }}

            Be thorough and provide actionable insights.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert recruiter with 15+ years of experience in talent acquisition and job-candidate matching."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(1))
            else:
                analysis = json.loads(analysis_text)
            
            self.set_status("active")
            return analysis
            
        except Exception as e:
            st.warning(f"AI job fit analysis failed: {str(e)}")
            return self._generate_fallback_analysis(resume_text, job_description)
    
    def _generate_fallback_analysis(self, resume_text: str, job_description: str) -> Dict:
        """Fallback analysis without OpenAI"""
        time.sleep(1)
        
        # Simple keyword matching for fallback
        resume_words = set(resume_text.lower().split())
        job_words = set(job_description.lower().split())
        
        common_words = resume_words.intersection(job_words)
        similarity = len(common_words) / len(job_words) if job_words else 0
        
        fit_score = min(100, similarity * 100 + random.uniform(10, 30))
        
        return {
            "overall_fit_score": round(fit_score, 1),
            "category_scores": {
                "skills_match": round(fit_score + random.uniform(-10, 10), 1),
                "experience_match": round(fit_score + random.uniform(-15, 15), 1),
                "education_match": round(fit_score + random.uniform(-5, 5), 1),
                "responsibility_match": round(fit_score + random.uniform(-10, 10), 1)
            },
            "strengths": [
                {"area": "Technical Skills", "details": "Good technical background", "impact": "High"},
                {"area": "Experience", "details": "Relevant work experience", "impact": "Medium"}
            ],
            "gaps": [
                {"area": "Specific Skills", "missing_skill": "Domain expertise", "importance": "Medium", "suggestion": "Consider highlighting related experience"}
            ],
            "matched_keywords": list(common_words)[:10],
            "missing_keywords": ["leadership", "agile", "collaboration"],
            "recommendation": {
                "should_apply": fit_score > 60,
                "confidence_level": "Medium",
                "reasoning": f"Fit score of {fit_score:.1f}% suggests moderate alignment",
                "application_strategy": "Tailor resume to highlight relevant experience"
            },
            "salary_fit": {
                "estimated_range": "Competitive",
                "candidate_level": "mid",
                "market_competitiveness": "Good match"
            }
        }

class CoverLetterAgent(AIAgent):
    """Agent for generating personalized cover letters using OpenAI GPT"""
    
    def __init__(self):
        super().__init__("Cover Letter", "GPT-Powered Cover Letter Generation")
        self.templates = {
            'professional': 'formal and professional tone, focusing on qualifications and experience',
            'creative': 'engaging and creative tone with storytelling elements',
            'technical': 'technical and detailed tone emphasizing technical skills and projects'
        }
    
    def generate_cover_letter(self, resume_data: Dict, job_data: Dict, style: str = 'professional') -> Dict:
        """Generate a personalized cover letter using OpenAI GPT"""
        self.set_status("working")
        
        if not openai_client:
            return self._generate_fallback_cover_letter(resume_data, job_data, style)
        
        try:
            resume_text = resume_data.get('text', '')[:2000]
            job_description = job_data.get('description', '')[:1500]
            company_name = job_data.get('company', 'the company')
            position = job_data.get('title', 'this position')
            
            skills = self._extract_top_skills(resume_data.get('skills', {}))
            
            prompt = f"""
            Create a personalized cover letter with a {self.templates[style]}. 

            RESUME SUMMARY:
            {resume_text}

            JOB DESCRIPTION:
            {job_description}

            KEY REQUIREMENTS:
            - Company: {company_name}
            - Position: {position}
            - Style: {style}
            - Relevant skills to highlight: {', '.join(skills)}

            INSTRUCTIONS:
            1. Write a complete, professional cover letter
            2. Include proper header placeholders
            3. Highlight relevant experience and skills that match the job
            4. Show enthusiasm for the specific company and role
            5. Keep it concise (3-4 paragraphs)
            6. Use a {self.templates[style]}
            7. End with a strong call to action

            Generate the cover letter now:
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert career coach and professional writer specializing in creating compelling cover letters that get results."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            cover_letter_content = response.choices[0].message.content
            
            self.set_status("active")
            
            return {
                'content': cover_letter_content,
                'style': style,
                'word_count': len(cover_letter_content.split()),
                'generated_at': datetime.now().isoformat(),
                'personalization_score': self._calculate_ai_personalization_score(resume_data, job_data, cover_letter_content),
                'ai_powered': True
            }
            
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            return self._generate_fallback_cover_letter(resume_data, job_data, style)
    
    def _calculate_ai_personalization_score(self, resume_data: Dict, job_data: Dict, cover_letter: str) -> float:
        """Use GPT to evaluate cover letter personalization"""
        try:
            evaluation_prompt = f"""
            Evaluate this cover letter's personalization on a scale of 1-10:

            COVER LETTER:
            {cover_letter[:1000]}

            JOB DESCRIPTION:
            {job_data.get('description', '')[:500]}

            Rate the personalization (1-10) based on:
            1. How well it matches the job requirements
            2. Specific company/role mentions
            3. Relevant skill highlighting
            4. Overall customization level

            Respond with just a number between 1-10:
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            score = float(response.choices[0].message.content.strip())
            return min(10.0, max(1.0, score))
            
        except:
            return 7.5
    
    def _generate_fallback_cover_letter(self, resume_data: Dict, job_data: Dict, style: str) -> Dict:
        """Fallback cover letter generation without OpenAI"""
        time.sleep(1)
        
        skills = self._extract_top_skills(resume_data.get('skills', {}))
        company_name = job_data.get('company', 'the company')
        position = job_data.get('title', 'this position')
        
        cover_letter = f"""[Your Name]
[Your Address]
[Your Email]
[Your Phone]

{datetime.now().strftime('%B %d, %Y')}

[Hiring Manager's Name]
{company_name}
[Company Address]

Dear Hiring Manager,

I am writing to express my strong interest in the {position} role at {company_name}. With my background in {', '.join(skills[:2])}, I am confident that I would be a valuable addition to your team.

My experience with {', '.join(skills)} aligns perfectly with your requirements. I am particularly drawn to {company_name}'s innovative approach and would bring both technical excellence and collaborative skills to your team.

I would welcome the opportunity to discuss how my skills and enthusiasm can contribute to {company_name}'s continued success. Thank you for considering my application.

Sincerely,
[Your Name]"""
        
        return {
            'content': cover_letter,
            'style': style,
            'word_count': len(cover_letter.split()),
            'generated_at': datetime.now().isoformat(),
            'personalization_score': 6.5,
            'ai_powered': False
        }
    
    def _extract_top_skills(self, skills_dict: Dict) -> List[str]:
        """Extract top skills from resume data"""
        top_skills = []
        for category, skill_list in skills_dict.items():
            top_skills.extend(skill_list[:3])
        return top_skills[:6]

class JobSearchAgent(AIAgent):
    """Agent for targeted job search by region"""
    
    def __init__(self):
        super().__init__("Job Search", "Regional Job Discovery & Matching")
        self.job_sources = ['indeed', 'linkedin', 'glassdoor', 'dice']
        self.location_cache = {}
    
    def search_jobs(self, query: str, location: str, radius: int = 25) -> Dict:
        """Search for jobs in specified location"""
        self.set_status("working")
        
        time.sleep(3)
        
        jobs = self._generate_job_listings(query, location, radius)
        
        self.set_status("active")
        
        return {
            'jobs': jobs,
            'total_found': len(jobs),
            'search_params': {
                'query': query,
                'location': location,
                'radius': radius,
                'timestamp': datetime.now().isoformat()
            },
            'location_data': self._get_location_insights(location)
        }
    
    def _generate_job_listings(self, query: str, location: str, radius: int) -> List[Dict]:
        """Generate realistic job listings"""
        companies = ['TechCorp', 'InnovateLabs', 'DataDriven Inc', 'CloudFirst', 'AI Solutions', 
                    'DevCorp', 'StartupXYZ', 'Enterprise Systems', 'FinTech Plus', 'HealthTech']
        
        job_titles = ['Software Engineer', 'Data Scientist', 'DevOps Engineer', 'Product Manager',
                     'Frontend Developer', 'Backend Developer', 'ML Engineer', 'Systems Analyst']
        
        jobs = []
        for i in range(random.randint(15, 30)):
            salary_base = random.randint(70000, 150000)
            jobs.append({
                'id': f"job_{i+1}",
                'title': random.choice(job_titles),
                'company': random.choice(companies),
                'location': location,
                'salary_min': salary_base,
                'salary_max': salary_base + random.randint(10000, 30000),
                'description': f"We are seeking a qualified {random.choice(job_titles)} to join our dynamic team...",
                'skills': random.sample(['Python', 'SQL', 'AWS', 'Docker', 'React', 'Java', 'Git'], 
                                      random.randint(3, 5)),
                'posted_date': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                'match_score': random.uniform(0.6, 0.95),
                'remote_option': random.choice([True, False, 'Hybrid']),
                'experience_level': random.choice(['Entry', 'Mid', 'Senior'])
            })
        
        jobs.sort(key=lambda x: x['match_score'], reverse=True)
        return jobs
    
    def _get_location_insights(self, location: str) -> Dict:
        """Get insights about the job market in the location"""
        return {
            'avg_salary': random.randint(80000, 120000),
            'job_growth': random.uniform(0.05, 0.25),
            'cost_of_living_index': random.uniform(0.8, 1.5),
            'tech_companies_count': random.randint(50, 500),
            'market_competitiveness': random.choice(['Low', 'Medium', 'High']),
            'top_skills_demand': ['Python', 'AWS', 'React', 'SQL', 'Docker']
        }
    
    def filter_jobs(self, jobs: List[Dict], filters: Dict) -> List[Dict]:
        """Filter jobs based on criteria"""
        filtered_jobs = jobs.copy()
        
        if filters.get('min_salary'):
            filtered_jobs = [j for j in filtered_jobs if j['salary_min'] >= filters['min_salary']]
        
        if filters.get('experience_level'):
            filtered_jobs = [j for j in filtered_jobs if j['experience_level'] == filters['experience_level']]
        
        if filters.get('remote_only'):
            filtered_jobs = [j for j in filtered_jobs if j['remote_option'] in [True, 'Hybrid']]
        
        if filters.get('required_skills'):
            required_skills = set(filters['required_skills'])
            filtered_jobs = [j for j in filtered_jobs 
                           if len(required_skills.intersection(set(j['skills']))) > 0]
        
        return filtered_jobs

class RecommendationAgent(AIAgent):
    """Agent for personalized career recommendations using OpenAI GPT"""
    
    def __init__(self):
        super().__init__("Recommendation", "GPT-Powered Career Guidance")
        self.recommendation_types = ['skill_development', 'career_path', 'job_match', 'salary_negotiation']
    
    def generate_recommendations(self, profile_data: Dict) -> Dict:
        """Generate personalized recommendations using OpenAI GPT"""
        self.set_status("working")
        
        if not openai_client:
            return self._generate_fallback_recommendations(profile_data)
        
        try:
            resume_text = profile_data.get('text', '')[:1500]
            skills = profile_data.get('skills', {})
            score = profile_data.get('score', 5)
            experience_level = profile_data.get('experience_level', 'mid')
            
            skills_summary = []
            for category, skill_list in skills.items():
                if skill_list:
                    skills_summary.append(f"{category}: {', '.join(skill_list)}")
            
            prompt = f"""
            As an expert career coach, analyze this professional profile and provide comprehensive career recommendations.

            PROFILE SUMMARY:
            - Resume Quality Score: {score}/10
            - Experience Level: {experience_level}
            - Skills: {'; '.join(skills_summary)}

            RESUME EXCERPT:
            {resume_text}

            Please provide detailed recommendations in JSON format with these categories:

            {{
                "skill_development": [
                    {{
                        "category": "category_name",
                        "skills": ["skill1", "skill2"],
                        "priority": "High/Medium/Low",
                        "learning_time": "timeframe",
                        "impact": "description of career impact"
                    }}
                ],
                "career_path": [
                    {{
                        "path": "role_title",
                        "timeline": "timeframe",
                        "requirements": ["requirement1", "requirement2"],
                        "salary_potential": "salary_range"
                    }}
                ],
                "job_applications": [
                    {{
                        "type": "strategy_name",
                        "priority": "High/Medium/Low",
                        "action": "specific_action",
                        "details": "explanation"
                    }}
                ],
                "profile_optimization": [
                    {{
                        "area": "improvement_area",
                        "suggestion": "specific_suggestion",
                        "impact": "High/Medium/Low"
                    }}
                ],
                "priority_score": {{
                    "resume_update": 0-10,
                    "skill_development": 0-10,
                    "job_applications": 0-10,
                    "networking": 0-10,
                    "interview_prep": 0-10
                }}
            }}

            Focus on actionable, specific advice based on current market trends and the individual's profile.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert career coach with deep knowledge of technology careers, market trends, and professional development. Provide specific, actionable recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.7
            )
            
            recommendations_text = response.choices[0].message.content
            
            try:
                json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', recommendations_text, re.DOTALL)
                if json_match:
                    recommendations = json.loads(json_match.group(1))
                else:
                    recommendations = json.loads(recommendations_text)
                
                self.set_status("active")
                return recommendations
                
            except json.JSONDecodeError:
                st.warning("GPT response parsing failed, using structured fallback")
                return self._generate_fallback_recommendations(profile_data)
            
        except Exception as e:
            st.error(f"OpenAI API Error in recommendations: {str(e)}")
            return self._generate_fallback_recommendations(profile_data)
    
    def _generate_fallback_recommendations(self, profile_data: Dict) -> Dict:
        """Fallback recommendations without OpenAI"""
        time.sleep(1)
        
        recommendations = {
            'skill_development': [
                {
                    'category': 'Programming',
                    'skills': ['Python', 'JavaScript'],
                    'priority': 'High',
                    'learning_time': '3-6 months',
                    'impact': 'Significant salary increase potential'
                }
            ],
            'career_path': [
                {
                    'path': 'Senior Software Engineer',
                    'timeline': '2-3 years',
                    'requirements': ['Advanced programming skills', 'Leadership experience'],
                    'salary_potential': '$120K - $160K'
                }
            ],
            'job_applications': [
                {
                    'type': 'Resume Optimization',
                    'priority': 'High',
                    'action': 'Add more quantified achievements',
                    'details': 'Include specific metrics and results'
                }
            ],
            'profile_optimization': [
                {
                    'area': 'Skills Section',
                    'suggestion': 'Add more relevant technical skills',
                    'impact': 'High'
                }
            ],
            'priority_score': {
                'resume_update': 7.5,
                'skill_development': 8.0,
                'job_applications': 6.5,
                'networking': 6.0,
                'interview_prep': 7.0
            }
        }
        
        return recommendations

class ResumeJobAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask'],
            'data_science': ['pandas', 'numpy', 'sql', 'tableau', 'power bi', 'tensorflow', 'pytorch'],
            'cloud': ['aws', 'azure', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'trello']
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
    
    def ai_analyze_resume(self, resume_text: str) -> Dict:
        """Use OpenAI GPT to analyze resume quality and provide insights"""
        if not openai_client:
            return {}
        
        try:
            prompt = f"""
            As an expert resume reviewer and career coach, analyze this resume comprehensively:

            RESUME:
            {resume_text[:2000]}

            Provide detailed analysis in JSON format:

            {{
                "overall_quality_score": 0-10,
                "strengths": ["strength1", "strength2", "strength3"],
                "weaknesses": ["weakness1", "weakness2", "weakness3"],
                "missing_elements": ["element1", "element2"],
                "experience_assessment": {{
                    "level": "entry/mid/senior",
                    "years_estimated": "X-Y years",
                    "leadership_indicators": true/false
                }},
                "improvement_priority": [
                    {{
                        "area": "improvement_area",
                        "priority": "High/Medium/Low",
                        "suggestion": "specific_suggestion"
                    }}
                ]
            }}

            Focus on actionable feedback and market-relevant insights.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert resume reviewer with 20+ years of experience in talent acquisition and career coaching."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                return json.loads(analysis_text)
                
        except Exception as e:
            st.warning(f"AI resume analysis failed: {str(e)}")
            return {}
    
    def analyze_fit(self, resume_text, job_text=""):
        resume_skills = self.extract_skills(resume_text)
        word_count = len(resume_text.split())
        
        ai_analysis = self.ai_analyze_resume(resume_text)
        
        base_score = ai_analysis.get('overall_quality_score', 5.0)
        
        if any(resume_skills.values()):
            base_score += 1.0
        if word_count > 200:
            base_score += 0.5
        if word_count > 500:
            base_score += 0.5
        if '@' in resume_text:
            base_score += 0.5
            
        if job_text:
            job_skills = self.extract_skills(job_text)
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
        
        result = {
            'score': round(scaled_score, 1),
            'resume_skills': resume_skills,
            'job_skills': job_skills,
            'recommendation': self.get_recommendation(scaled_score),
            'text': resume_text,
            'experience_level': self._determine_experience_level(resume_text),
            'ai_analysis': ai_analysis
        }
        
        return result
    
    def _determine_experience_level(self, text):
        """Determine experience level from resume text"""
        text_lower = text.lower()
        
        senior_indicators = ['senior', 'lead', 'manager', 'director', 'principal', '10+', 'years']
        entry_indicators = ['graduate', 'junior', 'entry', 'recent', 'internship', 'new grad']
        
        senior_count = sum(1 for indicator in senior_indicators if indicator in text_lower)
        entry_count = sum(1 for indicator in entry_indicators if indicator in text_lower)
        
        if senior_count > entry_count and senior_count > 1:
            return 'senior'
        elif entry_count > 0:
            return 'entry'
        else:
            return 'mid'
    
    def get_recommendation(self, score):
        if score >= 8:
            return "🟢 Excellent fit! Please apply - you meet most requirements."
        elif score >= 6:
            return "🟡 Good fit! Consider applying - you have many relevant qualifications."
        elif score >= 4:
            return "🟠 Moderate fit. Consider highlighting transferable skills."
        else:
            return "🔴 Focus on skill development for better alignment."

def create_agent_status_display(agents: List[AIAgent]):
    """Display AI agents status"""
    st.markdown("### Please select from one of the options below")
    
    cols = st.columns(len(agents))
    for idx, agent in enumerate(agents):
        with cols[idx]:
            st.markdown(f"""
                <div class="ai-agent-card">
                    <h4 style="margin: 0 0 0.5rem 0;">{agent.name}</h4>
                    <p style="margin: 0 0 1rem 0; font-size: 0.9rem; opacity: 0.9;">{agent.specialty}</p>
                    {agent.get_status_badge()}
                </div>
            """, unsafe_allow_html=True)

def display_recommendations(recommendations: Dict):
    """Display AI recommendations"""
    st.markdown("### AI-Powered Recommendations")
    
    priority_scores = recommendations.get('priority_score', {})
    if priority_scores:
        st.markdown("#### Priority Actions")
        sorted_priorities = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        
        cols = st.columns(len(sorted_priorities))
        for idx, (action, score) in enumerate(sorted_priorities):
            with cols[idx]:
                color = "#10b981" if score >= 8 else "#f59e0b" if score >= 6 else "#ef4444"
                st.markdown(f"""
                    <div style="background: {color}; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                        <div style="font-size: 1.5rem; font-weight: 700;">{score:.1f}</div>
                        <div style="font-size: 0.9rem;">{action.replace('_', ' ').title()}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    skill_recs = recommendations.get('skill_development', [])
    if skill_recs:
        st.markdown("#### Skill Development")
        for rec in skill_recs:
            st.markdown(f"""
                <div class="recommendation-card">
                    <h5 style="margin: 0 0 0.5rem 0;">{rec['category'].replace('_', ' ').title()} Skills</h5>
                    <p style="margin: 0 0 0.5rem 0;"><strong>Skills to Learn:</strong> {', '.join(rec['skills'])}</p>
                    <p style="margin: 0 0 0.5rem 0;"><strong>Priority:</strong> {rec['priority']} • <strong>Time:</strong> {rec['learning_time']}</p>
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">{rec['impact']}</p>
                </div>
            """, unsafe_allow_html=True)

def display_job_fit_analysis(fit_analysis: Dict):
    """Display job fit analysis results"""
    st.markdown("### Job Fit Analysis")
    
    # Overall fit score
    fit_score = fit_analysis.get('overall_fit_score', 0)
    color = "#10b981" if fit_score >= 80 else "#f59e0b" if fit_score >= 60 else "#ef4444"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div style="background: {color}; color: white; padding: 2rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 2.5rem; font-weight: 700;">{fit_score:.0f}%</div>
                <div style="font-size: 1rem;">Overall Fit Score</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        recommendation = fit_analysis.get('recommendation', {})
        should_apply = recommendation.get('should_apply', False)
        apply_text = "APPLY" if should_apply else "RECONSIDER"
        apply_color = "#10b981" if should_apply else "#ef4444"
        
        st.markdown(f"""
            <div style="background: {apply_color}; color: white; padding: 2rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700;">{apply_text}</div>
                <div style="font-size: 0.9rem;">{recommendation.get('confidence_level', 'Medium')} Confidence</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        candidate_level = fit_analysis.get('salary_fit', {}).get('candidate_level', 'mid')
        st.markdown(f"""
            <div style="background: #6366f1; color: white; padding: 2rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700;">{candidate_level.upper()}</div>
                <div style="font-size: 0.9rem;">Experience Level</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Category scores
    category_scores = fit_analysis.get('category_scores', {})
    if category_scores:
        st.markdown("#### Category Breakdown")
        
        categories = list(category_scores.keys())
        scores = list(category_scores.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=scores,
                marker_color=['#10b981' if s >= 80 else '#f59e0b' if s >= 60 else '#ef4444' for s in scores],
                text=[f"{s:.1f}%" for s in scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Fit Score by Category",
            xaxis_title="Categories",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Strengths and gaps
    col1, col2 = st.columns(2)
    
    with col1:
        strengths = fit_analysis.get('strengths', [])
        if strengths:
            st.markdown("#### Strengths")
            for strength in strengths:
                impact_color = "#10b981" if strength.get('impact') == 'High' else "#f59e0b" if strength.get('impact') == 'Medium' else "#6b7280"
                st.markdown(f"""
                    <div style="background: {impact_color}; color: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                        <strong>{strength.get('area', 'N/A')}</strong><br>
                        <small>{strength.get('details', 'N/A')}</small>
                    </div>
                """, unsafe_allow_html=True)
    
    with col2:
        gaps = fit_analysis.get('gaps', [])
        if gaps:
            st.markdown("#### Areas for Improvement")
            for gap in gaps:
                importance_color = "#ef4444" if gap.get('importance') == 'High' else "#f59e0b" if gap.get('importance') == 'Medium' else "#6b7280"
                st.markdown(f"""
                    <div style="background: {importance_color}; color: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                        <strong>{gap.get('area', 'N/A')}</strong><br>
                        <small>Missing: {gap.get('missing_skill', 'N/A')}</small><br>
                        <small>Suggestion: {gap.get('suggestion', 'N/A')}</small>
                    </div>
                """, unsafe_allow_html=True)
    
    # Recommendation details
    recommendation = fit_analysis.get('recommendation', {})
    if recommendation:
        st.markdown("#### AI Recommendation")
        st.markdown(f"""
            <div class="fit-analysis-card">
                <h5 style="margin: 0 0 1rem 0;">Application Strategy</h5>
                <p style="margin: 0 0 1rem 0;"><strong>Reasoning:</strong> {recommendation.get('reasoning', 'N/A')}</p>
                <p style="margin: 0;"><strong>Strategy:</strong> {recommendation.get('application_strategy', 'N/A')}</p>
            </div>
        """, unsafe_allow_html=True)

def display_interview_practice():
    """Display interview practice interface"""
    st.markdown("### Interview Practice")
    
    practice_type = st.selectbox(
        "Choose Practice Type",
        ["Behavioral Questions", "SQL Challenges", "Python Challenges"]
    )
    
    if practice_type == "Behavioral Questions":
        if st.button("Get New Behavioral Question", type="primary"):
            question = st.session_state.ai_agents['interview'].generate_behavioral_question()
            st.session_state.current_behavioral_question = question
        
        if 'current_behavioral_question' in st.session_state:
            st.markdown(f"""
                <div class="interview-question-card">
                    <h5 style="margin: 0 0 1rem 0;">Behavioral Question</h5>
                    <p style="margin: 0; font-size: 1.1rem;">{st.session_state.current_behavioral_question}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Your Answer")
            st.markdown("*Use the STAR method: Situation, Task, Action, Result*")
            
            answer = st.text_area(
                "Type your answer here:",
                height=200,
                placeholder="Describe the situation, task, actions you took, and the results..."
            )
            
            if st.button("Get AI Feedback") and answer.strip():
                with st.spinner("AI is evaluating your answer..."):
                    evaluation = st.session_state.ai_agents['interview'].evaluate_behavioral_answer(
                        st.session_state.current_behavioral_question, answer
                    )
                    st.session_state.behavioral_evaluation = evaluation
            
            if 'behavioral_evaluation' in st.session_state:
                eval_data = st.session_state.behavioral_evaluation
                
                # Overall score
                score = eval_data.get('overall_score', 0)
                score_color = "#10b981" if score >= 8 else "#f59e0b" if score >= 6 else "#ef4444"
                
                st.markdown(f"""
                    <div style="background: {score_color}; color: white; padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;">
                        <div style="font-size: 2rem; font-weight: 700;">{score:.1f}/10</div>
                        <div>Overall Score</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # STAR analysis
                star_analysis = eval_data.get('star_analysis', {})
                if star_analysis:
                    st.markdown("#### STAR Method Analysis")
                    
                    star_cols = st.columns(4)
                    star_components = ['situation', 'task', 'action', 'result']
                    
                    for idx, component in enumerate(star_components):
                        with star_cols[idx]:
                            comp_data = star_analysis.get(component, {})
                            present = comp_data.get('present', False)
                            clarity = comp_data.get('clarity', 0)
                            
                            status_color = "#10b981" if present and clarity >= 7 else "#f59e0b" if present else "#ef4444"
                            
                            st.markdown(f"""
                                <div style="background: {status_color}; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                                    <div style="font-weight: 700;">{component.upper()}</div>
                                    <div style="font-size: 0.8rem;">{'✓' if present else '✗'} | {clarity}/10</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.caption(comp_data.get('feedback', ''))
                
                # Strengths and improvements
                col1, col2 = st.columns(2)
                
                with col1:
                    strengths = eval_data.get('strengths', [])
                    if strengths:
                        st.markdown("#### Strengths")
                        for strength in strengths:
                            st.success(f"• {strength}")
                
                with col2:
                    improvements = eval_data.get('improvements', [])
                    if improvements:
                        st.markdown("#### Areas for Improvement")
                        for improvement in improvements:
                            st.warning(f"• {improvement}")
                
                # Sample improved answer
                sample_answer = eval_data.get('sample_improved_answer', '')
                if sample_answer:
                    st.markdown("#### Improvement Suggestion")
                    st.info(sample_answer)
    
    elif practice_type in ["SQL Challenges", "Python Challenges"]:
        challenge_type = "sql" if practice_type == "SQL Challenges" else "python"
        
        difficulty = st.selectbox("Choose Difficulty", ["Easy", "Medium", "Hard"])
        
        if st.button(f"Get New {challenge_type.upper()} Challenge", type="primary"):
            challenge = st.session_state.ai_agents['interview'].get_coding_challenge(challenge_type, difficulty)
            st.session_state.current_coding_challenge = challenge
        
        if 'current_coding_challenge' in st.session_state:
            challenge = st.session_state.current_coding_challenge
            
            st.markdown(f"""
                <div class="coding-challenge-card">
                    <h5 style="margin: 0 0 1rem 0;">{challenge_type.upper()} Challenge - {challenge.get('difficulty', 'Unknown')}</h5>
                    <p style="margin: 0; font-size: 1.1rem;">{challenge.get('question', 'No question available')}</p>
                    {f'<p style="margin: 1rem 0 0 0; font-size: 0.9rem;"><strong>Schema:</strong> {challenge.get("schema", "")}</p>' if challenge.get("schema") else ''}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Your Solution")
            solution = st.text_area(
                f"Write your {challenge_type.upper()} code:",
                height=200,
                placeholder=f"Enter your {challenge_type.upper()} solution here..."
            )
            
            if st.button("Show Expected Solution"):
                expected = challenge.get('expected_solution', 'No solution available')
                st.markdown("#### Expected Solution")
                st.code(expected, language=challenge_type)

def main():
    load_css()
    
    st.markdown("""
        <div class="main-header">
            <h1 style="color: #1f2937; font-size: 2.5rem; margin-bottom: 0.5rem;">
                Resume & Interview Coach Pro
            </h1>
            <p style="color: #6b7280; font-size: 1.1rem;">
                Powered by Data2Trend
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize AI agents
    if 'ai_agents' not in st.session_state:
        st.session_state.ai_agents = {
            'cover_letter': CoverLetterAgent(),
            'job_search': JobSearchAgent(),
            'recommendation': RecommendationAgent(),
            'interview': InterviewAgent(),
            'job_alignment': JobAlignmentAgent()
        }
    
    analyzer = ResumeJobAnalyzer()
    
    # Initialize session state
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""
    if 'job_description_text' not in st.session_state:
        st.session_state.job_description_text = ""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'job_search_results' not in st.session_state:
        st.session_state.job_search_results = None
    if 'cover_letters' not in st.session_state:
        st.session_state.cover_letters = []
    if 'job_fit_analysis' not in st.session_state:
        st.session_state.job_fit_analysis = None
    
    agents_list = list(st.session_state.ai_agents.values())
    create_agent_status_display(agents_list)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Upload Files", 
        "Resume Analysis", 
        "Job Fit Analysis",
        "Interview Practice", 
        "Job Search", 
        "Cover Letters", 
        "AI Recommendations"
    ])
    
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("## Upload Resume & Job Description")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Upload Resume File")
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
                st.success(f"Resume uploaded: {resume_file.name}")
                
                preview_text = resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
                st.text_area("Resume Preview", value=preview_text, height=100, disabled=True)
            
            st.markdown("#### Or Paste Resume Text")
            resume_text_input = st.text_area(
                "Resume content",
                height=200,
                placeholder="Paste your resume content here..."
            )
            
            if resume_text_input:
                st.session_state.resume_text = resume_text_input
        
        with col2:
            st.markdown("#### Upload Job Description")
            job_file = st.file_uploader(
                "Choose job description file (optional)",
                type=['pdf', 'txt'],
                help="Upload a PDF or TXT file containing the job description"
            )
            
            if job_file:
                if job_file.type == "application/pdf":
                    job_text = analyzer.extract_text_from_pdf(job_file)
                else:
                    job_text = str(job_file.read(), "utf-8")
                
                st.session_state.job_description_text = job_text
                st.success(f"Job description uploaded: {job_file.name}")
                
                preview_text = job_text[:200] + "..." if len(job_text) > 200 else job_text
                st.text_area("Job Description Preview", value=preview_text, height=100, disabled=True)
            
            st.markdown("#### Or Paste Job Description")
            job_text_input = st.text_area(
                "Job description content",
                height=200,
                placeholder="Paste the job description here for better analysis..."
            )
            
            if job_text_input:
                st.session_state.job_description_text = job_text_input
        
        if st.session_state.resume_text:
            if st.button("Analyze Resume with AI", type="primary", use_container_width=True):
                with st.spinner("AI is analyzing your resume..."):
                    st.session_state.ai_agents['recommendation'].set_status("working")
                    
                    # Analyze resume with or without job description
                    results = analyzer.analyze_fit(
                        st.session_state.resume_text, 
                        st.session_state.job_description_text
                    )
                    st.session_state.analysis_results = results
                    
                    # Generate job fit analysis if job description is provided
                    if st.session_state.job_description_text:
                        st.session_state.ai_agents['job_alignment'].set_status("working")
                        job_fit = st.session_state.ai_agents['job_alignment'].analyze_job_fit(
                            st.session_state.resume_text,
                            st.session_state.job_description_text
                        )
                        st.session_state.job_fit_analysis = job_fit
                    
                    # Generate recommendations
                    recommendations = st.session_state.ai_agents['recommendation'].generate_recommendations(results)
                    st.session_state.recommendations = recommendations
                    
                    st.success("AI analysis complete! Check all tabs for insights.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            st.markdown("## Resume Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                score_class = "score-excellent" if results['score'] >= 8 else "score-good" if results['score'] >= 6 else "score-moderate"
                st.markdown(f'<div class="score-badge {score_class}">{results["score"]}/10</div>', unsafe_allow_html=True)
                st.caption("Overall Score")
            
            with col2:
                skill_count = sum(len(skills) for skills in results['resume_skills'].values())
                st.metric("Skills Found", skill_count)
            
            with col3:
                word_count = len(st.session_state.resume_text.split())
                st.metric("Word Count", word_count)
            
            with col4:
                exp_level = results.get('experience_level', 'Mid').title()
                st.metric("Experience Level", exp_level)
            
            st.markdown("### AI Assessment")
            st.info(results['recommendation'])
            
            # Display AI analysis insights if available
            ai_analysis = results.get('ai_analysis', {})
            if ai_analysis:
                st.markdown("### Detailed AI Insights")
                
                col1, col2 = st.columns(2)
                with col1:
                    strengths = ai_analysis.get('strengths', [])
                    if strengths:
                        st.markdown("#### Strengths")
                        for strength in strengths:
                            st.success(f"• {strength}")
                
                with col2:
                    weaknesses = ai_analysis.get('weaknesses', [])
                    if weaknesses:
                        st.markdown("#### Areas for Improvement")
                        for weakness in weaknesses:
                            st.warning(f"• {weakness}")
                
                missing_elements = ai_analysis.get('missing_elements', [])
                if missing_elements:
                    st.markdown("#### Missing Elements")
                    for element in missing_elements:
                        st.error(f"• {element}")
            
            if results['resume_skills']:
                st.markdown("### Your Skills Portfolio")
                
                skills_data = []
                for category, skills in results['resume_skills'].items():
                    for skill in skills:
                        skills_data.append({'Category': category.replace('_', ' ').title(), 'Skill': skill})
                
                if skills_data:
                    df = pd.DataFrame(skills_data)
                    fig = px.treemap(df, path=['Category', 'Skill'], 
                                   title="Your Skills Breakdown",
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Please upload and analyze your resume first.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if st.session_state.job_fit_analysis:
            display_job_fit_analysis(st.session_state.job_fit_analysis)
        elif st.session_state.job_description_text and st.session_state.resume_text:
            st.info("Job description detected! Generate job fit analysis by running resume analysis.")
            if st.button("Generate Job Fit Analysis", type="primary"):
                with st.spinner("AI is analyzing job fit..."):
                    job_fit = st.session_state.ai_agents['job_alignment'].analyze_job_fit(
                        st.session_state.resume_text,
                        st.session_state.job_description_text
                    )
                    st.session_state.job_fit_analysis = job_fit
                    st.rerun()
        else:
            st.warning("Please upload both your resume and a job description to analyze fit.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        display_interview_practice()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("## AI-Powered Job Search")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Search Parameters")
            
            job_query = st.text_input(
                "Job Title/Keywords",
                placeholder="e.g., Software Engineer, Data Scientist"
            )
            
            location = st.text_input(
                "Location (City, State or ZIP)",
                placeholder="e.g., San Francisco, CA or 94105"
            )
            
            search_radius = st.slider("Search Radius (miles)", 5, 100, 25)
            
            if st.button("Search Jobs with AI", type="primary", use_container_width=True):
                if job_query and location:
                    with st.spinner("AI is searching jobs for you..."):
                        st.session_state.ai_agents['job_search'].set_status("working")
                        
                        search_results = st.session_state.ai_agents['job_search'].search_jobs(
                            job_query, location, search_radius
                        )
                        
                        st.session_state.job_search_results = search_results
                        
                        st.success(f"Found {len(search_results['jobs'])} matching jobs!")
                else:
                    st.error("Please enter both job title and location.")
        
        with col2:
            if st.session_state.job_search_results:
                results = st.session_state.job_search_results
                
                st.markdown(f"### Job Results ({results['total_found']} found)")
                
                for job in results['jobs'][:10]:
                    salary_range = f"${job['salary_min']:,} - ${job['salary_max']:,}"
                    
                    st.markdown(f"""
                        <div class="job-card">
                            <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">{job['title']}</h4>
                            <p style="margin: 0 0 0.5rem 0; color: #6b7280; font-weight: 600;">{job['company']} • {job['location']}</p>
                            <p style="margin: 0; color: #059669; font-weight: 600;">{salary_range}</p>
                            <div style="margin: 1rem 0;">
                                <strong>Skills:</strong> {', '.join(job['skills'])}
                            </div>
                            <div style="font-size: 0.8rem; color: #6b7280;">
                                Match: {job['match_score']:.1%} • Posted: {datetime.fromisoformat(job['posted_date']).strftime('%B %d, %Y')}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.info("Use the search form to find jobs with AI assistance!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("## AI Cover Letter Generator")
        
        if st.session_state.analysis_results:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Letter Configuration")
                
                company_name = st.text_input("Company Name", placeholder="e.g., Google")
                position_title = st.text_input("Position Title", placeholder="e.g., Software Engineer")
                
                # Use uploaded job description if available
                job_description = st.text_area(
                    "Job Description (Optional)",
                    value=st.session_state.job_description_text,
                    height=150,
                    placeholder="Paste job description for better personalization..."
                )
                
                cover_letter_style = st.selectbox(
                    "Cover Letter Style",
                    ["professional", "creative", "technical"],
                    format_func=lambda x: x.title()
                )
                
                if st.button("Generate Cover Letter", type="primary", use_container_width=True):
                    if company_name and position_title:
                        with st.spinner("AI is crafting your personalized cover letter..."):
                            job_data = {
                                'company': company_name,
                                'title': position_title,
                                'description': job_description,
                                'skills': analyzer.extract_skills(job_description) if job_description else {}
                            }
                            
                            cover_letter_result = st.session_state.ai_agents['cover_letter'].generate_cover_letter(
                                st.session_state.analysis_results,
                                job_data,
                                cover_letter_style
                            )
                            
                            cover_letter_result['job_data'] = job_data
                            st.session_state.cover_letters.append(cover_letter_result)
                            
                            st.success("Cover letter generated successfully!")
                    else:
                        st.error("Please enter company name and position title.")
            
            with col2:
                st.markdown("### Generated Cover Letters")
                
                if st.session_state.cover_letters:
                    latest_letter = st.session_state.cover_letters[-1]
                    
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("Words", latest_letter['word_count'])
                    with metric_cols[1]:
                        st.metric("Style", latest_letter['style'].title())
                    with metric_cols[2]:
                        st.metric("Personalization", f"{latest_letter['personalization_score']:.1f}/10")
                    
                    st.markdown("#### Preview")
                    st.markdown(f"""
                        <div class="cover-letter-preview">
                            {latest_letter['content'].replace(chr(10), '<br>')}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        "Download as TXT",
                        latest_letter['content'],
                        file_name=f"cover_letter_{latest_letter['job_data']['company']}.txt",
                        mime="text/plain"
                    )
                
                else:
                    st.info("Configure and generate your first AI cover letter!")
        
        else:
            st.warning("Please upload and analyze your resume first.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab7:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if 'recommendations' in st.session_state:
            display_recommendations(st.session_state.recommendations)
        elif st.session_state.analysis_results:
            st.info("Generate AI recommendations by analyzing your resume first!")
            if st.button("Generate AI Recommendations", type="primary"):
                with st.spinner("AI is analyzing your profile for recommendations..."):
                    recommendations = st.session_state.ai_agents['recommendation'].generate_recommendations(
                        st.session_state.analysis_results
                    )
                    st.session_state.recommendations = recommendations
                    st.rerun()
        else:
            st.warning("Please upload and analyze your resume first.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            Enhanced By Data2Trend
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()