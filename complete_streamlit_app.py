import streamlit as st
import openai
import requests
import json
import os
from uszipcode import SearchEngine
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import re
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import icalendar
from icalendar import Calendar, Event, vText
import uuid

# Set page configuration
st.set_page_config(
    page_title="AI Job Search Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .job-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .interview-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class JobListing:
    title: str
    company: str
    location: str
    description: str
    requirements: str
    salary_range: Optional[str] = None
    url: Optional[str] = None
    posted_date: Optional[str] = None
    job_type: Optional[str] = None
    source: Optional[str] = None

@dataclass
class ResumeAnalysis:
    ats_score: int
    strengths: List[str]
    weaknesses: List[str]
    keyword_matches: List[str]
    missing_keywords: List[str]
    formatting_issues: List[str]
    recommendations: List[str]

@dataclass
class ApplicationRecord:
    job_id: str
    company: str
    position: str
    applied_date: str
    status: str
    interview_date: Optional[str] = None
    notes: str = ""

@dataclass
class InterviewSchedule:
    id: str
    company: str
    position: str
    interview_date: datetime
    interview_type: str
    interviewer_name: str = ""
    interviewer_email: str = ""
    meeting_link: str = ""
    notes: str = ""

# Initialize session state
if 'job_results' not in st.session_state:
    st.session_state.job_results = []
if 'resume_analysis' not in st.session_state:
    st.session_state.resume_analysis = None
if 'cover_letter' not in st.session_state:
    st.session_state.cover_letter = ""
if 'interview_questions' not in st.session_state:
    st.session_state.interview_questions = {"behavioral": [], "technical": []}
if 'applications' not in st.session_state:
    st.session_state.applications = []
if 'interviews' not in st.session_state:
    st.session_state.interviews = []
if 'email_config' not in st.session_state:
    st.session_state.email_config = {}
if 'practice_history' not in st.session_state:
    st.session_state.practice_history = []
if 'saved_answers' not in st.session_state:
    st.session_state.saved_answers = []

class ZipcodeManager:
    """Handle zipcode validation and location services"""
    
    def __init__(self):
        self.search_engine = SearchEngine()
    
    def validate_zipcode(self, zipcode: str) -> bool:
        """Validate if zipcode exists"""
        try:
            result = self.search_engine.by_zipcode(zipcode)
            return result.zipcode is not None
        except:
            return False
    
    def get_location_info(self, zipcode: str) -> dict:
        """Get detailed location information for zipcode"""
        try:
            result = self.search_engine.by_zipcode(zipcode)
            if result.zipcode:
                return {
                    'zipcode': result.zipcode,
                    'city': result.major_city,
                    'state': result.state,
                    'county': result.county,
                    'lat': result.lat,
                    'lng': result.lng,
                    'population': result.population,
                    'area_code': result.area_code
                }
        except Exception as e:
            st.error(f"Error getting location info: {e}")
        return None
    
    def find_nearby_zipcodes(self, zipcode: str, radius_miles: int = 25) -> List[dict]:
        """Find zipcodes within radius"""
        try:
            center = self.search_engine.by_zipcode(zipcode)
            if not center.zipcode:
                return []
            
            # Search by coordinates and radius
            nearby = self.search_engine.by_coordinates(
                lat=center.lat,
                lng=center.lng,
                radius=radius_miles,
                returns=50  # Max number of results
            )
            
            results = []
            for zip_info in nearby:
                results.append({
                    'zipcode': zip_info.zipcode,
                    'city': zip_info.major_city,
                    'state': zip_info.state,
                    'distance': zip_info.distance_in_miles
                })
            
            return sorted(results, key=lambda x: x.get('distance', 0))
            
        except Exception as e:
            st.error(f"Error finding nearby zipcodes: {e}")
            return []
    
    def zipcode_to_city_state(self, zipcode: str) -> str:
        """Convert zipcode to 'City, State' format"""
        try:
            result = self.search_engine.by_zipcode(zipcode)
            if result.zipcode:
                return f"{result.major_city}, {result.state}"
        except:
            pass
        return zipcode  # Return original if conversion fails

class EmailManager:
    """Handle email sending for job applications"""
    
    def __init__(self):
        self.smtp_server = None
        self.smtp_port = None
        self.email_address = None
        self.email_password = None
    
    def configure_email(self, email_provider: str, email_address: str, email_password: str):
        """Configure email settings based on provider"""
        
        providers = {
            "Gmail": {"server": "smtp.gmail.com", "port": 587},
            "Outlook": {"server": "smtp-mail.outlook.com", "port": 587},
            "Yahoo": {"server": "smtp.mail.yahoo.com", "port": 587},
            "Custom": {"server": "", "port": 587}
        }
        
        if email_provider in providers:
            self.smtp_server = providers[email_provider]["server"]
            self.smtp_port = providers[email_provider]["port"]
            self.email_address = email_address
            self.email_password = email_password
            return True
        return False
    
    def send_application_email(self, recipient_email: str, subject: str, 
                             cover_letter: str, resume_text: str, 
                             applicant_name: str) -> bool:
        """Send job application email with attachments"""
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.email_address
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            # Email body
            body = f"""Dear Hiring Manager,

{cover_letter}

I have attached my resume for your review. I look forward to hearing from you soon.

Best regards,
{applicant_name}

---
This email was sent via AI Job Search Assistant
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Attach resume
            resume_attachment = MimeBase('application', 'octet-stream')
            resume_attachment.set_payload(resume_text.encode())
            encoders.encode_base64(resume_attachment)
            resume_attachment.add_header(
                'Content-Disposition',
                f'attachment; filename= {applicant_name.replace(" ", "_")}_Resume.txt'
            )
            msg.attach(resume_attachment)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_address, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_address, recipient_email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            st.error(f"Email sending failed: {e}")
            return False

class CalendarManager:
    """Handle interview scheduling and calendar integration"""
    
    @staticmethod
    def create_interview_event(interview: InterviewSchedule) -> str:
        """Create iCalendar event for interview"""
        
        cal = Calendar()
        cal.add('prodid', '-//AI Job Search Assistant//Interview//EN')
        cal.add('version', '2.0')
        
        event = Event()
        event.add('uid', interview.id)
        event.add('summary', f'Interview: {interview.position} at {interview.company}')
        event.add('dtstart', interview.interview_date)
        event.add('dtend', interview.interview_date + timedelta(hours=1))
        event.add('description', f"""
Interview Details:
Position: {interview.position}
Company: {interview.company}
Type: {interview.interview_type}
Interviewer: {interview.interviewer_name}
Meeting Link: {interview.meeting_link}
Notes: {interview.notes}

Preparation Tips:
- Review company background
- Prepare STAR method examples
- Test meeting link 15 minutes early
        """)
        
        if interview.meeting_link:
            event.add('location', interview.meeting_link)
        else:
            event.add('location', f'{interview.company} Office')
        
        # Add reminder (30 minutes before)
        alarm = icalendar.Alarm()
        alarm.add('action', 'DISPLAY')
        alarm.add('description', f'Interview Reminder: {interview.position} at {interview.company}')
        alarm.add('trigger', timedelta(minutes=-30))
        event.add_component(alarm)
        
        cal.add_component(event)
        
        return cal.to_ical().decode('utf-8')
    
    @staticmethod
    def create_follow_up_reminders(application: ApplicationRecord) -> str:
        """Create follow-up reminder events"""
        
        cal = Calendar()
        cal.add('prodid', '-//AI Job Search Assistant//Follow-up//EN')
        cal.add('version', '2.0')
        
        # Follow-up reminders at 1 week and 2 weeks
        base_date = datetime.fromisoformat(application.applied_date)
        
        # 1-week follow-up
        event1 = Event()
        event1.add('uid', str(uuid.uuid4()))
        event1.add('summary', f'Follow-up: {application.position} at {application.company}')
        event1.add('dtstart', base_date + timedelta(days=7))
        event1.add('dtend', base_date + timedelta(days=7, hours=1))
        event1.add('description', f'Send follow-up email for {application.position} application')
        cal.add_component(event1)
        
        # 2-week follow-up
        event2 = Event()
        event2.add('uid', str(uuid.uuid4()))
        event2.add('summary', f'Second Follow-up: {application.position} at {application.company}')
        event2.add('dtstart', base_date + timedelta(days=14))
        event2.add('dtend', base_date + timedelta(days=14, hours=1))
        event2.add('description', f'Final follow-up for {application.position} if no response')
        cal.add_component(event2)
        
        return cal.to_ical().decode('utf-8')

class StreamlitJobSearchAssistant:
    """Streamlit-integrated job search assistant with enhanced features"""
    
    def __init__(self):
        self.load_environment()
        self.setup_apis()
        self.zipcode_manager = ZipcodeManager()
    
    def load_environment(self):
        """Load environment variables"""
        env_path = st.sidebar.text_input(
            "📁 Path to .env file", 
            value=r"C:\Users\pjone\OneDrive\Desktop\NewPython\test.env",
            help="Path to your environment file containing API keys"
        )
        
        if env_path and os.path.exists(env_path):
            load_dotenv(env_path)
            st.sidebar.success("✅ Environment file loaded")
            return True
        elif env_path:
            st.sidebar.error("❌ Environment file not found")
            return False
        return False
    
    def setup_apis(self):
        """Setup API clients"""
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.adzuna_app_id = os.getenv('ADZUNA_APP_ID')
        self.adzuna_app_key = os.getenv('ADZUNA_APP_KEY')
        self.rapidapi_key = os.getenv('RAPIDAPI_KEY')
        
        if self.openai_key:
            self.client = openai.OpenAI(api_key=self.openai_key)
            st.sidebar.success("✅ OpenAI API connected")
        else:
            st.sidebar.error("❌ OpenAI API key not found")
    
    def analyze_resume(self, resume_text: str, job_description: str = "") -> ResumeAnalysis:
        """Analyze resume for ATS compatibility"""
        
        if not self.openai_key:
            st.error("OpenAI API key required for resume analysis")
            return None
        
        with st.spinner("🔍 Analyzing resume for ATS compatibility..."):
            prompt = f"""
            As an expert ATS analyzer, analyze this resume and provide detailed feedback.
            
            Resume: {resume_text}
            Job Description: {job_description}
            
            Provide analysis as JSON:
            {{
                "ats_score": 85,
                "strengths": ["specific strengths"],
                "weaknesses": ["areas to improve"],
                "keyword_matches": ["matched keywords"],
                "missing_keywords": ["important missing keywords"],
                "formatting_issues": ["formatting problems"],
                "recommendations": ["actionable improvements"]
            }}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.3
                )
                
                # Clean response and parse JSON
                response_text = response.choices[0].message.content.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:-3]
                
                analysis_data = json.loads(response_text)
                return ResumeAnalysis(**analysis_data)
                
            except Exception as e:
                st.error(f"Error analyzing resume: {e}")
                return None
    
    def improve_resume(self, resume_text: str, job_description: str = "") -> str:
        """Generate improved resume"""
        
        with st.spinner("✨ Improving your resume..."):
            prompt = f"""
            Improve this resume for ATS compatibility and job relevance:
            
            Original Resume: {resume_text}
            Target Job: {job_description}
            
            Return only the improved resume text with:
            1. Better ATS formatting
            2. Optimized keywords
            3. Stronger action verbs
            4. Quantified achievements
            5. Improved structure
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.4
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                st.error(f"Error improving resume: {e}")
                return resume_text
    
    def generate_cover_letter(self, resume_text: str, job_info: Dict, applicant_name: str) -> str:
        """Generate personalized cover letter"""
        
        with st.spinner("📝 Creating your personalized cover letter..."):
            prompt = f"""
            Create a compelling cover letter:
            
            Applicant: {applicant_name}
            Company: {job_info.get('company', 'Target Company')}
            Position: {job_info.get('title', 'Target Position')}
            
            Job Description: {job_info.get('description', '')}
            
            Resume: {resume_text}
            
            Create a professional cover letter (under 400 words) that:
            1. Shows genuine interest
            2. Highlights relevant experience
            3. Addresses key requirements
            4. Has compelling opening/closing
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.6
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                st.error(f"Error generating cover letter: {e}")
                return "Error generating cover letter"
    
    def search_jobs_adzuna(self, job_title: str, location: str, radius: int = 20) -> List[JobListing]:
        """Search Adzuna API for jobs with zipcode support"""
        
        # Convert zipcode to city, state if needed
        if location.isdigit() and len(location) == 5:
            location_info = self.zipcode_manager.get_location_info(location)
            if location_info:
                location = f"{location_info['city']}, {location_info['state']}"
        
        if not self.adzuna_app_id or not self.adzuna_app_key:
            return self.generate_sample_jobs(job_title, location)
        
        try:
            url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
            
            params = {
                'app_id': self.adzuna_app_id,
                'app_key': self.adzuna_app_key,
                'what': job_title,
                'where': location,
                'distance': radius,
                'results_per_page': 20,
                'sort_by': 'relevance'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            jobs = []
            
            for job_data in data.get('results', []):
                salary_min = job_data.get('salary_min')
                salary_max = job_data.get('salary_max')
                salary_range = None
                if salary_min and salary_max:
                    salary_range = f"${salary_min:,.0f} - ${salary_max:,.0f}"
                
                job = JobListing(
                    title=job_data.get('title', ''),
                    company=job_data.get('company', {}).get('display_name', ''),
                    location=job_data.get('location', {}).get('display_name', ''),
                    description=self._clean_html(job_data.get('description', '')),
                    requirements=job_data.get('description', ''),
                    salary_range=salary_range,
                    url=job_data.get('redirect_url', ''),
                    posted_date=job_data.get('created', ''),
                    source="Adzuna"
                )
                jobs.append(job)
            
            return jobs
            
        except Exception as e:
            st.warning(f"Adzuna API error: {e}")
            return self.generate_sample_jobs(job_title, location)
    
    def search_jobs_with_zipcode_expansion(self, job_title: str, location: str, radius: int = 20) -> List[JobListing]:
        """Enhanced job search with zipcode expansion"""
        
        search_locations = [location]
        
        # If location looks like a zipcode, expand search to nearby areas
        if location.isdigit() and len(location) == 5:
            if self.zipcode_manager.validate_zipcode(location):
                # Get primary location name
                location_info = self.zipcode_manager.get_location_info(location)
                if location_info:
                    primary_location = f"{location_info['city']}, {location_info['state']}"
                    search_locations = [primary_location, location]
                    
                    # Add nearby zipcodes for broader search
                    nearby_zips = self.zipcode_manager.find_nearby_zipcodes(location, radius)
                    for nearby in nearby_zips[:5]:  # Limit to top 5 nearby areas
                        nearby_location = f"{nearby['city']}, {nearby['state']}"
                        if nearby_location not in search_locations:
                            search_locations.append(nearby_location)
        
        # Search jobs in all locations
        all_jobs = []
        for search_loc in search_locations:
            jobs = self.search_jobs_adzuna(job_title, search_loc, radius)
            all_jobs.extend(jobs)
        
        # Remove duplicates based on title + company
        seen = set()
        unique_jobs = []
        for job in all_jobs:
            key = (job.title.lower(), job.company.lower())
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        
        return unique_jobs
    
    def _clean_html(self, text: str) -> str:
        """Clean HTML tags from text"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def generate_sample_jobs(self, job_title: str, location: str) -> List[JobListing]:
        """Generate sample jobs when API is not available"""
        
        sample_jobs = [
            JobListing(
                title=f"Senior {job_title}",
                company="Tech Solutions Inc",
                location=location,
                description=f"We are seeking an experienced {job_title} to join our growing team...",
                requirements=f"3+ years experience in {job_title.lower()}, strong technical skills...",
                salary_range="$80,000 - $120,000",
                source="Sample Data"
            ),
            JobListing(
                title=f"Junior {job_title}",
                company="Startup Innovations",
                location=location,
                description=f"Entry-level {job_title} position with growth opportunities...",
                requirements="Bachelor's degree, willingness to learn...",
                salary_range="$50,000 - $70,000",
                source="Sample Data"
            ),
            JobListing(
                title=f"{job_title} Manager",
                company="Global Corp",
                location=location,
                description=f"Lead a team of {job_title.lower()}s in a fast-paced environment...",
                requirements="5+ years management experience, technical background...",
                salary_range="$100,000 - $150,000",
                source="Sample Data"
            )
        ]
        
        return sample_jobs
    
    def generate_interview_questions(self, job_title: str, question_type: str = "behavioral") -> List[str]:
        """Generate interview questions"""
        
        with st.spinner(f"🎯 Generating {question_type} interview questions..."):
            if question_type == "behavioral":
                prompt = f"""
                Generate 10 behavioral interview questions for a {job_title} position.
                Focus on STAR method, leadership, teamwork, problem-solving.
                Return as JSON array of questions only.
                """
            else:
                prompt = f"""
                Generate 10 technical interview questions for a {job_title} position.
                Include coding, system design, and role-specific technical concepts.
                Return as JSON array of questions only.
                """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.6
                )
                
                response_text = response.choices[0].message.content.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:-3]
                
                questions = json.loads(response_text)
                return questions
                
            except Exception as e:
                st.error(f"Error generating questions: {e}")
                return []

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Job Search Assistant</h1>
        <p>Analyze resumes • Generate cover letters • Find jobs • Practice interviews • Send applications • Schedule interviews</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize assistant
    assistant = StreamlitJobSearchAssistant()
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["🏠 Home", "📄 Resume Analysis", "🔍 Job Search", "📝 Cover Letter", 
         "🎯 Interview Prep", "📧 Email & Applications", "📅 Interview Calendar", "📊 Dashboard"]
    )
    
    # API Status Check
    with st.sidebar.expander("🔑 API Status"):
        if assistant.openai_key:
            st.success("✅ OpenAI API")
        else:
            st.error("❌ OpenAI API")
        
        if assistant.adzuna_app_id and assistant.adzuna_app_key:
            st.success("✅ Adzuna API")
        else:
            st.warning("⚠️ Adzuna API not configured")
        
        if assistant.rapidapi_key:
            st.success("✅ RapidAPI")
        else:
            st.warning("⚠️ RapidAPI not configured")
        
        # Zipcode functionality status (using web API)
        try:
            test_zip = ZipcodeManager()
            if test_zip.validate_zipcode("10001"):
                st.success("✅ Zipcode Service (Web API)")
            else:
                st.warning("⚠️ Zipcode Service (Limited)")
        except:
            st.warning("⚠️ Zipcode Service (Offline)")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📄 Resume Analysis":
        show_resume_analysis_page(assistant)
    elif page == "🔍 Job Search":
        show_job_search_page(assistant)
    elif page == "📝 Cover Letter":
        show_cover_letter_page(assistant)
    elif page == "🎯 Interview Prep":
        show_interview_prep_page(assistant)
    elif page == "📧 Email & Applications":
        show_email_applications_page(assistant)
    elif page == "📅 Interview Calendar":
        show_calendar_page(assistant)
    elif page == "📊 Dashboard":
        show_dashboard_page()

def show_home_page():
    """Enhanced home page with progress tracking"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📄 Resume Analysis</h3>
            <p>Get your ATS score and improvement suggestions</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Analyze Resume", key="home_resume"):
            st.session_state.current_page = "📄 Resume Analysis"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Job Search</h3>
            <p>Find opportunities near you with AI-powered matching</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Search Jobs", key="home_jobs"):
            st.session_state.current_page = "🔍 Job Search"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Interview Prep</h3>
            <p>Practice with AI coach and get personalized feedback</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Practice Interview", key="home_interview"):
            st.session_state.current_page = "🎯 Interview Prep"
            st.rerun()
    
    # Progress overview
    st.subheader("📈 Your Job Search Progress")
    create_progress_overview()
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📧 Send Application"):
            st.session_state.current_page = "📧 Email & Applications"
            st.rerun()
    
    with col2:
        if st.button("📅 Schedule Interview"):
            st.session_state.current_page = "📅 Interview Calendar"
            st.rerun()
    
    with col3:
        if st.button("📝 Generate Cover Letter"):
            st.session_state.current_page = "📝 Cover Letter"
            st.rerun()
    
    with col4:
        if st.button("📊 View Analytics"):
            st.session_state.current_page = "📊 Dashboard"
            st.rerun()
    
    # Recent activity
    show_recent_activity()

def create_progress_overview():
    """Create comprehensive progress overview"""
    
    progress_items = [
        ("Resume optimized", bool(st.session_state.resume_analysis)),
        ("Jobs searched", len(st.session_state.job_results) > 0),
        ("Cover letter ready", bool(st.session_state.cover_letter)),
        ("Interview questions prepared", len(st.session_state.interview_questions["behavioral"]) > 0),
        ("Email configured", bool(st.session_state.email_config)),
        ("Applications sent", len(st.session_state.applications) > 0),
        ("Interviews scheduled", len(st.session_state.interviews) > 0)
    ]
    
    completed = sum(1 for _, done in progress_items if done)
    total = len(progress_items)
    progress_percentage = (completed / total) * 100
    
    # Progress bar
    st.progress(progress_percentage / 100)
    st.write(f"**Progress: {completed}/{total} tasks completed ({progress_percentage:.0f}%)**")
    
    # Progress grid
    col1, col2 = st.columns(2)
    for i, (task, done) in enumerate(progress_items):
        col = col1 if i % 2 == 0 else col2
        icon = "✅" if done else "⏳"
        col.write(f"{icon} {task}")

def show_recent_activity():
    """Show recent activity summary"""
    
    st.subheader("📱 Recent Activity")
    
    activities = []
    
    # Recent applications
    if st.session_state.applications:
        for app in st.session_state.applications[-3:]:
            activities.append({
                'date': app.applied_date[:10],
                'action': 'Applied',
                'details': f"{app.position} at {app.company}",
                'status': app.status
            })
    
    # Recent interviews
    if st.session_state.interviews:
        for interview in st.session_state.interviews[-3:]:
            activities.append({
                'date': interview.interview_date.strftime('%Y-%m-%d'),
                'action': 'Interview Scheduled',
                'details': f"{interview.position} at {interview.company}",
                'status': interview.interview_type
            })
    
    if activities:
        # Sort by date
        activities.sort(key=lambda x: x['date'], reverse=True)
        
        for activity in activities[:5]:
            st.write(f"📅 **{activity['date']}** - {activity['action']}: {activity['details']}")
    else:
        st.info("No recent activity. Start by analyzing your resume or searching for jobs!")

def show_resume_analysis_page(assistant):
    """Resume analysis and improvement page"""
    
    st.header("📄 Resume Analysis & Improvement")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Your Resume")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['txt', 'pdf', 'docx'],
            help="Upload your resume in TXT, PDF, or DOCX format"
        )
        
        # Text area for direct input
        resume_text = st.text_area(
            "Or paste your resume text here:",
            height=300,
            placeholder="Paste your resume content here..."
        )
        
        # Job description for targeted analysis
        job_description = st.text_area(
            "Target Job Description (optional):",
            height=150,
            placeholder="Paste the job description you're targeting..."
        )
        
        analyze_button = st.button("🔍 Analyze Resume", type="primary")
    
    with col2:
        st.subheader("Analysis Results")
        
        if analyze_button and (uploaded_file or resume_text):
            # Process uploaded file
            if uploaded_file:
                if uploaded_file.type == "text/plain":
                    resume_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("PDF/DOCX processing requires additional libraries. Using text input for now.")
            
            if resume_text:
                # Perform analysis
                analysis = assistant.analyze_resume(resume_text, job_description)
                
                if analysis:
                    st.session_state.resume_analysis = analysis
                    
                    # Display ATS Score with color coding
                    score = analysis.ats_score
                    if score >= 80:
                        score_color = "green"
                        score_emoji = "🟢"
                    elif score >= 60:
                        score_color = "orange"
                        score_emoji = "🟡"
                    else:
                        score_color = "red"
                        score_emoji = "🔴"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {score_color};">
                        <h2>{score_emoji} ATS Score: {score}/100</h2>
                        <p>Your resume's compatibility with Applicant Tracking Systems</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Strengths
                    with st.expander("✅ Strengths", expanded=True):
                        for strength in analysis.strengths:
                            st.success(f"• {strength}")
                    
                    # Recommendations
                    with st.expander("🚀 Recommendations", expanded=True):
                        for rec in analysis.recommendations:
                            st.info(f"• {rec}")
                    
                    # Keyword Analysis
                    if job_description:
                        col_match, col_missing = st.columns(2)
                        
                        with col_match:
                            st.subheader("✅ Matched Keywords")
                            for keyword in analysis.keyword_matches:
                                st.markdown(f'<span style="background-color: #d4edda; padding: 2px 8px; border-radius: 4px; margin: 2px;">{keyword}</span>', unsafe_allow_html=True)
                        
                        with col_missing:
                            st.subheader("❌ Missing Keywords")
                            for keyword in analysis.missing_keywords:
                                st.markdown(f'<span style="background-color: #f8d7da; padding: 2px 8px; border-radius: 4px; margin: 2px;">{keyword}</span>', unsafe_allow_html=True)
        
        # Generate improved resume
        if st.session_state.resume_analysis and st.button("✨ Generate Improved Resume"):
            improved_resume = assistant.improve_resume(resume_text, job_description)
            
            st.subheader("✨ Improved Resume")
            st.text_area("Your optimized resume:", value=improved_resume, height=400)
            
            # Download button
            st.download_button(
                label="📥 Download Improved Resume",
                data=improved_resume,
                file_name="improved_resume.txt",
                mime="text/plain"
            )

def show_job_search_page(assistant):
    """Enhanced job search page with zipcode support"""
    
    st.header("🔍 AI-Powered Job Search with Location Intelligence")
    
    # Search form with zipcode validation
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            job_title = st.text_input(
                "💼 Job Title",
                value="Software Engineer",
                placeholder="e.g., Software Engineer, Data Scientist, Marketing Manager"
            )
        
        with col2:
            location_input = st.text_input(
                "📍 Location",
                value="10001",
                placeholder="City, State OR Zipcode (e.g., 10001, San Francisco, CA)"
            )
            
            # Validate and show location info if zipcode
            if location_input and location_input.isdigit() and len(location_input) == 5:
                location_info = assistant.zipcode_manager.get_location_info(location_input)
                if location_info:
                    st.success(f"✅ {location_info['city']}, {location_info['state']}")
                    
                    # Show location insights
                    with st.expander("📊 Location Details"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**County:** {location_info['county']}")
                            st.write(f"**Population:** {location_info['population']:,}")
                        with col_b:
                            st.write(f"**Area Code:** {location_info['area_code']}")
                            st.write(f"**Coordinates:** {location_info['lat']:.2f}, {location_info['lng']:.2f}")
                else:
                    st.error("❌ Invalid zipcode")
        
        with col3:
            radius = st.select_slider(
                "🎯 Search Radius",
                options=[10, 20, 30, 50, 100],
                value=25,
                format_func=lambda x: f"{x} miles"
            )
    
    # Show nearby areas if zipcode entered
    if location_input and location_input.isdigit() and len(location_input) == 5:
        if assistant.zipcode_manager.validate_zipcode(location_input):
            with st.expander("🌐 Nearby Areas to Include in Search"):
                nearby_areas = assistant.zipcode_manager.find_nearby_zipcodes(location_input, radius)
                if nearby_areas:
                    for area in nearby_areas[:8]:
                        st.write(f"📍 {area['city']}, {area['state']} ({area['distance']:.1f} miles)")
    
    # Advanced search options
    with st.expander("🔧 Advanced Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            job_type = st.selectbox(
                "Job Type",
                ["All", "Full-time", "Part-time", "Contract", "Temporary", "Internship"]
            )
            
            salary_range = st.selectbox(
                "Salary Range",
                ["Any", "$30k-$50k", "$50k-$75k", "$75k-$100k", "$100k-$150k", "$150k+"]
            )
        
        with col2:
            experience_level = st.selectbox(
                "Experience Level", 
                ["Any", "Entry Level", "Mid Level", "Senior Level", "Executive"]
            )
            
            company_size = st.selectbox(
                "Company Size",
                ["Any", "Startup (1-50)", "Small (51-200)", "Medium (201-1000)", "Large (1000+)"]
            )
    
    # Search button
    if st.button("🔍 Search Jobs", type="primary"):
        if job_title and location_input:
            with st.spinner("🔍 Searching for jobs..."):
                # Use enhanced search with zipcode support
                job_results = assistant.search_jobs_with_zipcode_expansion(job_title, location_input, radius)
                st.session_state.job_results = job_results
                
                if job_results:
                    st.success(f"✅ Found {len(job_results)} jobs!")
                    
                    # Show search area info
                    if location_input.isdigit() and len(location_input) == 5:
                        nearby_areas = assistant.zipcode_manager.find_nearby_zipcodes(location_input, radius)
                        if nearby_areas:
                            st.info(f"🌐 Expanded search to {len(nearby_areas)} areas within {radius} miles")
                else:
                    st.warning("No jobs found. Try adjusting your search criteria.")
        else:
            st.error("Please provide both job title and location")
    
    # Display results
    if st.session_state.job_results:
        st.subheader(f"📋 Job Results ({len(st.session_state.job_results)} found)")
        
        # Filters for results
        with st.expander("🎛️ Filter Results"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                companies = ["All"] + list(set([job.company for job in st.session_state.job_results if job.company]))
                selected_company = st.selectbox("Filter by Company", companies)
            
            with col2:
                locations = ["All"] + list(set([job.location for job in st.session_state.job_results if job.location]))
                selected_location = st.selectbox("Filter by Location", locations)
            
            with col3:
                show_remote = st.checkbox("Show Remote Jobs Only", value=False)
        
        # Apply filters
        filtered_jobs = st.session_state.job_results
        if selected_company != "All":
            filtered_jobs = [job for job in filtered_jobs if job.company == selected_company]
        if selected_location != "All":
            filtered_jobs = [job for job in filtered_jobs if job.location == selected_location]
        if show_remote:
            filtered_jobs = [job for job in filtered_jobs if "remote" in job.location.lower() or "remote" in job.description.lower()]
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by:",
            ["Relevance", "Company A-Z", "Location", "Date Posted"],
            index=0
        )
        
        if sort_by == "Company A-Z":
            filtered_jobs.sort(key=lambda x: x.company.lower())
        elif sort_by == "Location":
            filtered_jobs.sort(key=lambda x: x.location.lower())
        elif sort_by == "Date Posted" and all(job.posted_date for job in filtered_jobs):
            filtered_jobs.sort(key=lambda x: job.posted_date, reverse=True)
        
        # Display jobs
        for i, job in enumerate(filtered_jobs):
            with st.container():
                st.markdown(f"""
                <div class="job-card">
                    <h3>💼 {job.title}</h3>
                    <p><strong>🏢 Company:</strong> {job.company}</p>
                    <p><strong>📍 Location:</strong> {job.location}</p>
                    {f'<p><strong>💰 Salary:</strong> {job.salary_range}</p>' if job.salary_range else ''}
                    {f'<p><strong>📅 Posted:</strong> {job.posted_date[:10] if job.posted_date else "N/A"}</p>' if job.posted_date else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Job description preview
                with st.expander(f"📝 View Details - {job.title}"):
                    st.write(job.description[:500] + "..." if len(job.description) > 500 else job.description)
                    
                    if job.url:
                        st.markdown(f"🔗 [Apply on Company Website]({job.url})")
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button(f"📄 Cover Letter", key=f"cover_{i}"):
                        st.session_state.selected_job = job
                        st.session_state.current_page = "📝 Cover Letter"
                        st.rerun()
                
                with col2:
                    if st.button(f"📧 Quick Apply", key=f"apply_{i}"):
                        st.session_state.selected_job = job
                        st.session_state.current_page = "📧 Email & Applications"
                        st.rerun()
                
                with col3:
                    if st.button(f"📊 Match Analysis", key=f"analyze_{i}"):
                        if st.session_state.resume_analysis:
                            analyze_job_match(job, st.session_state.resume_analysis)
                        else:
                            st.warning("Please analyze your resume first!")
                
                with col4:
                    if st.button(f"💾 Save Job", key=f"save_{i}"):
                        save_job_to_applications(job)
                
                st.divider()

def show_cover_letter_page(assistant):
    """Cover letter generation page"""
    
    st.header("📝 AI Cover Letter Generator")
    
    # Check if job is selected from job search
    selected_job = st.session_state.get('selected_job')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Job Information")
        
        # Job details input
        if selected_job:
            st.success(f"✅ Using job: {selected_job.title} at {selected_job.company}")
            company = selected_job.company
            position = selected_job.title
            job_description = selected_job.description
        else:
            company = st.text_input("🏢 Company Name", placeholder="e.g., Google, Microsoft, Amazon")
            position = st.text_input("💼 Position Title", placeholder="e.g., Software Engineer")
            job_description = st.text_area(
                "📋 Job Description", 
                height=200,
                placeholder="Paste the job description here..."
            )
        
        # Applicant information
        st.subheader("Your Information")
        applicant_name = st.text_input("👤 Your Name", placeholder="John Doe")
        
        # Resume text
        resume_text = st.text_area(
            "📄 Your Resume", 
            height=250,
            placeholder="Paste your resume content here..."
        )
        
        # Cover letter style
        cover_letter_style = st.selectbox(
            "✍️ Writing Style",
            ["Professional", "Enthusiastic", "Conservative", "Creative", "Technical"]
        )
        
        generate_button = st.button("📝 Generate Cover Letter", type="primary")
    
    with col2:
        st.subheader("Generated Cover Letter")
        
        if generate_button and company and position and applicant_name:
            if not resume_text:
                st.warning("Please provide your resume for better personalization")
            
            job_info = {
                'company': company,
                'title': position,
                'description': job_description
            }
            
            # Generate cover letter
            cover_letter = assistant.generate_cover_letter(resume_text, job_info, applicant_name)
            st.session_state.cover_letter = cover_letter
            
            # Display cover letter
            st.text_area(
                "Your personalized cover letter:",
                value=cover_letter,
                height=400,
                key="cover_letter_display"
            )
            
            # Action buttons
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.download_button(
                    "📥 Download",
                    data=cover_letter,
                    file_name=f"{applicant_name.replace(' ', '_')}_Cover_Letter.txt",
                    mime="text/plain"
                )
            
            with col_b:
                if st.button("📧 Send Application"):
                    st.session_state.current_page = "📧 Email & Applications"
                    st.rerun()
            
            with col_c:
                if st.button("🔄 Regenerate"):
                    st.rerun()

def show_interview_prep_page(assistant):
    """Interview preparation page with AI coaching"""
    
    st.header("🎯 AI Interview Coach")
    
    tab1, tab2, tab3 = st.tabs(["📝 Question Practice", "🎭 Mock Interview", "📚 Interview Tips"])
    
    with tab1:
        st.subheader("Interview Question Practice")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            job_title = st.text_input(
                "💼 Job Title for Questions",
                value="Software Engineer",
                placeholder="Enter the position you're preparing for"
            )
            
            question_type = st.radio(
                "Question Type:",
                ["Behavioral", "Technical", "Both"]
            )
            
            if st.button("🎯 Generate Questions", type="primary"):
                questions = {}
                
                if question_type in ["Behavioral", "Both"]:
                    behavioral_qs = assistant.generate_interview_questions(job_title, "behavioral")
                    questions["behavioral"] = behavioral_qs
                    st.session_state.interview_questions["behavioral"] = behavioral_qs
                
                if question_type in ["Technical", "Both"]:
                    technical_qs = assistant.generate_interview_questions(job_title, "technical")
                    questions["technical"] = technical_qs
                    st.session_state.interview_questions["technical"] = technical_qs
        
        with col2:
            if st.session_state.interview_questions["behavioral"] or st.session_state.interview_questions["technical"]:
                st.subheader("Practice Questions")
                
                # Behavioral questions
                if st.session_state.interview_questions["behavioral"]:
                    with st.expander("🧠 Behavioral Questions"):
                        for i, question in enumerate(st.session_state.interview_questions["behavioral"]):
                            st.write(f"**Q{i+1}:** {question}")
                            
                            # Answer practice
                            answer_key = f"behavioral_answer_{i}"
                            if answer_key not in st.session_state:
                                st.session_state[answer_key] = ""
                            
                            answer = st.text_area(
                                f"Your answer:",
                                key=answer_key,
                                height=100,
                                placeholder="Practice your STAR method answer here..."
                            )
                            
                            if answer and st.button(f"💡 Get Feedback", key=f"feedback_{i}"):
                                get_answer_feedback(question, answer, assistant)
                
                # Technical questions
                if st.session_state.interview_questions["technical"]:
                    with st.expander("⚙️ Technical Questions"):
                        for i, question in enumerate(st.session_state.interview_questions["technical"]):
                            st.write(f"**Q{i+1}:** {question}")
    
    with tab2:
        st.subheader("🎭 AI Mock Interview")
        st.info("Coming soon: Real-time mock interview with AI feedback!")
        
        # Placeholder for mock interview feature
        mock_interview_setup()
    
    with tab3:
        st.subheader("📚 Interview Preparation Tips")
        show_interview_tips()

def show_email_applications_page(assistant):
    """Email and application management page"""
    
    st.header("📧 Email & Application Manager")
    
    tab1, tab2, tab3 = st.tabs(["⚙️ Email Setup", "📤 Send Applications", "📊 Track Applications"])
    
    with tab1:
        st.subheader("Configure Email Settings")
        setup_email_configuration()
    
    with tab2:
        st.subheader("Send Job Applications")
        send_application_interface(assistant)
    
    with tab3:
        st.subheader("Application Tracking")
        show_application_tracker()

def show_calendar_page(assistant):
    """Interview calendar and scheduling page"""
    
    st.header("📅 Interview Calendar & Scheduling")
    
    tab1, tab2 = st.tabs(["📅 Schedule Interview", "📋 View Calendar"])
    
    with tab1:
        st.subheader("Schedule New Interview")
        schedule_interview_interface()
    
    with tab2:
        st.subheader("Upcoming Interviews")
        show_interview_calendar()

def show_dashboard_page():
    """Analytics and dashboard page"""
    
    st.header("📊 Job Search Analytics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📄 Resumes Analyzed", 
            1 if st.session_state.resume_analysis else 0
        )
    
    with col2:
        st.metric(
            "🔍 Jobs Found", 
            len(st.session_state.job_results)
        )
    
    with col3:
        st.metric(
            "📧 Applications Sent", 
            len(st.session_state.applications)
        )
    
    with col4:
        st.metric(
            "📅 Interviews Scheduled", 
            len(st.session_state.interviews)
        )
    
    # Application status chart
    if st.session_state.applications:
        st.subheader("📈 Application Status Overview")
        
        status_counts = {}
        for app in st.session_state.applications:
            status_counts[app.status] = status_counts.get(app.status, 0) + 1
        
        if status_counts:
            fig = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Application Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Timeline chart
    show_application_timeline()

def analyze_job_match(job: JobListing, resume_analysis: ResumeAnalysis):
    """Analyze how well a job matches the resume"""
    
    st.subheader(f"📊 Job Match Analysis: {job.title}")
    
    # Calculate match score based on keyword overlap
    job_keywords = set(job.description.lower().split())
    resume_keywords = set([kw.lower() for kw in resume_analysis.keyword_matches])
    
    common_keywords = job_keywords.intersection(resume_keywords)
    match_score = min(100, int((len(common_keywords) / max(len(job_keywords), 20)) * 100))
    
    # Display match score
    if match_score >= 75:
        match_color = "green"
        match_emoji = "🟢"
    elif match_score >= 50:
        match_color = "orange"
        match_emoji = "🟡"
    else:
        match_color = "red"
        match_emoji = "🔴"
    
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {match_color};">
        <h2>{match_emoji} Match Score: {match_score}%</h2>
        <p>How well your resume aligns with this job</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show matching keywords
    if common_keywords:
        st.subheader("✅ Matching Skills/Keywords")
        for keyword in list(common_keywords)[:10]:
            st.markdown(f'<span style="background-color: #d4edda; padding: 2px 8px; border-radius: 4px; margin: 2px;">{keyword}</span>', unsafe_allow_html=True)

def save_job_to_applications(job: JobListing):
    """Save job to application tracking"""
    
    application = ApplicationRecord(
        job_id=str(uuid.uuid4()),
        company=job.company,
        position=job.title,
        applied_date=datetime.now().isoformat(),
        status="Interested",
        notes=f"Found via job search - {job.source}"
    )
    
    st.session_state.applications.append(application)
    st.success(f"✅ Saved {job.title} at {job.company} to your applications!")

def get_answer_feedback(question: str, answer: str, assistant):
    """Get AI feedback on interview answer"""
    
    if not answer.strip():
        st.warning("Please provide an answer to get feedback")
        return
    
    with st.spinner("🤖 Getting AI feedback..."):
        prompt = f"""
        Evaluate this interview answer using the STAR method:
        
        Question: {question}
        Answer: {answer}
        
        Provide feedback on:
        1. STAR structure (Situation, Task, Action, Result)
        2. Clarity and conciseness
        3. Specific improvements
        4. Score out of 10
        
        Keep feedback concise and actionable.
        """
        
        try:
            response = assistant.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.4
            )
            
            feedback = response.choices[0].message.content
            st.success("💡 AI Feedback:")
            st.write(feedback)
            
        except Exception as e:
            st.error(f"Error getting feedback: {e}")

def mock_interview_setup():
    """Setup for mock interview feature"""
    
    st.write("🎭 **Mock Interview Setup**")
    
    interview_type = st.selectbox(
        "Interview Type:",
        ["Behavioral", "Technical", "Mixed", "Case Study"]
    )
    
    duration = st.slider(
        "Interview Duration (minutes):",
        min_value=15,
        max_value=60,
        value=30,
        step=15
    )
    
    difficulty = st.select_slider(
        "Difficulty Level:",
        options=["Easy", "Medium", "Hard"],
        value="Medium"
    )
    
    if st.button("🚀 Start Mock Interview"):
        st.info("Mock interview feature coming soon! This will provide real-time AI coaching.")

def setup_email_configuration():
    """Email configuration interface"""
    
    st.write("Configure your email settings to send applications directly from the app.")
    
    email_provider = st.selectbox(
        "📧 Email Provider:",
        ["Gmail", "Outlook", "Yahoo", "Custom"]
    )
    
    email_address = st.text_input(
        "📨 Email Address:",
        placeholder="your.email@example.com"
    )
    
    email_password = st.text_input(
        "🔐 Password/App Password:",
        type="password",
        help="For Gmail, use an App Password, not your regular password"
    )
    
    if st.button("💾 Save Email Configuration"):
        email_manager = EmailManager()
        if email_manager.configure_email(email_provider, email_address, email_password):
            st.session_state.email_config = {
                'provider': email_provider,
                'address': email_address,
                'password': email_password
            }
            st.success("✅ Email configuration saved!")
        else:
            st.error("❌ Failed to configure email")

def send_application_interface(assistant):
    """Interface for sending job applications"""
    
    if not st.session_state.email_config:
        st.warning("⚠️ Please configure email settings first in the 'Email Setup' tab")
        return
    
    st.write("Send professional job applications with AI-generated content")
    
    # Job selection
    selected_job = st.session_state.get('selected_job')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if selected_job:
            st.success(f"✅ Selected: {selected_job.title} at {selected_job.company}")
            company = selected_job.company
            position = selected_job.title
        else:
            company = st.text_input("🏢 Company", placeholder="Company name")
            position = st.text_input("💼 Position", placeholder="Job title")
        
        hr_email = st.text_input(
            "📨 HR/Recruiter Email:",
            placeholder="hr@company.com or recruiter@company.com"
        )
        
        applicant_name = st.text_input("👤 Your Name:", placeholder="Your full name")
    
    with col2:
        # Email subject
        subject = st.text_input(
            "📑 Email Subject:",
            value=f"Application for {position} Position" if position else "Job Application",
            placeholder="Application for [Position] Position"
        )
        
        # Cover letter
        cover_letter_text = st.text_area(
            "📝 Cover Letter:",
            value=st.session_state.cover_letter,
            height=200,
            placeholder="Your cover letter content..."
        )
        
        # Resume text
        resume_text = st.text_area(
            "📄 Resume:",
            height=150,
            placeholder="Your resume content for attachment..."
        )
    
    # Send application
    if st.button("📤 Send Application", type="primary"):
        if all([company, position, hr_email, applicant_name, cover_letter_text, resume_text]):
            email_manager = EmailManager()
            email_manager.configure_email(
                st.session_state.email_config['provider'],
                st.session_state.email_config['address'],
                st.session_state.email_config['password']
            )
            
            with st.spinner("📤 Sending application..."):
                success = email_manager.send_application_email(
                    hr_email, subject, cover_letter_text, resume_text, applicant_name
                )
                
                if success:
                    st.success("✅ Application sent successfully!")
                    
                    # Add to application tracker
                    application = ApplicationRecord(
                        job_id=str(uuid.uuid4()),
                        company=company,
                        position=position,
                        applied_date=datetime.now().isoformat(),
                        status="Applied",
                        notes=f"Sent via email to {hr_email}"
                    )
                    st.session_state.applications.append(application)
                    
                else:
                    st.error("❌ Failed to send application")
        else:
            st.error("Please fill in all required fields")

def show_application_tracker():
    """Application tracking interface"""
    
    if not st.session_state.applications:
        st.info("📝 No applications tracked yet. Start applying to jobs to see them here!")
        return
    
    st.write(f"**Tracking {len(st.session_state.applications)} applications**")
    
    # Add new application manually
    with st.expander("➕ Add Application Manually"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_company = st.text_input("Company", key="new_app_company")
            new_position = st.text_input("Position", key="new_app_position")
            new_status = st.selectbox(
                "Status",
                ["Interested", "Applied", "Phone Screen", "Interview", "Offer", "Rejected", "Withdrawn"],
                key="new_app_status"
            )
        
        with col2:
            new_date = st.date_input("Application Date", key="new_app_date")
            new_notes = st.text_area("Notes", key="new_app_notes", height=100)
        
        if st.button("➕ Add Application"):
            if new_company and new_position:
                application = ApplicationRecord(
                    job_id=str(uuid.uuid4()),
                    company=new_company,
                    position=new_position,
                    applied_date=new_date.isoformat(),
                    status=new_status,
                    notes=new_notes
                )
                st.session_state.applications.append(application)
                st.success("✅ Application added!")
                st.rerun()
    
    # Display applications table
    applications_data = []
    for app in st.session_state.applications:
        applications_data.append({
            "Company": app.company,
            "Position": app.position,
            "Applied Date": app.applied_date[:10],
            "Status": app.status,
            "Notes": app.notes[:50] + "..." if len(app.notes) > 50 else app.notes
        })
    
    if applications_data:
        df = pd.DataFrame(applications_data)
        st.dataframe(df, use_container_width=True)
        
        # Quick status updates
        st.subheader("📝 Quick Status Updates")
        
        for i, app in enumerate(st.session_state.applications):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{app.position}** at {app.company}")
            
            with col2:
                new_status = st.selectbox(
                    "Status",
                    ["Interested", "Applied", "Phone Screen", "Interview", "Offer", "Rejected", "Withdrawn"],
                    index=["Interested", "Applied", "Phone Screen", "Interview", "Offer", "Rejected", "Withdrawn"].index(app.status),
                    key=f"status_update_{i}"
                )
            
            with col3:
                if st.button("Update", key=f"update_app_{i}"):
                    st.session_state.applications[i].status = new_status
                    st.success(f"✅ Updated {app.company} status")
                    st.rerun()

def schedule_interview_interface():
    """Interface for scheduling interviews"""
    
    st.write("Schedule and manage your job interviews")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic interview info
        company = st.text_input("🏢 Company:", placeholder="Company name")
        position = st.text_input("💼 Position:", placeholder="Job title")
        
        interview_date = st.date_input("📅 Interview Date:")
        interview_time = st.time_input("⏰ Interview Time:")
        
        interview_type = st.selectbox(
            "📞 Interview Type:",
            ["Phone Screen", "Video Interview", "In-Person", "Panel Interview", "Technical Interview"]
        )
    
    with col2:
        # Additional details
        interviewer_name = st.text_input("👤 Interviewer Name:", placeholder="Optional")
        interviewer_email = st.text_input("📨 Interviewer Email:", placeholder="Optional")
        meeting_link = st.text_input("🔗 Meeting Link:", placeholder="Zoom/Teams link")
        notes = st.text_area("📝 Notes:", height=100, placeholder="Preparation notes, directions, etc.")
    
    if st.button("📅 Schedule Interview", type="primary"):
        if company and position and interview_date:
            # Combine date and time
            interview_datetime = datetime.combine(interview_date, interview_time)
            
            interview = InterviewSchedule(
                id=str(uuid.uuid4()),
                company=company,
                position=position,
                interview_date=interview_datetime,
                interview_type=interview_type,
                interviewer_name=interviewer_name,
                interviewer_email=interviewer_email,
                meeting_link=meeting_link,
                notes=notes
            )
            
            st.session_state.interviews.append(interview)
            st.success("✅ Interview scheduled successfully!")
            
            # Generate calendar event
            calendar_event = CalendarManager.create_interview_event(interview)
            
            st.download_button(
                "📥 Download Calendar Event",
                data=calendar_event,
                file_name=f"{company}_{position}_interview.ics",
                mime="text/calendar"
            )
        else:
            st.error("Please fill in company, position, and date")

def show_interview_calendar():
    """Display scheduled interviews"""
    
    if not st.session_state.interviews:
        st.info("📅 No interviews scheduled yet")
        return
    
    # Sort interviews by date
    sorted_interviews = sorted(
        st.session_state.interviews, 
        key=lambda x: x.interview_date
    )
    
    now = datetime.now()
    upcoming = [i for i in sorted_interviews if i.interview_date > now]
    past = [i for i in sorted_interviews if i.interview_date <= now]
    
    # Upcoming interviews
    if upcoming:
        st.subheader("🔜 Upcoming Interviews")
        for interview in upcoming:
            with st.container():
                st.markdown(f"""
                <div class="interview-card">
                    <h3>📅 {interview.position} at {interview.company}</h3>
                    <p><strong>📅 Date:</strong> {interview.interview_date.strftime('%Y-%m-%d %I:%M %p')}</p>
                    <p><strong>📞 Type:</strong> {interview.interview_type}</p>
                    {f'<p><strong>👤 Interviewer:</strong> {interview.interviewer_name}</p>' if interview.interviewer_name else ''}
                    {f'<p><strong>🔗 Link:</strong> <a href="{interview.meeting_link}" target="_blank">Join Meeting</a></p>' if interview.meeting_link else ''}
                </div>
                """, unsafe_allow_html=True)
                
                if interview.notes:
                    with st.expander("📝 Notes"):
                        st.write(interview.notes)
    
    # Past interviews
    if past:
        with st.expander(f"📚 Past Interviews ({len(past)})"):
            for interview in past:
                st.write(f"**{interview.position}** at {interview.company} - {interview.interview_date.strftime('%Y-%m-%d')}")

def show_interview_tips():
    """Display interview preparation tips"""
    
    st.markdown("""
    ### 🎯 Interview Preparation Checklist
    
    #### 📚 Research Phase
    - ✅ Research company background, mission, values
    - ✅ Understand the role and requirements
    - ✅ Review recent company news and developments
    - ✅ Know your interviewer's background (LinkedIn)
    
    #### 📝 Content Preparation
    - ✅ Prepare STAR method examples for behavioral questions
    - ✅ Practice technical skills relevant to the role
    - ✅ Prepare thoughtful questions to ask the interviewer
    - ✅ Review your resume and be ready to discuss any item
    
    #### 🎭 Practice & Logistics
    - ✅ Practice answers out loud, not just in your head
    - ✅ Test video conferencing setup (lighting, audio, internet)
    - ✅ Prepare professional attire
    - ✅ Plan your route/travel time for in-person interviews
    
    #### 💡 Pro Tips
    - Use the STAR method: Situation, Task, Action, Result
    - Quantify achievements with specific numbers/metrics
    - Show enthusiasm and cultural fit
    - Ask about next steps and timeline
    """)
    
    # Common questions by category
    with st.expander("❓ Common Interview Questions by Category"):
        
        st.markdown("""
        **🧠 Behavioral Questions:**
        - Tell me about yourself
        - Describe a challenging situation you overcame
        - How do you handle stress and pressure?
        - Give an example of teamwork
        
        **⚙️ Technical Questions:**
        - Walk me through your technical experience
        - How would you approach [specific problem]?
        - What technologies are you most comfortable with?
        - Describe a technical project you're proud of
        
        **🏢 Company/Role Questions:**
        - Why do you want to work here?
        - What interests you about this role?
        - Where do you see yourself in 5 years?
        - What are your salary expectations?
        """)

def show_application_timeline():
    """Show application timeline chart"""
    
    if len(st.session_state.applications) < 2:
        return
    
    st.subheader("📈 Application Timeline")
    
    # Prepare timeline data
    timeline_data = []
    for app in st.session_state.applications:
        timeline_data.append({
            'Date': app.applied_date[:10],
            'Company': app.company,
            'Position': app.position,
            'Status': app.status
        })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Create timeline chart
        fig = px.scatter(
            df, 
            x='Date', 
            y='Company',
            color='Status',
            hover_data=['Position'],
            title="Application Timeline"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def zipcode_location_widget():
    """Reusable zipcode location input widget"""
    
    zipcode_manager = ZipcodeManager()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        location = st.text_input(
            "📍 Enter Location",
            placeholder="Enter zipcode (e.g., 10001) or city, state (e.g., New York, NY)",
            help="You can enter a 5-digit zipcode or city and state"
        )
    
    with col2:
        if st.button("🔍 Validate"):
            if location:
                if location.isdigit() and len(location) == 5:
                    # Validate zipcode
                    if zipcode_manager.validate_zipcode(location):
                        info = zipcode_manager.get_location_info(location)
                        if info:
                            st.success(f"✅ {info['city']}, {info['state']}")
                            st.write(f"📊 Population: {info['population']:,}")
                            return location, f"{info['city']}, {info['state']}"
                    else:
                        st.error("❌ Invalid zipcode")
                        return None, None
                else:
                    # Assume city, state format
                    st.info(f"📍 Using: {location}")
                    return location, location
    
    return location, location

def show_zipcode_insights(zipcode: str):
    """Show demographic and economic insights for a zipcode"""
    
    zipcode_manager = ZipcodeManager()
    info = zipcode_manager.get_location_info(zipcode)
    
    if info:
        st.subheader("📊 Location Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏙️ City", info['city'])
            st.metric("🗺️ State", info['state'])
        
        with col2:
            st.metric("👥 Population", f"{info['population']:,}" if info['population'] else "N/A")
            st.metric("📞 Area Code", info['area_code'] if info['area_code'] else "N/A")
        
        with col3:
            st.metric("📍 Coordinates", f"{info['lat']:.2f}, {info['lng']:.2f}")
        
        # Show nearby areas for job search expansion
        st.subheader("🌐 Nearby Areas for Job Search")
        nearby = zipcode_manager.find_nearby_zipcodes(zipcode, 25)
        
        if nearby:
            nearby_df = pd.DataFrame(nearby[:10])  # Show top 10
            st.dataframe(
                nearby_df,
                column_config={
                    "zipcode": "Zipcode",
                    "city": "City", 
                    "state": "State",
                    "distance": st.column_config.NumberColumn(
                        "Distance (miles)",
                        format="%.1f"
                    )
                },
                hide_index=True
            )

# Helper function for sample job generation
def generate_sample_jobs(job_title: str, location: str) -> List[JobListing]:
    """Generate sample jobs when API is not available"""
    
    import random
    
    companies = ["TechCorp", "InnovateSoft", "DataDyne", "CloudFirst", "StartupHub", "MegaCorp"]
    job_types = ["Full-time", "Contract", "Part-time"]
    
    sample_jobs = []
    
    for i in range(5):
        company = random.choice(companies)
        salary_base = random.randint(60000, 150000)
        salary_range = f"${salary_base:,} - ${salary_base + 30000:,}"
        
        job = JobListing(
            title=f"{job_title}",
            company=company,
            location=location,
            description=f"We are seeking a talented {job_title} to join our dynamic team. This role offers excellent growth opportunities and the chance to work with cutting-edge technology...",
            requirements=f"Bachelor's degree, {random.randint(2, 8)}+ years experience, strong problem-solving skills...",
            salary_range=salary_range,
            job_type=random.choice(job_types),
            posted_date=datetime.now().isoformat(),
            source="Sample Data"
        )
        sample_jobs.append(job)
    
    return sample_jobs

# Page navigation helper
def navigate_to_page(page_name: str):
    """Helper function for page navigation"""
    st.session_state.current_page = page_name
    st.rerun()

# Main application entry point
if __name__ == "__main__":
    main()