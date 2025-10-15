import streamlit as st
import json
import requests
from datetime import datetime
import io
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Page configuration
st.set_page_config(
    page_title="Resume-Job Matcher & Application Suite",
    page_icon="📄",
    layout="wide"
)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        import docx
        doc = docx.Document(io.BytesIO(docx_file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def call_claude_api(prompt, max_tokens=4000):
    """Call Claude API for analysis - No API key needed!"""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['content'][0]['text']
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error calling API: {str(e)}"

def search_jobs_online(job_title, location, num_results=10):
    """Search for jobs using web search"""
    search_query = f"{job_title} jobs {location} site:linkedin.com OR site:indeed.com OR site:glassdoor.com"
    
    # Using a simple approach - in production, you'd use proper APIs
    st.info(f"🔍 Searching for: {search_query}")
    
    # Placeholder for actual job search results
    # In a real implementation, you would integrate with:
    # - JSearch API (RapidAPI)
    # - Adzuna API
    # - Google Jobs API
    # - SerpAPI for job search results
    
    return {
        "search_query": search_query,
        "message": "Manual job search recommended. Copy URLs from LinkedIn, Indeed, or Glassdoor and paste in 'Add Jobs' tab.",
        "api_recommendations": [
            "JSearch API (RapidAPI) - Free tier available",
            "Adzuna API - Job board aggregator",
            "Google Custom Search API - Can search job sites",
            "SerpAPI - Google Jobs search results"
        ]
    }

def analyze_job_match(resume_text, job_description):
    """Analyze how well the resume matches the job description"""
    prompt = f"""
    Analyze the alignment between this resume and job description. Provide a match score from 0-100 and detailed reasoning.

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}

    Respond ONLY with a valid JSON object in this exact format:
    {{
        "match_score": <number between 0-100>,
        "key_alignments": ["alignment 1", "alignment 2", "alignment 3"],
        "gaps": ["gap 1", "gap 2"],
        "reasoning": "Brief explanation of the score"
    }}

    DO NOT include any text outside the JSON structure, including backticks or markdown formatting.
    """
    
    response = call_claude_api(prompt)
    
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        
        return json.loads(cleaned)
    except:
        return {
            "match_score": 0,
            "key_alignments": ["Error parsing response"],
            "gaps": [],
            "reasoning": "Could not parse match analysis"
        }

def tailor_resume(resume_text, job_description, job_title, company_name):
    """Generate a tailored resume for the specific job"""
    prompt = f"""
    Create a tailored version of this resume for the following job. Keep all truthful information but reorder, emphasize, and phrase things to best match the job requirements.

    ORIGINAL RESUME:
    {resume_text}

    JOB TITLE: {job_title}
    COMPANY: {company_name}
    
    JOB DESCRIPTION:
    {job_description}

    Create a tailored resume that:
    1. Keeps the same structure and format
    2. Emphasizes relevant skills and experiences
    3. Uses keywords from the job description naturally
    4. Maintains all factual information
    5. Reorders bullet points to highlight most relevant experience first

    Return the complete tailored resume in plain text format.
    """
    
    return call_claude_api(prompt, max_tokens=6000)

def generate_cover_letter(resume_text, job_description, job_title, company_name):
    """Generate a cover letter for the specific job"""
    prompt = f"""
    Write a compelling, professional cover letter for this job application.

    CANDIDATE RESUME:
    {resume_text}

    JOB TITLE: {job_title}
    COMPANY: {company_name}
    
    JOB DESCRIPTION:
    {job_description}

    Write a cover letter that:
    1. Is 3-4 paragraphs long
    2. Shows genuine enthusiasm for the role and company
    3. Highlights 2-3 most relevant achievements from the resume
    4. Explains why the candidate is a great fit
    5. Has a professional but warm tone
    6. Includes proper formatting with date and addresses

    Return the complete cover letter.
    """
    
    return call_claude_api(prompt, max_tokens=3000)

def generate_interview_questions(resume_text, job_description, job_title, company_name):
    """Generate likely interview questions and suggested answers"""
    prompt = f"""
    Generate likely interview questions for this job and provide suggested answers based on the candidate's background.

    CANDIDATE RESUME:
    {resume_text}

    JOB TITLE: {job_title}
    COMPANY: {company_name}
    
    JOB DESCRIPTION:
    {job_description}

    Respond ONLY with a valid JSON object in this format:
    {{
        "technical_questions": [
            {{"question": "...", "suggested_answer": "..."}},
            {{"question": "...", "suggested_answer": "..."}}
        ],
        "behavioral_questions": [
            {{"question": "...", "suggested_answer": "..."}},
            {{"question": "...", "suggested_answer": "..."}}
        ],
        "company_specific_questions": [
            {{"question": "...", "suggested_answer": "..."}}
        ]
    }}

    DO NOT include any text outside the JSON structure.
    """
    
    response = call_claude_api(prompt, max_tokens=4000)
    
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        
        return json.loads(cleaned)
    except:
        return {
            "technical_questions": [],
            "behavioral_questions": [],
            "company_specific_questions": []
        }

def send_application_email(to_email, subject, body, attachments=None, smtp_config=None):
    """Send application email with attachments"""
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['email']
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Add attachments
        if attachments:
            for filename, content in attachments.items():
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(content.encode())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={filename}')
                msg.attach(part)
        
        # Connect to SMTP server
        server = smtplib.SMTP(smtp_config['server'], smtp_config['port'])
        server.starttls()
        server.login(smtp_config['email'], smtp_config['password'])
        
        text = msg.as_string()
        server.sendmail(smtp_config['email'], to_email, text)
        server.quit()
        
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"Error sending email: {str(e)}"

# Initialize session state
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None

if 'jobs' not in st.session_state:
    st.session_state.jobs = []

if 'analyzed_jobs' not in st.session_state:
    st.session_state.analyzed_jobs = []

if 'smtp_config' not in st.session_state:
    st.session_state.smtp_config = None

# Main App UI
st.title("📄 Resume-Job Matcher & Application Suite")
st.write("**Complete Job Application Management System**")
st.divider()

# Sidebar
with st.sidebar:
    st.header("📋 Quick Start")
    st.write("""
    1. **Upload Resume** 
    2. **Search or Add Jobs**
    3. **Analyze Matches**
    4. **Generate Materials**
    5. **Send Applications**
    """)
    
    st.divider()
    
    # Resume Upload Section
    st.subheader("📤 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, DOCX, TXT"
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner("Processing resume..."):
            if file_type == 'pdf':
                st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
            elif file_type == 'docx':
                st.session_state.resume_text = extract_text_from_docx(uploaded_file)
            else:  # txt
                st.session_state.resume_text = uploaded_file.read().decode('utf-8')
        
        st.success("✅ Resume loaded!")
        
        with st.expander("📄 Preview Resume"):
            st.text_area("Resume Content", st.session_state.resume_text[:500] + "...", height=200, disabled=True)
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    match_threshold = st.slider("Minimum Match Score", 0, 100, 80)
    
    st.divider()
    
    # Email Configuration
    st.subheader("📧 Email Setup")
    with st.expander("Configure Email"):
        email = st.text_input("Your Email", type="default")
        password = st.text_input("App Password", type="password", help="Use app-specific password for Gmail")
        smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
        smtp_port = st.number_input("SMTP Port", value=587)
        
        if st.button("💾 Save Email Config"):
            if email and password:
                st.session_state.smtp_config = {
                    'email': email,
                    'password': password,
                    'server': smtp_server,
                    'port': smtp_port
                }
                st.success("✅ Email configured!")
            else:
                st.error("Please provide email and password")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Search Jobs", 
    "➕ Add Jobs", 
    "📊 Analyze Matches", 
    "✨ Generate Materials",
    "📧 Send Applications"
])

with tab1:
    st.header("🔍 Search for Jobs Online")
    
    if not st.session_state.resume_text:
        st.warning("⚠️ Please upload your resume in the sidebar first!")
    else:
        st.info("🚀 **Pro Tip**: For best results, manually copy job descriptions from LinkedIn, Indeed, or Glassdoor to the 'Add Jobs' tab.")
        
        col1, col2 = st.columns(2)
        with col1:
            search_job_title = st.text_input("Job Title", placeholder="e.g., Analytics Engineer")
        with col2:
            search_location = st.text_input("Location", placeholder="e.g., New York, NY")
        
        if st.button("🔍 Search Jobs", type="primary"):
            if search_job_title:
                results = search_jobs_online(search_job_title, search_location)
                
                st.write("**Recommended Job Search APIs:**")
                for api in results['api_recommendations']:
                    st.write(f"• {api}")
                
                st.divider()
                st.write("**Manual Search Links:**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    linkedin_url = f"https://www.linkedin.com/jobs/search/?keywords={search_job_title.replace(' ', '%20')}&location={search_location.replace(' ', '%20')}"
                    st.link_button("🔗 LinkedIn", linkedin_url)
                
                with col2:
                    indeed_url = f"https://www.indeed.com/jobs?q={search_job_title.replace(' ', '+')}&l={search_location.replace(' ', '+')}"
                    st.link_button("🔗 Indeed", indeed_url)
                
                with col3:
                    glassdoor_url = f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={search_job_title.replace(' ', '%20')}&locT=C&locId={search_location.replace(' ', '%20')}"
                    st.link_button("🔗 Glassdoor", glassdoor_url)
                
                st.info("💡 Copy job descriptions from these sites and paste them in the 'Add Jobs' tab")
            else:
                st.error("Please enter a job title")

with tab2:
    st.header("➕ Add Job Descriptions")
    
    if not st.session_state.resume_text:
        st.warning("⚠️ Please upload your resume in the sidebar first!")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Analytics Engineer")
            company_name = st.text_input("Company Name", placeholder="e.g., Tech Corp")
        
        with col2:
            job_url = st.text_input("Job URL (optional)", placeholder="https://...")
            job_location = st.text_input("Location (optional)", placeholder="e.g., New York, NY")
        
        job_description = st.text_area(
            "Job Description",
            placeholder="Paste the full job description here...",
            height=300
        )
        
        recruiter_email = st.text_input("Recruiter/HR Email (optional)", placeholder="recruiter@company.com")
        
        if st.button("➕ Add Job", type="primary"):
            if job_title and company_name and job_description:
                new_job = {
                    "id": len(st.session_state.jobs) + 1,
                    "title": job_title,
                    "company": company_name,
                    "description": job_description,
                    "url": job_url,
                    "location": job_location,
                    "recruiter_email": recruiter_email,
                    "added_date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.jobs.append(new_job)
                st.success(f"✅ Added: {job_title} at {company_name}")
                st.rerun()
            else:
                st.error("Please fill in Job Title, Company Name, and Job Description")
        
        st.divider()
        
        if st.session_state.jobs:
            st.subheader(f"📋 Saved Jobs ({len(st.session_state.jobs)})")
            for job in st.session_state.jobs:
                with st.expander(f"{job['title']} - {job['company']}"):
                    st.write(f"**Added:** {job['added_date']}")
                    if job['location']:
                        st.write(f"**Location:** {job['location']}")
                    if job['url']:
                        st.write(f"**URL:** {job['url']}")
                    if job.get('recruiter_email'):
                        st.write(f"**Recruiter Email:** {job['recruiter_email']}")
                    st.text_area("Description Preview", job['description'][:500] + "...", height=100, key=f"preview_{job['id']}", disabled=True)
                    
                    if st.button("🗑️ Remove", key=f"remove_{job['id']}"):
                        st.session_state.jobs = [j for j in st.session_state.jobs if j['id'] != job['id']]
                        st.session_state.analyzed_jobs = [j for j in st.session_state.analyzed_jobs if j['id'] != job['id']]
                        st.rerun()

with tab3:
    st.header("📊 Analyze Job Matches")
    
    if not st.session_state.resume_text:
        st.warning("⚠️ Please upload your resume in the sidebar first!")
    elif not st.session_state.jobs:
        st.info("👈 Add some jobs in the 'Add Jobs' tab first!")
    else:
        st.write(f"**{len(st.session_state.jobs)} job(s) ready to analyze**")
        
        if st.button("🔍 Analyze All Jobs", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            analyzed_jobs = []
            for idx, job in enumerate(st.session_state.jobs):
                status_text.text(f"Analyzing {job['title']} at {job['company']}...")
                
                match_result = analyze_job_match(st.session_state.resume_text, job['description'])
                
                analyzed_job = {**job, **match_result}
                analyzed_jobs.append(analyzed_job)
                
                progress_bar.progress((idx + 1) / len(st.session_state.jobs))
            
            st.session_state.analyzed_jobs = analyzed_jobs
            status_text.text("✅ Analysis complete!")
            st.success(f"Analyzed {len(analyzed_jobs)} jobs!")
            st.rerun()
        
        st.divider()
        
        if st.session_state.analyzed_jobs:
            matching_jobs = [j for j in st.session_state.analyzed_jobs if j.get('match_score', 0) >= match_threshold]
            
            st.subheader(f"✅ Matching Jobs ({len(matching_jobs)} with {match_threshold}%+ match)")
            
            if matching_jobs:
                matching_jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
                
                for job in matching_jobs:
                    score = job.get('match_score', 0)
                    
                    if score >= 90:
                        score_color = "🟢"
                    elif score >= 80:
                        score_color = "🟡"
                    else:
                        score_color = "🟠"
                    
                    with st.expander(f"{score_color} {score}% - {job['title']} at {job['company']}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.metric("Match Score", f"{score}%")
                            st.write("**Reasoning:**")
                            st.write(job.get('reasoning', 'N/A'))
                        
                        with col2:
                            if job.get('location'):
                                st.write(f"📍 {job['location']}")
                            if job.get('url'):
                                st.link_button("🔗 View Job", job['url'])
                        
                        st.write("**✅ Key Alignments:**")
                        for alignment in job.get('key_alignments', []):
                            st.write(f"• {alignment}")
                        
                        if job.get('gaps'):
                            st.write("**⚠️ Gaps to Address:**")
                            for gap in job.get('gaps', []):
                                st.write(f"• {gap}")
            else:
                st.warning(f"No jobs found with {match_threshold}%+ match. Try lowering the threshold in the sidebar.")
            
            low_scoring = [j for j in st.session_state.analyzed_jobs if j.get('match_score', 0) < match_threshold]
            if low_scoring:
                with st.expander(f"📊 Lower Scoring Jobs ({len(low_scoring)} jobs)"):
                    for job in sorted(low_scoring, key=lambda x: x.get('match_score', 0), reverse=True):
                        st.write(f"🔴 {job.get('match_score', 0)}% - {job['title']} at {job['company']}")

with tab4:
    st.header("✨ Generate Tailored Materials")
    
    if not st.session_state.resume_text:
        st.warning("⚠️ Please upload your resume in the sidebar first!")
    elif not st.session_state.analyzed_jobs:
        st.info("👈 Analyze jobs first in the 'Analyze Matches' tab!")
    else:
        matching_jobs = [j for j in st.session_state.analyzed_jobs if j.get('match_score', 0) >= match_threshold]
        
        if not matching_jobs:
            st.warning(f"No jobs with {match_threshold}%+ match found. Lower the threshold to generate materials.")
        else:
            job_options = [f"{j['title']} at {j['company']} ({j.get('match_score', 0)}%)" for j in matching_jobs]
            selected_job_idx = st.selectbox("Select Job", range(len(matching_jobs)), format_func=lambda x: job_options[x])
            
            selected_job = matching_jobs[selected_job_idx]
            
            # Store generated materials in session state
            if 'generated_materials' not in st.session_state:
                st.session_state.generated_materials = {}
            
            job_key = f"{selected_job['id']}"
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Generate Resume", type="primary", use_container_width=True):
                    with st.spinner("Creating tailored resume..."):
                        resume = tailor_resume(
                            st.session_state.resume_text,
                            selected_job['description'],
                            selected_job['title'],
                            selected_job['company']
                        )
                        
                        if job_key not in st.session_state.generated_materials:
                            st.session_state.generated_materials[job_key] = {}
                        st.session_state.generated_materials[job_key]['resume'] = resume
                        st.success("✅ Resume generated!")
            
            with col2:
                if st.button("✉️ Generate Cover Letter", type="primary", use_container_width=True):
                    with st.spinner("Writing cover letter..."):
                        letter = generate_cover_letter(
                            st.session_state.resume_text,
                            selected_job['description'],
                            selected_job['title'],
                            selected_job['company']
                        )
                        
                        if job_key not in st.session_state.generated_materials:
                            st.session_state.generated_materials[job_key] = {}
                        st.session_state.generated_materials[job_key]['cover_letter'] = letter
                        st.success("✅ Cover letter generated!")
            
            with col3:
                if st.button("🎤 Generate Interview Prep", type="primary", use_container_width=True):
                    with st.spinner("Preparing interview questions..."):
                        questions = generate_interview_questions(
                            st.session_state.resume_text,
                            selected_job['description'],
                            selected_job['title'],
                            selected_job['company']
                        )
                        
                        if job_key not in st.session_state.generated_materials:
                            st.session_state.generated_materials[job_key] = {}
                        st.session_state.generated_materials[job_key]['interview_prep'] = questions
                        st.success("✅ Interview prep generated!")
            
            st.divider()
            
            # Display generated materials
            if job_key in st.session_state.generated_materials:
                materials = st.session_state.generated_materials[job_key]
                
                if 'resume' in materials:
                    st.subheader("📄 Tailored Resume")
                    st.text_area("Resume", materials['resume'], height=400, key=f"display_resume_{job_key}")
                    st.download_button(
                        "📥 Download Resume",
                        materials['resume'],
                        f"Resume_{selected_job['company']}_{selected_job['title'].replace(' ', '_')}.txt",
                        key=f"dl_resume_{job_key}"
                    )
                
                if 'cover_letter' in materials:
                    st.subheader("✉️ Cover Letter")
                    st.text_area("Letter", materials['cover_letter'], height=400, key=f"display_letter_{job_key}")
                    st.download_button(
                        "📥 Download Cover Letter",
                        materials['cover_letter'],
                        f"CoverLetter_{selected_job['company']}_{selected_job['title'].replace(' ', '_')}.txt",
                        key=f"dl_letter_{job_key}"
                    )
                
                if 'interview_prep' in materials:
                    st.subheader("🎤 Interview Preparation")
                    
                    prep = materials['interview_prep']
                    
                    if prep.get('technical_questions'):
                        st.write("**💻 Technical Questions:**")
                        for i, qa in enumerate(prep['technical_questions'], 1):
                            with st.expander(f"Q{i}: {qa.get('question', '')}"):
                                st.write("**Suggested Answer:**")
                                st.write(qa.get('suggested_answer', ''))
                    
                    if prep.get('behavioral_questions'):
                        st.write("**🧠 Behavioral Questions:**")
                        for i, qa in enumerate(prep['behavioral_questions'], 1):
                            with st.expander(f"Q{i}: {qa.get('question', '')}"):
                                st.write("**Suggested Answer:**")
                                st.write(qa.get('suggested_answer', ''))
                    
                    if prep.get('company_specific_questions'):
                        st.write("**🏢 Company-Specific Questions:**")
                        for i, qa in enumerate(prep['company_specific_questions'], 1):
                            with st.expander(f"Q{i}: {qa.get('question', '')}"):
                                st.write("**Suggested Answer:**")
                                st.write(qa.get('suggested_answer', ''))

with tab5:
    st.header("📧 Send Applications")
    
    if not st.session_state.smtp_config:
        st.warning("⚠️ Please configure your email in the sidebar first!")
    elif not st.session_state.analyzed_jobs:
        st.info("👈 Generate materials first in the 'Generate Materials' tab!")
    else:
        matching_jobs = [j for j in st.session_state.analyzed_jobs if j.get('match_score', 0) >= match_threshold]
        
        if not matching_jobs:
            st.warning(f"No jobs with {match_threshold}%+ match found.")
        else:
            # Filter jobs that have generated materials and recruiter email
            sendable_jobs = [
                j for j in matching_jobs 
                if f"{j['id']}" in st.session_state.get('generated_materials', {}) 
                and j.get('recruiter_email')
            ]
            
            if not sendable_jobs:
                st.info("💡 Generate materials and add recruiter emails for jobs to send applications")
            else:
                job_options = [f"{j['title']} at {j['company']} ({j.get('match_score', 0)}%)" for j in sendable_jobs]
                selected_job_idx = st.selectbox("Select Job to Send", range(len(sendable_jobs)), format_func=lambda x: job_options[x])
                
                selected_job = sendable_jobs[selected_job_idx]
                job_key = f"{selected_job['id']}"
                materials = st.session_state.generated_materials.get(job_key, {})
                
                st.write(f"**Sending to:** {selected_job.get('recruiter_email')}")
                
                email_subject = st.text_input(
                    "Email Subject",
                    value=f"Application for {selected_job['title']} Position"
                )
                
                default_body = f"""Dear Hiring Manager,

I am writing to express my strong interest in the {selected_job['title']} position at {selected_job['company']}.

Please find attached my resume and cover letter for your review. I believe my experience and skills make me an excellent fit for this role.

I look forward to the opportunity to discuss how I can contribute to your team.

Best regards,
Your Name"""
                
                email_body = st.text_area("Email Body", value=default_body, height=300)
                
                st.write("**Attachments:**")
                attach_resume = st.checkbox("Attach Tailored Resume", value='resume' in materials)
                attach_cover = st.checkbox("Attach Cover Letter", value='cover_letter' in materials)
                
                if st.button("📤 Send Application", type="primary"):
                    attachments = {}
                    
                    if attach_resume and 'resume' in materials:
                        attachments[f"Resume_{selected_job['company']}.txt"] = materials['resume']
                    
                    if attach_cover and 'cover_letter' in materials:
                        attachments[f"CoverLetter_{selected_job['company']}.txt"] = materials['cover_letter']
                    
                    with st.spinner("Sending email..."):
                        success, message = send_application_email(
                            selected_job['recruiter_email'],
                            email_subject,
                            email_body,
                            attachments,
                            st.session_state.smtp_config
                        )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")

# Footer
st.divider()
st.caption("Built with Streamlit | Powered by Claude AI | Resume Matcher & Application Suite v2.0")
st.caption("⚡ No API key required - Claude API access is built-in!")
