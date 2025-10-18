import streamlit as st
import PyPDF2
import re
from collections import Counter
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="Resume ATS Scanner",
    page_icon="📋",
    layout="wide"
)

# Title and description
st.title("📋 Resume ATS Scanner")
st.markdown("Automated resume screening tool to identify top candidates")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state['analysis_complete'] = False
    st.session_state['rankings'] = None
    st.session_state['uploaded_files'] = None
    st.session_state['job_description'] = None

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    company_name = st.text_input(
        "Company Name",
        value="Your Company",
        help="Enter your company name"
    )
    
    position_title = st.text_input(
        "Position Title",
        value="Senior Business Analyst",
        help="Job title for the position"
    )

# Main content area
st.subheader("1️⃣ Job Description")

default_jd = """We are seeking an experienced and highly-motivated Senior Business Analyst (SBA) to join our team. The Senior Business Analyst will play a critical role in analyzing complex business problems and opportunities, developing strategic solutions, and driving process improvements across the organization. This role requires a blend of deep analytical skills, strong leadership capabilities, and exceptional stakeholder management.

Key Responsibilities:
• Strategic Analysis and Planning: Investigate, analyze, and document complex business problems, opportunities, and strategic initiatives. Develop and define business cases, including cost-benefit analysis, risk assessment, and impact analysis for proposed solutions.
• Requirements Management: Plan, arrange, and facilitate engaging workshops and meetings with senior business and IT stakeholders to elicit, analyze, and prioritize detailed business requirements. Translate high-level business needs into clear, concise, and detailed functional and non-functional requirements.
• Process Improvement and Solution Design: Analyze current business processes, identify gaps, and design improved 'to-be' processes. Work closely with technical teams to assess the feasibility of solutions.
• Stakeholder Engagement and Leadership: Act as the primary liaison, building and maintaining strong relationships with stakeholders across multiple departments.

Key Requirements:
• 7+ years of business analysis experience
• Strong experience with process improvement and business process management
• Proficiency in requirements gathering and documentation
• Experience with SQL and data analysis
• Knowledge of system design and technical architecture
• Excellent communication and leadership skills
• Experience with Agile/Scrum methodologies
• Strong problem-solving and critical thinking abilities"""

job_description = st.text_area(
    "Paste the job description below:",
    value=default_jd,
    height=250,
    help="Enter the full job description for the position"
)

st.session_state['job_description'] = job_description

st.subheader("2️⃣ Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload resume files (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload up to 5 resumes for analysis"
)

st.session_state['uploaded_files'] = uploaded_files

# Analysis functions
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.warning(f"Error reading PDF: {e}")
        return ""

def extract_text_from_file(file) -> str:
    """Extract text from uploaded file"""
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

def extract_keywords(text: str) -> List[str]:
    """Extract keywords and phrases from text"""
    # Convert to lowercase and remove special characters
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Split into words
    words = re.findall(r'\b\w+\b', text)
    return words

def calculate_match_score(resume_text: str, job_description: str) -> float:
    """Calculate match score between resume and job description"""
    
    # Define key skills and requirements to look for
    key_skills = {
        "business analysis": 15,
        "requirements management": 12,
        "process improvement": 12,
        "stakeholder management": 10,
        "sql": 8,
        "data analysis": 8,
        "agile": 7,
        "scrum": 7,
        "risk assessment": 8,
        "cost-benefit analysis": 8,
        "user stories": 7,
        "process mapping": 8,
        "workshop facilitation": 6,
        "technical design": 7,
        "uat": 6,
        "python": 5,
        "analytics": 5,
        "communication": 5,
        "leadership": 6,
        "problem solving": 5,
        "critical thinking": 5,
    }
    
    experience_keywords = {
        "5+ years": 10,
        "7+ years": 15,
        "10+ years": 20,
        "8+ years": 18,
        "senior": 8,
        "lead": 8,
        "director": 10,
    }
    
    resume_lower = resume_text.lower()
    score = 0
    matches = []
    
    # Check for key skills
    for skill, points in key_skills.items():
        if skill in resume_lower:
            score += points
            matches.append(skill)
    
    # Check for experience level
    for exp, points in experience_keywords.items():
        if exp in resume_lower:
            score += points
            matches.append(exp)
    
    # Check for education
    if any(degree in resume_lower for degree in ["mba", "master", "bachelor", "degree", "b.a.", "b.s."]):
        score += 5
        matches.append("education")
    
    # Check for certifications
    if any(cert in resume_lower for cert in ["pmi", "cbap", "cissp", "certified", "pmp"]):
        score += 5
        matches.append("certification")
    
    # Normalize score (cap at 100)
    normalized_score = min(score, 100)
    
    return normalized_score, matches

def rank_resumes(uploaded_files, job_description: str) -> List[Tuple]:
    """Rank all uploaded resumes against job description"""
    
    rankings = []
    
    for file in uploaded_files:
        resume_text = extract_text_from_file(file)
        
        if resume_text:
            score, matches = calculate_match_score(resume_text, job_description)
            rankings.append({
                "filename": file.name,
                "score": score,
                "matches": matches,
                "text": resume_text[:500]  # First 500 chars for preview
            })
    
    # Sort by score descending
    rankings.sort(key=lambda x: x["score"], reverse=True)
    
    return rankings

# Analyze button
if uploaded_files and len(uploaded_files) > 0:
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔍 Analyze Resumes", type="primary", use_container_width=True):
            with st.spinner("Analyzing resumes..."):
                rankings = rank_resumes(uploaded_files, job_description)
                st.session_state['rankings'] = rankings
                st.session_state['analysis_complete'] = True
else:
    st.info("📤 Upload at least one resume to begin analysis")

# Display results
if st.session_state['analysis_complete'] and st.session_state['rankings']:
    rankings = st.session_state['rankings']
    
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Resumes Analyzed", len(rankings))
    with col2:
        st.metric("Top Score", f"{rankings[0]['score']:.1f}/100")
    with col3:
        st.metric("Recommended for Interview", "2")
    
    st.markdown("---")
    
    # Top 2 candidates
    st.subheader("🏆 Top 2 Recommended Candidates")
    
    for idx, candidate in enumerate(rankings[:2], 1):
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### #{idx} - {candidate['filename']}")
                st.markdown(f"**Match Score: {candidate['score']:.1f}/100**")
            
            with col2:
                # Color coding based on score
                if candidate['score'] >= 80:
                    color = "green"
                    status = "Excellent"
                elif candidate['score'] >= 60:
                    color = "orange"
                    status = "Good"
                else:
                    color = "red"
                    status = "Fair"
                
                st.markdown(f":{color}[{status}]")
            
            # Matched skills
            if candidate['matches']:
                st.markdown("**Key Matches:**")
                # Display matches as tags
                match_text = " • ".join(candidate['matches'][:8])
                st.markdown(f"`{match_text}`")
            
            with st.expander("View Resume Preview"):
                st.text(candidate['text'])
    
    st.markdown("---")
    
    # All rankings table
    st.subheader("📋 All Candidates Ranked")
    
    results_data = []
    for idx, candidate in enumerate(rankings, 1):
        results_data.append({
            "Rank": idx,
            "Candidate": candidate['filename'],
            "Match Score": f"{candidate['score']:.1f}",
            "Recommended": "✅ Yes" if idx <= 2 else "❌ No"
        })
    
    st.dataframe(results_data, use_container_width=True, hide_index=True)
    
    # Export recommendations
    st.markdown("---")
    st.subheader("📑 Recommendations")
    
    recommendation_text = f"""
    **HIRING RECOMMENDATION REPORT**
    
    Company: {company_name}
    Position: {position_title}
    Analysis Date: {st.session_state.get('analysis_date', 'Today')}
    
    **RECOMMENDED CANDIDATES FOR FINAL INTERVIEW:**
    
    """
    
    for idx, candidate in enumerate(rankings[:2], 1):
        recommendation_text += f"\n{idx}. **{candidate['filename']}** (Match Score: {candidate['score']:.1f}/100)\n"
        recommendation_text += f"   Key Qualifications: {', '.join(candidate['matches'][:5])}\n"
    
    recommendation_text += f"\n---\n\nTotal Resumes Reviewed: {len(rankings)}\n"
    recommendation_text += f"Recommendation: Proceed with interviews for the above 2 candidates.\n"
    
    st.text_area(
        "Recommendation Report:",
        value=recommendation_text,
        height=300,
        disabled=True
    )
    
    # Copy button
    st.download_button(
        label="📥 Download Report",
        data=recommendation_text,
        file_name=f"hiring_recommendation_{position_title.replace(' ', '_')}.txt",
        mime="text/plain"
    )
