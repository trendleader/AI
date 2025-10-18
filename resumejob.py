import streamlit as st
import requests
import json
from datetime import datetime

st.set_page_config(page_title="Resume Job Matcher", layout="wide")

st.title("🎯 Resume-to-Job Matcher")
st.markdown("Compare your resume to job descriptions and get tailored resumes & cover letters")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Claude API Key", type="password", help="Get your API key from https://console.anthropic.com")
    alignment_threshold = st.slider("Alignment Threshold (%)", 0, 100, 75, help="Only show jobs above this match percentage")
    st.markdown("---")
    st.info("📝 Paste job descriptions below to analyze them against your resume")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📄 Resume Upload", "🔍 Job Analysis", "✨ Generated Materials"])

with tab1:
    st.header("Your Resume")
    resume_text = st.text_area(
        "Paste your resume content here:",
        height=300,
        placeholder="Paste your complete resume text...",
        key="resume_input"
    )
    
    if resume_text:
        st.success(f"✓ Resume loaded ({len(resume_text.split())} words)")
        
        # Show resume preview
        with st.expander("Preview Resume"):
            st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)

with tab2:
    st.header("Job Description Analysis")
    
    if not resume_text:
        st.warning("⚠️ Please upload your resume first in the Resume Upload tab")
    elif not api_key:
        st.warning("⚠️ Please enter your Claude API key in the sidebar")
    else:
        num_jobs = st.number_input("How many job descriptions?", min_value=1, max_value=5, value=1)
        
        jobs = []
        for i in range(num_jobs):
            st.subheader(f"Job Description {i+1}")
            
            col1, col2 = st.columns(2)
            with col1:
                job_title = st.text_input(f"Job Title {i+1}", key=f"title_{i}")
            with col2:
                company = st.text_input(f"Company {i+1}", key=f"company_{i}")
            
            job_desc = st.text_area(
                f"Paste job description {i+1}:",
                height=200,
                key=f"job_{i}",
                placeholder="Paste the job description..."
            )
            
            if job_title and job_desc:
                jobs.append({
                    "title": job_title,
                    "company": company,
                    "description": job_desc,
                    "index": i
                })
        
        if st.button("🔍 Analyze Jobs", key="analyze_btn", type="primary"):
            if not jobs:
                st.error("Please enter at least one job description")
            else:
                st.session_state.analyzed_jobs = []
                
                for job in jobs:
                    with st.spinner(f"Analyzing {job['title']} at {job['company']}..."):
                        prompt = f"""ANALYZE RESUME VS JOB POSTING - RETURN ONLY VALID JSON

RESUME:
{resume_text}

JOB POSTING:
Title: {job['title']}
Company: {job['company']}
{job['description']}

YOUR RESPONSE MUST BE VALID JSON ONLY. NO OTHER TEXT. NO MARKDOWN. NO CODE BLOCKS.

{{
  "alignment_score": 75,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "strengths": "Candidate has X years experience with Y. Strong background in Z.",
  "gaps": "Missing experience in A. Limited exposure to B.",
  "recommendation": "Good Match"
}}

NOW ANALYZE AND RESPOND WITH ONLY THE JSON OBJECT:"""

                        try:
                            response = requests.post(
                                "https://api.anthropic.com/v1/messages",
                                headers={
                                    "x-api-key": api_key,
                                    "anthropic-version": "2023-06-01",
                                    "content-type": "application/json"
                                },
                                json={
                                    "model": "claude-sonnet-4-5-20250929",
                                    "max_tokens": 1000,
                                    "messages": [{"role": "user", "content": prompt}]
                                }
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                analysis_text = result['content'][0]['text']
                                
                                # Clean up markdown code blocks if present
                                if "```" in analysis_text:
                                    analysis_text = analysis_text.split("```")[1]
                                    if analysis_text.startswith("json"):
                                        analysis_text = analysis_text[4:]
                                
                                # Find JSON content between { and }
                                start_idx = analysis_text.find('{')
                                end_idx = analysis_text.rfind('}') + 1
                                
                                if start_idx != -1 and end_idx > start_idx:
                                    analysis_text = analysis_text[start_idx:end_idx]
                                
                                analysis_text = analysis_text.strip()
                                
                                # Parse JSON from response
                                analysis = json.loads(analysis_text)
                                analysis['job'] = job
                                st.session_state.analyzed_jobs.append(analysis)
                            else:
                                st.error(f"API Error: {response.status_code} - {response.text}")
                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing JSON for {job['title']}: {str(e)}\n\nRaw response: {analysis_text[:200] if 'analysis_text' in locals() else 'N/A'}")
                        except Exception as e:
                            st.error(f"Unexpected error analyzing {job['title']}: {str(e)}")
                        except Exception as e:
                            st.error(f"Error analyzing {job['title']}: {str(e)}")
                
                # Display results
                if st.session_state.analyzed_jobs:
                    st.subheader("Analysis Results")
                    
                    # Filter by alignment threshold
                    qualified_jobs = [j for j in st.session_state.analyzed_jobs 
                                    if j.get('alignment_score', 0) >= alignment_threshold]
                    
                    st.info(f"Found {len(qualified_jobs)} job(s) above {alignment_threshold}% alignment threshold")
                    
                    for analysis in qualified_jobs:
                        with st.expander(
                            f"✓ {analysis['job']['title']} @ {analysis['job']['company']} "
                            f"({analysis.get('alignment_score', 0)}%)",
                            expanded=False
                        ):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Alignment Score", f"{analysis.get('alignment_score', 0)}%")
                                st.write("**Recommendation:**", analysis.get('recommendation', 'N/A'))
                                st.write("**Matching Skills:**")
                                for skill in analysis.get('matching_skills', [])[:5]:
                                    st.write(f"  • {skill}")
                            
                            with col2:
                                st.write("**Missing Skills:**")
                                for skill in analysis.get('missing_skills', [])[:5]:
                                    st.write(f"  • {skill}")
                            
                            st.write("**Why it's a good fit:**", analysis.get('strengths', 'N/A'))
                            st.write("**Gaps to address:**", analysis.get('gaps', 'N/A'))
                            
                            # Store for later use
                            st.session_state.qualified_jobs = qualified_jobs

with tab3:
    st.header("Generate Customized Materials")
    
    if not resume_text:
        st.warning("⚠️ Please upload your resume first")
    elif not api_key:
        st.warning("⚠️ Please enter your Claude API key")
    elif 'qualified_jobs' not in st.session_state or not st.session_state.qualified_jobs:
        st.warning("⚠️ Please analyze jobs first in the Job Analysis tab")
    else:
        selected_job_idx = st.selectbox(
            "Select a job to generate materials for:",
            range(len(st.session_state.qualified_jobs)),
            format_func=lambda i: f"{st.session_state.qualified_jobs[i]['job']['title']} @ {st.session_state.qualified_jobs[i]['job']['company']}"
        )
        
        selected_job = st.session_state.qualified_jobs[selected_job_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Generate Tailored Resume", key="gen_resume"):
                with st.spinner("Generating customized resume..."):
                    prompt = f"""Based on this resume and job description, create a tailored resume that:
1. Highlights relevant experience from the original
2. Reorders sections to emphasize job-relevant content
3. Uses language and keywords from the job posting
4. Maintains truthfulness - don't add false information
5. Keeps the same person but optimizes presentation

ORIGINAL RESUME:
{resume_text}

TARGET JOB:
Title: {selected_job['job']['title']}
Company: {selected_job['job']['company']}
{selected_job['job']['description']}

Generate the tailored resume. Start directly with the resume content."""

                    try:
                        response = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={
                                "x-api-key": api_key,
                                "anthropic-version": "2023-06-01",
                                "content-type": "application/json"
                            },
                            json={
                                "model": "claude-opus-4-1-20250805",
                                "max_tokens": 2000,
                                "messages": [{"role": "user", "content": prompt}]
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            tailored_resume = result['content'][0]['text']
                            st.session_state.tailored_resume = tailored_resume
                            st.success("✓ Resume generated!")
                        else:
                            st.error(f"API Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("💌 Generate Cover Letter", key="gen_letter"):
                with st.spinner("Generating cover letter..."):
                    prompt = f"""Write a professional cover letter for this job application:

RESUME:
{resume_text}

JOB POSTING:
Title: {selected_job['job']['title']}
Company: {selected_job['job']['company']}
{selected_job['job']['description']}

Create a compelling cover letter that:
1. Opens with enthusiasm for the specific role and company
2. Highlights 2-3 relevant achievements from the resume
3. Addresses key requirements from the job posting
4. Explains why the candidate is interested
5. Closes with a call to action

Make it professional, personalized, and about 250-300 words."""

                    try:
                        response = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={
                                "x-api-key": api_key,
                                "anthropic-version": "2023-06-01",
                                "content-type": "application/json"
                            },
                            json={
                                "model": "claude-opus-4-1-20250805",
                                "max_tokens": 1000,
                                "messages": [{"role": "user", "content": prompt}]
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            cover_letter = result['content'][0]['text']
                            st.session_state.cover_letter = cover_letter
                            st.success("✓ Cover letter generated!")
                        else:
                            st.error(f"API Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        if 'tailored_resume' in st.session_state:
            with st.expander("📄 View Tailored Resume", expanded=False):
                st.text_area(
                    "Tailored Resume:",
                    value=st.session_state.tailored_resume,
                    height=400,
                    disabled=True
                )
                st.download_button(
                    "⬇️ Download Resume",
                    st.session_state.tailored_resume,
                    f"resume_{selected_job['job']['company']}.txt"
                )
        
        if 'cover_letter' in st.session_state:
            with st.expander("💌 View Cover Letter", expanded=False):
                st.text_area(
                    "Cover Letter:",
                    value=st.session_state.cover_letter,
                    height=300,
                    disabled=True
                )
                st.download_button(
                    "⬇️ Download Cover Letter",
                    st.session_state.cover_letter,
                    f"cover_letter_{selected_job['job']['company']}.txt"
                )