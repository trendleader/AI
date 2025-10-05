import streamlit as st
import json
import datetime
import requests
import os

# Set page config
st.set_page_config(
    page_title="Technology Career Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > div {
        background-color: #f8fafc;
    }
    .career-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .skill-tag {
        background-color: #e2e8f0;
        color: #2d3748;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        display: inline-block;
        margin: 0.25rem 0.25rem 0.25rem 0;
    }
    .salary-badge {
        background-color: #48bb78;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: bold;
    }
    .step-number {
        background-color: #4299e1;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.875rem;
        font-weight: bold;
        margin-right: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

def get_course_recommendations(course_name):
    """Get career recommendations for a specific course using Claude API"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        st.error("❌ Please set your ANTHROPIC_API_KEY environment variable")
        st.info("Get your API key from: https://console.anthropic.com/")
        return None
    
    prompt = f"""As an educational career counselor, provide personalized recommendations for someone interested in learning {course_name}. 

Please respond with a JSON object in this exact format:
{{
  "courseOverview": "Brief description of the course and its importance",
  "careerPaths": [
    {{
      "title": "Career Path Name",
      "description": "What this role involves",
      "salaryRange": "Typical salary range",
      "skillsNeeded": ["skill1", "skill2", "skill3"]
    }}
  ],
  "recommendedProgram": {{
    "beginner": "Specific training recommendations for beginners",
    "intermediate": "Recommendations for those with some experience",
    "advanced": "Advanced learning path suggestions"
  }},
  "nextSteps": ["step1", "step2", "step3"]
}}

Your entire response must be a single, valid JSON object. DO NOT include any text outside of the JSON structure."""

    try:
        # Using requests to make the API call directly
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1500,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )
        
        if response.status_code != 200:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")
            return None
            
        data = response.json()
        response_text = data['content'][0]['text']
        
        # Clean up any markdown formatting
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        recommendations = json.loads(response_text)
        return recommendations
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network Error: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON Parse Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        return None

def get_sample_recommendations(course_name):
    """Provide comprehensive sample recommendations when API is not available"""
    sample_data = {
        "SQL (Database Management)": {
            "courseOverview": "SQL (Structured Query Language) is a fundamental skill for working with databases and is essential in today's data-driven world. It enables professionals to extract, analyze, and manipulate data from relational databases, making it valuable across industries including technology, finance, healthcare, marketing, and business intelligence.",
            "careerPaths": [
                {
                    "title": "Data Analyst",
                    "description": "Analyze data trends, create reports, and provide insights to support business decisions using SQL queries and visualization tools",
                    "salaryRange": "$50,000 - $85,000",
                    "skillsNeeded": ["SQL", "Excel", "Tableau/Power BI", "Statistical Analysis", "Python/R"]
                },
                {
                    "title": "Database Administrator",
                    "description": "Manage and maintain database systems, ensure data security, optimize performance, and troubleshoot database issues",
                    "salaryRange": "$70,000 - $120,000",
                    "skillsNeeded": ["Advanced SQL", "Database Design", "Security Management", "Backup/Recovery", "Server Administration"]
                },
                {
                    "title": "Business Intelligence Developer",
                    "description": "Design and develop data warehouses, ETL processes, and reporting solutions to support business intelligence initiatives",
                    "salaryRange": "$75,000 - $110,000",
                    "skillsNeeded": ["SQL", "ETL Tools", "Data Modeling", "SSIS/Talend", "Business Analysis"]
                }
            ],
            "recommendedProgram": {
                "beginner": "Start with free resources like SQLBolt or W3Schools SQL Tutorial, then progress to platforms like Codecademy SQL course. Practice with sample databases like Sakila or Northwind.",
                "intermediate": "Enroll in intermediate courses on Coursera or Udemy focusing on advanced joins, subqueries, and window functions. Practice with real datasets on platforms like HackerRank SQL.",
                "advanced": "Pursue advanced certifications like Microsoft SQL Server certification or Oracle Database certification. Focus on performance tuning, stored procedures, and database design."
            },
            "nextSteps": [
                "Complete a beginner SQL tutorial and practice basic queries on sample databases",
                "Build a portfolio project demonstrating your SQL skills with real-world data analysis",
                "Network with professionals in your target field and consider informational interviews",
                "Apply for entry-level positions or internships that utilize SQL skills"
            ]
        },
        "Tableau (Data Visualization)": {
            "courseOverview": "Tableau is a leading data visualization and business intelligence platform that enables users to connect, visualize, and share data insights. It's essential for transforming raw data into meaningful, interactive dashboards and reports that drive business decisions across all industries.",
            "careerPaths": [
                {
                    "title": "Data Visualization Specialist",
                    "description": "Create compelling visual stories from data using Tableau to help stakeholders understand complex information and make data-driven decisions",
                    "salaryRange": "$55,000 - $90,000",
                    "skillsNeeded": ["Tableau", "Data Analysis", "SQL", "Design Principles", "Business Intelligence"]
                },
                {
                    "title": "Business Intelligence Analyst",
                    "description": "Develop dashboards, reports, and data models to support business operations and strategic planning using Tableau and other BI tools",
                    "salaryRange": "$65,000 - $105,000",
                    "skillsNeeded": ["Tableau", "SQL", "Business Analysis", "Statistics", "Project Management"]
                },
                {
                    "title": "Data Analyst",
                    "description": "Analyze business data, identify trends, and create visualizations to communicate insights to stakeholders and support decision-making processes",
                    "salaryRange": "$50,000 - $85,000",
                    "skillsNeeded": ["Tableau", "Excel", "SQL", "Statistical Analysis", "Critical Thinking"]
                },
                {
                    "title": "Tableau Developer",
                    "description": "Design and build complex Tableau workbooks, optimize performance, and maintain enterprise-level Tableau implementations",
                    "salaryRange": "$70,000 - $115,000",
                    "skillsNeeded": ["Advanced Tableau", "SQL", "Data Modeling", "Server Administration", "Programming"]
                }
            ],
            "recommendedProgram": {
                "beginner": "Start with Tableau Public (free version) and complete Tableau's official training videos. Take online courses like 'Tableau Fundamentals' on Coursera or Udemy. Practice with public datasets and build your first dashboards.",
                "intermediate": "Pursue Tableau Desktop Specialist certification, learn advanced chart types, calculated fields, and table calculations. Work on real-world projects and start building a portfolio on Tableau Public.",
                "advanced": "Earn Tableau Desktop Certified Associate or Professional certifications. Learn Tableau Server/Cloud administration, advanced analytics, and integration with other tools like R and Python."
            },
            "nextSteps": [
                "Download Tableau Public and complete the built-in tutorial with sample data",
                "Build 3-5 dashboard projects using different datasets and publish them on Tableau Public",
                "Study for and take the Tableau Desktop Specialist certification exam",
                "Join Tableau community forums and attend local Tableau User Group meetups",
                "Apply for roles that use data visualization and highlight your Tableau portfolio"
            ]
        },
        "Power BI (Business Intelligence)": {
            "courseOverview": "Microsoft Power BI is a comprehensive business analytics platform that enables users to visualize data, share insights, and make data-driven decisions. As part of the Microsoft ecosystem, it's widely adopted in enterprises and offers powerful integration with other Microsoft tools.",
            "careerPaths": [
                {
                    "title": "Power BI Developer",
                    "description": "Design, develop, and maintain Power BI reports and dashboards, work with data models, and ensure optimal performance for business users",
                    "salaryRange": "$65,000 - $110,000",
                    "skillsNeeded": ["Power BI", "DAX", "SQL", "Data Modeling", "Microsoft Stack"]
                },
                {
                    "title": "Business Intelligence Analyst",
                    "description": "Create analytical solutions using Power BI to help organizations understand their data and improve business processes",
                    "salaryRange": "$60,000 - $95,000",
                    "skillsNeeded": ["Power BI", "Business Analysis", "SQL", "Excel", "Data Visualization"]
                },
                {
                    "title": "Data Analyst",
                    "description": "Analyze business data using Power BI and other tools to identify trends, patterns, and opportunities for business improvement",
                    "salaryRange": "$50,000 - $85,000",
                    "skillsNeeded": ["Power BI", "Excel", "SQL", "Statistics", "Business Acumen"]
                }
            ],
            "recommendedProgram": {
                "beginner": "Start with Microsoft's free Power BI learning path, download Power BI Desktop, and work through guided tutorials. Practice with sample datasets and learn basic visualization principles.",
                "intermediate": "Learn DAX (Data Analysis Expressions), advanced data modeling, and Power Query. Work on real business scenarios and start preparing for Microsoft PL-300 certification.",
                "advanced": "Master Power BI Service administration, implement row-level security, and learn advanced analytics features. Pursue Microsoft Certified: Data Analyst Associate certification."
            },
            "nextSteps": [
                "Download Power BI Desktop and complete Microsoft's guided learning modules",
                "Practice building reports with different data sources (Excel, SQL, web APIs)",
                "Learn DAX fundamentals and create calculated columns and measures",
                "Build a portfolio of Power BI projects showcasing different visualization types",
                "Study for Microsoft PL-300 certification exam to validate your skills"
            ]
        }
    }
    
    # Default template for courses not in sample data
    default_template = {
        "courseOverview": f"{course_name} is a valuable technology skill that opens doors to various career opportunities in today's digital workplace. This skill is increasingly important across industries as organizations continue to digitize and leverage technology for competitive advantage.",
        "careerPaths": [
            {
                "title": f"{course_name} Specialist",
                "description": f"Work as a specialist using {course_name} to solve business problems, improve processes, and drive digital transformation initiatives",
                "salaryRange": "$55,000 - $95,000",
                "skillsNeeded": [course_name, "Problem Solving", "Communication", "Project Management"]
            },
            {
                "title": f"Senior {course_name} Consultant",
                "description": f"Provide expert guidance on {course_name} implementations, lead projects, and mentor junior team members",
                "salaryRange": "$75,000 - $125,000",
                "skillsNeeded": [f"Advanced {course_name}", "Leadership", "Business Analysis", "Client Relations"]
            }
        ],
        "recommendedProgram": {
            "beginner": f"Start with online tutorials and introductory courses for {course_name}. Focus on understanding basic concepts and getting hands-on practice with simple projects.",
            "intermediate": f"Take advanced {course_name} courses, work on real-world projects, and consider earning relevant certifications to validate your skills.",
            "advanced": f"Pursue expert-level certifications, specialize in specific areas of {course_name}, and consider leadership or consulting roles."
        },
        "nextSteps": [
            f"Complete a comprehensive beginner course in {course_name}",
            "Practice with hands-on projects and build a portfolio",
            "Network with professionals in the field and join relevant communities",
            "Apply for entry-level positions that use {course_name}",
            "Continue learning complementary skills and technologies"
        ]
    }
    
    return sample_data.get(course_name, default_template)

def display_recommendations(recommendations, course_name):
    """Display the career recommendations in a formatted way"""
    
    # Course Overview
    st.markdown(f"""
    <div class="recommendation-card">
        <h3>📚 Course Overview</h3>
        <p>{recommendations['courseOverview']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Career Paths
    st.markdown("### 💼 Career Paths")
    for i, path in enumerate(recommendations['careerPaths']):
        skills_html = ''.join([f'<span class="skill-tag">{skill}</span>' for skill in path['skillsNeeded']])
        
        st.markdown(f"""
        <div class="recommendation-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h4 style="margin: 0; color: #2d3748;">{path['title']}</h4>
                <span class="salary-badge">{path['salaryRange']}</span>
            </div>
            <p style="color: #4a5568; margin-bottom: 0.75rem;">{path['description']}</p>
            <div><strong>Skills Needed:</strong><br>{skills_html}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Training Programs
    st.markdown("### 🎯 Recommended Training Programs")
    for level, description in recommendations['recommendedProgram'].items():
        st.markdown(f"""
        <div class="recommendation-card">
            <h4 style="color: #805ad5; text-transform: capitalize; margin-bottom: 0.5rem;">{level} Level</h4>
            <p style="color: #4a5568;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Next Steps
    st.markdown("### 🚀 Your Next Steps")
    for i, step in enumerate(recommendations['nextSteps'], 1):
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
            <span class="step-number">{i}</span>
            <p style="margin: 0; color: #4a5568; line-height: 1.5;">{step}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="career-card">
        <h1>🎓 Technology Career Advisor</h1>
        <p>Discover your path in tech with personalized career guidance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Course selection
    courses = [
        "Select a course...",
        "SQL (Database Management)",
        "Power BI (Business Intelligence)",
        "Excel (Data Analysis)",
        "Python (Programming)",
        "Tableau (Data Visualization)",
        "AWS (Cloud Computing)",
        "JavaScript (Web Development)",
        "Data Science",
        "Cybersecurity"
    ]
    
    selected_course = st.selectbox(
        "Choose a technology course to explore:",
        courses,
        key="course_selector"
    )
    
    if selected_course != "Select a course...":
        with st.spinner(f"🤖 Getting personalized recommendations for {selected_course}..."):
            # Try to get recommendations from Claude API first
            recommendations = get_course_recommendations(selected_course)
            
            # Fall back to sample data if API is not available
            if not recommendations:
                st.warning("⚠️ Using sample data. Set up your Anthropic API key for personalized recommendations.")
                recommendations = get_sample_recommendations(selected_course)
            
            if recommendations:
                st.success(f"✅ Here are your personalized recommendations for {selected_course}:")
                display_recommendations(recommendations, selected_course)
                
                # Add to chat history
                if selected_course not in [msg.get('course') for msg in st.session_state.messages]:
                    st.session_state.messages.append({
                        'course': selected_course,
                        'recommendations': recommendations,
                        'timestamp': datetime.datetime.now()
                    })
    
    # Display chat history
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 📝 Your Exploration History")
        for msg in st.session_state.messages:
            with st.expander(f"📚 {msg['course']} - {msg['timestamp'].strftime('%I:%M %p')}"):
                display_recommendations(msg['recommendations'], msg['course'])
    
    # Setup instructions
    with st.sidebar:
        st.markdown("### 🔧 Setup Instructions")
        st.markdown("""
        **To use with Claude API:**
        1. Get your API key from [Anthropic Console](https://console.anthropic.com/)
        2. Set environment variable:
           ```bash
           export ANTHROPIC_API_KEY="your-key-here"
           ```
           
           **Windows:**
           ```cmd
           set ANTHROPIC_API_KEY=your-key-here
           ```
           
        3. Install required packages:
           ```bash
           pip install streamlit requests
           ```
        4. Run the app:
           ```bash
           streamlit run career_chatbot.py
           ```
        """)
        
        st.markdown("### 📋 Requirements")
        st.code("""
streamlit>=1.28.0
requests>=2.25.0
        """)
        
        # API Key status
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            st.success("✅ API Key detected")
        else:
            st.warning("⚠️ No API Key found")

if __name__ == "__main__":
    main()
