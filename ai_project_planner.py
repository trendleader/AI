import streamlit as st
from openai import OpenAI
import os

# Set page configuration
st.set_page_config(
    page_title="AI/ML Project Planner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚀 AI/ML Project Planner</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<strong>Welcome!</strong> This tool helps you plan the implementation of your AI/ML business project. 
Simply provide your OpenAI API key and describe your project, and receive a detailed roadmap with steps, 
requirements, and best practices.
</div>
""", unsafe_allow_html=True)

# Sidebar for API key input
st.sidebar.header("🔑 API Configuration")
api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    help="Your API key is not stored and only used for this session"
)

if not api_key:
    st.sidebar.warning("⚠️ Please enter your OpenAI API key to proceed")

# Main content area
st.subheader("📋 Describe Your Project")

project_name = st.text_input(
    "Project Name",
    placeholder="e.g., Customer Churn Prediction System",
    help="Enter a descriptive name for your AI/ML project"
)

project_description = st.text_area(
    "Project Description",
    placeholder="Describe what your project should accomplish, who it's for, and any specific requirements...",
    height=150,
    help="Provide details about your business problem, goals, and constraints"
)

# Additional options
col1, col2 = st.columns(2)
with col1:
    model_choice = st.selectbox(
        "Select OpenAI Model",
        ["gpt-4o-mini", "gpt-4o"],
        help="Choose the model to use for planning"
    )

with col2:
    experience_level = st.selectbox(
        "Your Team's Experience Level",
        ["Beginner", "Intermediate", "Advanced"],
        help="This helps tailor the complexity of recommendations"
    )

# Generate button
if st.button("🎯 Generate Project Plan", type="primary", use_container_width=True):
    if not api_key:
        st.error("❌ Please enter your OpenAI API key in the sidebar")
    elif not project_name or not project_description:
        st.error("❌ Please fill in both the project name and description")
    else:
        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)
            
            # Create the prompt
            prompt = f"""You are an expert AI/ML architect and project manager. Based on the following project description, 
provide a comprehensive, actionable implementation plan.

Project Name: {project_name}
Project Description: {project_description}
Team Experience Level: {experience_level}

Please structure your response with the following sections:

1. **Project Overview** - Brief summary of the project and its business value
2. **Success Metrics** - Key metrics to measure project success
3. **Phase 1: Planning & Data Preparation**
   - Specific steps and deliverables
4. **Phase 2: Model Development**
   - Algorithm selection reasoning
   - Specific tools and libraries recommended
   - Training approach
5. **Phase 3: Evaluation & Optimization**
   - Testing strategy
   - Performance benchmarks
6. **Phase 4: Deployment & Monitoring**
   - Deployment approach
   - Monitoring and maintenance strategy
7. **Required Resources**
   - Team skills and roles
   - Infrastructure requirements
   - Estimated timeline and budget considerations
8. **Potential Challenges & Mitigation**
   - Common pitfalls and how to avoid them
9. **Recommended Tools & Technologies** - Specific stack recommendations for {experience_level} level teams
10. **Next Steps** - Immediate action items to get started

Be specific, practical, and consider the team's {experience_level} experience level. Provide concrete examples and best practices."""

            # Show loading spinner while generating
            with st.spinner("🔄 Generating your project plan..."):
                response = client.chat.completions.create(
                    model=model_choice,
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            
            # Display the response
            st.success("✅ Project plan generated successfully!")
            
            st.markdown("---")
            st.subheader("📊 Your AI/ML Project Plan")
            st.markdown(response.choices[0].message.content)
            
            # Add download button
            plan_text = response.choices[0].message.content
            st.download_button(
                label="📥 Download Plan as Text",
                data=plan_text,
                file_name=f"{project_name.replace(' ', '_')}_plan.txt",
                mime="text/plain"
            )
            
        except ValueError as e:
            st.error(f"❌ Invalid API Key: Please check your OpenAI API key")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Tips: Make sure your API key is valid and has available credits")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔐 Privacy: Your API key is never stored")
with col2:
    st.caption("📚 Powered by OpenAI")
with col3:
    st.caption("⚡ Built with Streamlit")
