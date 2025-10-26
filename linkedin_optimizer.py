import streamlit as st
from openai import OpenAI
import openai
import sys

# Check OpenAI version
try:
    from packaging import version
    openai_version = openai.__version__
    if version.parse(openai_version) < version.parse("1.0.0"):
        st.error("❌ OpenAI library version 1.0.0 or higher is required. Please run: pip install --upgrade openai")
        st.stop()
except:
    pass

# Page configuration
st.set_page_config(
    page_title="LinkedIn Profile Optimizer",
    page_icon="💼",
    layout="wide"
)

# Title and description
st.title("💼 LinkedIn Profile Optimizer")
st.markdown(
    "Leverage OpenAI to get actionable suggestions for optimizing your LinkedIn profile."
)

# Sidebar for API key input
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Your API key is never stored and only used for this session"
)

if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to get started.")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=api_key.strip())
    # Test the connection
    client.models.list()
except openai.AuthenticationError:
    st.error("❌ Invalid API Key. Please verify your OpenAI API key is correct and has not expired.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error connecting to OpenAI: {str(e)}")
    st.stop()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Your LinkedIn Profile Content")
    st.markdown(
        """
        **Instructions:**
        1. Copy your LinkedIn profile content (headline, about, experience, skills, etc.)
        2. Paste it in the text area below
        3. Click "Analyze Profile" to get optimization suggestions
        """
    )
    
    profile_content = st.text_area(
        "Paste your LinkedIn profile content here:",
        height=300,
        placeholder="Example:\n\nHeadline: Senior Data Scientist at Tech Company\n\nAbout:\nI'm passionate about data science...\n\nExperience:\n- Senior Data Scientist at Tech Company (2022-Present)\n- Data Scientist at StartupCo (2020-2022)\n\nSkills:\nPython, Machine Learning, Data Analysis...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("⚙️ Optimization Settings")
    
    optimization_focus = st.multiselect(
        "What areas would you like to focus on?",
        [
            "Headline Impact",
            "About Section",
            "Experience Descriptions",
            "Skills & Endorsements",
            "SEO & Keywords",
            "Tone & Professionalism",
            "Call-to-Action",
            "Overall Strategy"
        ],
        default=[
            "Headline Impact",
            "About Section",
            "SEO & Keywords",
            "Overall Strategy"
        ]
    )
    
    tone = st.select_slider(
        "Desired Tone:",
        options=["Casual", "Professional", "Executive"],
        value="Professional"
    )
    
    length_preference = st.radio(
        "Response Length:",
        ["Concise", "Detailed", "Comprehensive"]
    )

# Analyze button
if st.button("🔍 Analyze Profile", type="primary", use_container_width=True):
    
    if not profile_content.strip():
        st.error("Please paste your LinkedIn profile content to analyze.")
    else:
        with st.spinner("Analyzing your profile..."):
            try:
                # Build the prompt
                focus_areas = ", ".join(optimization_focus) if optimization_focus else "all areas"
                
                if length_preference == "Concise":
                    length_instruction = "Provide brief, actionable suggestions (2-3 sentences per point)."
                elif length_preference == "Detailed":
                    length_instruction = "Provide detailed suggestions with examples (3-5 sentences per point)."
                else:
                    length_instruction = "Provide comprehensive suggestions with detailed examples and rationale (5-7 sentences per point)."
                
                prompt = f"""You are an expert LinkedIn profile optimizer with 10+ years of experience helping professionals enhance their profiles for maximum impact and visibility.

Analyze the following LinkedIn profile content and provide actionable optimization suggestions:

PROFILE CONTENT:
---
{profile_content}
---

OPTIMIZATION FOCUS AREAS: {focus_areas}
DESIRED TONE: {tone}

{length_instruction}

Please structure your response as follows:
1. **Executive Summary** - A 2-3 sentence overview of the top 3 opportunities for improvement
2. **Detailed Recommendations** - Specific suggestions for each focus area
3. **Quick Wins** - 3-5 immediate actions they can take (these should be the easiest/highest impact)
4. **Revised Examples** - Provide 1-2 specific examples of rewritten content from their profile
5. **Next Steps** - Strategic recommendations for ongoing LinkedIn optimization

Focus on being constructive, specific, and actionable. Include percentages or metrics when relevant to demonstrate potential impact."""

                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                suggestions = response.choices[0].message.content
                
                # Display results
                st.success("✅ Analysis complete!")
                
                st.markdown("---")
                st.subheader("📊 Optimization Suggestions")
                
                st.markdown(suggestions)
                
                # Add export option
                st.markdown("---")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("📋 Copy to Clipboard"):
                        st.success("Suggestions copied! You can now paste them elsewhere.")
                
                with col2:
                    st.download_button(
                        label="📥 Download as Text",
                        data=suggestions,
                        file_name="linkedin_profile_suggestions.txt",
                        mime="text/plain"
                    )
                
            except openai.APIConnectionError as e:
                st.error(f"Connection error: {str(e)}")
            except openai.AuthenticationError as e:
                st.error(f"Authentication failed: Invalid API key. Please check your OpenAI API key.")
            except openai.RateLimitError as e:
                st.error(f"Rate limit reached: Please wait a moment and try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    **Tips for best results:**
    - Include your complete LinkedIn profile information (headline, about, experience, skills, recommendations)
    - Be specific about your goals (e.g., "looking for leadership roles", "wanting to stand out in tech industry")
    - The more detail you provide, the more tailored the suggestions will be
    
    **Note:** This app doesn't store your data. Your API key is used only for this session.
    """
)
