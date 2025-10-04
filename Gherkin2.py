import streamlit as st
import openai
import os
from pathlib import Path

# Load API key - works for both local and Streamlit Cloud
try:
    # Try Streamlit secrets first (for cloud deployment)
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    # Fallback to environment variable or dotenv (for local development)
    try:
        from dotenv import load_dotenv
        env_path = Path(r"C:\Users\pjone\OneDrive\Desktop\NewPython\test.env")
        load_dotenv(dotenv_path=env_path)
        openai.api_key = os.getenv("OPENAI_API_KEY")
    except:
        openai.api_key = os.getenv("OPENAI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Gherkin Assistant",
    page_icon="🥒",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
    .highlight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    .success-badge {
        background-color: #00c853;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🥒 Gherkin Assistant Pro")
st.markdown("Convert, enhance, and refine Gherkin syntax with grammar and style checking")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Mode Selection")
    
    app_mode = st.radio(
        "Choose Operation",
        ["Convert to Gherkin", "Enhance Existing Gherkin", "Validate & Fix Gherkin"],
        help="Select what you want to do"
    )
    
    st.divider()
    
    # Grammar and Style Rules
    st.subheader("📝 Grammar & Style Rules")
    
    enforce_capitalization = st.checkbox("Enforce proper capitalization", value=True,
                                        help="Ensure each step starts with a capital letter")
    
    no_contractions = st.checkbox("No contractions (I am not I'm)", value=True,
                                  help="Expand all contractions to full words")
    
    check_punctuation = st.checkbox("Check punctuation", value=True,
                                   help="Ensure proper punctuation throughout")
    
    enforce_grammar = st.checkbox("Enforce proper grammar", value=True,
                                 help="Check and fix grammatical errors")
    
    check_flow = st.checkbox("Check context and flow", value=True,
                            help="Ensure logical flow between steps")
    
    use_present_tense = st.checkbox("Use present tense", value=True,
                                   help="Ensure steps use present tense")
    
    st.divider()
    
    if app_mode == "Convert to Gherkin":
        st.subheader("Conversion Settings")
        conversion_mode = st.selectbox(
            "Conversion Method",
            ["AI-Powered (OpenAI)", "Rule-Based"],
            help="Choose between AI-powered conversion or rule-based logic"
        )
    
    elif app_mode == "Enhance Existing Gherkin":
        st.subheader("Enhancement Settings")
        enhancement_options = st.multiselect(
            "What to improve?",
            [
                "Clarity and readability",
                "Consistency in language",
                "Better Given/When/Then structure",
                "More descriptive steps",
                "Remove redundancy",
                "Add missing context"
            ],
            default=["Clarity and readability", "Consistency in language"]
        )
        
        enhancement_level = st.select_slider(
            "Enhancement Level",
            options=["Light", "Moderate", "Comprehensive"],
            value="Moderate"
        )
    
    elif app_mode == "Validate & Fix Gherkin":
        st.subheader("Validation Settings")
        check_options = st.multiselect(
            "Check for:",
            [
                "Syntax errors",
                "Inconsistent formatting",
                "Weak assertions",
                "Missing context",
                "Anti-patterns",
                "Best practices"
            ],
            default=["Syntax errors", "Inconsistent formatting", "Best practices"]
        )
    
    st.divider()
    
    # Common settings
    include_scenario = st.checkbox("Include Scenario wrapper", value=True)
    
    if include_scenario:
        scenario_name = st.text_input("Scenario Name", value="Test Scenario")
        tags_input = st.text_input("Tags (comma-separated)", placeholder="smoke, regression")
        tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
    
    include_feature = st.checkbox("Include Feature wrapper", value=False)
    
    if include_feature:
        feature_name = st.text_input("Feature Name", value="Feature Name")
        feature_description = st.text_area("Feature Description", 
                                          value="As a user, I want to...",
                                          height=100)
    
    st.divider()
    
    st.markdown("### 💡 Style Guide")
    st.markdown("""
    **Enforced Rules:**
    - ✅ Proper capitalization
    - ✅ No contractions
    - ✅ Correct punctuation
    - ✅ Present tense
    - ✅ Clear and concise
    - ✅ Logical flow
    """)

# Main content area with two columns
col1, col2 = st.columns(2)

with col1:
    if app_mode == "Convert to Gherkin":
        st.subheader("📝 Input Steps")
        sample_text = """user is logged into the application
user navigates to the profile page
user clicks on the edit button
user updates the email address
user clicks save
the profile should be updated
user should see a success message"""
    
    elif app_mode == "Enhance Existing Gherkin":
        st.subheader("📝 Original Gherkin")
        sample_text = """Scenario: Update profile
Given user's logged in
When user goes to profile
And clicks edit
And types new email
And saves
Then profile updated
And message shown"""
    
    else:  # Validate & Fix
        st.subheader("📝 Gherkin to Validate")
        sample_text = """Scenario: Login
given user's on page
when he enters credentials
then should login
and he'll see dashboard"""
    
    input_text = st.text_area(
        "Enter your text:",
        height=400,
        placeholder=sample_text,
        key="input_text"
    )
    
    # Buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if app_mode == "Convert to Gherkin":
            action_button = st.button("🔄 Convert", type="primary", use_container_width=True)
        elif app_mode == "Enhance Existing Gherkin":
            action_button = st.button("✨ Enhance", type="primary", use_container_width=True)
        else:
            action_button = st.button("🔍 Validate", type="primary", use_container_width=True)
    
    with col_btn2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    with col_btn3:
        if app_mode == "Enhance Existing Gherkin":
            compare_button = st.button("🔀 Compare", use_container_width=True)

with col2:
    if app_mode == "Convert to Gherkin":
        st.subheader("✨ Gherkin Output")
    elif app_mode == "Enhance Existing Gherkin":
        st.subheader("✨ Enhanced Gherkin")
    else:
        st.subheader("✅ Validation Results")
    
    output_placeholder = st.empty()


def build_style_rules() -> str:
    """Build style rules based on checkbox settings"""
    rules = []
    
    if enforce_capitalization:
        rules.append("- Each step MUST start with a capital letter (e.g., 'The user' not 'the user')")
    
    if no_contractions:
        rules.append("- NO contractions allowed - expand all (e.g., 'I am' not 'I'm', 'do not' not 'don't', 'user is' not 'user's')")
    
    if check_punctuation:
        rules.append("- Proper punctuation required - no periods at the end of steps unless absolutely necessary")
    
    if enforce_grammar:
        rules.append("- Correct grammar and sentence structure required")
    
    if check_flow:
        rules.append("- Ensure logical flow and proper context between steps")
    
    if use_present_tense:
        rules.append("- Use present tense consistently (e.g., 'clicks' not 'clicked', 'enters' not 'entered')")
    
    return "\n".join(rules)