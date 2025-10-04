import streamlit as st
from openai import OpenAI
import os
from pathlib import Path

# Load API key - works for both local and Streamlit Cloud
try:
    # Try Streamlit secrets first (for cloud deployment)
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    # Fallback to environment variable or dotenv (for local development)
    try:
        from dotenv import load_dotenv
        env_path = Path(r"C:\Users\pjone\OneDrive\Desktop\NewPython\test.env")
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv("OPENAI_API_KEY")
    except:
        api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is available
if not api_key:
    st.error("⚠️ OpenAI API key not found! Please set it in Streamlit secrets or your .env file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

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
        else:
            compare_button = False

with col2:
    if app_mode == "Convert to Gherkin":
        st.subheader("✨ Gherkin Output")
    elif app_mode == "Enhance Existing Gherkin":
        st.subheader("✨ Enhanced Gherkin")
    else:
        st.subheader("✅ Validation Results")


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
    
    return "\n".join(rules) if rules else "Standard Gherkin formatting"


def convert_with_openai(steps_text: str, scenario_name: str = None, 
                       tags: list = None, include_scenario: bool = True,
                       include_feature: bool = False, feature_name: str = None,
                       feature_description: str = None) -> str:
    """Convert steps to Gherkin using OpenAI"""
    
    style_rules = build_style_rules()
    
    prompt = f"""Convert the following test steps into proper Gherkin format (Given/When/Then/And/But).

MANDATORY FORMATTING RULES:
{style_rules}

GHERKIN STRUCTURE RULES:
1. Use "Given" for preconditions and setup
2. Use "When" for actions and events
3. Use "Then" for expected outcomes and assertions
4. Use "And" for consecutive steps of the same type
5. Use "But" for negative assertions when appropriate
6. Each step should start with the appropriate keyword
7. Format with proper indentation (2 spaces before each step)
8. Steps should be clear, specific, and independently understandable

EXAMPLES OF CORRECT FORMATTING:
✅ Given the user is logged into the application
✅ When the user clicks the "Submit" button
✅ Then the user should see a confirmation message
❌ given user's logged in (wrong: lowercase, contraction)
❌ when user clicked button (wrong: lowercase, past tense)

Test Steps:
{steps_text}

Please provide ONLY the converted Gherkin steps without any explanations or markdown code blocks."""

    if include_scenario:
        prompt += f"\n\nInclude this as a Scenario named: '{scenario_name}'"
        if tags:
            prompt += f"\nWith tags: {', '.join(tags)}"
    
    if include_feature:
        prompt += f"\n\nInclude this in a Feature named: '{feature_name}'"
        prompt += f"\nWith description: '{feature_description}'"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in BDD (Behavior Driven Development) and Gherkin syntax. You convert test steps into proper Gherkin format with strict adherence to grammar, capitalization, and style rules. You NEVER use contractions and always ensure proper capitalization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1200
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"❌ Error occurred:\n\n{str(e)}\n\nPlease check your OpenAI API key configuration."


def enhance_gherkin(gherkin_text: str, enhancement_options: list, 
                    enhancement_level: str) -> str:
    """Enhance existing Gherkin using OpenAI"""
    
    improvements = "\n".join([f"- {opt}" for opt in enhancement_options])
    style_rules = build_style_rules()
    
    level_instructions = {
        "Light": "Make minimal improvements while keeping the structure mostly intact.",
        "Moderate": "Improve clarity and consistency while maintaining the original intent.",
        "Comprehensive": "Thoroughly refine and restructure for maximum clarity and best practices."
    }
    
    prompt = f"""You are a BDD and Gherkin expert. Enhance the following Gherkin scenario.

MANDATORY FORMATTING RULES (MUST BE ENFORCED):
{style_rules}

SPECIFIC IMPROVEMENTS TO MAKE:
{improvements}

Enhancement Level: {enhancement_level}
{level_instructions[enhancement_level]}

BEST PRACTICES TO APPLY:
1. Proper capitalization at the start of each step
2. NO contractions whatsoever (I am, not I'm; do not, not don't; user is, not user's)
3. Use present tense consistently
4. Be specific and concrete in descriptions
5. Avoid technical jargon when possible
6. Use consistent language patterns
7. Make steps independently understandable
8. Ensure proper indentation (2 spaces)
9. Ensure logical flow and context between steps
10. Proper grammar throughout

EXAMPLES OF CORRECT VS INCORRECT:
✅ Given the user is logged into the application
❌ Given user's logged into the app (contraction, not capitalized properly)

✅ When the admin clicks the "Delete" button
❌ When admin clicked delete button (past tense, missing details)

✅ Then the system should display an error message
❌ Then it'll show error (contraction, vague)

Original Gherkin:
{gherkin_text}

Please provide the enhanced version without explanations or markdown code blocks. Only output the improved Gherkin."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in BDD and Gherkin syntax. You enhance and refine Gherkin scenarios to follow best practices with strict grammar and formatting rules. You NEVER use contractions and always enforce proper capitalization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"❌ Error occurred:\n\n{str(e)}\n\nPlease check your OpenAI API key configuration."


def validate_gherkin(gherkin_text: str, check_options: list) -> str:
    """Validate and fix Gherkin using OpenAI"""
    
    checks = "\n".join([f"- {opt}" for opt in check_options])
    style_rules = build_style_rules()
    
    prompt = f"""You are a Gherkin syntax validator and expert. Analyze the following Gherkin and provide detailed feedback.

MANDATORY FORMATTING RULES TO CHECK:
{style_rules}

VALIDATION CHECKS TO PERFORM:
{checks}

SPECIFIC ISSUES TO LOOK FOR:
1. Lowercase letters at the start of steps (MUST be capitalized)
2. Contractions (I'm, don't, user's, etc.) - MUST be expanded to full words
3. Past tense usage - MUST be present tense
4. Missing or incorrect punctuation
5. Grammar errors
6. Poor context or flow between steps
7. Vague or unclear language
8. Inconsistent formatting

Provide your response in this format:

=== ISSUES FOUND ===
[List each issue with severity: CRITICAL, MAJOR, or MINOR]
Example:
- CRITICAL: Line 2 uses lowercase "given" instead of "Given"
- MAJOR: Line 3 contains contraction "I'm" should be "I am"
- MINOR: Step could be more descriptive

=== FIXED VERSION ===
[Corrected Gherkin with all issues resolved]

=== IMPROVEMENTS MADE ===
- [Specific improvement 1]
- [Specific improvement 2]
...

=== STYLE COMPLIANCE ===
✅ Capitalization: [Pass/Fail]
✅ No Contractions: [Pass/Fail]
✅ Punctuation: [Pass/Fail]
✅ Grammar: [Pass/Fail]
✅ Context & Flow: [Pass/Fail]
✅ Present Tense: [Pass/Fail]

Gherkin to Validate:
{gherkin_text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a meticulous Gherkin syntax validator. You identify grammar, punctuation, capitalization, and style issues with precision. You NEVER allow contractions or lowercase letters at the start of steps."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"❌ Error occurred:\n\n{str(e)}\n\nPlease check your OpenAI API key configuration."


def convert_rule_based(steps_text: str, scenario_name: str = None,
                      tags: list = None, include_scenario: bool = True,
                      include_feature: bool = False, feature_name: str = None,
                      feature_description: str = None) -> str:
    """Rule-based conversion (fallback method)"""
    
    given_keywords = ['have', 'has', 'exist', 'logged in', 'authenticated', 'is on', 'navigates to']
    when_keywords = ['click', 'press', 'submit', 'enter', 'type', 'select', 'update', 'delete', 'add']
    then_keywords = ['should', 'must', 'will', 'see', 'display', 'show', 'receive']
    
    steps = [step.strip() for step in steps_text.split('\n') if step.strip()]
    
    result = []
    
    if include_feature:
        result.append(f"Feature: {feature_name}")
        result.append(f"  {feature_description}")
        result.append("")
    
    if include_scenario:
        if tags:
            result.append(" ".join(f"@{tag}" for tag in tags))
        result.append(f"Scenario: {scenario_name}")
    
    previous_keyword = None
    
    for step in steps:
        # Apply style rules
        if no_contractions:
            # Expand common contractions
            contractions = {
                "i'm": "I am", "i've": "I have", "i'll": "I will", "i'd": "I would",
                "you're": "you are", "you've": "you have", "you'll": "you will",
                "he's": "he is", "she's": "she is", "it's": "it is",
                "we're": "we are", "we've": "we have", "we'll": "we will",
                "they're": "they are", "they've": "they have", "they'll": "they will",
                "isn't": "is not", "aren't": "are not", "wasn't": "was not",
                "weren't": "were not", "hasn't": "has not", "haven't": "have not",
                "won't": "will not", "wouldn't": "would not", "don't": "do not",
                "doesn't": "does not", "didn't": "did not", "can't": "cannot",
                "couldn't": "could not", "shouldn't": "should not",
                "user's": "user is", "system's": "system is"
            }
            for contraction, expansion in contractions.items():
                step = step.replace(contraction, expansion)
                step = step.replace(contraction.capitalize(), expansion.capitalize())
        
        if enforce_capitalization:
            step = step[0].upper() + step[1:] if step else step
        
        step_lower = step.lower()
        
        # Detect keyword
        if any(kw in step_lower for kw in then_keywords):
            keyword = "Then"
        elif any(kw in step_lower for kw in when_keywords):
            keyword = "When"
        elif any(kw in step_lower for kw in given_keywords):
            keyword = "Given"
        else:
            keyword = "Given"  # Default
        
        # Use And if same as previous
        if previous_keyword == keyword:
            result.append(f"  And {step}")
        else:
            result.append(f"  {keyword} {step}")
            previous_keyword = keyword
    
    return "\n".join(result)


# Handle button clicks
if clear_button:
    st.rerun()

if action_button:
    if not input_text:
        st.error("Please enter some text!")
    else:
        with col2:
            with st.spinner("Processing with grammar and style checks..."):
                
                if app_mode == "Convert to Gherkin":
                    if conversion_mode == "AI-Powered (OpenAI)":
                        output = convert_with_openai(
                            input_text,
                            scenario_name if include_scenario else None,
                            tags if include_scenario else None,
                            include_scenario,
                            include_feature,
                            feature_name if include_feature else None,
                            feature_description if include_feature else None
                        )
                    else:
                        output = convert_rule_based(
                            input_text,
                            scenario_name if include_scenario else None,
                            tags if include_scenario else None,
                            include_scenario,
                            include_feature,
                            feature_name if include_feature else None,
                            feature_description if include_feature else None
                        )
                
                elif app_mode == "Enhance Existing Gherkin":
                    output = enhance_gherkin(input_text, enhancement_options, enhancement_level)
                
                else:  # Validate & Fix
                    output = validate_gherkin(input_text, check_options)
                
                # Display output
                st.text_area("Result:", value=output, height=400, key="output", label_visibility="collapsed")
                
                # Download button
                st.download_button(
                    label="📥 Download .feature file",
                    data=output,
                    file_name="scenario.feature",
                    mime="text/plain",
                    use_container_width=True
                )

# Comparison feature for enhancement mode
if app_mode == "Enhance Existing Gherkin" and compare_button:
    if input_text:
        with st.spinner("Generating comparison with grammar checks..."):
            enhanced = enhance_gherkin(input_text, enhancement_options, enhancement_level)
            
            st.subheader("🔀 Side-by-Side Comparison")
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("**Original**")
                st.code(input_text, language="gherkin")
            
            with comp_col2:
                st.markdown("**Enhanced (Grammar & Style Corrected)**")
                st.code(enhanced, language="gherkin")
            
            # Show what was fixed
            st.info("✨ **Key Improvements:** Capitalization, contractions removed, grammar corrected, present tense enforced, proper flow established")

# Footer
st.divider()

# Display active rules
st.markdown("### 📋 Active Style Rules")
rule_cols = st.columns(3)
with rule_cols[0]:
    if enforce_capitalization:
        st.markdown("✅ **Proper Capitalization**")
    if no_contractions:
        st.markdown("✅ **No Contractions**")

with rule_cols[1]:
    if check_punctuation:
        st.markdown("✅ **Punctuation Check**")
    if enforce_grammar:
        st.markdown("✅ **Grammar Enforcement**")

with rule_cols[2]:
    if check_flow:
        st.markdown("✅ **Context & Flow**")
    if use_present_tense:
        st.markdown("✅ **Present Tense**")

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🥒 Gherkin Assistant Pro | Grammar-Aware | Powered by OpenAI & Streamlit</p>
</div>
""", unsafe_allow_html=True)