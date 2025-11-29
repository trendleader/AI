import streamlit as st
from openai import OpenAI
from huggingface_hub import InferenceClient
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Legal Research Assistant - Dual AI",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4788;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .legal-box {
        background-color: #ffffff !important;
        padding: 1.5rem;
        border-left: 4px solid #1f4788;
        margin: 1rem 0;
        border-radius: 0.3rem;
        color: #000000 !important;
    }
    .legal-box * {
        color: #000000 !important;
    }
    .case-box {
        background-color: #f0f7ff !important;
        padding: 1.5rem;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
        border-radius: 0.3rem;
        color: #000000 !important;
    }
    .case-box * {
        color: #000000 !important;
    }
    .api-info {
        background-color: #e7f3ff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9em;
        color: #1e1e1e !important;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724 !important;
        padding: 0.75rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24 !important;
        padding: 0.75rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Legal topic templates
LEGAL_TEMPLATES = {
    "Torts - Negligence": {
        "overview": "Research the four elements of negligence",
        "elements": ["Duty of Care", "Breach of Duty", "Causation (Actual and Proximate)", "Damages"],
        "key_concepts": ["Reasonable Person Standard", "Res Ipsa Loquitur", "Negligence Per Se", "Comparative/Contributory Negligence"]
    },
    "Torts - Intentional Torts": {
        "overview": "Research intentional torts to persons and property",
        "elements": ["Intent", "Act", "Causation", "Harm/Offense"],
        "key_concepts": ["Battery", "Assault", "False Imprisonment", "IIED", "Trespass to Land", "Conversion"]
    },
    "Contracts - Formation": {
        "overview": "Research contract formation requirements",
        "elements": ["Offer", "Acceptance", "Consideration", "Mutual Assent"],
        "key_concepts": ["Mailbox Rule", "Mirror Image Rule", "Bargain Theory", "Promissory Estoppel"]
    },
    "Contracts - Defenses": {
        "overview": "Research defenses to contract enforcement",
        "elements": ["Lack of Capacity", "Duress", "Undue Influence", "Mistake", "Fraud", "Illegality"],
        "key_concepts": ["Statute of Frauds", "Unconscionability", "Impossibility", "Frustration of Purpose"]
    },
    "Criminal Law - Elements": {
        "overview": "Research general elements of crimes",
        "elements": ["Actus Reus", "Mens Rea", "Concurrence", "Causation"],
        "key_concepts": ["Specific Intent", "General Intent", "Strict Liability", "Accomplice Liability"]
    },
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'research_notes' not in st.session_state:
    st.session_state.research_notes = []
if 'openai_validated' not in st.session_state:
    st.session_state.openai_validated = False
if 'hf_validated' not in st.session_state:
    st.session_state.hf_validated = False
if 'openai_error' not in st.session_state:
    st.session_state.openai_error = None
if 'hf_error' not in st.session_state:
    st.session_state.hf_error = None

# Header
st.markdown('<p class="main-header">⚖️ Legal Research Assistant - Dual AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Choose Between OpenAI or Hugging Face Models</p>', unsafe_allow_html=True)

# Validation functions
def validate_openai_key(api_key):
    """Test OpenAI API key with a simple request"""
    if not api_key or len(api_key) < 20:
        return False, "API key appears to be invalid (too short)"
    
    try:
        client = OpenAI(api_key=api_key)
        # Make a minimal test request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return True, "OpenAI API key validated successfully!"
    except Exception as e:
        error_msg = str(e)
        if "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
            return False, "Invalid API key. Please check your key and try again."
        elif "insufficient_quota" in error_msg:
            return False, "API key valid but insufficient quota/credits. Please add billing."
        elif "rate_limit" in error_msg:
            return False, "Rate limit exceeded. Please wait and try again."
        else:
            return False, f"Error: {error_msg[:100]}"

def validate_huggingface_key(api_key):
    """Test Hugging Face API key with a simple request"""
    if not api_key or len(api_key) < 20:
        return False, "API key appears to be invalid (too short)"
    
    try:
        client = InferenceClient(api_key=api_key)
        # Make a minimal test request
        response = client.text_generation(
            "test",
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_new_tokens=5
        )
        return True, "Hugging Face API key validated successfully!"
    except Exception as e:
        error_msg = str(e)
        if "Invalid token" in error_msg or "401" in error_msg:
            return False, "Invalid API key. Please check your key and try again."
        elif "rate" in error_msg.lower():
            return False, "Rate limit exceeded. Please wait and try again."
        else:
            return False, f"Error: {error_msg[:100]}"

# Sidebar
with st.sidebar:
    st.header("🔧 AI Provider Selection")
    
    # Choose AI provider
    ai_provider = st.radio(
        "Select AI Provider:",
        ["OpenAI", "Hugging Face"],
        help="Choose which AI service to use for research"
    )
    
    st.markdown("---")
    
    # API Key input and validation based on provider
    if ai_provider == "OpenAI":
        st.subheader("🔑 OpenAI Configuration")
        
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your API key at https://platform.openai.com/api-keys",
            key="openai_key_input"
        )
        
        # Validation button and status
        col1, col2 = st.columns([2, 1])
        with col1:
            validate_openai = st.button("🔍 Validate Key", key="validate_openai", use_container_width=True)
        with col2:
            if st.session_state.openai_validated:
                st.markdown("✅")
        
        if validate_openai and openai_api_key:
            with st.spinner("Validating OpenAI key..."):
                is_valid, message = validate_openai_key(openai_api_key)
                st.session_state.openai_validated = is_valid
                st.session_state.openai_error = None if is_valid else message
                
                if is_valid:
                    st.success(message)
                else:
                    st.error(message)
        
        # Show validation status
        if openai_api_key:
            if st.session_state.openai_validated:
                st.markdown("""
                <div class='success-box'>
                ✅ <strong>OpenAI API Key: VALIDATED</strong><br>
                Ready to use!
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.openai_error:
                st.markdown(f"""
                <div class='error-box'>
                ❌ <strong>Validation Failed</strong><br>
                {st.session_state.openai_error}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("👆 Click 'Validate Key' to test your API key")
        
        api_key = openai_api_key
        
        model_choice = st.selectbox(
            "Select Model:",
            [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ],
            index=1,  # Default to gpt-4o-mini (affordable)
            help="GPT-4o-mini recommended for best value"
        )
        
        st.markdown("""
        <div class='api-info'>
        <b>💰 Pricing:</b><br>
        • GPT-4o: Best quality<br>
        • GPT-4o-mini: Fast & affordable ⭐<br>
        • GPT-3.5-turbo: Budget option
        </div>
        """, unsafe_allow_html=True)
        
    else:  # Hugging Face
        st.subheader("🔑 Hugging Face Configuration")
        
        hf_api_key = st.text_input(
            "Hugging Face API Key",
            type="password",
            help="Get your API key at https://huggingface.co/settings/tokens",
            key="hf_key_input"
        )
        
        # Validation button and status
        col1, col2 = st.columns([2, 1])
        with col1:
            validate_hf = st.button("🔍 Validate Key", key="validate_hf", use_container_width=True)
        with col2:
            if st.session_state.hf_validated:
                st.markdown("✅")
        
        if validate_hf and hf_api_key:
            with st.spinner("Validating Hugging Face key..."):
                is_valid, message = validate_huggingface_key(hf_api_key)
                st.session_state.hf_validated = is_valid
                st.session_state.hf_error = None if is_valid else message
                
                if is_valid:
                    st.success(message)
                else:
                    st.error(message)
        
        # Show validation status
        if hf_api_key:
            if st.session_state.hf_validated:
                st.markdown("""
                <div class='success-box'>
                ✅ <strong>Hugging Face API Key: VALIDATED</strong><br>
                Ready to use!
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.hf_error:
                st.markdown(f"""
                <div class='error-box'>
                ❌ <strong>Validation Failed</strong><br>
                {st.session_state.hf_error}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("👆 Click 'Validate Key' to test your API key")
        
        api_key = hf_api_key
        
        model_choice = st.selectbox(
            "Select Model:",
            [
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "meta-llama/Meta-Llama-3-8B-Instruct",
            ],
            index=0,
            help="Mixtral-8x7B recommended for best results"
        )
        
        st.markdown("""
        <div class='api-info'>
        <b>🆓 Free Tier Available!</b><br>
        Hugging Face offers free API access with rate limits.
        </div>
        """, unsafe_allow_html=True)
    
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    st.markdown("---")
    st.header("📚 Research Template")
    
    template_choice = st.selectbox(
        "Select a legal topic template:",
        ["Custom Query"] + list(LEGAL_TEMPLATES.keys())
    )
    
    if template_choice != "Custom Query":
        template = LEGAL_TEMPLATES[template_choice]
        st.info(f"**Overview:** {template['overview']}")
        
        with st.expander("📋 Elements to Research"):
            for element in template['elements']:
                st.write(f"• {element}")
        
        with st.expander("🔑 Key Concepts"):
            for concept in template['key_concepts']:
                st.write(f"• {concept}")
    
    st.markdown("---")
    st.header("🎯 Research Tools")
    
    research_tool = st.radio(
        "Choose research tool:",
        [
            "💡 Concept Explanation",
            "⚖️ Case Analysis",
            "📊 Element Breakdown",
            "🧩 Hypothetical Solver",
            "📝 Exam Outline",
        ]
    )
    
    st.markdown("---")
    st.header("⚙️ Settings")
    
    detail_level = st.select_slider(
        "Response Detail Level:",
        options=["Brief", "Moderate", "Detailed", "Comprehensive"],
        value="Detailed"
    )
    
    include_cases = st.checkbox("Include Case Examples", value=True)
    include_examples = st.checkbox("Include Hypotheticals", value=True)
    
    st.markdown("---")
    
    # Debug info
    with st.expander("🔧 Debug Info"):
        st.write(f"Provider: {ai_provider}")
        st.write(f"OpenAI Key Present: {bool(openai_api_key) if ai_provider == 'OpenAI' else 'N/A'}")
        st.write(f"HF Key Present: {bool(hf_api_key) if ai_provider == 'Hugging Face' else 'N/A'}")
        st.write(f"OpenAI Validated: {st.session_state.openai_validated}")
        st.write(f"HF Validated: {st.session_state.hf_validated}")
    
    if st.button("🗑️ Clear All History"):
        st.session_state.chat_history = []
        st.session_state.research_notes = []
        st.rerun()

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🔍 Research", "📝 Notes", "📚 Resources"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Research Query")
        
        if template_choice != "Custom Query":
            template = LEGAL_TEMPLATES[template_choice]
            default_query = f"Explain {template_choice}: {template['overview']}"
        else:
            default_query = ""
        
        user_query = st.text_area(
            "Enter your legal research question:",
            value=default_query,
            height=120,
            placeholder="Example: What are the elements of breach of contract? or Analyze the reasonable person standard in negligence."
        )
        
        with st.expander("➕ Advanced Options"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                jurisdiction = st.text_input(
                    "Jurisdiction:",
                    placeholder="e.g., Federal, CA, NY"
                )
                
                year_level = st.selectbox(
                    "Law School Year:",
                    ["1L", "2L", "3L"]
                )
            
            with col_b:
                course = st.text_input(
                    "Course:",
                    placeholder="e.g., Torts, Contracts"
                )
                
                focus_area = st.text_input(
                    "Specific Focus:",
                    placeholder="e.g., Exam prep, Paper"
                )
            
            context = st.text_area(
                "Additional Context:",
                height=80,
                placeholder="Add hypothetical facts, specific questions, or areas needing clarification."
            )
        
        submit_button = st.button("🔎 Research", type="primary", use_container_width=True)
    
    with col2:
        st.header("💡 Quick Actions")
        
        if st.button("📋 Generate Outline", use_container_width=True):
            if template_choice != "Custom Query":
                st.session_state.quick_action = f"Create a comprehensive study outline for {template_choice} including all elements, key concepts, and important cases."
        
        if st.button("🎯 Practice Hypo", use_container_width=True):
            if template_choice != "Custom Query":
                st.session_state.quick_action = f"Generate a practice hypothetical problem for {template_choice} with a detailed analysis."
        
        st.markdown("---")
        
        # Show API status
        if ai_provider == "OpenAI":
            if st.session_state.openai_validated:
                st.success(f"✅ {ai_provider} Ready")
            else:
                st.warning("⚠️ Validate your API key first!")
        else:
            if st.session_state.hf_validated:
                st.success(f"✅ {ai_provider} Ready")
            else:
                st.warning("⚠️ Validate your API key first!")
        
        st.info(f"""
        **Current Setup:**
        
        🤖 Provider: {ai_provider}
        
        📊 Model: {model_choice.split('/')[-1] if '/' in model_choice else model_choice}
        
        🎚️ Temperature: {temperature}
        """)

# Functions
def create_research_prompt(query, tool, detail, cases, examples, **kwargs):
    detail_instructions = {
        "Brief": "Provide a concise, focused response.",
        "Moderate": "Provide a balanced response with key details.",
        "Detailed": "Provide a thorough explanation with examples.",
        "Comprehensive": "Provide an exhaustive analysis with multiple examples and perspectives."
    }
    
    tool_instructions = {
        "💡 Concept Explanation": "Explain the legal concept clearly, defining terms and providing context.",
        "⚖️ Case Analysis": "Analyze relevant case law, including holdings, reasoning, and significance.",
        "📊 Element Breakdown": "Break down the legal test or rule into its component elements with explanations for each.",
        "🧩 Hypothetical Solver": "Analyze the hypothetical using IRAC (Issue, Rule, Analysis, Conclusion) methodology.",
        "📝 Exam Outline": "Create a structured outline suitable for exam preparation with clear organization.",
    }
    
    prompt = f"""You are an expert legal educator helping a law student understand legal concepts.

Research Tool: {tool}
{tool_instructions.get(tool, "")}

Detail Level: {detail}
{detail_instructions[detail]}

Student's Question: {query}

"""
    
    if kwargs.get('jurisdiction'):
        prompt += f"Jurisdiction: {kwargs['jurisdiction']}\n"
    
    if kwargs.get('course'):
        prompt += f"Course Context: {kwargs['course']}\n"
    
    if kwargs.get('context'):
        prompt += f"Additional Context: {kwargs['context']}\n"
    
    prompt += f"""
Include Case Examples: {"Yes" if cases else "No"}
Include Hypotheticals: {"Yes" if examples else "No"}

Please provide a response that:
1. Uses clear legal terminology while remaining accessible
2. Structures information with headings and organization
3. {"Cites landmark cases with brief explanations" if cases else ""}
4. {"Provides practical examples or hypotheticals" if examples else ""}
5. Breaks down complex concepts systematically
6. Highlights important distinctions and nuances
7. Is suitable for {kwargs.get('year_level', '1L')} study

Format your response professionally with clear structure."""
    
    return prompt

def query_openai_llm(prompt, api_key, model, temp, max_tokens=3000):
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert legal educator specializing in helping first-year law students understand complex legal concepts. Provide clear, structured, and educational responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temp
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"❌ Error with OpenAI API: {str(e)}\n\nPlease verify your API key is validated and try again."

def query_huggingface_llm(prompt, api_key, model, temp, max_tokens=2500):
    try:
        client = InferenceClient(api_key=api_key)
        
        response = client.text_generation(
            prompt,
            model=model,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=0.95,
            return_full_text=False
        )
        
        return response
    
    except Exception as e:
        return f"❌ Error with Hugging Face API: {str(e)}\n\nPlease verify your API key is validated and try again."

# Process research request
if submit_button or 'quick_action' in st.session_state:
    if 'quick_action' in st.session_state:
        user_query = st.session_state.quick_action
        del st.session_state.quick_action
    
    # Validation check before processing
    if not api_key:
        st.error(f"⚠️ Please enter your {ai_provider} API key in the sidebar.")
    elif ai_provider == "OpenAI" and not st.session_state.openai_validated:
        st.error("⚠️ Please validate your OpenAI API key before making queries. Click the 'Validate Key' button in the sidebar.")
    elif ai_provider == "Hugging Face" and not st.session_state.hf_validated:
        st.error("⚠️ Please validate your Hugging Face API key before making queries. Click the 'Validate Key' button in the sidebar.")
    elif not user_query:
        st.error("⚠️ Please enter a research question.")
    else:
        max_tokens = 3000
        
        with st.spinner(f"🔍 Conducting legal research with {ai_provider}..."):
            prompt = create_research_prompt(
                user_query,
                research_tool,
                detail_level,
                include_cases,
                include_examples,
                jurisdiction=jurisdiction,
                course=course,
                context=context,
                year_level=year_level,
                focus_area=focus_area
            )
            
            # Query the appropriate API based on provider
            if ai_provider == "OpenAI":
                response = query_openai_llm(prompt, api_key, model_choice, temperature, max_tokens)
            else:  # Hugging Face
                response = query_huggingface_llm(prompt, api_key, model_choice, temperature, max_tokens)
            
            st.session_state.chat_history.insert(0, {
                "timestamp": datetime.now(),
                "tool": research_tool,
                "template": template_choice,
                "query": user_query,
                "response": response,
                "detail": detail_level,
                "model": model_choice,
                "provider": ai_provider
            })
            
            st.rerun()

# Display research results
if st.session_state.chat_history:
    st.markdown("---")
    st.header("📚 Research Results")
    
    for idx, item in enumerate(st.session_state.chat_history):
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"### {item['tool']} - {item['timestamp'].strftime('%b %d, %I:%M %p')}")
                provider_icon = "🟢" if item['provider'] == "OpenAI" else "🔵"
                model_display = item['model'].split('/')[-1] if '/' in item['model'] else item['model']
                st.caption(f"{provider_icon} {item['provider']} | 📋 {item['template']} | 🤖 {model_display}")
            
            with col2:
                if st.button("💾 Save Note", key=f"save_{idx}", use_container_width=True):
                    note = {
                        "type": "research",
                        "timestamp": item['timestamp'],
                        "template": item['template'],
                        "query": item['query'],
                        "response": item['response'],
                        "provider": item['provider']
                    }
                    st.session_state.research_notes.append(note)
                    st.success("✓ Saved!")
            
            with col3:
                if st.button("🔄 Re-run", key=f"rerun_{idx}", use_container_width=True):
                    st.session_state.quick_action = item['query']
                    st.rerun()
            
            st.markdown(f"**Query:** {item['query']}")
            
            # Create a container with white background and dark text
            with st.container():
                st.markdown(f"""
                <div style='background-color: #ffffff; padding: 1.5rem; border-left: 4px solid #1f4788; 
                            border-radius: 0.5rem; margin: 1rem 0;'>
                    <div style='color: #1e1e1e; line-height: 1.6;'>
                        {item['response']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 5])
            with col1:
                st.button("👍", key=f"up_{idx}")
            with col2:
                st.button("👎", key=f"down_{idx}")
            
            st.markdown("---")

with tab2:
    st.header("📝 Research Notes")
    
    if st.session_state.research_notes:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📥 Export All Notes", use_container_width=True):
                notes_text = ""
                for note in st.session_state.research_notes:
                    notes_text += f"{'='*60}\n"
                    notes_text += f"{note['template']} - {note['timestamp'].strftime('%Y-%m-%d %I:%M %p')}\n"
                    notes_text += f"Provider: {note['provider']}\n"
                    notes_text += f"{'='*60}\n\n"
                    notes_text += f"Query: {note['query']}\n\n"
                    notes_text += f"{note['response']}\n\n\n"
                
                st.download_button(
                    "Download Notes (.txt)",
                    notes_text,
                    file_name=f"legal_research_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
        
        for idx, note in enumerate(st.session_state.research_notes):
            provider_icon = "🟢" if note['provider'] == "OpenAI" else "🔵"
            with st.expander(f"{provider_icon} {note['template']} - {note['timestamp'].strftime('%b %d, %I:%M %p')}"):
                st.markdown(f"**Query:** {note['query']}")
                st.markdown(note['response'])
                
                if st.button("🗑️ Delete", key=f"del_note_{idx}"):
                    st.session_state.research_notes.pop(idx)
                    st.rerun()
    else:
        st.info("No saved notes yet. Click '💾 Save Note' on any research result to save it here.")

with tab3:
    st.header("📚 Legal Research Resources & Setup Help")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Free Legal Databases")
        st.markdown("""
        - [Google Scholar](https://scholar.google.com/) - Case law search
        - [Cornell LII](https://www.law.cornell.edu/) - Legal Information Institute
        - [Justia](https://www.justia.com/) - Free case law & codes
        - [Oyez](https://www.oyez.org/) - Supreme Court multimedia
        """)
        
        st.subheader("🤖 OpenAI Setup")
        st.markdown("""
        **Get API Key:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
        
        **Steps:**
        1. Sign up for OpenAI account
        2. Add billing information
        3. Create API key
        4. Paste key in sidebar
        5. Click "Validate Key"
        6. Wait for ✅ confirmation
        
        **Troubleshooting:**
        - Key must start with "sk-"
        - Billing must be set up
        - Check usage limits
        """)
    
    with col2:
        st.subheader("💡 Study Tips")
        st.markdown("""
        **For First Year Students:**
        - Focus on understanding elements
        - Brief cases using IRAC method
        - Create outlines early
        - Practice hypotheticals regularly
        - Use multiple resources
        """)
        
        st.subheader("🔵 Hugging Face Setup")
        st.markdown("""
        **Get API Key:** [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        
        **Steps:**
        1. Sign up for Hugging Face
        2. Go to Settings → Access Tokens
        3. Create new token (Read access)
        4. Paste key in sidebar
        5. Click "Validate Key"
        6. Wait for ✅ confirmation
        
        **Troubleshooting:**
        - Key must start with "hf_"
        - Free tier has rate limits
        - Wait if rate limited
        """)

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
<p><strong>Legal Research Assistant - Dual AI Edition with Validation</strong></p>
<p style='font-size: 0.9em;'>⚠️ <em>For educational purposes only. Not legal advice. Always verify with authoritative sources.</em></p>
</div>
""", unsafe_allow_html=True)
