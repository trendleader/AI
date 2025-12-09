import streamlit as st
from openai import OpenAI
from huggingface_hub import InferenceClient
import os
from datetime import datetime
import PyPDF2
import docx
import io

# Page configuration
st.set_page_config(
    page_title="Legal Research Assistant - With Document Chat",
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
    .document-box {
        background-color: #fff8e1 !important;
        padding: 1rem;
        border-left: 4px solid #ffa000;
        margin: 1rem 0;
        border-radius: 0.3rem;
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
    "Contracts - Formation": {
        "overview": "Research contract formation requirements",
        "elements": ["Offer", "Acceptance", "Consideration", "Mutual Assent"],
        "key_concepts": ["Mailbox Rule", "Mirror Image Rule", "Bargain Theory", "Promissory Estoppel"]
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
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = {}
if 'document_chat_history' not in st.session_state:
    st.session_state.document_chat_history = []
if 'openai_validated' not in st.session_state:
    st.session_state.openai_validated = False
if 'hf_validated' not in st.session_state:
    st.session_state.hf_validated = False

# Document processing functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        return txt_file.read().decode('utf-8')
    except Exception as e:
        return f"Error reading TXT: {str(e)}"

# Validation functions
def validate_openai_key(api_key):
    """Test OpenAI API key with a simple request"""
    if not api_key or len(api_key) < 20:
        return False, "API key appears to be invalid (too short)"
    
    try:
        client = OpenAI(api_key=api_key)
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
        else:
            return False, f"Error: {error_msg[:100]}"

def validate_huggingface_key(api_key):
    """Test Hugging Face API key with a simple request"""
    if not api_key or len(api_key) < 20:
        return False, "API key appears to be invalid (too short)"
    
    if not api_key.startswith('hf_'):
        return False, "Hugging Face tokens should start with 'hf_'"
    
    try:
        client = InferenceClient(api_key=api_key)
        response = client.text_generation(
            "Hello",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=10,
            timeout=30
        )
        return True, "Hugging Face API key validated successfully!"
    except Exception as e:
        return False, f"Validation error: {str(e)[:150]}"

# Header
st.markdown('<p class="main-header">⚖️ Legal Research Assistant - With Document Chat</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Research Legal Topics & Chat With Your Legal Documents</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 AI Provider Selection")
    
    ai_provider = st.radio(
        "Select AI Provider:",
        ["OpenAI", "Hugging Face"],
        help="Choose which AI service to use"
    )
    
    st.markdown("---")
    
    # API Key configuration
    if ai_provider == "OpenAI":
        st.subheader("🔑 OpenAI Configuration")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your API key at https://platform.openai.com/api-keys",
            key="openai_key_input"
        )
        
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
                if is_valid:
                    st.success(message)
                else:
                    st.error(message)
        
        if openai_api_key and st.session_state.openai_validated:
            st.markdown("""
            <div class='success-box'>
            ✅ <strong>OpenAI API Key: VALIDATED</strong><br>
            Ready to use!
            </div>
            """, unsafe_allow_html=True)
        
        api_key = openai_api_key
        
        model_choice = st.selectbox(
            "Select Model:",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=1,
            help="GPT-4o-mini recommended for best value"
        )
        
    else:  # Hugging Face
        st.subheader("🔑 Hugging Face Configuration")
        hf_api_key = st.text_input(
            "Hugging Face API Key",
            type="password",
            help="Get your API key at https://huggingface.co/settings/tokens",
            key="hf_key_input"
        )
        
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
                if is_valid:
                    st.success(message)
                else:
                    st.error(message)
        
        if hf_api_key and st.session_state.hf_validated:
            st.markdown("""
            <div class='success-box'>
            ✅ <strong>Hugging Face API Key: VALIDATED</strong><br>
            Ready to use!
            </div>
            """, unsafe_allow_html=True)
        
        api_key = hf_api_key
        
        model_choice = st.selectbox(
            "Select Model:",
            ["mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"],
            index=0
        )
    
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    st.markdown("---")
    
    # Document upload section
    st.header("📄 Upload Legal Documents")
    
    uploaded_files = st.file_uploader(
        "Upload your legal documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload cases, notes, textbooks, or exam materials"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_documents:
                # Process the file
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    if file_extension == 'pdf':
                        text = extract_text_from_pdf(uploaded_file)
                    elif file_extension == 'docx':
                        text = extract_text_from_docx(uploaded_file)
                    elif file_extension == 'txt':
                        text = extract_text_from_txt(uploaded_file)
                    else:
                        text = "Unsupported file type"
                    
                    st.session_state.uploaded_documents[uploaded_file.name] = {
                        'text': text,
                        'uploaded_at': datetime.now(),
                        'type': file_extension
                    }
        
        st.success(f"✅ {len(st.session_state.uploaded_documents)} document(s) loaded")
        
        # Show uploaded documents
        with st.expander(f"📚 Loaded Documents ({len(st.session_state.uploaded_documents)})"):
            for doc_name, doc_info in st.session_state.uploaded_documents.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {doc_name}")
                    st.caption(f"Uploaded: {doc_info['uploaded_at'].strftime('%I:%M %p')}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc_name}"):
                        del st.session_state.uploaded_documents[doc_name]
                        st.rerun()
    
    if st.button("🗑️ Clear All Documents"):
        st.session_state.uploaded_documents = {}
        st.session_state.document_chat_history = []
        st.rerun()

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🔍 Research", "💬 Chat with Documents", "📝 Notes"])

with tab1:
    st.header("General Legal Research")
    
    template_choice = st.selectbox(
        "Select a legal topic template:",
        ["Custom Query"] + list(LEGAL_TEMPLATES.keys())
    )
    
    user_query = st.text_area(
        "Enter your legal research question:",
        height=120,
        placeholder="Example: What are the elements of negligence? or Explain the mailbox rule."
    )
    
    if st.button("🔎 Research", type="primary"):
        if not api_key:
            st.error(f"⚠️ Please enter and validate your {ai_provider} API key in the sidebar.")
        elif not user_query:
            st.error("⚠️ Please enter a research question.")
        else:
            with st.spinner(f"🔍 Researching with {ai_provider}..."):
                # Create prompt
                prompt = f"""You are an expert legal educator helping a law student.

Question: {user_query}

Provide a clear, educational response with:
1. Clear definitions of legal terms
2. Relevant legal rules and tests
3. Examples or case references where helpful
4. Well-organized structure

Format your response for easy studying."""

                try:
                    if ai_provider == "OpenAI":
                        client = OpenAI(api_key=api_key)
                        response = client.chat.completions.create(
                            model=model_choice,
                            messages=[
                                {"role": "system", "content": "You are an expert legal educator."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=2000,
                            temperature=temperature
                        )
                        answer = response.choices[0].message.content
                    else:
                        client = InferenceClient(api_key=api_key)
                        answer = client.text_generation(
                            prompt,
                            model=model_choice,
                            max_new_tokens=2000,
                            temperature=temperature
                        )
                    
                    st.session_state.chat_history.insert(0, {
                        "timestamp": datetime.now(),
                        "query": user_query,
                        "response": answer,
                        "provider": ai_provider
                    })
                    
                    st.markdown("### 📚 Research Results")
                    st.markdown(f"<div class='legal-box'>{answer}</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.header("💬 Chat with Your Legal Documents")
    
    if not st.session_state.uploaded_documents:
        st.info("""
        📄 **No documents uploaded yet!**
        
        Upload your legal documents in the sidebar to:
        - Ask questions about cases
        - Get summaries of chapters
        - Analyze exam hypotheticals
        - Create study notes from readings
        - Compare multiple cases
        """)
    else:
        # Show active documents
        st.markdown(f"**📚 Active Documents:** {len(st.session_state.uploaded_documents)}")
        doc_names = ", ".join(list(st.session_state.uploaded_documents.keys())[:3])
        if len(st.session_state.uploaded_documents) > 3:
            doc_names += f" and {len(st.session_state.uploaded_documents) - 3} more..."
        st.caption(doc_names)
        
        st.markdown("---")
        
        # Document chat interface
        document_query = st.text_area(
            "Ask a question about your documents:",
            height=100,
            placeholder="Examples:\n- Summarize the key holding in this case\n- What are the elements discussed in this chapter?\n- Create an outline from these notes\n- What's the issue in this hypothetical?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            ask_button = st.button("💬 Ask Question", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.document_chat_history = []
                st.rerun()
        
        if ask_button:
            if not api_key:
                st.error(f"⚠️ Please enter and validate your {ai_provider} API key.")
            elif not document_query:
                st.error("⚠️ Please enter a question.")
            else:
                with st.spinner("🤔 Analyzing your documents..."):
                    # Combine all document texts
                    combined_docs = "\n\n---DOCUMENT SEPARATOR---\n\n".join([
                        f"DOCUMENT: {name}\n\n{info['text'][:8000]}"  # Limit each doc to 8000 chars
                        for name, info in st.session_state.uploaded_documents.items()
                    ])
                    
                    # Create prompt with document context
                    doc_prompt = f"""You are a legal education assistant helping a law student study from their course materials.

UPLOADED DOCUMENTS:
{combined_docs}

STUDENT'S QUESTION: {document_query}

Please provide a helpful answer based on the documents above. If the documents contain relevant information, cite specific parts. If they don't contain the answer, let the student know and provide general legal knowledge if helpful.

Format your response clearly for studying."""
                    
                    try:
                        if ai_provider == "OpenAI":
                            client = OpenAI(api_key=api_key)
                            response = client.chat.completions.create(
                                model=model_choice,
                                messages=[
                                    {"role": "system", "content": "You are a helpful legal education assistant."},
                                    {"role": "user", "content": doc_prompt}
                                ],
                                max_tokens=2500,
                                temperature=temperature
                            )
                            answer = response.choices[0].message.content
                        else:
                            client = InferenceClient(api_key=api_key)
                            answer = client.text_generation(
                                doc_prompt,
                                model=model_choice,
                                max_new_tokens=2500,
                                temperature=temperature
                            )
                        
                        st.session_state.document_chat_history.append({
                            "timestamp": datetime.now(),
                            "question": document_query,
                            "answer": answer,
                            "docs_used": len(st.session_state.uploaded_documents)
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Display chat history
        if st.session_state.document_chat_history:
            st.markdown("---")
            st.markdown("### 💬 Conversation History")
            
            for idx, chat in enumerate(reversed(st.session_state.document_chat_history)):
                with st.container():
                    st.markdown(f"**📝 You asked:** {chat['question']}")
                    st.markdown(f"<div class='document-box'>{chat['answer']}</div>", unsafe_allow_html=True)
                    st.caption(f"🕐 {chat['timestamp'].strftime('%I:%M %p')} | 📚 Used {chat['docs_used']} document(s)")
                    
                    if st.button("💾 Save to Notes", key=f"save_doc_chat_{idx}"):
                        note = {
                            "type": "document_chat",
                            "timestamp": chat['timestamp'],
                            "question": chat['question'],
                            "answer": chat['answer']
                        }
                        st.session_state.research_notes.append(note)
                        st.success("✓ Saved to notes!")
                    
                    st.markdown("---")

with tab3:
    st.header("📝 Research Notes")
    
    if st.session_state.research_notes:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📥 Export All Notes", use_container_width=True):
                notes_text = ""
                for note in st.session_state.research_notes:
                    notes_text += f"{'='*60}\n"
                    notes_text += f"{note.get('type', 'research')} - {note['timestamp'].strftime('%Y-%m-%d %I:%M %p')}\n"
                    notes_text += f"{'='*60}\n\n"
                    
                    if note.get('type') == 'document_chat':
                        notes_text += f"Question: {note['question']}\n\n"
                        notes_text += f"Answer:\n{note['answer']}\n\n\n"
                    else:
                        notes_text += f"Query: {note.get('query', note.get('question', ''))}\n\n"
                        notes_text += f"{note.get('response', note.get('answer', ''))}\n\n\n"
                
                st.download_button(
                    "Download Notes (.txt)",
                    notes_text,
                    file_name=f"legal_research_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
        
        for idx, note in enumerate(st.session_state.research_notes):
            note_type = note.get('type', 'research')
            icon = "💬" if note_type == 'document_chat' else "🔍"
            with st.expander(f"{icon} {note_type.title()} - {note['timestamp'].strftime('%b %d, %I:%M %p')}"):
                if note_type == 'document_chat':
                    st.markdown(f"**Question:** {note['question']}")
                    st.markdown(note['answer'])
                else:
                    st.markdown(f"**Query:** {note.get('query', '')}")
                    st.markdown(note.get('response', ''))
                
                if st.button("🗑️ Delete", key=f"del_note_{idx}"):
                    st.session_state.research_notes.pop(idx)
                    st.rerun()
    else:
        st.info("No saved notes yet. Save research results or document chats to build your study materials!")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>Legal Research Assistant with Document Chat</strong> | Educational Tool for Law Students</p>
<p style='font-size: 0.9em;'>⚠️ <em>For educational purposes only. Not legal advice. Always verify with authoritative sources.</em></p>
</div>
""", unsafe_allow_html=True)
