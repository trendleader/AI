import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import docx
import PyPDF2
from io import BytesIO

# Load environment variables from the specified path
env_path = Path(r"C:\Users\pjone\OneDrive\Desktop\NewPython\test.env")
load_dotenv(dotenv_path=env_path)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    """Extract text from Word document"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def generate_document(doc_type, source_text):
    """Generate BRD, FRD, or Acceptance Criteria using OpenAI"""
    
    prompts = {
        "BRD": """You are a business analyst expert. Based on the following business case document, create a comprehensive Business Requirements Document (BRD).

The BRD should include:
1. Executive Summary
2. Business Objectives
3. Scope (In-Scope and Out-of-Scope)
4. Stakeholders
5. Business Requirements (detailed list)
6. Assumptions and Constraints
7. Success Criteria

Source Document:
{source_text}

Please provide a well-structured, professional BRD.""",

        "FRD": """You are a technical business analyst expert. Based on the following business case document, create a comprehensive Functional Requirements Document (FRD).

The FRD should include:
1. Introduction and Purpose
2. System Overview
3. Functional Requirements (detailed, numbered list with descriptions)
4. User Interface Requirements
5. Data Requirements
6. Integration Requirements
7. Performance Requirements
8. Security Requirements

Source Document:
{source_text}

Please provide a well-structured, professional FRD with specific, measurable functional requirements.""",

        "Acceptance Criteria": """You are a QA and business analyst expert. Based on the following business case document, create comprehensive Acceptance Criteria.

The Acceptance Criteria should include:
1. User Stories with acceptance criteria in Given-When-Then format
2. Test Scenarios
3. Success Metrics
4. Edge Cases and Negative Scenarios
5. Performance Benchmarks

Source Document:
{source_text}

Please provide detailed, testable acceptance criteria that can be used for validation."""
    }
    
    prompt = prompts[doc_type].format(source_text=source_text)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert business analyst with deep experience in creating professional business and technical documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating {doc_type}: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Business Requirements Generator", page_icon="📋", layout="wide")

st.title("📋 Business Requirements Document Generator")
st.markdown("Upload a business case or overview document to automatically generate BRD, FRD, and Acceptance Criteria")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This tool uses OpenAI to analyze your business case documents and generate:
    - **BRD**: Business Requirements Document
    - **FRD**: Functional Requirements Document
    - **Acceptance Criteria**: Test scenarios and success metrics
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Upload a Word (.docx) or PDF file
    2. Click 'Generate Documents'
    3. Review and download the results
    """)

# Main content
uploaded_file = st.file_uploader("Upload Business Case Document", type=['pdf', 'docx'])

if uploaded_file is not None:
    # Display file info
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Extract text based on file type
    with st.spinner("Extracting text from document..."):
        if uploaded_file.name.endswith('.pdf'):
            source_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.docx'):
            source_text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file format")
            source_text = None
    
    if source_text:
        # Show preview of extracted text
        with st.expander("Preview Extracted Text"):
            st.text_area("Document Content", source_text, height=200)
        
        # Generate button
        if st.button("🚀 Generate Documents", type="primary"):
            
            # Create tabs for each document type
            tab1, tab2, tab3 = st.tabs(["📄 BRD", "⚙️ FRD", "✅ Acceptance Criteria"])
            
            with tab1:
                with st.spinner("Generating Business Requirements Document..."):
                    brd = generate_document("BRD", source_text)
                    st.markdown(brd)
                    st.download_button(
                        label="Download BRD",
                        data=brd,
                        file_name="BRD.txt",
                        mime="text/plain"
                    )
            
            with tab2:
                with st.spinner("Generating Functional Requirements Document..."):
                    frd = generate_document("FRD", source_text)
                    st.markdown(frd)
                    st.download_button(
                        label="Download FRD",
                        data=frd,
                        file_name="FRD.txt",
                        mime="text/plain"
                    )
            
            with tab3:
                with st.spinner("Generating Acceptance Criteria..."):
                    ac = generate_document("Acceptance Criteria", source_text)
                    st.markdown(ac)
                    st.download_button(
                        label="Download Acceptance Criteria",
                        data=ac,
                        file_name="Acceptance_Criteria.txt",
                        mime="text/plain"
                    )
            
            st.success("✅ All documents generated successfully!")

else:
    st.info("👆 Please upload a business case document to get started")

# Footer
st.markdown("---")
st.markdown("*Powered by OpenAI GPT-4*")