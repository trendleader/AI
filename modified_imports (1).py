# MUST BE FIRST - Set page config before any other Streamlit commands
import streamlit as st
st.set_page_config(
    page_title="AI Resume & Interview Coach Pro",
    page_icon="🤖",
    layout="wide"
)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import PyPDF2
from docx import Document
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import random
import time
import requests
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional
import hashlib
import os

# Handle dotenv import gracefully
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    st.warning("python-dotenv not available. Using Streamlit secrets for API keys.")

# Handle OpenAI import gracefully
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.error("OpenAI package not available. Please install: pip install openai")

# Initialize OpenAI client
openai_client = None
if OPENAI_AVAILABLE:
    # Try to get API key from multiple sources
    api_key = None
    
    # First try Streamlit secrets (recommended for deployment)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("✅ OpenAI API connected via Streamlit secrets!")
    except:
        pass
    
    # Then try environment variables if dotenv is available
    if not api_key and DOTENV_AVAILABLE:
        env_path = r"C:\Users\pjone\OneDrive\Desktop\NewPython\test.env"
        if os.path.exists(env_path):
            load_dotenv(env_path)
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                st.success("✅ OpenAI API connected via environment file!")
    
    # Finally try direct environment variable
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            st.success("✅ OpenAI API connected via environment variable!")
    
    if api_key:
        try:
            openai.api_key = api_key
            openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"OpenAI connection error: {str(e)}")
    else:
        st.warning("⚠️ OpenAI API key not found. Some AI features will use fallback methods.")
else:
    st.error("OpenAI package not installed. AI features will use fallback methods.")