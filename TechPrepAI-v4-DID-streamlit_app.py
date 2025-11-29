"""
TechPrep AI - Technical Interview Practice Platform
Version 4.0 - With D-ID Talking Avatars & Voice Synthesis
"""

import streamlit as st
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any
import os
import requests
import base64

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="TechPrep AI - Interview Practice",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #0A0A0F;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #6B7280;
        font-size: 0.875rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* Avatar Container */
    .avatar-video-container {
        width: 200px;
        height: 200px;
        margin: 0 auto;
        border-radius: 50%;
        overflow: hidden;
        border: 4px solid #6366F1;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
    }
    
    .avatar-video-container video {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .avatar-image-container {
        width: 150px;
        height: 150px;
        margin: 0 auto;
        border-radius: 50%;
        overflow: hidden;
        border: 3px solid #6366F1;
        position: relative;
    }
    
    .avatar-image-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .avatar-speaking {
        animation: pulse-glow 1s ease-in-out infinite alternate;
    }
    
    @keyframes pulse-glow {
        0% {
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
            border-color: #6366F1;
        }
        100% {
            box-shadow: 0 0 40px rgba(99, 102, 241, 0.6);
            border-color: #818CF8;
        }
    }
    
    .speaking-indicator {
        display: flex;
        justify-content: center;
        gap: 6px;
        margin-top: 12px;
    }
    
    .speaking-dot {
        width: 10px;
        height: 10px;
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        border-radius: 50%;
        animation: bounce 0.6s ease-in-out infinite;
    }
    
    .speaking-dot:nth-child(2) { animation-delay: 0.15s; }
    .speaking-dot:nth-child(3) { animation-delay: 0.3s; }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Loading spinner for video generation */
    .video-loading {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 200px;
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid #1F1F2E;
        border-top-color: #6366F1;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Card styles */
    .interviewer-card {
        background-color: #111118;
        border: 1px solid #1F1F2E;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .interviewer-card:hover {
        border-color: #6366F1;
        transform: translateY(-2px);
    }
    
    .interviewer-card.selected {
        border-color: #6366F1;
        background-color: #13131D;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
    }
    
    .question-box {
        background-color: #1A1A24;
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid #6366F1;
    }
    
    .feedback-card {
        background-color: #111118;
        border: 1px solid #1F1F2E;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .coaching-card {
        background-color: #0F172A;
        border: 1px solid #1E3A5F;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .hint-card {
        background-color: #1E1B4B;
        border: 1px solid #312E81;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-connected {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-disconnected {
        background-color: rgba(239, 68, 68, 0.1);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Override Streamlit defaults */
    .stTextArea textarea {
        background-color: #111118 !important;
        border: 1px solid #1F1F2E !important;
        border-radius: 12px !important;
        color: #E5E5E5 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stTextInput input {
        background-color: #111118 !important;
        border: 1px solid #1F1F2E !important;
        border-radius: 8px !important;
        color: #E5E5E5 !important;
    }
    
    .stButton > button {
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0D0D12;
    }
</style>
""", unsafe_allow_html=True)

# D-ID Voice options
DID_VOICES = {
    "sarah": {
        "provider": "microsoft",
        "voice_id": "en-US-JennyNeural",
        "name": "Jenny (US Female)"
    },
    "marcus": {
        "provider": "microsoft", 
        "voice_id": "en-US-GuyNeural",
        "name": "Guy (US Male)"
    },
    "priya": {
        "provider": "microsoft",
        "voice_id": "en-IN-NeerjaNeural", 
        "name": "Neerja (Indian Female)"
    }
}

# Interviewer data with D-ID compatible images
INTERVIEWERS = [
    {
        "id": "sarah",
        "name": "Sarah Chen",
        "title": "Senior Data Engineer",
        "avatar": "👩‍💻",
        "style": "technical",
        "personality": "Direct and thorough. Focuses on technical accuracy and best practices.",
        "openai_voice": "nova",
        # High-quality portrait for D-ID (must be front-facing, good lighting)
        "image_url": "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?w=512&h=512&fit=crop&crop=face",
        "did_voice": DID_VOICES["sarah"],
        "feedback_templates": {
            "high": "Excellent technical depth. Your understanding of the underlying concepts is clear.",
            "medium": "Solid foundation. I'd like to see more attention to edge cases and optimization.",
            "low": "Let's work on strengthening the fundamentals here. Review the core concepts and try again."
        }
    },
    {
        "id": "marcus",
        "name": "Marcus Williams",
        "title": "Engineering Manager",
        "avatar": "👨‍💼",
        "style": "behavioral",
        "personality": "Warm but probing. Interested in problem-solving approach and communication.",
        "openai_voice": "onyx",
        "image_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512&h=512&fit=crop&crop=face",
        "did_voice": DID_VOICES["marcus"],
        "feedback_templates": {
            "high": "Great job communicating your thought process. You explained complex ideas clearly.",
            "medium": "Good approach to the problem. Try to be more explicit about trade-offs in your decisions.",
            "low": "I appreciate the effort. Focus on structuring your answer with a clear beginning, middle, and end."
        }
    },
    {
        "id": "priya",
        "name": "Priya Patel",
        "title": "Principal Architect",
        "avatar": "👩‍🔬",
        "style": "strategic",
        "personality": "Big-picture thinker. Values scalability and architectural decisions.",
        "openai_voice": "shimmer",
        "image_url": "https://images.unsplash.com/photo-1580489944761-15a19d654956?w=512&h=512&fit=crop&crop=face",
        "did_voice": DID_VOICES["priya"],
        "feedback_templates": {
            "high": "You're thinking at the right level. Good consideration of scalability and architecture.",
            "medium": "Nice start. Consider how this solution would work at 10x or 100x scale.",
            "low": "Let's zoom out a bit. Think about the bigger picture before diving into implementation details."
        }
    }
]

# Question Bank
QUESTION_BANK = {
    "sql": {
        "name": "SQL & Database",
        "icon": "🗃️",
        "questions": [
            {
                "id": "sql-1",
                "difficulty": "Medium",
                "question": "Write a SQL query to find all employees who earn more than their department average salary.",
                "hints": [
                    "Consider using a subquery or CTE to calculate department averages",
                    "You'll need to join the result back to the employees table",
                    "Think about using a correlated subquery or window function"
                ],
                "sample_answer": """SELECT e.employee_name, e.salary, e.department_id
FROM employees e
WHERE e.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.department_id = e.department_id
);""",
                "key_points": [
                    "Uses a correlated subquery to compare each employee to their department average",
                    "Alternative approaches include CTEs or window functions",
                    "Consider performance implications"
                ]
            },
            {
                "id": "sql-2",
                "difficulty": "Hard",
                "question": "Explain the difference between INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN.",
                "hints": [
                    "Think about what happens to non-matching rows",
                    "Consider scenarios with customer orders",
                    "NULL values play an important role"
                ],
                "sample_answer": """INNER JOIN: Returns only matching rows from both tables.
LEFT JOIN: All rows from left table + matching from right (NULLs for no match).
RIGHT JOIN: All rows from right table + matching from left.
FULL OUTER JOIN: All rows from both tables with NULLs where no match exists.""",
                "key_points": [
                    "Understanding NULL behavior is critical",
                    "LEFT JOIN is most commonly used for optional relationships",
                    "FULL OUTER JOIN useful for data reconciliation"
                ]
            }
        ]
    },
    "python": {
        "name": "Python",
        "icon": "🐍",
        "questions": [
            {
                "id": "py-1",
                "difficulty": "Medium",
                "question": "Explain the difference between a list and a tuple in Python. When would you choose one over the other?",
                "hints": [
                    "Think about mutability and its implications",
                    "Consider memory efficiency and performance",
                    "Think about use cases like dictionary keys"
                ],
                "sample_answer": """Lists are mutable (can be modified), tuples are immutable.

Use lists when you need to modify the collection.
Use tuples for fixed data, dictionary keys, or function returns.

Tuples are more memory efficient and can be used as dictionary keys.""",
                "key_points": [
                    "Immutability makes tuples hashable",
                    "Tuples have slight performance advantages",
                    "Lists are more flexible but use more memory"
                ]
            },
            {
                "id": "py-2",
                "difficulty": "Hard",
                "question": "What is a Python decorator? Create one that logs function execution time.",
                "hints": [
                    "A decorator wraps another function",
                    "Use @decorator syntax",
                    "You'll need the time module and *args, **kwargs"
                ],
                "sample_answer": """import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.4f}s")
        return result
    return wrapper""",
                "key_points": [
                    "Decorators modify function behavior without changing source",
                    "@wraps preserves function metadata",
                    "*args, **kwargs provide flexibility"
                ]
            }
        ]
    },
    "dax": {
        "name": "DAX & Power BI",
        "icon": "📊",
        "questions": [
            {
                "id": "dax-1",
                "difficulty": "Medium",
                "question": "What is the difference between CALCULATE and CALCULATETABLE in DAX?",
                "hints": [
                    "Think about what data type each returns",
                    "Consider how filter context modification works",
                    "When do you need a table vs a scalar value?"
                ],
                "sample_answer": """CALCULATE returns a scalar value (single number/text/date).
CALCULATETABLE returns a table.

Both modify filter context before evaluation.

CALCULATE example: Sales_2023 = CALCULATE(SUM(Sales[Amount]), Year = 2023)
CALCULATETABLE example: TopProducts = CALCULATETABLE(TOPN(5, Products, [Sales]))""",
                "key_points": [
                    "Both functions modify filter context",
                    "CALCULATE for measures, CALCULATETABLE for tables",
                    "Understanding filter context is essential"
                ]
            }
        ]
    },
    "system_design": {
        "name": "System Design",
        "icon": "🏗️",
        "questions": [
            {
                "id": "sd-1",
                "difficulty": "Hard",
                "question": "Design a URL shortening service like bit.ly. What are the key components?",
                "hints": [
                    "Consider ID generation strategy",
                    "Think about caching for frequently accessed URLs",
                    "Consider read vs write patterns"
                ],
                "sample_answer": """Key Components:
1. API Gateway - Rate limiting, load balancing
2. App Servers - Generate short codes, handle redirects
3. Database - Store URL mappings (SQL or NoSQL)
4. Cache (Redis) - Cache hot URLs for read-heavy workload
5. Analytics - Track clicks asynchronously

Short code: Base62 encoding, 7 chars = 3.5 trillion combinations""",
                "key_points": [
                    "Read-heavy system (100:1 ratio)",
                    "Caching is critical for performance",
                    "ID generation affects scalability"
                ]
            }
        ]
    }
}


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'current_view': 'home',
        'selected_domain': None,
        'current_question': None,
        'current_interviewer': INTERVIEWERS[0],
        'user_answer': '',
        'feedback': None,
        'hints_used': 0,
        'show_sample': False,
        'interview_history': [],
        'start_time': None,
        'elapsed_time': 0,
        # API Keys
        'openai_api_key': '',
        'did_api_key': '',
        'huggingface_api_key': '',
        # Settings
        'ai_provider': 'rule_based',
        'avatar_mode': 'static',  # 'static', 'did_video'
        'voice_enabled': True,
        'api_key_validated': {'openai': False, 'did': False},
        # Video state
        'current_video_url': None,
        'video_generating': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============== D-ID API Functions ==============

def validate_did_key(api_key: str) -> bool:
    """Validate D-ID API key."""
    if not api_key:
        return False
    try:
        headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(
            "https://api.d-id.com/credits",
            headers=headers,
            timeout=10
        )
        return response.status_code == 200
    except Exception:
        return False


def get_did_credits(api_key: str) -> Optional[dict]:
    """Get remaining D-ID credits."""
    try:
        headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(
            "https://api.d-id.com/credits",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def create_did_talk(
    text: str,
    source_url: str,
    api_key: str,
    voice_config: dict
) -> Optional[str]:
    """
    Create a talking avatar video using D-ID API.
    Returns the talk_id for polling.
    """
    try:
        headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "source_url": source_url,
            "script": {
                "type": "text",
                "input": text,
                "provider": {
                    "type": voice_config.get("provider", "microsoft"),
                    "voice_id": voice_config.get("voice_id", "en-US-JennyNeural")
                }
            },
            "config": {
                "fluent": True,
                "pad_audio": 0.5
            }
        }
        
        response = requests.post(
            "https://api.d-id.com/talks",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            result = response.json()
            return result.get("id")
        else:
            st.error(f"D-ID API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"D-ID request failed: {e}")
        return None


def get_did_talk_status(talk_id: str, api_key: str) -> Optional[dict]:
    """Poll D-ID for talk video status."""
    try:
        headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"https://api.d-id.com/talks/{talk_id}",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        return None
        
    except Exception:
        return None


def generate_talking_avatar(
    text: str,
    interviewer: dict,
    api_key: str,
    progress_placeholder
) -> Optional[str]:
    """
    Generate a talking avatar video and return the video URL.
    Shows progress in the provided placeholder.
    """
    # Create the talk request
    progress_placeholder.markdown("""
    <div class="video-loading">
        <div class="loading-spinner"></div>
        <p style="color: #9CA3AF; margin-top: 1rem;">Generating avatar video...</p>
    </div>
    """, unsafe_allow_html=True)
    
    talk_id = create_did_talk(
        text=text,
        source_url=interviewer["image_url"],
        api_key=api_key,
        voice_config=interviewer["did_voice"]
    )
    
    if not talk_id:
        return None
    
    # Poll for completion (max 60 seconds)
    max_attempts = 30
    for attempt in range(max_attempts):
        time.sleep(2)
        
        status = get_did_talk_status(talk_id, api_key)
        if not status:
            continue
            
        state = status.get("status")
        
        if state == "done":
            return status.get("result_url")
        elif state == "error":
            error_msg = status.get("error", {}).get("description", "Unknown error")
            st.error(f"Video generation failed: {error_msg}")
            return None
        
        # Update progress
        progress_placeholder.markdown(f"""
        <div class="video-loading">
            <div class="loading-spinner"></div>
            <p style="color: #9CA3AF; margin-top: 1rem;">Generating avatar video... ({attempt + 1}/{max_attempts})</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.warning("Video generation timed out. Using fallback.")
    return None


# ============== OpenAI Functions ==============

def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key."""
    if not api_key or not OPENAI_AVAILABLE:
        return False
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False


def generate_speech_openai(text: str, voice: str, api_key: str) -> Optional[bytes]:
    """Generate speech using OpenAI TTS."""
    try:
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        return response.content
    except Exception as e:
        st.warning(f"OpenAI TTS error: {e}")
        return None


def evaluate_with_openai(question: dict, answer: str, interviewer: dict, api_key: str) -> Optional[dict]:
    """Use OpenAI to evaluate the answer."""
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are {interviewer['name']}, a {interviewer['title']} conducting a technical interview.
Your style: {interviewer['personality']}

Evaluate this interview answer:

QUESTION: {question['question']}

CANDIDATE'S ANSWER: {answer}

KEY POINTS THAT SHOULD BE COVERED:
{chr(10).join(f'- {point}' for point in question.get('key_points', []))}

Provide evaluation in JSON format:
{{"score": <0-100>, "strengths": [...], "improvements": [...], "coaching_tips": [...], "interviewer_comment": "..."}}

Be constructive and specific."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"OpenAI error: {e}")
        return None


# ============== Evaluation Functions ==============

def evaluate_answer_rules(question: dict, answer: str, interviewer: dict) -> dict:
    """Rule-based evaluation fallback."""
    score = 0
    strengths = []
    improvements = []
    coaching_tips = []
    
    answer_lower = answer.lower()
    answer_length = len(answer.strip())
    
    if answer_length > 300:
        score += 25
        strengths.append("Comprehensive response with good detail")
    elif answer_length > 150:
        score += 15
        improvements.append("Consider expanding with more details")
    else:
        score += 5
        improvements.append("Answer could benefit from more depth")
    
    has_code = any(kw in answer for kw in ['SELECT', 'def ', '```', 'FROM', 'WHERE', '='])
    if has_code:
        score += 25
        strengths.append("Included practical code examples")
    elif 'SELECT' in question.get('sample_answer', '') or 'def ' in question.get('sample_answer', ''):
        improvements.append("Adding code examples would strengthen your answer")
    
    key_points = question.get('key_points', [])
    key_points_hit = sum(1 for point in key_points
                        if any(word.lower() in answer_lower for word in point.split()[:3]))
    
    if key_points_hit >= len(key_points) * 0.5:
        score += 25
        strengths.append("Addressed key technical concepts")
    elif key_points_hit > 0:
        score += 15
        improvements.append("Cover more core concepts")
    else:
        improvements.append("Review the fundamental concepts")
    
    if '\n' in answer or '-' in answer or '1.' in answer:
        score += 15
        strengths.append("Well-structured response")
    
    style_tips = {
        "technical": "Technical interviewers appreciate precise terminology and edge cases",
        "behavioral": "Explain your thought process as you work through problems",
        "strategic": "Consider discussing scalability and trade-offs"
    }
    coaching_tips.append(style_tips.get(interviewer['style'], ""))
    
    score = min(score, 100)
    tier = 'high' if score >= 75 else 'medium' if score >= 50 else 'low'
    
    return {
        "score": score,
        "strengths": strengths,
        "improvements": improvements,
        "coaching_tips": [t for t in coaching_tips if t],
        "interviewer_comment": interviewer['feedback_templates'][tier]
    }


def evaluate_answer(question: dict, answer: str, interviewer: dict) -> dict:
    """Main evaluation function."""
    result = None
    
    if st.session_state.ai_provider == 'openai' and st.session_state.openai_api_key:
        result = evaluate_with_openai(question, answer, interviewer, st.session_state.openai_api_key)
    
    if result is None:
        result = evaluate_answer_rules(question, answer, interviewer)
    
    return result


# ============== Audio/Video Rendering ==============

def render_avatar_with_video(interviewer: dict, video_url: Optional[str] = None, is_speaking: bool = False):
    """Render avatar - either video or static image."""
    
    if video_url:
        # Show D-ID video
        st.markdown(f"""
        <div class="avatar-video-container">
            <video autoplay playsinline>
                <source src="{video_url}" type="video/mp4">
            </video>
        </div>
        <div style="text-align: center; margin-top: 1rem;">
            <div style="color: #FFFFFF; font-weight: 600; font-size: 1.1rem;">{interviewer['name']}</div>
            <div style="color: #6366F1; font-size: 0.8rem;">{interviewer['title']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show static image with animation
        speaking_class = "avatar-speaking" if is_speaking else ""
        speaking_indicator = '''
        <div class="speaking-indicator">
            <div class="speaking-dot"></div>
            <div class="speaking-dot"></div>
            <div class="speaking-dot"></div>
        </div>
        ''' if is_speaking else ''
        
        st.markdown(f"""
        <div class="avatar-image-container {speaking_class}">
            <img src="{interviewer['image_url']}" alt="{interviewer['name']}"
                 onerror="this.style.display='none'; this.parentElement.innerHTML='<div style=\\'display:flex;align-items:center;justify-content:center;height:100%;font-size:3rem;\\'>{interviewer['avatar']}</div>';">
        </div>
        {speaking_indicator}
        <div style="text-align: center; margin-top: 0.75rem;">
            <div style="color: #FFFFFF; font-weight: 600;">{interviewer['name']}</div>
            <div style="color: #6366F1; font-size: 0.75rem;">{interviewer['title']}</div>
        </div>
        """, unsafe_allow_html=True)


def play_audio_openai(text: str, voice: str, api_key: str):
    """Play audio using OpenAI TTS."""
    audio_data = generate_speech_openai(text, voice, api_key)
    if audio_data:
        audio_b64 = base64.b64encode(audio_data).decode()
        st.markdown(f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)
        return True
    return False


def play_audio_browser(text: str):
    """Play audio using browser TTS."""
    escaped_text = text.replace("'", "\\'").replace("\n", " ").replace('"', '\\"')
    st.markdown(f"""
    <script>
    (function() {{
        const utterance = new SpeechSynthesisUtterance("{escaped_text}");
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }})();
    </script>
    """, unsafe_allow_html=True)


# ============== Sidebar ==============

def render_sidebar():
    """Render sidebar with all configuration options."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # Avatar Mode Selection
        st.markdown("### 🎭 Avatar Mode")
        avatar_mode = st.radio(
            "Select avatar type",
            options=['static', 'did_video'],
            format_func=lambda x: {
                'static': '🖼️ Static Image + Voice',
                'did_video': '🎬 D-ID Talking Avatar'
            }[x],
            index=['static', 'did_video'].index(st.session_state.avatar_mode),
            help="D-ID creates realistic talking videos"
        )
        st.session_state.avatar_mode = avatar_mode
        
        # Voice toggle (for static mode)
        if avatar_mode == 'static':
            st.session_state.voice_enabled = st.toggle(
                "🔊 Enable Voice",
                value=st.session_state.voice_enabled
            )
        
        st.markdown("---")
        
        # D-ID Configuration
        if avatar_mode == 'did_video':
            st.markdown("### 🎬 D-ID API Key")
            did_key = st.text_input(
                "D-ID API Key",
                type="password",
                value=st.session_state.did_api_key,
                placeholder="Enter your D-ID API key",
                help="Get your key at studio.d-id.com"
            )
            
            if did_key != st.session_state.did_api_key:
                st.session_state.did_api_key = did_key
                st.session_state.api_key_validated['did'] = False
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Validate D-ID", use_container_width=True):
                    with st.spinner("..."):
                        if validate_did_key(did_key):
                            st.session_state.api_key_validated['did'] = True
                            st.success("✓ Valid")
                        else:
                            st.error("✗ Invalid")
            
            with col2:
                if st.session_state.api_key_validated.get('did'):
                    st.markdown('<span class="status-badge status-connected">● Connected</span>', unsafe_allow_html=True)
            
            # Show credits
            if st.session_state.api_key_validated.get('did'):
                credits = get_did_credits(did_key)
                if credits:
                    remaining = credits.get('remaining', 0)
                    st.caption(f"💳 Credits remaining: {remaining}")
            
            st.markdown("""
            <div style="font-size: 0.7rem; color: #6B7280; margin-top: 0.5rem;">
            📹 D-ID creates lip-synced talking avatar videos<br>
            🆓 Free tier: 5 minutes of video<br>
            💰 Paid plans start at $5.90/month
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        # AI Provider for Evaluation
        st.markdown("### 🤖 Evaluation Provider")
        provider = st.radio(
            "Select provider",
            options=['rule_based', 'openai'],
            format_func=lambda x: {
                'rule_based': '📊 Rule-Based (Free)',
                'openai': '🤖 OpenAI GPT-4o-mini'
            }[x],
            index=['rule_based', 'openai'].index(st.session_state.ai_provider)
        )
        st.session_state.ai_provider = provider
        
        # OpenAI Configuration
        if provider == 'openai' or st.session_state.avatar_mode == 'static':
            st.markdown("---")
            st.markdown("### 🔑 OpenAI API Key")
            st.caption("Used for evaluation and/or voice synthesis")
            
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.openai_api_key,
                placeholder="sk-..."
            )
            
            if openai_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = openai_key
                st.session_state.api_key_validated['openai'] = False
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Validate OpenAI", use_container_width=True):
                    with st.spinner("..."):
                        if validate_openai_key(openai_key):
                            st.session_state.api_key_validated['openai'] = True
                            st.success("✓ Valid")
                        else:
                            st.error("✗ Invalid")
            
            with col2:
                if st.session_state.api_key_validated.get('openai'):
                    st.markdown('<span class="status-badge status-connected">● Connected</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Session Stats
        st.markdown("### 📊 Session Stats")
        if st.session_state.interview_history:
            avg = sum(h['score'] for h in st.session_state.interview_history) / len(st.session_state.interview_history)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Score", f"{avg:.0f}%")
            with col2:
                st.metric("Questions", len(st.session_state.interview_history))
        else:
            st.caption("No interviews completed yet")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.7rem; color: #4B5563;">
        TechPrep AI v4.0<br>
        Built for Data2Trend
        </div>
        """, unsafe_allow_html=True)


# ============== Main Views ==============

def start_interview(domain: str):
    """Start an interview session."""
    import random
    st.session_state.selected_domain = domain
    questions = QUESTION_BANK[domain]['questions']
    st.session_state.current_question = random.choice(questions)
    st.session_state.current_view = 'interview'
    st.session_state.start_time = time.time()
    st.session_state.user_answer = ''
    st.session_state.feedback = None
    st.session_state.hints_used = 0
    st.session_state.current_video_url = None


def submit_answer():
    """Submit and evaluate the answer."""
    if st.session_state.user_answer.strip():
        st.session_state.elapsed_time = int(time.time() - st.session_state.start_time)
        
        with st.spinner("Evaluating your answer..."):
            feedback = evaluate_answer(
                st.session_state.current_question,
                st.session_state.user_answer,
                st.session_state.current_interviewer
            )
        
        if st.session_state.hints_used > 0:
            feedback['score'] = max(0, feedback['score'] - (st.session_state.hints_used * 5))
            feedback['improvements'].append(f"Score adjusted for {st.session_state.hints_used} hint(s)")
        
        score = feedback['score']
        feedback['grade'] = 'Excellent' if score >= 85 else 'Good' if score >= 70 else 'Satisfactory' if score >= 50 else 'Needs Improvement'
        
        st.session_state.feedback = feedback
        st.session_state.current_video_url = None  # Reset for feedback video
        
        st.session_state.interview_history.append({
            'domain': st.session_state.selected_domain,
            'score': feedback['score'],
            'time': st.session_state.elapsed_time,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M')
        })


def render_home():
    """Render the home screen."""
    st.markdown("""
    <div class="main-header">
        <h1>⚡ TechPrep AI</h1>
        <p>Master Your Technical Interview</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicators
    avatar_label = "🎬 D-ID Video" if st.session_state.avatar_mode == 'did_video' else "🖼️ Static + Voice"
    eval_label = "🤖 OpenAI" if st.session_state.ai_provider == 'openai' else "📊 Rule-Based"
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem; display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
        <span class="status-badge" style="background: #1F1F2E; color: #9CA3AF; border: 1px solid #374151;">
            Avatar: {avatar_label}
        </span>
        <span class="status-badge" style="background: #1F1F2E; color: #9CA3AF; border: 1px solid #374151;">
            Evaluation: {eval_label}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Interviewer Selection
    st.markdown("### Choose Your Interviewer")
    
    cols = st.columns(3)
    for idx, interviewer in enumerate(INTERVIEWERS):
        with cols[idx]:
            is_selected = st.session_state.current_interviewer['id'] == interviewer['id']
            card_class = "interviewer-card selected" if is_selected else "interviewer-card"
            
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            render_avatar_with_video(interviewer, is_speaking=False)
            st.markdown(f"""
            <div style="color: #6B7280; font-size: 0.7rem; margin-top: 1rem; padding: 0 0.5rem;">
                {interviewer['personality']}
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(
                "✓ Selected" if is_selected else "Select",
                key=f"iv_{interviewer['id']}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.current_interviewer = interviewer
                st.rerun()
    
    st.markdown("---")
    
    # Domain Selection
    st.markdown("### Select Interview Domain")
    
    domain_cols = st.columns(4)
    for idx, (domain_key, domain_data) in enumerate(QUESTION_BANK.items()):
        with domain_cols[idx]:
            st.markdown(f"""
            <div style="background-color: #111118; border: 1px solid #1F1F2E; 
                        border-radius: 12px; padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem;">{domain_data['icon']}</div>
                <div style="color: #FFFFFF; font-weight: 600; margin-top: 0.5rem;">{domain_data['name']}</div>
                <div style="color: #6B7280; font-size: 0.75rem;">{len(domain_data['questions'])} questions</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Start Practice", key=f"d_{domain_key}", use_container_width=True):
                start_interview(domain_key)
                st.rerun()
    
    # History
    if st.session_state.interview_history:
        st.markdown("---")
        st.markdown("### Recent Sessions")
        for session in st.session_state.interview_history[-5:][::-1]:
            domain_info = QUESTION_BANK[session['domain']]
            col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
            with col1:
                st.markdown(f"{domain_info['icon']} **{domain_info['name']}**")
            with col2:
                color = "#10B981" if session['score'] >= 70 else "#F59E0B" if session['score'] >= 50 else "#EF4444"
                st.markdown(f"<span style='color: {color};'>{session['score']}%</span>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"⏱️ {session['time'] // 60}:{session['time'] % 60:02d}")
            with col4:
                st.caption(session['date'])


def render_interview():
    """Render the interview screen."""
    domain_data = QUESTION_BANK[st.session_state.selected_domain]
    question = st.session_state.current_question
    interviewer = st.session_state.current_interviewer
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Home"):
            st.session_state.current_view = 'home'
            st.session_state.current_video_url = None
            st.rerun()
    
    with col2:
        if st.session_state.start_time and not st.session_state.feedback:
            elapsed = int(time.time() - st.session_state.start_time)
            st.markdown(f"<h2 style='text-align: center; color: #FFFFFF;'>⏱️ {elapsed // 60}:{elapsed % 60:02d}</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content
    col_avatar, col_content = st.columns([1, 2])
    
    with col_avatar:
        # Avatar section
        video_placeholder = st.empty()
        
        # Generate D-ID video for question if enabled and not yet generated
        if (st.session_state.avatar_mode == 'did_video' and 
            st.session_state.did_api_key and 
            st.session_state.api_key_validated.get('did') and
            not st.session_state.feedback and
            not st.session_state.current_video_url):
            
            with video_placeholder:
                video_url = generate_talking_avatar(
                    text=question['question'],
                    interviewer=interviewer,
                    api_key=st.session_state.did_api_key,
                    progress_placeholder=video_placeholder
                )
                
                if video_url:
                    st.session_state.current_video_url = video_url
        
        # Render avatar/video
        with video_placeholder:
            render_avatar_with_video(
                interviewer,
                video_url=st.session_state.current_video_url,
                is_speaking=(not st.session_state.feedback and st.session_state.voice_enabled)
            )
        
        # Audio controls for static mode
        if st.session_state.avatar_mode == 'static' and st.session_state.voice_enabled:
            if st.button("🔊 Play Question", use_container_width=True, key="play_q"):
                if st.session_state.openai_api_key:
                    play_audio_openai(question['question'], interviewer['openai_voice'], st.session_state.openai_api_key)
                else:
                    play_audio_browser(question['question'])
        
        # Regenerate video button
        if st.session_state.avatar_mode == 'did_video' and st.session_state.current_video_url:
            if st.button("🔄 Regenerate Video", use_container_width=True, key="regen"):
                st.session_state.current_video_url = None
                st.rerun()
    
    with col_content:
        # Question badges
        diff_colors = {"Easy": "#10B981", "Medium": "#F59E0B", "Hard": "#EF4444"}
        diff_color = diff_colors.get(question['difficulty'], "#6B7280")
        
        st.markdown(f"""
        <div style="display: flex; gap: 0.75rem; margin-bottom: 1rem;">
            <span style="background-color: #1F1F2E; color: #9CA3AF; padding: 0.25rem 0.75rem; 
                         border-radius: 999px; font-size: 0.75rem;">
                {domain_data['icon']} {domain_data['name']}
            </span>
            <span style="background-color: {diff_color}22; color: {diff_color}; padding: 0.25rem 0.75rem; 
                         border-radius: 999px; font-size: 0.75rem; font-weight: 600;">
                {question['difficulty']}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Question
        st.markdown(f"""
        <div class="question-box">
            <p style="color: #E5E5E5; line-height: 1.8; margin: 0; font-size: 1rem;">
                {question['question']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Hints
        if st.session_state.hints_used > 0:
            st.markdown(f"""
            <div class="hint-card">
                <h4 style="color: #A5B4FC; margin: 0 0 0.75rem 0; font-size: 0.9rem;">
                    💡 Hints ({st.session_state.hints_used}/{len(question['hints'])})
                </h4>
                <ul style="color: #C7D2FE; margin: 0; padding-left: 1.25rem; font-size: 0.85rem;">
                    {''.join(f'<li style="margin-bottom: 0.5rem;">{h}</li>' for h in question['hints'][:st.session_state.hints_used])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
        
        # Answer or Feedback
        if not st.session_state.feedback:
            st.session_state.user_answer = st.text_area(
                "Your Answer",
                value=st.session_state.user_answer,
                height=200,
                placeholder="Type your answer here... Be thorough and include code examples where relevant.",
                label_visibility="collapsed"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                remaining = len(question['hints']) - st.session_state.hints_used
                if st.button(f"💡 Use Hint ({remaining} left)", disabled=remaining == 0, use_container_width=True):
                    st.session_state.hints_used += 1
                    st.rerun()
            
            with col2:
                if st.button("Submit Answer", type="primary", disabled=not st.session_state.user_answer.strip(), use_container_width=True):
                    submit_answer()
                    st.rerun()
        
        else:
            # === FEEDBACK SECTION ===
            feedback = st.session_state.feedback
            
            # Score display
            grade_colors = {"Excellent": "#10B981", "Good": "#3B82F6", "Satisfactory": "#F59E0B", "Needs Improvement": "#EF4444"}
            grade_color = grade_colors.get(feedback['grade'], "#6B7280")
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 2rem; margin-bottom: 1.5rem;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; font-weight: 700; color: #FFFFFF;">{feedback['score']}</div>
                    <div style="color: #6B7280;">/100</div>
                </div>
                <div>
                    <span style="background-color: {grade_color}22; color: {grade_color}; 
                               padding: 0.5rem 1rem; border-radius: 999px; font-weight: 600;">
                        {feedback['grade']}
                    </span>
                    <div style="color: #6B7280; margin-top: 0.5rem; font-size: 0.85rem;">
                        ⏱️ Completed in {st.session_state.elapsed_time // 60}:{st.session_state.elapsed_time % 60:02d}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interviewer comment
            st.markdown(f"""
            <div style="background-color: #1A1A24; padding: 1rem; border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #6366F1;">
                <p style="color: #E5E5E5; font-style: italic; margin: 0;">"{feedback['interviewer_comment']}"</p>
                <p style="color: #6B7280; font-size: 0.75rem; margin: 0.5rem 0 0 0;">— {interviewer['name']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Play feedback audio/video
            if st.session_state.avatar_mode == 'did_video' and st.session_state.did_api_key:
                if st.button("🎬 Generate Feedback Video", use_container_width=True, key="fb_video"):
                    with col_avatar:
                        video_url = generate_talking_avatar(
                            text=feedback['interviewer_comment'],
                            interviewer=interviewer,
                            api_key=st.session_state.did_api_key,
                            progress_placeholder=video_placeholder
                        )
                        if video_url:
                            st.session_state.current_video_url = video_url
                            st.rerun()
            elif st.session_state.voice_enabled:
                if st.button("🔊 Play Feedback", use_container_width=True, key="play_fb"):
                    if st.session_state.openai_api_key:
                        play_audio_openai(feedback['interviewer_comment'], interviewer['openai_voice'], st.session_state.openai_api_key)
                    else:
                        play_audio_browser(feedback['interviewer_comment'])
            
            # Detailed feedback
            col1, col2 = st.columns(2)
            with col1:
                if feedback.get('strengths'):
                    st.markdown("**✅ Strengths**")
                    for s in feedback['strengths']:
                        st.markdown(f"- {s}")
            
            with col2:
                if feedback.get('improvements'):
                    st.markdown("**📈 Areas to Improve**")
                    for i in feedback['improvements']:
                        st.markdown(f"- {i}")
            
            if feedback.get('coaching_tips'):
                st.markdown("**🎯 Coaching Tips**")
                for t in feedback['coaching_tips']:
                    st.markdown(f"- {t}")
            
            # Sample answer
            if st.checkbox("📖 Show Sample Answer"):
                st.code(question['sample_answer'])
                st.markdown("**Key Points:**")
                for kp in question.get('key_points', []):
                    st.markdown(f"• {kp}")
            
            # Actions
            st.markdown("")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Try Again", use_container_width=True):
                    st.session_state.user_answer = ''
                    st.session_state.feedback = None
                    st.session_state.hints_used = 0
                    st.session_state.start_time = time.time()
                    st.session_state.current_video_url = None
                    st.rerun()
            
            with col2:
                if st.button("Next Question →", type="primary", use_container_width=True):
                    import random
                    questions = QUESTION_BANK[st.session_state.selected_domain]['questions']
                    available = [q for q in questions if q['id'] != question['id']]
                    if available:
                        st.session_state.current_question = random.choice(available)
                        st.session_state.user_answer = ''
                        st.session_state.feedback = None
                        st.session_state.hints_used = 0
                        st.session_state.start_time = time.time()
                        st.session_state.current_video_url = None
                        st.rerun()
                    else:
                        st.success("🎉 You've completed all questions in this domain!")


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    
    if st.session_state.current_view == 'home':
        render_home()
    else:
        render_interview()


if __name__ == "__main__":
    main()
