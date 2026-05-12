"""
Claude Certified Architect Study Guide
LLM-powered Streamlit app for mastering Claude architecture concepts.
"""

import streamlit as st
import anthropic
import json
import random
import time
from datetime import datetime

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claude Certified Architect Study Guide",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #6B46C1 0%, #4C1D95 50%, #1E1B4B 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .hero p  { font-size: 1.1rem; margin-top: .5rem; opacity: .85; }

    .domain-card {
        background: white;
        border: 1.5px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        border-left: 5px solid #7C3AED;
        transition: box-shadow .2s;
    }
    .domain-card:hover { box-shadow: 0 4px 16px rgba(109,40,217,.15); }
    .domain-card h3 { margin: 0 0 .3rem 0; color: #4C1D95; font-size: 1rem; }
    .domain-card p  { margin: 0; color: #6B7280; font-size: .88rem; }

    .concept-badge {
        display: inline-block;
        background: #EDE9FE;
        color: #5B21B6;
        border-radius: 20px;
        padding: .25rem .75rem;
        font-size: .8rem;
        font-weight: 600;
        margin: .2rem .2rem .2rem 0;
    }

    .quiz-question {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #7C3AED;
        margin-bottom: 1rem;
    }
    .quiz-question h4 { color: #1F2937; margin-top: 0; }

    .correct   { background: #D1FAE5; border-left-color: #059669 !important; }
    .incorrect { background: #FEE2E2; border-left-color: #DC2626 !important; }

    .stat-box {
        background: linear-gradient(135deg, #7C3AED, #4C1D95);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-box .num  { font-size: 2rem; font-weight: 700; }
    .stat-box .lbl  { font-size: .85rem; opacity: .85; }

    .chat-user      { background: #EDE9FE; border-radius: 12px; padding: .9rem 1.1rem; margin: .5rem 0; }
    .chat-assistant { background: #F3F4F6; border-radius: 12px; padding: .9rem 1.1rem; margin: .5rem 0; }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4C1D95;
        border-bottom: 2px solid #EDE9FE;
        padding-bottom: .4rem;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stSidebar"] { background: #1E1B4B; }
    div[data-testid="stSidebar"] * { color: #E5E7EB !important; }
    div[data-testid="stSidebar"] .stSelectbox label { color: #C4B5FD !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"

DOMAINS = {
    "1. Claude Models & Capabilities": {
        "weight": "22%",
        "icon": "🧠",
        "topics": [
            "Model family overview (Haiku, Sonnet, Opus)",
            "Context window sizes and token limits",
            "Multimodal capabilities (vision, documents)",
            "Streaming responses and latency trade-offs",
            "Model selection criteria for use-cases",
            "claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-7 IDs",
        ],
        "key_concepts": ["Context window", "Token budget", "Model tiers", "Multimodal"],
    },
    "2. Prompt Engineering": {
        "weight": "20%",
        "icon": "✍️",
        "topics": [
            "System prompt design patterns",
            "Few-shot and zero-shot prompting",
            "Chain-of-thought (CoT) reasoning",
            "XML tagging for structured output",
            "Role assignment and persona framing",
            "Output format control (JSON, Markdown)",
        ],
        "key_concepts": ["System prompt", "Few-shot", "CoT", "XML tags", "Temperature"],
    },
    "3. Tool Use & Function Calling": {
        "weight": "18%",
        "icon": "🔧",
        "topics": [
            "Tool definition schema (name, description, input_schema)",
            "Tool call / tool result message flow",
            "Parallel tool calls",
            "Forced tool use with tool_choice",
            "Error handling in tool results",
            "Computer use (beta) capabilities",
        ],
        "key_concepts": ["Tool schema", "tool_choice", "Parallel calls", "tool_result"],
    },
    "4. Multi-Agent Systems": {
        "weight": "18%",
        "icon": "🤝",
        "topics": [
            "Orchestrator / subagent patterns",
            "Agent memory: in-context, external, semantic",
            "Agent SDK: Claude as orchestrator",
            "Inter-agent communication protocols",
            "Guardrails and agent safety",
            "Long-running tasks and checkpointing",
        ],
        "key_concepts": ["Orchestrator", "Subagent", "Memory types", "Agent loop"],
    },
    "5. Prompt Caching & Cost Optimization": {
        "weight": "12%",
        "icon": "💰",
        "topics": [
            "cache_control breakpoints (ephemeral)",
            "Minimum cacheable token thresholds",
            "Cache hit rate optimization strategies",
            "Input/output token pricing",
            "Batch API for async workloads",
            "Token counting with count_tokens API",
        ],
        "key_concepts": ["cache_control", "Ephemeral cache", "Batch API", "Token counting"],
    },
    "6. Safety, Trust & Responsible AI": {
        "weight": "10%",
        "icon": "🛡️",
        "topics": [
            "Constitutional AI principles",
            "Harmlessness, helpfulness, honesty",
            "Content filtering and refusal patterns",
            "Prompt injection mitigation",
            "Human-in-the-loop design",
            "Sensitive domain handling",
        ],
        "key_concepts": ["Constitutional AI", "RLHF", "Prompt injection", "HHH"],
    },
}

PRACTICE_QUESTIONS = [
    {
        "q": "Which parameter forces Claude to always call a specific tool instead of choosing?",
        "options": ["A. tool_required", "B. tool_choice={\"type\":\"tool\",\"name\":\"...\"}", "C. force_tool=True", "D. tool_mode=\"forced\""],
        "answer": "B",
        "explanation": "Setting tool_choice to {\"type\": \"tool\", \"name\": \"<tool_name>\"} forces Claude to call that specific tool. You can also use {\"type\": \"any\"} to require some tool call without specifying which.",
        "domain": "3. Tool Use & Function Calling",
    },
    {
        "q": "What is the minimum number of tokens required for a message block to be eligible for prompt caching?",
        "options": ["A. 512 tokens", "B. 1,024 tokens", "C. 2,048 tokens", "D. 4,096 tokens"],
        "answer": "B",
        "explanation": "Prompt caching requires at least 1,024 tokens in the cached content block (or 2,048 for certain older models). Below this threshold, caching is not applied.",
        "domain": "5. Prompt Caching & Cost Optimization",
    },
    {
        "q": "In a multi-agent architecture, which role is responsible for delegating tasks to worker agents?",
        "options": ["A. Subagent", "B. Evaluator", "C. Orchestrator", "D. Router"],
        "answer": "C",
        "explanation": "The orchestrator decomposes complex tasks and delegates them to subagents. Subagents focus on specific subtasks and report results back to the orchestrator.",
        "domain": "4. Multi-Agent Systems",
    },
    {
        "q": "Which Claude model ID should you use for the most capable general-purpose tasks as of 2026?",
        "options": ["A. claude-3-opus-20240229", "B. claude-opus-4-7", "C. claude-sonnet-4-6", "D. claude-haiku-4-5-20251001"],
        "answer": "B",
        "explanation": "claude-opus-4-7 is Anthropic's most capable model in the Claude 4 family. claude-sonnet-4-6 balances capability and speed. claude-haiku-4-5 is the fastest/cheapest option.",
        "domain": "1. Claude Models & Capabilities",
    },
    {
        "q": "What XML structure should you use to pass multiple document chunks to Claude for analysis?",
        "options": [
            "A. <input><doc>...</doc></input>",
            "B. <documents><document index=\"1\"><content>...</content></document></documents>",
            "C. [DOCUMENT]...[/DOCUMENT]",
            "D. ```document ... ```",
        ],
        "answer": "B",
        "explanation": "The recommended pattern is to wrap documents in <documents> with indexed <document> tags containing <source> and <content> children. This clearly delineates document boundaries for Claude.",
        "domain": "2. Prompt Engineering",
    },
    {
        "q": "Which API endpoint allows you to submit many requests asynchronously for up to 50% cost savings?",
        "options": ["A. /v1/stream", "B. /v1/messages/count_tokens", "C. /v1/messages/batches", "D. /v1/complete/bulk"],
        "answer": "C",
        "explanation": "The Messages Batches API (/v1/messages/batches) processes requests asynchronously and offers up to 50% cost reduction. Results are available within 24 hours.",
        "domain": "5. Prompt Caching & Cost Optimization",
    },
    {
        "q": "Which principle from Constitutional AI describes Claude refusing to help with tasks that could harm others?",
        "options": ["A. Honesty", "B. Harmlessness", "C. Helpfulness", "D. Humility"],
        "answer": "B",
        "explanation": "Constitutional AI is built on HHH: Helpful, Harmless, and Honest. Harmlessness covers refusal of requests that could cause harm to users, third parties, or society.",
        "domain": "6. Safety, Trust & Responsible AI",
    },
    {
        "q": "When streaming a response using the Python SDK, which event signals the start of a content block?",
        "options": ["A. stream.text()", "B. content_block_start", "C. message_delta", "D. stream_begin"],
        "answer": "B",
        "explanation": "The streaming event lifecycle includes: message_start → content_block_start → content_block_delta (repeated) → content_block_stop → message_delta → message_stop.",
        "domain": "1. Claude Models & Capabilities",
    },
    {
        "q": "What does setting temperature=0 do to Claude's responses?",
        "options": [
            "A. Makes responses faster",
            "B. Disables tool use",
            "C. Makes responses maximally deterministic",
            "D. Enables extended thinking",
        ],
        "answer": "C",
        "explanation": "Temperature controls randomness in token sampling. temperature=0 selects the highest-probability token at each step, producing near-deterministic (reproducible) responses. Higher values increase creativity/diversity.",
        "domain": "2. Prompt Engineering",
    },
    {
        "q": "Which cache_control type is currently supported for prompt caching?",
        "options": ["A. persistent", "B. session", "C. ephemeral", "D. permanent"],
        "answer": "C",
        "explanation": "The only supported cache_control type is \"ephemeral\". Cached content is retained for approximately 5 minutes (with TTL refresh on each hit). Persistent/session caches are not currently available.",
        "domain": "5. Prompt Caching & Cost Optimization",
    },
]

SYSTEM_PROMPT = """You are an expert Claude Certified Architect tutor and study coach.
You have deep knowledge of:
- All Claude model families (Haiku, Sonnet, Opus) and their specifications
- The Anthropic API: messages, streaming, tool use, vision, batch processing
- Prompt engineering best practices and patterns
- Multi-agent architectures using Claude as orchestrator or subagent
- Prompt caching with cache_control and cost optimization strategies
- Constitutional AI, safety principles, and responsible deployment
- The Claude Agent SDK and MCP (Model Context Protocol)

Your role:
1. Answer questions clearly, accurately, and concisely
2. Use concrete code examples (Python SDK) when helpful
3. Highlight exam-relevant details with "📌 Exam tip:" callouts
4. Connect concepts to real architectural decisions
5. Correct misconceptions gently but clearly

Always ground your answers in the official Anthropic documentation and current (2026) model IDs:
- claude-haiku-4-5-20251001 (fast, cost-efficient)
- claude-sonnet-4-6 (balanced)
- claude-opus-4-7 (most capable)
"""

# ─── Helpers ───────────────────────────────────────────────────────────────────
def get_client() -> anthropic.Anthropic | None:
    api_key = st.session_state.get("api_key") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def stream_response(client: anthropic.Anthropic, messages: list, system: str = SYSTEM_PROMPT) -> str:
    full_text = ""
    placeholder = st.empty()
    with client.messages.stream(
        model=MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            full_text += text
            placeholder.markdown(full_text + "▌")
    placeholder.markdown(full_text)
    return full_text


def score_badge(score: int, total: int) -> str:
    pct = score / total * 100
    if pct >= 80:
        return f"🏆 {pct:.0f}% — Excellent! Ready for the exam."
    elif pct >= 60:
        return f"📚 {pct:.0f}% — Good progress. Review weak areas."
    else:
        return f"🔄 {pct:.0f}% — Keep studying. You've got this!"


# ─── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "chat_history": [],
        "quiz_index": 0,
        "quiz_score": 0,
        "quiz_answered": False,
        "quiz_questions": random.sample(PRACTICE_QUESTIONS, len(PRACTICE_QUESTIONS)),
        "flashcard_index": 0,
        "quiz_results": [],
        "api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ Claude Architect")
    st.markdown("---")

    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-ant-...",
        help="Required for AI Tutor and Explain features",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("---")
    nav = st.radio(
        "Navigate",
        ["🏠 Overview", "📚 Study Domains", "💬 AI Tutor", "🧩 Practice Quiz", "🃏 Flashcards", "📊 Progress"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Exam at a Glance**")
    st.markdown("- 65 questions · 130 min")
    st.markdown("- Pass score: ~72%")
    st.markdown("- Format: Multiple choice")
    st.markdown("- Validity: 2 years")
    st.markdown("---")
    st.markdown(f"*Model: `{MODEL}`*")

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏛️ Claude Certified Architect</h1>
    <p>AI-powered study guide · Master the architecture · Pass the exam</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Overview":
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("6", "Exam Domains"),
        (str(len(PRACTICE_QUESTIONS)), "Practice Questions"),
        ("65", "Exam Questions"),
        ("72%", "Pass Score"),
    ]
    for col, (num, lbl) in zip([col1, col2, col3, col4], stats):
        col.markdown(f"""
        <div class="stat-box">
            <div class="num">{num}</div>
            <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Exam Domain Weights</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (domain, info) in enumerate(DOMAINS.items()):
        with cols[i % 2]:
            badges = " ".join(f'<span class="concept-badge">{c}</span>' for c in info["key_concepts"])
            st.markdown(f"""
            <div class="domain-card">
                <h3>{info['icon']} {domain} <small style="color:#9CA3AF">({info['weight']})</small></h3>
                <p>{"  ·  ".join(info['topics'][:3])} …</p>
                <div style="margin-top:.5rem">{badges}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Quick Study Tips</div>', unsafe_allow_html=True)
    tips = [
        ("💡", "Memorize current model IDs", "claude-haiku-4-5-20251001, claude-sonnet-4-6, claude-opus-4-7"),
        ("🔧", "Understand tool_choice values", "\"auto\", \"any\", {\"type\":\"tool\",\"name\":\"...\"} — know when to use each"),
        ("💰", "Know caching rules cold", "Min 1,024 tokens, ephemeral type, ~5 min TTL, up to 90% input cost savings"),
        ("🤝", "Draw the agent loop", "Orchestrator → tool call → subagent → result → orchestrator loop"),
        ("✍️", "Practice XML prompting", "Use <documents>, <examples>, <thinking> tags to structure complex prompts"),
    ]
    for icon, title, detail in tips:
        st.markdown(f"**{icon} {title}** — {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STUDY DOMAINS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📚 Study Domains":
    st.markdown('<div class="section-header">Study Domains</div>', unsafe_allow_html=True)

    selected = st.selectbox("Choose a domain to study", list(DOMAINS.keys()))
    info = DOMAINS[selected]

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown(f"### {info['icon']} {selected}")
        st.markdown(f"**Exam weight:** `{info['weight']}`")
        st.markdown("**Topics covered:**")
        for t in info["topics"]:
            st.markdown(f"- {t}")

    with col_r:
        st.markdown("**Key concepts:**")
        badges = " ".join(f'<span class="concept-badge">{c}</span>' for c in info["key_concepts"])
        st.markdown(badges, unsafe_allow_html=True)

    st.markdown("---")

    client = get_client()
    if client:
        if st.button(f"🤖 Generate deep-dive explanation for {selected.split('.')[1].strip()}", use_container_width=True):
            with st.spinner("Generating study material…"):
                prompt = f"""Generate a comprehensive architect-level study guide for the exam domain: "{selected}".

Structure your response as:
1. **Core Concepts** — precise definitions with examples
2. **Architecture Patterns** — how to design systems using these features
3. **Python SDK Examples** — concrete code snippets
4. **Common Pitfalls** — mistakes candidates make
5. **📌 Exam Tips** — high-probability exam topics

Be specific, accurate, and use current 2026 model IDs and API formats."""
                stream_response(client, [{"role": "user", "content": prompt}])
    else:
        st.info("Enter your Anthropic API key in the sidebar to generate AI explanations.")
        st.markdown("**Sample topic summary:**")
        st.markdown(f"Domain **{selected}** covers: {', '.join(info['topics'])}.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI TUTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "💬 AI Tutor":
    st.markdown('<div class="section-header">AI Tutor — Ask Anything</div>', unsafe_allow_html=True)

    client = get_client()
    if not client:
        st.warning("Enter your Anthropic API key in the sidebar to use the AI Tutor.")
        st.stop()

    # Suggested questions
    st.markdown("**Suggested questions:**")
    suggestions = [
        "How does prompt caching work and when should I use it?",
        "Explain the difference between orchestrator and subagent patterns",
        "What are the tool_choice options and when do I use each?",
        "Walk me through designing a multi-agent RAG pipeline with Claude",
        "How do I handle tool errors gracefully in an agentic loop?",
        "What's the streaming event lifecycle for a Claude response?",
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, key=f"sugg_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": s})

    st.markdown("---")

    # Chat history
    for msg in st.session_state.chat_history:
        css_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{css_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Your question", placeholder="Ask about Claude architecture, APIs, prompt engineering…", height=80)
        submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

    # Auto-respond if last message is from user
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        with st.spinner("Thinking…"):
            # Keep last 10 turns for context
            messages = st.session_state.chat_history[-10:]
            response = stream_response(client, messages)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PRACTICE QUIZ
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🧩 Practice Quiz":
    st.markdown('<div class="section-header">Practice Quiz</div>', unsafe_allow_html=True)

    questions = st.session_state.quiz_questions
    idx = st.session_state.quiz_index

    # Progress bar
    progress = idx / len(questions)
    st.progress(progress, text=f"Question {idx + 1} of {len(questions)} · Score: {st.session_state.quiz_score}/{idx}")

    if idx >= len(questions):
        # Results screen
        pct = st.session_state.quiz_score / len(questions) * 100
        st.markdown(f"## Quiz Complete! {score_badge(st.session_state.quiz_score, len(questions))}")

        # Domain breakdown
        domain_results: dict[str, dict] = {}
        for r in st.session_state.quiz_results:
            d = r["domain"]
            if d not in domain_results:
                domain_results[d] = {"correct": 0, "total": 0}
            domain_results[d]["total"] += 1
            if r["correct"]:
                domain_results[d]["correct"] += 1

        st.markdown("### Domain Breakdown")
        for domain, counts in domain_results.items():
            pct_d = counts["correct"] / counts["total"] * 100
            st.markdown(f"**{domain}:** {counts['correct']}/{counts['total']} ({pct_d:.0f}%)")
            st.progress(pct_d / 100)

        # Review wrong answers
        wrong = [r for r in st.session_state.quiz_results if not r["correct"]]
        if wrong:
            with st.expander(f"📖 Review {len(wrong)} incorrect answers"):
                for r in wrong:
                    st.markdown(f"**Q:** {r['question']}")
                    st.markdown(f"- Your answer: {r['selected']} | Correct: {r['answer']}")
                    st.markdown(f"- *{r['explanation']}*")
                    st.markdown("---")

        if st.button("🔄 Restart Quiz", use_container_width=True):
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answered = False
            st.session_state.quiz_questions = random.sample(PRACTICE_QUESTIONS, len(PRACTICE_QUESTIONS))
            st.session_state.quiz_results = []
            st.rerun()

    else:
        q = questions[idx]
        css_class = ""
        if st.session_state.quiz_answered:
            last = st.session_state.quiz_results[-1] if st.session_state.quiz_results else None
            if last:
                css_class = "correct" if last["correct"] else "incorrect"

        st.markdown(f"""
        <div class="quiz-question {css_class}">
            <h4>Q{idx + 1}. {q['q']}</h4>
            <small style="color:#6B7280">Domain: {q['domain']}</small>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.quiz_answered:
            selected = st.radio("Select your answer:", q["options"], key=f"q_{idx}")
            if st.button("Submit Answer", use_container_width=True):
                chosen_letter = selected[0]
                correct = chosen_letter == q["answer"]
                if correct:
                    st.session_state.quiz_score += 1
                st.session_state.quiz_results.append({
                    "domain": q["domain"],
                    "question": q["q"],
                    "selected": selected,
                    "answer": q["answer"],
                    "correct": correct,
                    "explanation": q["explanation"],
                })
                st.session_state.quiz_answered = True
                st.rerun()
        else:
            last = st.session_state.quiz_results[-1]
            if last["correct"]:
                st.success(f"✅ Correct! **{q['answer']}** is right.")
            else:
                st.error(f"❌ Incorrect. The correct answer is **{q['answer']}**.")
            st.info(f"💡 **Explanation:** {q['explanation']}")

            client = get_client()
            if client:
                if st.button("🤖 Ask tutor to explain further", use_container_width=True):
                    with st.spinner("Getting detailed explanation…"):
                        prompt = f"""A student got this question {'correct' if last['correct'] else 'wrong'} on a practice quiz.

Question: {q['q']}
Options: {', '.join(q['options'])}
Correct answer: {q['answer']}
Explanation: {q['explanation']}

Please provide a thorough architect-level explanation of the underlying concept, including any related API details, code examples, and exam tips."""
                        stream_response(client, [{"role": "user", "content": prompt}])

            if st.button("Next Question →", use_container_width=True):
                st.session_state.quiz_index += 1
                st.session_state.quiz_answered = False
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FLASHCARDS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🃏 Flashcards":
    st.markdown('<div class="section-header">Flashcards</div>', unsafe_allow_html=True)

    FLASHCARDS = [
        {"front": "What is the cache_control type for prompt caching?", "back": "\"ephemeral\" — the only supported type. Content is cached for ~5 minutes, refreshed on each cache hit."},
        {"front": "Name the 3 Claude model tiers and their 2026 IDs", "back": "Haiku: claude-haiku-4-5-20251001\nSonnet: claude-sonnet-4-6\nOpus: claude-opus-4-7"},
        {"front": "What is the minimum token count for prompt caching?", "back": "1,024 tokens (for most models). Content blocks smaller than this threshold will not be cached."},
        {"front": "What does tool_choice {\"type\": \"any\"} do?", "back": "Forces Claude to call at least one tool from the available tools list, but lets Claude choose which tool to use."},
        {"front": "Describe the orchestrator/subagent pattern", "back": "Orchestrator: high-level Claude instance that decomposes tasks and delegates.\nSubagent: specialized Claude instance that executes a specific subtask and returns results."},
        {"front": "What streaming events occur in order for a Claude response?", "back": "1. message_start\n2. content_block_start\n3. content_block_delta (×N)\n4. content_block_stop\n5. message_delta\n6. message_stop"},
        {"front": "What are the 3 Hs of Constitutional AI?", "back": "Helpful — assists users effectively\nHarmless — avoids causing harm\nHonest — truthful and calibrated"},
        {"front": "How does the Batch API save costs?", "back": "The Messages Batches API (/v1/messages/batches) offers up to 50% cost reduction for async workloads processed within 24 hours."},
        {"front": "What is the role of the system prompt?", "back": "Sets persistent context, persona, constraints, and instructions for Claude before the conversation begins. Not visible to users in most UIs."},
        {"front": "What does temperature=0 produce?", "back": "Near-deterministic (maximally greedy) responses. Claude always picks the highest-probability next token. Best for factual/consistent outputs."},
        {"front": "Name 3 types of agent memory", "back": "1. In-context: messages within the current context window\n2. External: databases, vector stores, files\n3. Semantic: embeddings-based long-term memory"},
        {"front": "What is MCP (Model Context Protocol)?", "back": "An open standard for connecting Claude to external tools, data sources, and services. Enables structured tool definitions and bi-directional communication between Claude and host applications."},
        {"front": "How do you count tokens before sending a request?", "back": "Use the count_tokens API endpoint: client.messages.count_tokens(model=..., messages=...). Returns an input_tokens count without making a full inference call."},
        {"front": "What XML tags are recommended for few-shot examples?", "back": "<examples>\n  <example>\n    <input>...</input>\n    <output>...</output>\n  </example>\n</examples>"},
        {"front": "What is a tool result message?", "back": "A user-role message with type \"tool_result\" that returns the output of a tool call back to Claude. Contains the tool_use_id and content of the result."},
    ]

    fc_idx = st.session_state.flashcard_index % len(FLASHCARDS)
    card = FLASHCARDS[fc_idx]

    st.markdown(f"**Card {fc_idx + 1} of {len(FLASHCARDS)}**")
    st.progress((fc_idx + 1) / len(FLASHCARDS))

    with st.expander("📋 **FRONT** — Click to reveal answer", expanded=True):
        st.markdown(f"### {card['front']}")

    with st.expander("✅ **BACK** — Answer"):
        st.markdown(f"```\n{card['back']}\n```" if "\n" in card["back"] else f"**{card['back']}**")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("◀ Previous", use_container_width=True):
            st.session_state.flashcard_index = (fc_idx - 1) % len(FLASHCARDS)
            st.rerun()
    with col2:
        if st.button("🔀 Shuffle", use_container_width=True):
            st.session_state.flashcard_index = random.randint(0, len(FLASHCARDS) - 1)
            st.rerun()
    with col3:
        if st.button("Next ▶", use_container_width=True):
            st.session_state.flashcard_index = (fc_idx + 1) % len(FLASHCARDS)
            st.rerun()

    client = get_client()
    if client:
        if st.button("🤖 Generate more flashcards with AI", use_container_width=True):
            with st.spinner("Generating flashcards…"):
                prompt = """Generate 5 new flashcard-style question-answer pairs for the Claude Certified Architect exam.
Focus on: API details, architecture patterns, and tricky exam topics.
Format each as:
Q: [question]
A: [concise answer]

Make them different from these topics: caching, model IDs, tool_choice, agent patterns, streaming."""
                stream_response(client, [{"role": "user", "content": prompt}])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PROGRESS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📊 Progress":
    st.markdown('<div class="section-header">Your Progress Dashboard</div>', unsafe_allow_html=True)

    results = st.session_state.quiz_results
    total_answered = len(results)
    total_correct = sum(1 for r in results if r["correct"])

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""<div class="stat-box"><div class="num">{total_answered}</div><div class="lbl">Questions Answered</div></div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div class="stat-box"><div class="num">{total_correct}</div><div class="lbl">Correct Answers</div></div>""", unsafe_allow_html=True)
    pct = f"{total_correct/total_answered*100:.0f}%" if total_answered else "—"
    col3.markdown(f"""<div class="stat-box"><div class="num">{pct}</div><div class="lbl">Accuracy</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if results:
        # Domain accuracy breakdown
        st.markdown("### Domain Performance")
        domain_stats: dict[str, dict] = {}
        for r in results:
            d = r["domain"]
            if d not in domain_stats:
                domain_stats[d] = {"correct": 0, "total": 0}
            domain_stats[d]["total"] += 1
            if r["correct"]:
                domain_stats[d]["correct"] += 1

        for domain, counts in domain_stats.items():
            accuracy = counts["correct"] / counts["total"]
            color = "green" if accuracy >= 0.8 else "orange" if accuracy >= 0.6 else "red"
            st.markdown(f"**{domain}**")
            st.progress(accuracy, text=f"{counts['correct']}/{counts['total']} ({accuracy*100:.0f}%)")

        st.markdown("### Study Recommendations")
        weak = [(d, s["correct"]/s["total"]) for d, s in domain_stats.items() if s["correct"]/s["total"] < 0.7]
        if weak:
            st.warning("Focus study on these domains:")
            for domain, acc in sorted(weak, key=lambda x: x[1]):
                st.markdown(f"- **{domain}** ({acc*100:.0f}% accuracy) — go to Study Domains and generate a deep-dive")
        else:
            st.success("Great performance across all attempted domains! Keep practicing to reinforce retention.")

        # History table
        st.markdown("### Answer History")
        history_data = [
            {
                "Question": r["question"][:60] + "…",
                "Domain": r["domain"].split(".")[1].strip() if "." in r["domain"] else r["domain"],
                "Result": "✅ Correct" if r["correct"] else "❌ Wrong",
                "Your Answer": r["selected"][:30],
                "Correct": r["answer"],
            }
            for r in results
        ]
        st.dataframe(history_data, use_container_width=True)
    else:
        st.info("No quiz results yet. Head to **Practice Quiz** to get started!")

    # Readiness estimate
    st.markdown("### Exam Readiness Estimate")
    if total_answered >= 5:
        readiness = total_correct / total_answered * 100
        st.progress(readiness / 100, text=f"Estimated readiness: {readiness:.0f}% (pass threshold: 72%)")
        if readiness >= 80:
            st.success("You appear well prepared! Consider scheduling your exam.")
        elif readiness >= 72:
            st.info("You're near pass threshold. A few more study sessions should get you there.")
        else:
            st.warning("More study needed. Focus on weak domains and use the AI Tutor.")
    else:
        st.info("Answer at least 5 quiz questions to see your readiness estimate.")
