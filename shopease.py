"""
ShopEase Customer Support – Streamlit UI
Agentic RAG pipeline with ChromaDB + Anthropic tool use.
"""

from __future__ import annotations

import streamlit as st

from knowledge_base import build_knowledge_base, get_kb_stats
from agent import run_agent

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ShopEase Customer Support",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []          # chat history for display
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []  # raw history for the agent
    if "last_tool_calls" not in st.session_state:
        st.session_state.last_tool_calls = []
    if "escalation_count" not in st.session_state:
        st.session_state.escalation_count = 0
    if "total_messages" not in st.session_state:
        st.session_state.total_messages = 0

_init_session_state()

# ---------------------------------------------------------------------------
# Knowledge base – cached so it is only built once per Streamlit session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ShopEase knowledge base…")
def get_collection():
    return build_knowledge_base()

collection = get_collection()
kb_stats = get_kb_stats(collection)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("ShopEase Support")
    st.caption("Internal agent dashboard")
    st.divider()

    # ── Knowledge Base Stats ─────────────────────────────────────────────────
    st.subheader("📚 Knowledge Base Stats")
    st.metric("Total Documents", kb_stats["total_documents"])
    st.markdown("**Categories:**")
    for cat, count in kb_stats["categories"].items():
        icon = {
            "returns": "↩️",
            "shipping": "📦",
            "payment": "💳",
            "account": "👤",
            "warranty": "🛡️",
        }.get(cat, "📄")
        st.markdown(f"- {icon} **{cat.title()}**: {count} docs")

    st.divider()

    # ── Last Tool Calls ──────────────────────────────────────────────────────
    st.subheader("🔧 Last Tool Calls")
    if st.session_state.last_tool_calls:
        for call in st.session_state.last_tool_calls:
            tool_name = call["tool"]
            summary = call["result_summary"]
            icon = {
                "search_knowledge_base": "🔍",
                "check_order_status": "📋",
                "escalate_to_human": "🚨",
            }.get(tool_name, "🔧")
            with st.expander(f"{icon} `{tool_name}`", expanded=False):
                st.markdown(f"**Result:** {summary}")
                if call.get("input"):
                    st.json(call["input"])
    else:
        st.caption("No tools called yet.")

    st.divider()

    # ── Session Stats ────────────────────────────────────────────────────────
    st.subheader("📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Messages", st.session_state.total_messages)
    col2.metric("Escalations", st.session_state.escalation_count)

    st.divider()

    # ── Clear Chat ───────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.last_tool_calls = []
        st.session_state.escalation_count = 0
        st.session_state.total_messages = 0
        st.rerun()

# ---------------------------------------------------------------------------
# Main area – title
# ---------------------------------------------------------------------------
st.title("🛍️ ShopEase Customer Support")
st.caption(
    "Powered by an Agentic RAG pipeline · claude-sonnet-4-6 · ChromaDB"
)
st.divider()

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

        # Assistant messages may carry extra metadata
        if role == "assistant":
            # Escalation banner
            if msg.get("escalated"):
                ticket = msg.get("escalation_ticket", "")
                st.error(
                    f"⚠️ **Escalated to human agent** "
                    + (f"· Ticket: `{ticket}`" if ticket else ""),
                    icon="🚨",
                )

            # Sources expander
            sources = msg.get("sources_used", [])
            if sources:
                with st.expander(f"📄 Sources used ({len(sources)})", expanded=False):
                    for src in sources:
                        st.markdown(
                            f"**`{src['doc_id']}`** · *{src['category'].title()}* "
                            f"· relevance: `{src['relevance']:.3f}`"
                        )
                        st.caption(src["snippet"])
                        st.divider()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input("How can I help you today?")

if user_input:
    # ── Display user message ─────────────────────────────────────────────────
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.total_messages += 1

    # ── Run agent ────────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Sage is thinking…"):
            agent_response = run_agent(
                user_message=user_input,
                conversation_history=st.session_state.conversation_history,
                collection=collection,
            )

        answer = agent_response.answer
        st.markdown(answer)

        # Escalation banner (live render)
        if agent_response.escalated:
            ticket = agent_response.escalation_ticket or ""
            st.error(
                f"⚠️ **Escalated to human agent** "
                + (f"· Ticket: `{ticket}`" if ticket else ""),
                icon="🚨",
            )
            st.session_state.escalation_count += 1

        # Sources expander (live render)
        sources = agent_response.sources_used
        if sources:
            with st.expander(f"📄 Sources used ({len(sources)})", expanded=False):
                for src in sources:
                    st.markdown(
                        f"**`{src['doc_id']}`** · *{src['category'].title()}* "
                        f"· relevance: `{src['relevance']:.3f}`"
                    )
                    st.caption(src["snippet"])
                    st.divider()

    # ── Update session state ─────────────────────────────────────────────────

    # Persist assistant message with metadata for re-render
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "escalated": agent_response.escalated,
        "escalation_ticket": agent_response.escalation_ticket,
        "sources_used": sources,
    })
    st.session_state.total_messages += 1

    # Update conversation history for the agent (raw Anthropic format)
    st.session_state.conversation_history.append(
        {"role": "user", "content": user_input}
    )
    st.session_state.conversation_history.append(
        {"role": "assistant", "content": answer}
    )

    # Update sidebar tool calls
    st.session_state.last_tool_calls = agent_response.tool_calls_made

    # Rerun so sidebar stats refresh
    st.rerun()
