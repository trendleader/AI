"""
Cybersecurity Review Agent -- Streamlit UI

Upload source files (or a zipped codebase) and get:
  - An automated static analysis pass (hardcoded secrets, injection flaws,
    insecure crypto, unsafe deserialization, etc.)
  - An optional LLM-powered deep-dive that reasons about exploitation
    scenarios and produces a prioritized remediation plan.

Scope note: this tool performs *static source-code review*. It does not send
traffic to, or run tests against, any running system. Only upload code you
own or are authorized to assess.
"""

import streamlit as st

from vuln_scanner import scan_uploaded_file, scan_paths, Severity, render_markdown_report
from llm_security_review import run_llm_review

st.set_page_config(
    page_title="Cybersecurity Review Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { color: #d64545; font-size: 2.3rem; margin-bottom: 0.25rem; }
    .subheader { color: #666; margin-bottom: 1rem; }
    .scope-box {
        background-color: #fff4e5; border-left: 4px solid #d68a00;
        padding: 0.9rem 1.1rem; border-radius: 0.5rem; margin-bottom: 1.2rem;
    }
    .sev-Critical { color: #b30000; font-weight: 700; }
    .sev-High { color: #d64545; font-weight: 700; }
    .sev-Medium { color: #c98a00; font-weight: 700; }
    .sev-Low { color: #2b7a2b; font-weight: 700; }
    .sev-Info { color: #555; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🛡️ Cybersecurity Review Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subheader">Static vulnerability scanning + AI-assisted exploitation reasoning for uploaded code</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="scope-box">
    <strong>Scope:</strong> This agent performs <em>static source-code review</em> only -- it never sends
    requests to a live system. Only upload code you own or are explicitly authorized to assess.
    Findings are a starting point, not a substitute for a full manual/dynamic security assessment.
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar -----------------------------------------------------------
st.sidebar.header("⚙️ Options")

run_llm = st.sidebar.checkbox("Run AI deep-dive review (uses Claude)", value=False)
api_key = ""
model = "claude-sonnet-5"
if run_llm:
    api_key = st.sidebar.text_input(
        "Anthropic API key",
        type="password",
        help="Used only for this session; never stored or logged.",
    )
    model = st.sidebar.selectbox("Model", ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"], index=0)

min_severity = st.sidebar.selectbox(
    "Minimum severity to display",
    [s.value for s in Severity],
    index=4,
)

st.sidebar.markdown("---")
st.sidebar.caption("🔐 Files are processed in-memory for this session only.")
st.sidebar.caption("📚 Static rules cover OWASP-style issues: injection, secrets, crypto, deserialization, SSRF/TLS, XSS, path traversal, and more.")

# --- Upload --------------------------------------------------------------
st.subheader("📤 Upload code")
uploaded_files = st.file_uploader(
    "Upload one or more source files, or a .zip of a codebase",
    accept_multiple_files=True,
    type=None,
)

pasted_code = st.text_area(
    "...or paste a code snippet instead",
    height=180,
    placeholder="Paste code here to scan it directly (treated as a single Python file unless you note otherwise).",
)

run_clicked = st.button("🔍 Run security review", type="primary", use_container_width=True)

if run_clicked:
    if not uploaded_files and not pasted_code.strip():
        st.error("❌ Upload at least one file (or paste a snippet) first.")
    elif run_llm and not api_key:
        st.error("❌ Enter your Anthropic API key in the sidebar, or uncheck the AI deep-dive option.")
    else:
        all_findings_result = None
        code_samples = []
        source_labels = []

        with st.spinner("🔎 Running static analysis..."):
            for uploaded in uploaded_files or []:
                raw = uploaded.getvalue()
                result = scan_uploaded_file(uploaded.name, raw)
                source_labels.append(uploaded.name)
                if all_findings_result is None:
                    all_findings_result = result
                else:
                    all_findings_result.findings.extend(result.findings)
                    all_findings_result.files_scanned += result.files_scanned
                    all_findings_result.files_skipped += result.files_skipped
                if not uploaded.name.lower().endswith(".zip"):
                    try:
                        code_samples.append((uploaded.name, raw.decode("utf-8", errors="ignore")))
                    except Exception:
                        pass

            if pasted_code.strip():
                snippet_result = scan_paths([("pasted_snippet.py", pasted_code)])
                source_labels.append("pasted_snippet.py")
                if all_findings_result is None:
                    all_findings_result = snippet_result
                else:
                    all_findings_result.findings.extend(snippet_result.findings)
                    all_findings_result.files_scanned += snippet_result.files_scanned
                code_samples.append(("pasted_snippet.py", pasted_code))

        result = all_findings_result
        source_label = ", ".join(source_labels) if len(source_labels) <= 3 else f"{len(source_labels)} files"

        st.success(f"✅ Scanned {result.files_scanned} file(s), skipped {result.files_skipped}.")

        # --- Summary metrics ---
        counts = result.counts_by_severity()
        cols = st.columns(6)
        cols[0].metric("Risk score", f"{result.risk_score()}/100")
        for i, sev in enumerate(Severity):
            cols[i + 1].metric(sev.value, counts[sev])

        st.markdown("---")

        # --- Findings table ---
        min_rank_order = [s.value for s in Severity].index(min_severity)
        visible = [
            f for f in result.sorted_findings()
            if [s.value for s in Severity].index(f.severity.value) <= min_rank_order
        ]

        st.subheader(f"📋 Findings ({len(visible)} shown / {len(result.findings)} total)")

        if not visible:
            st.info("No findings at or above the selected severity threshold.")
        else:
            for f in visible:
                sev_class = f"sev-{f.severity.value}"
                with st.expander(f"[{f.severity.value}] {f.title} — {f.file}:{f.line}", expanded=(f.severity == Severity.CRITICAL)):
                    st.markdown(f'<span class="{sev_class}">{f.severity.value}</span> · CWE: {f.cwe} · Confidence: {f.confidence}', unsafe_allow_html=True)
                    st.code(f.snippet or "(snippet unavailable)")
                    st.markdown(f"**Description:** {f.description}")
                    st.markdown(f"**Exploitation scenario:** {f.exploitation_scenario}")
                    st.markdown(f"**Remediation:** {f.remediation}")

        # --- Downloadable report ---
        report_md = render_markdown_report(result, source_label)
        st.download_button(
            "📥 Download full report (Markdown)",
            data=report_md,
            file_name="security_review_report.md",
            mime="text/markdown",
        )

        # --- Optional AI deep dive ---
        if run_llm:
            st.markdown("---")
            st.subheader("🤖 AI deep-dive review")
            with st.spinner("Reasoning about exploitation scenarios and remediation priorities..."):
                try:
                    narrative = run_llm_review(api_key, result, code_samples, model=model)
                    st.markdown(narrative)
                    st.download_button(
                        "📥 Download AI review (Markdown)",
                        data=narrative,
                        file_name="ai_security_review.md",
                        mime="text/markdown",
                    )
                except Exception as e:
                    st.error(f"❌ AI review failed: {e}")

st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.caption("🔐 Static analysis runs locally; nothing is sent anywhere unless you enable the AI review.")
c2.caption("📚 Covers OWASP Top 10-style issues + secrets/crypto/config hygiene.")
c3.caption("⚡ Built with Streamlit")
