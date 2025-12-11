"""
Technical Writing Coach - Main Application
A Streamlit application for scenario-based technical documentation writing practice.
"""

import streamlit as st
from typing import Dict, List, Optional
from scenarios import SCENARIOS, WritingScenario, Difficulty
from analyzer import WritingAnalyzer, DocumentAnalysis, IssueSeverity, WritingIssue
from ai_feedback import get_ai_feedback, get_example_solution, AIFeedback

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Technical Spec Writing Coach",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom Styling
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=Source+Serif+Pro:wght@400;600&display=swap');
    
    :root {
        --primary: #1e3a5f;
        --secondary: #3d5a80;
        --accent: #ee6c4d;
        --success: #2d6a4f;
        --warning: #e9c46a;
        --error: #d62828;
        --bg-light: #f8f9fa;
    }
    
    .main { font-family: 'Source Sans Pro', sans-serif; background-color: var(--bg-light); }
    h1, h2, h3 { font-family: 'Source Serif Pro', serif; color: var(--primary); }
    
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: var(--secondary);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
    }
    
    .scenario-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #3d5a80 100%);
        color: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
    }
    .scenario-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .scenario-context {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .scenario-task {
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        white-space: pre-wrap;
    }
    .audience-tag {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-top: 1rem;
    }
    
    .feedback-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid var(--primary);
    }
    .issue-critical { border-left-color: var(--error); background: linear-gradient(135deg, #fff 0%, #fff5f5 100%); }
    .issue-warning { border-left-color: var(--warning); background: linear-gradient(135deg, #fff 0%, #fffbeb 100%); }
    .issue-suggestion { border-left-color: var(--success); background: linear-gradient(135deg, #fff 0%, #f0fdf4 100%); }
    
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: var(--primary); }
    .metric-label { color: #6c757d; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
    
    .score-badge { display: inline-block; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 700; font-size: 1.1rem; }
    .score-excellent { background: #d4edda; color: #155724; }
    .score-good { background: #cce5ff; color: #004085; }
    .score-needs-work { background: #fff3cd; color: #856404; }
    .score-poor { background: #f8d7da; color: #721c24; }
    
    .section-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
    }
    
    .tip-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1976d2;
        color: #1e3a5f;
        font-weight: 500;
    }
    
    .criteria-box {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .criteria-met { border-left: 4px solid #2d6a4f; }
    .criteria-not-met { border-left: 4px solid #d62828; }
    .criteria-partial { border-left: 4px solid #e9c46a; }
    
    .difficulty-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-left: 0.5rem; }
    .difficulty-beginner { background: #d4edda; color: #155724; }
    .difficulty-intermediate { background: #fff3cd; color: #856404; }
    .difficulty-advanced { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UI Components
# ============================================================================

def render_header():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">📋 Technical Spec Writing Coach</h1>
        <p style="color: #6c757d; font-size: 1.1rem;">
            Master the art of writing clear, compelling technical specifications for business stakeholders
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_scenario_selector() -> tuple[WritingScenario, Optional[str]]:
    with st.sidebar:
        st.markdown("### 🔑 AI-Powered Feedback")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key for AI-powered feedback. Leave blank for basic analysis only.",
            placeholder="sk-..."
        )
        
        if api_key:
            st.success("✅ AI feedback enabled")
        else:
            st.info("💡 Add API key for intelligent feedback")
        
        st.markdown("---")
        st.markdown("### 📋 Select a Scenario")
        
        difficulty_filter = st.radio(
            "Difficulty Level",
            options=["All", "Beginner", "Intermediate", "Advanced"],
            horizontal=True
        )
        
        if difficulty_filter == "Beginner":
            filtered = [s for s in SCENARIOS if s.difficulty == Difficulty.BEGINNER]
        elif difficulty_filter == "Intermediate":
            filtered = [s for s in SCENARIOS if s.difficulty == Difficulty.INTERMEDIATE]
        elif difficulty_filter == "Advanced":
            filtered = [s for s in SCENARIOS if s.difficulty == Difficulty.ADVANCED]
        else:
            filtered = SCENARIOS
        
        scenario_titles = {s.id: s.title for s in filtered}
        
        selected_id = st.selectbox(
            "Choose a scenario",
            options=list(scenario_titles.keys()),
            format_func=lambda x: scenario_titles[x]
        )
        
        selected_scenario = next((s for s in SCENARIOS if s.id == selected_id), None)
        
        st.markdown("---")
        
        if selected_scenario:
            st.markdown("### 📝 Requirements")
            for req in selected_scenario.requirements:
                st.markdown(f"- {req.description}")
            
            st.markdown("---")
            st.markdown("### 💡 Writing Tips")
            for hint in selected_scenario.hints[:3]:
                st.markdown(f"""<div class="tip-box">💡 {hint}</div>""", unsafe_allow_html=True)
        
        return selected_scenario, api_key if api_key else None


def render_scenario_card(scenario: WritingScenario):
    difficulty_class = {
        Difficulty.BEGINNER: "difficulty-beginner",
        Difficulty.INTERMEDIATE: "difficulty-intermediate",
        Difficulty.ADVANCED: "difficulty-advanced"
    }[scenario.difficulty]
    
    st.markdown(f"""
    <div class="scenario-card">
        <div class="scenario-title">
            📋 {scenario.title}
            <span class="difficulty-badge {difficulty_class}">{scenario.difficulty.value.title()}</span>
        </div>
        
        <div class="scenario-context">
            <strong>📍 Context:</strong><br/>
            {scenario.context}
        </div>
        
        <div class="scenario-task">
            <strong>✏️ Your Task:</strong><br/><br/>
            {scenario.task}
        </div>
        
        <div>
            <span class="audience-tag">👥 Audience: {scenario.audience}</span>
            <span class="audience-tag">📏 {scenario.min_words}-{scenario.max_words} words</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(analysis: DocumentAnalysis, scenario: WritingScenario):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score_class = (
            "score-excellent" if analysis.overall_score >= 85 else
            "score-good" if analysis.overall_score >= 70 else
            "score-needs-work" if analysis.overall_score >= 50 else
            "score-poor"
        )
        score_label = (
            "Excellent" if analysis.overall_score >= 85 else
            "Good" if analysis.overall_score >= 70 else
            "Needs Work" if analysis.overall_score >= 50 else
            "Poor"
        )
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{analysis.overall_score:.0f}</div>
            <div class="metric-label">Overall Score</div>
            <span class="score-badge {score_class}">{score_label}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{analysis.readability_score:.0f}</div>
            <div class="metric-label">Readability</div>
            <small style="color: #6c757d;">Higher = easier to read</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        word_status = "✅" if scenario.min_words <= analysis.word_count <= scenario.max_words else "⚠️"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{analysis.word_count}</div>
            <div class="metric-label">Words</div>
            <small style="color: #6c757d;">{word_status} Target: {scenario.min_words}-{scenario.max_words}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        length_status = (
            "✅ Good" if 15 <= analysis.avg_sentence_length <= 25 else
            "⚠️ Short" if analysis.avg_sentence_length < 15 else
            "⚠️ Long"
        )
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{analysis.avg_sentence_length:.1f}</div>
            <div class="metric-label">Avg Words/Sentence</div>
            <small style="color: #6c757d;">{length_status}</small>
        </div>
        """, unsafe_allow_html=True)


def render_requirement_scores(requirement_scores: Dict[str, float], scenario: WritingScenario):
    st.markdown("""<div class="section-header">📋 Scenario Requirements Evaluation</div>""", unsafe_allow_html=True)
    
    for req in scenario.requirements:
        score = requirement_scores.get(req.name, 0)
        
        if score >= 0.7:
            status_class, status_icon, status_text = "criteria-met", "✅", "Met"
        elif score >= 0.3:
            status_class, status_icon, status_text = "criteria-partial", "⚠️", "Partial"
        else:
            status_class, status_icon, status_text = "criteria-not-met", "❌", "Not Met"
        
        st.markdown(f"""
        <div class="criteria-box {status_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span><strong>{req.description}</strong></span>
                <span>{status_icon} {status_text} ({score*100:.0f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_issues(issues: List[WritingIssue], title: str, icon: str):
    if not issues:
        return
    
    st.markdown(f"""<div class="section-header">{icon} {title} ({len(issues)} {"issue" if len(issues) == 1 else "issues"})</div>""", unsafe_allow_html=True)
    
    for issue in issues:
        severity_class = {
            IssueSeverity.CRITICAL: "issue-critical",
            IssueSeverity.WARNING: "issue-warning",
            IssueSeverity.SUGGESTION: "issue-suggestion",
        }[issue.severity]
        
        severity_label = {
            IssueSeverity.CRITICAL: "🔴 Critical",
            IssueSeverity.WARNING: "🟡 Warning",
            IssueSeverity.SUGGESTION: "🟢 Suggestion",
        }[issue.severity]
        
        st.markdown(f"""
        <div class="feedback-card {severity_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <strong>{issue.category}</strong>
                <span style="font-size: 0.85rem;">{severity_label}</span>
            </div>
            <div style="font-family: monospace; background: #f8f9fa; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0;">
                "{issue.text}"
            </div>
            <div style="color: #495057;">
                💡 <strong>Suggestion:</strong> {issue.suggestion}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_summary(analysis: DocumentAnalysis, scenario: WritingScenario):
    st.markdown("""<div class="section-header">📊 Summary & Next Steps</div>""", unsafe_allow_html=True)
    
    all_issues = (analysis.spelling_issues + analysis.grammar_issues + 
                  analysis.punctuation_issues + analysis.flow_issues + analysis.clarity_issues)
    
    critical_count = sum(1 for i in all_issues if i.severity == IssueSeverity.CRITICAL)
    warning_count = sum(1 for i in all_issues if i.severity == IssueSeverity.WARNING)
    suggestion_count = sum(1 for i in all_issues if i.severity == IssueSeverity.SUGGESTION)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Critical Issues", critical_count)
    with col2:
        st.metric("🟡 Warnings", warning_count)
    with col3:
        st.metric("🟢 Suggestions", suggestion_count)
    
    st.markdown("### 📝 Recommendations")
    
    recommendations = []
    
    low_reqs = [req for req in scenario.requirements 
                if analysis.requirement_scores.get(req.name, 0) < 0.5]
    for req in low_reqs[:2]:
        recommendations.append(f"📋 **{req.description}** - This requirement needs more attention.")
    
    if critical_count > 0:
        recommendations.append("✏️ Fix critical spelling and grammar errors first for professional credibility.")
    
    if analysis.avg_sentence_length > 25:
        recommendations.append("✂️ Break down long sentences for better readability (aim for 15-25 words).")
    
    if analysis.word_count < scenario.min_words:
        recommendations.append(f"📏 Add more detail - your document is below the minimum {scenario.min_words} words.")
    elif analysis.word_count > scenario.max_words:
        recommendations.append(f"📏 Tighten your writing - aim to stay under {scenario.max_words} words.")
    
    if len(analysis.flow_issues) > 5:
        recommendations.append("🔄 Remove filler words and simplify phrases for more direct communication.")
    
    if not recommendations:
        recommendations.append("🎉 Great work! Your documentation meets the key requirements.")
    
    for rec in recommendations[:5]:
        st.markdown(f"- {rec}")


def render_ai_feedback(ai_feedback: AIFeedback, scenario: WritingScenario):
    """Render AI-powered feedback section."""
    
    st.markdown("""
    <div class="section-header">
        🤖 AI-Powered Analysis
    </div>
    """, unsafe_allow_html=True)
    
    # Overall Assessment
    st.markdown(f"""
    <div class="feedback-card" style="border-left-color: #6366f1;">
        <h4 style="margin-top: 0;">📝 Overall Assessment</h4>
        <p>{ai_feedback.overall_assessment}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Scores
    st.markdown("#### 📊 AI Quality Scores")
    col1, col2, col3, col4 = st.columns(4)
    
    def score_color(score):
        if score >= 8: return "#2d6a4f"
        elif score >= 6: return "#3d5a80"
        elif score >= 4: return "#e9c46a"
        else: return "#d62828"
    
    with col1:
        color = score_color(ai_feedback.audience_fit_score)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: {color};">{ai_feedback.audience_fit_score}/10</div>
            <div class="metric-label">Audience Fit</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = score_color(ai_feedback.technical_accuracy_score)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: {color};">{ai_feedback.technical_accuracy_score}/10</div>
            <div class="metric-label">Technical Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = score_color(ai_feedback.clarity_score)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: {color};">{ai_feedback.clarity_score}/10</div>
            <div class="metric-label">Clarity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        color = score_color(ai_feedback.completeness_score)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: {color};">{ai_feedback.completeness_score}/10</div>
            <div class="metric-label">Completeness</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Strengths and Improvements
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Strengths")
        for strength in ai_feedback.strengths:
            st.markdown(f"""
            <div style="background: #d4edda; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #2d6a4f;">
                ✓ {strength}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🔧 Areas for Improvement")
        for improvement in ai_feedback.improvements:
            st.markdown(f"""
            <div style="background: #fff3cd; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #e9c46a;">
                → {improvement}
            </div>
            """, unsafe_allow_html=True)
    
    # Requirement-specific feedback
    if ai_feedback.requirement_feedback:
        st.markdown("#### 📋 Requirement-Specific Feedback")
        for req in scenario.requirements:
            feedback = ai_feedback.requirement_feedback.get(req.name, "")
            if feedback:
                st.markdown(f"""
                <div class="criteria-box" style="margin: 0.5rem 0;">
                    <strong>{req.description}</strong><br/>
                    <span style="color: #495057;">{feedback}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Rewrite Suggestions
    if ai_feedback.rewrite_suggestions:
        st.markdown("#### ✨ Suggested Rewrites")
        for i, suggestion in enumerate(ai_feedback.rewrite_suggestions[:3]):
            with st.expander(f"Suggestion {i+1}: Improve this passage"):
                st.markdown("**Original:**")
                st.code(suggestion.get("original", ""), language=None)
                st.markdown("**Suggested:**")
                st.code(suggestion.get("suggested", ""), language=None)
                st.markdown(f"**Why:** {suggestion.get('reason', '')}")


def render_example_solution(example: str):
    """Render an example solution."""
    st.markdown("""
    <div class="section-header">
        📚 Example Solution
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border: 1px solid #e0e0e0;">
    """, unsafe_allow_html=True)
    
    st.markdown(example)
    
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Main Application
# ============================================================================

def main():
    render_header()
    
    selected_scenario, api_key = render_scenario_selector()
    
    if not selected_scenario:
        st.info("Please select a scenario from the sidebar to begin.")
        return
    
    render_scenario_card(selected_scenario)
    
    # Show Example button if API key is available
    if api_key:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📚 Show Example Solution", use_container_width=True):
                with st.spinner("Generating example solution..."):
                    example = get_example_solution(selected_scenario, api_key)
                    if example:
                        st.session_state['example_solution'] = example
        
        if 'example_solution' in st.session_state:
            render_example_solution(st.session_state['example_solution'])
            if st.button("Hide Example"):
                del st.session_state['example_solution']
                st.rerun()
    
    st.markdown("### ✏️ Write Your Documentation")
    
    user_text = st.text_area(
        "Enter your documentation here:",
        height=400,
        placeholder=f"Write {selected_scenario.min_words}-{selected_scenario.max_words} words addressing the scenario above...",
        help=f"Target: {selected_scenario.min_words}-{selected_scenario.max_words} words"
    )
    
    current_words = len(user_text.split()) if user_text.strip() else 0
    word_color = (
        "#2d6a4f" if selected_scenario.min_words <= current_words <= selected_scenario.max_words else
        "#e9c46a" if current_words > 0 else "#6c757d"
    )
    st.markdown(f"""
    <div style="text-align: right; color: {word_color}; margin-top: -10px; margin-bottom: 10px;">
        {current_words} / {selected_scenario.min_words}-{selected_scenario.max_words} words
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        button_text = "🔍 Evaluate with AI" if api_key else "🔍 Evaluate My Documentation"
        evaluate_button = st.button(button_text, use_container_width=True)
    
    if evaluate_button:
        if not user_text.strip():
            st.warning("Please write some documentation before evaluating.")
            return
        
        if current_words < 50:
            st.warning("Please write at least 50 words for a meaningful evaluation.")
            return
        
        # Basic analysis
        with st.spinner("Analyzing your documentation..."):
            analyzer = WritingAnalyzer()
            analysis = analyzer.analyze(user_text, selected_scenario)
        
        st.markdown("---")
        render_metrics(analysis, selected_scenario)
        
        # AI Feedback (if API key provided)
        ai_feedback = None
        if api_key:
            with st.spinner("🤖 Getting AI-powered feedback..."):
                ai_feedback = get_ai_feedback(user_text, selected_scenario, api_key)
        
        if ai_feedback:
            st.markdown("---")
            render_ai_feedback(ai_feedback, selected_scenario)
        
        st.markdown("---")
        render_requirement_scores(analysis.requirement_scores, selected_scenario)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            render_issues(analysis.spelling_issues, "Spelling Issues", "📝")
            render_issues(analysis.grammar_issues, "Grammar Issues", "📖")
            render_issues(analysis.punctuation_issues, "Punctuation Issues", "✏️")
        with col2:
            render_issues(analysis.flow_issues, "Flow & Style", "🔄")
            render_issues(analysis.clarity_issues, "Clarity Issues", "💡")
        
        st.markdown("---")
        render_summary(analysis, selected_scenario)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <small>
            Technical Spec Writing Coach • Powered by Data2Trend Analytics<br/>
            Build your business writing skills for technical documentation
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
