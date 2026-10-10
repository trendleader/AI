"""
AWS Certified Solutions Architect – Associate (SAA-C03) Simulated Exam

- 65 medium/hard multiple-choice and multiple-response questions across all 4 domains
- 90-minute countdown timer (exam auto-submits when time runs out)
- Scaled score 100–1000, passing score 720 (same scale as the real exam)
- Results e-mailed to the instructor when the exam is completed

Run:
    streamlit run aws_saa_exam_app.py

E-mail notification is configured with Streamlit secrets (.streamlit/secrets.toml)
or environment variables. See .streamlit/secrets.toml.example.
"""

import csv
import io
import os
import random
import smtplib
import ssl
import time
from datetime import datetime
from email.message import EmailMessage

import pandas as pd
import streamlit as st

from aws_saa_exam_questions import DOMAINS, QUESTIONS

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

EXAM_TITLE = "AWS Certified Solutions Architect – Associate (SAA-C03) Practice Exam"
EXAM_DURATION_SECONDS = 90 * 60
MIN_SCALED_SCORE = 100
MAX_SCALED_SCORE = 1000
PASSING_SCORE = 720
DEFAULT_INSTRUCTOR_EMAIL = "philjones820@gmail.com"
LETTERS = "ABCDEFGH"

st.set_page_config(page_title="AWS SAA-C03 Practice Exam", page_icon="☁️", layout="wide")


def get_setting(name, default=None):
    """Read a setting from st.secrets['smtp'], st.secrets, or the environment."""
    try:
        if "smtp" in st.secrets and name in st.secrets["smtp"]:
            return st.secrets["smtp"][name]
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # no secrets.toml present
    return os.environ.get(name, default)


# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "stage": "start",          # start -> exam -> results
        "student_name": "",
        "student_email": "",
        "order": [],               # question ids in display order
        "option_order": {},        # qid -> shuffled option indexes
        "answers": {},             # qid -> set of selected option indexes
        "flagged": set(),
        "current": 0,
        "start_time": None,
        "end_time": None,
        "auto_submitted": False,
        "result": None,
        "email_status": None,
        "confirm_submit": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


QUESTIONS_BY_ID = {q["id"]: q for q in QUESTIONS}


def start_exam(name, email):
    rng = random.Random()
    order = [q["id"] for q in QUESTIONS]
    rng.shuffle(order)
    option_order = {}
    for q in QUESTIONS:
        idx = list(range(len(q["options"])))
        rng.shuffle(idx)
        option_order[q["id"]] = idx

    st.session_state.update(
        stage="exam",
        student_name=name.strip(),
        student_email=email.strip(),
        order=order,
        option_order=option_order,
        answers={},
        flagged=set(),
        current=0,
        start_time=time.time(),
        end_time=None,
        auto_submitted=False,
        result=None,
        email_status=None,
        confirm_submit=False,
    )


def remaining_seconds():
    if st.session_state.start_time is None:
        return EXAM_DURATION_SECONDS
    elapsed = time.time() - st.session_state.start_time
    return max(0, int(EXAM_DURATION_SECONDS - elapsed))


def fmt_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────

def scaled_score(correct, total):
    """Linear mapping of raw score onto the 100–1000 AWS scale."""
    return round(MIN_SCALED_SCORE + (MAX_SCALED_SCORE - MIN_SCALED_SCORE) * correct / total)


def grade_exam():
    answers = st.session_state.answers
    details = []
    domain_stats = {d: {"correct": 0, "total": 0} for d in DOMAINS}

    for number, qid in enumerate(st.session_state.order, start=1):
        q = QUESTIONS_BY_ID[qid]
        selected = set(answers.get(qid, set()))
        correct_set = set(q["answer"])
        is_correct = selected == correct_set  # multiple-response: all-or-nothing, like the real exam
        domain_stats[q["domain"]]["total"] += 1
        if is_correct:
            domain_stats[q["domain"]]["correct"] += 1
        details.append({
            "number": number,
            "qid": qid,
            "domain": q["domain"],
            "difficulty": q["difficulty"],
            "selected": selected,
            "correct": correct_set,
            "is_correct": is_correct,
            "answered": bool(selected),
        })

    total = len(QUESTIONS)
    correct = sum(d["is_correct"] for d in details)
    score = scaled_score(correct, total)
    end = st.session_state.end_time or time.time()
    return {
        "correct": correct,
        "total": total,
        "answered": sum(d["answered"] for d in details),
        "score": score,
        "passed": score >= PASSING_SCORE,
        "domains": domain_stats,
        "details": details,
        "duration": min(end - st.session_state.start_time, EXAM_DURATION_SECONDS),
        "started": datetime.fromtimestamp(st.session_state.start_time),
        "finished": datetime.fromtimestamp(end),
        "auto_submitted": st.session_state.auto_submitted,
    }


def submit_exam(auto=False):
    if st.session_state.stage != "exam":
        return
    st.session_state.end_time = time.time()
    st.session_state.auto_submitted = auto
    st.session_state.result = grade_exam()
    st.session_state.stage = "results"
    st.session_state.email_status = send_results_email(st.session_state.result)


# ─────────────────────────────────────────────────────────────
# E-MAIL NOTIFICATION
# ─────────────────────────────────────────────────────────────

def letters_for(qid, indexes):
    """Map original option indexes to the letters the student saw."""
    order = st.session_state.option_order[qid]
    return ", ".join(sorted(LETTERS[order.index(i)] for i in indexes)) or "—"


def build_email(result):
    name = st.session_state.student_name or "Unnamed student"
    status = "PASS" if result["passed"] else "FAIL"
    subject = f"[AWS SAA Practice Exam] {name}: {result['score']}/1000 ({status})"

    domain_rows_txt = []
    domain_rows_html = []
    for d, s in result["domains"].items():
        pct = 100 * s["correct"] / s["total"] if s["total"] else 0
        domain_rows_txt.append(f"  - {d}: {s['correct']}/{s['total']} ({pct:.0f}%)")
        domain_rows_html.append(
            f"<tr><td>{d}</td><td align='center'>{s['correct']}/{s['total']}</td>"
            f"<td align='center'>{pct:.0f}%</td></tr>"
        )

    missed = [d for d in result["details"] if not d["is_correct"]]
    missed_txt = [
        f"  Q{d['number']} [{d['domain']}] chose {letters_for(d['qid'], d['selected'])}, "
        f"correct {letters_for(d['qid'], d['correct'])}"
        for d in missed
    ]

    submitted = "Auto-submitted (time expired)" if result["auto_submitted"] else "Submitted by student"
    text = "\n".join([
        f"{EXAM_TITLE}",
        "",
        f"Student:        {name}",
        f"Student email:  {st.session_state.student_email or '—'}",
        f"Started:        {result['started']:%Y-%m-%d %H:%M:%S}",
        f"Finished:       {result['finished']:%Y-%m-%d %H:%M:%S}",
        f"Time used:      {fmt_duration(result['duration'])} of {fmt_duration(EXAM_DURATION_SECONDS)}",
        f"Submission:     {submitted}",
        "",
        f"Scaled score:   {result['score']} / 1000 (passing {PASSING_SCORE}) -> {status}",
        f"Raw score:      {result['correct']} / {result['total']} correct, {result['answered']} answered",
        "",
        "Domain breakdown:",
        *domain_rows_txt,
        "",
        f"Incorrect / unanswered questions ({len(missed)}):",
        *(missed_txt or ["  None — perfect score!"]),
    ])

    color = "#1a7f37" if result["passed"] else "#cf222e"
    html = f"""
    <html><body style="font-family:Arial,sans-serif;color:#222">
      <h2>{EXAM_TITLE}</h2>
      <p><b>Student:</b> {name}<br>
         <b>Student email:</b> {st.session_state.student_email or '—'}<br>
         <b>Started:</b> {result['started']:%Y-%m-%d %H:%M:%S}<br>
         <b>Finished:</b> {result['finished']:%Y-%m-%d %H:%M:%S}<br>
         <b>Time used:</b> {fmt_duration(result['duration'])} of {fmt_duration(EXAM_DURATION_SECONDS)}<br>
         <b>Submission:</b> {submitted}</p>
      <h1 style="color:{color}">{result['score']} / 1000 — {status}</h1>
      <p>Passing score: {PASSING_SCORE}. Raw score: {result['correct']}/{result['total']} correct
         ({result['answered']} answered).</p>
      <h3>Domain breakdown</h3>
      <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
        <tr style="background:#f0f0f0"><th>Domain</th><th>Correct</th><th>%</th></tr>
        {''.join(domain_rows_html)}
      </table>
      <p>The attached CSV lists every question with the student's answer and the correct answer.</p>
    </body></html>
    """
    return subject, text, html


def build_csv(result):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["#", "Domain", "Difficulty", "Question", "Student answer", "Correct answer", "Result"])
    for d in result["details"]:
        q = QUESTIONS_BY_ID[d["qid"]]
        writer.writerow([
            d["number"], d["domain"], d["difficulty"], q["question"],
            letters_for(d["qid"], d["selected"]), letters_for(d["qid"], d["correct"]),
            "Correct" if d["is_correct"] else ("Unanswered" if not d["answered"] else "Incorrect"),
        ])
    return buf.getvalue()


def send_results_email(result):
    """Send the results to the instructor. Returns (ok: bool, message: str)."""
    host = get_setting("SMTP_HOST", "smtp.gmail.com")
    port = int(get_setting("SMTP_PORT", 587))
    user = get_setting("SMTP_USER")
    password = get_setting("SMTP_PASSWORD")
    sender = get_setting("SMTP_FROM", user)
    recipient = get_setting("NOTIFY_EMAIL", DEFAULT_INSTRUCTOR_EMAIL)

    if not (user and password):
        return False, "E-mail is not configured (SMTP_USER / SMTP_PASSWORD missing)."

    subject, text, html = build_email(result)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    if st.session_state.student_email:
        msg["Reply-To"] = st.session_state.student_email
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    safe_name = "".join(c for c in (st.session_state.student_name or "student") if c.isalnum() or c in "-_") or "student"
    msg.add_attachment(
        build_csv(result).encode("utf-8"), maintype="text", subtype="csv",
        filename=f"saa_exam_{safe_name}_{result['finished']:%Y%m%d_%H%M}.csv",
    )

    try:
        context = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.starttls(context=context)
                server.login(user, password)
                server.send_message(msg)
        return True, "Your results have been sent to your instructor."
    except Exception as exc:  # report but never lose the student's results
        return False, f"Could not send results e-mail: {exc}"


# ─────────────────────────────────────────────────────────────
# UI – START PAGE
# ─────────────────────────────────────────────────────────────

def render_start():
    st.title("☁️ " + EXAM_TITLE)
    st.markdown(
        f"""
        **Format:** {len(QUESTIONS)} questions · multiple choice and multiple response · medium to hard
        **Time limit:** {EXAM_DURATION_SECONDS // 60} minutes — the exam submits automatically when time runs out
        **Scoring:** scaled score from {MIN_SCALED_SCORE} to {MAX_SCALED_SCORE} · passing score **{PASSING_SCORE}**
        **Multiple-response questions** ("Choose TWO") are scored all-or-nothing, like the real exam. There is no penalty for guessing.
        """
    )

    domain_counts = pd.Series([q["domain"] for q in QUESTIONS]).value_counts().reindex(DOMAINS)
    st.subheader("Domains covered")
    st.dataframe(
        pd.DataFrame({
            "Domain": DOMAINS,
            "Questions": domain_counts.values,
            "Weight": [f"{100 * c / len(QUESTIONS):.0f}%" for c in domain_counts.values],
        }),
        hide_index=True, width="stretch",
    )

    st.subheader("Candidate details")
    with st.form("start_form"):
        name = st.text_input("Full name *")
        email = st.text_input("Your e-mail (optional)")
        agree = st.checkbox("I understand the timer starts as soon as I begin and cannot be paused.")
        submitted = st.form_submit_button("🚀 Start exam", type="primary")
    if submitted:
        if not name.strip():
            st.error("Please enter your name.")
        elif not agree:
            st.error("Please confirm you understand the timer rules.")
        else:
            start_exam(name, email)
            st.rerun()


# ─────────────────────────────────────────────────────────────
# UI – EXAM
# ─────────────────────────────────────────────────────────────

@st.fragment(run_every=1)
def render_timer():
    """Live countdown; re-runs every second and auto-submits at zero."""
    if st.session_state.stage != "exam":
        return
    left = remaining_seconds()
    if left <= 0:
        submit_exam(auto=True)
        st.rerun()
    color = "#cf222e" if left <= 300 else ("#bf8700" if left <= 900 else "#1a7f37")
    st.markdown(
        f"<div style='font-size:2rem;font-weight:700;color:{color};text-align:center'>⏱ {fmt_duration(left)}</div>"
        "<div style='text-align:center;color:gray'>time remaining</div>",
        unsafe_allow_html=True,
    )
    if left <= 300:
        st.warning("Less than 5 minutes left!")


def go_to(index):
    st.session_state.current = index
    st.session_state.confirm_submit = False


def save_single(qid, key):
    value = st.session_state.get(key)
    st.session_state.answers[qid] = set() if value is None else {value}


def save_multi(qid, option_index, key):
    selected = set(st.session_state.answers.get(qid, set()))
    if st.session_state.get(key):
        selected.add(option_index)
    else:
        selected.discard(option_index)
    st.session_state.answers[qid] = selected


def toggle_flag(qid):
    st.session_state.flagged ^= {qid}


def render_sidebar():
    order = st.session_state.order
    answers = st.session_state.answers
    flagged = st.session_state.flagged
    with st.sidebar:
        render_timer()
        st.divider()
        answered = sum(1 for qid in order if answers.get(qid))
        st.progress(answered / len(order), text=f"Answered {answered} / {len(order)}")
        st.caption("✅ answered · 🚩 flagged · ⬜ not answered")
        cols_per_row = 5
        for row_start in range(0, len(order), cols_per_row):
            cols = st.columns(cols_per_row)
            for offset, col in enumerate(cols):
                i = row_start + offset
                if i >= len(order):
                    break
                qid = order[i]
                icon = "🚩" if qid in flagged else ("✅" if answers.get(qid) else "⬜")
                col.button(
                    f"{icon}{i + 1}", key=f"nav_{i}", on_click=go_to, args=(i,),
                    type="primary" if i == st.session_state.current else "secondary",
                    width="stretch",
                )
        st.divider()
        if st.button("📝 Review & submit", width="stretch"):
            st.session_state.confirm_submit = True


def render_question():
    order = st.session_state.order
    i = st.session_state.current
    qid = order[i]
    q = QUESTIONS_BY_ID[qid]
    option_order = st.session_state.option_order[qid]
    selected = st.session_state.answers.get(qid, set())
    multi = len(q["answer"]) > 1

    st.caption(f"Question {i + 1} of {len(order)} · {q['domain']}")
    st.markdown(f"#### {q['question']}")
    if multi:
        st.info(f"Select **{len(q['answer'])}** answers.")

    if multi:
        for pos, opt_idx in enumerate(option_order):
            key = f"q{qid}_opt{opt_idx}"
            if key not in st.session_state:  # restore after navigating away
                st.session_state[key] = opt_idx in selected
            st.checkbox(
                f"{LETTERS[pos]}. {q['options'][opt_idx]}", key=key,
                on_change=save_multi, args=(qid, opt_idx, key),
            )
        if len(selected) > len(q["answer"]):
            st.warning(f"You selected {len(selected)} options; this question requires {len(q['answer'])}.")
    else:
        key = f"q{qid}_radio"
        if key not in st.session_state:
            st.session_state[key] = next(iter(selected), None)
        st.radio(
            "Select one answer:", option_order, key=key, label_visibility="collapsed",
            format_func=lambda idx: f"{LETTERS[option_order.index(idx)]}. {q['options'][idx]}",
            on_change=save_single, args=(qid, key),
        )

    st.write("")
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    c1.button("⬅ Previous", disabled=i == 0, on_click=go_to, args=(i - 1,), width="stretch")
    c2.button(
        "🚩 Unflag" if qid in st.session_state.flagged else "🚩 Flag for review",
        on_click=toggle_flag, args=(qid,), width="stretch",
    )
    c3.button("Next ➡", disabled=i == len(order) - 1, on_click=go_to, args=(i + 1,), width="stretch")


def render_submit_panel():
    order = st.session_state.order
    answers = st.session_state.answers
    unanswered = [n for n, qid in enumerate(order, 1) if not answers.get(qid)]
    flagged = [n for n, qid in enumerate(order, 1) if qid in st.session_state.flagged]

    st.subheader("Review & submit")
    if unanswered:
        st.warning(f"**{len(unanswered)} unanswered:** " + ", ".join(f"Q{n}" for n in unanswered))
    else:
        st.success("All questions answered.")
    if flagged:
        st.info("**Flagged for review:** " + ", ".join(f"Q{n}" for n in flagged))
    st.write("Once submitted you cannot change your answers.")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("✅ Submit exam", type="primary", width="stretch"):
        with st.spinner("Scoring your exam and notifying your instructor..."):
            submit_exam(auto=False)
        st.rerun()
    if c2.button("Return to exam", width="stretch"):
        st.session_state.confirm_submit = False
        st.rerun()


def render_exam():
    # Server-side guard in case the browser was inactive when time ran out.
    if remaining_seconds() <= 0:
        submit_exam(auto=True)
        st.rerun()

    render_sidebar()
    st.title("AWS SAA-C03 Practice Exam")
    st.caption(f"Candidate: {st.session_state.student_name}")
    if st.session_state.confirm_submit:
        render_submit_panel()
    else:
        render_question()


# ─────────────────────────────────────────────────────────────
# UI – RESULTS
# ─────────────────────────────────────────────────────────────

def render_results():
    r = st.session_state.result
    st.title("📊 Exam results")
    st.caption(f"Candidate: {st.session_state.student_name} · finished {r['finished']:%Y-%m-%d %H:%M}")
    if r["auto_submitted"]:
        st.warning("⏰ Time expired — the exam was submitted automatically.")

    ok, message = st.session_state.email_status or (False, "")
    if ok:
        st.success("📧 " + message)
    elif message:
        st.error("📧 " + message)
        if st.button("Retry sending e-mail"):
            st.session_state.email_status = send_results_email(r)
            st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scaled score", f"{r['score']} / 1000")
    c2.metric("Result", "PASS ✅" if r["passed"] else "FAIL ❌", f"passing score {PASSING_SCORE}", delta_color="off")
    c3.metric("Correct", f"{r['correct']} / {r['total']}")
    c4.metric("Time used", fmt_duration(r["duration"]))
    st.progress((r["score"] - MIN_SCALED_SCORE) / (MAX_SCALED_SCORE - MIN_SCALED_SCORE))

    st.subheader("Performance by domain")
    domain_df = pd.DataFrame([
        {
            "Domain": d,
            "Correct": s["correct"],
            "Total": s["total"],
            "Score %": round(100 * s["correct"] / s["total"]) if s["total"] else 0,
            "Status": "Meets competency" if s["total"] and s["correct"] / s["total"] >= 0.72 else "Needs improvement",
        }
        for d, s in r["domains"].items()
    ])
    st.dataframe(domain_df, hide_index=True, width="stretch")
    st.bar_chart(domain_df.set_index("Domain")["Score %"], horizontal=True)

    st.subheader("Answer review")
    view = st.radio("Show", ["Incorrect / unanswered", "All questions"], horizontal=True)
    for d in r["details"]:
        if view != "All questions" and d["is_correct"]:
            continue
        q = QUESTIONS_BY_ID[d["qid"]]
        order = st.session_state.option_order[d["qid"]]
        icon = "✅" if d["is_correct"] else ("⬜" if not d["answered"] else "❌")
        with st.expander(f"{icon} Q{d['number']} · {q['domain']} · {q['difficulty']}"):
            st.markdown(f"**{q['question']}**")
            for pos, opt_idx in enumerate(order):
                text = f"{LETTERS[pos]}. {q['options'][opt_idx]}"
                if opt_idx in d["correct"]:
                    marker = "✅ **correct answer**" + (" — your choice" if opt_idx in d["selected"] else "")
                    st.markdown(f"- {text}  \n  {marker}")
                elif opt_idx in d["selected"]:
                    st.markdown(f"- {text}  \n  ❌ your choice")
                else:
                    st.markdown(f"- {text}")
            st.info(q["explanation"])

    st.download_button(
        "⬇ Download results (CSV)", build_csv(r), file_name="saa_exam_results.csv", mime="text/csv",
    )
    if st.button("🔁 Take the exam again"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

init_state()
stage = st.session_state.stage
if stage == "start":
    render_start()
elif stage == "exam":
    render_exam()
else:
    render_results()
