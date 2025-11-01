import streamlit as st
import openai
import json
import os
from typing import Optional
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SQL Challenges - Interview Prep",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Built-in SQL challenges data
BUILTIN_CHALLENGES = [
    {
        "id": 1,
        "title": "The Second-Highest Salary Trap",
        "difficulty": "Medium",
        "question": "Find the second-highest salary in each department without using window functions, subqueries, or CTEs.",
        "setup": """CREATE TABLE employees (
    id INT,
    name VARCHAR(50),
    department VARCHAR(50),
    salary INT
);

INSERT INTO employees VALUES
(1, 'Alice', 'Engineering', 95000),
(2, 'Bob', 'Engineering', 87000),
(3, 'Carol', 'Engineering', 92000),
(4, 'David', 'Sales', 78000),
(5, 'Eve', 'Sales', 82000),
(6, 'Frank', 'Sales', 75000);""",
        "solution": """SELECT e1.department, MAX(e1.salary) as second_highest
FROM employees e1
JOIN employees e2 ON e1.department = e2.department
    AND e1.salary < e2.salary
GROUP BY e1.department;""",
        "explanation": "Uses a self-join with an inequality condition to eliminate the highest salary, then takes MAX of remaining salaries.",
        "key_insight": "Self-join with inequality (< or >) is a pattern most analysts never consider because JOINs are typically equality operations.",
        "is_custom": False
    },
    {
        "id": 2,
        "title": "The Consecutive Days Puzzle",
        "difficulty": "Hard",
        "question": "Find users who logged in on 3 or more consecutive days.",
        "setup": """CREATE TABLE user_logins (
    user_id INT,
    login_date DATE
);

INSERT INTO user_logins VALUES
(1, '2024-01-01'), (1, '2024-01-02'), (1, '2024-01-03'),
(1, '2024-01-05'), (1, '2024-01-06'),
(2, '2024-01-01'), (2, '2024-01-03'), (2, '2024-01-04');""",
        "solution": """WITH numbered_logins AS (
    SELECT user_id, login_date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) as rn
    FROM user_logins
),
grouped_sequences AS (
    SELECT user_id, login_date,
        DATE_SUB(login_date, INTERVAL rn DAY) as group_date
    FROM numbered_logins
),
consecutive_counts AS (
    SELECT user_id, group_date, COUNT(*) as consecutive_days
    FROM grouped_sequences
    GROUP BY user_id, group_date
)
SELECT DISTINCT user_id
FROM consecutive_counts
WHERE consecutive_days >= 3;""",
        "explanation": "Uses date arithmetic trick: DATE_SUB(login_date, INTERVAL rn DAY) creates identical values for consecutive dates.",
        "key_insight": "For consecutive dates, subtracting the row number produces the same value. This 'date arithmetic trick' is what 90% of analysts miss.",
        "is_custom": False
    },
    {
        "id": 3,
        "title": "The Percentage of Total Paradox",
        "difficulty": "Medium",
        "question": "Show each region's sales as a percentage of total, and each product's percentage within its region.",
        "setup": """CREATE TABLE sales (
    region VARCHAR(50),
    product VARCHAR(50),
    amount INT
);

INSERT INTO sales VALUES
('North', 'Laptop', 1000), ('North', 'Phone', 800),
('South', 'Laptop', 1200), ('South', 'Phone', 600),
('East', 'Laptop', 900), ('East', 'Phone', 700);""",
        "solution": """SELECT
    region,
    product,
    amount,
    ROUND(100.0 * SUM(amount) OVER (PARTITION BY region) /
        SUM(amount) OVER (), 1) as region_pct_of_total,
    ROUND(100.0 * amount /
        SUM(amount) OVER (PARTITION BY region), 1) as product_pct_of_region
FROM sales
ORDER BY region, product;""",
        "explanation": "Uses multiple window functions with different PARTITION BY clauses in the same query.",
        "key_insight": "Most analysts think you can only use one partition per query. SUM(amount) OVER () (empty partition) calculates the grand total.",
        "is_custom": False
    },
    {
        "id": 4,
        "title": "The Islands and Gaps Challenge",
        "difficulty": "Hard",
        "question": "Find the start and end of each consecutive sequence of ticket numbers.",
        "setup": """CREATE TABLE tickets (
    ticket_id INT PRIMARY KEY
);

INSERT INTO tickets VALUES (1), (2), (3), (7), (8), (9), (10), (15), (16);""",
        "solution": """WITH numbered AS (
    SELECT ticket_id,
        ticket_id - ROW_NUMBER() OVER (ORDER BY ticket_id) as group_id
    FROM tickets
),
groups AS (
    SELECT group_id,
        MIN(ticket_id) as sequence_start,
        MAX(ticket_id) as sequence_end
    FROM numbered
    GROUP BY group_id
)
SELECT sequence_start, sequence_end
FROM groups
ORDER BY sequence_start;""",
        "explanation": "For consecutive integers, subtracting the row number creates identical group identifiers.",
        "key_insight": "This mathematical approach breaks most analysts' mental models of GROUP BY operations.",
        "is_custom": False
    },
    {
        "id": 5,
        "title": "The Running Balance Mind-Bender",
        "difficulty": "Hard",
        "question": "Calculate running balance for each account, showing balance BEFORE and AFTER each transaction.",
        "setup": """CREATE TABLE transactions (
    account_id INT,
    transaction_date DATE,
    amount DECIMAL(10,2),
    transaction_type VARCHAR(10)
);

INSERT INTO transactions VALUES
(1, '2024-01-01', 1000.00, 'credit'),
(1, '2024-01-02', 500.00, 'debit'),
(1, '2024-01-03', 200.00, 'credit'),
(1, '2024-01-04', 800.00, 'debit'),
(2, '2024-01-01', 2000.00, 'credit'),
(2, '2024-01-02', 300.00, 'debit');""",
        "solution": """WITH transaction_amounts AS (
    SELECT account_id, transaction_date, amount,
        CASE WHEN transaction_type = 'debit' THEN -amount ELSE amount END as signed_amount
    FROM transactions
),
running_totals AS (
    SELECT account_id, transaction_date, amount, signed_amount,
        SUM(signed_amount) OVER (
            PARTITION BY account_id
            ORDER BY transaction_date
            ROWS UNBOUNDED PRECEDING
        ) as balance_after
    FROM transaction_amounts
)
SELECT account_id, transaction_date,
    ABS(amount) as amount,
    COALESCE(
        LAG(balance_after) OVER (PARTITION BY account_id ORDER BY transaction_date),
        0
    ) as balance_before,
    balance_after
FROM running_totals
ORDER BY account_id, transaction_date;""",
        "explanation": "Combines sign conversion, window functions with ROWS frame, and LAG function.",
        "key_insight": "Combines three advanced SQL concepts in an elegant way. ROWS UNBOUNDED PRECEDING creates true running totals.",
        "is_custom": False
    }
]

# Challenge persistence file
CUSTOM_CHALLENGES_FILE = "custom_challenges.json"

# Initialize session state
if "openai_key" not in st.session_state:
    st.session_state.openai_key = None

if "selected_challenge" not in st.session_state:
    st.session_state.selected_challenge = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "custom_challenges" not in st.session_state:
    # Try to load from file
    if os.path.exists(CUSTOM_CHALLENGES_FILE):
        try:
            with open(CUSTOM_CHALLENGES_FILE, 'r') as f:
                st.session_state.custom_challenges = json.load(f)
        except:
            st.session_state.custom_challenges = []
    else:
        st.session_state.custom_challenges = []

if "show_add_challenge_form" not in st.session_state:
    st.session_state.show_add_challenge_form = False

def save_custom_challenges():
    """Save custom challenges to JSON file"""
    try:
        with open(CUSTOM_CHALLENGES_FILE, 'w') as f:
            json.dump(st.session_state.custom_challenges, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving challenges: {str(e)}")
        return False

def get_all_challenges():
    """Combine built-in and custom challenges"""
    return BUILTIN_CHALLENGES + st.session_state.custom_challenges

def delete_custom_challenge(challenge_id):
    """Delete a custom challenge"""
    st.session_state.custom_challenges = [
        c for c in st.session_state.custom_challenges 
        if c["id"] != challenge_id
    ]
    save_custom_challenges()
    st.success("Challenge deleted!")

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# API Key input
with st.sidebar.expander("OpenAI API Key", expanded=not st.session_state.openai_key):
    api_key_input = st.text_input(
        "Enter your OpenAI API key:",
        type="password",
        value=st.session_state.openai_key or "",
        help="Your API key is only used for this session and not stored."
    )
    if api_key_input:
        st.session_state.openai_key = api_key_input
        st.success("✅ API key configured!")

# Main content
st.title("💾 SQL Interview Challenges")
st.markdown("Master SQL challenges - built-in & custom")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📚 Challenges", "🤖 Ask AI", "📖 Patterns", "➕ Manage Challenges"])

with tab1:
    st.header("Select a Challenge")
    
    all_challenges = get_all_challenges()
    
    # Separate built-in and custom
    builtin = [c for c in all_challenges if not c.get("is_custom", False)]
    custom = [c for c in all_challenges if c.get("is_custom", False)]
    
    # Challenge selector
    challenge_groups = {}
    
    if builtin:
        for c in builtin:
            key = f"[Built-in] {c['id']}. {c['title']} ({c['difficulty']})"
            challenge_groups[key] = c
    
    if custom:
        for c in custom:
            key = f"[Custom] {c['id']}. {c['title']} ({c['difficulty']})"
            challenge_groups[key] = c
    
    if not challenge_groups:
        st.warning("No challenges available. Add one in the 'Manage Challenges' tab.")
    else:
        selected_text = st.selectbox("Choose a challenge:", list(challenge_groups.keys()))
        challenge = challenge_groups[selected_text]
        
        # Display challenge
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric("Difficulty", challenge["difficulty"])
        
        with col2:
            if challenge["difficulty"] == "Easy":
                st.metric("Challenge Level", "🟢")
            elif challenge["difficulty"] == "Medium":
                st.metric("Challenge Level", "🟡")
            else:
                st.metric("Challenge Level", "🔴")
        
        with col3:
            if challenge.get("is_custom"):
                st.metric("Type", "Custom ✏️")
            else:
                st.metric("Type", "Built-in")
        
        st.markdown("---")
        
        # Question
        st.subheader("❓ Challenge Question")
        st.write(challenge["question"])
        
        # Setup
        with st.expander("📋 Table Setup", expanded=True):
            st.code(challenge["setup"], language="sql")
        
        # Solution
        with st.expander("✅ Solution"):
            st.code(challenge["solution"], language="sql")
            st.markdown(f"**How it works:** {challenge['explanation']}")
            st.markdown(f"**Key Insight:** {challenge['key_insight']}")
        
        # Try to solve section
        st.markdown("---")
        st.subheader("💭 Try to Solve")
        user_solution = st.text_area(
            "Write your SQL solution here:",
            height=150,
            placeholder="SELECT ..."
        )
        
        if user_solution:
            if st.button("Check against AI"):
                if not st.session_state.openai_key:
                    st.error("❌ Please configure your OpenAI API key first!")
                else:
                    with st.spinner("Analyzing your solution..."):
                        try:
                            client = openai.OpenAI(api_key=st.session_state.openai_key)
                            
                            response = client.chat.completions.create(
                                model="gpt-4-turbo",
                                messages=[
                                    {
                                        "role": "user",
                                        "content": f"""Compare this SQL solution to the correct one and provide feedback.

Challenge: {challenge['question']}

Correct Solution:
{challenge['solution']}

User's Solution:
{user_solution}

Please provide:
1. Does the logic match? (Yes/No)
2. Will it produce the correct results?
3. What are the strengths?
4. What could be improved?
5. Performance considerations if any."""
                                    }
                                ],
                                temperature=0.7
                            )
                            
                            st.info(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

with tab2:
    st.header("🤖 Ask AI about SQL Challenges")
    
    if not st.session_state.openai_key:
        st.warning("⚠️ Please configure your OpenAI API key in the sidebar first!")
    else:
        st.markdown("Ask questions about any of the SQL challenges or SQL concepts in general.")
        
        all_challenges = get_all_challenges()
        challenge_labels = ["General SQL Questions"] + [f"{c['id']}. {c['title']}" for c in all_challenges]
        
        # Challenge context selector
        context_challenge = st.selectbox(
            "Ask about a specific challenge (optional):",
            challenge_labels
        )
        
        # Chat interface
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        user_question = st.chat_input("Ask your question...")
        
        if user_question:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = openai.OpenAI(api_key=st.session_state.openai_key)
                        
                        # Build context
                        context = ""
                        if context_challenge != "General SQL Questions":
                            challenge_id = int(context_challenge.split(".")[0])
                            challenge_data = next((c for c in all_challenges if c["id"] == challenge_id), None)
                            if challenge_data:
                                context = f"""
Context - Challenge #{challenge_data['id']}: {challenge_data['title']}
Question: {challenge_data['question']}
Setup: {challenge_data['setup']}
Solution: {challenge_data['solution']}
Explanation: {challenge_data['explanation']}
"""
                        
                        response = client.chat.completions.create(
                            model="gpt-4-turbo",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"""You are an expert SQL tutor helping someone master advanced SQL concepts. 
Be clear, educational, and provide examples when helpful.{context}"""
                                }
                            ] + [
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in st.session_state.chat_history
                            ],
                            temperature=0.7
                        )
                        
                        assistant_message = response.choices[0].message.content
                        st.markdown(assistant_message)
                        
                        # Add to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": assistant_message
                        })
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

with tab3:
    st.header("📖 Advanced SQL Patterns")
    
    patterns = [
        {
            "name": "Mathematical Grouping",
            "description": "Use arithmetic operations (subtraction, modulo) to create group identifiers for complex partitioning.",
            "used_in": ["Islands and Gaps Challenge"],
            "example": "ticket_id - ROW_NUMBER() creates groups for consecutive sequences"
        },
        {
            "name": "Multiple Window Function Partitions",
            "description": "Use different PARTITION BY clauses in the same query for multi-level aggregations.",
            "used_in": ["Percentage of Total Paradox"],
            "example": "SUM(amount) OVER (PARTITION BY region) vs SUM(amount) OVER ()"
        },
        {
            "name": "Self-Joins with Inequality",
            "description": "Join a table to itself using < or > conditions instead of = to find relationships.",
            "used_in": ["Second-Highest Salary Trap"],
            "example": "ON e1.salary < e2.salary to exclude highest values"
        },
        {
            "name": "Frame-Aware Window Functions",
            "description": "Master ROWS BETWEEN and RANGE BETWEEN to control which rows participate in calculations.",
            "used_in": ["Running Balance Mind-Bender"],
            "example": "ROWS UNBOUNDED PRECEDING for running totals"
        },
        {
            "name": "LAG/LEAD for State Transitions",
            "description": "Use previous/next row values to calculate differences, detect changes, or maintain state.",
            "used_in": ["Running Balance Mind-Bender"],
            "example": "LAG(balance) to show balance before each transaction"
        }
    ]
    
    for i, pattern in enumerate(patterns, 1):
        with st.expander(f"{i}. {pattern['name']}", expanded=(i==1)):
            st.markdown(f"**Description:** {pattern['description']}")
            st.markdown(f"**Used in:** {', '.join(pattern['used_in'])}")
            st.markdown(f"**Example:** {pattern['example']}")

with tab4:
    st.header("➕ Manage Challenges")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Add New Challenge")
        
        with st.form("add_challenge_form", clear_on_submit=True):
            title = st.text_input("Challenge Title", placeholder="e.g., Finding Duplicate Records")
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
            question = st.text_area("Challenge Question", height=100, placeholder="What should the user solve?")
            setup = st.text_area("Table Setup (SQL)", height=150, placeholder="CREATE TABLE...")
            solution = st.text_area("Solution (SQL)", height=150, placeholder="SELECT...")
            explanation = st.text_area("Explanation", height=100, placeholder="How does the solution work?")
            key_insight = st.text_area("Key Insight", height=80, placeholder="What's the main takeaway?")
            
            submitted = st.form_submit_button("➕ Add Challenge")
            
            if submitted:
                if not all([title, question, setup, solution, explanation, key_insight]):
                    st.error("Please fill in all fields!")
                else:
                    # Generate new ID
                    new_id = max([c["id"] for c in get_all_challenges()]) + 1
                    
                    new_challenge = {
                        "id": new_id,
                        "title": title,
                        "difficulty": difficulty,
                        "question": question,
                        "setup": setup,
                        "solution": solution,
                        "explanation": explanation,
                        "key_insight": key_insight,
                        "is_custom": True
                    }
                    
                    st.session_state.custom_challenges.append(new_challenge)
                    if save_custom_challenges():
                        st.success(f"✅ Challenge '{title}' added successfully!")
                        st.rerun()
    
    with col2:
        st.subheader("🗑️ Manage Custom Challenges")
        
        if not st.session_state.custom_challenges:
            st.info("No custom challenges yet. Create one on the left!")
        else:
            for challenge in st.session_state.custom_challenges:
                with st.expander(f"🔹 {challenge['title']} ({challenge['difficulty']})"):
                    st.write(f"**Question:** {challenge['question'][:100]}...")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"✏️ Edit", key=f"edit_{challenge['id']}"):
                            st.info("Edit feature coming soon! For now, delete and recreate.")
                    
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_{challenge['id']}"):
                            delete_custom_challenge(challenge['id'])
                            st.rerun()
    
    st.markdown("---")
    
    # Import/Export section
    st.subheader("📤 Import / Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export custom challenges to JSON:**")
        if st.session_state.custom_challenges:
            export_data = json.dumps(st.session_state.custom_challenges, indent=2)
            st.download_button(
                label="📥 Download Custom Challenges",
                data=export_data,
                file_name=f"sql_challenges_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("No custom challenges to export")
    
    with col2:
        st.write("**Import challenges from JSON:**")
        uploaded_file = st.file_uploader("Upload JSON file with challenges", type="json")
        
        if uploaded_file is not None:
            try:
                imported_challenges = json.load(uploaded_file)
                
                # Validate structure
                for challenge in imported_challenges:
                    required_fields = ["title", "difficulty", "question", "setup", "solution", "explanation", "key_insight"]
                    if not all(field in challenge for field in required_fields):
                        raise ValueError("Invalid challenge structure")
                
                # Get max ID
                max_id = max([c["id"] for c in get_all_challenges()]) + 1
                
                # Add imported challenges
                for idx, challenge in enumerate(imported_challenges):
                    challenge["id"] = max_id + idx
                    challenge["is_custom"] = True
                    st.session_state.custom_challenges.append(challenge)
                
                if save_custom_challenges():
                    st.success(f"✅ Imported {len(imported_challenges)} challenge(s)!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing challenges: {str(e)}")
    
    st.markdown("---")
    st.subheader("📊 Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Built-in Challenges", len(BUILTIN_CHALLENGES))
    with col2:
        st.metric("Custom Challenges", len(st.session_state.custom_challenges))
    with col3:
        st.metric("Total Challenges", len(get_all_challenges()))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>SQL Interview Challenges App | Based on "5 SQL Questions That Stump Even Senior Analysts"</p>
    <p style='font-size: 0.8em; color: gray;'>Built-in + Custom Challenges | API keys are not stored. All queries use your provided OpenAI key for this session only.</p>
</div>
""", unsafe_allow_html=True)
