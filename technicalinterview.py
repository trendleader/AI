import streamlit as st
import sqlite3
import pandas as pd
import time
from datetime import datetime
import json
import os
from pathlib import Path

# Try to import OpenAI, handle if not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("OpenAI package not installed. AI features will be disabled.")

# Page configuration
st.set_page_config(
    page_title="CodeMaster - Interview Platform",
    page_icon="Lightning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client
client = None
if OPENAI_AVAILABLE:
    try:
        openai_api_key = None
        
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            openai_api_key = st.secrets['OPENAI_API_KEY']
        elif 'OPENAI_API_KEY' in os.environ:
            openai_api_key = os.environ['OPENAI_API_KEY']
        
        if openai_api_key:
            client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        st.warning(f"OpenAI initialization error: {str(e)}")

# Initialize session state
if 'user_solutions' not in st.session_state:
    st.session_state.user_solutions = {}
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'ai_feedback' not in st.session_state:
    st.session_state.ai_feedback = {}
if 'hints_used' not in st.session_state:
    st.session_state.hints_used = {}

# Problem database
PROBLEMS = {
    "algorithms": [
        {
            "id": "two_sum",
            "title": "Two Sum",
            "difficulty": "Easy",
            "category": "Array/Hash Table",
            "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution.",
            "template": "def two_sum(nums, target):\n    # Your code here\n    pass\n\nprint(two_sum([2, 7, 11, 15], 9))",
            "solution": "def two_sum(nums, target):\n    num_map = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in num_map:\n            return [num_map[complement], i]\n        num_map[num] = i\n    return []",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)"
        },
        {
            "id": "palindrome",
            "title": "Valid Palindrome",
            "difficulty": "Easy",
            "category": "String/Two Pointers",
            "description": "A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward.",
            "template": "def is_palindrome(s):\n    # Your code here\n    pass\n\nprint(is_palindrome('A man, a plan, a canal: Panama'))",
            "solution": "def is_palindrome(s):\n    cleaned = ''.join(char.lower() for char in s if char.isalnum())\n    return cleaned == cleaned[::-1]",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)"
        }
    ],
    "sql": [
        {
            "id": "employee_salary",
            "title": "Second Highest Salary",
            "difficulty": "Medium",
            "category": "SQL Query",
            "description": "Write a solution to find the second highest salary from the Employee table.",
            "template": "-- Write your SQL query statement below\nSELECT\n    -- Your solution here",
            "solution": "SELECT MAX(salary) AS SecondHighestSalary\nFROM Employee\nWHERE salary < (SELECT MAX(salary) FROM Employee);"
        }
    ],
    "dax": [
        {
            "id": "total_sales",
            "title": "Calculate Total Sales",
            "difficulty": "Easy",
            "category": "DAX Measures",
            "description": "Create a DAX measure to calculate the total sales amount from a Sales table.",
            "template": "-- Create a DAX measure for Total Sales\nTotal Sales =\n-- Your DAX formula here",
            "solution": "Total Sales = SUM(Sales[SalesAmount])",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)"
        }
    ],
    "data_structures": [
        {
            "id": "valid_parentheses",
            "title": "Valid Parentheses",
            "difficulty": "Easy",
            "category": "Stack",
            "description": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
            "template": "def is_valid(s):\n    # Your code here\n    pass",
            "solution": "def is_valid(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping:\n            if not stack or stack.pop() != mapping[char]:\n                return False\n        else:\n            stack.append(char)\n    return not stack"
        }
    ],
    "system_design": [
        {
            "id": "url_shortener",
            "title": "Design URL Shortener",
            "difficulty": "Medium",
            "category": "System Design",
            "description": "Design a URL shortening service like bit.ly or TinyURL.",
            "template": "class URLShortener:\n    def __init__(self):\n        pass\n    \n    def encode(self, long_url):\n        pass\n    \n    def decode(self, short_url):\n        pass",
            "solution": "class URLShortener:\n    def __init__(self):\n        self.url_to_short = {}\n        self.short_to_url = {}\n        self.counter = 1000000\n    \n    def encode(self, long_url):\n        if long_url in self.url_to_short:\n            return self.url_to_short[long_url]\n        code = str(self.counter)\n        self.counter += 1\n        short_url = 'http://short.ly/' + code\n        self.url_to_short[long_url] = short_url\n        self.short_to_url[short_url] = long_url\n        return short_url\n    \n    def decode(self, short_url):\n        return self.short_to_url.get(short_url, 'Not found')"
        }
    ]
}

def get_ai_feedback(code, problem_description, problem_category):
    if not client:
        return "OpenAI client not available."
    
    try:
        prompt = f"As a technical interviewer, review this code:\n\nProblem: {problem_description}\nCategory: {problem_category}\n\nCode:\n{code}\n\nProvide: correctness, complexity, quality, edge cases, improvements."
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a technical interviewer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_hint(problem_description, problem_category, hint_level=1):
    if not client:
        return "OpenAI client not available."
    
    try:
        hints = {
            1: "Provide a gentle hint about the approach",
            2: "Provide details about the algorithm or data structure",
            3: "Provide step-by-step solution approach"
        }
        
        prompt = f"Problem: {problem_description}\nCategory: {problem_category}\n\n{hints.get(hint_level)}"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a coding mentor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def create_database():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    cursor.execute('CREATE TABLE Employee (Id INTEGER PRIMARY KEY, Salary INTEGER)')
    cursor.execute('CREATE TABLE Person (Id INTEGER PRIMARY KEY, Email TEXT)')
    
    cursor.executemany('INSERT INTO Employee (Id, Salary) VALUES (?, ?)', 
                       [(1, 100), (2, 200), (3, 300), (4, 200)])
    cursor.executemany('INSERT INTO Person (Id, Email) VALUES (?, ?)', 
                       [(1, 'a@b.com'), (2, 'c@d.com'), (3, 'a@b.com')])
    
    conn.commit()
    return conn

def main():
    st.title("CodeMaster - Interview Platform")
    st.markdown("Master Your Coding Interview Skills")
    
    if not client:
        st.info("AI features disabled. Add OpenAI API key to enable.")
    
    with st.sidebar:
        st.subheader("Problem Navigator")
        
        category = st.selectbox(
            "Category",
            ["algorithms", "sql", "dax", "data_structures", "system_design"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        problems = PROBLEMS[category]
        problem_titles = [f"{p['title']} ({p['difficulty']})" for p in problems]
        
        selected_idx = st.selectbox(
            "Select Problem",
            range(len(problems)),
            format_func=lambda i: problem_titles[i]
        )
        
        current_problem = problems[selected_idx]
        problem_id = current_problem['id']
        
        st.markdown("---")
        st.subheader("Session Timer")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start"):
                st.session_state.start_time = time.time()
        with col2:
            if st.button("Reset"):
                st.session_state.start_time = None
        
        if st.session_state.start_time:
            elapsed = int(time.time() - st.session_state.start_time)
            st.metric("Time", f"{elapsed//60:02d}:{elapsed%60:02d}")
        
        if client:
            st.markdown("---")
            st.subheader("AI Assistant")
            
            if st.button("Get Hint 1"):
                if problem_id not in st.session_state.hints_used:
                    st.session_state.hints_used[problem_id] = 0
                st.session_state.hints_used[problem_id] += 1
                hint = get_hint(current_problem['description'], current_problem['category'], 1)
                st.info(hint)
            
            if st.button("Get Hint 2"):
                hint = get_hint(current_problem['description'], current_problem['category'], 2)
                st.info(hint)
        
        st.markdown("---")
        st.subheader("Your Progress")
        
        total_problems = sum(len(p) for p in PROBLEMS.values())
        solved_problems = len(st.session_state.user_solutions)
        
        st.progress(solved_problems / total_problems if total_problems > 0 else 0)
        st.text(f"Solved: {solved_problems}/{total_problems}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(current_problem['title'])
        st.badge(current_problem['difficulty'])
        st.badge(current_problem['category'])
        
        st.markdown("**Problem Description**")
        st.write(current_problem['description'])
        
        if 'time_complexity' in current_problem:
            st.info(f"Time: {current_problem['time_complexity']} | Space: {current_problem['space_complexity']}")
        
        if st.button("Show Solution"):
            st.code(current_problem['solution'], language='python' if category not in ['sql', 'dax'] else category)
    
    with col2:
        st.subheader("Code Editor")
        
        if category == 'sql':
            language = 'sql'
        elif category == 'dax':
            language = 'dax'
        else:
            language = st.selectbox("Language", ["python", "java", "javascript", "cpp"])
        
        user_code = st.text_area(
            "Write your solution:",
            value=current_problem['template'],
            height=400,
            key=f"code_{problem_id}"
        )
        
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            if st.button("Save"):
                st.session_state.user_solutions[problem_id] = {
                    'code': user_code,
                    'language': language,
                    'timestamp': datetime.now().isoformat(),
                    'hints_used': st.session_state.hints_used.get(problem_id, 0)
                }
                st.success("Solution saved!")
        
        with col2b:
            if client and st.button("AI Review"):
                if user_code.strip() and user_code != current_problem['template']:
                    feedback = get_ai_feedback(user_code, current_problem['description'], current_problem['category'])
                    st.session_state.ai_feedback[problem_id] = feedback
                else:
                    st.warning("Please write some code first!")
        
        with col2c:
            if category == 'sql' and st.button("Run Query"):
                try:
                    conn = create_database()
                    df = pd.read_sql_query(user_code, conn)
                    st.dataframe(df)
                    conn.close()
                    st.success("Query executed!")
                except Exception as e:
                    st.error(f"SQL Error: {str(e)}")
            elif category == 'dax' and st.button("Validate DAX"):
                st.info("Test in Power BI Desktop. Common functions: SUM, CALCULATE, FILTER, ALL")
            else:
                if st.button("Test Code"):
                    st.info("Copy code to local environment to test.")
        
        if problem_id in st.session_state.ai_feedback:
            st.markdown("**AI Code Review**")
            st.write(st.session_state.ai_feedback[problem_id])
        
        if problem_id in st.session_state.user_solutions:
            solution = st.session_state.user_solutions[problem_id]
            st.success(f"Last saved: {solution['timestamp'][:16]} | Hints: {solution.get('hints_used', 0)}")

if __name__ == "__main__":
    main()