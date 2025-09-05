import streamlit as st
import sqlite3
import pandas as pd
import time
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# Load environment variables from the specified path
env_path = Path(r"C:\Users\pjone\OneDrive\Desktop\NewPython\test.env")
load_dotenv(dotenv_path=env_path)

# Page configuration
st.set_page_config(
    page_title="AI-Powered Coding Interview Platform",
    page_icon="🤖",
    layout="wide"
)

# Initialize OpenAI client
try:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
    else:
        client = None
        st.error("OpenAI API key not found in test.env file. Please add OPENAI_API_KEY=your_key_here to the file.")
except Exception as e:
    client = None
    st.error(f"Error loading OpenAI API key: {str(e)}")

# Initialize session state
if 'current_problem' not in st.session_state:
    st.session_state.current_problem = None
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
            "description": """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].""",
            "template": """def two_sum(nums, target):
    # Your code here
    pass

# Test cases
print(two_sum([2, 7, 11, 15], 9))  # Expected: [0, 1]
print(two_sum([3, 2, 4], 6))       # Expected: [1, 2]""",
            "solution": """def two_sum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []""",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)"
        },
        {
            "id": "palindrome",
            "title": "Valid Palindrome",
            "difficulty": "Easy",
            "category": "String/Two Pointers",
            "description": """A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward.

Given a string s, return true if it is a palindrome, or false otherwise.

Example:
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.""",
            "template": """def is_palindrome(s):
    # Your code here
    pass

# Test cases
print(is_palindrome("A man, a plan, a canal: Panama"))  # Expected: True
print(is_palindrome("race a car"))                      # Expected: False""",
            "solution": """def is_palindrome(s):
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]""",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)"
        },
        {
            "id": "binary_search",
            "title": "Binary Search",
            "difficulty": "Easy",
            "category": "Binary Search",
            "description": """Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

Example:
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4""",
            "template": """def search(nums, target):
    # Your code here
    pass

# Test cases
print(search([-1,0,3,5,9,12], 9))   # Expected: 4
print(search([-1,0,3,5,9,12], 2))   # Expected: -1""",
            "solution": """def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
            "time_complexity": "O(log n)",
            "space_complexity": "O(1)"
        }
    ],
    "sql": [
        {
            "id": "employee_salary",
            "title": "Find Second Highest Salary",
            "difficulty": "Medium",
            "category": "SQL Query",
            "description": """Given an Employee table with columns Id and Salary, write a SQL query to get the second highest salary from the Employee table.

Employee table:
| Id | Salary |
|----|--------|
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |

Expected output: 200""",
            "template": """-- Write your SQL query here
SELECT 
-- Your solution here""",
            "solution": """SELECT MAX(Salary) as SecondHighestSalary
FROM Employee 
WHERE Salary < (SELECT MAX(Salary) FROM Employee);

-- Alternative solution:
SELECT DISTINCT Salary as SecondHighestSalary
FROM Employee 
ORDER BY Salary DESC 
LIMIT 1 OFFSET 1;"""
        },
        {
            "id": "duplicate_emails",
            "title": "Duplicate Emails",
            "difficulty": "Easy",
            "category": "SQL Query",
            "description": """Write a SQL query to find all duplicate emails in a table named Person.

Person table:
| Id | Email   |
|----|---------|
| 1  | a@b.com |
| 2  | c@d.com |
| 3  | a@b.com |

Expected output:
| Email   |
|---------|
| a@b.com |""",
            "template": """-- Write your SQL query here
SELECT 
-- Your solution here""",
            "solution": """SELECT Email
FROM Person
GROUP BY Email
HAVING COUNT(Email) > 1;"""
        }
    ],
    "data_structures": [
        {
            "id": "linked_list_cycle",
            "title": "Linked List Cycle",
            "difficulty": "Easy",
            "category": "Linked List",
            "description": """Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer.

Return true if there is a cycle in the linked list. Otherwise, return false.""",
            "template": """# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def has_cycle(head):
    # Your code here
    pass

# Test case setup would require creating linked list nodes""",
            "solution": """def has_cycle(head):
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next
    
    return True""",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)"
        },
        {
            "id": "valid_parentheses",
            "title": "Valid Parentheses",
            "difficulty": "Easy",
            "category": "Stack",
            "description": """Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.

Example:
Input: s = "()[]{}"
Output: true""",
            "template": """def is_valid(s):
    # Your code here
    pass

# Test cases
print(is_valid("()"))       # Expected: True
print(is_valid("()[]{}")    # Expected: True
print(is_valid("(]"))       # Expected: False""",
            "solution": """def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack""",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)"
        }
    ],
    "system_design": [
        {
            "id": "url_shortener",
            "title": "Design URL Shortener",
            "difficulty": "Medium",
            "category": "System Design",
            "description": """Design a URL shortening service like bit.ly or TinyURL.

Requirements:
1. Shorten long URLs to short URLs
2. Redirect short URLs to original URLs
3. Handle millions of URLs
4. URLs should be as short as possible
5. System should be highly available

Provide a high-level design and implement basic functionality.""",
            "template": """class URLShortener:
    def __init__(self):
        # Your initialization code here
        pass
    
    def encode(self, long_url):
        # Encode long URL to short URL
        pass
    
    def decode(self, short_url):
        # Decode short URL to long URL
        pass

# Test your implementation
shortener = URLShortener()
short_url = shortener.encode("https://www.google.com/very/long/url")
print(f"Short URL: {short_url}")
print(f"Decoded: {shortener.decode(short_url)}")""",
            "solution": """import hashlib
import base64

class URLShortener:
    def __init__(self):
        self.url_to_short = {}
        self.short_to_url = {}
        self.base_url = "http://short.ly/"
        self.counter = 1000000  # Start from a large number
    
    def encode(self, long_url):
        if long_url in self.url_to_short:
            return self.url_to_short[long_url]
        
        # Generate short code using counter
        short_code = self._encode_base62(self.counter)
        self.counter += 1
        
        short_url = self.base_url + short_code
        
        self.url_to_short[long_url] = short_url
        self.short_to_url[short_url] = long_url
        
        return short_url
    
    def decode(self, short_url):
        return self.short_to_url.get(short_url, "URL not found")
    
    def _encode_base62(self, num):
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = ""
        while num > 0:
            result = chars[num % 62] + result
            num //= 62
        return result or "0" """
        }
    ]
}

def get_ai_feedback(code, problem_description, problem_category):
    """Get AI feedback on user's code solution"""
    if not client:
        return "OpenAI client not available. Please check your API key."
    
    try:
        prompt = f"""
        As a technical interviewer, please review this coding solution:
        
        Problem: {problem_description}
        Category: {problem_category}
        
        User's Code:
        {code}
        
        Please provide:
        1. Code correctness assessment
        2. Time and space complexity analysis
        3. Code quality and style feedback
        4. Potential edge cases to consider
        5. Suggestions for improvement
        
        Keep the feedback constructive and interview-focused.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an experienced technical interviewer providing code review feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI feedback: {str(e)}"

def get_hint(problem_description, problem_category, hint_level=1):
    """Get AI-generated hint for the problem"""
    if not client:
        return "OpenAI client not available. Please check your API key."
    
    try:
        hint_prompts = {
            1: "Provide a gentle hint about the approach without giving away the solution",
            2: "Provide a more detailed hint about the algorithm or data structure to use",
            3: "Provide a step-by-step breakdown of the solution approach"
        }
        
        prompt = f"""
        Problem: {problem_description}
        Category: {problem_category}
        
        {hint_prompts.get(hint_level, hint_prompts[1])}
        
        Keep the hint concise and educational, as if you're guiding a student during an interview.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful coding mentor providing hints for interview problems."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting hint: {str(e)}"

def generate_similar_problem(problem_description, difficulty):
    """Generate a similar problem using AI"""
    if not client:
        return "OpenAI client not available. Please check your API key."
    
    try:
        prompt = f"""
        Based on this coding problem:
        {problem_description}
        
        Generate a similar problem with {difficulty} difficulty that tests the same concepts but with a different scenario.
        
        Format your response as:
        Title: [Problem Title]
        Description: [Problem Description]
        Example: [Input/Output Example]
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a technical interview question generator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating similar problem: {str(e)}"

def create_database():
    """Create SQLite database for storing user solutions"""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create tables for SQL practice
    cursor.execute('''
        CREATE TABLE Employee (
            Id INTEGER PRIMARY KEY,
            Salary INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE Person (
            Id INTEGER PRIMARY KEY,
            Email TEXT
        )
    ''')
    
    # Insert sample data
    cursor.executemany('INSERT INTO Employee (Id, Salary) VALUES (?, ?)', 
                       [(1, 100), (2, 200), (3, 300), (4, 200)])
    
    cursor.executemany('INSERT INTO Person (Id, Email) VALUES (?, ?)', 
                       [(1, 'a@b.com'), (2, 'c@d.com'), (3, 'a@b.com')])
    
    conn.commit()
    return conn

def main():
    st.title("🤖 AI-Powered Coding Interview Platform")
    st.markdown("Practice coding problems with AI-powered hints and feedback")
    
    # Check if OpenAI is configured
    if not client:
        st.warning("⚠️ OpenAI API key not configured. AI features will be disabled.")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Problem category selection
    category = st.sidebar.selectbox(
        "Select Category",
        ["algorithms", "sql", "data_structures", "system_design"],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Problem selection within category
    problems = PROBLEMS[category]
    problem_titles = [f"{p['title']} ({p['difficulty']})" for p in problems]
    
    selected_problem_idx = st.sidebar.selectbox(
        "Select Problem",
        range(len(problems)),
        format_func=lambda i: problem_titles[i]
    )
    
    current_problem = problems[selected_problem_idx]
    problem_id = current_problem['id']
    
    # Timer functionality
    if st.sidebar.button("Start Timer"):
        st.session_state.start_time = time.time()
    
    if st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        st.sidebar.metric("Time Elapsed", f"{elapsed//60}:{elapsed%60:02d}")
    
    # AI Features section in sidebar
    if client:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 AI Assistant")
        
        # Hint system
        if st.sidebar.button("Get Hint (Level 1)"):
            if problem_id not in st.session_state.hints_used:
                st.session_state.hints_used[problem_id] = 0
            st.session_state.hints_used[problem_id] += 1
            hint = get_hint(current_problem['description'], current_problem['category'], 1)
            st.sidebar.info(f"💡 Hint: {hint}")
        
        if st.sidebar.button("Get Detailed Hint (Level 2)"):
            hint = get_hint(current_problem['description'], current_problem['category'], 2)
            st.sidebar.info(f"💡 Detailed Hint: {hint}")
        
        if st.sidebar.button("Get Solution Approach (Level 3)"):
            hint = get_hint(current_problem['description'], current_problem['category'], 3)
            st.sidebar.info(f"💡 Approach: {hint}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header(current_problem['title'])
        st.badge(current_problem['difficulty'])
        st.badge(current_problem['category'])
        
        st.subheader("Problem Description")
        st.markdown(current_problem['description'])
        
        # Show complexity information if available
        if 'time_complexity' in current_problem:
            st.info(f"**Expected Time Complexity:** {current_problem['time_complexity']}")
            st.info(f"**Expected Space Complexity:** {current_problem['space_complexity']}")
        
        # Show solution button
        if st.button("Show Solution"):
            st.subheader("Solution")
            st.code(current_problem['solution'], language='python' if category != 'sql' else 'sql')
        
        # AI-generated similar problem
        if client and st.button("Generate Similar Problem"):
            with st.spinner("Generating similar problem..."):
                similar_problem = generate_similar_problem(
                    current_problem['description'], 
                    current_problem['difficulty']
                )
                st.subheader("🎯 Similar Problem")
                st.markdown(similar_problem)
    
    with col2:
        st.subheader("Your Solution")
        
        # Code editor
        code_language = 'sql' if category == 'sql' else 'python'
        user_code = st.text_area(
            "Write your code here:",
            value=current_problem['template'],
            height=300,
            key=f"code_{problem_id}"
        )
        
        # Button row
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            if st.button("Save Solution"):
                st.session_state.user_solutions[problem_id] = {
                    'code': user_code,
                    'timestamp': datetime.now().isoformat(),
                    'hints_used': st.session_state.hints_used.get(problem_id, 0)
                }
                st.success("Solution saved!")
        
        with col2b:
            # AI Code Review
            if client and st.button("Get AI Review"):
                if user_code.strip() and user_code != current_problem['template']:
                    with st.spinner("Getting AI feedback..."):
                        feedback = get_ai_feedback(
                            user_code, 
                            current_problem['description'],
                            current_problem['category']
                        )
                        st.session_state.ai_feedback[problem_id] = feedback
                else:
                    st.warning("Please write some code first!")
        
        with col2c:
            # Test code button (for SQL problems)
            if category == 'sql' and st.button("Test Query"):
                try:
                    conn = create_database()
                    df = pd.read_sql_query(user_code, conn)
                    st.subheader("Query Result:")
                    st.dataframe(df)
                    conn.close()
                except Exception as e:
                    st.error(f"SQL Error: {str(e)}")
        
        # Show AI feedback if available
        if problem_id in st.session_state.ai_feedback:
            st.subheader("🤖 AI Code Review")
            st.markdown(st.session_state.ai_feedback[problem_id])
        
        # For non-SQL categories, show testing note
        if category != 'sql':
            st.info("💡 To test your code, copy it to your local Python environment or online IDE.")
    
    # Progress tracking
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Progress")
    
    total_problems = sum(len(problems) for problems in PROBLEMS.values())
    solved_problems = len(st.session_state.user_solutions)
    
    progress = solved_problems / total_problems if total_problems > 0 else 0
    st.sidebar.progress(progress)
    st.sidebar.text(f"Solved: {solved_problems}/{total_problems}")
    
    # Show performance stats
    if st.session_state.user_solutions:
        st.sidebar.subheader("🏆 Your Stats")
        
        total_hints = sum(st.session_state.hints_used.values())
        st.sidebar.metric("Total Hints Used", total_hints)
        
        # Show solved problems with hint usage
        for pid, solution in st.session_state.user_solutions.items():
            hints_used = solution.get('hints_used', 0)
            hint_indicator = f" ({hints_used} hints)" if hints_used > 0 else ""
            st.sidebar.text(f"✅ {pid}{hint_indicator}")

if __name__ == "__main__":
    main()