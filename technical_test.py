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
    page_title="CodeMaster - Interview Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for LeetCode/HackerRank style
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Custom font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Problem card styling */
    .problem-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
    }
    
    .problem-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .problem-meta {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    
    .difficulty-easy {
        background: #d4edda;
        color: #155724;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .difficulty-medium {
        background: #fff3cd;
        color: #856404;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .difficulty-hard {
        background: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .category-tag {
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Code editor styling */
    .code-container {
        background: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #3e3e3e;
    }
    
    /* Button styling */
    .custom-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.2rem;
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .hint-button {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #8b4513;
    }
    
    .test-button {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2c3e50;
    }
    
    .success-button {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        color: #2c3e50;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Progress bar custom styling */
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        padding: 0.2rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    
    /* Stats card */
    .stats-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Alert styling */
    .success-alert {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-alert {
        background: #e3f2fd;
        border: 1px solid #90caf9;
        color: #1565c0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-alert {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Timer styling */
    .timer-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    
    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom selectbox styling */
    .stSelectbox > label {
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Custom text area styling */
    .stTextArea > label {
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Complexity info styling */
    .complexity-info {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* AI Features styling */
    .ai-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .ai-section h3 {
        margin-top: 0;
        color: white;
    }
    
    /* Solution container */
    .solution-container {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
try:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
    else:
        client = None
except Exception as e:
    client = None

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

**Example 1:**
```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
```

**Example 2:**
```
Input: nums = [3,2,4], target = 6
Output: [1,2]
```

**Constraints:**
- 2 ≤ nums.length ≤ 10⁴
- -10⁹ ≤ nums[i] ≤ 10⁹
- -10⁹ ≤ target ≤ 10⁹
- Only one valid answer exists.""",
            "template": """def two_sum(nums, target):
    \"\"\"
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    \"\"\"
    # Your code here
    pass

# Test cases
if __name__ == "__main__":
    print(two_sum([2, 7, 11, 15], 9))  # Expected: [0, 1]
    print(two_sum([3, 2, 4], 6))       # Expected: [1, 2]
    print(two_sum([3, 3], 6))          # Expected: [0, 1]""",
            "solution": """def two_sum(nums, target):
    \"\"\"
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    \"\"\"
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []""",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "acceptance_rate": "52.1%",
            "total_submissions": "8.2M"
        },
        {
            "id": "palindrome",
            "title": "Valid Palindrome",
            "difficulty": "Easy",
            "category": "String/Two Pointers",
            "description": """A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward.

Given a string s, return true if it is a palindrome, or false otherwise.

**Example 1:**
```
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
```

**Example 2:**
```
Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.
```

**Constraints:**
- 1 ≤ s.length ≤ 2 * 10⁵
- s consists only of printable ASCII characters.""",
            "template": """def is_palindrome(s):
    \"\"\"
    :type s: str
    :rtype: bool
    \"\"\"
    # Your code here
    pass

# Test cases
if __name__ == "__main__":
    print(is_palindrome("A man, a plan, a canal: Panama"))  # Expected: True
    print(is_palindrome("race a car"))                      # Expected: False
    print(is_palindrome(" "))                               # Expected: True""",
            "solution": """def is_palindrome(s):
    \"\"\"
    :type s: str
    :rtype: bool
    \"\"\"
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]""",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "acceptance_rate": "45.8%",
            "total_submissions": "2.1M"
        },
        {
            "id": "binary_search",
            "title": "Binary Search",
            "difficulty": "Easy",
            "category": "Binary Search",
            "description": """Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

**Example 1:**
```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
```

**Example 2:**
```
Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
```

**Constraints:**
- 1 ≤ nums.length ≤ 10⁴
- -10⁴ < nums[i], target < 10⁴
- All the integers in nums are unique.
- nums is sorted in ascending order.""",
            "template": """def search(nums, target):
    \"\"\"
    :type nums: List[int]
    :type target: int
    :rtype: int
    \"\"\"
    # Your code here
    pass

# Test cases
if __name__ == "__main__":
    print(search([-1,0,3,5,9,12], 9))   # Expected: 4
    print(search([-1,0,3,5,9,12], 2))   # Expected: -1""",
            "solution": """def search(nums, target):
    \"\"\"
    :type nums: List[int]
    :type target: int
    :rtype: int
    \"\"\"
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
            "space_complexity": "O(1)",
            "acceptance_rate": "58.2%",
            "total_submissions": "1.8M"
        }
    ],
    "sql": [
        {
            "id": "employee_salary",
            "title": "Second Highest Salary",
            "difficulty": "Medium",
            "category": "SQL Query",
            "description": """Write a solution to find the second highest salary from the Employee table. If there is no second highest salary, return null (return None in Pandas).

**Employee Table:**
```
+----+--------+
| id | salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+
```

**Output:**
```
+---------------------+
| SecondHighestSalary |
+---------------------+
| 200                 |
+---------------------+
```""",
            "template": """-- Write your SQL query statement below
SELECT 
    -- Your solution here
    """,
            "solution": """-- Solution 1: Using subquery
SELECT MAX(salary) AS SecondHighestSalary
FROM Employee 
WHERE salary < (SELECT MAX(salary) FROM Employee);

-- Solution 2: Using LIMIT and OFFSET
SELECT DISTINCT salary AS SecondHighestSalary
FROM Employee 
ORDER BY salary DESC 
LIMIT 1 OFFSET 1;""",
            "acceptance_rate": "34.2%",
            "total_submissions": "892K"
        },
        {
            "id": "duplicate_emails",
            "title": "Duplicate Emails",
            "difficulty": "Easy",
            "category": "SQL Query",
            "description": """Write a solution to report all the duplicate emails. Note that it's guaranteed that the email field is not NULL.

**Person Table:**
```
+----+---------+
| id | email   |
+----+---------+
| 1  | a@b.com |
| 2  | c@d.com |
| 3  | a@b.com |
+----+---------+
```

**Output:**
```
+---------+
| Email   |
+---------+
| a@b.com |
+---------+
```""",
            "template": """-- Write your SQL query statement below
SELECT 
    -- Your solution here
    """,
            "solution": """SELECT email AS Email
FROM Person
GROUP BY email
HAVING COUNT(email) > 1;""",
            "acceptance_rate": "67.8%",
            "total_submissions": "756K"
        }
    ],
    "data_structures": [
        {
            "id": "linked_list_cycle",
            "title": "Linked List Cycle",
            "difficulty": "Easy",
            "category": "Linked List",
            "description": """Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

**Example 1:**
```
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
```

**Follow up:** Can you solve it using O(1) (i.e. constant) memory?""",
            "template": """# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def has_cycle(head):
    \"\"\"
    :type head: ListNode
    :rtype: bool
    \"\"\"
    # Your code here
    pass

# Test case setup would require creating linked list nodes""",
            "solution": """def has_cycle(head):
    \"\"\"
    :type head: ListNode
    :rtype: bool
    \"\"\"
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
            "space_complexity": "O(1)",
            "acceptance_rate": "48.6%",
            "total_submissions": "2.3M"
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
3. Every close bracket has a corresponding open bracket of the same type.

**Example 1:**
```
Input: s = "()"
Output: true
```

**Example 2:**
```
Input: s = "()[]{}"
Output: true
```

**Example 3:**
```
Input: s = "(]"
Output: false
```""",
            "template": """def is_valid(s):
    \"\"\"
    :type s: str
    :rtype: bool
    \"\"\"
    # Your code here
    pass

# Test cases
if __name__ == "__main__":
    print(is_valid("()"))         # Expected: True
    print(is_valid("()[]{}")      # Expected: True
    print(is_valid("(]"))         # Expected: False""",
            "solution": """def is_valid(s):
    \"\"\"
    :type s: str
    :rtype: bool
    \"\"\"
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
            "space_complexity": "O(n)",
            "acceptance_rate": "40.8%",
            "total_submissions": "3.1M"
        }
    ],
    "system_design": [
        {
            "id": "url_shortener",
            "title": "Design URL Shortener",
            "difficulty": "Medium",
            "category": "System Design",
            "description": """Design a URL shortening service like bit.ly or TinyURL.

**Requirements:**
1. Shorten long URLs to short URLs
2. Redirect short URLs to original URLs  
3. Handle millions of URLs
4. URLs should be as short as possible
5. System should be highly available

**Components to Design:**
- URL encoding/decoding algorithm
- Database schema
- API endpoints
- Caching strategy
- Load balancing

**Follow-up Questions:**
- How would you handle custom aliases?
- What about analytics and click tracking?
- How to handle peak traffic?

Provide a high-level design and implement basic functionality.""",
            "template": """class URLShortener:
    def __init__(self):
        # Your initialization code here
        # Consider: storage, counter, base URL
        pass
    
    def encode(self, long_url):
        \"\"\"
        Encode a long URL to a short URL
        :type long_url: str
        :rtype: str
        \"\"\"
        # Your encoding logic here
        pass
    
    def decode(self, short_url):
        \"\"\"
        Decode a short URL to the original long URL
        :type short_url: str
        :rtype: str
        \"\"\"
        # Your decoding logic here
        pass
    
    def _encode_base62(self, num):
        \"\"\"
        Helper method to encode number to base62
        \"\"\"
        # Your base62 encoding here
        pass

# Test your implementation
if __name__ == "__main__":
    shortener = URLShortener()
    
    # Test cases
    long_url1 = "https://www.google.com/search?q=very+long+url+example"
    short_url1 = shortener.encode(long_url1)
    print(f"Short URL: {short_url1}")
    print(f"Decoded: {shortener.decode(short_url1)}")
    
    # Test duplicate encoding
    short_url2 = shortener.encode(long_url1)
    print(f"Same URL again: {short_url2}")
    print(f"URLs match: {short_url1 == short_url2}")""",
            "solution": """import hashlib
import base64

class URLShortener:
    def __init__(self):
        # Storage for bidirectional mapping
        self.url_to_short = {}
        self.short_to_url = {}
        self.base_url = "http://short.ly/"
        self.counter = 1000000  # Start from large number for shorter codes
    
    def encode(self, long_url):
        \"\"\"
        Encode a long URL to a short URL
        Time: O(1), Space: O(1)
        \"\"\"
        # Return existing short URL if already encoded
        if long_url in self.url_to_short:
            return self.url_to_short[long_url]
        
        # Generate short code using counter approach
        short_code = self._encode_base62(self.counter)
        self.counter += 1
        
        # Create full short URL
        short_url = self.base_url + short_code
        
        # Store bidirectional mapping
        self.url_to_short[long_url] = short_url
        self.short_to_url[short_url] = long_url
        
        return short_url
    
    def decode(self, short_url):
        \"\"\"
        Decode a short URL to original long URL
        Time: O(1), Space: O(1)
        \"\"\"
        return self.short_to_url.get(short_url, "URL not found")
    
    def _encode_base62(self, num):
        \"\"\"
        Convert number to base62 string (0-9, a-z, A-Z)
        This gives us 62^n possible combinations
        \"\"\"
        if num == 0:
            return "0"
            
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = ""
        
        while num > 0:
            result = chars[num % 62] + result
            num //= 62
            
        return result

# Alternative implementation using hash-based approach
class URLShortenerHash:
    def __init__(self):
        self.url_to_short = {}
        self.short_to_url = {}
        self.base_url = "http://short.ly/"
    
    def encode(self, long_url):
        if long_url in self.url_to_short:
            return self.url_to_short[long_url]
        
        # Use MD5 hash and take first 6 characters
        hash_object = hashlib.md5(long_url.encode())
        short_code = hash_object.hexdigest()[:6]
        
        # Handle collisions by appending counter
        original_code = short_code
        counter = 1
        while short_code in self.short_to_url:
            short_code = original_code + str(counter)
            counter += 1
        
        short_url = self.base_url + short_code
        
        self.url_to_short[long_url] = short_url
        self.short_to_url[short_code] = long_url
        
        return short_url
    
    def decode(self, short_url):
        short_code = short_url.replace(self.base_url, "")
        return self.short_to_url.get(short_code, "URL not found")""",
            "time_complexity": "O(1)",
            "space_complexity": "O(n)",
            "acceptance_rate": "42.1%",
            "total_submissions": "2.8K"
        },
        {
            "id": "chat_system",
            "title": "Design Chat System",
            "difficulty": "Hard",
            "category": "System Design",
            "description": """Design a real-time chat system like WhatsApp or Slack.

**Requirements:**
1. Support 1-on-1 and group conversations
2. Real-time message delivery
3. Message history and persistence
4. Online/offline status
5. Support for multimedia messages
6. Handle millions of users

**Key Components:**
- WebSocket connections for real-time communication
- Message queue system
- Database design for messages and users
- Notification system
- Media storage

**Follow-up Questions:**
- How to handle message ordering?
- What about message encryption?
- How to scale to multiple data centers?
- Read receipts and typing indicators?

Design the high-level architecture and implement core message handling.""",
            "template": """class ChatSystem:
    def __init__(self):
        # Initialize your chat system components
        pass
    
    def send_message(self, sender_id, recipient_id, message, message_type="text"):
        \"\"\"
        Send a message from sender to recipient
        \"\"\"
        pass
    
    def get_conversation(self, user1_id, user2_id, limit=50):
        \"\"\"
        Get conversation history between two users
        \"\"\"
        pass
    
    def create_group(self, creator_id, member_ids, group_name):
        \"\"\"
        Create a group chat
        \"\"\"
        pass
    
    def send_group_message(self, sender_id, group_id, message):
        \"\"\"
        Send message to a group
        \"\"\"
        pass

# Test implementation
if __name__ == "__main__":
    chat = ChatSystem()
    
    # Test 1-on-1 messaging
    chat.send_message("user1", "user2", "Hello!")
    chat.send_message("user2", "user1", "Hi there!")
    
    # Get conversation
    messages = chat.get_conversation("user1", "user2")
    print("Conversation:", messages)""",
            "solution": """from datetime import datetime
import uuid
from collections import defaultdict, deque

class ChatSystem:
    def __init__(self):
        # Message storage: {conversation_id: [messages]}
        self.conversations = defaultdict(deque)
        
        # User conversations: {user_id: [conversation_ids]}
        self.user_conversations = defaultdict(set)
        
        # Group information: {group_id: {members, name, created_by}}
        self.groups = {}
        
        # Online users: {user_id: websocket_connection}
        self.online_users = {}
        
        # Message metadata
        self.messages = {}  # {message_id: message_data}
    
    def send_message(self, sender_id, recipient_id, message, message_type="text"):
        \"\"\"
        Send 1-on-1 message
        Time: O(1), Space: O(1)
        \"\"\"
        # Create conversation ID (consistent regardless of order)
        conv_id = self._get_conversation_id(sender_id, recipient_id)
        
        # Create message
        message_id = str(uuid.uuid4())
        message_data = {
            'id': message_id,
            'sender': sender_id,
            'recipient': recipient_id,
            'content': message,
            'type': message_type,
            'timestamp': datetime.now(),
            'conversation_id': conv_id
        }
        
        # Store message
        self.messages[message_id] = message_data
        self.conversations[conv_id].append(message_data)
        
        # Update user conversations
        self.user_conversations[sender_id].add(conv_id)
        self.user_conversations[recipient_id].add(conv_id)
        
        # Send real-time notification if recipient is online
        if recipient_id in self.online_users:
            self._notify_user(recipient_id, message_data)
        
        return message_id
    
    def get_conversation(self, user1_id, user2_id, limit=50):
        \"\"\"
        Get conversation history
        Time: O(limit), Space: O(limit)
        \"\"\"
        conv_id = self._get_conversation_id(user1_id, user2_id)
        messages = list(self.conversations[conv_id])
        
        # Return most recent messages
        return messages[-limit:] if len(messages) > limit else messages
    
    def create_group(self, creator_id, member_ids, group_name):
        \"\"\"
        Create group chat
        Time: O(n) where n is number of members
        \"\"\"
        group_id = str(uuid.uuid4())
        
        # Include creator in members
        all_members = set(member_ids)
        all_members.add(creator_id)
        
        self.groups[group_id] = {
            'name': group_name,
            'members': all_members,
            'created_by': creator_id,
            'created_at': datetime.now()
        }
        
        # Add group to each member's conversations
        for member_id in all_members:
            self.user_conversations[member_id].add(group_id)
        
        return group_id
    
    def send_group_message(self, sender_id, group_id, message):
        \"\"\"
        Send message to group
        Time: O(n) where n is group size
        \"\"\"
        if group_id not in self.groups:
            raise ValueError("Group not found")
        
        group = self.groups[group_id]
        if sender_id not in group['members']:
            raise ValueError("User not in group")
        
        # Create message
        message_id = str(uuid.uuid4())
        message_data = {
            'id': message_id,
            'sender': sender_id,
            'group_id': group_id,
            'content': message,
            'type': 'text',
            'timestamp': datetime.now(),
            'conversation_id': group_id
        }
        
        # Store message
        self.messages[message_id] = message_data
        self.conversations[group_id].append(message_data)
        
        # Notify all online group members except sender
        for member_id in group['members']:
            if member_id != sender_id and member_id in self.online_users:
                self._notify_user(member_id, message_data)
        
        return message_id
    
    def _get_conversation_id(self, user1_id, user2_id):
        \"\"\"Create consistent conversation ID\"\"\"
        return f"{'_'.join(sorted([user1_id, user2_id]))}"
    
    def _notify_user(self, user_id, message_data):
        \"\"\"Send real-time notification (WebSocket simulation)\"\"\"
        print(f"Notifying {user_id}: New message from {message_data['sender']}")
        # In real implementation, this would send via WebSocket
    
    def connect_user(self, user_id, websocket=None):
        \"\"\"User comes online\"\"\"
        self.online_users[user_id] = websocket
        print(f"User {user_id} connected")
    
    def disconnect_user(self, user_id):
        \"\"\"User goes offline\"\"\"
        if user_id in self.online_users:
            del self.online_users[user_id]
            print(f"User {user_id} disconnected")

# Database Schema Design (SQL)
SCHEMA = '''
-- Users table
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(50) UNIQUE,
    email VARCHAR(100),
    last_seen TIMESTAMP,
    created_at TIMESTAMP
);

-- Conversations table (for 1-on-1 chats)
CREATE TABLE conversations (
    id VARCHAR(36) PRIMARY KEY,
    user1_id VARCHAR(36),
    user2_id VARCHAR(36),
    created_at TIMESTAMP,
    FOREIGN KEY (user1_id) REFERENCES users(id),
    FOREIGN KEY (user2_id) REFERENCES users(id)
);

-- Groups table
CREATE TABLE groups (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100),
    created_by VARCHAR(36),
    created_at TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id)
);

-- Group members
CREATE TABLE group_members (
    group_id VARCHAR(36),
    user_id VARCHAR(36),
    joined_at TIMESTAMP,
    PRIMARY KEY (group_id, user_id),
    FOREIGN KEY (group_id) REFERENCES groups(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Messages table
CREATE TABLE messages (
    id VARCHAR(36) PRIMARY KEY,
    sender_id VARCHAR(36),
    conversation_id VARCHAR(36), -- can be conversation or group id
    content TEXT,
    message_type VARCHAR(20) DEFAULT 'text',
    timestamp TIMESTAMP,
    FOREIGN KEY (sender_id) REFERENCES users(id)
);

-- Indexes for performance
CREATE INDEX idx_messages_conversation ON messages(conversation_id, timestamp);
CREATE INDEX idx_messages_sender ON messages(sender_id);
CREATE INDEX idx_users_last_seen ON users(last_seen);
'''""",
            "time_complexity": "O(1) for send, O(n) for groups",
            "space_complexity": "O(m) where m is total messages",
            "acceptance_rate": "28.3%",
            "total_submissions": "1.4K"
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

def get_difficulty_class(difficulty):
    """Get CSS class for difficulty badge"""
    return f"difficulty-{difficulty.lower()}"

def display_problem_stats(problem):
    """Display problem statistics like LeetCode"""
    col1, col2 = st.columns(2)
    with col1:
        if 'acceptance_rate' in problem:
            st.metric("Acceptance Rate", problem['acceptance_rate'])
    with col2:
        if 'total_submissions' in problem:
            st.metric("Total Submissions", problem['total_submissions'])

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚡ CodeMaster</h1>
        <p>Master Your Coding Interview Skills</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if OpenAI is configured
    if not client:
        st.markdown("""
        <div class="warning-alert">
            ⚠️ <strong>AI Features Disabled:</strong> OpenAI API key not configured. Add your key to test.env file to enable AI-powered hints and feedback.
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Problem Navigator")
        
        # Problem category selection
        category = st.selectbox(
            "📂 Category",
            ["algorithms", "sql", "dax", "data_structures", "system_design"],
            format_func=lambda x: {
                "algorithms": "🧮 Algorithms",
                "sql": "🗃️ SQL Database",
                "dax": "📊 DAX Analytics", 
                "data_structures": "🏗️ Data Structures", 
                "system_design": "🏛️ System Design"
            }[x],
            index=0
        )
        
        # Problem selection within category
        problems = PROBLEMS[category]
        problem_options = [(i, f"{p['title']} ({p['difficulty']})") for i, p in enumerate(problems)]
        
        selected_problem_idx = st.selectbox(
            "📋 Select Problem",
            options=[opt[0] for opt in problem_options],
            format_func=lambda i: problem_options[i][1]
        )
        
        current_problem = problems[selected_problem_idx]
        problem_id = current_problem['id']
        
        st.markdown("---")
        
        # Timer section
        st.markdown("### ⏱️ Session Timer")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.start_time = time.time()
        with col2:
            if st.button("⏹️ Reset", use_container_width=True):
                st.session_state.start_time = None
        
        if st.session_state.start_time:
            elapsed = int(time.time() - st.session_state.start_time)
            st.markdown(f"""
            <div class="timer-display">
                🕐 {elapsed//60:02d}:{elapsed%60:02d}
            </div>
            """, unsafe_allow_html=True)
        
        # AI Features section
        if client:
            st.markdown("""
            <div class="ai-section">
                <h3>🤖 AI Assistant</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Hint system
            hint_col1, hint_col2 = st.columns(2)
            with hint_col1:
                if st.button("💡 Hint 1", use_container_width=True):
                    if problem_id not in st.session_state.hints_used:
                        st.session_state.hints_used[problem_id] = 0
                    st.session_state.hints_used[problem_id] += 1
                    hint = get_hint(current_problem['description'], current_problem['category'], 1)
                    st.info(f"💡 {hint}")
            
            with hint_col2:
                if st.button("💡 Hint 2", use_container_width=True):
                    hint = get_hint(current_problem['description'], current_problem['category'], 2)
                    st.info(f"💡 {hint}")
            
            if st.button("🎯 Solution Approach", use_container_width=True):
                hint = get_hint(current_problem['description'], current_problem['category'], 3)
                st.success(f"🎯 {hint}")
        
        st.markdown("---")
        
        # Progress tracking
        st.markdown("### 📊 Your Progress")
        
        total_problems = sum(len(problems) for problems in PROBLEMS.values())
        solved_problems = len(st.session_state.user_solutions)
        
        progress_percentage = int((solved_problems / total_problems) * 100) if total_problems > 0 else 0
        
        st.markdown(f"""
        <div class="stats-card">
            <h4>🏆 Statistics</h4>
            <p><strong>Solved:</strong> {solved_problems}/{total_problems} ({progress_percentage}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(solved_problems / total_problems if total_problems > 0 else 0)
        
        # Show performance stats
        if st.session_state.user_solutions:
            total_hints = sum(st.session_state.hints_used.values())
            st.markdown(f"""
            <div class="stats-card">
                <h4>📈 Performance</h4>
                <p><strong>Hints Used:</strong> {total_hints}</p>
                <p><strong>Avg Hints/Problem:</strong> {total_hints/solved_problems:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show solved problems
            st.markdown("#### ✅ Completed")
            for pid, solution in st.session_state.user_solutions.items():
                hints_used = solution.get('hints_used', 0)
                hint_indicator = f" ({hints_used}💡)" if hints_used > 0 else " (💪)"
                st.text(f"{pid}{hint_indicator}")
    
    # Main content area
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        # Problem header
        st.markdown(f"""
        <div class="problem-card">
            <div class="problem-title">{current_problem['title']}</div>
            <div class="problem-meta">
                <span class="{get_difficulty_class(current_problem['difficulty'])}">{current_problem['difficulty']}</span>
                <span class="category-tag">{current_problem['category']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Problem stats
        display_problem_stats(current_problem)
        
        # Problem description
        st.markdown("#### 📝 Problem Description")
        st.markdown(current_problem['description'])
        
        # Show complexity information if available
        if 'time_complexity' in current_problem:
            st.markdown(f"""
            <div class="complexity-info">
                <strong>⏰ Expected Time Complexity:</strong> {current_problem['time_complexity']}<br>
                <strong>💾 Expected Space Complexity:</strong> {current_problem['space_complexity']}
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            if st.button("👁️ Show Solution", use_container_width=True):
                st.markdown("""
                <div class="solution-container">
                    <h4>💡 Official Solution</h4>
                </div>
                """, unsafe_allow_html=True)
                st.code(current_problem['solution'], language='python' if category not in ['sql', 'dax'] else category)
        
        with button_col2:
            # AI-generated similar problem
            if client and st.button("🎲 Similar Problem", use_container_width=True):
                with st.spinner("🤖 Generating similar problem..."):
                    similar_problem = generate_similar_problem(
                        current_problem['description'], 
                        current_problem['difficulty']
                    )
                    st.markdown("""
                    <div class="info-alert">
                        <h4>🎯 AI-Generated Similar Problem</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(similar_problem)
        
        with button_col3:
            if st.button("📚 Discuss", use_container_width=True):
                st.info("💬 Discussion feature coming soon! Join our community to discuss solutions.")
    
    with col2:
        st.markdown("### 💻 Code Editor")
        
        # Language selector for multi-language support
        if category == 'sql':
            language = 'sql'
            st.markdown("**Language:** SQL")
        elif category == 'dax':
            language = 'dax'
            st.markdown("**Language:** DAX (Data Analysis Expressions)")
        else:
            language = st.selectbox(
                "Programming Language",
                ["python", "java", "javascript", "cpp"],
                format_func=lambda x: {
                    "python": "🐍 Python",
                    "java": "☕ Java", 
                    "javascript": "🟨 JavaScript",
                    "cpp": "⚡ C++"
                }[x]
            )
        
        # Code editor
        user_code = st.text_area(
            "Write your solution:",
            value=current_problem['template'],
            height=400,
            key=f"code_{problem_id}",
            help="💡 Tip: Use good variable names and add comments to explain your approach"
        )
        
        # Action buttons row
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("💾 Save", use_container_width=True):
                st.session_state.user_solutions[problem_id] = {
                    'code': user_code,
                    'language': language,
                    'timestamp': datetime.now().isoformat(),
                    'hints_used': st.session_state.hints_used.get(problem_id, 0)
                }
                st.markdown("""
                <div class="success-alert">
                    ✅ <strong>Solution saved successfully!</strong>
                </div>
                """, unsafe_allow_html=True)
        
        with action_col2:
            # AI Code Review
            if client and st.button("🤖 AI Review", use_container_width=True):
                if user_code.strip() and user_code != current_problem['template']:
                    with st.spinner("🤖 Analyzing your code..."):
                        feedback = get_ai_feedback(
                            user_code, 
                            current_problem['description'],
                            current_problem['category']
                        )
                        st.session_state.ai_feedback[problem_id] = feedback
                else:
                    st.warning("⚠️ Please write some code first!")
        
        with action_col3:
            # Test code button
            if category == 'sql' and st.button("🧪 Run Query", use_container_width=True):
                try:
                    with st.spinner("🔄 Executing query..."):
                        conn = create_database()
                        df = pd.read_sql_query(user_code, conn)
                        st.markdown("#### 📊 Query Results:")
                        st.dataframe(df, use_container_width=True)
                        conn.close()
                        st.markdown("""
                        <div class="success-alert">
                            ✅ <strong>Query executed successfully!</strong>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-alert">
                        ❌ <strong>SQL Error:</strong> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
            elif category == 'dax' and st.button("📊 Validate DAX", use_container_width=True):
                st.markdown("""
                <div class="info-alert">
                    📊 <strong>DAX Validation:</strong><br>
                    • Check syntax: Ensure proper function names and operators<br>
                    • Verify table/column references: [TableName][ColumnName]<br>
                    • Test in Power BI Desktop or Power BI Service<br>
                    • Common functions: SUM, CALCULATE, FILTER, ALL, VALUES
                </div>
                """, unsafe_allow_html=True)
            elif category not in ['sql', 'dax'] and st.button("🧪 Test Code", use_container_width=True):
                st.markdown("""
                <div class="info-alert">
                    💡 <strong>Testing Tip:</strong> Copy your code to your local environment or online IDE (like repl.it, CodePen) to test with the provided test cases.
                </div>
                """, unsafe_allow_html=True)
        
        # Show AI feedback if available
        if problem_id in st.session_state.ai_feedback:
            st.markdown("#### 🤖 AI Code Review")
            st.markdown(f"""
            <div class="solution-container">
                {st.session_state.ai_feedback[problem_id]}
            </div>
            """, unsafe_allow_html=True)
        
        # Code submission status
        if problem_id in st.session_state.user_solutions:
            solution = st.session_state.user_solutions[problem_id]
            submission_time = datetime.fromisoformat(solution['timestamp']).strftime("%Y-%m-%d %H:%M")
            hints_used = solution.get('hints_used', 0)
            
            st.markdown(f"""
            <div class="success-alert">
                <strong>✅ Last Submission:</strong><br>
                📅 {submission_time}<br>
                💡 Hints used: {hints_used}<br>
                🗣️ Language: {solution.get('language', 'python').title()}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
                