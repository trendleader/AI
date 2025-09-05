"""
Fixed Chainlit SQL Interview Prep RAG Chatbot
Resolved the Message.update() error
"""

import os
import asyncio
from typing import List, Dict, Optional
import chainlit as cl
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

class SQLRAGChatbot:
    """SQL RAG chatbot for interview preparation"""
    
    def __init__(self):
        self.vectorstore = None
        self.embeddings = None
        self.conversation_history = []
        
    async def setup_embeddings(self):
        """Setup embedding model"""
        await cl.Message(content="🔧 Loading embedding model...").send()
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        await cl.Message(content="✅ Embeddings ready!").send()
    
    def load_sql_interview_content(self) -> List[Document]:
        """Load the SQL interview content from your original PDF"""
        
        # Content extracted from your original PDF
        sql_content = {
            "WHERE vs HAVING": """
            What is the difference between WHERE and HAVING?
            
            WHERE filters rows before aggregation.
            HAVING filters groups after aggregation.
            
            Example:
            SELECT department, COUNT(*) 
            FROM employees 
            WHERE status = 'active' 
            GROUP BY department 
            HAVING COUNT(*) > 5;
            
            Key differences:
            - WHERE is applied to individual rows before any grouping
            - HAVING is applied to grouped rows after GROUP BY
            - WHERE cannot use aggregate functions like COUNT(), SUM()
            - HAVING can use aggregate functions
            """,
            
            "Window Functions": """
            Explain the use of ROW_NUMBER(), RANK(), and DENSE_RANK().
            
            These are window functions used for ranking rows:
            
            ROW_NUMBER(): Unique row numbers even if values are the same.
            - Always assigns sequential numbers: 1, 2, 3, 4...
            
            RANK(): Gives the same rank to ties, but skips the next rank(s).
            - With ties: 1, 2, 2, 4...
            
            DENSE_RANK(): Gives the same rank to ties without skipping ranks.
            - With ties: 1, 2, 2, 3...
            
            Example:
            SELECT name, salary,
                ROW_NUMBER() OVER (ORDER BY salary DESC) as row_num,
                RANK() OVER (ORDER BY salary DESC) as rank_val,
                DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank_val
            FROM employees;
            """,
            
            "CTEs vs Subqueries": """
            What is a CTE and how is it different from a subquery?
            
            A Common Table Expression (CTE) is a temporary result set defined using WITH.
            
            CTEs vs Subqueries:
            - CTEs are more readable and can be reused within the same query
            - Subqueries are nested inside the main query and not reusable
            - CTEs can be recursive
            - CTEs appear at the beginning of the query
            
            CTE Example:
            WITH active_customers AS (
                SELECT * FROM customers WHERE status = 'active'
            )
            SELECT city, COUNT(*) FROM active_customers GROUP BY city;
            
            Subquery Example:
            SELECT city, COUNT(*)
            FROM (SELECT * FROM customers WHERE status = 'active') subq
            GROUP BY city;
            """,
            
            "JOIN Types": """
            How does a LEFT JOIN differ from an INNER JOIN?
            
            INNER JOIN: Returns only matching rows from both tables.
            - Only includes records where the join condition is met in both tables
            - Smaller result set
            
            LEFT JOIN: Returns all rows from the left table and matching rows from the right.
            - Includes all records from the left table
            - Shows NULL for unmatched records from the right table
            - Also called LEFT OUTER JOIN
            
            Example:
            -- INNER JOIN (only employees with departments)
            SELECT e.name, d.department_name
            FROM employees e
            INNER JOIN departments d ON e.dept_id = d.dept_id;
            
            -- LEFT JOIN (all employees, even without departments)
            SELECT e.name, d.department_name
            FROM employees e
            LEFT JOIN departments d ON e.dept_id = d.dept_id;
            """,
            
            "UNION Operations": """
            What is the difference between UNION and UNION ALL?
            
            UNION: Removes duplicates.
            - Performs automatic deduplication
            - Slower execution due to duplicate checking
            - Returns unique rows only
            
            UNION ALL: Keeps duplicates.
            - No deduplication performed
            - Faster execution
            - Returns all rows including duplicates
            
            Example:
            -- UNION (removes duplicates)
            SELECT city FROM customers
            UNION
            SELECT city FROM suppliers;
            
            -- UNION ALL (keeps duplicates)
            SELECT city FROM customers
            UNION ALL
            SELECT city FROM suppliers;
            """,
            
            "Nth Highest Value": """
            How can you find the Nth highest value in a column?
            
            Multiple approaches to find the Nth highest value:
            
            Method 1 - Using DENSE_RANK() (Recommended):
            SELECT salary 
            FROM (
                SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk 
                FROM employees
            ) ranked
            WHERE rnk = 3;  -- This gives the 3rd highest salary
            
            Method 2 - Using LIMIT/OFFSET (MySQL, PostgreSQL):
            SELECT DISTINCT salary 
            FROM employees 
            ORDER BY salary DESC 
            LIMIT 1 OFFSET 2;  -- 3rd highest (0-indexed)
            
            Method 3 - Using ROW_NUMBER():
            SELECT salary
            FROM (
                SELECT DISTINCT salary, 
                       ROW_NUMBER() OVER (ORDER BY salary DESC) as rn
                FROM employees
            ) t
            WHERE rn = 3;
            
            Key considerations:
            - DENSE_RANK() handles ties better than ROW_NUMBER()
            - Use DISTINCT to avoid counting duplicate salary values
            - Different databases have slightly different syntax
            """,
            
            "Database Indexes": """
            What are indexes and how do they affect performance?
            
            Indexes are database objects that speed up data retrieval operations.
            Think of them like an index in a book - they help find data quickly.
            
            Performance Effects:
            ✅ Speed up SELECT queries (faster reads)
            ✅ Speed up WHERE, ORDER BY, JOIN operations
            ✅ Speed up sorting and grouping operations
            ❌ Slow down INSERT, UPDATE, DELETE (slower writes)
            ❌ Require additional storage space
            ❌ Need maintenance overhead
            
            Common Index Types:
            - B-tree: Most common, good for range queries and sorting
            - Hash: Very fast for exact matches, not good for ranges
            - Full-text: Specialized for text search operations
            - Bitmap: Good for low-cardinality data
            
            When to use indexes:
            - Columns frequently used in WHERE clauses
            - Columns used in JOIN conditions
            - Columns used in ORDER BY clauses
            - Foreign key columns
            
            When NOT to use indexes:
            - Tables with frequent INSERT/UPDATE/DELETE operations
            - Very small tables (overhead not worth it)
            - Columns that change frequently
            """,
            
            "Correlated Subqueries": """
            What is a correlated subquery and how is it different from a regular subquery?
            
            Regular Subquery:
            - Runs independently of the outer query
            - Executes once and returns a result
            - Can be executed separately from the outer query
            - Generally more efficient
            
            Correlated Subquery:
            - References columns from the outer query
            - Executes once for each row in the outer query
            - Cannot be executed independently
            - Often less efficient for large datasets
            
            Regular Subquery Example:
            SELECT name FROM employees 
            WHERE salary > (SELECT AVG(salary) FROM employees);
            
            Correlated Subquery Example:
            SELECT name, salary 
            FROM employees e1
            WHERE salary > (
                SELECT AVG(salary) 
                FROM employees e2
                WHERE e2.department = e1.department  -- References outer query
            );
            
            In the correlated example, the subquery runs once for each employee,
            comparing their salary to the average of their specific department.
            
            Performance Considerations:
            - Regular subqueries are generally faster
            - Correlated subqueries can be expensive for large datasets
            - Consider using JOINs or window functions as alternatives
            """,
            
            "CASE Statements": """
            When would you use a CASE statement in SQL?
            
            A CASE statement is used for conditional logic, like if-else in programming.
            
            Two types of CASE statements:
            
            1. Searched CASE (uses conditions):
            SELECT name,
                CASE 
                    WHEN salary >= 100000 THEN 'High'
                    WHEN salary >= 50000 THEN 'Medium'
                    ELSE 'Low'
                END AS salary_band
            FROM employees;
            
            2. Simple CASE (compares to specific values):
            SELECT name,
                CASE department
                    WHEN 'IT' THEN 'Technology'
                    WHEN 'HR' THEN 'Human Resources'
                    WHEN 'FIN' THEN 'Finance'
                    ELSE 'Other'
                END AS dept_category
            FROM employees;
            
            Common Use Cases:
            - Creating categories from continuous data
            - Conditional aggregations
            - Data transformation and cleansing
            - Custom sorting logic
            - Pivot-like operations
            - Handling NULL values conditionally
            
            It's useful for creating new labels or categories from existing data
            without needing separate lookup tables.
            """,
            
            "Duplicate Handling": """
            How do you detect and remove duplicates from a table while keeping only one record?
            
            Detecting Duplicates:
            SELECT name, email, COUNT(*) as duplicate_count
            FROM customers 
            GROUP BY name, email
            HAVING COUNT(*) > 1;
            
            Removing Duplicates (Multiple Methods):
            
            Method 1 - Using MIN/MAX with subquery:
            DELETE FROM customers
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM customers
                GROUP BY name, email
            );
            
            Method 2 - Using ROW_NUMBER() window function:
            DELETE FROM customers
            WHERE id IN (
                SELECT id FROM (
                    SELECT id, 
                           ROW_NUMBER() OVER (
                               PARTITION BY name, email 
                               ORDER BY id
                           ) as rn
                    FROM customers
                ) t
                WHERE rn > 1
            );
            
            Method 3 - Using self-join:
            DELETE c1 FROM customers c1
            INNER JOIN customers c2
            WHERE c1.id > c2.id 
            AND c1.name = c2.name 
            AND c1.email = c2.email;
            
            Prevention Strategies:
            - Use UNIQUE constraints on combinations of columns
            - Use UPSERT operations (INSERT ... ON CONFLICT)
            - Validate data before insertion
            - Use primary keys effectively
            
            This keeps only the row with the smallest ID for each unique combination.
            """,
            
            "Interview Tips": """
            Final Tips for SQL Interview Success
            
            Technical Preparation:
            - Practice regularly using real datasets
            - Understand performance implications of joins, subqueries, and indexing
            - Be ready to explain your logic clearly, not just write queries
            - Know the differences between database systems (MySQL, PostgreSQL, SQL Server)
            - Practice with sample databases like Northwind or Sakila
            
            During the Interview:
            - Think out loud - explain your reasoning
            - Ask clarifying questions about the requirements
            - Start with a simple solution, then optimize
            - Consider edge cases (NULL values, empty results)
            - Discuss performance implications of your queries
            
            Common Topics to Review:
            - Window functions and their use cases
            - Different types of JOINs and when to use each
            - Subqueries vs CTEs vs JOINs performance
            - Indexing strategies and trade-offs
            - Normalization and denormalization
            - Transaction concepts (ACID properties)
            
            Practical Skills:
            - Be able to write queries from scratch
            - Debug and optimize existing queries
            - Explain execution plans
            - Handle data quality issues
            
            SQL mastery is about more than syntax—it's about thinking through problems, 
            choosing efficient solutions, and understanding data at its core.
            
            Remember: Employers want to see your problem-solving process, not just the final answer.
            """
        }
        
        # Convert to Document objects
        documents = []
        for topic, content in sql_content.items():
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "source": "sql_interview_prep.pdf",
                    "topic": topic,
                    "type": "interview_question"
                }
            )
            documents.append(doc)
        
        return documents
    
    async def build_vectorstore(self):
        """Build the vector store from documents"""
        await cl.Message(content="📚 Loading SQL interview content...").send()
        
        # Load documents
        documents = self.load_sql_interview_content()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        await cl.Message(content=f"📄 Created {len(chunks)} content chunks").send()
        await cl.Message(content="🔧 Building vector store...").send()
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        await cl.Message(content="✅ Vector store ready!").send()
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        try:
            # Get relevant documents
            docs = self.vectorstore.similarity_search(question, k=3)
            
            # Combine relevant content
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Extract topics from retrieved docs
            topics = [doc.metadata.get("topic", "Unknown") for doc in docs]
            main_topic = topics[0] if topics else "SQL Concepts"
            
            # Generate structured response
            response = f"""## {main_topic}

{context}

---
**Related Topics:** {', '.join(set(topics))}
**Sources:** {len(docs)} relevant sections found"""
            
            return response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"

# Global chatbot instance
chatbot = SQLRAGChatbot()

@cl.on_chat_start
async def start():
    """Initialize the chatbot when a new chat starts"""
    
    # Welcome message
    welcome_msg = """# 🎯 SQL Interview Prep Assistant

Welcome! I'm your AI assistant for SQL interview preparation.

## 📚 Topics I Can Help With:
- **Basic Concepts**: WHERE vs HAVING, JOIN types
- **Advanced Features**: Window functions, CTEs, Subqueries  
- **Performance**: Indexing, Query optimization
- **Practical Skills**: Finding Nth highest values, Removing duplicates
- **Interview Tips**: Best practices and common questions

## 💡 Example Questions:
- "What's the difference between WHERE and HAVING?"
- "How do window functions work?"
- "Explain CTEs vs subqueries"
- "How do I find the 3rd highest salary?"
- "What are some SQL interview tips?"

---
**Setting up the system...**"""
    
    await cl.Message(content=welcome_msg).send()
    
    try:
        # Initialize components
        await chatbot.setup_embeddings()
        await chatbot.build_vectorstore()
        
        ready_msg = """✅ **System Ready!**

🚀 You can now ask me any SQL interview questions. I'll provide detailed explanations with examples and best practices.

What would you like to learn about?"""
        
        await cl.Message(content=ready_msg).send()
        
    except Exception as e:
        error_msg = f"❌ **Setup Error**: {str(e)}\n\nPlease check your environment and try again."
        await cl.Message(content=error_msg).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    
    user_question = message.content
    
    # Add to conversation history
    chatbot.conversation_history.append({"type": "user", "content": user_question})
    
    try:
        # Show thinking indicator
        thinking_msg = await cl.Message(content="🤔 Searching for relevant information...").send()
        
        # Get response from chatbot
        response = chatbot.query(user_question)
        
        # Send the final response as a new message instead of updating
        await cl.Message(content=response).send()
        
        # Add to conversation history
        chatbot.conversation_history.append({"type": "assistant", "content": response})
        
        # Suggest related questions for first few interactions
        if len(chatbot.conversation_history) <= 4:
            suggestions = get_related_questions(user_question)
            if suggestions:
                suggestion_msg = f"""💡 **You might also ask:**

{chr(10).join([f"• {q}" for q in suggestions])}"""
                await cl.Message(content=suggestion_msg).send()
                
    except Exception as e:
        error_response = f"""❌ **Error**: I encountered an issue processing your question.

**Error details**: {str(e)}

Please try rephrasing your question or ask about a different SQL topic."""
        
        await cl.Message(content=error_response).send()

def get_related_questions(user_question):
    """Get related questions based on user input"""
    question_lower = user_question.lower()
    
    if any(word in question_lower for word in ["where", "having"]):
        return [
            "What are aggregate functions?",
            "How do GROUP BY clauses work?",
            "What's the difference between INNER and LEFT JOIN?"
        ]
    elif any(word in question_lower for word in ["window", "rank", "row_number"]):
        return [
            "What are CTEs and when should I use them?",
            "How do I find the Nth highest value?",
            "What's the difference between RANK and DENSE_RANK?"
        ]
    elif any(word in question_lower for word in ["3rd", "third", "highest", "nth"]):
        return [
            "How do window functions work?",
            "What's the difference between RANK and DENSE_RANK?",
            "How do I optimize queries with subqueries?"
        ]
    elif any(word in question_lower for word in ["cte", "with"]):
        return [
            "What are correlated subqueries?",
            "When should I use subqueries vs JOINs?",
            "How do recursive CTEs work?"
        ]
    elif any(word in question_lower for word in ["join", "inner", "left"]):
        return [
            "What are the different types of JOINs?",
            "How do I optimize JOIN performance?",
            "What's a self-join?"
        ]
    elif any(word in question_lower for word in ["index", "performance"]):
        return [
            "How do I optimize slow queries?",
            "What are execution plans?",
            "When should I avoid indexes?"
        ]
    else:
        return [
            "What are some common SQL interview mistakes?",
            "How do I prepare for a technical SQL interview?",
            "What are the most important SQL concepts to know?"
        ]

if __name__ == "__main__":
    # This would be run with: python -m chainlit run fixed_sql_chatbot.py
    pass