import streamlit as st
import json
import random
from datetime import datetime
from io import BytesIO
import pandas as pd
import openai
import os

# Set page configuration
st.set_page_config(
    page_title="Interactive Excel Training",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'scores' not in st.session_state:
    st.session_state.scores = {}
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'question_order' not in st.session_state:
    st.session_state.question_order = []

# Excel Training Questions Database
EXCEL_QUESTIONS = [
    {
        "id": 1,
        "category": "Formatting",
        "difficulty": "Easy",
        "question": "How do you center text in a column?",
        "question_type": "multiple_choice",
        "options": [
            "Home Tab → Alignment Group → Center Button",
            "Format Menu → Cell Format → Alignment",
            "Right-click → Format Cells → Center",
            "All of the above are correct"
        ],
        "correct_answer": 3,  # Index 0-based, so 3 = "All of the above"
        "explanation": "You can center text using the Center button in the Alignment group on the Home tab, through Format Menu, or by right-clicking and selecting Format Cells. All methods work!",
        "steps": [
            "1. Select the cells you want to center",
            "2. Go to Home tab in the ribbon",
            "3. Look for the Alignment group",
            "4. Click the Center button (or use Ctrl+E)"
        ]
    },
    {
        "id": 2,
        "category": "Data Analysis",
        "difficulty": "Easy",
        "question": "What is the first step to create a Pivot Table?",
        "question_type": "multiple_choice",
        "options": [
            "Go to Insert tab → Pivot Table",
            "Select your data range",
            "Go to Data tab → Pivot Table",
            "Format cells as Table"
        ],
        "correct_answer": 1,  # "Select your data range"
        "explanation": "Before creating a pivot table, you must first select the data range you want to analyze. Excel needs to know which data to work with.",
        "steps": [
            "1. Select all your data including headers",
            "2. Go to Insert tab in the ribbon",
            "3. Click Pivot Table",
            "4. Choose where to place the pivot table",
            "5. Drag fields to rows, columns, and values"
        ]
    },
    {
        "id": 3,
        "category": "Data Analysis",
        "difficulty": "Easy",
        "question": "How do you create a Pivot Table?",
        "question_type": "multiple_choice",
        "options": [
            "Data tab → Pivot Table",
            "Insert tab → Pivot Table",
            "Format tab → Create Pivot Table",
            "View tab → Pivot Table"
        ],
        "correct_answer": 1,  # "Insert tab → Pivot Table"
        "explanation": "In modern Excel, Pivot Tables are created via the Insert tab. Select your data first, then Insert → Pivot Table.",
        "steps": [
            "1. Select data (A1:D100)",
            "2. Click Insert tab",
            "3. Click Pivot Table button",
            "4. Select pivot table location",
            "5. Drag fields from Field List"
        ]
    },
    {
        "id": 4,
        "category": "Formulas",
        "difficulty": "Easy",
        "question": "How do you insert a SUM formula?",
        "question_type": "multiple_choice",
        "options": [
            "Type =SUM() directly in the cell",
            "Formulas tab → AutoSum → Sum",
            "Home tab → Sum button",
            "Both A and B are correct"
        ],
        "correct_answer": 3,  # "Both A and B are correct"
        "explanation": "You can insert a SUM formula by typing it directly (=SUM(A1:A10)) or by using the AutoSum button in the Formulas tab.",
        "steps": [
            "1. Select the cell where you want the sum",
            "2. Option A: Type =SUM(range)",
            "2. Option B: Go to Formulas tab → AutoSum",
            "3. Select or type the range",
            "4. Press Enter"
        ]
    },
    {
        "id": 5,
        "category": "Charts",
        "difficulty": "Easy",
        "question": "How do you create a chart?",
        "question_type": "multiple_choice",
        "options": [
            "Select data → Insert tab → Charts section",
            "Right-click data → Create Chart",
            "Format tab → Chart",
            "Charts option appears only in Excel Premium"
        ],
        "correct_answer": 0,  # "Select data → Insert tab → Charts section"
        "explanation": "To create a chart: select your data, go to the Insert tab, and choose from the Charts section. Excel will create a chart based on your selected data.",
        "steps": [
            "1. Select the data you want to chart",
            "2. Go to Insert tab",
            "3. In Charts section, click desired chart type",
            "4. Excel creates the chart",
            "5. Customize using Chart Design and Format tabs"
        ]
    },
    {
        "id": 6,
        "category": "Data Organization",
        "difficulty": "Easy",
        "question": "Where is the Sort button located?",
        "question_type": "multiple_choice",
        "options": [
            "Home tab → Editing group",
            "Data tab → Sort & Filter group",
            "Insert tab → Tables",
            "Format tab → Styles"
        ],
        "correct_answer": 1,  # "Data tab → Sort & Filter group"
        "explanation": "The Sort button is in the Data tab under the Sort & Filter group. You can also access it by going to Data → Sort.",
        "steps": [
            "1. Select your data range",
            "2. Go to Data tab",
            "3. In Sort & Filter group, click Sort A→Z, Z→A, or Sort",
            "4. Choose your sort criteria",
            "5. Click OK"
        ]
    },
    {
        "id": 7,
        "category": "Filters",
        "difficulty": "Easy",
        "question": "How do you apply an AutoFilter?",
        "question_type": "multiple_choice",
        "options": [
            "Data tab → Filter button",
            "Home tab → Filter",
            "Format tab → AutoFilter",
            "View tab → Filter"
        ],
        "correct_answer": 0,  # "Data tab → Filter button"
        "explanation": "AutoFilter is applied from the Data tab. Once applied, dropdown arrows appear in the header row allowing you to filter data.",
        "steps": [
            "1. Select your data range",
            "2. Go to Data tab",
            "3. Click Filter button (looks like a funnel)",
            "4. Dropdown arrows appear in headers",
            "5. Click dropdown to filter specific values"
        ]
    },
    {
        "id": 8,
        "category": "Formatting",
        "difficulty": "Easy",
        "question": "How do you freeze panes?",
        "question_type": "multiple_choice",
        "options": [
            "Format tab → Freeze",
            "View tab → Freeze Panes",
            "Data tab → Freeze",
            "Home tab → Lock Cells"
        ],
        "correct_answer": 1,  # "View tab → Freeze Panes"
        "explanation": "The Freeze Panes option is in the View tab. It keeps rows or columns visible while scrolling through data.",
        "steps": [
            "1. Click on cell below and to the right of what you want frozen",
            "2. Go to View tab",
            "3. Click Freeze Panes",
            "4. Select Freeze Panes option",
            "5. Lines appear showing frozen areas"
        ]
    },
    {
        "id": 9,
        "category": "Formulas",
        "difficulty": "Medium",
        "question": "How do you use a VLOOKUP formula?",
        "question_type": "multiple_choice",
        "options": [
            "=VLOOKUP(lookup_value, table_array, col_index, [range_lookup])",
            "Go to Formulas tab → VLOOKUP",
            "Format tab → Lookup",
            "Right-click → Insert Function"
        ],
        "correct_answer": 0,  # "=VLOOKUP..."
        "explanation": "VLOOKUP searches for a value in the leftmost column and returns a value from another column. Syntax: =VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])",
        "steps": [
            "1. Type =VLOOKUP(",
            "2. First parameter: the value to search for",
            "3. Second parameter: the table range",
            "4. Third parameter: which column to return (1, 2, 3...)",
            "5. Fourth parameter: FALSE for exact match",
            "6. Example: =VLOOKUP(A2,B:D,3,FALSE)"
        ]
    },
    {
        "id": 10,
        "category": "Data Organization",
        "difficulty": "Medium",
        "question": "How do you convert data to a Table?",
        "question_type": "multiple_choice",
        "options": [
            "Select data → Home tab → Format as Table",
            "Select data → Insert tab → Table",
            "Right-click → Create Table",
            "Data tab → Create Table"
        ],
        "correct_answer": 0,  # "Select data → Home tab → Format as Table"
        "explanation": "To convert data to a Table: select your data, go to Home tab, and click Format as Table. This adds filtering, formatting, and table functionality.",
        "steps": [
            "1. Select your data range",
            "2. Go to Home tab",
            "3. In Styles group, click Format as Table",
            "4. Choose a table style",
            "5. Confirm the data range",
            "6. Click OK"
        ]
    },
    {
        "id": 11,
        "category": "Charts",
        "difficulty": "Medium",
        "question": "How do you add a chart title?",
        "question_type": "multiple_choice",
        "options": [
            "Click chart → Chart Design tab → Add Chart Element → Title",
            "Right-click chart → Edit Title",
            "Format tab → Chart Title",
            "Click chart → Type title directly"
        ],
        "correct_answer": 0,  # "Click chart → Chart Design tab → Add Chart Element → Title"
        "explanation": "To add a chart title: click the chart, go to Chart Design tab, click Add Chart Element, then select Title from the dropdown.",
        "steps": [
            "1. Click on the chart to select it",
            "2. Go to Chart Design tab",
            "3. Click Add Chart Element dropdown",
            "4. Select Title",
            "5. Choose Above Chart or Centered Overlay Title",
            "6. Type your title"
        ]
    },
    {
        "id": 12,
        "category": "Formatting",
        "difficulty": "Easy",
        "question": "How do you apply conditional formatting?",
        "question_type": "multiple_choice",
        "options": [
            "Home tab → Conditional Formatting",
            "Format tab → Rules",
            "Data tab → Conditional",
            "View tab → Format"
        ],
        "correct_answer": 0,  # "Home tab → Conditional Formatting"
        "explanation": "Conditional Formatting is in the Home tab. It allows you to format cells based on their values or conditions.",
        "steps": [
            "1. Select the cells to format",
            "2. Go to Home tab",
            "3. In Styles group, click Conditional Formatting",
            "4. Choose: Highlight Cell Rules, Top/Bottom Rules, or New Rule",
            "5. Set your condition",
            "6. Choose formatting style",
            "7. Click OK"
        ]
    },
    {
        "id": 13,
        "category": "Formulas",
        "difficulty": "Medium",
        "question": "Where do you find the Function Wizard?",
        "question_type": "multiple_choice",
        "options": [
            "Formulas tab → Function Library → Function Wizard",
            "Click fx button next to formula bar",
            "Right-click cell → Insert Function",
            "All of the above"
        ],
        "correct_answer": 3,  # "All of the above"
        "explanation": "The Function Wizard can be accessed multiple ways: Formulas tab → Function Library → Function Wizard, clicking fx next to the formula bar, or right-clicking a cell.",
        "steps": [
            "1. Method 1: Formulas tab → Function Wizard button",
            "2. Method 2: Click fx icon (next to formula bar)",
            "3. Method 3: Right-click → Insert Function",
            "4. Function Wizard dialog opens",
            "5. Search for or select function",
            "6. Enter parameters and click OK"
        ]
    },
    {
        "id": 14,
        "category": "Data Validation",
        "difficulty": "Medium",
        "question": "How do you create a data validation list (dropdown)?",
        "question_type": "multiple_choice",
        "options": [
            "Data tab → Data Validation",
            "Home tab → Validation",
            "Format tab → Data List",
            "Insert tab → List"
        ],
        "correct_answer": 0,  # "Data tab → Data Validation"
        "explanation": "Data Validation is in the Data tab. Select cells, go to Data → Data Validation, set type to List, and enter your values.",
        "steps": [
            "1. Select the cells for the dropdown",
            "2. Go to Data tab",
            "3. Click Data Validation",
            "4. Set Allow: List",
            "5. Enter values: Item1, Item2, Item3 (comma-separated)",
            "6. Click OK",
            "7. Dropdown appears in selected cells"
        ]
    },
    {
        "id": 15,
        "category": "Charts",
        "difficulty": "Easy",
        "question": "What types of charts can you create?",
        "question_type": "multiple_choice",
        "options": [
            "Column, Bar, Line, Pie, Area, XY (Scatter) charts",
            "Only Column and Pie charts",
            "Only charts available in Office 365",
            "Bar and Line charts only"
        ],
        "correct_answer": 0,  # Column, Bar, Line, Pie, Area, XY (Scatter)
        "explanation": "Excel supports many chart types: Column (vertical), Bar (horizontal), Line, Pie, Area, XY (Scatter), Bubble, Stock, and Surface charts.",
        "steps": [
            "1. Select your data",
            "2. Go to Insert tab",
            "3. In Charts section, click Recommended Charts",
            "4. View available chart types",
            "5. Select desired chart type",
            "6. Click Insert"
        ]
    }
]

def get_openai_client():
    """Initialize OpenAI client from session state, secrets, or environment"""
    if st.session_state.openai_api_key:
        return openai.OpenAI(api_key=st.session_state.openai_api_key)
    
    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        st.session_state.openai_api_key = api_key
        st.session_state.api_key_set = True
        return openai.OpenAI(api_key=api_key)
    
    return None

def display_api_key_setup():
    """Display API key input interface"""
    st.title("🔐 OpenAI API Key Configuration")
    st.markdown("---")
    
    st.info("""
    To use this Excel Training app, you need to provide your OpenAI API key.
    
    **How to get an API key:**
    1. Go to https://platform.openai.com/api/keys
    2. Sign in with your OpenAI account
    3. Click "Create new secret key"
    4. Copy the key and paste it below
    
    **Security Note:** Your API key is stored securely in this session only.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        api_key_input = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            placeholder="sk-..."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Set API Key", use_container_width=True, type="primary"):
            if api_key_input and api_key_input.startswith("sk-"):
                st.session_state.openai_api_key = api_key_input
                st.session_state.api_key_set = True
                st.success("✅ API key configured!")
                st.rerun()
            else:
                st.error("❌ Invalid API key format.")
    
    st.markdown("---")
    
    # Check for existing key
    if not st.session_state.api_key_set:
        client = get_openai_client()
        if client:
            st.success("✅ API key detected from environment. Click below to proceed.")
            st.session_state.api_key_set = True
            return True
    
    return st.session_state.api_key_set

def display_welcome_screen():
    """Display welcome screen"""
    st.title("📚 Interactive Excel Training Center")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Learn Excel Through Interactive Quizzes")
        st.write("""
        Master Excel by answering real-world questions about Excel features and functions.
        
        **How It Works:**
        1. You'll be presented with Excel training questions
        2. Each question asks about Excel features, ribbon locations, and best practices
        3. Multiple choice answers with clear explanations
        4. Learn step-by-step instructions for each feature
        5. Get scored and see your progress
        
        **Training Features:**
        - 15+ interactive questions
        - Questions cover: Formatting, Charts, Formulas, Data Analysis, and more
        - Easy to Medium difficulty levels
        - Detailed explanations and step-by-step guides
        - Performance tracking and scoring
        - Learn at your own pace
        """)
    
    with col2:
        st.info("""
        **Topics Covered:**
        ✓ Data Formatting
        ✓ Pivot Tables
        ✓ Charts & Graphs
        ✓ Formulas (SUM, VLOOKUP)
        ✓ Filtering & Sorting
        ✓ Conditional Formatting
        ✓ Data Validation
        ✓ And more!
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📖 Start Training", use_container_width=True, type="primary", key="start_training"):
            # Randomize question order
            st.session_state.question_order = list(range(len(EXCEL_QUESTIONS)))
            random.shuffle(st.session_state.question_order)
            st.session_state.training_started = True
            st.session_state.current_question_idx = 0
            st.rerun()
    
    with col2:
        if st.button("📊 View Sample Questions", use_container_width=True):
            st.session_state.training_started = False
            show_sample_questions()
    
    with col3:
        if st.button("📚 Learn More", use_container_width=True):
            show_help_section()

def show_sample_questions():
    """Show sample questions"""
    st.subheader("📋 Sample Questions")
    
    for q in EXCEL_QUESTIONS[:3]:
        with st.expander(f"Question {q['id']}: {q['question']}", expanded=False):
            st.write(f"**Difficulty:** {q['difficulty']}")
            st.write(f"**Category:** {q['category']}")
            st.write(f"**Question:** {q['question']}")
            st.write("**Options:**")
            for i, option in enumerate(q['options']):
                st.write(f"  {chr(65+i)}) {option}")

def show_help_section():
    """Show help section"""
    st.subheader("❓ How to Use This Training App")
    
    st.write("""
    **Getting Started:**
    1. Click "Start Training" to begin
    2. Answer multiple-choice questions about Excel
    3. After each answer, see the explanation
    4. Learn step-by-step instructions for each feature
    5. Get scoring based on correct answers
    
    **Question Types:**
    - Multiple choice with 4 options
    - Questions cover ribbon locations, menu paths, and Excel functions
    - Each question has detailed explanation and steps
    
    **Tips for Success:**
    - Read all options carefully
    - Understanding ribbon layout helps
    - Each correct answer builds your Excel knowledge
    - Review explanations to learn faster
    """)

def display_question_screen():
    """Display training question"""
    if not st.session_state.question_order:
        st.session_state.question_order = list(range(len(EXCEL_QUESTIONS)))
        random.shuffle(st.session_state.question_order)
    
    question_idx = st.session_state.question_order[st.session_state.current_question_idx]
    question = EXCEL_QUESTIONS[question_idx]
    
    # Progress
    progress = st.session_state.current_question_idx / len(EXCEL_QUESTIONS)
    st.progress(progress, text=f"Question {st.session_state.current_question_idx + 1} of {len(EXCEL_QUESTIONS)}")
    
    # Question header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title(f"Q{st.session_state.current_question_idx + 1}: {question['question']}")
    with col2:
        difficulty_color = "🟢" if question['difficulty'] == "Easy" else "🟡"
        st.metric("Level", f"{difficulty_color} {question['difficulty']}")
    with col3:
        st.metric("Category", question['category'])
    
    st.markdown("---")
    
    # Question
    st.subheader("Question")
    st.write(question['question'])
    
    # Options
    st.subheader("Choose your answer:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Option A:**")
        option_a = st.radio(
            "Select option A:",
            [question['options'][0]],
            label_visibility="collapsed",
            key=f"opt_a_{question['id']}"
        )
    
    with col2:
        st.write("**Option B:**")
        option_b = st.radio(
            "Select option B:",
            [question['options'][1]],
            label_visibility="collapsed",
            key=f"opt_b_{question['id']}"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Option C:**")
        option_c = st.radio(
            "Select option C:",
            [question['options'][2]],
            label_visibility="collapsed",
            key=f"opt_c_{question['id']}"
        )
    
    with col2:
        st.write("**Option D:**")
        option_d = st.radio(
            "Select option D:",
            [question['options'][3]],
            label_visibility="collapsed",
            key=f"opt_d_{question['id']}"
        )
    
    # Better option selection
    st.subheader("Select your answer:")
    selected_option = st.radio(
        "Which option is correct?",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Option {chr(65+x)}: {question['options'][x]}",
        horizontal=True,
        key=f"answer_{question['id']}"
    )
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.current_question_idx > 0:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.current_question_idx -= 1
                st.rerun()
    
    with col2:
        if st.button("💡 Show Explanation", use_container_width=True):
            st.info(f"""
            **Correct Answer:** Option {chr(65 + question['correct_answer'])}
            
            **Explanation:** {question['explanation']}
            
            **Steps to Master This:**
            """)
            for step in question['steps']:
                st.write(step)
    
    with col3:
        if st.button("✅ Submit Answer", use_container_width=True, type="primary"):
            is_correct = (selected_option == question['correct_answer'])
            st.session_state.answers[question_idx] = {
                "selected": selected_option,
                "correct": question['correct_answer'],
                "is_correct": is_correct
            }
            
            if is_correct:
                st.session_state.scores[question_idx] = 100
                st.success("✅ Correct! Well done!")
            else:
                st.session_state.scores[question_idx] = 0
                st.error(f"❌ Incorrect. The correct answer is Option {chr(65 + question['correct_answer'])}: {question['options'][question['correct_answer']]}")
            
            st.info(f"""
            **Explanation:** {question['explanation']}
            
            **How to do this:**
            """)
            for step in question['steps']:
                st.write(step)
            
            st.write("---")
            if st.button("Next Question →", use_container_width=True):
                if st.session_state.current_question_idx < len(EXCEL_QUESTIONS) - 1:
                    st.session_state.current_question_idx += 1
                    st.rerun()
                else:
                    st.session_state.training_completed = True
                    st.rerun()
    
    with col4:
        if st.button("⏹️ End Training", use_container_width=True):
            st.session_state.training_completed = True
            st.rerun()

def display_results_screen():
    """Display results"""
    st.title("📊 Training Complete!")
    st.markdown("---")
    
    if not st.session_state.answers:
        st.info("You haven't answered any questions yet. Start training to see results.")
        if st.button("🚀 Go Back to Welcome", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['api_key_set', 'openai_api_key']:
                    del st.session_state[key]
            st.rerun()
        return
    
    # Calculate score
    total_questions = len(st.session_state.answers)
    correct_answers = sum(1 for score in st.session_state.scores.values() if score == 100)
    percentage = (correct_answers / total_questions * 100) if total_questions > 0 else 0
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Score", f"{correct_answers}/{total_questions}")
    col2.metric("Percentage", f"{percentage:.1f}%")
    col3.metric("Questions", total_questions)
    col4.metric("Correct", correct_answers)
    
    # Performance level
    st.markdown("---")
    if percentage >= 90:
        st.success("🌟 Excellent! You have strong Excel knowledge!")
    elif percentage >= 70:
        st.info("👍 Good! You have solid Excel skills.")
    elif percentage >= 50:
        st.warning("📈 Fair. Keep practicing to improve!")
    else:
        st.error("⚠️ Keep learning! Review the explanations and try again.")
    
    st.markdown("---")
    
    # Detailed results
    st.subheader("📋 Question Breakdown")
    
    results_data = []
    for idx, question_idx in enumerate(st.session_state.question_order):
        if question_idx in st.session_state.answers:
            question = EXCEL_QUESTIONS[question_idx]
            answer_data = st.session_state.answers[question_idx]
            
            results_data.append({
                "Q#": idx + 1,
                "Question": question['question'][:50] + "...",
                "Category": question['category'],
                "Your Answer": chr(65 + answer_data['selected']),
                "Correct Answer": chr(65 + answer_data['correct']),
                "Result": "✅ Correct" if answer_data['is_correct'] else "❌ Incorrect"
            })
    
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Detailed review
    st.subheader("📚 Review Your Answers")
    
    for idx, question_idx in enumerate(st.session_state.question_order):
        if question_idx in st.session_state.answers:
            question = EXCEL_QUESTIONS[question_idx]
            answer_data = st.session_state.answers[question_idx]
            
            result_icon = "✅" if answer_data['is_correct'] else "❌"
            with st.expander(f"{result_icon} Q{idx+1}: {question['question']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Your Answer:** Option {chr(65 + answer_data['selected'])}")
                    st.write(f"{question['options'][answer_data['selected']]}")
                
                with col2:
                    st.write(f"**Correct Answer:** Option {chr(65 + answer_data['correct'])}")
                    st.write(f"{question['options'][answer_data['correct']]}")
                
                st.write("---")
                st.write(f"**Explanation:** {question['explanation']}")
                st.write("**Steps:**")
                for step in question['steps']:
                    st.write(step)
    
    st.markdown("---")
    
    # Export results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df_results.to_csv(index=False)
        st.download_button(
            "📥 Download Results (CSV)",
            data=csv,
            file_name=f"excel_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Retake Training", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['api_key_set', 'openai_api_key']:
                    del st.session_state[key]
            st.rerun()
    
    with col3:
        if st.button("🏠 Back to Home", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['api_key_set', 'openai_api_key']:
                    del st.session_state[key]
            st.rerun()

def main():
    """Main app logic"""
    
    # Sidebar
    with st.sidebar:
        st.title("📚 Excel Training")
        st.info(f"""
        **Status:** {'🟢 In Progress' if st.session_state.training_started else '🟡 Not Started'}
        
        **Questions:** {len(EXCEL_QUESTIONS)}
        
        **Difficulty:** Easy - Medium
        """)
        
        if st.session_state.training_started and not st.session_state.training_completed:
            st.markdown("---")
            st.subheader("Progress")
            st.write(f"Question {st.session_state.current_question_idx + 1} of {len(EXCEL_QUESTIONS)}")
        
        if st.session_state.api_key_set:
            st.markdown("---")
            st.success("✅ API Key Configured")
            if st.button("🔄 Change API Key"):
                st.session_state.api_key_set = False
                st.session_state.openai_api_key = None
                st.rerun()
    
    # Main content
    if not st.session_state.api_key_set:
        display_api_key_setup()
    elif not st.session_state.training_started:
        display_welcome_screen()
    elif st.session_state.training_completed:
        display_results_screen()
    else:
        display_question_screen()

if __name__ == "__main__":
    main()
