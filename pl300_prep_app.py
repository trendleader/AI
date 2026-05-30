import streamlit as st
import random
import json
from datetime import datetime

st.set_page_config(
    page_title="PL-300 Exam Prep",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0078D4 0%, #00BCF2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p { margin: 0.5rem 0 0; opacity: 0.9; font-size: 1.1rem; }

    .question-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-left: 5px solid #0078D4;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .question-card h4 { color: #0078D4; margin-top: 0; }

    .correct-answer {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .wrong-answer {
        background: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .explanation-box {
        background: #e8f4fd;
        border: 1px solid #0078D4;
        border-radius: 6px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .score-card {
        background: linear-gradient(135deg, #0078D4, #00BCF2);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .topic-badge {
        display: inline-block;
        background: #0078D4;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .exam-info {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .domain-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .domain-card h4 { color: #0078D4; margin-top: 0; margin-bottom: 0.3rem; }
    .progress-text { font-size: 0.85rem; color: #666; margin-top: 0.3rem; }
    .stButton > button {
        border-radius: 6px;
    }
    .flashcard-front {
        background: linear-gradient(135deg, #0078D4, #005a9e);
        color: white;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .flashcard-back {
        background: #f0f8ff;
        border: 2px solid #0078D4;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1rem;
        font-size: 1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ── Question Bank ──────────────────────────────────────────────────────────────

QUESTIONS = [
    # ── Prepare the Data (25-30%) ──────────────────────────────────────────────
    {
        "id": 1,
        "domain": "Prepare the Data",
        "topic": "Power Query",
        "difficulty": "Medium",
        "question": "You need to combine data from two tables where the first table contains sales transactions and the second contains product details. You want all sales records to appear even if there is no matching product. Which type of merge should you use in Power Query?",
        "options": [
            "Inner Join",
            "Left Outer Join",
            "Right Outer Join",
            "Full Outer Join"
        ],
        "answer": 1,
        "explanation": "A Left Outer Join returns all rows from the left (first) table and matching rows from the right (second) table. Rows from the left table with no match show null for right-table columns. This ensures all sales records are included regardless of product match."
    },
    {
        "id": 2,
        "domain": "Prepare the Data",
        "topic": "Data Types",
        "difficulty": "Easy",
        "question": "A column in Power Query contains values like '1,234.56'. When you import this column, it is classified as Text. What is the most efficient way to convert it to a usable decimal number?",
        "options": [
            "Use Replace Values to remove commas, then change the data type to Decimal Number",
            "Add a custom column using Number.FromText()",
            "Split the column by delimiter",
            "Create a calculated column in the data model"
        ],
        "answer": 0,
        "explanation": "The correct approach is to use Replace Values to remove the comma separator, then change the column's data type to Decimal Number. This is a common data-cleansing step in Power Query before loading to the model."
    },
    {
        "id": 3,
        "domain": "Prepare the Data",
        "topic": "Parameters",
        "difficulty": "Hard",
        "question": "You want Power Query to dynamically switch the source file path based on a value stored in a SharePoint list cell. Which feature enables this?",
        "options": [
            "Query folding",
            "Power Query parameters",
            "Dataflow incremental refresh",
            "Custom connector"
        ],
        "answer": 1,
        "explanation": "Power Query parameters let you store a value externally and reference it in queries. You can bind a parameter to a SharePoint list value, and when the list cell changes the parameter value updates, changing the source path dynamically."
    },
    {
        "id": 4,
        "domain": "Prepare the Data",
        "topic": "Query Folding",
        "difficulty": "Hard",
        "question": "Which of the following Power Query transformations will break query folding against a SQL Server source?",
        "options": [
            "Filtering rows by a column value",
            "Removing columns",
            "Adding an index column",
            "Sorting rows"
        ],
        "answer": 2,
        "explanation": "Adding an Index Column is a client-side operation that cannot be translated into SQL. It breaks query folding, meaning subsequent steps cannot be pushed back to the source database, which can significantly reduce performance on large datasets."
    },
    {
        "id": 5,
        "domain": "Prepare the Data",
        "topic": "Dataflows",
        "difficulty": "Medium",
        "question": "You have a Dataflow in Power BI Service that multiple datasets consume. The underlying source data changes every hour. What is the most efficient refresh strategy?",
        "options": [
            "Set each dataset to refresh independently every hour",
            "Enable incremental refresh on each dataset",
            "Configure the Dataflow to refresh hourly and set datasets to refresh after the Dataflow completes",
            "Use DirectQuery on all datasets"
        ],
        "answer": 2,
        "explanation": "When multiple datasets share a Dataflow, the best practice is to refresh the Dataflow first (transformations run once) and then cascade dataset refreshes. This avoids redundant source queries and ensures consistency across all consuming datasets."
    },
    {
        "id": 6,
        "domain": "Prepare the Data",
        "topic": "Power Query M",
        "difficulty": "Hard",
        "question": "In Power Query M, what does the #shared environment contain?",
        "options": [
            "The list of all tables in the current data model",
            "All built-in Power Query functions, constants, and values",
            "Shared queries between datasets in the same workspace",
            "User-defined parameters for the current query"
        ],
        "answer": 1,
        "explanation": "#shared is a special M record that contains all built-in Power Query functions, types, and library values. Typing #shared in the Advanced Editor exposes the full M library, which is useful for discovery and documentation purposes."
    },

    # ── Model the Data (25-30%) ────────────────────────────────────────────────
    {
        "id": 7,
        "domain": "Model the Data",
        "topic": "Star Schema",
        "difficulty": "Medium",
        "question": "You are designing a data model for a retail company. You have a Sales fact table and dimension tables for Date, Product, Store, and Customer. A Product belongs to a Category, and a Category belongs to a Department. How should you handle the Product-Category-Department hierarchy?",
        "options": [
            "Create three separate dimension tables: Product, Category, Department with individual relationships to Sales",
            "Flatten the hierarchy into a single Product dimension table with Category and Department columns",
            "Create a snowflake schema with Product → Category → Department",
            "Create a bridge table between Product and Department"
        ],
        "answer": 1,
        "explanation": "Best practice in Power BI is to flatten hierarchical dimensions into a single wide dimension table (star schema). This avoids the extra joins of a snowflake schema, improves query performance, and simplifies DAX calculations. The Category and Department columns are denormalized into the Product dimension."
    },
    {
        "id": 8,
        "domain": "Model the Data",
        "topic": "Relationships",
        "difficulty": "Medium",
        "question": "You have a Sales table and a Date table. The Sales table has both an OrderDate and a ShipDate column. How do you create relationships so that measures can filter by either date?",
        "options": [
            "Create two active relationships between Sales and Date",
            "Create one active relationship (OrderDate) and one inactive relationship (ShipDate); use USERELATIONSHIP() in measures that need ShipDate",
            "Duplicate the Date table and relate each copy to one date column",
            "Use a calculated column to merge the dates"
        ],
        "answer": 1,
        "explanation": "Power BI only allows one active relationship between two tables. The standard pattern is to keep one relationship active (OrderDate) and make the other inactive (ShipDate). Use USERELATIONSHIP() in DAX measures to activate the inactive relationship at query time."
    },
    {
        "id": 9,
        "domain": "Model the Data",
        "topic": "DAX",
        "difficulty": "Hard",
        "question": "You write the measure: Sales YTD = TOTALYTD(SUM(Sales[Amount]), 'Date'[Date]). The fiscal year ends on June 30. Which modification makes this measure respect the fiscal year?",
        "options": [
            "TOTALYTD(SUM(Sales[Amount]), 'Date'[Date], \"06-30\")",
            "TOTALYTD(SUM(Sales[Amount]), 'Date'[Date], ALL('Date'), \"06-30\")",
            "DATESYTD('Date'[Date], \"06-30\")",
            "TOTALYTD(SUM(Sales[Amount]), DATESYTD('Date'[Date], \"06-30\"))"
        ],
        "answer": 0,
        "explanation": "TOTALYTD accepts an optional year-end-date argument as its third parameter. TOTALYTD(expression, dates, \"06-30\") calculates year-to-date up to June 30 as the fiscal year end. The date format is \"MM-DD\"."
    },
    {
        "id": 10,
        "domain": "Model the Data",
        "topic": "DAX",
        "difficulty": "Hard",
        "question": "You have a Sales table with a SalesPersonID column and a Hierarchy table with EmployeeID and ManagerID columns. You need a measure that calculates total sales for a salesperson and all their subordinates. Which DAX function is essential?",
        "options": [
            "RELATED()",
            "PATH() and PATHCONTAINS()",
            "TREATAS()",
            "CROSSFILTER()"
        ],
        "answer": 1,
        "explanation": "PATH() generates a delimited text string of all ancestors for each node in a parent-child hierarchy. PATHCONTAINS() checks whether a specific ID appears in that path. Together they let you filter for a manager and all descendants without knowing the hierarchy depth, enabling recursive rollup measures."
    },
    {
        "id": 11,
        "domain": "Model the Data",
        "topic": "DAX",
        "difficulty": "Medium",
        "question": "What is the difference between CALCULATE and CALCULATETABLE in DAX?",
        "options": [
            "CALCULATE works on measures; CALCULATETABLE works on columns",
            "CALCULATE returns a scalar value; CALCULATETABLE returns a table",
            "CALCULATE modifies row context; CALCULATETABLE modifies filter context",
            "They are identical; CALCULATETABLE is deprecated"
        ],
        "answer": 1,
        "explanation": "CALCULATE evaluates an expression in a modified filter context and returns a scalar (single value). CALCULATETABLE does the same but returns a table instead of a scalar. CALCULATETABLE is used when you need to pass a filtered table to another function."
    },
    {
        "id": 12,
        "domain": "Model the Data",
        "topic": "Row-Level Security",
        "difficulty": "Medium",
        "question": "You want to implement dynamic row-level security so that each salesperson sees only their own sales data. The Sales table has a SalesPersonEmail column. Which DAX expression should you use in the RLS role filter?",
        "options": [
            "[SalesPersonEmail] = USERNAME()",
            "[SalesPersonEmail] = USERPRINCIPALNAME()",
            "[SalesPersonEmail] = CURRENTUSER()",
            "[SalesPersonEmail] = CUSTOMDATA()"
        ],
        "answer": 1,
        "explanation": "USERPRINCIPALNAME() returns the UPN (email address) of the currently authenticated Power BI user, which matches the format of work/school email addresses. USERNAME() returns the domain\\user format used in on-premises SSAS but is less reliable for Power BI Service. USERPRINCIPALNAME() is the correct function for dynamic RLS in the cloud."
    },
    {
        "id": 13,
        "domain": "Model the Data",
        "topic": "Calculated Columns vs Measures",
        "difficulty": "Easy",
        "question": "You need to calculate a profit margin percentage for each product row to use in a slicer. Should you use a calculated column or a measure?",
        "options": [
            "Measure — because it calculates dynamically based on filter context",
            "Calculated column — because it needs to evaluate per row and be used in a slicer",
            "Either; there is no functional difference",
            "Neither; use Power Query to add the column"
        ],
        "answer": 1,
        "explanation": "Slicers require a column, not a measure. Calculated columns compute row by row during data refresh and are stored in the model, making them suitable for slicers and filters. Measures compute dynamically at query time and cannot be directly used as slicer sources."
    },
    {
        "id": 14,
        "domain": "Model the Data",
        "topic": "DAX Context",
        "difficulty": "Hard",
        "question": "Inside an ADDCOLUMNS function, you use SUM(Sales[Amount]). What context does SUM operate in?",
        "options": [
            "The current filter context of the report visual",
            "The row context created by ADDCOLUMNS iterating over the table",
            "No context — SUM always aggregates the entire table",
            "The row context is automatically converted to filter context by ADDCOLUMNS"
        ],
        "answer": 3,
        "explanation": "ADDCOLUMNS is an iterator that creates a row context for each row of the table it iterates. When you call SUM() inside ADDCOLUMNS, DAX performs implicit context transition: the row context is automatically converted to an equivalent filter context, so SUM returns the value filtered to match that row's key values."
    },
    {
        "id": 15,
        "domain": "Model the Data",
        "topic": "Aggregations",
        "difficulty": "Hard",
        "question": "You configure a user-defined aggregation table in Power BI. For the aggregation to be hit by a query, which conditions must be true?",
        "options": [
            "The query grouping columns must map to columns in the aggregation table, and the aggregation measures must match the requested aggregation",
            "The aggregation table must be in DirectQuery mode while the detail table is in Import mode",
            "Aggregations only work with composite models where the detail table is in DirectQuery",
            "The aggregation table must have fewer than 1 million rows"
        ],
        "answer": 0,
        "explanation": "For a query to hit an aggregation table, the query's GROUP BY columns must all have corresponding mappings in the aggregation, and the requested measure aggregations must match what the aggregation table stores. If any column or aggregation is missing from the aggregation, the engine falls back to the detail table."
    },

    # ── Visualize and Analyze the Data (25-30%) ────────────────────────────────
    {
        "id": 16,
        "domain": "Visualize and Analyze",
        "topic": "Report Design",
        "difficulty": "Easy",
        "question": "A user reports that the report page takes too long to load. You notice the page has 15 visuals all querying the same large table. What is the best first action?",
        "options": [
            "Switch the dataset to DirectQuery mode",
            "Reduce the number of visuals or use bookmarks to show visuals on demand",
            "Increase the Power BI Premium capacity SKU",
            "Enable query caching for the dataset"
        ],
        "answer": 1,
        "explanation": "Each visual on a page sends a separate query. Reducing the number of visuals or hiding infrequently-used visuals behind bookmarks is the most impactful and immediate fix for slow page loads. Query caching helps for repeated loads but not the first load, and DirectQuery would likely make it slower."
    },
    {
        "id": 17,
        "domain": "Visualize and Analyze",
        "topic": "Conditional Formatting",
        "difficulty": "Medium",
        "question": "You want to color a table's revenue column red/yellow/green based on whether the value is below target, at target, or above target. The target varies by region. How do you achieve this?",
        "options": [
            "Use a static color scale in conditional formatting",
            "Write a DAX measure that returns a hex color code and apply it via 'Format by field value'",
            "Use a calculated column with IF() to return color names",
            "Apply a Power Query conditional column"
        ],
        "answer": 1,
        "explanation": "The 'Format by field value' option in conditional formatting allows a measure to return hex color codes (e.g., \"#FF0000\") dynamically based on context-aware logic. Since the target varies by region, a DAX measure can compare revenue to the correct regional target and return the appropriate color."
    },
    {
        "id": 18,
        "domain": "Visualize and Analyze",
        "topic": "AI Visuals",
        "difficulty": "Medium",
        "question": "A business user asks: 'Why did sales drop in Q3?' Which Power BI visual is designed to automatically identify the key factors driving a metric change?",
        "options": [
            "Decomposition Tree",
            "Key Influencers visual",
            "Smart Narrative",
            "Q&A visual"
        ],
        "answer": 1,
        "explanation": "The Key Influencers visual uses machine learning to automatically analyze a dataset and surface the top factors that increase or decrease a selected metric. It is ideal for explaining why a KPI changed. Decomposition Tree is useful for manual exploration; Smart Narrative generates text summaries."
    },
    {
        "id": 19,
        "domain": "Visualize and Analyze",
        "topic": "Filters",
        "difficulty": "Medium",
        "question": "You have a report with 5 pages. You add a filter on a visual's filter pane and set it to 'This page'. A user requests the same filter apply across all pages. What should you do?",
        "options": [
            "Recreate the filter on every page individually",
            "Change the filter scope from 'This page' to 'All pages' in the Filters pane",
            "Use a slicer with the 'Sync slicers' feature",
            "Both B and C are valid approaches"
        ],
        "answer": 3,
        "explanation": "Both approaches work. Changing a filter's scope to 'All pages' in the Filters pane applies it report-wide without a visual. Alternatively, synced slicers propagate user selections across pages and give users interactive control. The best choice depends on whether the filter should be user-adjustable."
    },
    {
        "id": 20,
        "domain": "Visualize and Analyze",
        "topic": "Drill-Through",
        "difficulty": "Medium",
        "question": "You create a drill-through page for a Product detail view. Which setting ensures that filters applied on the source page are automatically passed to the drill-through page?",
        "options": [
            "Enable 'Keep all filters' in the drill-through configuration",
            "Add every report filter as a drill-through field",
            "Use cross-report drill-through",
            "Publish the report to a Premium workspace"
        ],
        "answer": 0,
        "explanation": "The 'Keep all filters' toggle in the drill-through setup (on by default) passes the entire filter context from the source page to the drill-through page. When disabled, only the drill-through field's filter is applied."
    },
    {
        "id": 21,
        "domain": "Visualize and Analyze",
        "topic": "Performance Analyzer",
        "difficulty": "Medium",
        "question": "You use Performance Analyzer and observe a visual showing high 'Other' time but low 'DAX query' and 'Visual display' time. What does high 'Other' time most likely indicate?",
        "options": [
            "Slow DAX measures",
            "High rendering complexity in the visual",
            "Wait time due to other queries or report rendering overhead",
            "A network latency issue with the data source"
        ],
        "answer": 2,
        "explanation": "'Other' time in Performance Analyzer represents overhead such as waiting for other queries to complete, rendering the report shell, or browser processing. It is not directly related to DAX engine time or visual rendering. High 'Other' time often means too many visuals are competing for query execution slots."
    },
    {
        "id": 22,
        "domain": "Visualize and Analyze",
        "topic": "Bookmarks",
        "difficulty": "Easy",
        "question": "You create two bookmarks — one showing a bar chart and another showing a table — to let users switch views. Which bookmark setting should you update to ensure the bookmark only captures the visibility state of visuals, not filter states?",
        "options": [
            "Uncheck 'Data' in the bookmark options",
            "Uncheck 'Display' in the bookmark options",
            "Uncheck 'Current page' in the bookmark options",
            "Enable 'Selected visuals' mode"
        ],
        "answer": 0,
        "explanation": "Each bookmark can capture three states: Data (filters/slicers), Display (visual visibility), and Current page. Unchecking 'Data' makes the bookmark only toggle visibility without affecting any filter or slicer selections, which is the correct approach for show/hide toggle patterns."
    },
    {
        "id": 23,
        "domain": "Visualize and Analyze",
        "topic": "Decomposition Tree",
        "difficulty": "Easy",
        "question": "In the Decomposition Tree visual, what does the 'AI split — High value' option do?",
        "options": [
            "Automatically splits by the dimension that produces the highest absolute value",
            "Uses machine learning to find the dimension that best explains variance in the metric",
            "Sorts the existing split in descending order",
            "Enables drill-through for the highest-value node"
        ],
        "answer": 1,
        "explanation": "The AI split options ('High value' and 'Low value') in the Decomposition Tree use machine learning to automatically identify which dimension explains the most variance in the metric being analyzed. This saves analysts from manually trying every dimension."
    },

    # ── Deploy and Maintain Assets (20-25%) ────────────────────────────────────
    {
        "id": 24,
        "domain": "Deploy and Maintain",
        "topic": "Workspaces",
        "difficulty": "Medium",
        "question": "Your organization uses a Development, Test, and Production deployment pipeline. You need to allow developers to publish to Dev, testers to promote to Test, and only admins to promote to Production. How should you configure this?",
        "options": [
            "Create three separate workspaces and use deployment pipelines; assign roles per workspace",
            "Use a single workspace with item-level permissions",
            "Create three workspaces; give all users Admin access to all workspaces",
            "Use Power BI Embedded for environment separation"
        ],
        "answer": 0,
        "explanation": "The correct pattern is three workspaces connected in a deployment pipeline. Developers are Members/Contributors in the Dev workspace; testers have pipeline deploy rights to the Test stage; only admins have rights to promote to Production. This enforces environment separation and controlled promotion."
    },
    {
        "id": 25,
        "domain": "Deploy and Maintain",
        "topic": "Sensitivity Labels",
        "difficulty": "Medium",
        "question": "A dataset is tagged with the 'Confidential' sensitivity label in Power BI. A user exports the data to Excel. What happens to the sensitivity label?",
        "options": [
            "The label is removed because Excel is a different application",
            "The user is prompted to choose a new label in Excel",
            "The label is inherited by the exported Excel file automatically",
            "The export is blocked for Confidential data"
        ],
        "answer": 2,
        "explanation": "Microsoft Information Protection sensitivity labels are inherited downstream. When data tagged as 'Confidential' is exported to Excel (or other Office apps), the label and its associated protection settings are automatically applied to the exported file, maintaining data governance across the ecosystem."
    },
    {
        "id": 26,
        "domain": "Deploy and Maintain",
        "topic": "Incremental Refresh",
        "difficulty": "Hard",
        "question": "You configure incremental refresh with a 2-year historical window and a 3-day refresh window. What happens to data older than 2 years during each refresh?",
        "options": [
            "It is re-imported from the source on every refresh",
            "It is archived to Azure Data Lake automatically",
            "It is dropped from the dataset",
            "It remains in the dataset but is not refreshed"
        ],
        "answer": 2,
        "explanation": "The historical window defines the maximum age of data retained in the dataset. When a refresh runs, partitions older than the historical window boundary are automatically removed (dropped). This keeps the dataset size manageable while maintaining a rolling history."
    },
    {
        "id": 27,
        "domain": "Deploy and Maintain",
        "topic": "Gateway",
        "difficulty": "Medium",
        "question": "You have an on-premises SQL Server and need to refresh a Power BI dataset daily. You install the on-premises data gateway. What is the minimum gateway mode required for scheduled refresh of Import mode datasets?",
        "options": [
            "Personal mode gateway",
            "Standard mode (enterprise) gateway",
            "VPN gateway",
            "No gateway is needed for Import mode"
        ],
        "answer": 0,
        "explanation": "The Personal mode gateway is sufficient for scheduled refresh of Import mode datasets for a single user's own datasets. However, for enterprise scenarios (shared datasets, multiple users, DirectQuery), a Standard mode gateway is required. The question asks for the minimum, which is Personal mode."
    },
    {
        "id": 28,
        "domain": "Deploy and Maintain",
        "topic": "Endorsement",
        "difficulty": "Easy",
        "question": "What is the difference between 'Promoted' and 'Certified' endorsement in Power BI?",
        "options": [
            "Promoted is for reports; Certified is for datasets only",
            "Promoted can be set by dataset owners; Certified requires a tenant administrator or designated certifier",
            "Promoted requires Premium; Certified works on all capacities",
            "They are equivalent; Certified is the newer term for Promoted"
        ],
        "answer": 1,
        "explanation": "Any dataset owner or workspace member with sufficient rights can mark content as 'Promoted'. 'Certified' is a higher-trust designation that can only be granted by users specified by the Power BI administrator in tenant settings, ensuring certified content meets organizational standards."
    },
    {
        "id": 29,
        "domain": "Deploy and Maintain",
        "topic": "Sharing",
        "difficulty": "Medium",
        "question": "You share a report link with a colleague. They can view the report but cannot see the underlying dataset in the workspace. Why?",
        "options": [
            "They need Build permission on the dataset to access it",
            "They need to be a workspace Member to see the dataset",
            "The dataset must be published to a separate dataset workspace",
            "They need Power BI Pro to view shared datasets"
        ],
        "answer": 0,
        "explanation": "Sharing a report grants view access to that report only. To explore the dataset in tools like Excel Analyze in Excel, create new reports from it, or see it in the workspace, a user needs Build permission on the underlying dataset. Build permission can be granted separately from report sharing."
    },
    {
        "id": 30,
        "domain": "Deploy and Maintain",
        "topic": "XMLA Endpoint",
        "difficulty": "Hard",
        "question": "A data engineer wants to use Tabular Editor to deploy model changes directly to a Power BI dataset in the service. Which requirement must be met?",
        "options": [
            "The workspace must be on Premium capacity (or Premium Per User)",
            "The dataset must be in DirectQuery mode",
            "The dataset must have fewer than 1 billion rows",
            "The user must have Power BI Embedded A SKU"
        ],
        "answer": 0,
        "explanation": "The XMLA endpoint — which tools like Tabular Editor, SSMS, and DAX Studio connect to for read/write operations — is only available for workspaces on Premium capacity (P SKUs), Premium Per User (PPU), or Embedded A SKUs. It is not available for shared capacity (Pro) workspaces."
    },
    # ── Additional questions ──────────────────────────────────────────────────
    {
        "id": 31,
        "domain": "Prepare the Data",
        "topic": "Power Query",
        "difficulty": "Easy",
        "question": "You import a CSV file that has the actual column headers in row 2, not row 1. What Power Query step resolves this?",
        "options": [
            "Remove Top Rows, then Use First Row as Headers",
            "Transpose the table",
            "Use First Row as Headers twice",
            "Split the first column by delimiter"
        ],
        "answer": 0,
        "explanation": "Remove Top Rows (remove 1 row) deletes the unwanted first row, and then Use First Row as Headers promotes the now-first row (original row 2) to become column names. This is the standard pattern when headers are not in the first row."
    },
    {
        "id": 32,
        "domain": "Model the Data",
        "topic": "DAX",
        "difficulty": "Medium",
        "question": "You want a measure that returns blank instead of zero when there are no sales. Which expression achieves this?",
        "options": [
            "IF(SUM(Sales[Amount]) = 0, BLANK(), SUM(Sales[Amount]))",
            "IFERROR(SUM(Sales[Amount]), BLANK())",
            "SUM(Sales[Amount]) + 0",
            "COALESCE(SUM(Sales[Amount]), 0)"
        ],
        "answer": 0,
        "explanation": "IF(SUM(Sales[Amount]) = 0, BLANK(), SUM(Sales[Amount])) explicitly returns BLANK() when the sum is zero. In Power BI, BLANK() is the correct way to suppress zeros in visuals. IFERROR handles errors, not zeros. COALESCE replaces blanks with zeros — the opposite of what's needed."
    },
    {
        "id": 33,
        "domain": "Visualize and Analyze",
        "topic": "Q&A",
        "difficulty": "Easy",
        "question": "The Q&A visual returns incorrect results when a user types 'total revenue last year'. You want to ensure 'revenue' is recognized as the Sales[Amount] column. What should you configure?",
        "options": [
            "Rename the Sales[Amount] column to 'Revenue'",
            "Add 'revenue' as a synonym for the Sales[Amount] field in the Q&A setup",
            "Create a measure named 'Revenue'",
            "Enable featured tables for the Sales table"
        ],
        "answer": 1,
        "explanation": "The Q&A setup tool (accessible from the dataset settings) lets you add synonyms for field names. Adding 'revenue' as a synonym for Sales[Amount] teaches the Q&A engine to recognize that term without renaming the underlying column, which would affect all other reports."
    },
    {
        "id": 34,
        "domain": "Deploy and Maintain",
        "topic": "Deployment Pipelines",
        "difficulty": "Medium",
        "question": "In a deployment pipeline, you update a report in the Development stage and deploy it to Test. The Test stage already contains a different version of the report. What happens to the Test version?",
        "options": [
            "Both versions coexist in the Test workspace",
            "The Test version is overwritten by the Development version",
            "The deployment fails and you must manually delete the Test version",
            "A version history is created and the Test version is archived"
        ],
        "answer": 1,
        "explanation": "Deployment pipelines overwrite the target stage's content with the source stage's content. There is no versioning or coexistence — the Test version is replaced. This is why change management and approvals are important before deploying to Production."
    },
    {
        "id": 35,
        "domain": "Model the Data",
        "topic": "DAX",
        "difficulty": "Hard",
        "question": "You have Sales[Amount] and a disconnected 'What-If' parameter table with values 0.05 to 0.20. You want a measure that applies the selected discount rate to sales. Which measure is correct?",
        "options": [
            "Discounted Sales = SUM(Sales[Amount]) * Discount[Discount Value]",
            "Discounted Sales = SUM(Sales[Amount]) * SELECTEDVALUE(Discount[Discount Value], 0.1)",
            "Discounted Sales = SUM(Sales[Amount]) * MAX(Discount[Discount Value])",
            "Discounted Sales = SUMX(Sales, Sales[Amount] * Discount[Discount Value])"
        ],
        "answer": 1,
        "explanation": "SELECTEDVALUE() returns the single value selected in the slicer, or the default (0.1) if no selection or multiple selections are made. This is the correct and safe pattern for What-If parameters. Using MAX() works but doesn't handle the no-selection case gracefully. Direct column multiplication won't work on disconnected tables."
    },
]

# ── Flashcards ─────────────────────────────────────────────────────────────────

FLASHCARDS = [
    {"term": "Star Schema", "definition": "A data model design with a central fact table surrounded by denormalized dimension tables. Preferred in Power BI for optimal DAX performance and simplicity.", "domain": "Model the Data"},
    {"term": "Query Folding", "definition": "The ability of Power Query to translate transformation steps into native source queries (e.g., SQL). Improves performance by processing data at the source. Breaks when using operations like adding index columns or custom M functions.", "domain": "Prepare the Data"},
    {"term": "CALCULATE()", "definition": "The most powerful DAX function. Evaluates an expression in a modified filter context. Accepts filter arguments that add, remove, or override filters. Triggers context transition when row context exists.", "domain": "Model the Data"},
    {"term": "Context Transition", "definition": "The automatic conversion of row context into an equivalent filter context when a measure (or aggregation function with CALCULATE) is called inside an iterator.", "domain": "Model the Data"},
    {"term": "USERELATIONSHIP()", "definition": "DAX function used inside CALCULATE to activate an inactive relationship for the duration of a calculation. Used when a table has multiple relationships to another table (e.g., OrderDate and ShipDate to Date).", "domain": "Model the Data"},
    {"term": "Incremental Refresh", "definition": "Divides a dataset into date-based partitions. Only the most recent 'refresh window' partition is refreshed each time, while historical partitions remain untouched. Requires RangeStart and RangeEnd Power Query parameters.", "domain": "Deploy and Maintain"},
    {"term": "Row-Level Security (RLS)", "definition": "Restricts data access for specific users by defining DAX filter rules on tables within roles. Can be static (fixed filters) or dynamic (using USERPRINCIPALNAME()).", "domain": "Deploy and Maintain"},
    {"term": "Dataflow", "definition": "A Power BI artifact that stores Power Query transformations in the service (Azure Data Lake Gen2). Multiple datasets can consume a single Dataflow, centralizing data preparation logic.", "domain": "Prepare the Data"},
    {"term": "XMLA Endpoint", "definition": "An Analysis Services-compatible endpoint available in Premium/PPU workspaces. Enables read/write access to datasets using tools like SSMS, Tabular Editor, and DAX Studio.", "domain": "Deploy and Maintain"},
    {"term": "SELECTEDVALUE()", "definition": "Returns the value of a column when it is filtered to exactly one distinct value; otherwise returns the alternate result. Ideal for What-If parameters and single-selection scenarios.", "domain": "Model the Data"},
    {"term": "Composite Model", "definition": "A Power BI model that combines Import and DirectQuery tables, or tables from multiple sources including Power BI datasets and Azure Analysis Services, within a single model.", "domain": "Model the Data"},
    {"term": "Deployment Pipeline", "definition": "A Power BI Premium feature that provides a Dev → Test → Prod workflow for promoting content between workspaces in a structured, controlled manner.", "domain": "Deploy and Maintain"},
    {"term": "Build Permission", "definition": "A dataset permission that allows users to create new reports, dashboards, and connect via Analyze in Excel. Separate from workspace roles; granted per dataset.", "domain": "Deploy and Maintain"},
    {"term": "TOTALYTD() / DATESYTD()", "definition": "Time-intelligence functions for year-to-date calculations. TOTALYTD wraps an expression; DATESYTD returns the date range. Both accept a fiscal year-end date (e.g., '06-30').", "domain": "Model the Data"},
    {"term": "Performance Analyzer", "definition": "A Power BI Desktop tool that measures the rendering time of each visual, broken down into DAX query, visual display, and other components. Used to identify performance bottlenecks.", "domain": "Visualize and Analyze"},
    {"term": "Sensitivity Labels", "definition": "Microsoft Information Protection labels (e.g., Confidential, Public) applied to Power BI content. Labels are inherited by downstream exports (Excel, CSV, etc.) maintaining data governance.", "domain": "Deploy and Maintain"},
    {"term": "ALLEXCEPT()", "definition": "Removes all filters from a table except for filters on specified columns. Often used in DAX measures to calculate totals or percentages while keeping certain context dimensions active.", "domain": "Model the Data"},
    {"term": "Decomposition Tree", "definition": "An AI-powered visualization that enables users to drill down across multiple dimensions to analyze a metric. Offers AI splits (High value/Low value) to automatically find the most explanatory dimension.", "domain": "Visualize and Analyze"},
    {"term": "Key Influencers Visual", "definition": "An AI visual that uses machine learning to identify and rank the factors (columns) that most influence a selected metric increasing or decreasing.", "domain": "Visualize and Analyze"},
    {"term": "TREATAS()", "definition": "Applies the values of a table expression as filters to columns in an unrelated table, effectively creating a virtual relationship. Useful for disconnected tables or complex filter propagation.", "domain": "Model the Data"},
]

# ── Study Guide Content ────────────────────────────────────────────────────────

STUDY_GUIDE = {
    "Prepare the Data (25-30%)": {
        "color": "#0078D4",
        "weight": "25-30%",
        "subtopics": [
            "**Get data from different sources** — Files (Excel, CSV, JSON), databases (SQL Server, Azure SQL, Oracle), online services, Power BI datasets, Dataflows",
            "**Profile the data** — Column distribution, column quality, column profile in Power Query Editor",
            "**Clean and transform data** — Remove duplicates, handle errors, unpivot/pivot, split columns, change data types",
            "**Combine queries** — Append (stacking rows) vs Merge (joining columns); join kinds (inner, left outer, right outer, full outer, anti)",
            "**Query folding** — Understand which steps fold; index columns, custom functions, and certain operations break folding",
            "**Parameters and templates** — Power Query parameters for dynamic source paths, query templates (.pqt files)",
            "**Dataflows** — Create, consume, incremental refresh, certified dataflows in Premium",
            "**Advanced M features** — List functions, Table functions, #shared library, try/otherwise error handling",
        ]
    },
    "Model the Data (25-30%)": {
        "color": "#107C10",
        "weight": "25-30%",
        "subtopics": [
            "**Star schema design** — Fact tables (measures, foreign keys), dimension tables (descriptive attributes), snowflake vs star",
            "**Relationships** — Active vs inactive, cardinality (1:M, M:M, 1:1), cross-filter direction, USERELATIONSHIP()",
            "**DAX fundamentals** — Row context vs filter context, context transition, CALCULATE(), iterators (SUMX, AVERAGEX)",
            "**Time intelligence** — TOTALYTD, SAMEPERIODLASTYEAR, DATEADD, DATESYTD, DATESBETWEEN; require a proper Date table marked as Date table",
            "**Calculated columns vs measures** — Columns stored in model, computed per row; measures computed at query time in filter context",
            "**Row-Level Security** — Static roles, dynamic RLS with USERPRINCIPALNAME(), OLS (object-level security) in Premium",
            "**Performance optimization** — Avoid bi-directional relationships, reduce cardinality, use aggregations, format columns appropriately",
            "**Aggregations** — User-defined aggregation tables, automatic aggregations (Premium), aggregation precedence",
            "**Composite models** — Mixing Import/DirectQuery, chaining to Power BI datasets/AAS",
        ]
    },
    "Visualize and Analyze the Data (25-30%)": {
        "color": "#8764B8",
        "weight": "25-30%",
        "subtopics": [
            "**Choose the right visual** — Bar/column for comparisons, line for trends, scatter for correlations, map for geography, matrix for cross-tab",
            "**Conditional formatting** — Background color, font color, data bars, icons; format by rules or field value (hex color measure)",
            "**Report interactivity** — Cross-filtering vs cross-highlighting, edit interactions, drill-through, drill-down",
            "**Filters pane** — Visual, page, report, drillthrough filter scopes; hide/lock filters for consumers",
            "**Bookmarks** — Capture data, display, current page states; use for navigation, show/hide toggles",
            "**Slicers** — Sync slicers across pages, slicer types (list, dropdown, between, relative date)",
            "**AI visuals** — Key Influencers, Decomposition Tree, Smart Narrative, Q&A; when to use each",
            "**Performance** — Performance Analyzer (DAX query, visual display, Other); reduce visual count, use aggregations",
            "**Accessibility** — Alt text, tab order, color contrast, screen reader support",
            "**Mobile layouts** — Canvas redesign for phone view, responsive/static layouts",
        ]
    },
    "Deploy and Maintain Assets (20-25%)": {
        "color": "#D83B01",
        "weight": "20-25%",
        "subtopics": [
            "**Workspaces** — Create, roles (Admin, Member, Contributor, Viewer), Classic vs new workspace",
            "**Deployment pipelines** — Dev/Test/Prod stages, deployment rules, parameterized connections",
            "**Sharing and permissions** — Share reports, Build permission on datasets, workspace access vs item access",
            "**Endorsement** — Promoted (by owner) vs Certified (by designated certifier); discoverability in data hub",
            "**Sensitivity labels** — MIP labels inherit to exports; requires Azure Information Protection integration",
            "**Gateway** — Personal mode (single user, Import only) vs Standard mode (enterprise, DirectQuery/Scheduled); gateway clusters",
            "**Scheduled refresh** — Up to 8x/day (Pro), 48x/day (Premium); incremental refresh partitions",
            "**XMLA endpoint** — Premium/PPU only; enables Tabular Editor, SSMS, DAX Studio read/write access",
            "**Apps** — Package workspace content into an App; audience-based permissions, custom navigation",
            "**Monitoring** — Activity log, audit logs, usage metrics; monitoring hub in Premium",
        ]
    },
}

# ── Session State Init ─────────────────────────────────────────────────────────

defaults = {
    "quiz_active": False,
    "quiz_questions": [],
    "quiz_index": 0,
    "quiz_answers": [],  # list of True/False
    "quiz_selected": None,
    "quiz_submitted": False,
    "score_history": [],
    "fc_index": 0,
    "fc_revealed": False,
    "fc_domain_filter": "All",
    "quiz_domain_filter": "All",
    "quiz_difficulty_filter": "All",
    "quiz_size": 10,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────

DOMAINS = ["All", "Prepare the Data", "Model the Data", "Visualize and Analyze", "Deploy and Maintain"]
DIFFICULTIES = ["All", "Easy", "Medium", "Hard"]

def get_filtered_questions(domain="All", difficulty="All"):
    qs = QUESTIONS
    if domain != "All":
        qs = [q for q in qs if q["domain"] == domain]
    if difficulty != "All":
        qs = [q for q in qs if q["difficulty"] == difficulty]
    return qs

def start_quiz(domain, difficulty, size):
    pool = get_filtered_questions(domain, difficulty)
    if not pool:
        return False
    sample = random.sample(pool, min(size, len(pool)))
    st.session_state.quiz_active = True
    st.session_state.quiz_questions = sample
    st.session_state.quiz_index = 0
    st.session_state.quiz_answers = []
    st.session_state.quiz_selected = None
    st.session_state.quiz_submitted = False
    return True

def submit_answer():
    if st.session_state.quiz_selected is None:
        return
    q = st.session_state.quiz_questions[st.session_state.quiz_index]
    correct = st.session_state.quiz_selected == q["answer"]
    st.session_state.quiz_answers.append(correct)
    st.session_state.quiz_submitted = True

def next_question():
    st.session_state.quiz_index += 1
    st.session_state.quiz_selected = None
    st.session_state.quiz_submitted = False

def finish_quiz():
    total = len(st.session_state.quiz_answers)
    correct = sum(st.session_state.quiz_answers)
    pct = round(correct / total * 100) if total else 0
    st.session_state.score_history.append({
        "date": datetime.now().strftime("%b %d %H:%M"),
        "score": pct,
        "correct": correct,
        "total": total,
        "domain": st.session_state.quiz_domain_filter,
    })
    st.session_state.quiz_active = False
    st.session_state.quiz_questions = []

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 PL-300 Exam Prep")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📝 Practice Quiz", "🃏 Flashcards", "📚 Study Guide", "📈 My Progress"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    if st.session_state.score_history:
        last = st.session_state.score_history[-1]
        st.metric("Last Score", f"{last['score']}%", f"{last['correct']}/{last['total']}")
        avg = round(sum(h["score"] for h in st.session_state.score_history) / len(st.session_state.score_history))
        st.metric("Avg Score", f"{avg}%")

    st.markdown("---")
    st.markdown("""
**Exam at a Glance**
- Duration: 100 minutes
- Questions: ~40-60
- Passing: ~700/1000
- Format: Multiple choice, case studies, drag-and-drop
    """)

# ── Pages ──────────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>📊 PL-300 Power BI Exam Prep</h1>
        <p>Microsoft Power BI Data Analyst Certification</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="exam-info">
        <strong>📋 Exam PL-300:</strong> Microsoft Power BI Data Analyst &nbsp;|&nbsp;
        <strong>⏱ Duration:</strong> 100 minutes &nbsp;|&nbsp;
        <strong>✅ Passing Score:</strong> ~700/1000 &nbsp;|&nbsp;
        <strong>❓ Questions:</strong> 40-60
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Questions", len(QUESTIONS))
    with col2:
        st.metric("🃏 Flashcards", len(FLASHCARDS))
    with col3:
        sessions = len(st.session_state.score_history)
        st.metric("🎯 Sessions Taken", sessions)
    with col4:
        if st.session_state.score_history:
            best = max(h["score"] for h in st.session_state.score_history)
            st.metric("🏆 Best Score", f"{best}%")
        else:
            st.metric("🏆 Best Score", "—")

    st.markdown("---")
    st.subheader("Exam Domain Breakdown")

    domain_data = [
        ("Prepare the Data", "25-30%", "#0078D4", "Connect to sources, clean and transform data in Power Query, configure dataflows, understand query folding"),
        ("Model the Data", "25-30%", "#107C10", "Design star schemas, write DAX measures, configure RLS, optimize model performance"),
        ("Visualize & Analyze", "25-30%", "#8764B8", "Build reports, choose visuals, implement interactivity, use AI features, optimize performance"),
        ("Deploy & Maintain", "20-25%", "#D83B01", "Manage workspaces, deployment pipelines, security, gateway, scheduled refresh"),
    ]

    for name, weight, color, desc in domain_data:
        domain_qs = [q for q in QUESTIONS if q["domain"] in name or name in q["domain"]]
        st.markdown(f"""
        <div class="domain-card">
            <h4 style="color:{color}">● {name} — {weight}</h4>
            <p style="margin:0;font-size:0.9rem;color:#333">{desc}</p>
            <p class="progress-text">{len(domain_qs)} practice questions available</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Quick Start")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**📝 Practice Quiz**\nTest yourself with timed multiple-choice questions. Review explanations after each answer.")
    with col_b:
        st.info("**🃏 Flashcards**\nReview key terms and concepts. Filter by domain for targeted study.")
    with col_c:
        st.info("**📚 Study Guide**\nDetailed notes on every exam domain with key concepts and exam tips.")


elif page == "📝 Practice Quiz":
    st.markdown("""
    <div class="main-header">
        <h1>📝 Practice Quiz</h1>
        <p>Test your knowledge with real exam-style questions</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.quiz_active:
        st.subheader("Configure Your Quiz")

        col1, col2, col3 = st.columns(3)
        with col1:
            domain = st.selectbox("Domain", DOMAINS, key="quiz_domain_filter")
        with col2:
            difficulty = st.selectbox("Difficulty", DIFFICULTIES, key="quiz_difficulty_filter")
        with col3:
            available = len(get_filtered_questions(
                st.session_state.quiz_domain_filter,
                st.session_state.quiz_difficulty_filter
            ))
            size = st.slider("Questions", 5, min(35, max(5, available)), min(10, available), key="quiz_size")

        st.info(f"**{available}** questions match your filters.")

        if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
            if start_quiz(
                st.session_state.quiz_domain_filter,
                st.session_state.quiz_difficulty_filter,
                st.session_state.quiz_size
            ):
                st.rerun()
            else:
                st.error("No questions match the selected filters.")

        if st.session_state.score_history:
            st.markdown("---")
            st.subheader("Recent Sessions")
            for h in reversed(st.session_state.score_history[-5:]):
                color = "#28a745" if h["score"] >= 70 else "#ffc107" if h["score"] >= 50 else "#dc3545"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:0.6rem 1rem;border:1px solid #dee2e6;border-radius:6px;margin-bottom:0.4rem">
                    <span>{h['date']} — {h['domain']}</span>
                    <span style="font-weight:bold;color:{color}">{h['score']}% ({h['correct']}/{h['total']})</span>
                </div>
                """, unsafe_allow_html=True)

    else:
        questions = st.session_state.quiz_questions
        idx = st.session_state.quiz_index

        if idx >= len(questions):
            finish_quiz()
            st.rerun()
        else:
            q = questions[idx]
            progress = idx / len(questions)
            st.progress(progress, text=f"Question {idx + 1} of {len(questions)}")

            st.markdown(f"""
            <div class="question-card">
                <span class="topic-badge">{q['domain']}</span>
                <span style="margin-left:0.5rem;font-size:0.8rem;color:#666">{q['topic']} · {q['difficulty']}</span>
                <h4>Q{idx + 1}. {q['question']}</h4>
            </div>
            """, unsafe_allow_html=True)

            if not st.session_state.quiz_submitted:
                for i, opt in enumerate(q["options"]):
                    selected = st.session_state.quiz_selected == i
                    btn_label = f"{'◉' if selected else '○'}  {opt}"
                    if st.button(btn_label, key=f"opt_{idx}_{i}", use_container_width=True):
                        st.session_state.quiz_selected = i
                        st.rerun()

                col_sub, col_skip = st.columns([3, 1])
                with col_sub:
                    disabled = st.session_state.quiz_selected is None
                    if st.button("✅ Submit Answer", type="primary", use_container_width=True, disabled=disabled):
                        submit_answer()
                        st.rerun()
                with col_skip:
                    if st.button("⏭ Skip", use_container_width=True):
                        st.session_state.quiz_answers.append(False)
                        next_question()
                        st.rerun()

            else:
                correct = st.session_state.quiz_answers[-1]
                selected_idx = st.session_state.quiz_selected

                for i, opt in enumerate(q["options"]):
                    if i == q["answer"]:
                        st.markdown(f'<div class="correct-answer">✅ <strong>{opt}</strong> — Correct answer</div>', unsafe_allow_html=True)
                    elif i == selected_idx and not correct:
                        st.markdown(f'<div class="wrong-answer">❌ <strong>{opt}</strong> — Your answer</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='padding:0.5rem 1rem;margin:0.3rem 0;border-radius:4px;background:#f8f9fa;color:#888'>{opt}</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="explanation-box">
                    <strong>💡 Explanation:</strong><br>{q['explanation']}
                </div>
                """, unsafe_allow_html=True)

                running_correct = sum(st.session_state.quiz_answers)
                running_total = len(st.session_state.quiz_answers)
                st.markdown(f"**Running score: {running_correct}/{running_total} ({round(running_correct/running_total*100)}%)**")

                col_next, col_end = st.columns([3, 1])
                with col_next:
                    label = "Next Question ➡" if idx + 1 < len(questions) else "🏁 Finish Quiz"
                    if st.button(label, type="primary", use_container_width=True):
                        if idx + 1 >= len(questions):
                            finish_quiz()
                        else:
                            next_question()
                        st.rerun()
                with col_end:
                    if st.button("🛑 End Quiz", use_container_width=True):
                        finish_quiz()
                        st.rerun()


elif page == "🃏 Flashcards":
    st.markdown("""
    <div class="main-header">
        <h1>🃏 Flashcards</h1>
        <p>Review key concepts and terminology</p>
    </div>
    """, unsafe_allow_html=True)

    fc_domains = ["All"] + sorted(set(f["domain"] for f in FLASHCARDS))
    domain_filter = st.selectbox("Filter by Domain", fc_domains, key="fc_domain_filter")

    cards = FLASHCARDS if domain_filter == "All" else [f for f in FLASHCARDS if f["domain"] == domain_filter]

    if not cards:
        st.warning("No flashcards for this domain.")
    else:
        idx = st.session_state.fc_index % len(cards)
        card = cards[idx]

        st.markdown(f"**Card {idx + 1} of {len(cards)}** — {card['domain']}")

        st.markdown(f"""
        <div class="flashcard-front">
            {card['term']}
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.fc_revealed:
            st.markdown(f"""
            <div class="flashcard-back">
                <strong>Definition:</strong><br><br>{card['definition']}
            </div>
            """, unsafe_allow_html=True)
            btn_label = "🙈 Hide Answer"
        else:
            btn_label = "👁 Reveal Answer"

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅ Previous", use_container_width=True):
                st.session_state.fc_index = (st.session_state.fc_index - 1) % len(cards)
                st.session_state.fc_revealed = False
                st.rerun()
        with col2:
            if st.button(btn_label, use_container_width=True, type="primary"):
                st.session_state.fc_revealed = not st.session_state.fc_revealed
                st.rerun()
        with col3:
            if st.button("Next ➡", use_container_width=True):
                st.session_state.fc_index = (st.session_state.fc_index + 1) % len(cards)
                st.session_state.fc_revealed = False
                st.rerun()

        st.markdown("---")
        if st.button("🔀 Shuffle", use_container_width=True):
            random.shuffle(cards)
            st.session_state.fc_index = 0
            st.session_state.fc_revealed = False
            st.rerun()

        st.markdown("---")
        st.subheader("All Cards at a Glance")
        for c in cards:
            with st.expander(f"**{c['term']}** — {c['domain']}"):
                st.write(c["definition"])


elif page == "📚 Study Guide":
    st.markdown("""
    <div class="main-header">
        <h1>📚 Study Guide</h1>
        <p>Comprehensive notes for every exam domain</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="exam-info">
        <strong>💡 Exam Tips:</strong>
        All four domains are weighted roughly equally (20-30% each). Focus on DAX (measures vs calculated columns, CALCULATE,
        time intelligence), Power Query (query folding, merge vs append, M language), and deployment topics
        (RLS, incremental refresh, XMLA, gateways) as these are heavily tested.
    </div>
    """, unsafe_allow_html=True)

    for domain, content in STUDY_GUIDE.items():
        with st.expander(f"**{domain}**", expanded=False):
            st.markdown(f"**Exam Weight: {content['weight']}**")
            st.markdown("---")
            for item in content["subtopics"]:
                st.markdown(f"• {item}")

    st.markdown("---")
    st.subheader("⚡ High-Priority DAX Functions")

    dax_functions = [
        ("CALCULATE(expr, filters)", "Core filter context modifier. Master this first."),
        ("SUMX(table, expr)", "Iterator — evaluates expr for each row, returns sum."),
        ("FILTER(table, condition)", "Returns a filtered table. Often used inside CALCULATE."),
        ("ALL(table/column)", "Removes all filters from a table or column."),
        ("ALLEXCEPT(table, col1, ...)", "Removes all filters except specified columns."),
        ("USERELATIONSHIP(col1, col2)", "Activates an inactive relationship in a calculation."),
        ("RELATED(column)", "Returns related value from a many-to-one (dimension) table."),
        ("RELATEDTABLE(table)", "Returns related rows from a one-to-many (fact) table."),
        ("SELECTEDVALUE(col, alt)", "Returns the single selected value or alternate result."),
        ("RANKX(table, expr, ...)", "Ranks rows in a table based on an expression."),
        ("DIVIDE(num, denom, alt)", "Safe division — returns alt (default BLANK) on divide-by-zero."),
        ("DATEADD(dates, n, interval)", "Shifts a date range by n intervals (DAY, MONTH, QUARTER, YEAR)."),
        ("SAMEPERIODLASTYEAR(dates)", "Returns the same period from the prior year."),
        ("TOTALYTD(expr, dates, ...)", "Year-to-date calculation; accepts optional fiscal year-end."),
        ("USERPRINCIPALNAME()", "Returns the current user's UPN email — used in dynamic RLS."),
        ("PATH(id, parentId)", "Generates ancestor path string for parent-child hierarchies."),
        ("TREATAS(table, col1, ...)", "Applies table values as filters to an unrelated table."),
        ("VAR / RETURN", "Stores intermediate results to improve readability and performance."),
    ]

    col1, col2 = st.columns(2)
    for i, (func, desc) in enumerate(dax_functions):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;
                        padding:0.7rem 1rem;margin-bottom:0.5rem">
                <code style="color:#0078D4;font-size:0.85rem">{func}</code><br>
                <span style="font-size:0.85rem;color:#333">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔑 Key Power Query Tips")
    pq_tips = [
        "Use **Remove Columns** early to reduce data volume before other transformations.",
        "**Merge Queries** = JOIN (horizontal); **Append Queries** = UNION (vertical).",
        "**Adding an Index Column** breaks query folding — do it as late as possible.",
        "RangeStart and RangeEnd parameters (DateTime type) are required for incremental refresh.",
        "**Table.TransformColumnTypes** and filters generally fold against SQL sources.",
        "Use **Group By** to pre-aggregate large tables before loading to the model.",
        "**Custom functions** (let … in …) are reusable M code — great for repeated patterns.",
        "Check query folding by right-clicking a step: 'View Native Query' (grayed out = no folding).",
    ]
    for tip in pq_tips:
        st.markdown(f"• {tip}")


elif page == "📈 My Progress":
    st.markdown("""
    <div class="main-header">
        <h1>📈 My Progress</h1>
        <p>Track your study sessions and improvement</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.score_history:
        st.info("No quiz sessions yet. Complete a practice quiz to see your progress here.")
        if st.button("🚀 Start a Quiz", type="primary"):
            st.session_state["_nav"] = "📝 Practice Quiz"
            st.rerun()
    else:
        history = st.session_state.score_history
        scores = [h["score"] for h in history]
        total_q = sum(h["total"] for h in history)
        total_correct = sum(h["correct"] for h in history)
        avg_score = round(sum(scores) / len(scores))
        best_score = max(scores)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sessions", len(history))
        with col2:
            st.metric("Questions Answered", total_q)
        with col3:
            st.metric("Average Score", f"{avg_score}%")
        with col4:
            st.metric("Best Score", f"{best_score}%")

        st.markdown("---")
        st.subheader("Score History")
        st.line_chart({"Score (%)": scores})

        st.markdown("---")
        st.subheader("All Sessions")
        for i, h in enumerate(reversed(history)):
            pct = h["score"]
            color = "#28a745" if pct >= 70 else "#ffc107" if pct >= 50 else "#dc3545"
            badge = "✅ Pass" if pct >= 70 else "⚠️ Near Pass" if pct >= 50 else "❌ Needs Work"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                padding:0.8rem 1.2rem;border:1px solid #dee2e6;border-radius:8px;margin-bottom:0.5rem">
                <div>
                    <strong>Session {len(history) - i}</strong> &nbsp;·&nbsp; {h['date']} &nbsp;·&nbsp;
                    <em>{h['domain']}</em>
                </div>
                <div style="text-align:right">
                    <span style="font-size:1.2rem;font-weight:bold;color:{color}">{pct}%</span>
                    &nbsp; <span style="font-size:0.8rem;color:{color}">{badge}</span>
                    <br><span style="font-size:0.8rem;color:#666">{h['correct']}/{h['total']} correct</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑 Clear History", type="secondary"):
            st.session_state.score_history = []
            st.rerun()

        st.markdown("---")
        st.subheader("Readiness Assessment")
        if avg_score >= 75:
            st.success(f"🎉 **Excellent!** Your average score of {avg_score}% suggests you are well-prepared for the PL-300 exam. Keep practicing to stay sharp!")
        elif avg_score >= 65:
            st.warning(f"📚 **Good progress!** Your average score of {avg_score}% is getting close to passing (~70%). Focus on your weaker domains.")
        else:
            st.error(f"📖 **Keep studying.** Your average score of {avg_score}% indicates more preparation is needed. Review the Study Guide and use Flashcards for key concepts.")

        # Domain breakdown
        st.markdown("---")
        st.subheader("Performance by Domain")
        domain_scores = {}
        domain_counts = {}
        for h in history:
            d = h["domain"]
            if d not in domain_scores:
                domain_scores[d] = []
                domain_counts[d] = 0
            domain_scores[d].append(h["score"])
            domain_counts[d] += 1

        if domain_scores:
            dcol1, dcol2 = st.columns(2)
            for i, (d, scores_list) in enumerate(domain_scores.items()):
                col = dcol1 if i % 2 == 0 else dcol2
                with col:
                    avg = round(sum(scores_list) / len(scores_list))
                    color = "#28a745" if avg >= 70 else "#ffc107" if avg >= 50 else "#dc3545"
                    st.markdown(f"""
                    <div style="background:#f8f9fa;border-left:4px solid {color};
                        padding:0.8rem 1rem;border-radius:6px;margin-bottom:0.5rem">
                        <strong>{d}</strong><br>
                        <span style="font-size:1.3rem;font-weight:bold;color:{color}">{avg}%</span>
                        <span style="font-size:0.8rem;color:#666"> avg over {domain_counts[d]} session(s)</span>
                    </div>
                    """, unsafe_allow_html=True)
