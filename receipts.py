import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re
from pathlib import Path
import io

# For OCR - you'll need to install: pip install pytesseract pillow pdf2image
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    st.warning("OCR libraries not installed. Install with: pip install pytesseract pillow pdf2image")

# Page config
st.set_page_config(
    page_title="Receipt Analyzer",
    page_icon="🧾",
    layout="wide"
)

# Initialize session state for storing receipts
if 'receipts' not in st.session_state:
    st.session_state.receipts = []

# Helper Functions
def extract_text_from_image(image):
    """Extract text from image using OCR"""
    if not OCR_AVAILABLE:
        return "OCR not available. Please install required libraries."
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def parse_receipt_text(text):
    """Parse receipt text to extract relevant information"""
    data = {
        'date': None,
        'merchant': None,
        'items': [],
        'total': None,
        'category': 'Uncategorized'
    }
    
    # Extract date (various formats)
    date_patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            try:
                date_str = date_match.group()
                # Try to parse the date
                for fmt in ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y']:
                    try:
                        data['date'] = datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                        break
                    except:
                        continue
            except:
                pass
            if data['date']:
                break
    
    # Extract total amount
    total_patterns = [
        r'total[:\s]+\$?(\d+\.?\d*)',
        r'amount[:\s]+\$?(\d+\.?\d*)',
        r'\$(\d+\.?\d*)\s*total'
    ]
    for pattern in total_patterns:
        total_match = re.search(pattern, text, re.IGNORECASE)
        if total_match:
            try:
                data['total'] = float(total_match.group(1))
                break
            except:
                pass
    
    # Extract merchant (usually first few lines)
    lines = text.split('\n')
    for line in lines[:5]:
        if len(line.strip()) > 3 and not re.match(r'^\d', line):
            data['merchant'] = line.strip()
            break
    
    # Categorize based on merchant name
    if data['merchant']:
        merchant_lower = data['merchant'].lower()
        if any(word in merchant_lower for word in ['walmart', 'target', 'costco', 'grocery', 'market', 'food']):
            data['category'] = 'Groceries'
        elif any(word in merchant_lower for word in ['restaurant', 'cafe', 'coffee', 'pizza', 'burger']):
            data['category'] = 'Dining'
        elif any(word in merchant_lower for word in ['gas', 'fuel', 'shell', 'exxon', 'chevron']):
            data['category'] = 'Transportation'
        elif any(word in merchant_lower for word in ['amazon', 'ebay', 'store', 'shop']):
            data['category'] = 'Shopping'
        elif any(word in merchant_lower for word in ['pharmacy', 'cvs', 'walgreens', 'health']):
            data['category'] = 'Health'
    
    return data

def add_receipt(receipt_data):
    """Add receipt to session state"""
    receipt_data['id'] = len(st.session_state.receipts)
    receipt_data['upload_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.receipts.append(receipt_data)

def get_receipts_df():
    """Convert receipts to DataFrame"""
    if not st.session_state.receipts:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.receipts)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

def calculate_insights(df):
    """Calculate spending insights"""
    insights = {}
    
    if df.empty:
        return insights
    
    # Total spending
    insights['total_spending'] = df['total'].sum()
    
    # Average per transaction
    insights['avg_transaction'] = df['total'].mean()
    
    # Spending by category
    insights['category_spending'] = df.groupby('category')['total'].sum().to_dict()
    
    # Most frequent merchant
    if 'merchant' in df.columns:
        insights['top_merchant'] = df['merchant'].mode()[0] if not df['merchant'].mode().empty else 'N/A'
    
    # Recent trends (last 30 days vs previous 30 days)
    if 'date' in df.columns and not df['date'].isna().all():
        now = datetime.now()
        last_30 = df[df['date'] >= (now - timedelta(days=30))]['total'].sum()
        prev_30 = df[(df['date'] >= (now - timedelta(days=60))) & 
                     (df['date'] < (now - timedelta(days=30)))]['total'].sum()
        
        if prev_30 > 0:
            insights['spending_change'] = ((last_30 - prev_30) / prev_30) * 100
        else:
            insights['spending_change'] = 0
    
    return insights

# Main App
st.title("🧾 Receipt Analyzer")
st.markdown("Upload your receipts and get real-time insights into your spending habits!")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Upload Receipts", "Dashboard", "Detailed Analysis", "Settings"])
    
    st.divider()
    st.metric("Total Receipts", len(st.session_state.receipts))
    if st.session_state.receipts:
        total = sum(r['total'] for r in st.session_state.receipts if r['total'])
        st.metric("Total Spent", f"${total:,.2f}")
    
    st.divider()
    if st.button("Clear All Data", type="secondary"):
        st.session_state.receipts = []
        st.rerun()

# Page: Upload Receipts
if page == "Upload Receipts":
    st.header("📤 Upload Receipts")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload receipt images or PDFs",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.subheader(f"Processing: {uploaded_file.name}")
                
                # Display image
                if uploaded_file.type.startswith('image'):
                    image = Image.open(uploaded_file)
                    st.image(image, width=300)
                    
                    if OCR_AVAILABLE:
                        with st.spinner("Extracting text..."):
                            text = extract_text_from_image(image)
                            
                        with st.expander("View extracted text"):
                            st.text(text)
                        
                        # Parse receipt
                        receipt_data = parse_receipt_text(text)
                    else:
                        receipt_data = {'merchant': None, 'date': None, 'total': None, 'category': 'Uncategorized'}
                        st.info("OCR not available. Please enter details manually.")
                else:
                    st.info("PDF processing requires OCR libraries")
                    receipt_data = {'merchant': None, 'date': None, 'total': None, 'category': 'Uncategorized'}
                
                # Manual entry form
                with st.form(f"receipt_form_{uploaded_file.name}"):
                    st.subheader("Verify/Edit Receipt Details")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        merchant = st.text_input("Merchant", value=receipt_data.get('merchant', '') or '')
                        date = st.date_input("Date", value=datetime.strptime(receipt_data['date'], '%Y-%m-%d') if receipt_data.get('date') else datetime.now())
                    
                    with col_b:
                        total = st.number_input("Total Amount ($)", min_value=0.0, value=float(receipt_data.get('total') or 0.0), step=0.01)
                        category = st.selectbox(
                            "Category",
                            ['Groceries', 'Dining', 'Transportation', 'Shopping', 'Health', 'Entertainment', 'Utilities', 'Other'],
                            index=0 if not receipt_data.get('category') else ['Groceries', 'Dining', 'Transportation', 'Shopping', 'Health', 'Entertainment', 'Utilities', 'Other'].index(receipt_data['category']) if receipt_data['category'] in ['Groceries', 'Dining', 'Transportation', 'Shopping', 'Health', 'Entertainment', 'Utilities', 'Other'] else 0
                        )
                    
                    notes = st.text_area("Notes (optional)")
                    
                    submitted = st.form_submit_button("Save Receipt")
                    
                    if submitted:
                        receipt = {
                            'merchant': merchant,
                            'date': date.strftime('%Y-%m-%d'),
                            'total': total,
                            'category': category,
                            'notes': notes
                        }
                        add_receipt(receipt)
                        st.success(f"Receipt saved! Total receipts: {len(st.session_state.receipts)}")
    
    with col2:
        st.info("**Quick Tips:**\n\n📸 Take clear, well-lit photos\n\n📄 Ensure text is readable\n\n✏️ You can edit OCR results\n\n💾 Data is saved automatically")

# Page: Dashboard
elif page == "Dashboard":
    st.header("📊 Spending Dashboard")
    
    df = get_receipts_df()
    
    if df.empty:
        st.info("No receipts uploaded yet. Go to 'Upload Receipts' to get started!")
    else:
        insights = calculate_insights(df)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Spending", f"${insights['total_spending']:,.2f}")
        
        with col2:
            st.metric("Avg Transaction", f"${insights['avg_transaction']:,.2f}")
        
        with col3:
            st.metric("# of Receipts", len(df))
        
        with col4:
            if 'spending_change' in insights:
                st.metric(
                    "30-Day Change",
                    f"{insights['spending_change']:+.1f}%",
                    delta=f"{insights['spending_change']:.1f}%"
                )
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending by Category
            st.subheader("Spending by Category")
            if insights['category_spending']:
                fig = px.pie(
                    values=list(insights['category_spending'].values()),
                    names=list(insights['category_spending'].keys()),
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top Categories
            st.subheader("Top Spending Categories")
            sorted_categories = sorted(insights['category_spending'].items(), key=lambda x: x[1], reverse=True)
            for cat, amount in sorted_categories[:5]:
                st.metric(cat, f"${amount:,.2f}")
        
        # Spending over time
        if 'date' in df.columns and not df['date'].isna().all():
            st.subheader("Spending Trends")
            daily_spending = df.groupby('date')['total'].sum().reset_index()
            
            fig = px.line(
                daily_spending,
                x='date',
                y='total',
                markers=True,
                title="Daily Spending"
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Amount ($)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Insights & Recommendations
        st.subheader("💡 Insights & Recommendations")
        
        if insights['category_spending']:
            top_category = max(insights['category_spending'].items(), key=lambda x: x[1])
            st.write(f"🎯 **Top Spending Category:** {top_category[0]} (${top_category[1]:,.2f})")
            
            if top_category[0] == 'Dining':
                st.info("💰 **Tip:** Consider meal prepping to reduce dining expenses. Cooking at home can save 40-50% on food costs!")
            elif top_category[0] == 'Shopping':
                st.info("💰 **Tip:** Try the 24-hour rule: wait a day before making non-essential purchases to avoid impulse buying.")
            elif top_category[0] == 'Groceries':
                st.info("💰 **Tip:** Make a shopping list and stick to it. Avoid shopping when hungry to reduce impulse purchases.")
        
        if 'spending_change' in insights:
            if insights['spending_change'] > 10:
                st.warning(f"⚠️ Your spending increased by {insights['spending_change']:.1f}% compared to the previous month. Review your spending categories!")
            elif insights['spending_change'] < -10:
                st.success(f"🎉 Great job! Your spending decreased by {abs(insights['spending_change']):.1f}% compared to the previous month!")

# Page: Detailed Analysis
elif page == "Detailed Analysis":
    st.header("📋 Detailed Analysis")
    
    df = get_receipts_df()
    
    if df.empty:
        st.info("No receipts uploaded yet.")
    else:
        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = ['All'] + sorted(df['category'].unique().tolist())
            selected_category = st.selectbox("Category", categories)
        
        with col2:
            if 'date' in df.columns and not df['date'].isna().all():
                date_range = st.date_input(
                    "Date Range",
                    value=(df['date'].min(), df['date'].max()),
                    key='date_range'
                )
        
        # Apply filters
        filtered_df = df.copy()
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        if 'date' in filtered_df.columns and not filtered_df['date'].isna().all() and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['date'] >= pd.Timestamp(date_range[0])) &
                (filtered_df['date'] <= pd.Timestamp(date_range[1]))
            ]
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", f"${filtered_df['total'].sum():,.2f}")
        with col2:
            st.metric("Average", f"${filtered_df['total'].mean():,.2f}")
        with col3:
            st.metric("Count", len(filtered_df))
        
        st.divider()
        
        # Receipt table
        st.subheader("Receipt History")
        display_df = filtered_df[['date', 'merchant', 'category', 'total']].copy()
        display_df['total'] = display_df['total'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export option
        if st.button("Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"receipts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Page: Settings
elif page == "Settings":
    st.header("⚙️ Settings")
    
    st.subheader("Budget Goals")
    
    col1, col2 = st.columns(2)
    with col1:
        monthly_budget = st.number_input("Monthly Budget ($)", min_value=0.0, value=2000.0, step=100.0)
    
    with col2:
        st.metric("Current Month Spending", f"${sum(r['total'] for r in st.session_state.receipts if r['total']):,.2f}")
    
    # Category budgets
    st.subheader("Category Budgets")
    categories = ['Groceries', 'Dining', 'Transportation', 'Shopping', 'Health', 'Entertainment', 'Utilities', 'Other']
    
    for category in categories:
        st.number_input(f"{category} Budget ($)", min_value=0.0, value=0.0, step=50.0, key=f"budget_{category}")
    
    st.divider()
    
    st.subheader("Data Management")
    
    if st.session_state.receipts:
        if st.button("Export All Data (JSON)", type="primary"):
            json_data = json.dumps(st.session_state.receipts, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"receipts_backup_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    uploaded_json = st.file_uploader("Import Data (JSON)", type=['json'])
    if uploaded_json:
        try:
            imported_data = json.load(uploaded_json)
            st.session_state.receipts = imported_data
            st.success(f"Imported {len(imported_data)} receipts!")
            st.rerun()
        except Exception as e:
            st.error(f"Error importing data: {str(e)}")

st.divider()
st.caption("💡 Tip: For best OCR results, ensure your receipt images are clear and well-lit.")