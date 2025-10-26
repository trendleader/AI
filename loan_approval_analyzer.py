import streamlit as st
import os
import requests
from dotenv import load_dotenv

# Load env file
load_dotenv("C:\\Users\\pjone\\OneDrive\\Desktop\\NewPython\\test.env")

# Configure page
st.set_page_config(page_title="Loan Approval Analyzer", page_icon="💰", layout="wide")

st.title("💰 Loan Approval Analyzer")
st.markdown("Analyze your debt-to-income ratio and get AI-powered loan approval insights")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This tool helps you understand:")
    st.write("✓ Your debt-to-income ratio")
    st.write("✓ Loan approval probability")
    st.write("✓ Financial recommendations")

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

# Layout
left, right = st.columns(2)

with left:
    st.subheader("💼 Your Financial Info")
    income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)
    debt = st.number_input("Monthly Debt ($)", min_value=0.0, value=1000.0, step=50.0)
    
    if income > 0:
        current_dti = (debt / income) * 100
        st.metric("Your DTI", f"{current_dti:.1f}%")
    else:
        current_dti = 0

with right:
    st.subheader("🏦 Loan Application")
    loan_amt = st.number_input("Loan Amount ($)", min_value=0.0, value=10000.0, step=500.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    loan_type = st.selectbox("Loan Type", ["Personal Loan", "Auto Loan", "Mortgage", "Credit Card", "Home Equity"])
    
    st.subheader("⚙️ Loan Terms")
    rate = st.slider("Interest Rate (%)", 1.0, 25.0, 7.5, 0.5)
    term = st.slider("Term (months)", 12, 360, 60)
    
    # Calculate payment
    if loan_amt > 0 and rate > 0:
        monthly_rate = rate / 100 / 12
        if monthly_rate > 0:
            payment = loan_amt * (monthly_rate * (1 + monthly_rate)**term) / ((1 + monthly_rate)**term - 1)
        else:
            payment = loan_amt / term
    else:
        payment = 0
    
    st.metric("Est. Monthly Payment", f"${payment:.2f}")

st.markdown("---")

# AI Analysis Button
st.subheader("🤖 AI Analysis")

if st.button("Get Approval Assessment", use_container_width=True):
    if not api_key:
        st.error("API key not found in test.env")
    elif income <= 0:
        st.error("Please enter valid income")
    else:
        with st.spinner("Analyzing..."):
            new_debt = debt + payment
            new_dti = (new_debt / income) * 100
            
            prompt = f"""Analyze this loan application:

CURRENT:
- Monthly Income: ${income:,.2f}
- Monthly Debt: ${debt:,.2f}
- DTI: {current_dti:.1f}%
- Credit Score: {credit_score}

APPLICATION:
- Type: {loan_type}
- Amount: ${loan_amt:,.2f}
- Rate: {rate}%
- Term: {term} months
- Payment: ${payment:.2f}

IF APPROVED:
- New Monthly Debt: ${new_debt:,.2f}
- New DTI: {new_dti:.1f}%

Provide:
1. Approval probability (0-100%)
2. Approval factors
3. Risk factors
4. DTI comparison to industry standards
5. Recommendations

Keep response concise and well-organized."""

            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result['choices'][0]['message']['content']
                    
                    st.success("Analysis Complete!")
                    st.markdown(analysis)
                    
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current DTI", f"{current_dti:.1f}%")
                    c2.metric("New DTI", f"{new_dti:.1f}%")
                    c3.metric("Change", f"{new_dti - current_dti:+.1f}%")
                    
                    st.info("""
                    **DTI Guidelines:**
                    - Below 36%: Excellent
                    - 36-43%: Good
                    - 43-50%: Fair
                    - Above 50%: Poor
                    """)
                    
                elif response.status_code == 401:
                    st.error("Invalid API key. Check test.env")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.warning("⚠️ Disclaimer: This is for educational purposes only. Not a guarantee of approval.")
