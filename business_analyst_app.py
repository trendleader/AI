import streamlit as st
import openai
import json
from typing import Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Business Analyst Assessment Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Business Analyst Assessment Tool")
st.markdown("*Practice writing business artifacts and receive AI-powered feedback*")

# ============================================================================
# BUSINESS SCENARIOS - Add more scenarios here as needed
# ============================================================================
SCENARIOS = {
    "Scenario 1: E-commerce Platform Enhancement": {
        "title": "E-commerce Platform Enhancement",
        "description": """Your company operates a mid-sized e-commerce platform. The product team has identified that 
        customers are abandoning their shopping carts at high rates during checkout. Current analytics show:
        - 45% cart abandonment rate (industry average is 35%)
        - Customers cite "complicated checkout process" as the top reason
        - Mobile checkout has a 60% abandonment rate vs. 35% on desktop
        - Estimated revenue loss: $500K per month
        
        Your task is to write a Business Requirements Document that would help the development team understand what needs to be changed. 
        Consider the problem, desired outcomes, user needs, and acceptance criteria.""",
        "artifact_type": "Business Requirements Document"
    },
    "Scenario 2: Data Privacy Compliance": {
        "title": "GDPR Compliance Initiative",
        "description": """Your organization is a SaaS company operating in Europe and has discovered that your current data 
        handling processes are not fully compliant with GDPR regulations. You need to:
        - Implement user data deletion capabilities
        - Create data consent management system
        - Establish audit logging for data access
        - Set compliance deadlines within 6 months
        
        Write a Business Case that justifies the investment in this initiative. Include business impact, 
        risks of non-compliance, resource requirements, and expected outcomes.""",
        "artifact_type": "Business Case"
    },
    "Scenario 3: Mobile App Feature": {
        "title": "Push Notification Feature Development",
        "description": """Your mobile banking app users have requested the ability to receive real-time notifications 
        for account activities (large transactions, login alerts, etc.). Currently, users must manually check the app 
        to stay informed, leading to missed fraud alerts and poor user experience.
        
        Your task is to write Acceptance Criteria for this feature. Consider different notification types, 
        user preferences, notification delivery methods (push, in-app, email), and edge cases.""",
        "artifact_type": "Acceptance Criteria"
    },
    "Scenario 4: Digital Transformation Initiative": {
        "title": "Legacy System Modernization",
        "description": """Your healthcare organization still relies on a 20-year-old legacy system for patient records, 
        scheduling, and billing. Current challenges include:
        - System crashes 2-3 times weekly, affecting patient care
        - Manual data entry required; duplicate records are common
        - Staff training takes 6 months for new employees
        - Patient portal is unavailable; all communication is phone-based
        - Estimated 15% of billing errors due to system limitations
        
        Write a Business Case for modernizing this system. Include risks of maintaining status quo, 
        benefits of modernization, implementation timeline, budget considerations, and success metrics.""",
        "artifact_type": "Business Case"
    },
    "Scenario 5: Customer Support Portal Redesign": {
        "title": "Self-Service Support Portal Enhancement",
        "description": """Your software company's support team handles 10,000 tickets monthly. Analysis shows:
        - 40% of tickets are duplicates or FAQs that customers could self-serve
        - Average wait time: 48 hours
        - Support team is at full capacity but customer satisfaction is declining (CSAT: 72%)
        - Customers request better documentation and video tutorials
        - Goal: Reduce ticket volume by 30% and improve CSAT to 85% within 3 months
        
        Write a Business Requirements Document for a new self-service support portal that includes 
        knowledge base, video tutorials, community forums, and AI-powered search capabilities.""",
        "artifact_type": "Business Requirements Document"
    },
    "Scenario 6: Inventory Management System": {
        "title": "Real-Time Inventory Tracking",
        "description": """Your retail company operates 250 physical stores and an online platform. Current issues:
        - Inventory data updates only once daily (overnight batch)
        - 30% stockouts occur because inventory shown as available is actually sold
        - Lost sales estimated at $2M annually
        - No real-time visibility into stock levels across locations
        - Customers cannot check in-store availability online
        
        Define comprehensive Acceptance Criteria for a real-time inventory tracking system that 
        updates inventory instantly as items are sold or received in stores. Consider point-of-sale integration, 
        omnichannel visibility, exception handling, and reporting requirements.""",
        "artifact_type": "Acceptance Criteria"
    },
    "Scenario 7: Cloud Migration Project": {
        "title": "Infrastructure Migration to Cloud",
        "description": """Your mid-sized financial services company currently hosts all systems on-premise with:
        - High capital expenditure for hardware and maintenance ($1.2M annually)
        - Limited scalability during peak trading periods
        - Single data center vulnerability; no disaster recovery site
        - Manual backup procedures prone to human error
        - Difficulty recruiting IT talent in expensive on-premise skill requirements
        
        Write a comprehensive Business Case for migrating to a cloud infrastructure (AWS/Azure). 
        Include cost-benefit analysis, risk mitigation strategies, security and compliance considerations, 
        implementation approach, and ROI projections over 5 years.""",
        "artifact_type": "Business Case"
    },
    "Scenario 8: Employee Onboarding Process": {
        "title": "Automated Employee Onboarding System",
        "description": """Your HR department processes 500 new hires annually across 15 locations. Current process is manual and inefficient:
        - New employees complete 50+ forms manually
        - IT provisioning takes 3-5 days; employees often start without computer access
        - Managers must gather approvals from multiple departments
        - No standardized experience; process varies by location
        - High administrative burden (HR team spends 30% of time on onboarding)
        - New employee satisfaction with onboarding: 65%
        
        Create detailed Business Requirements for an automated onboarding platform that includes 
        digital forms, IT provisioning automation, manager workflows, and employee self-service portal.""",
        "artifact_type": "Business Requirements Document"
    },
    "Scenario 9: Analytics Dashboard for Marketing": {
        "title": "Real-Time Marketing Analytics Dashboard",
        "description": """Your marketing team manages campaigns across 8 channels (email, social, paid search, organic, 
        affiliate, webinars, events, content). Currently:
        - Data is scattered across 6 different tools and platforms
        - Marketers manually compile weekly reports (4 hours per week)
        - Campaign decisions are delayed 2-3 days while gathering data
        - ROI calculation is manual and error-prone
        - No real-time visibility into campaign performance
        
        Write Acceptance Criteria for a unified analytics dashboard that consolidates all marketing data, 
        provides real-time performance metrics, automated reporting, predictive analytics, and budget tracking.""",
        "artifact_type": "Acceptance Criteria"
    },
    "Scenario 10: Supply Chain Optimization": {
        "title": "Supply Chain Visibility Platform",
        "description": """Your manufacturing company produces complex products requiring 50+ component suppliers globally. Challenges:
        - No real-time visibility into supplier performance or delivery status
        - Frequent delays cause production shutdowns (lost revenue: $500K per incident)
        - Manual tracking of shipments; average tracking time: 3 hours per shipment
        - No early warning system for supplier issues
        - Inventory buffers are excessive due to uncertainty (tied up capital: $5M)
        - Quality issues sometimes discovered too late in the production cycle
        
        Create a comprehensive Business Case for implementing a supply chain visibility platform. 
        Include expected benefits (cost savings, reduced downtime, improved quality), implementation risks, 
        technology requirements, and strategic value to the organization.""",
        "artifact_type": "Business Case"
    },
    "Scenario 11: Third-Party API Integration": {
        "title": "Payment Gateway Integration",
        "description": """Your e-commerce platform currently accepts only credit card payments through one provider. 
        Analysis shows:
        - 15% of potential customers abandon due to limited payment options
        - Customers request: PayPal, Apple Pay, Google Pay, bank transfers, and Buy Now Pay Later (BNPL)
        - International customers need local payment methods (Alipay, WeChat Pay, etc.)
        - Current provider charges 2.9% + $0.30 per transaction
        - New providers offer competitive rates but require integration
        
        Write a detailed Business Requirements Document for integrating multiple payment gateways. 
        Include merchant onboarding flow, payment processing, error handling, security considerations, 
        and customer experience requirements.""",
        "artifact_type": "Business Requirements Document"
    },
    "Scenario 12: Cost Reduction Program": {
        "title": "Operational Efficiency and Cost Reduction Initiative",
        "description": """Your organization has ambitious growth targets but rising operational costs threaten profitability:
        - Operating expenses grew 15% YoY while revenue grew only 8%
        - Software licensing costs: $2M annually (many unused licenses)
        - Vendor contracts lack competitive pricing (haven't been re-negotiated in 3-5 years)
        - Manual processes waste 20% of staff time (equivalent to 40 FTEs)
        - Consulting and temporary staff costs: $1.5M annually
        - Target: Reduce costs by $4M in 12 months without impacting revenue
        
        Develop a comprehensive Business Case for a cost reduction program. Include detailed savings 
        opportunities, implementation roadmap, resource requirements, potential risks of cost-cutting, 
        and strategies to maintain customer satisfaction and employee morale.""",
        "artifact_type": "Business Case"
    },
    "Scenario 13: Labor Union LMS Platform": {
        "title": "Learning Management System for Skilled Trades",
        "description": """Your labor union represents 25,000 skilled craftspeople including electricians, plumbers, 
        HVAC technicians, carpenters, and heavy equipment operators across 15 states. Currently:
        - Training is fragmented: Some courses in-person only, others online through outdated systems
        - No centralized course catalog; members struggle to find relevant professional development
        - Training materials are inconsistently updated (some content is 5+ years old)
        - In-person workshops have limited capacity and scheduling conflicts with work
        - New apprentices receive inconsistent onboarding depending on local chapter
        - Members cannot track their own progress or certifications across chapters
        - Union struggles to demonstrate value to members, affecting recruitment and retention
        
        Write a comprehensive Business Requirements Document for a modern, mobile-first LMS that 
        serves your diverse membership. Consider: course types (videos, interactive modules, certifications), 
        mobile access for field workers, apprenticeship tracking, skill assessments, progress tracking, 
        multi-role support (instructors, apprentices, union staff), and integration with existing member database.""",
        "artifact_type": "Business Requirements Document"
    },
    "Scenario 14: Union Training Investment": {
        "title": "Strategic Investment in Member Skill Development",
        "description": """Your labor union faces critical challenges that threaten relevance and membership:
        - Membership has declined 8% over 3 years; younger workers are not joining
        - Member earnings are stagnant; average age of membership is 52 (older than before)
        - Competing trade schools and online platforms offer training that union doesn't provide
        - Employers increasingly request workers with advanced certifications union doesn't support
        - Only 12% of members take any training in a given year (union benchmark: 40%)
        - Union dues revenue declining with membership; unable to fund existing services
        - Members cite "lack of professional development opportunities" as #2 reason for leaving
        
        Develop a comprehensive Business Case for investing $2.5M over 3 years in a modern LMS and 
        training program. Include: business objectives (membership growth, retention, earnings improvement), 
        financial projections (impact on dues revenue, cost avoidance, member earnings), member value 
        proposition, competitive advantages vs. other training providers, implementation roadmap, and 
        success metrics. Address the question: How does this investment position the union as THE 
        preferred training provider for skilled trades?""",
        "artifact_type": "Business Case"
    },
    "Scenario 15: LMS Implementation Success Criteria": {
        "title": "LMS Adoption and Performance Metrics",
        "description": """Your union has approved a $2.5M investment in a new LMS and you need to define clear, 
        measurable success criteria for the implementation. Key stakeholders include:
        - Union leadership wanting ROI and membership growth
        - Members wanting accessible, high-quality training
        - Local chapter coordinators managing courses
        - Instructors developing and delivering content
        - HR staff handling administration
        
        Define comprehensive Acceptance Criteria for the LMS that ensure:
        - High adoption rates among diverse membership (consider technology literacy varies)
        - Content quality and relevance to actual job requirements
        - Engagement metrics (courses completed, certifications earned)
        - Impact on member skills and job advancement
        - Accessibility for field workers (mobile, offline capability)
        - Integration with union systems (member database, chapter management)
        - Instructor and administrator usability
        - Performance and reliability standards
        
        Include both technical criteria (system performance, uptime, data security) and business metrics 
        (adoption %, course completion %, member satisfaction, certification achievement, job placement impact).""",
        "artifact_type": "Acceptance Criteria"
    }
}

# ============================================================================
# GRADING RUBRIC - Customize scoring criteria here
# ============================================================================
GRADING_RUBRIC = """
You are an expert business analyst evaluator. Grade the submitted document on the following criteria:

1. CONTENT COMPLETENESS (0-25 points)
   - All key requirements/criteria are identified
   - Addresses the business problem thoroughly
   - Includes measurable objectives
   - Covers success metrics or acceptance criteria

2. LOGICAL FLOW & ORGANIZATION (0-25 points)
   - Clear structure and logical progression
   - Information is well-organized
   - Proper use of sections/headings if applicable
   - Smooth transitions between ideas

3. CLARITY & PROFESSIONAL TONE (0-25 points)
   - Uses clear, concise language
   - Avoids jargon or explains it properly
   - Maintains professional tone throughout
   - Specific and avoids vague statements

4. GRAMMAR, SPELLING & PUNCTUATION (0-25 points)
   - No spelling errors
   - Proper grammar usage
   - Correct punctuation
   - Overall writing quality

RESPONSE FORMAT:
Provide your assessment as a JSON object with this exact structure:
{
    "content_completeness": {
        "score": <0-25>,
        "feedback": "<specific feedback>"
    },
    "logical_flow": {
        "score": <0-25>,
        "feedback": "<specific feedback>"
    },
    "clarity_tone": {
        "score": <0-25>,
        "feedback": "<specific feedback>"
    },
    "grammar_spelling": {
        "score": <0-25>,
        "feedback": "<specific feedback>"
    },
    "total_score": <0-100>,
    "overall_assessment": "<2-3 sentence summary>",
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "areas_for_improvement": ["<area 1>", "<area 2>", ...]
}
"""

# ============================================================================
# SIDEBAR - OpenAI API Key Input
# ============================================================================
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input(
    "Enter your OpenAI API key:",
    type="password",
    help="Your API key is used only for this session and is not stored"
)

if api_key:
    st.sidebar.success("✅ API key configured")
else:
    st.sidebar.warning("⚠️ Please enter your OpenAI API key to use this app")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Scenario Selection
st.subheader("Step 1: Select a Business Scenario")
scenario_choice = st.selectbox(
    "Choose a scenario to work on:",
    list(SCENARIOS.keys()),
    label_visibility="collapsed"
)

scenario = SCENARIOS[scenario_choice]

# Display scenario details
with st.container(border=True):
    st.markdown(f"### {scenario['title']}")
    st.markdown(f"**Artifact Type:** {scenario['artifact_type']}")
    st.markdown(f"\n{scenario['description']}")

# Document input
st.subheader("Step 2: Write Your Document")
st.markdown(f"Please write your {scenario['artifact_type']} below:")

user_document = st.text_area(
    "Your document:",
    placeholder="Enter your business requirements document, business case, or acceptance criteria here...",
    height=250,
    label_visibility="collapsed"
)

# Assessment submission
st.subheader("Step 3: Submit for Assessment")

col1, col2 = st.columns([3, 1])
with col2:
    submit_button = st.button(
        "🔍 Assess Document",
        type="primary",
        use_container_width=True,
        disabled=not (api_key and user_document and len(user_document.strip()) > 50)
    )

if not api_key:
    st.info("💡 Add your OpenAI API key in the sidebar to assess your document.")
elif not user_document or len(user_document.strip()) < 50:
    st.info("💡 Write at least 50 characters in your document to enable assessment.")

# ============================================================================
# ASSESSMENT PROCESSING
# ============================================================================

if submit_button:
    if not api_key:
        st.error("Please provide your OpenAI API key in the sidebar.")
        st.stop()
    
    if not user_document or len(user_document.strip()) < 50:
        st.error("Please write a more substantial document (at least 50 characters).")
        st.stop()
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Create assessment prompt
        assessment_prompt = f"""
{GRADING_RUBRIC}

SCENARIO:
{scenario['description']}

DOCUMENT TYPE REQUESTED: {scenario['artifact_type']}

USER'S SUBMITTED DOCUMENT:
{user_document}

Please evaluate this document and provide your assessment in the JSON format specified above.
"""
        
        with st.spinner("🤖 Analyzing your document with OpenAI..."):
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert business analyst and evaluator. Provide detailed, constructive feedback on business documents."
                    },
                    {
                        "role": "user",
                        "content": assessment_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
        
        # Parse response
        response_text = response.choices[0].message.content
        
        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                assessment = json.loads(json_match.group())
            else:
                assessment = json.loads(response_text)
        except json.JSONDecodeError:
            st.error("Could not parse assessment response. Please try again.")
            st.text(response_text)
            st.stop()
        
        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================
        st.success("✅ Assessment Complete!")
        
        # Overall Score - Prominent Display
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Content Completeness",
                f"{assessment['content_completeness']['score']}/25"
            )
        with col2:
            st.metric(
                "Logical Flow",
                f"{assessment['logical_flow']['score']}/25"
            )
        with col3:
            st.metric(
                "Clarity & Tone",
                f"{assessment['clarity_tone']['score']}/25"
            )
        with col4:
            st.metric(
                "Grammar & Spelling",
                f"{assessment['grammar_spelling']['score']}/25"
            )
        with col5:
            total = assessment['total_score']
            color = "green" if total >= 80 else "orange" if total >= 70 else "red"
            st.markdown(f"<h3 style='text-align: center; color: {color};'>{total}/100</h3>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Overall Assessment Summary
        st.subheader("📋 Overall Assessment")
        st.info(assessment['overall_assessment'])
        
        # Detailed Feedback
        st.subheader("📝 Detailed Feedback")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Content Completeness",
            "Logical Flow",
            "Clarity & Tone",
            "Grammar & Spelling"
        ])
        
        with tab1:
            st.write(f"**Score: {assessment['content_completeness']['score']}/25**")
            st.write(assessment['content_completeness']['feedback'])
        
        with tab2:
            st.write(f"**Score: {assessment['logical_flow']['score']}/25**")
            st.write(assessment['logical_flow']['feedback'])
        
        with tab3:
            st.write(f"**Score: {assessment['clarity_tone']['score']}/25**")
            st.write(assessment['clarity_tone']['feedback'])
        
        with tab4:
            st.write(f"**Score: {assessment['grammar_spelling']['score']}/25**")
            st.write(assessment['grammar_spelling']['feedback'])
        
        # Strengths and Areas for Improvement
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💪 Strengths")
            for strength in assessment['strengths']:
                st.write(f"✓ {strength}")
        
        with col2:
            st.subheader("🎯 Areas for Improvement")
            for area in assessment['areas_for_improvement']:
                st.write(f"→ {area}")
        
        st.markdown("---")
        st.markdown("### Try Again?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Revise This Document"):
                st.rerun()
        with col2:
            if st.button("🔄 Choose Another Scenario"):
                st.session_state.clear()
                st.rerun()
    
    except openai.AuthenticationError:
        st.error("❌ Authentication failed. Please check your OpenAI API key.")
    except openai.RateLimitError:
        st.error("❌ Rate limit exceeded. Please wait a moment and try again.")
    except openai.APIError as e:
        st.error(f"❌ OpenAI API error: {str(e)}")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Business Analyst Assessment Tool | Powered by OpenAI</p>
    <p>Your documents are assessed in real-time. API keys are not stored.</p>
</div>
""", unsafe_allow_html=True)
