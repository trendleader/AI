"""
Business Requirements Generator - Streamlit Web App with OpenAI Integration

SETUP INSTRUCTIONS FOR LOCAL:
1. Install required packages:
   pip install streamlit openai python-dotenv

2. Make sure your test.env file contains:
   OPENAI_API_KEY=your-openai-api-key-here

3. Run the app:
   streamlit run business_requirements_app.py

SETUP FOR STREAMLIT CLOUD:
1. Create requirements.txt with:
   streamlit
   openai

2. In Streamlit Cloud, go to App Settings > Secrets and add:
   OPENAI_API_KEY = "your-openai-api-key-here"
"""

import streamlit as st
import re
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Load API key - works for both local and Streamlit Cloud
def load_api_key():
    """Load API key from Streamlit secrets (cloud) or environment (local)."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except (AttributeError, FileNotFoundError):
        # Secrets not available (running locally without secrets.toml)
        pass
    
    # Try environment variable
    if 'OPENAI_API_KEY' in os.environ:
        return os.environ['OPENAI_API_KEY']
    
    # Try loading from local .env file (optional, for local development)
    try:
        from dotenv import load_dotenv
        # Try test.env in current directory first
        if Path('test.env').exists():
            load_dotenv('test.env')
        # Try standard .env file
        elif Path('.env').exists():
            load_dotenv('.env')
        # Try the specific path if running locally
        else:
            env_path = Path(r"C:\Users\pjone\OneDrive\Desktop\NewPython\test.env")
            if env_path.exists():
                load_dotenv(env_path)
        
        return os.environ.get('OPENAI_API_KEY')
    except ImportError:
        # dotenv not installed (e.g., on Streamlit Cloud)
        pass
    
    return None

# Load the API key once
OPENAI_API_KEY = load_api_key()


class BusinessRequirementsGenerator:
    """
    Generates structured business requirements from natural language input.
    Supports both template-based and OpenAI-based generation.
    """
    
    def __init__(self, use_ai: bool = False, api_key: Optional[str] = None):
        self.use_ai = use_ai
        self.api_key = api_key or OPENAI_API_KEY
        self.requirement_counter = 1
    
    def generate_requirements(self, natural_language_input: str) -> Dict:
        """Generate requirements using selected method."""
        if self.use_ai and self.api_key:
            return self._generate_with_openai(natural_language_input)
        else:
            return self._generate_with_templates(natural_language_input)
    
    def _generate_with_openai(self, input_text: str) -> Dict:
        """Generate requirements using OpenAI API."""
        try:
            from openai import OpenAI
            
            if not self.api_key:
                st.error("❌ OpenAI API key not found. Using template-based generation.")
                return self._generate_with_templates(input_text)
            
            client = OpenAI(api_key=self.api_key)
            
            prompt = f"""Analyze the following natural language input and create a comprehensive, structured business requirements document. 

Input: {input_text}

Please provide a JSON response with the following structure. Be thorough and extract as much detail as possible:

{{
    "document_info": {{
        "title": "Business Requirements Document",
        "date_created": "{datetime.now().strftime('%Y-%m-%d')}",
        "version": "1.0"
    }},
    "project_overview": {{
        "description": "Comprehensive project description",
        "goals": ["List of specific, measurable project goals"],
        "scope": "Detailed project scope including what's in and out of scope"
    }},
    "functional_requirements": [
        {{
            "id": "FR-001",
            "description": "Specific, clear, and testable requirement description",
            "priority": "High/Medium/Low",
            "category": "Functional"
        }}
    ],
    "non_functional_requirements": [
        {{
            "id": "NFR-001",
            "description": "Specific, measurable non-functional requirement",
            "category": "Performance/Security/Usability/Scalability/Reliability/Maintainability",
            "priority": "High/Medium/Low"
        }}
    ],
    "stakeholders": ["List of identified stakeholders with their roles"],
    "assumptions": ["List of project assumptions"],
    "constraints": ["List of project constraints including budget, time, technology, etc."]
}}

IMPORTANT GUIDELINES:
- Make requirements specific, clear, and testable
- Avoid vague language like "should", "could", "appropriate"
- Include measurable criteria where possible
- Each requirement should be atomic (one requirement, not multiple)
- Prioritize requirements based on business value and dependencies
- Include both functional and non-functional requirements
- Be comprehensive but realistic

Provide ONLY valid JSON, no additional text or markdown formatting."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a business analyst expert who creates detailed, structured business requirements documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content
            # Clean up markdown code blocks if present
            response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
            
            requirements = json.loads(response_text)
            return requirements
            
        except ImportError:
            st.error("❌ 'openai' package not installed. Install with: pip install openai")
            return self._generate_with_templates(input_text)
        except json.JSONDecodeError as e:
            st.error(f"❌ Error parsing AI response: {e}")
            st.info("Falling back to template-based generation.")
            return self._generate_with_templates(input_text)
        except Exception as e:
            st.error(f"❌ Error with OpenAI generation: {e}")
            st.info("Falling back to template-based generation.")
            return self._generate_with_templates(input_text)
    
    def _generate_with_templates(self, input_text: str) -> Dict:
        """Generate requirements using pattern matching and templates."""
        
        requirements = {
            "document_info": {
                "title": "Business Requirements Document",
                "date_created": datetime.now().strftime("%Y-%m-%d"),
                "version": "1.0"
            },
            "project_overview": self._extract_overview(input_text),
            "functional_requirements": self._extract_functional_requirements(input_text),
            "non_functional_requirements": self._extract_non_functional_requirements(input_text),
            "stakeholders": self._extract_stakeholders(input_text),
            "assumptions": self._extract_assumptions(input_text),
            "constraints": self._extract_constraints(input_text)
        }
        
        return requirements
    
    def _extract_overview(self, text: str) -> Dict:
        """Extract project overview information."""
        overview = {
            "description": "",
            "goals": [],
            "scope": ""
        }
        
        goal_patterns = [
            r"(?:goal|objective|aim|purpose)(?:s)?(?:\s+is|\s+are)?:?\s*([^.!?]+)",
            r"(?:we want to|need to|should|must)\s+([^.!?]+)",
            r"(?:in order to|to)\s+([^.!?]+)"
        ]
        
        goals = []
        for pattern in goal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                goal = match.group(1).strip()
                if goal and len(goal) > 10:
                    goals.append(goal)
        
        overview["goals"] = list(set(goals))[:5]
        overview["description"] = text[:500]
        
        return overview
    
    def _extract_functional_requirements(self, text: str) -> List[Dict]:
        """Extract functional requirements from text."""
        requirements = []
        
        patterns = [
            r"(?:system|application|user|platform)(?:\s+should|\s+must|\s+shall|\s+will)\s+([^.!?]+)",
            r"(?:user|admin|customer)(?:\s+can|\s+should be able to)\s+([^.!?]+)",
            r"(?:the system|it)(?:\s+allows|\s+enables|\s+provides|\s+supports)\s+([^.!?]+)",
            r"(?:feature|functionality|capability)(?:\s+to|\s+for|\s+that)\s+([^.!?]+)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                requirement_text = match.group(1).strip()
                if requirement_text and len(requirement_text) > 15:
                    requirements.append({
                        "id": f"FR-{self.requirement_counter:03d}",
                        "description": requirement_text,
                        "priority": self._determine_priority(match.group(0)),
                        "category": "Functional"
                    })
                    self.requirement_counter += 1
        
        if not requirements:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences[:10]:
                sentence = sentence.strip()
                if len(sentence) > 30:
                    requirements.append({
                        "id": f"FR-{self.requirement_counter:03d}",
                        "description": sentence,
                        "priority": "Medium",
                        "category": "Functional"
                    })
                    self.requirement_counter += 1
        
        return requirements[:15]
    
    def _extract_non_functional_requirements(self, text: str) -> List[Dict]:
        """Extract non-functional requirements."""
        nfr = []
        
        if re.search(r"(fast|quick|speed|performance|response time)", text, re.IGNORECASE):
            nfr.append({
                "id": f"NFR-{len(nfr)+1:03d}",
                "description": "System should provide fast response times and optimal performance",
                "category": "Performance",
                "priority": "High"
            })
        
        if re.search(r"(secure|security|authentication|authorization|encrypted?)", text, re.IGNORECASE):
            nfr.append({
                "id": f"NFR-{len(nfr)+1:03d}",
                "description": "System must implement robust security measures and data protection",
                "category": "Security",
                "priority": "High"
            })
        
        if re.search(r"(scalable?|scale|growth|expand|increase)", text, re.IGNORECASE):
            nfr.append({
                "id": f"NFR-{len(nfr)+1:03d}",
                "description": "System should be scalable to handle growing user base and data volume",
                "category": "Scalability",
                "priority": "Medium"
            })
        
        if re.search(r"(user.friendly|easy to use|intuitive|simple)", text, re.IGNORECASE):
            nfr.append({
                "id": f"NFR-{len(nfr)+1:03d}",
                "description": "System must provide an intuitive and user-friendly interface",
                "category": "Usability",
                "priority": "High"
            })
        
        return nfr
    
    def _extract_stakeholders(self, text: str) -> List[str]:
        """Extract stakeholders from text."""
        stakeholders = []
        stakeholder_patterns = [
            r"(users?|customers?|clients?|administrators?|managers?|developers?|team|stakeholders?)",
        ]
        
        for pattern in stakeholder_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                stakeholder = match.group(1).strip().capitalize()
                if stakeholder not in stakeholders:
                    stakeholders.append(stakeholder)
        
        return stakeholders if stakeholders else ["End Users", "Project Team", "Management"]
    
    def _extract_assumptions(self, text: str) -> List[str]:
        """Extract assumptions from text."""
        assumptions = []
        assumption_patterns = [
            r"(?:assume|assuming|assumed)(?:\s+that)?\s+([^.!?]+)",
            r"(?:provided that|given that)\s+([^.!?]+)"
        ]
        
        for pattern in assumption_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                assumptions.append(match.group(1).strip())
        
        return assumptions if assumptions else ["All stakeholders are available for consultation", "Required resources will be provided"]
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints from text."""
        constraints = []
        constraint_patterns = [
            r"(?:constraint|limitation|restricted?)(?:\s+is|\s+are)?\s+([^.!?]+)",
            r"(?:limited by|cannot|must not)\s+([^.!?]+)",
            r"(?:budget|timeline|deadline)(?:\s+of|\s+is)?\s+([^.!?]+)"
        ]
        
        for pattern in constraint_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                constraints.append(match.group(1).strip())
        
        return constraints if constraints else ["Budget constraints to be confirmed", "Timeline to be determined"]
    
    def _determine_priority(self, text: str) -> str:
        """Determine priority based on keywords."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["must", "critical", "essential", "required"]):
            return "High"
        elif any(word in text_lower for word in ["should", "important"]):
            return "Medium"
        else:
            return "Low"
    
    def generate_acceptance_criteria(self, requirement: Dict) -> List[str]:
        """Generate acceptance criteria for a given requirement."""
        criteria = []
        description = requirement['description'].lower()
        
        if any(word in description for word in ['user', 'customer', 'admin', 'login', 'click', 'view']):
            criteria.append(f"Given the user is authenticated")
            criteria.append(f"When the user {requirement['description']}")
            criteria.append(f"Then the system should complete the action successfully")
            criteria.append(f"And appropriate feedback is provided to the user")
        else:
            criteria.append(f"The functionality described is fully implemented")
            criteria.append(f"All edge cases and error conditions are handled appropriately")
            criteria.append(f"The implementation meets performance standards")
            criteria.append(f"The feature is tested and verified")
        
        if 'data' in description or 'information' in description:
            criteria.append("Data integrity is maintained throughout the process")
        
        if 'search' in description or 'filter' in description:
            criteria.append("Results are accurate and relevant")
            criteria.append("Response time is within acceptable limits")
        
        if 'save' in description or 'update' in description or 'create' in description:
            criteria.append("Changes are persisted correctly")
            criteria.append("Appropriate confirmation is provided")
        
        if 'delete' in description or 'remove' in description:
            criteria.append("Confirmation is required before deletion")
            criteria.append("Deletion is properly logged for audit purposes")
        
        return criteria[:6]
    
    def add_acceptance_criteria_to_requirements(self, requirements: Dict) -> Dict:
        """Add acceptance criteria to all functional requirements."""
        for req in requirements['functional_requirements']:
            req['acceptance_criteria'] = self.generate_acceptance_criteria(req)
        return requirements
    
    def validate_requirement(self, requirement: Dict) -> Dict:
        """Validate a single requirement for quality and completeness."""
        issues = []
        warnings = []
        score = 100
        
        description = requirement.get('description', '')
        
        if not description or len(description) < 10:
            issues.append("Description is too short or missing")
            score -= 30
        
        vague_words = ['should', 'could', 'might', 'maybe', 'possibly', 'approximately', 
                       'reasonable', 'adequate', 'appropriate', 'etc', 'and so on']
        found_vague = [word for word in vague_words if word in description.lower()]
        if found_vague:
            warnings.append(f"Contains vague language: {', '.join(found_vague)}")
            score -= 10
        
        measurable_indicators = ['number', 'percentage', 'time', 'count', 'rate', 
                                'seconds', 'minutes', 'hours', '%', 'maximum', 'minimum']
        has_measurable = any(indicator in description.lower() for indicator in measurable_indicators)
        
        if requirement.get('category') == 'Functional' and not has_measurable:
            warnings.append("Requirement may not be measurable or testable")
            score -= 5
        
        ambiguous_pronouns = ['it', 'this', 'that', 'they', 'them']
        if any(f" {pronoun} " in description.lower() for pronoun in ambiguous_pronouns):
            warnings.append("Contains ambiguous pronouns that may reduce clarity")
            score -= 5
        
        and_count = description.lower().count(' and ')
        if and_count > 2:
            warnings.append(f"May contain multiple requirements in one ({and_count} 'and' statements)")
            score -= 10
        
        if not requirement.get('priority') or requirement['priority'] not in ['High', 'Medium', 'Low']:
            issues.append("Priority is missing or invalid")
            score -= 15
        
        req_id = requirement.get('id', '')
        if not re.match(r'^[A-Z]+-\d{3}$', req_id):
            warnings.append("ID format doesn't follow standard pattern")
            score -= 5
        
        if len(description) > 50:
            score += 5
        
        if has_measurable:
            score += 5
        
        score = max(0, min(100, score))
        
        if score >= 80:
            status = "Good"
        elif score >= 60:
            status = "Needs Improvement"
        else:
            status = "Poor"
        
        return {
            "requirement_id": req_id,
            "status": status,
            "score": score,
            "issues": issues,
            "warnings": warnings
        }
    
    def validate_all_requirements(self, requirements: Dict) -> Dict:
        """Validate all requirements in the document."""
        validation_report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": "Pass",
            "total_requirements": 0,
            "average_score": 0,
            "functional_requirements_validation": [],
            "non_functional_requirements_validation": [],
            "summary": {
                "good": 0,
                "needs_improvement": 0,
                "poor": 0
            },
            "recommendations": []
        }
        
        scores = []
        
        for req in requirements.get('functional_requirements', []):
            validation = self.validate_requirement(req)
            validation_report['functional_requirements_validation'].append(validation)
            scores.append(validation['score'])
            
            if validation['status'] == "Good":
                validation_report['summary']['good'] += 1
            elif validation['status'] == "Needs Improvement":
                validation_report['summary']['needs_improvement'] += 1
            else:
                validation_report['summary']['poor'] += 1
        
        for req in requirements.get('non_functional_requirements', []):
            validation = self.validate_requirement(req)
            validation_report['non_functional_requirements_validation'].append(validation)
            scores.append(validation['score'])
            
            if validation['status'] == "Good":
                validation_report['summary']['good'] += 1
            elif validation['status'] == "Needs Improvement":
                validation_report['summary']['needs_improvement'] += 1
            else:
                validation_report['summary']['poor'] += 1
        
        validation_report['total_requirements'] = len(scores)
        if scores:
            validation_report['average_score'] = round(sum(scores) / len(scores), 2)
        
        if validation_report['average_score'] >= 75:
            validation_report['overall_status'] = "Pass"
        elif validation_report['average_score'] >= 60:
            validation_report['overall_status'] = "Pass with Warnings"
        else:
            validation_report['overall_status'] = "Fail"
        
        if validation_report['summary']['poor'] > 0:
            validation_report['recommendations'].append(
                f"Address {validation_report['summary']['poor']} poor-quality requirements immediately"
            )
        
        if validation_report['summary']['needs_improvement'] > validation_report['total_requirements'] * 0.3:
            validation_report['recommendations'].append(
                "Over 30% of requirements need improvement. Consider a review session."
            )
        
        if validation_report['average_score'] < 75:
            validation_report['recommendations'].append(
                "Consider making requirements more specific and measurable"
            )
        
        return validation_report
    
    def export_to_markdown(self, requirements: Dict) -> str:
        """Export requirements to markdown format."""
        md = f"""# {requirements['document_info']['title']}

**Version:** {requirements['document_info']['version']}  
**Date:** {requirements['document_info']['date_created']}

---

## Project Overview

### Description
{requirements['project_overview']['description']}

### Goals
"""
        for goal in requirements['project_overview']['goals']:
            md += f"- {goal}\n"
        
        md += "\n### Scope\n"
        md += requirements['project_overview'].get('scope', 'To be defined') + "\n\n"
        
        md += "---\n\n## Functional Requirements\n\n"
        for req in requirements['functional_requirements']:
            md += f"### {req['id']}\n"
            md += f"**Description:** {req['description']}  \n"
            md += f"**Priority:** {req['priority']}  \n"
            
            if 'acceptance_criteria' in req:
                md += f"\n**Acceptance Criteria:**\n"
                for criterion in req['acceptance_criteria']:
                    md += f"- {criterion}\n"
            md += "\n"
        
        md += "---\n\n## Non-Functional Requirements\n\n"
        for req in requirements['non_functional_requirements']:
            md += f"### {req['id']}\n"
            md += f"**Description:** {req['description']}  \n"
            md += f"**Category:** {req['category']}  \n"
            md += f"**Priority:** {req['priority']}  \n\n"
        
        md += "---\n\n## Stakeholders\n\n"
        for stakeholder in requirements['stakeholders']:
            md += f"- {stakeholder}\n"
        
        md += "\n---\n\n## Assumptions\n\n"
        for assumption in requirements['assumptions']:
            md += f"- {assumption}\n"
        
        md += "\n---\n\n## Constraints\n\n"
        for constraint in requirements['constraints']:
            md += f"- {constraint}\n"
        
        return md


def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Business Requirements Generator",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Business Requirements Generator")
    st.markdown("Transform natural language descriptions into structured business requirements")
    
    # Check API key status
    api_key = OPENAI_API_KEY
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Generation method selector
        generation_method = st.radio(
            "Generation Method:",
            ["Template-based (Fast, No API needed)", "AI-Powered (OpenAI GPT-4)"],
            help="Template uses pattern matching. AI provides more intelligent extraction."
        )
        
        use_ai = "AI-Powered" in generation_method
        
        # API Key status
        if use_ai:
            if api_key:
                st.success("✅ OpenAI API Key loaded")
            else:
                st.error("❌ OpenAI API Key not found")
                with st.expander("ℹ️ How to add API Key"):
                    st.write("**For Local Development:**")
                    st.code("Create test.env file with:\nOPENAI_API_KEY=your-key")
                    st.write("**For Streamlit Cloud:**")
                    st.code("Add to App Settings > Secrets:\nOPENAI_API_KEY = \"your-key\"")
        
        st.divider()
        
        st.header("ℹ️ About")
        st.info(
            """
            This tool helps you create professional business requirements documents 
            from natural language input.
            
            **Features:**
            - Extract functional & non-functional requirements
            - Generate acceptance criteria
            - Validate requirement quality
            - Export to JSON & Markdown
            
            **AI-Powered Mode:**
            - More accurate extraction
            - Better context understanding
            - Comprehensive requirements
            """
        )
        
        st.divider()
        
        st.header("📝 Example")
        if st.button("Load Example", use_container_width=True):
            st.session_state.example_loaded = True
    
    # Load example text
    example_text = """We need to build a comprehensive customer relationship management (CRM) system for our sales team. The system must allow sales representatives to track leads, manage customer interactions, and monitor the sales pipeline from initial contact through to closing deals.

Users should be able to log customer calls, emails, and meetings with timestamps and notes. The system needs to provide dashboard views showing key metrics like conversion rates, average deal size, and sales forecast. Sales managers must be able to assign leads to team members and track individual performance.

The application should send automated email reminders for follow-ups and integrate with our existing email system. We need robust reporting capabilities with customizable filters and export to Excel functionality.

Security is critical - the system must have role-based access control, encrypted data storage, and audit logging. It should be accessible via web browser and mobile devices. Response time for page loads should be under 2 seconds.

The system needs to scale to support 500 concurrent users and store data for up to 100,000 customers. We have a 6-month development timeline and need to integrate with Salesforce and HubSpot APIs.

Our goal is to increase sales team productivity by 30% and improve lead conversion rates by 25% within the first year of deployment."""
    
    # Main content
    st.header("1. Enter Your Requirements")
    
    if 'example_loaded' in st.session_state and st.session_state.example_loaded:
        user_input = st.text_area(
            "Describe your project requirements in natural language:",
            value=example_text,
            height=250,
            help="Describe what you want to build, including features, goals, and constraints"
        )
        st.session_state.example_loaded = False
    else:
        user_input = st.text_area(
            "Describe your project requirements in natural language:",
            height=250,
            placeholder="Example: We need to build a mobile app that allows users to...",
            help="Describe what you want to build, including features, goals, and constraints"
        )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        generate_button = st.button("🚀 Generate Requirements", type="primary")
    
    with col2:
        add_criteria = st.checkbox("Add Acceptance Criteria", value=True)
    
    if generate_button and user_input:
        with st.spinner(f"Generating requirements using {'AI' if use_ai else 'template-based'} method..."):
            generator = BusinessRequirementsGenerator(use_ai=use_ai, api_key=api_key)
            
            # Generate requirements
            requirements = generator.generate_requirements(user_input)
            
            # Add acceptance criteria if requested
            if add_criteria:
                requirements = generator.add_acceptance_criteria_to_requirements(requirements)
            
            # Validate requirements
            validation_report = generator.validate_all_requirements(requirements)
            
            # Store in session state
            st.session_state.requirements = requirements
            st.session_state.validation_report = validation_report
            st.session_state.generated = True
            st.session_state.generation_method = "AI-Powered (OpenAI)" if use_ai else "Template-based"
    
    # Display results
    if 'generated' in st.session_state and st.session_state.generated:
        requirements = st.session_state.requirements
        validation_report = st.session_state.validation_report
        
        st.success(f"✅ Requirements generated successfully using {st.session_state.generation_method}!")
        
        # Validation Summary
        st.header("2. Validation Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🟢" if validation_report['overall_status'] == "Pass" else "🟡" if "Warning" in validation_report['overall_status'] else "🔴"
            st.metric("Overall Status", f"{status_color} {validation_report['overall_status']}")
        
        with col2:
            score = validation_report['average_score']
            st.metric("Quality Score", f"{score}/100")
        
        with col3:
            st.metric("Total Requirements", validation_report['total_requirements'])
        
        with col4:
            good_count = validation_report['summary']['good']
            st.metric("Good Quality", f"✓ {good_count}")
        
        # Quality breakdown
        st.subheader("Quality Breakdown")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"✓ Good: {validation_report['summary']['good']}")
        with col2:
            st.warning(f"⚠ Needs Improvement: {validation_report['summary']['needs_improvement']}")
        with col3:
            st.error(f"✗ Poor: {validation_report['summary']['poor']}")
        
        # Recommendations
        if validation_report['recommendations']:
            st.subheader("📝 Recommendations")
            for rec in validation_report['recommendations']:
                st.info(f"💡 {rec}")
        
        # Display Requirements
        st.header("3. Generated Requirements")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "⚙️ Functional", "🔧 Non-Functional", "👥 Stakeholders"])
        
        with tab1:
            st.subheader("Project Overview")
            st.write("**Description:**")
            st.write(requirements['project_overview']['description'])
            
            st.write("**Goals:**")
            for goal in requirements['project_overview']['goals']:
                st.write(f"- {goal}")
            
            if requirements['project_overview'].get('scope'):
                st.write("**Scope:**")
                st.write(requirements['project_overview']['scope'])
        
        with tab2:
            st.subheader("Functional Requirements")
            for req in requirements['functional_requirements']:
                with st.expander(f"{req['id']}: {req['description'][:60]}..."):
                    st.write(f"**Description:** {req['description']}")
                    
                    priority_color = "🔴" if req['priority'] == "High" else "🟡" if req['priority'] == "Medium" else "🟢"
                    st.write(f"**Priority:** {priority_color} {req['priority']}")
                    
                    if 'acceptance_criteria' in req:
                        st.write("**Acceptance Criteria:**")
                        for criterion in req['acceptance_criteria']:
                            st.write(f"- {criterion}")
                    
                    # Show validation for this requirement
                    validation = next((v for v in validation_report['functional_requirements_validation'] 
                                     if v['requirement_id'] == req['id']), None)
                    if validation:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            status_emoji = "✅" if validation['status'] == "Good" else "⚠️" if validation['status'] == "Needs Improvement" else "❌"
                            st.write(f"**Quality:** {status_emoji} {validation['score']}/100")
                        with col2:
                            if validation['warnings']:
                                for warning in validation['warnings']:
                                    st.caption(f"⚠️ {warning}")
        
        with tab3:
            st.subheader("Non-Functional Requirements")
            for req in requirements['non_functional_requirements']:
                with st.expander(f"{req['id']}: {req['category']}"):
                    st.write(f"**Description:** {req['description']}")
                    st.write(f"**Category:** {req['category']}")
                    
                    priority_color = "🔴" if req['priority'] == "High" else "🟡" if req['priority'] == "Medium" else "🟢"
                    st.write(f"**Priority:** {priority_color} {req['priority']}")
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Stakeholders")
                for stakeholder in requirements['stakeholders']:
                    st.write(f"- {stakeholder}")
                
                st.subheader("Assumptions")
                for assumption in requirements['assumptions']:
                    st.write(f"- {assumption}")
            
            with col2:
                st.subheader("Constraints")
                for constraint in requirements['constraints']:
                    st.write(f"- {constraint}")
        
        # Download Section
        st.header("4. Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        generator = BusinessRequirementsGenerator()
        
        with col1:
            # JSON download
            json_str = json.dumps(requirements, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="requirements.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Markdown download
            md_content = generator.export_to_markdown(requirements)
            st.download_button(
                label="📥 Download Markdown",
                data=md_content,
                file_name="requirements.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col3:
            # Validation report download
            validation_json = json.dumps(validation_report, indent=2)
            st.download_button(
                label="📥 Download Validation Report",
                data=validation_json,
                file_name="validation_report.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == "__main__":
    main()