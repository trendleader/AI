"""
AI Feedback Module
Provides intelligent, context-aware feedback using OpenAI's API.
"""

import os
from typing import Optional, Dict, List
from dataclasses import dataclass
from scenarios import WritingScenario


@dataclass
class AIFeedback:
    """Container for AI-generated feedback."""
    overall_assessment: str
    strengths: List[str]
    improvements: List[str]
    requirement_feedback: Dict[str, str]
    rewrite_suggestions: List[Dict[str, str]]
    audience_fit_score: int  # 1-10
    technical_accuracy_score: int  # 1-10
    clarity_score: int  # 1-10
    completeness_score: int  # 1-10


def get_ai_feedback(
    user_text: str,
    scenario: WritingScenario,
    api_key: Optional[str] = None
) -> Optional[AIFeedback]:
    """
    Get AI-powered feedback on the user's documentation.
    
    Args:
        user_text: The documentation written by the user
        scenario: The scenario being addressed
        api_key: OpenAI API key (uses env var if not provided)
    
    Returns:
        AIFeedback object with detailed analysis, or None if API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None
    
    # Get API key from parameter or environment
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    
    client = OpenAI(api_key=key)
    
    # Build the requirements list for the prompt
    requirements_list = "\n".join([
        f"- {req.description}" for req in scenario.requirements
    ])
    
    # Create the analysis prompt
    prompt = f"""You are an expert business writing coach specializing in technical specifications and business documentation. 
You help professionals write clear, compelling technical documents for mixed business and technical audiences.

## SCENARIO CONTEXT
Title: {scenario.title}
Difficulty: {scenario.difficulty.value}
Target Audience: {scenario.audience}

Context: {scenario.context}

Task: {scenario.task}

Requirements to evaluate:
{requirements_list}

Word count target: {scenario.min_words}-{scenario.max_words} words

## USER'S DOCUMENTATION
{user_text}

## YOUR TASK
Evaluate this business document for:
1. AUDIENCE APPROPRIATENESS - Is the language right for the stated audience? Too technical? Too vague?
2. CLARITY & CONCISENESS - Is it easy to understand? Free of jargon and wordiness?
3. STRUCTURE & FLOW - Is it well-organized? Does it guide the reader logically?
4. BUSINESS VALUE - Does it clearly articulate business impact and value?
5. ACTIONABILITY - Does it lead to clear decisions or next steps?
6. COMPLETENESS - Does it address all required elements?

Respond in the following JSON format exactly:
{{
    "overall_assessment": "2-3 sentence summary focusing on business communication effectiveness",
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "improvements": ["specific improvement 1", "specific improvement 2", "specific improvement 3"],
    "requirement_feedback": {{
        "requirement_name": "specific feedback on how well this requirement was met"
    }},
    "rewrite_suggestions": [
        {{
            "original": "problematic sentence or phrase from the text",
            "suggested": "improved version using clearer business writing",
            "reason": "why this is more effective for the audience"
        }}
    ],
    "audience_fit_score": 7,
    "technical_accuracy_score": 8,
    "clarity_score": 6,
    "completeness_score": 7
}}

For requirement_feedback, use these exact keys: {[req.name for req in scenario.requirements]}

Scores should be 1-10 where:
- 1-3: Poor - Would confuse stakeholders or fail to achieve document's purpose
- 4-5: Below average - Key information missing or poorly communicated  
- 6-7: Adequate - Gets the point across but could be clearer or more compelling
- 8-9: Good - Professional quality, minor polish needed
- 10: Excellent - Could be used as a template for others

Focus on business communication effectiveness, not just technical accuracy. 
Highlight where the writer succeeded in connecting technical content to business value."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert technical writing coach. Provide detailed, constructive feedback in valid JSON format only. No markdown, no code blocks, just pure JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Parse the response
        import json
        content = response.choices[0].message.content.strip()
        
        # Clean up potential markdown formatting
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        data = json.loads(content)
        
        return AIFeedback(
            overall_assessment=data.get("overall_assessment", ""),
            strengths=data.get("strengths", []),
            improvements=data.get("improvements", []),
            requirement_feedback=data.get("requirement_feedback", {}),
            rewrite_suggestions=data.get("rewrite_suggestions", []),
            audience_fit_score=data.get("audience_fit_score", 5),
            technical_accuracy_score=data.get("technical_accuracy_score", 5),
            clarity_score=data.get("clarity_score", 5),
            completeness_score=data.get("completeness_score", 5)
        )
        
    except Exception as e:
        print(f"AI Feedback error: {e}")
        return None


def get_ai_rewrite(
    original_text: str,
    scenario: WritingScenario,
    focus_area: str = "clarity",
    api_key: Optional[str] = None
) -> Optional[str]:
    """
    Get an AI-powered rewrite suggestion for a section of text.
    
    Args:
        original_text: The text to improve
        scenario: The scenario context
        focus_area: What to focus on (clarity, conciseness, technical_accuracy, structure)
        api_key: OpenAI API key
    
    Returns:
        Rewritten text or None if API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None
    
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    
    client = OpenAI(api_key=key)
    
    focus_instructions = {
        "clarity": "Make it clearer and easier to understand. Use simpler words and shorter sentences.",
        "conciseness": "Make it more concise. Remove unnecessary words and phrases while keeping all important information.",
        "technical_accuracy": "Improve technical precision. Use correct terminology and be more specific.",
        "structure": "Improve the organization and flow. Add better transitions and logical structure."
    }
    
    prompt = f"""Rewrite the following technical documentation excerpt to improve it.

Context: This is part of a {scenario.title} for {scenario.audience}.

Focus: {focus_instructions.get(focus_area, focus_instructions['clarity'])}

Original text:
{original_text}

Provide only the rewritten text, no explanations or preamble."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert technical writer. Provide only the improved text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"AI Rewrite error: {e}")
        return None


def get_example_solution(
    scenario: WritingScenario,
    api_key: Optional[str] = None
) -> Optional[str]:
    """
    Generate an example solution for a scenario.
    
    Args:
        scenario: The scenario to generate an example for
        api_key: OpenAI API key
    
    Returns:
        Example documentation or None if API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None
    
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    
    client = OpenAI(api_key=key)
    
    requirements_list = "\n".join([
        f"- {req.description}" for req in scenario.requirements
    ])
    
    prompt = f"""Write an excellent example of technical documentation for the following scenario.

## SCENARIO
Title: {scenario.title}
Target Audience: {scenario.audience}

Context: {scenario.context}

Task: {scenario.task}

Requirements to meet:
{requirements_list}

Word count: Aim for {scenario.min_words}-{scenario.max_words} words.

## INSTRUCTIONS
Write professional, clear technical documentation that:
1. Meets all the requirements listed above
2. Is appropriate for the target audience
3. Uses proper formatting (headers, code blocks, lists where appropriate)
4. Is concise but complete
5. Could serve as a model example for others to learn from

Write only the documentation, no meta-commentary."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior technical writer creating exemplary documentation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Example generation error: {e}")
        return None
