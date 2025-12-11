"""
Business Writing Analyzer for Technical Specifications
Evaluates documents for clarity, structure, audience-appropriateness, and professional quality.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from scenarios import WritingScenario


class IssueSeverity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    SUGGESTION = "suggestion"


@dataclass
class WritingIssue:
    category: str
    severity: IssueSeverity
    text: str
    suggestion: str
    context: str = ""
    position: int = 0


@dataclass
class DocumentAnalysis:
    spelling_issues: List[WritingIssue]
    grammar_issues: List[WritingIssue]
    punctuation_issues: List[WritingIssue]
    flow_issues: List[WritingIssue]
    clarity_issues: List[WritingIssue]
    requirement_scores: Dict[str, float]
    overall_score: float
    readability_score: float
    word_count: int
    sentence_count: int
    avg_sentence_length: float


class WritingAnalyzer:
    """Analyzes business writing for technical specifications."""
    
    # Common misspellings in business/technical writing
    COMMON_MISSPELLINGS = {
        'recieve': 'receive', 'occured': 'occurred', 'seperate': 'separate',
        'definately': 'definitely', 'occurance': 'occurrence', 'accomodate': 'accommodate',
        'acheive': 'achieve', 'apparant': 'apparent', 'begining': 'beginning',
        'beleive': 'believe', 'calender': 'calendar', 'catagory': 'category',
        'commited': 'committed', 'concensus': 'consensus', 'consistant': 'consistent',
        'dependant': 'dependent', 'enviroment': 'environment', 'existance': 'existence',
        'guarentee': 'guarantee', 'immediatly': 'immediately', 'independant': 'independent',
        'liason': 'liaison', 'maintainance': 'maintenance', 'neccessary': 'necessary',
        'noticable': 'noticeable', 'occurence': 'occurrence', 'persistant': 'persistent',
        'priviledge': 'privilege', 'recomend': 'recommend', 'refered': 'referred',
        'relevent': 'relevant', 'succesful': 'successful', 'threshhold': 'threshold',
        'untill': 'until', 'implimentation': 'implementation', 'developement': 'development',
        'paramater': 'parameter', 'configuation': 'configuration', 'dependancy': 'dependency',
        'accross': 'across', 'agressive': 'aggressive', 'alot': 'a lot',
        'analisis': 'analysis', 'arguement': 'argument', 'buisness': 'business',
        'collegue': 'colleague', 'completly': 'completely', 'efficent': 'efficient',
        'embarass': 'embarrass', 'experiance': 'experience', 'finaly': 'finally',
        'goverment': 'government', 'happend': 'happened', 'heirarchy': 'hierarchy',
        'knowlege': 'knowledge', 'lisence': 'license', 'managment': 'management',
        'millenium': 'millennium', 'mispell': 'misspell', 'neccessity': 'necessity',
        'occassion': 'occasion', 'paralell': 'parallel', 'percieve': 'perceive',
        'posession': 'possession', 'potental': 'potential', 'preceed': 'precede',
        'prefered': 'preferred', 'prevelant': 'prevalent', 'probaly': 'probably',
        'profesional': 'professional', 'realy': 'really', 'refering': 'referring',
        'reguardless': 'regardless', 'remeber': 'remember', 'repitition': 'repetition',
        'resistence': 'resistance', 'responsability': 'responsibility', 'scheldule': 'schedule',
        'similiar': 'similar', 'succede': 'succeed', 'supercede': 'supersede',
        'suprise': 'surprise', 'tecnical': 'technical', 'tommorow': 'tomorrow',
        'truely': 'truly', 'unfortunatly': 'unfortunately', 'untill': 'until',
        'usualy': 'usually', 'wierd': 'weird', 'writting': 'writing',
        'aknowledge': 'acknowledge', 'aquire': 'acquire', 'assesment': 'assessment',
        'benifit': 'benefit', 'challanges': 'challenges', 'commision': 'commission',
        'comparision': 'comparison', 'competance': 'competence', 'concious': 'conscious',
        'criterea': 'criteria', 'decison': 'decision', 'discription': 'description',
        'dissappoint': 'disappoint', 'enviromental': 'environmental', 'equiptment': 'equipment',
        'excercise': 'exercise', 'existant': 'existent', 'facillitate': 'facilitate',
        'fulfil': 'fulfill', 'furthur': 'further', 'guidence': 'guidance',
        'harrass': 'harass', 'identiy': 'identify', 'incorportate': 'incorporate',
        'infrastucture': 'infrastructure', 'initally': 'initially', 'inovation': 'innovation',
        'intergrate': 'integrate', 'interuption': 'interruption', 'judgement': 'judgment',
        'lable': 'label', 'lenght': 'length', 'likelyhood': 'likelihood',
        'maintance': 'maintenance', 'manuever': 'maneuver', 'milestones': 'milestones',
        'minumum': 'minimum', 'miscellaneous': 'miscellaneous', 'necesary': 'necessary',
        'negociate': 'negotiate', 'occuring': 'occurring', 'oppurtunity': 'opportunity',
        'optimium': 'optimum', 'orginal': 'original', 'performace': 'performance',
        'persistance': 'persistence', 'personel': 'personnel', 'planing': 'planning',
        'posible': 'possible', 'proceedure': 'procedure', 'programing': 'programming',
        'propogate': 'propagate', 'reciept': 'receipt', 'recognise': 'recognize',
        'reoccur': 'recur', 'requirment': 'requirement', 'resourse': 'resource',
        'retreive': 'retrieve', 'scaleable': 'scalable', 'senario': 'scenario',
        'specifiy': 'specify', 'statment': 'statement', 'stratagies': 'strategies',
        'strenght': 'strength', 'subsquent': 'subsequent', 'sucess': 'success',
        'summery': 'summary', 'teamplate': 'template', 'throught': 'through',
        'transfered': 'transferred', 'utilise': 'utilize', 'varient': 'variant',
        'visable': 'visible', 'vulnerabilty': 'vulnerability', 'waranty': 'warranty',
    }
    
    # Grammar patterns
    GRAMMAR_PATTERNS = [
        (r'\b(\w+)\s+\1\b', 'Repeated word detected'),
        (r'\ba\s+[aeiou]', 'Use "an" before vowel sounds'),
        (r'\ban\s+[^aeiou\s]', 'Use "a" before consonant sounds (check exceptions)'),
        (r'\b(should|could|would)\s+of\b', 'Use "have" instead of "of"'),
        (r'\b(irregardless)\b', 'Use "regardless" instead'),
        (r'\b(alot)\b', '"A lot" should be two words'),
        (r'\bits\s+important\s+to\s+note\b', 'Wordy - consider removing'),
        (r'\b(very|really|extremely)\s+(very|really|extremely)\b', 'Redundant intensifiers'),
    ]
    
    # Punctuation patterns
    PUNCTUATION_PATTERNS = [
        (r'\s+[,.]', 'Space before punctuation'),
        (r'[,]\s*[,]', 'Double comma'),
        (r'[!]\s*[!]+', 'Multiple exclamation marks - unprofessional in business writing'),
        (r'[?]\s*[?]+', 'Multiple question marks'),
        (r'\([^)]*$', 'Unclosed parenthesis'),
    ]
    
    # Business writing anti-patterns
    BUSINESS_WRITING_ISSUES = [
        # Wordiness
        (r'\b(in order to)\b', 'Wordy: Use "to" instead of "in order to"'),
        (r'\b(due to the fact that)\b', 'Wordy: Use "because" instead'),
        (r'\b(at this point in time)\b', 'Wordy: Use "now" or "currently"'),
        (r'\b(in the event that)\b', 'Wordy: Use "if" instead'),
        (r'\b(for the purpose of)\b', 'Wordy: Use "to" or "for" instead'),
        (r'\b(in spite of the fact that)\b', 'Wordy: Use "although" or "despite"'),
        (r'\b(with regard to|with respect to)\b', 'Wordy: Use "about" or "regarding"'),
        (r'\b(a large number of)\b', 'Wordy: Use "many" instead'),
        (r'\b(the majority of)\b', 'Wordy: Use "most" instead'),
        (r'\b(at the present time)\b', 'Wordy: Use "now" or "currently"'),
        (r'\b(it is important to note that)\b', 'Wordy: Remove and state the point directly'),
        (r'\b(it should be noted that)\b', 'Wordy: Remove and state the point directly'),
        (r'\b(needless to say)\b', 'If needless, remove it'),
        (r'\b(as a matter of fact)\b', 'Wordy: Use "in fact" or remove'),
        (r'\b(in the near future)\b', 'Vague: Specify the timeframe'),
        
        # Weak/Vague language
        (r'\b(very|really|quite|rather|somewhat)\b', 'Weak modifier: Use stronger, specific language'),
        (r'\b(thing|stuff)\b', 'Vague: Be more specific'),
        (r'\b(nice|good|bad)\b', 'Vague: Use more precise adjectives'),
        (r'\b(a lot)\b', 'Vague: Quantify when possible'),
        (r'\b(etc)\b', 'Vague: List specific items or use "and more"'),
        (r'\b(various|numerous|several)\b', 'Vague: Quantify when possible'),
        
        # Business jargon to avoid
        (r'\b(synergy|synergize|synergistic)\b', 'Jargon: Use clearer language'),
        (r'\b(leverage)\b', 'Jargon: Consider "use" or "apply"'),
        (r'\b(circle back)\b', 'Jargon: Use "follow up" or "revisit"'),
        (r'\b(deep dive)\b', 'Jargon: Use "detailed analysis" or "thorough review"'),
        (r'\b(move the needle)\b', 'Jargon: Use "make progress" or "improve"'),
        (r'\b(low-hanging fruit)\b', 'Jargon: Use "easy wins" or "quick improvements"'),
        (r'\b(boil the ocean)\b', 'Jargon: Be more specific about scope'),
        (r'\b(paradigm shift)\b', 'Overused: Use "fundamental change" or be specific'),
        (r'\b(best of breed)\b', 'Jargon: Use "leading" or specify criteria'),
        (r'\b(bandwidth)\b(?!.*(network|internet|data))', 'Jargon: Use "capacity" or "availability" for non-technical contexts'),
        (r'\b(ping)\b(?!.*(server|network))', 'Jargon: Use "contact" or "message"'),
        
        # Passive voice indicators (simplified)
        (r'\bwas\s+\w+ed\b', 'Consider active voice for clearer, more direct writing'),
        (r'\bwere\s+\w+ed\b', 'Consider active voice for clearer, more direct writing'),
        (r'\bbeen\s+\w+ed\b', 'Consider active voice for clearer, more direct writing'),
        
        # Hedging language
        (r'\b(I think|I believe|I feel)\b', 'Hedging: State conclusions confidently in business writing'),
        (r'\b(maybe|perhaps|possibly)\b', 'Hedging: Be more definitive or explain uncertainty'),
        (r'\b(sort of|kind of)\b', 'Hedging: Be more precise'),
    ]
    
    # Technical spec-specific patterns
    TECH_SPEC_PATTERNS = [
        (r'\b(easy|simple|straightforward|obviously|clearly|just)\b', 
         'Subjective: What seems easy may not be for all readers'),
        (r'\b(ASAP|as soon as possible)\b', 'Vague timeline: Specify a date or timeframe'),
        (r'\b(TBD|TBA)\b', 'Incomplete: Resolve before finalizing document'),
        (r'\b(should)\b(?!.*shall)', 'Ambiguous: Use "shall" for requirements, "should" for recommendations'),
    ]

    def __init__(self):
        self.issues = []
    
    def analyze(self, text: str, scenario: Optional[WritingScenario] = None) -> DocumentAnalysis:
        """Perform comprehensive analysis of the document."""
        self.issues = []
        
        words = text.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences) if sentences else 1
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        spelling_issues = self._check_spelling(text)
        grammar_issues = self._check_grammar(text)
        punctuation_issues = self._check_punctuation(text)
        flow_issues = self._check_business_writing(text)
        clarity_issues = self._check_clarity(text, sentences)
        
        requirement_scores = {}
        if scenario:
            requirement_scores = self._check_requirements(text, scenario)
        
        total_issues = len(spelling_issues) + len(grammar_issues) + len(punctuation_issues) + len(flow_issues)
        critical_issues = sum(1 for issues in [spelling_issues, grammar_issues, punctuation_issues, flow_issues, clarity_issues]
                            for issue in issues if issue.severity == IssueSeverity.CRITICAL)
        
        issue_score = max(0, 100 - (critical_issues * 5) - (total_issues * 2))
        
        if requirement_scores:
            req_avg = sum(requirement_scores.values()) / len(requirement_scores) * 100
            overall_score = (issue_score * 0.4) + (req_avg * 0.6)
        else:
            overall_score = issue_score
        
        readability_score = self._calculate_readability(text, avg_sentence_length)
        
        return DocumentAnalysis(
            spelling_issues=spelling_issues,
            grammar_issues=grammar_issues,
            punctuation_issues=punctuation_issues,
            flow_issues=flow_issues,
            clarity_issues=clarity_issues,
            requirement_scores=requirement_scores,
            overall_score=overall_score,
            readability_score=readability_score,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length
        )
    
    def _check_spelling(self, text: str) -> List[WritingIssue]:
        issues = []
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        found = set()
        
        for word in words:
            if word in self.COMMON_MISSPELLINGS and word not in found:
                found.add(word)
                issues.append(WritingIssue(
                    category="Spelling",
                    severity=IssueSeverity.CRITICAL,
                    text=word,
                    suggestion=f'"{word}" should be "{self.COMMON_MISSPELLINGS[word]}"',
                    context=self._get_context(text, word)
                ))
        
        return issues
    
    def _check_grammar(self, text: str) -> List[WritingIssue]:
        issues = []
        
        for pattern, description in self.GRAMMAR_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issues.append(WritingIssue(
                    category="Grammar",
                    severity=IssueSeverity.WARNING,
                    text=match.group(),
                    suggestion=description,
                    context=self._get_context(text, match.group()),
                    position=match.start()
                ))
        
        return issues[:10]
    
    def _check_punctuation(self, text: str) -> List[WritingIssue]:
        issues = []
        
        for pattern, description in self.PUNCTUATION_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                issues.append(WritingIssue(
                    category="Punctuation",
                    severity=IssueSeverity.WARNING,
                    text=match.group()[:50],
                    suggestion=description,
                    context=self._get_context(text, match.group())
                ))
        
        return issues[:10]
    
    def _check_business_writing(self, text: str) -> List[WritingIssue]:
        issues = []
        
        for pattern, description in self.BUSINESS_WRITING_ISSUES:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issues.append(WritingIssue(
                    category="Business Writing",
                    severity=IssueSeverity.SUGGESTION,
                    text=match.group(),
                    suggestion=description,
                    context=self._get_context(text, match.group())
                ))
        
        for pattern, description in self.TECH_SPEC_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                issues.append(WritingIssue(
                    category="Technical Writing",
                    severity=IssueSeverity.SUGGESTION,
                    text=match.group(),
                    suggestion=description,
                    context=self._get_context(text, match.group())
                ))
        
        return issues[:20]
    
    def _check_clarity(self, text: str, sentences: List[str]) -> List[WritingIssue]:
        issues = []
        
        for sentence in sentences:
            word_count = len(sentence.split())
            if word_count > 35:
                issues.append(WritingIssue(
                    category="Clarity",
                    severity=IssueSeverity.WARNING,
                    text=sentence[:80] + "..." if len(sentence) > 80 else sentence,
                    suggestion=f"Sentence has {word_count} words. Business writing should use shorter sentences (15-25 words) for clarity.",
                    context=sentence
                ))
            elif word_count > 28:
                issues.append(WritingIssue(
                    category="Clarity",
                    severity=IssueSeverity.SUGGESTION,
                    text=sentence[:80] + "..." if len(sentence) > 80 else sentence,
                    suggestion=f"Consider breaking this {word_count}-word sentence into two for better readability.",
                    context=sentence
                ))
        
        return issues[:10]
    
    def _check_requirements(self, text: str, scenario: WritingScenario) -> Dict[str, float]:
        """Check scenario-specific requirements for business writing."""
        scores = {}
        text_lower = text.lower()
        
        for req in scenario.requirements:
            score = self._evaluate_requirement(req.check_function, text, text_lower)
            scores[req.name] = score
        
        return scores
    
    def _evaluate_requirement(self, check_function: str, text: str, text_lower: str) -> float:
        """Evaluate a specific requirement."""
        
        # Business context and problem framing
        if check_function == "check_business_context":
            markers = ['business', 'problem', 'need', 'requirement', 'stakeholder', 'user', 'customer', 'goal', 'objective', 'challenge', 'opportunity']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_problem_statement":
            markers = ['problem', 'issue', 'challenge', 'impact', 'affect', 'cause', 'result', 'currently', 'today']
            found = sum(1 for m in markers if m in text_lower)
            has_quantification = bool(re.search(r'\d+\s*(%|percent|users|customers|hours|minutes|seconds)', text_lower))
            return min(1.0, (found / 3) + (0.3 if has_quantification else 0))
        
        elif check_function == "check_problem_solution":
            has_problem = any(m in text_lower for m in ['problem', 'challenge', 'issue', 'current', 'today'])
            has_solution = any(m in text_lower for m in ['solution', 'propose', 'recommend', 'approach', 'implement'])
            return 1.0 if (has_problem and has_solution) else 0.5 if (has_problem or has_solution) else 0.0
        
        # Requirements clarity
        elif check_function == "check_clear_requirements":
            shall_statements = len(re.findall(r'\b(shall|must|will)\b', text_lower))
            has_specifics = bool(re.search(r'\d+', text))
            return min(1.0, (shall_statements / 3) + (0.3 if has_specifics else 0))
        
        elif check_function == "check_functional_requirements":
            markers = ['shall', 'must', 'will', 'support', 'provide', 'enable', 'allow', 'function', 'feature', 'capability']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 4)
        
        elif check_function == "check_non_functional":
            markers = ['performance', 'scalability', 'reliability', 'availability', 'security', 'latency', 'throughput', 'uptime', 'sla', 'response time']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_acceptance_criteria":
            markers = ['acceptance', 'criteria', 'success', 'metric', 'measure', 'verify', 'validate', 'test', 'when', 'given', 'then']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        # Audience appropriateness
        elif check_function == "check_stakeholder_language":
            jargon_count = len(re.findall(r'\b(API|SDK|REST|JSON|SQL|HTTP|TCP|UDP|DNS|SSL|TLS)\b', text))
            has_explanations = any(m in text_lower for m in ['means', 'refers to', 'in other words', 'specifically'])
            if jargon_count > 5 and not has_explanations:
                return 0.4
            return 0.8 if jargon_count < 3 or has_explanations else 0.6
        
        elif check_function == "check_executive_language":
            business_terms = ['revenue', 'cost', 'roi', 'investment', 'value', 'benefit', 'risk', 'timeline', 'budget', 'resource']
            found = sum(1 for m in business_terms if m in text_lower)
            tech_jargon = len(re.findall(r'\b(API|SDK|REST|JSON|SQL|HTTP|microservice|kubernetes|docker)\b', text, re.IGNORECASE))
            score = min(1.0, found / 4) - (tech_jargon * 0.1)
            return max(0.0, score)
        
        # Scope and boundaries
        elif check_function == "check_scope":
            has_in_scope = any(m in text_lower for m in ['in scope', 'includes', 'will include', 'covers'])
            has_out_scope = any(m in text_lower for m in ['out of scope', 'excludes', 'does not include', 'will not', 'not included'])
            return 1.0 if (has_in_scope and has_out_scope) else 0.5 if has_in_scope else 0.2
        
        # Value and benefits
        elif check_function == "check_business_value":
            markers = ['value', 'benefit', 'roi', 'return', 'save', 'reduce', 'increase', 'improve', 'revenue', 'cost', 'efficiency']
            found = sum(1 for m in markers if m in text_lower)
            has_numbers = bool(re.search(r'\d+\s*(%|percent|\$|dollars|hours)', text_lower))
            return min(1.0, (found / 3) + (0.3 if has_numbers else 0))
        
        elif check_function == "check_business_case":
            markers = ['business', 'value', 'benefit', 'cost', 'roi', 'revenue', 'competitive', 'market', 'customer']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_key_findings":
            markers = ['finding', 'found', 'result', 'conclusion', 'key', 'important', 'significant', 'notable']
            found = sum(1 for m in markers if m in text_lower)
            has_numbers = bool(re.search(r'\d+', text))
            return min(1.0, (found / 2) + (0.3 if has_numbers else 0))
        
        # Risk and mitigation
        elif check_function == "check_risk_acknowledgment":
            markers = ['risk', 'challenge', 'concern', 'limitation', 'constraint', 'caveat', 'however', 'although']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        elif check_function == "check_risk_mitigation":
            has_risk = any(m in text_lower for m in ['risk', 'challenge', 'concern', 'issue'])
            has_mitigation = any(m in text_lower for m in ['mitigate', 'address', 'reduce', 'prevent', 'contingency', 'fallback'])
            return 1.0 if (has_risk and has_mitigation) else 0.5 if has_risk else 0.0
        
        # Recommendations and decisions
        elif check_function == "check_recommendation":
            markers = ['recommend', 'suggest', 'propose', 'next step', 'action', 'decision', 'proceed', 'approve']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        elif check_function == "check_recommendation_justified":
            has_recommendation = any(m in text_lower for m in ['recommend', 'suggest', 'propose'])
            has_justification = any(m in text_lower for m in ['because', 'since', 'therefore', 'as a result', 'given that', 'due to'])
            return 1.0 if (has_recommendation and has_justification) else 0.5 if has_recommendation else 0.0
        
        # Solution and implementation
        elif check_function == "check_proposed_solution":
            markers = ['solution', 'approach', 'proposal', 'implement', 'design', 'architecture', 'will', 'plan']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_implementation_plan":
            markers = ['phase', 'timeline', 'milestone', 'week', 'month', 'quarter', 'step', 'stage', 'sprint']
            found = sum(1 for m in markers if m in text_lower)
            has_dates = bool(re.search(r'(q[1-4]|january|february|march|april|may|june|july|august|september|october|november|december|\d{4})', text_lower))
            return min(1.0, (found / 2) + (0.3 if has_dates else 0))
        
        # Impact and assessment
        elif check_function == "check_impact_assessment":
            markers = ['impact', 'affect', 'risk', 'downtime', 'outage', 'user', 'customer', 'service']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_business_impact":
            markers = ['impact', 'revenue', 'cost', 'customer', 'user', 'affected', 'lost', 'damage']
            found = sum(1 for m in markers if m in text_lower)
            has_numbers = bool(re.search(r'\$?\d+', text))
            return min(1.0, (found / 3) + (0.3 if has_numbers else 0))
        
        # Rollback and contingency
        elif check_function == "check_rollback_plan":
            markers = ['rollback', 'revert', 'backout', 'undo', 'restore', 'fallback', 'contingency']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 1)
        
        # Timeline and resources
        elif check_function == "check_timeline":
            markers = ['timeline', 'schedule', 'deadline', 'date', 'week', 'month', 'hour', 'minute', 'duration']
            found = sum(1 for m in markers if m in text_lower)
            has_specifics = bool(re.search(r'\d+\s*(hour|minute|day|week|month)', text_lower))
            return min(1.0, (found / 2) + (0.3 if has_specifics else 0))
        
        # Integration and interfaces
        elif check_function == "check_integration_points":
            markers = ['integrate', 'interface', 'api', 'connect', 'endpoint', 'service', 'system', 'communication']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_integration_overview":
            markers = ['integration', 'scope', 'overview', 'approach', 'architecture', 'system', 'component']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_data_flow":
            markers = ['data', 'flow', 'transfer', 'send', 'receive', 'input', 'output', 'request', 'response']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        # Security and compliance
        elif check_function == "check_security_compliance":
            markers = ['security', 'compliance', 'pci', 'soc', 'hipaa', 'gdpr', 'encrypt', 'audit', 'access', 'authentication']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        # Trade-offs and alternatives
        elif check_function == "check_trade_offs":
            markers = ['trade-off', 'tradeoff', 'however', 'although', 'versus', 'compared', 'alternative', 'option', 'consideration']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        elif check_function == "check_alternatives":
            markers = ['alternative', 'option', 'considered', 'rejected', 'versus', 'compared', 'instead']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        elif check_function == "check_options_analysis":
            markers = ['option', 'alternative', 'approach', 'compare', 'versus', 'analysis', 'evaluate']
            found = sum(1 for m in markers if m in text_lower)
            has_multiple = bool(re.search(r'option\s*[abc123]|alternative\s*[abc123]', text_lower))
            return min(1.0, (found / 3) + (0.3 if has_multiple else 0))
        
        # Cost and financial
        elif check_function == "check_cost_benefit":
            markers = ['cost', 'benefit', 'roi', 'investment', 'save', 'expense', 'budget', 'price', 'value']
            found = sum(1 for m in markers if m in text_lower)
            has_numbers = bool(re.search(r'\$\d+|\d+%', text))
            return min(1.0, (found / 3) + (0.3 if has_numbers else 0))
        
        elif check_function == "check_total_cost":
            markers = ['total cost', 'tco', 'ownership', 'maintenance', 'operational', 'training', 'migration', 'ongoing']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        # Success metrics
        elif check_function == "check_success_metrics":
            markers = ['metric', 'kpi', 'measure', 'success', 'target', 'goal', 'benchmark', 'performance']
            found = sum(1 for m in markers if m in text_lower)
            has_numbers = bool(re.search(r'\d+\s*(%|percent|ms|seconds|per)', text_lower))
            return min(1.0, (found / 2) + (0.3 if has_numbers else 0))
        
        # Cross-functional
        elif check_function == "check_cross_functional":
            markers = ['finance', 'legal', 'security', 'engineering', 'product', 'operations', 'compliance', 'team', 'stakeholder']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        # Migration and dependencies
        elif check_function == "check_migration_approach":
            markers = ['migration', 'transition', 'cutover', 'rollout', 'phase', 'parallel', 'pilot']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        elif check_function == "check_dependencies_risks":
            has_deps = any(m in text_lower for m in ['dependency', 'depends', 'requires', 'prerequisite', 'blocker'])
            has_risks = any(m in text_lower for m in ['risk', 'concern', 'issue', 'challenge'])
            return 1.0 if (has_deps and has_risks) else 0.5 if (has_deps or has_risks) else 0.0
        
        # ADR-specific
        elif check_function == "check_context_forces":
            markers = ['context', 'background', 'situation', 'currently', 'today', 'force', 'constraint', 'driver']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_decision_stated":
            markers = ['decision', 'decided', 'will', 'adopt', 'choose', 'select', 'implement']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        elif check_function == "check_consequences":
            markers = ['consequence', 'result', 'impact', 'effect', 'benefit', 'drawback', 'positive', 'negative', 'trade-off']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_rationale":
            markers = ['because', 'reason', 'rationale', 'why', 'since', 'therefore', 'given', 'due to']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        elif check_function == "check_future_oriented":
            markers = ['future', 'later', 'eventually', 'long-term', 'sustainable', 'maintainable', 'scalable']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        # Vendor evaluation
        elif check_function == "check_evaluation_framework":
            markers = ['criteria', 'framework', 'evaluate', 'assess', 'compare', 'score', 'weight', 'factor']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_objective_analysis":
            markers = ['analysis', 'compare', 'versus', 'strength', 'weakness', 'pro', 'con', 'finding']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 3)
        
        elif check_function == "check_weighted_scoring":
            markers = ['score', 'weight', 'rank', 'priority', 'criteria', 'rating', 'point']
            found = sum(1 for m in markers if m in text_lower)
            has_numbers = bool(re.search(r'\d+\s*(point|score|%|/10|/5)', text_lower))
            return min(1.0, (found / 2) + (0.3 if has_numbers else 0))
        
        elif check_function == "check_strategic_alignment":
            markers = ['strategic', 'align', 'vision', 'roadmap', 'long-term', 'future', 'growth', 'direction']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        elif check_function == "check_actionable_recommendation":
            markers = ['recommend', 'action', 'next step', 'proceed', 'approve', 'decision', 'select', 'choose']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        # Post-mortem specific
        elif check_function == "check_executive_summary":
            has_summary = any(m in text_lower for m in ['summary', 'overview', 'executive'])
            is_early = text_lower[:500].find('summary') != -1 or text_lower[:500].find('overview') != -1
            return 1.0 if (has_summary and is_early) else 0.5 if has_summary else 0.0
        
        elif check_function == "check_timeline_clear":
            time_markers = len(re.findall(r'\d{1,2}:\d{2}|\d{1,2}\s*(am|pm)|(\d{1,2}/\d{1,2})', text_lower))
            has_sequence = any(m in text_lower for m in ['first', 'then', 'next', 'finally', 'subsequently'])
            return min(1.0, (time_markers / 3) + (0.3 if has_sequence else 0))
        
        elif check_function == "check_root_cause":
            markers = ['root cause', 'cause', 'reason', 'because', 'due to', 'resulted from', 'triggered by']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        elif check_function == "check_accountability":
            has_ownership = any(m in text_lower for m in ['responsibility', 'accountable', 'owner', 'team'])
            has_blameless = not any(m in text_lower for m in ['fault', 'blame', 'mistake by', 'error by'])
            return 1.0 if (has_ownership and has_blameless) else 0.5 if has_ownership else 0.3
        
        elif check_function == "check_action_items":
            markers = ['action', 'task', 'todo', 'will', 'assign', 'owner', 'due', 'deadline', 'by']
            found = sum(1 for m in markers if m in text_lower)
            has_dates = bool(re.search(r'(by|due|before)\s*\w+\s*\d+', text_lower))
            return min(1.0, (found / 3) + (0.3 if has_dates else 0))
        
        elif check_function == "check_prevention":
            markers = ['prevent', 'avoid', 'improve', 'enhance', 'monitor', 'alert', 'detect', 'safeguard', 'future']
            found = sum(1 for m in markers if m in text_lower)
            return min(1.0, found / 2)
        
        # Default
        return 0.5
    
    def _get_context(self, text: str, target: str, context_chars: int = 50) -> str:
        idx = text.lower().find(target.lower())
        if idx == -1:
            return target
        
        start = max(0, idx - context_chars)
        end = min(len(text), idx + len(target) + context_chars)
        
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context
    
    def _calculate_readability(self, text: str, avg_sentence_length: float) -> float:
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 0
        
        total_syllables = sum(self._count_syllables(word) for word in words)
        avg_syllables = total_syllables / len(words) if words else 0
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        vowels = "aeiou"
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
