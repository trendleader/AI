"""
Technical Specification Writing Scenarios
Focused on business writing for technical documents - specs, requirements, proposals.
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum


class Difficulty(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class ScenarioRequirement:
    name: str
    description: str
    check_function: str
    weight: float = 1.0


@dataclass
class WritingScenario:
    id: str
    title: str
    difficulty: Difficulty
    context: str
    task: str
    audience: str
    requirements: List[ScenarioRequirement]
    hints: List[str]
    min_words: int = 100
    max_words: int = 500
    example_topics: List[str] = field(default_factory=list)


SCENARIOS = [
    # =========================================================================
    # BEGINNER SCENARIOS
    # =========================================================================
    WritingScenario(
        id="feature_requirements",
        title="Feature Requirements Document",
        difficulty=Difficulty.BEGINNER,
        context="""Your product team has decided to add a password reset feature to the customer portal. 
        The VP of Engineering needs a clear requirements document to share with both the development 
        team and business stakeholders before approving the sprint planning.""",
        task="""Write a feature requirements document for a "Forgot Password" functionality.

The feature should include:
• Email-based password reset flow
• Security token with 24-hour expiration
• Rate limiting (max 3 requests per hour)
• Email notification when password is changed
• Audit logging for compliance

Your document should clearly communicate WHAT needs to be built and WHY, not HOW to build it.""",
        audience="VP of Engineering, Product Managers, and Development Team Lead",
        requirements=[
            ScenarioRequirement("business_context", "Explains the business need/problem being solved", "check_business_context"),
            ScenarioRequirement("clear_requirements", "Requirements are specific and unambiguous", "check_clear_requirements"),
            ScenarioRequirement("acceptance_criteria", "Includes measurable acceptance criteria", "check_acceptance_criteria"),
            ScenarioRequirement("stakeholder_appropriate", "Language appropriate for mixed technical/business audience", "check_stakeholder_language"),
            ScenarioRequirement("scope_defined", "Clearly defines what is in/out of scope", "check_scope"),
        ],
        hints=[
            "Start with the business problem, not the technical solution",
            "Write requirements as 'The system shall...' statements",
            "Include success metrics stakeholders care about",
            "Avoid implementation details - focus on outcomes",
            "Define scope boundaries explicitly"
        ],
        min_words=200,
        max_words=600,
    ),
    
    WritingScenario(
        id="technical_summary",
        title="Executive Technical Summary",
        difficulty=Difficulty.BEGINNER,
        context="""Your team completed a proof-of-concept for migrating the company's data warehouse 
        from on-premise SQL Server to cloud-based Snowflake. The CTO has asked for a one-page 
        summary to present to the executive leadership team who will decide on funding.""",
        task="""Write an executive summary of the data warehouse migration proof-of-concept.

Key findings from the POC:
• Query performance improved 3x on average
• Estimated 40% cost reduction over 3 years
• Migration timeline: 6 months with dedicated team of 4
• Risk: Some legacy reports need rewriting (estimated 15% of total)
• Cloud security meets SOC 2 compliance requirements

The executives need to understand the opportunity and make a go/no-go decision.""",
        audience="CTO, CFO, and Executive Leadership Team (limited technical background)",
        requirements=[
            ScenarioRequirement("executive_appropriate", "Written for executive audience, not engineers", "check_executive_language"),
            ScenarioRequirement("key_findings", "Highlights key findings prominently", "check_key_findings"),
            ScenarioRequirement("business_value", "Clearly articulates business value and ROI", "check_business_value"),
            ScenarioRequirement("risk_acknowledgment", "Acknowledges risks without burying them", "check_risk_acknowledgment"),
            ScenarioRequirement("clear_recommendation", "Includes clear recommendation or next steps", "check_recommendation"),
        ],
        hints=[
            "Lead with the business impact, not technical details",
            "Translate technical metrics into business value",
            "Present risks honestly but with mitigation plans",
            "Keep paragraphs short and scannable",
            "End with a clear call to action"
        ],
        min_words=200,
        max_words=500,
    ),
    
    WritingScenario(
        id="system_change_request",
        title="System Change Request",
        difficulty=Difficulty.BEGINNER,
        context="""The customer support team has reported that response times for the ticket 
        search feature have degraded significantly over the past month. After investigation, 
        the engineering team identified that adding a database index would resolve the issue. 
        You need to document this change for the Change Advisory Board (CAB).""",
        task="""Write a change request document for adding a database index to improve ticket search performance.

Technical details:
• Add composite index on tickets table (customer_id, created_date, status)
• Current avg query time: 4.2 seconds
• Expected query time after change: < 200ms
• Requires 15-minute maintenance window
• Rollback: Drop index (immediate, no data impact)

The CAB includes IT managers, security officers, and business representatives.""",
        audience="Change Advisory Board (mixed technical and business stakeholders)",
        requirements=[
            ScenarioRequirement("problem_statement", "Clearly states the problem and business impact", "check_problem_statement"),
            ScenarioRequirement("proposed_solution", "Explains the proposed solution clearly", "check_proposed_solution"),
            ScenarioRequirement("impact_assessment", "Assesses impact and risks", "check_impact_assessment"),
            ScenarioRequirement("rollback_plan", "Includes rollback/backout plan", "check_rollback_plan"),
            ScenarioRequirement("timeline_resources", "Specifies timeline and resource requirements", "check_timeline"),
        ],
        hints=[
            "Start with the business problem, not the technical fix",
            "Quantify the current impact (support tickets, user complaints)",
            "Be clear about downtime and user impact",
            "Make the rollback plan explicit and simple",
            "Include who approved the technical approach"
        ],
        min_words=200,
        max_words=500,
    ),

    # =========================================================================
    # INTERMEDIATE SCENARIOS
    # =========================================================================
    WritingScenario(
        id="system_design_spec",
        title="System Design Specification",
        difficulty=Difficulty.INTERMEDIATE,
        context="""Your company is building a new notification service to replace multiple 
        siloed notification systems across different products. The Architecture Review Board 
        needs a design specification to evaluate the proposal before allocating engineering 
        resources for Q3.""",
        task="""Write a system design specification for a unified notification service.

The service should support:
• Multiple channels: email, SMS, push notifications, in-app
• Template management with variable substitution
• User preference management (opt-in/opt-out per channel)
• Delivery tracking and analytics
• Rate limiting and throttling
• Integration via REST API and message queue

Address scalability (target: 1M notifications/day) and reliability requirements.""",
        audience="Architecture Review Board, Engineering Directors, and Platform Team",
        requirements=[
            ScenarioRequirement("problem_solution", "Clearly frames the problem and proposed solution", "check_problem_solution"),
            ScenarioRequirement("functional_requirements", "Documents functional requirements comprehensively", "check_functional_requirements"),
            ScenarioRequirement("non_functional_requirements", "Addresses non-functional requirements (scale, performance, reliability)", "check_non_functional"),
            ScenarioRequirement("integration_points", "Defines integration points and interfaces", "check_integration_points"),
            ScenarioRequirement("trade_offs", "Discusses design trade-offs and decisions", "check_trade_offs"),
            ScenarioRequirement("success_metrics", "Defines success metrics and KPIs", "check_success_metrics"),
        ],
        hints=[
            "Start with the problem statement and business drivers",
            "Separate functional from non-functional requirements",
            "Be explicit about what's NOT in scope",
            "Document key design decisions and alternatives considered",
            "Include measurable success criteria",
            "Consider failure modes and mitigation"
        ],
        min_words=350,
        max_words=800,
    ),
    
    WritingScenario(
        id="integration_spec",
        title="Third-Party Integration Specification",
        difficulty=Difficulty.INTERMEDIATE,
        context="""Your company is integrating with Stripe for payment processing to replace 
        the legacy payment system. The project requires coordination between Engineering, 
        Finance, Legal, and Security teams. You need to document the integration approach 
        for cross-functional alignment.""",
        task="""Write an integration specification for implementing Stripe payment processing.

Integration scope:
• Payment methods: Credit cards, ACH bank transfers
• Subscription billing for SaaS products
• Invoice generation and management
• Webhook handling for payment events
• PCI compliance requirements
• Refund and dispute handling

The Finance team needs to understand reporting changes, Security needs to approve the data flow, 
and Legal needs to verify compliance requirements are addressed.""",
        audience="Engineering Lead, Finance Director, Security Officer, Legal Counsel",
        requirements=[
            ScenarioRequirement("integration_overview", "Provides clear integration overview and scope", "check_integration_overview"),
            ScenarioRequirement("data_flow", "Documents data flows between systems", "check_data_flow"),
            ScenarioRequirement("security_compliance", "Addresses security and compliance requirements", "check_security_compliance"),
            ScenarioRequirement("cross_functional", "Addresses concerns of all stakeholder groups", "check_cross_functional"),
            ScenarioRequirement("migration_approach", "Outlines migration/transition approach", "check_migration_approach"),
            ScenarioRequirement("dependencies_risks", "Identifies dependencies and risks", "check_dependencies_risks"),
        ],
        hints=[
            "Address each stakeholder group's specific concerns",
            "Be explicit about data handling and PCI scope",
            "Document what changes for Finance/Accounting workflows",
            "Include rollback and contingency plans",
            "Clarify timeline dependencies clearly"
        ],
        min_words=350,
        max_words=800,
    ),
    
    WritingScenario(
        id="technical_proposal",
        title="Technical Solution Proposal",
        difficulty=Difficulty.INTERMEDIATE,
        context="""The Sales team has been losing deals because competitors offer real-time 
        analytics while your platform only updates dashboards daily. Product Management has 
        asked Engineering to propose a solution. Budget decisions will be made by the VP of 
        Product and CFO together.""",
        task="""Write a technical proposal for implementing real-time analytics capabilities.

Technical approach options considered:
• Option A: Stream processing with Kafka + Flink (most scalable, highest cost)
• Option B: Change data capture with Debezium (moderate complexity, moderate cost)  
• Option C: Frequent batch processing (simplest, limited real-time capability)

Current state: Daily batch ETL, 24-hour data latency
Target state: < 5 minute data latency for key metrics

Include cost-benefit analysis and your recommendation.""",
        audience="VP of Product, CFO, and Engineering Leadership",
        requirements=[
            ScenarioRequirement("business_case", "Establishes compelling business case", "check_business_case"),
            ScenarioRequirement("options_analysis", "Presents and analyzes multiple options", "check_options_analysis"),
            ScenarioRequirement("cost_benefit", "Includes cost-benefit analysis", "check_cost_benefit"),
            ScenarioRequirement("recommendation", "Makes clear, justified recommendation", "check_recommendation_justified"),
            ScenarioRequirement("implementation_plan", "Outlines implementation approach and timeline", "check_implementation_plan"),
            ScenarioRequirement("risk_mitigation", "Identifies risks with mitigation strategies", "check_risk_mitigation"),
        ],
        hints=[
            "Quantify the business problem (lost deals, revenue impact)",
            "Present options objectively before recommending",
            "Show total cost of ownership, not just build cost",
            "Connect recommendation to business priorities",
            "Include quick wins vs. long-term investment trade-offs"
        ],
        min_words=400,
        max_words=900,
    ),

    # =========================================================================
    # ADVANCED SCENARIOS
    # =========================================================================
    WritingScenario(
        id="architecture_decision_record",
        title="Architecture Decision Record (ADR)",
        difficulty=Difficulty.ADVANCED,
        context="""Your platform team is proposing to migrate from a monolithic architecture 
        to microservices. This is a significant architectural decision that will affect 
        development practices, team structure, and infrastructure costs for years. The 
        decision needs to be documented for the Architecture Review Board and future reference.""",
        task="""Write an Architecture Decision Record for adopting a microservices architecture.

Context:
• Current monolith: 500K lines of code, 12 developers, 2-week release cycles
• Pain points: Long build times (45 min), difficult to scale specific features, 
  team stepping on each other's code, single database bottleneck
• Target: Independent service deployments, team autonomy, selective scaling

Consider trade-offs around operational complexity, data consistency, and team readiness.
Document the decision in a way that future engineers can understand WHY this choice was made.""",
        audience="Architecture Review Board, Future Engineering Teams, CTO",
        requirements=[
            ScenarioRequirement("context_documented", "Documents the context and forces driving the decision", "check_context_forces"),
            ScenarioRequirement("decision_stated", "States the decision clearly and unambiguously", "check_decision_stated"),
            ScenarioRequirement("alternatives_considered", "Documents alternatives that were considered", "check_alternatives"),
            ScenarioRequirement("consequences_documented", "Documents both positive and negative consequences", "check_consequences"),
            ScenarioRequirement("rationale_clear", "Provides clear rationale for the decision", "check_rationale"),
            ScenarioRequirement("future_oriented", "Written for future readers who need to understand context", "check_future_oriented"),
        ],
        hints=[
            "Follow the standard ADR format: Context, Decision, Consequences",
            "Be honest about the downsides and costs",
            "Document what alternatives were rejected and why",
            "Include the constraints that influenced the decision",
            "Write as if explaining to someone who joins in 2 years"
        ],
        min_words=400,
        max_words=900,
    ),
    
    WritingScenario(
        id="vendor_evaluation",
        title="Vendor Evaluation and Recommendation",
        difficulty=Difficulty.ADVANCED,
        context="""Your company needs to select a cloud infrastructure provider for a new 
        product line. The decision involves significant 3-year commitment ($2M+ spend) and 
        will be reviewed by the CTO, CFO, and Procurement. Engineering has evaluated 
        AWS, Azure, and Google Cloud against your requirements.""",
        task="""Write a vendor evaluation document with your recommendation.

Evaluation criteria and findings:
• Compute/Container services: All three adequate; GCP strongest for Kubernetes
• Data services: AWS most mature; Azure best SQL Server integration
• AI/ML capabilities: GCP leads, AWS close second, Azure improving
• Pricing (3-year estimate): AWS $2.1M, Azure $1.9M, GCP $2.0M
• Enterprise support: Azure best (existing Microsoft relationship), AWS good, GCP limited
• Team expertise: 60% AWS experience, 30% Azure, 10% GCP
• Compliance: All meet SOC 2, HIPAA requirements

Current infrastructure: On-premise VMware + some Azure AD integration""",
        audience="CTO, CFO, Procurement Director, Enterprise Architecture Team",
        requirements=[
            ScenarioRequirement("evaluation_framework", "Establishes clear evaluation framework/criteria", "check_evaluation_framework"),
            ScenarioRequirement("objective_analysis", "Presents objective analysis of each option", "check_objective_analysis"),
            ScenarioRequirement("weighted_scoring", "Uses weighted scoring or structured comparison", "check_weighted_scoring"),
            ScenarioRequirement("total_cost", "Considers total cost of ownership beyond licensing", "check_total_cost"),
            ScenarioRequirement("strategic_alignment", "Addresses strategic fit and long-term implications", "check_strategic_alignment"),
            ScenarioRequirement("actionable_recommendation", "Provides clear, actionable recommendation", "check_actionable_recommendation"),
        ],
        hints=[
            "Define evaluation criteria before presenting findings",
            "Weight criteria based on business priorities",
            "Consider hidden costs: training, migration, operations",
            "Address the team expertise gap explicitly",
            "Make recommendation clear even if choice is close"
        ],
        min_words=450,
        max_words=1000,
    ),
    
    WritingScenario(
        id="incident_postmortem",
        title="Incident Post-Mortem Report",
        difficulty=Difficulty.ADVANCED,
        context="""Last week, a production outage took down the customer portal for 4 hours, 
        affecting 12,000 users during peak business hours. The CEO has asked for a detailed 
        post-mortem to share with the Board of Directors. The report needs to explain what 
        happened, demonstrate accountability, and show how it will be prevented.""",
        task="""Write a post-mortem report for the production outage.

Incident details:
• Duration: 4 hours 23 minutes (2:15 PM - 6:38 PM EST)
• Root cause: Database connection pool exhaustion after deployment of new feature
• Detection: Customer complaints (monitoring didn't alert for 47 minutes)
• Contributing factors: 
  - Load testing didn't simulate production traffic patterns
  - Connection pool settings not reviewed in deployment checklist
  - Monitoring threshold too high to catch gradual degradation
• Business impact: ~$180K estimated revenue impact, 847 support tickets

Write for a board-level audience while maintaining technical accuracy.""",
        audience="CEO, Board of Directors, VP of Engineering, Customer Success Leadership",
        requirements=[
            ScenarioRequirement("executive_summary", "Leads with clear executive summary", "check_executive_summary"),
            ScenarioRequirement("timeline_clear", "Provides clear incident timeline", "check_timeline_clear"),
            ScenarioRequirement("root_cause", "Explains root cause accurately but accessibly", "check_root_cause"),
            ScenarioRequirement("business_impact", "Quantifies business impact", "check_business_impact"),
            ScenarioRequirement("accountability", "Demonstrates accountability without blame", "check_accountability"),
            ScenarioRequirement("action_items", "Includes specific, assigned action items with dates", "check_action_items"),
            ScenarioRequirement("prevention_measures", "Outlines prevention measures for future", "check_prevention"),
        ],
        hints=[
            "Lead with impact and resolution, not technical details",
            "Use blameless language focused on systems, not individuals",
            "Make timeline scannable (use clear time markers)",
            "Action items need owners and due dates",
            "Show how monitoring/processes will prevent recurrence",
            "Acknowledge customer impact with empathy"
        ],
        min_words=450,
        max_words=1000,
    ),
]


def get_scenario_by_id(scenario_id: str) -> WritingScenario:
    """Get a scenario by its ID."""
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    return None


def get_scenarios_by_difficulty(difficulty: Difficulty) -> List[WritingScenario]:
    """Get all scenarios of a given difficulty."""
    return [s for s in SCENARIOS if s.difficulty == difficulty]
