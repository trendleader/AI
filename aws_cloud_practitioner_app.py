"""
AWS Cloud Practitioner Study App
Full-featured study app with flashcards, study notes, and practice exams
"""

import streamlit as st
import random
import json
import time
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# DATA: FLASHCARDS
# ─────────────────────────────────────────────────────────────

FLASHCARD_DECKS = {
    "Core Services": [
        {"front": "Amazon S3", "back": "Simple Storage Service — object storage with 99.999999999% durability. Stores files as objects in buckets. Use cases: backups, static websites, data lakes.", "category": "Storage"},
        {"front": "Amazon EC2", "back": "Elastic Compute Cloud — resizable virtual servers (instances) in the cloud. You choose instance type, OS, and size. Pay per hour/second.", "category": "Compute"},
        {"front": "Amazon RDS", "back": "Relational Database Service — managed relational DB (MySQL, PostgreSQL, Oracle, SQL Server, MariaDB, Aurora). Handles backups, patching, HA automatically.", "category": "Database"},
        {"front": "Amazon VPC", "back": "Virtual Private Cloud — logically isolated section of AWS cloud. You define IP ranges, subnets, route tables, and gateways.", "category": "Networking"},
        {"front": "AWS Lambda", "back": "Serverless compute — run code without provisioning servers. Triggered by events. Pay only for execution time (per ms). Max 15 min runtime.", "category": "Compute"},
        {"front": "Amazon CloudFront", "back": "Content Delivery Network (CDN) — delivers content globally through 400+ edge locations. Reduces latency, integrates with S3, EC2, ALB.", "category": "Networking"},
        {"front": "Amazon DynamoDB", "back": "Fully managed NoSQL key-value and document database. Single-digit millisecond performance at any scale. Serverless, auto-scaling.", "category": "Database"},
        {"front": "Amazon EBS", "back": "Elastic Block Store — persistent block storage volumes for EC2. Like a hard drive you attach to an instance. Survives instance stop/start.", "category": "Storage"},
        {"front": "AWS IAM", "back": "Identity and Access Management — control who (authentication) and what (authorization) can access AWS resources. Users, Groups, Roles, Policies.", "category": "Security"},
        {"front": "Amazon SQS", "back": "Simple Queue Service — fully managed message queuing. Decouple application components. Standard (at-least-once) and FIFO (exactly-once) queues.", "category": "Application Integration"},
        {"front": "Amazon SNS", "back": "Simple Notification Service — pub/sub messaging. Fan-out messages to SQS, Lambda, HTTP, email, SMS. Topic-based.", "category": "Application Integration"},
        {"front": "Amazon Route 53", "back": "Scalable DNS and domain registration service. Routes users to AWS or on-premises endpoints. Health checks and failover routing.", "category": "Networking"},
        {"front": "Elastic Load Balancing (ELB)", "back": "Automatically distributes incoming traffic across multiple targets. Types: ALB (HTTP/HTTPS), NLB (TCP/UDP), CLB (legacy), GWLB.", "category": "Networking"},
        {"front": "Amazon ECS", "back": "Elastic Container Service — fully managed container orchestration for Docker. Run containers on EC2 or Fargate (serverless).", "category": "Compute"},
        {"front": "Amazon EKS", "back": "Elastic Kubernetes Service — managed Kubernetes. Deploy and scale containerized apps using Kubernetes without managing the control plane.", "category": "Compute"},
        {"front": "AWS Elastic Beanstalk", "back": "PaaS — deploy and manage web apps without managing infrastructure. Upload code, Beanstalk handles provisioning, load balancing, auto scaling.", "category": "Compute"},
        {"front": "Amazon Aurora", "back": "MySQL/PostgreSQL-compatible relational DB built for the cloud. 5x faster than MySQL, 3x faster than PostgreSQL. Auto-scales storage up to 128TB.", "category": "Database"},
        {"front": "Amazon ElastiCache", "back": "Managed in-memory caching (Redis or Memcached). Improves app performance by retrieving data from fast in-memory caches.", "category": "Database"},
        {"front": "AWS CloudTrail", "back": "Records API calls and actions in your AWS account for auditing and compliance. Stores logs in S3. Enabled by default for 90 days.", "category": "Security"},
        {"front": "Amazon CloudWatch", "back": "Monitoring and observability service. Collect metrics, logs, events. Set alarms. Dashboard visualization. Monitor AWS resources and applications.", "category": "Management"},
    ],
    "Cloud Concepts": [
        {"front": "What is Cloud Computing?", "back": "On-demand delivery of IT resources over the internet with pay-as-you-go pricing. No need to buy/own hardware.", "category": "Concepts"},
        {"front": "6 Advantages of Cloud", "back": "1) Trade fixed for variable expense\n2) Benefit from massive economies of scale\n3) Stop guessing capacity\n4) Increase speed and agility\n5) Stop spending on data centers\n6) Go global in minutes", "category": "Concepts"},
        {"front": "3 Cloud Deployment Models", "back": "• Cloud (Public): Fully deployed in cloud\n• Hybrid: Connect cloud to on-premises\n• On-premises (Private Cloud): Deploy in your own data center using cloud-like tools", "category": "Concepts"},
        {"front": "IaaS", "back": "Infrastructure as a Service — you manage OS and above. Provider manages hardware, networking, virtualization.\nExample: Amazon EC2", "category": "Concepts"},
        {"front": "PaaS", "back": "Platform as a Service — you manage applications and data. Provider manages OS, runtime, middleware.\nExample: AWS Elastic Beanstalk, RDS", "category": "Concepts"},
        {"front": "SaaS", "back": "Software as a Service — provider manages everything. You just use the software.\nExample: Gmail, Salesforce, AWS Rekognition", "category": "Concepts"},
        {"front": "AWS Shared Responsibility Model", "back": "AWS: Security OF the cloud (hardware, data centers, managed services)\nCustomer: Security IN the cloud (data, IAM, encryption, OS patching, network config)", "category": "Security"},
        {"front": "AWS Well-Architected Framework Pillars", "back": "1) Operational Excellence\n2) Security\n3) Reliability\n4) Performance Efficiency\n5) Cost Optimization\n6) Sustainability", "category": "Concepts"},
        {"front": "High Availability", "back": "System remains operational and accessible despite component failures. Achieved via redundancy, load balancing, multi-AZ deployments.", "category": "Concepts"},
        {"front": "Scalability vs Elasticity", "back": "Scalability: Ability to handle increased load by adding resources.\nElasticity: Ability to automatically scale resources up AND down based on demand.", "category": "Concepts"},
        {"front": "Fault Tolerance", "back": "System continues operating even when components fail. Built with redundancy. More robust than high availability.", "category": "Concepts"},
        {"front": "AWS Global Infrastructure", "back": "Regions: Geographic areas with 2+ AZs\nAvailability Zones (AZs): Isolated data centers within a region\nEdge Locations: CDN endpoints (CloudFront)\nLocal Zones: Extend AWS to metro areas", "category": "Concepts"},
        {"front": "Total Cost of Ownership (TCO)", "back": "Comparison of on-premises costs vs cloud. Cloud reduces: hardware, data center space, power, cooling, staff. AWS TCO Calculator helps estimate savings.", "category": "Concepts"},
        {"front": "CapEx vs OpEx", "back": "CapEx (Capital Expenditure): Upfront investment in physical assets (on-prem servers)\nOpEx (Operating Expenditure): Pay-as-you-go ongoing costs (cloud computing)", "category": "Concepts"},
        {"front": "Economies of Scale", "back": "Because AWS buys massive amounts of hardware, their per-unit costs are lower, so they pass savings to customers via lower prices.", "category": "Concepts"},
    ],
    "Security & Compliance": [
        {"front": "AWS IAM Best Practices", "back": "• Enable MFA for root\n• Create individual IAM users\n• Use groups for permissions\n• Grant least privilege\n• Use roles for applications\n• Rotate credentials regularly\n• Never share root credentials", "category": "Security"},
        {"front": "AWS Shield", "back": "DDoS protection service.\nStandard: Free, automatic, basic DDoS protection.\nAdvanced: $3,000/month, enhanced protection, 24/7 DRT support, cost protection.", "category": "Security"},
        {"front": "AWS WAF", "back": "Web Application Firewall — filter web traffic based on rules. Protects against SQL injection, XSS, bad bots. Works with CloudFront, ALB, API Gateway.", "category": "Security"},
        {"front": "AWS KMS", "back": "Key Management Service — create, manage, and control cryptographic keys for data encryption. Integrates with most AWS services.", "category": "Security"},
        {"front": "Amazon Inspector", "back": "Automated security assessment service. Finds vulnerabilities in EC2 instances and container images. Network reachability analysis.", "category": "Security"},
        {"front": "Amazon GuardDuty", "back": "Intelligent threat detection. Analyzes CloudTrail, VPC Flow Logs, DNS logs. Detects malicious activity and unauthorized behavior.", "category": "Security"},
        {"front": "AWS Artifact", "back": "Free, self-service portal for on-demand access to AWS compliance reports and select agreements. SOC, ISO, PCI, HIPAA reports.", "category": "Compliance"},
        {"front": "AWS Config", "back": "Assess, audit, and evaluate AWS resource configurations over time. Continuous monitoring, compliance auditing, change management.", "category": "Security"},
        {"front": "Amazon Macie", "back": "Uses ML to automatically discover, classify, and protect sensitive data (PII) in S3. Helps with data privacy and compliance.", "category": "Security"},
        {"front": "AWS Security Hub", "back": "Centralized security findings across AWS accounts and services. Aggregates alerts from GuardDuty, Inspector, Macie, and more.", "category": "Security"},
        {"front": "Principle of Least Privilege", "back": "Grant only the minimum permissions required to perform a task. Do not give broad access. Core IAM best practice.", "category": "Security"},
        {"front": "MFA (Multi-Factor Authentication)", "back": "Adds extra layer of protection beyond username/password. Virtual MFA (Google Authenticator), hardware MFA (YubiKey), SMS. Required for root account.", "category": "Security"},
    ],
    "Billing & Pricing": [
        {"front": "AWS Pricing Fundamentals", "back": "Three fundamentals:\n1) Compute: charged per hour/second\n2) Storage: charged per GB stored\n3) Data Transfer: charged per GB OUT (inbound usually free)", "category": "Pricing"},
        {"front": "EC2 Pricing Models", "back": "• On-Demand: Pay by hour/second, no commitment\n• Reserved: 1-3 year term, up to 72% savings\n• Spot: Bid on unused capacity, up to 90% savings\n• Dedicated Host: Physical server for compliance\n• Savings Plans: Flexible commitment for compute savings", "category": "Pricing"},
        {"front": "AWS Free Tier", "back": "12-month free: EC2 (750hrs t2.micro), S3 (5GB), RDS (750hrs)\nAlways Free: Lambda (1M requests/mo), DynamoDB (25GB), CloudFront (1TB)\nTrials: Short-term free trials for specific services", "category": "Pricing"},
        {"front": "AWS Cost Explorer", "back": "Visualize, understand, and manage AWS costs and usage over time. Forecast future costs, identify cost drivers, right-size recommendations.", "category": "Billing"},
        {"front": "AWS Budgets", "back": "Set custom cost and usage budgets. Get alerts when costs/usage exceed (or are forecast to exceed) thresholds. Can trigger actions.", "category": "Billing"},
        {"front": "AWS Cost & Usage Report (CUR)", "back": "Most comprehensive billing data. Detailed hourly/daily/monthly data. Integrates with Athena, S3, QuickSight for analysis.", "category": "Billing"},
        {"front": "AWS Pricing Calculator", "back": "Estimate the cost of AWS services before using them. Model your solution, add services, see monthly/annual cost estimates.", "category": "Pricing"},
        {"front": "AWS Organizations", "back": "Centrally manage multiple AWS accounts. Consolidated billing (single bill for all accounts), volume discounts, SCPs (Service Control Policies) for governance.", "category": "Billing"},
        {"front": "Consolidated Billing", "back": "Feature of AWS Organizations. One bill for all accounts, share Reserved Instance discounts and Savings Plans across accounts, potential volume discounts.", "category": "Billing"},
        {"front": "Reserved Instances Types", "back": "• Standard RI: Biggest discount (up to 72%), can't change instance type\n• Convertible RI: Lower discount, can change instance family/OS\n• Scheduled RI: Recurring windows (deprecated)", "category": "Pricing"},
    ],
    "Support & Governance": [
        {"front": "AWS Support Plans", "back": "• Basic: Free, billing/account support\n• Developer: $29/mo, business-hours email, 1 contact\n• Business: $100/mo, 24/7 phone/chat, all checks\n• Enterprise On-Ramp: $5,500/mo, pool of TAMs\n• Enterprise: $15,000/mo, dedicated TAM, concierge", "category": "Support"},
        {"front": "AWS Trusted Advisor", "back": "Inspects your environment and recommends improvements across:\n• Cost Optimization\n• Performance\n• Security\n• Fault Tolerance\n• Service Limits\nFull checks require Business/Enterprise support.", "category": "Management"},
        {"front": "AWS Personal Health Dashboard", "back": "Personalized view of AWS service health events that may affect your account. Proactive notifications, remediation guidance.", "category": "Management"},
        {"front": "AWS Service Health Dashboard", "back": "Public page showing real-time status of all AWS services across all regions. Shows historical incidents. health.aws.amazon.com", "category": "Management"},
        {"front": "AWS CloudFormation", "back": "Infrastructure as Code (IaC) — provision and manage AWS resources using templates (JSON/YAML). Repeatable, version-controlled infrastructure.", "category": "Management"},
        {"front": "AWS Systems Manager", "back": "Visibility and control of your infrastructure at scale. Run commands, patch management, parameter store, session manager (no SSH needed).", "category": "Management"},
        {"front": "AWS Control Tower", "back": "Set up and govern a secure, multi-account AWS environment (Landing Zone). Implements guardrails, account factory, dashboard.", "category": "Management"},
        {"front": "Service Control Policies (SCPs)", "back": "Feature of AWS Organizations. JSON policies that set maximum permissions for accounts in an OU. Cannot grant permissions, only restrict them.", "category": "Governance"},
        {"front": "AWS License Manager", "back": "Manage software licenses from vendors (Microsoft, SAP, Oracle). Track usage, enforce rules, prevent license violations.", "category": "Management"},
        {"front": "AWS Marketplace", "back": "Digital catalog with thousands of software listings from independent vendors. Buy and deploy software to AWS. Pay via AWS bill.", "category": "Management"},
    ],
    "Migration & Innovation": [
        {"front": "AWS Snow Family", "back": "Physical devices for large data migrations:\n• Snowcone: 8TB, portable\n• Snowball Edge: 80TB, compute+storage\n• Snowmobile: 100PB, truck-sized\nUse when internet transfer would take too long.", "category": "Migration"},
        {"front": "AWS Migration Hub", "back": "Central hub to track application migrations across AWS and partner migration tools. Single pane of glass for migration progress.", "category": "Migration"},
        {"front": "AWS Database Migration Service (DMS)", "back": "Migrate databases to AWS quickly and securely. Source stays operational during migration. Supports homogeneous and heterogeneous migrations.", "category": "Migration"},
        {"front": "AWS Application Migration Service (MGN)", "back": "Lift-and-shift migration service. Replicates servers to AWS, then cut over. Formerly CloudEndure Migration.", "category": "Migration"},
        {"front": "6 Rs of Migration", "back": "• Rehost (lift & shift)\n• Replatform (lift, tinker & shift)\n• Repurchase (drop & shop)\n• Refactor/Re-architect\n• Retire (turn off)\n• Retain (keep on-prem)", "category": "Migration"},
        {"front": "AWS DataSync", "back": "Online data transfer service. Automate and accelerate moving data between on-premises storage and AWS. Up to 10x faster than open-source tools.", "category": "Migration"},
        {"front": "AWS IoT Core", "back": "Managed cloud service for IoT devices to connect to AWS cloud. Secure, bi-directional communication for billions of devices.", "category": "Innovation"},
        {"front": "Amazon SageMaker", "back": "Fully managed ML platform. Build, train, and deploy ML models at scale. Integrated Jupyter notebooks, built-in algorithms, one-click deployment.", "category": "Innovation"},
        {"front": "AWS Rekognition", "back": "AI/ML image and video analysis. Object detection, facial recognition, content moderation, text detection. No ML expertise needed.", "category": "Innovation"},
        {"front": "Amazon Lex", "back": "Build conversational interfaces (chatbots, voice bots). Same technology that powers Alexa. NLP and speech recognition.", "category": "Innovation"},
    ],
}

# ─────────────────────────────────────────────────────────────
# DATA: PRACTICE EXAM QUESTIONS
# ─────────────────────────────────────────────────────────────

PRACTICE_QUESTIONS = [
    # Cloud Concepts
    {
        "domain": "Cloud Concepts",
        "difficulty": "Easy",
        "question": "Which of the following best describes a benefit of cloud computing?",
        "options": [
            "Upfront capital expenditure for infrastructure",
            "Fixed capacity that cannot be changed",
            "Trade capital expense for variable (operational) expense",
            "Requires dedicated hardware for each customer"
        ],
        "correct": 2,
        "explanation": "Cloud computing allows you to trade capital expenses (buying hardware) for variable/operational expenses (paying only for what you use). This eliminates large upfront investments."
    },
    {
        "domain": "Cloud Concepts",
        "difficulty": "Easy",
        "question": "What does 'elasticity' mean in the context of AWS Cloud?",
        "options": [
            "The ability to store large amounts of data",
            "The ability to automatically scale resources up or down based on demand",
            "The durability of data stored in the cloud",
            "The geographic distribution of data centers"
        ],
        "correct": 1,
        "explanation": "Elasticity refers to the ability to automatically acquire and release resources based on current demand. This prevents over-provisioning while ensuring capacity when needed."
    },
    {
        "domain": "Cloud Concepts",
        "difficulty": "Medium",
        "question": "A company is evaluating moving to AWS. Which pillar of the AWS Well-Architected Framework focuses on protecting information and systems?",
        "options": [
            "Reliability",
            "Performance Efficiency",
            "Security",
            "Operational Excellence"
        ],
        "correct": 2,
        "explanation": "The Security pillar of the AWS Well-Architected Framework focuses on protecting information, systems, and assets while delivering business value through risk assessments and mitigation strategies."
    },
    {
        "domain": "Cloud Concepts",
        "difficulty": "Medium",
        "question": "Which cloud deployment model connects on-premises infrastructure with AWS cloud resources?",
        "options": [
            "Public Cloud",
            "Private Cloud",
            "Hybrid Cloud",
            "Community Cloud"
        ],
        "correct": 2,
        "explanation": "A Hybrid Cloud deployment model connects on-premises (private) infrastructure with public cloud resources (AWS), allowing data and applications to be shared between them."
    },
    {
        "domain": "Cloud Concepts",
        "difficulty": "Hard",
        "question": "Which statement accurately describes the relationship between AWS Regions and Availability Zones?",
        "options": [
            "Each Availability Zone contains multiple Regions",
            "Each Region contains one Availability Zone",
            "Each Region contains multiple Availability Zones that are physically separated but connected via low-latency links",
            "Regions and Availability Zones are the same thing"
        ],
        "correct": 2,
        "explanation": "Each AWS Region contains multiple (minimum 2, typically 3+) Availability Zones. AZs are distinct physical data centers within a region, isolated from failures in other AZs, but connected with low-latency networking."
    },
    # Security & Compliance
    {
        "domain": "Security & Compliance",
        "difficulty": "Easy",
        "question": "Under the AWS Shared Responsibility Model, which of the following is the CUSTOMER's responsibility?",
        "options": [
            "Physical security of data centers",
            "Hardware maintenance",
            "Managing IAM user permissions and access keys",
            "Virtualization infrastructure"
        ],
        "correct": 2,
        "explanation": "Customers are responsible for security IN the cloud, which includes managing IAM users, setting permissions, and rotating access keys. AWS is responsible for security OF the cloud (physical infrastructure)."
    },
    {
        "domain": "Security & Compliance",
        "difficulty": "Easy",
        "question": "What is the primary purpose of AWS IAM?",
        "options": [
            "Monitor application performance",
            "Store application data",
            "Manage user access and permissions to AWS services",
            "Distribute content globally"
        ],
        "correct": 2,
        "explanation": "AWS Identity and Access Management (IAM) enables you to manage access to AWS services and resources securely. You create users, groups, and roles, and use policies to define their permissions."
    },
    {
        "domain": "Security & Compliance",
        "difficulty": "Medium",
        "question": "A company wants to protect its web application from SQL injection and cross-site scripting (XSS) attacks. Which AWS service should they use?",
        "options": [
            "AWS Shield",
            "Amazon GuardDuty",
            "AWS WAF",
            "AWS KMS"
        ],
        "correct": 2,
        "explanation": "AWS WAF (Web Application Firewall) protects web applications from common web exploits like SQL injection and XSS. AWS Shield protects against DDoS attacks, GuardDuty detects threats, and KMS manages encryption keys."
    },
    {
        "domain": "Security & Compliance",
        "difficulty": "Medium",
        "question": "Which AWS service provides detailed records of all API calls made in your AWS account for auditing purposes?",
        "options": [
            "Amazon CloudWatch",
            "AWS CloudTrail",
            "AWS Config",
            "AWS Trusted Advisor"
        ],
        "correct": 1,
        "explanation": "AWS CloudTrail records API calls made in your AWS account, including the identity of the caller, time of the call, source IP address, and more. It's essential for security auditing and compliance."
    },
    {
        "domain": "Security & Compliance",
        "difficulty": "Hard",
        "question": "A company needs to ensure that no AWS accounts in their organization can disable CloudTrail logging. Which service and feature should they use?",
        "options": [
            "AWS IAM with deny policies on all users",
            "AWS Organizations with Service Control Policies (SCPs)",
            "AWS Config with remediation actions",
            "AWS Shield Advanced"
        ],
        "correct": 1,
        "explanation": "Service Control Policies (SCPs) in AWS Organizations set maximum permissions boundaries for all accounts in an Organizational Unit (OU). An SCP denying cloudtrail:StopLogging would prevent any account from disabling CloudTrail, even for administrators."
    },
    {
        "domain": "Security & Compliance",
        "difficulty": "Medium",
        "question": "Which AWS service uses machine learning to detect and alert on sensitive data (like PII) stored in S3?",
        "options": [
            "Amazon Inspector",
            "AWS Security Hub",
            "Amazon Macie",
            "Amazon GuardDuty"
        ],
        "correct": 2,
        "explanation": "Amazon Macie uses machine learning to automatically discover, classify, and protect sensitive data in Amazon S3, including personally identifiable information (PII). GuardDuty detects threats, Inspector finds vulnerabilities, Security Hub aggregates findings."
    },
    # Technology (Core Services)
    {
        "domain": "Technology",
        "difficulty": "Easy",
        "question": "Which AWS service provides scalable object storage?",
        "options": [
            "Amazon EBS",
            "Amazon EFS",
            "Amazon S3",
            "AWS Storage Gateway"
        ],
        "correct": 2,
        "explanation": "Amazon S3 (Simple Storage Service) provides scalable object storage. It stores data as objects within buckets and offers 99.999999999% (11 nines) durability."
    },
    {
        "domain": "Technology",
        "difficulty": "Easy",
        "question": "A developer wants to run code without managing servers. Which AWS service should they use?",
        "options": [
            "Amazon EC2",
            "Amazon ECS",
            "AWS Lambda",
            "Amazon Lightsail"
        ],
        "correct": 2,
        "explanation": "AWS Lambda is a serverless compute service that runs your code in response to events without requiring you to provision or manage servers. You pay only for compute time consumed."
    },
    {
        "domain": "Technology",
        "difficulty": "Medium",
        "question": "A company needs a managed NoSQL database that can scale to handle millions of requests per second with single-digit millisecond latency. Which service should they choose?",
        "options": [
            "Amazon RDS",
            "Amazon Aurora",
            "Amazon DynamoDB",
            "Amazon Redshift"
        ],
        "correct": 2,
        "explanation": "Amazon DynamoDB is a fully managed NoSQL database service that delivers single-digit millisecond performance at any scale. RDS and Aurora are relational databases. Redshift is a data warehouse."
    },
    {
        "domain": "Technology",
        "difficulty": "Medium",
        "question": "Which AWS service automatically distributes incoming application traffic across multiple EC2 instances?",
        "options": [
            "Amazon Route 53",
            "Elastic Load Balancing (ELB)",
            "Amazon CloudFront",
            "AWS Auto Scaling"
        ],
        "correct": 1,
        "explanation": "Elastic Load Balancing (ELB) automatically distributes incoming application traffic across multiple targets (EC2 instances, containers, IPs). Route 53 is DNS, CloudFront is CDN, Auto Scaling adjusts capacity."
    },
    {
        "domain": "Technology",
        "difficulty": "Medium",
        "question": "A company wants to send notifications to multiple subscribers when an event occurs. Which AWS service is BEST suited for this?",
        "options": [
            "Amazon SQS",
            "Amazon SNS",
            "Amazon MQ",
            "AWS Step Functions"
        ],
        "correct": 1,
        "explanation": "Amazon SNS (Simple Notification Service) is a pub/sub messaging service that can fan-out messages to multiple subscribers simultaneously (SQS queues, Lambda functions, email, SMS, HTTP). SQS is for queuing, not broadcasting."
    },
    {
        "domain": "Technology",
        "difficulty": "Hard",
        "question": "A company's application runs on EC2 and needs a database that is MySQL-compatible, automatically scales storage, and is 5x faster than standard MySQL. Which service meets all these requirements?",
        "options": [
            "Amazon RDS for MySQL",
            "Amazon Aurora",
            "Amazon DynamoDB",
            "Amazon ElastiCache for Redis"
        ],
        "correct": 1,
        "explanation": "Amazon Aurora is a MySQL and PostgreSQL-compatible relational database that is 5x faster than standard MySQL and 3x faster than standard PostgreSQL. It automatically scales storage up to 128TB and is fully managed."
    },
    {
        "domain": "Technology",
        "difficulty": "Medium",
        "question": "Which AWS service provides a virtual private network environment within the AWS cloud?",
        "options": [
            "AWS Direct Connect",
            "Amazon VPC",
            "AWS VPN",
            "Amazon Route 53"
        ],
        "correct": 1,
        "explanation": "Amazon VPC (Virtual Private Cloud) lets you provision a logically isolated section of AWS Cloud where you can launch resources in a virtual network you define. Direct Connect provides dedicated network connection to AWS."
    },
    {
        "domain": "Technology",
        "difficulty": "Hard",
        "question": "A company needs to run containerized microservices without managing the underlying EC2 infrastructure. Which combination of services achieves this?",
        "options": [
            "Amazon ECS with EC2 launch type",
            "Amazon ECS or EKS with AWS Fargate",
            "Amazon EC2 with Docker installed manually",
            "AWS Lambda with container packaging"
        ],
        "correct": 1,
        "explanation": "AWS Fargate is a serverless compute engine for containers. When used with ECS or EKS, you run containers without managing EC2 instances — you just define your containers and Fargate handles the underlying infrastructure."
    },
    # Billing & Pricing
    {
        "domain": "Billing & Pricing",
        "difficulty": "Easy",
        "question": "Which EC2 pricing option provides the MOST savings for a steady-state workload that runs continuously for 1-3 years?",
        "options": [
            "On-Demand Instances",
            "Spot Instances",
            "Reserved Instances",
            "Dedicated Hosts"
        ],
        "correct": 2,
        "explanation": "Reserved Instances provide up to 72% savings compared to On-Demand for workloads that run continuously. They require a 1 or 3-year commitment. Spot Instances can offer up to 90% savings but can be interrupted."
    },
    {
        "domain": "Billing & Pricing",
        "difficulty": "Easy",
        "question": "Which AWS tool allows you to estimate the cost of AWS services before using them?",
        "options": [
            "AWS Cost Explorer",
            "AWS Budgets",
            "AWS Pricing Calculator",
            "AWS Trusted Advisor"
        ],
        "correct": 2,
        "explanation": "The AWS Pricing Calculator lets you estimate the cost of AWS services for your use case before you start using them. Cost Explorer analyzes actual/historical costs, Budgets sets alerts, Trusted Advisor gives recommendations."
    },
    {
        "domain": "Billing & Pricing",
        "difficulty": "Medium",
        "question": "A company has multiple AWS accounts and wants a single bill. Which AWS feature enables this?",
        "options": [
            "AWS Cost Explorer",
            "Consolidated Billing via AWS Organizations",
            "AWS Budgets",
            "AWS Cost and Usage Report"
        ],
        "correct": 1,
        "explanation": "Consolidated Billing is a feature of AWS Organizations that provides a single bill for all accounts in the organization. It also enables sharing of Reserved Instance discounts and Savings Plans across accounts."
    },
    {
        "domain": "Billing & Pricing",
        "difficulty": "Medium",
        "question": "Which AWS pricing model allows customers to use excess EC2 capacity at up to 90% discount but with the possibility of interruption?",
        "options": [
            "On-Demand Instances",
            "Reserved Instances",
            "Spot Instances",
            "Savings Plans"
        ],
        "correct": 2,
        "explanation": "Spot Instances allow you to use spare EC2 capacity at up to 90% discount. However, AWS can reclaim (interrupt) them with a 2-minute warning when capacity is needed elsewhere. Best for fault-tolerant, flexible workloads."
    },
    {
        "domain": "Billing & Pricing",
        "difficulty": "Hard",
        "question": "Which of the following data transfer scenarios does NOT incur an AWS data transfer charge?",
        "options": [
            "Data transferred OUT from EC2 to the internet",
            "Data transferred between EC2 instances in different Regions",
            "Data transferred IN to AWS from the internet",
            "Data transferred between EC2 in different AZs within the same Region"
        ],
        "correct": 2,
        "explanation": "Inbound data transfer (from the internet INTO AWS) is generally free. Outbound to internet is charged. Cross-region transfer is charged. Cross-AZ transfer within the same region incurs a small charge."
    },
    # Support & Governance
    {
        "domain": "Support & Governance",
        "difficulty": "Easy",
        "question": "Which AWS Support plan includes access to a dedicated Technical Account Manager (TAM)?",
        "options": [
            "Basic",
            "Developer",
            "Business",
            "Enterprise"
        ],
        "correct": 3,
        "explanation": "A dedicated Technical Account Manager (TAM) is only available with the Enterprise Support plan ($15,000/month). Enterprise On-Ramp provides access to a pool of TAMs but not a dedicated one."
    },
    {
        "domain": "Support & Governance",
        "difficulty": "Medium",
        "question": "Which AWS service provides Infrastructure as Code (IaC) to model and provision AWS resources?",
        "options": [
            "AWS Systems Manager",
            "AWS OpsWorks",
            "AWS CloudFormation",
            "AWS Service Catalog"
        ],
        "correct": 2,
        "explanation": "AWS CloudFormation provides Infrastructure as Code (IaC), allowing you to model and provision AWS and third-party resources using JSON or YAML templates. This enables repeatable, version-controlled infrastructure deployments."
    },
    {
        "domain": "Support & Governance",
        "difficulty": "Medium",
        "question": "Which AWS tool checks your environment against AWS best practices in categories like security, cost optimization, and performance?",
        "options": [
            "Amazon Inspector",
            "AWS Trusted Advisor",
            "AWS Config",
            "AWS Well-Architected Tool"
        ],
        "correct": 1,
        "explanation": "AWS Trusted Advisor inspects your AWS environment and provides real-time recommendations in five categories: Cost Optimization, Performance, Security, Fault Tolerance, and Service Limits. Full checks require Business/Enterprise support."
    },
    {
        "domain": "Support & Governance",
        "difficulty": "Hard",
        "question": "A company wants to ensure all new AWS accounts in their organization automatically have CloudTrail enabled and S3 public access blocked. Which service best achieves this?",
        "options": [
            "AWS Config Rules applied to each account",
            "AWS Control Tower with guardrails",
            "AWS Trusted Advisor across the organization",
            "Manual IAM policies in each account"
        ],
        "correct": 1,
        "explanation": "AWS Control Tower provides pre-packaged governance for multi-account environments. It uses 'guardrails' (preventive and detective controls) and an account factory to automatically configure new accounts with required security settings like CloudTrail and S3 settings."
    },
    # Migration & Innovation
    {
        "domain": "Migration & Innovation",
        "difficulty": "Easy",
        "question": "A company has 50TB of data to migrate to AWS but has limited internet bandwidth. Which AWS service should they use?",
        "options": [
            "AWS DataSync",
            "AWS Transfer Family",
            "AWS Snowball Edge",
            "AWS Direct Connect"
        ],
        "correct": 2,
        "explanation": "AWS Snowball Edge is a physical device that allows you to transfer large amounts of data (up to 80TB per device) to AWS without using the internet. When internet-based transfer would take too long, physical devices are the answer."
    },
    {
        "domain": "Migration & Innovation",
        "difficulty": "Medium",
        "question": "Which migration strategy involves moving an application to AWS without making any changes to the application?",
        "options": [
            "Replatform",
            "Refactor",
            "Rehost (Lift and Shift)",
            "Repurchase"
        ],
        "correct": 2,
        "explanation": "Rehost (Lift and Shift) means moving an application to AWS without making changes. You simply re-host the application on AWS infrastructure (e.g., moving VMs to EC2). It's the fastest way to migrate but may not fully leverage cloud benefits."
    },
    {
        "domain": "Migration & Innovation",
        "difficulty": "Medium",
        "question": "Which AWS service enables you to build, train, and deploy machine learning models?",
        "options": [
            "Amazon Rekognition",
            "Amazon Comprehend",
            "Amazon SageMaker",
            "AWS DeepLens"
        ],
        "correct": 2,
        "explanation": "Amazon SageMaker is a fully managed ML service that provides every developer and data scientist with the ability to build, train, and deploy ML models quickly. Rekognition is for image analysis, Comprehend for NLP."
    },
]

# ─────────────────────────────────────────────────────────────
# DATA: STUDY GUIDE CONTENT
# ─────────────────────────────────────────────────────────────

STUDY_GUIDE = {
    "Exam Overview": """
## AWS Cloud Practitioner (CLF-C02) Exam Guide

### Exam Details
| Item | Detail |
|------|--------|
| Exam Code | CLF-C02 |
| Duration | 90 minutes |
| Questions | 65 questions |
| Passing Score | 700/1000 |
| Cost | $100 USD |
| Format | Multiple choice, Multiple response |
| Delivery | Pearson VUE or PSI (testing center or online) |

### Exam Domains & Weightings
| Domain | Weight |
|--------|--------|
| Cloud Concepts | 24% |
| Security & Compliance | 30% |
| Cloud Technology & Services | 34% |
| Billing, Pricing & Support | 12% |

### Key Tips
- **Domain 3 (Technology)** is the largest section — know your AWS services well
- **Domain 2 (Security)** is second — understand the Shared Responsibility Model deeply
- Read all answer choices before selecting — AWS questions often have "more correct" answers
- Know the difference between SIMILAR services (e.g., SNS vs SQS, CloudTrail vs CloudWatch)
- Understand pricing models and when to use each type
""",
    "Cloud Concepts": """
## Domain 1: Cloud Concepts (24%)

### The 6 Advantages of Cloud Computing
1. **Trade fixed expense for variable expense** — Pay only for what you consume
2. **Benefit from massive economies of scale** — Lower costs due to AWS's purchasing power
3. **Stop guessing capacity** — Scale up/down based on actual need
4. **Increase speed and agility** — Deploy in minutes vs weeks
5. **Stop spending on data center operations** — Focus on business, not infrastructure
6. **Go global in minutes** — Deploy to multiple regions easily

### Cloud Deployment Models
| Model | Description | Example |
|-------|-------------|---------|
| Public Cloud | Fully on AWS | All resources in AWS |
| Hybrid | Mix of AWS + on-premises | AWS + corporate data center |
| On-Premises (Private) | Your own data center | VMware on-prem |

### AWS Global Infrastructure
- **Regions**: 33+ geographic regions (e.g., us-east-1)
- **Availability Zones**: 105+ AZs; each Region has 3+ AZs
- **Edge Locations**: 400+ for CloudFront CDN
- **Local Zones**: Extend AWS to specific metro areas

### Well-Architected Framework (6 Pillars)
1. **Operational Excellence** — Run and monitor systems, continuous improvement
2. **Security** — Protect information and systems
3. **Reliability** — Recover from failures, scale dynamically
4. **Performance Efficiency** — Use computing resources efficiently
5. **Cost Optimization** — Avoid unnecessary costs
6. **Sustainability** — Minimize environmental impact

### Service Models
- **IaaS** (Infrastructure): You manage OS+. Example: EC2
- **PaaS** (Platform): You manage app+data. Example: Beanstalk, RDS
- **SaaS** (Software): Provider manages everything. Example: Rekognition, Gmail
""",
    "Security & Compliance": """
## Domain 2: Security & Compliance (30%)

### Shared Responsibility Model
| AWS Responsibility (OF the cloud) | Customer Responsibility (IN the cloud) |
|-----------------------------------|----------------------------------------|
| Physical data centers | IAM users, groups, roles |
| Hardware, networking | OS patches on EC2 |
| Managed service security | Data encryption |
| Virtualization | Network/firewall configuration |
| Global infrastructure | Application security |

### IAM Key Concepts
- **Users**: Individual accounts for people/applications
- **Groups**: Collection of users sharing permissions
- **Roles**: Temporary credentials for services/apps (preferred over keys)
- **Policies**: JSON documents defining permissions
- **Least Privilege**: Grant only what's needed

### IAM Best Practices
✅ Lock away root account keys
✅ Create individual IAM users
✅ Use groups to assign permissions
✅ Grant least privilege
✅ Enable MFA for privileged users
✅ Use roles for applications
✅ Rotate credentials regularly
✅ Use IAM Access Analyzer

### Security Services Quick Reference
| Service | Purpose |
|---------|---------|
| AWS Shield | DDoS protection (Standard = free) |
| AWS WAF | Web app firewall (SQLi, XSS) |
| Amazon GuardDuty | Threat detection via ML |
| Amazon Inspector | Vulnerability scanning |
| Amazon Macie | Sensitive data discovery in S3 |
| AWS KMS | Encryption key management |
| AWS CloudTrail | API call auditing/logging |
| AWS Config | Resource configuration compliance |
| AWS Security Hub | Centralized security findings |
| AWS Artifact | Compliance reports and docs |

### Compliance Programs
AWS is compliant with many standards:
- **HIPAA** (healthcare), **PCI DSS** (payments), **SOC 1/2/3**, **ISO 27001**, **FedRAMP**
- AWS Artifact provides access to compliance documentation
""",
    "Technology & Services": """
## Domain 3: Cloud Technology & Services (34%)

### Compute Services
| Service | Type | Key Facts |
|---------|------|-----------|
| EC2 | IaaS VMs | Full control, many instance types |
| Lambda | Serverless | Event-driven, up to 15 min, pay per ms |
| ECS | Containers | Docker orchestration |
| EKS | Containers | Managed Kubernetes |
| Fargate | Serverless Containers | No EC2 management needed |
| Elastic Beanstalk | PaaS | Deploy web apps, manages infrastructure |
| Lightsail | Simple VPS | Simplified, predictable pricing |
| Batch | Batch Jobs | Run batch computing workloads |

### Storage Services
| Service | Type | Use Case |
|---------|------|---------|
| S3 | Object | Files, backups, static sites, data lakes |
| EBS | Block | EC2 hard drive, databases |
| EFS | File | Shared file system, Linux |
| FSx | File | Windows (FSx for Windows), HPC (FSx for Lustre) |
| S3 Glacier | Archive | Long-term archival, cheap |
| Storage Gateway | Hybrid | On-prem to cloud bridge |

### Database Services
| Service | Type | Use Case |
|---------|------|---------|
| RDS | Relational | MySQL, PostgreSQL, Oracle, SQL Server |
| Aurora | Relational | High-performance MySQL/PostgreSQL |
| DynamoDB | NoSQL | Key-value, millisecond latency |
| ElastiCache | In-memory | Redis/Memcached caching layer |
| Redshift | Data Warehouse | Analytics, BI queries |
| DocumentDB | Document | MongoDB-compatible |
| Neptune | Graph | Graph databases |
| QLDB | Ledger | Immutable transaction history |

### Networking Services
| Service | Purpose |
|---------|---------|
| VPC | Virtual network isolation |
| CloudFront | CDN, edge caching |
| Route 53 | DNS, domain registration |
| ELB | Load balancing (ALB/NLB/CLB/GWLB) |
| API Gateway | REST/WebSocket APIs |
| Direct Connect | Dedicated network to AWS |
| VPN | Encrypted tunnel to AWS |
| Transit Gateway | Hub-and-spoke network |

### Management & Monitoring
| Service | Purpose |
|---------|---------|
| CloudWatch | Metrics, logs, alarms, dashboards |
| CloudTrail | API call logging/auditing |
| Config | Configuration compliance |
| CloudFormation | Infrastructure as Code |
| Systems Manager | Operations management |
| Trusted Advisor | Best practice checks |
| Control Tower | Multi-account governance |
| Organizations | Account management, SCPs |
""",
    "Billing & Pricing": """
## Domain 4: Billing, Pricing & Support (12%)

### EC2 Pricing Models
| Model | Discount | Commitment | Best For |
|-------|----------|------------|---------|
| On-Demand | None | None | Short-term, unpredictable |
| Reserved (Standard) | Up to 72% | 1-3 years | Steady-state workloads |
| Reserved (Convertible) | Up to 66% | 1-3 years | Need flexibility to change type |
| Savings Plans | Up to 72% | 1-3 years | Flexible commitment across services |
| Spot | Up to 90% | None (can be interrupted) | Fault-tolerant, flexible workloads |
| Dedicated Host | Varies | On-demand or reserved | Compliance, licensing |

### Billing & Cost Management Tools
| Tool | Purpose |
|------|---------|
| Cost Explorer | Visualize/analyze historical costs |
| AWS Budgets | Set cost/usage alerts and actions |
| Pricing Calculator | Estimate costs before use |
| Cost & Usage Report | Detailed billing data |
| Consolidated Billing | Single bill for org accounts |
| Savings Plans | Flexible cost reduction |

### AWS Free Tier (3 Types)
1. **Always Free**: Lambda (1M requests), DynamoDB (25GB), SNS (1M publishes)
2. **12-Month Free**: EC2 t2.micro (750 hrs), S3 (5GB), RDS (750 hrs)
3. **Short-term Trials**: 30-60 day trials for specific services

### Support Plans Comparison
| Feature | Basic | Developer | Business | Enterprise |
|---------|-------|-----------|----------|------------|
| Price | Free | $29/mo | $100/mo | $15K/mo |
| Response (critical) | N/A | N/A | 1 hr | 15 min |
| TAM | No | No | No | Yes (dedicated) |
| Trusted Advisor | 7 checks | 7 checks | All checks | All checks |
| Support channels | Docs only | Email | Phone/Chat | Phone/Chat |

### Cost Optimization Strategies
- **Right-sizing**: Match instance type to actual workload
- **Reserved Instances / Savings Plans**: Commit for steady workloads
- **Auto Scaling**: Scale down during low demand
- **Spot Instances**: For flexible, fault-tolerant workloads
- **S3 Lifecycle Policies**: Move old data to cheaper tiers
- **Delete unused resources**: Unattached EBS, idle EIPs
""",
}

# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "fc_deck": list(FLASHCARD_DECKS.keys())[0],
        "fc_index": 0,
        "fc_flipped": False,
        "fc_known": set(),
        "fc_unknown": set(),
        "fc_shuffled": None,
        "exam_questions": None,
        "exam_answers": {},
        "exam_submitted": False,
        "exam_start_time": None,
        "exam_score": None,
        "exam_num_questions": 30,
        "exam_domain_filter": "All Domains",
        "active_tab": "Home",
        "progress": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_domain_color(domain):
    colors = {
        "Cloud Concepts": "#3498db",
        "Security & Compliance": "#e74c3c",
        "Technology": "#2ecc71",
        "Billing & Pricing": "#f39c12",
        "Support & Governance": "#9b59b6",
        "Migration & Innovation": "#1abc9c",
    }
    return colors.get(domain, "#95a5a6")


def get_difficulty_badge(difficulty):
    badges = {
        "Easy": "🟢 Easy",
        "Medium": "🟡 Medium",
        "Hard": "🔴 Hard",
    }
    return badges.get(difficulty, difficulty)


# ─────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────

def render_home():
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <h1 style='color:#FF9900; font-size:2.8em; margin-bottom:5px;'>☁️ AWS Cloud Practitioner</h1>
        <h2 style='color:#232F3E; font-size:1.4em; font-weight:400;'>CLF-C02 Complete Study Platform</h2>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    total_cards = sum(len(v) for v in FLASHCARD_DECKS.values())
    known_count = len(st.session_state.fc_known)
    questions_attempted = len([q for q in PRACTICE_QUESTIONS if q.get("attempted")])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Flashcards", total_cards, help="Total flashcards across all decks")
    with col2:
        st.metric("✅ Cards Mastered", known_count, help="Flashcards you've marked as known")
    with col3:
        st.metric("❓ Practice Questions", len(PRACTICE_QUESTIONS), help="Total practice exam questions")
    with col4:
        pct = int((known_count / total_cards) * 100) if total_cards else 0
        st.metric("📈 Study Progress", f"{pct}%", help="Percentage of flashcards mastered")

    st.markdown("---")

    # Exam domain breakdown
    st.subheader("📊 Exam Domain Breakdown")
    domains = {
        "☁️ Cloud Concepts": ("24%", "#3498db", "Understand cloud computing fundamentals"),
        "🔐 Security & Compliance": ("30%", "#e74c3c", "Shared responsibility, IAM, security services"),
        "⚙️ Technology & Services": ("34%", "#2ecc71", "Core AWS services and infrastructure"),
        "💰 Billing & Pricing": ("12%", "#f39c12", "Pricing models, billing tools, support plans"),
    }
    cols = st.columns(4)
    for col, (domain, (pct, color, desc)) in zip(cols, domains.items()):
        with col:
            st.markdown(f"""
            <div style='background:{color}15; border-left: 4px solid {color}; padding: 15px; border-radius: 8px; height:120px;'>
                <div style='font-size:1.3em; font-weight:bold; color:{color};'>{pct}</div>
                <div style='font-weight:600; margin:4px 0;'>{domain}</div>
                <div style='font-size:0.8em; color:#666;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🚀 Quick Start")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**📖 Flashcards**\nReview key AWS concepts with interactive flashcards organized by topic.")
    with c2:
        st.warning("**📝 Practice Exam**\nTest your knowledge with timed multiple-choice questions from all domains.")
    with c3:
        st.success("**📚 Study Guide**\nComprehensive notes covering all exam domains and key services.")

    st.markdown("---")
    st.subheader("🎯 Exam Day Tips")
    tips = [
        "**Read carefully**: AWS questions often have two seemingly correct answers — pick the BEST one.",
        "**Know the Shared Responsibility Model** cold — it appears in many questions.",
        "**Understand pricing models**: When to use On-Demand vs Reserved vs Spot vs Savings Plans.",
        "**Learn service pairs**: CloudTrail (audit) vs CloudWatch (monitoring), SNS (notify) vs SQS (queue).",
        "**Managed = AWS manages**: RDS, DynamoDB, Lambda — AWS handles the infrastructure.",
        "**Focus on use cases**: Know WHEN to use each service, not just what it does.",
        "**Skip and return**: Flag uncertain questions and revisit at the end.",
    ]
    for tip in tips:
        st.markdown(f"• {tip}")


# ─────────────────────────────────────────────────────────────
# PAGE: FLASHCARDS
# ─────────────────────────────────────────────────────────────

def render_flashcards():
    st.title("📖 Flashcards")

    # Deck selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_deck = st.selectbox("Choose a Deck", list(FLASHCARD_DECKS.keys()),
                                      index=list(FLASHCARD_DECKS.keys()).index(st.session_state.fc_deck))
    with col2:
        shuffle = st.checkbox("🔀 Shuffle", value=True)
    with col3:
        filter_unknown = st.checkbox("⭐ Review Only", help="Show only cards you haven't mastered")

    # Deck changed — reset
    if selected_deck != st.session_state.fc_deck:
        st.session_state.fc_deck = selected_deck
        st.session_state.fc_index = 0
        st.session_state.fc_flipped = False
        st.session_state.fc_shuffled = None
        st.rerun()

    deck = FLASHCARD_DECKS[selected_deck].copy()

    # Filter to unknown cards
    if filter_unknown:
        deck = [c for i, c in enumerate(deck) if i not in st.session_state.fc_known]
        if not deck:
            st.success("🎉 You've mastered all cards in this deck! Uncheck 'Review Only' to review them again.")
            return

    # Build shuffled order once per deck session
    if st.session_state.fc_shuffled is None or len(st.session_state.fc_shuffled) != len(deck):
        order = list(range(len(deck)))
        if shuffle:
            random.shuffle(order)
        st.session_state.fc_shuffled = order
        st.session_state.fc_index = 0

    if not st.session_state.fc_shuffled:
        st.info("No cards available.")
        return

    idx = st.session_state.fc_index % len(st.session_state.fc_shuffled)
    card_idx = st.session_state.fc_shuffled[idx]
    card = deck[card_idx]
    total = len(st.session_state.fc_shuffled)
    known_in_deck = len([i for i in range(len(FLASHCARD_DECKS[selected_deck]))
                         if i in st.session_state.fc_known])

    # Progress bar
    st.markdown(f"**Card {idx + 1} of {total}** &nbsp;|&nbsp; ✅ {known_in_deck} mastered in this deck")
    progress_val = (idx + 1) / total if total else 0
    st.progress(progress_val)

    # Category badge
    cat_color = {"Storage": "#3498db", "Compute": "#2ecc71", "Database": "#9b59b6",
                 "Networking": "#e67e22", "Security": "#e74c3c", "Management": "#1abc9c",
                 "Concepts": "#f39c12", "Pricing": "#16a085", "Compliance": "#8e44ad",
                 "Billing": "#2980b9", "Governance": "#c0392b", "Support": "#7f8c8d",
                 "Migration": "#27ae60", "Innovation": "#d35400", "Application Integration": "#2c3e50"}.get(card.get("category", ""), "#95a5a6")

    st.markdown(f"<span style='background:{cat_color}; color:white; padding:3px 10px; border-radius:12px; font-size:0.8em;'>{card.get('category','')}</span>", unsafe_allow_html=True)

    # Card display
    st.markdown("<br>", unsafe_allow_html=True)
    if not st.session_state.fc_flipped:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #232F3E, #37475A); color: white; padding: 50px 40px;
                    border-radius: 16px; text-align: center; min-height: 200px; cursor: pointer;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3); display: flex; align-items: center; justify-content: center;'>
            <div>
                <div style='color:#FF9900; font-size:0.9em; margin-bottom:10px; text-transform:uppercase; letter-spacing:2px;'>QUESTION</div>
                <div style='font-size:1.6em; font-weight:600; line-height:1.4;'>{card["front"]}</div>
                <div style='color:#aaa; margin-top:20px; font-size:0.85em;'>👆 Click "Reveal Answer" to flip</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        back_html = card["back"].replace("\n", "<br>")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a6b3c, #27ae60); color: white; padding: 40px;
                    border-radius: 16px; min-height: 200px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
            <div style='color:#a8f5c8; font-size:0.9em; margin-bottom:10px; text-transform:uppercase; letter-spacing:2px;'>ANSWER</div>
            <div style='font-size:1.1em; line-height:1.8;'>{back_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Buttons
    if not st.session_state.fc_flipped:
        col_flip = st.columns([1, 2, 1])[1]
        with col_flip:
            if st.button("🔄 Reveal Answer", use_container_width=True, type="primary"):
                st.session_state.fc_flipped = True
                st.rerun()
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("❌ Still Learning", use_container_width=True):
                st.session_state.fc_unknown.add(card_idx)
                st.session_state.fc_known.discard(card_idx)
                st.session_state.fc_index = (idx + 1) % total
                st.session_state.fc_flipped = False
                st.rerun()
        with col2:
            if st.button("✅ Got It!", use_container_width=True, type="primary"):
                st.session_state.fc_known.add(card_idx)
                st.session_state.fc_unknown.discard(card_idx)
                st.session_state.fc_index = (idx + 1) % total
                st.session_state.fc_flipped = False
                st.rerun()
        with col3:
            if st.button("⏭️ Skip", use_container_width=True):
                st.session_state.fc_index = (idx + 1) % total
                st.session_state.fc_flipped = False
                st.rerun()

    # Navigation row
    st.markdown("---")
    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("⬅️ Previous"):
            st.session_state.fc_index = (idx - 1) % total
            st.session_state.fc_flipped = False
            st.rerun()
    with nav2:
        jump = st.number_input("Jump to card #", min_value=1, max_value=total, value=idx + 1, label_visibility="collapsed")
        if jump != idx + 1:
            st.session_state.fc_index = int(jump) - 1
            st.session_state.fc_flipped = False
            st.rerun()
    with nav3:
        if st.button("Next ➡️"):
            st.session_state.fc_index = (idx + 1) % total
            st.session_state.fc_flipped = False
            st.rerun()

    # Deck overview
    with st.expander("📋 All Cards in This Deck"):
        for i, c in enumerate(deck):
            status = "✅" if i in st.session_state.fc_known else ("❌" if i in st.session_state.fc_unknown else "⬜")
            st.markdown(f"{status} **{c['front']}**")


# ─────────────────────────────────────────────────────────────
# PAGE: PRACTICE EXAM
# ─────────────────────────────────────────────────────────────

def render_practice_exam():
    st.title("📝 Practice Exam")

    if not st.session_state.exam_submitted and st.session_state.exam_questions is None:
        _render_exam_setup()
    elif not st.session_state.exam_submitted and st.session_state.exam_questions is not None:
        _render_exam_in_progress()
    else:
        _render_exam_results()


def _render_exam_setup():
    st.subheader("⚙️ Configure Your Practice Exam")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        num_q = st.slider("Number of Questions", min_value=10, max_value=len(PRACTICE_QUESTIONS),
                          value=min(30, len(PRACTICE_QUESTIONS)), step=5)
        st.session_state.exam_num_questions = num_q
    with col2:
        domains = ["All Domains"] + sorted(set(q["domain"] for q in PRACTICE_QUESTIONS))
        domain_filter = st.selectbox("Filter by Domain", domains)
        st.session_state.exam_domain_filter = domain_filter

    difficulty_filter = st.multiselect("Difficulty", ["Easy", "Medium", "Hard"], default=["Easy", "Medium", "Hard"])

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.info(f"**{len(PRACTICE_QUESTIONS)} total questions** available covering all CLF-C02 domains.")
    with col_b:
        if st.button("🚀 Start Exam", type="primary", use_container_width=True):
            pool = PRACTICE_QUESTIONS.copy()
            if domain_filter != "All Domains":
                pool = [q for q in pool if q["domain"] == domain_filter]
            if difficulty_filter:
                pool = [q for q in pool if q["difficulty"] in difficulty_filter]
            if not pool:
                st.error("No questions match your filters. Please adjust and try again.")
                return
            selected = random.sample(pool, min(num_q, len(pool)))
            st.session_state.exam_questions = selected
            st.session_state.exam_answers = {}
            st.session_state.exam_submitted = False
            st.session_state.exam_start_time = time.time()
            st.rerun()

    # Previous exam history preview
    if st.session_state.exam_score is not None:
        st.markdown("---")
        st.subheader("📊 Last Exam Result")
        score = st.session_state.exam_score
        pct = score["percentage"]
        color = "#2ecc71" if pct >= 70 else "#e74c3c"
        st.markdown(f"""
        <div style='background:{color}20; border-left: 4px solid {color}; padding:15px; border-radius:8px;'>
            <b>Score: {score['correct']}/{score['total']} ({pct:.1f}%)</b>
            {"✅ PASS" if pct >= 70 else "❌ Did not pass"} &nbsp;|&nbsp; Time: {score['time_taken']}
        </div>
        """, unsafe_allow_html=True)


def _render_exam_in_progress():
    questions = st.session_state.exam_questions
    elapsed = int(time.time() - st.session_state.exam_start_time)
    remaining = max(0, 90 * 60 - elapsed)
    mins, secs = divmod(remaining, 60)

    # Header
    answered = len(st.session_state.exam_answers)
    total = len(questions)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Progress", f"{answered}/{total} answered")
    with col2:
        color_class = "normal" if remaining > 600 else "inverse"
        st.metric("⏱️ Time Remaining", f"{mins:02d}:{secs:02d}", delta_color=color_class)
    with col3:
        st.metric("🔢 Questions", total)

    st.progress(answered / total if total else 0)
    st.markdown("---")

    # Questions
    for i, q in enumerate(questions):
        domain_color = get_domain_color(q["domain"])
        st.markdown(f"""
        <div style='margin-bottom:8px;'>
            <span style='background:{domain_color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.75em;'>
                {q['domain']}
            </span>
            &nbsp;
            <span style='color:#666; font-size:0.8em;'>{get_difficulty_badge(q["difficulty"])}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Q{i+1}.** {q['question']}")

        current_answer = st.session_state.exam_answers.get(i)
        selected = st.radio(
            f"q_{i}",
            options=q["options"],
            index=current_answer if current_answer is not None else None,
            key=f"exam_q_{i}",
            label_visibility="collapsed",
        )
        if selected is not None:
            new_answer = q["options"].index(selected)
            if st.session_state.exam_answers.get(i) != new_answer:
                st.session_state.exam_answers[i] = new_answer
        st.markdown("<hr style='border:none; border-top:1px solid #eee; margin:20px 0;'>", unsafe_allow_html=True)

    # Submit
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📊 Submit Exam", type="primary", use_container_width=True):
            _score_exam()
            st.rerun()
    with col1:
        if answered < total:
            st.warning(f"⚠️ {total - answered} questions unanswered. You can still submit.")
        else:
            st.success("✅ All questions answered! Ready to submit.")


def _score_exam():
    questions = st.session_state.exam_questions
    answers = st.session_state.exam_answers
    elapsed = int(time.time() - st.session_state.exam_start_time)
    mins, secs = divmod(elapsed, 60)

    correct = 0
    domain_results = {}
    for i, q in enumerate(questions):
        domain = q["domain"]
        if domain not in domain_results:
            domain_results[domain] = {"correct": 0, "total": 0}
        domain_results[domain]["total"] += 1
        if answers.get(i) == q["correct"]:
            correct += 1
            domain_results[domain]["correct"] += 1

    total = len(questions)
    pct = (correct / total * 100) if total else 0
    st.session_state.exam_score = {
        "correct": correct,
        "total": total,
        "percentage": pct,
        "domain_results": domain_results,
        "time_taken": f"{mins}m {secs}s",
    }
    st.session_state.exam_submitted = True


def _render_exam_results():
    score = st.session_state.exam_score
    pct = score["percentage"]
    passed = pct >= 70
    questions = st.session_state.exam_questions
    answers = st.session_state.exam_answers

    # Score banner
    color = "#2ecc71" if passed else "#e74c3c"
    emoji = "🎉" if passed else "📚"
    st.markdown(f"""
    <div style='background:linear-gradient(135deg, {color}, {color}cc); color:white; padding:30px;
                border-radius:16px; text-align:center; margin-bottom:20px;'>
        <div style='font-size:3em;'>{emoji}</div>
        <div style='font-size:2.5em; font-weight:bold;'>{pct:.1f}%</div>
        <div style='font-size:1.2em;'>{score["correct"]}/{score["total"]} correct</div>
        <div style='font-size:1.5em; margin-top:10px; font-weight:600;'>
            {"✅ PASS — Great work!" if passed else "❌ Keep Studying — You've got this!"}
        </div>
        <div style='color:rgba(255,255,255,0.8); margin-top:5px;'>Time: {score["time_taken"]} &nbsp;|&nbsp; Passing: 70% (700/1000)</div>
    </div>
    """, unsafe_allow_html=True)

    # Domain breakdown
    st.subheader("📊 Performance by Domain")
    for domain, res in score["domain_results"].items():
        domain_pct = (res["correct"] / res["total"] * 100) if res["total"] else 0
        d_color = "#2ecc71" if domain_pct >= 70 else ("#f39c12" if domain_pct >= 50 else "#e74c3c")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{domain}**")
            st.progress(domain_pct / 100)
        with col2:
            st.markdown(f"<div style='color:{d_color}; font-weight:bold; padding-top:8px;'>{res['correct']}/{res['total']} ({domain_pct:.0f}%)</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Answer review
    st.subheader("📖 Detailed Answer Review")
    filter_opt = st.radio("Show:", ["All Questions", "Incorrect Only", "Correct Only"], horizontal=True)

    for i, q in enumerate(questions):
        user_ans = answers.get(i)
        is_correct = user_ans == q["correct"]

        if filter_opt == "Incorrect Only" and is_correct:
            continue
        if filter_opt == "Correct Only" and not is_correct:
            continue

        with st.expander(f"{'✅' if is_correct else '❌'} Q{i+1}: {q['question'][:80]}..."):
            st.markdown(f"**Domain:** {q['domain']} &nbsp; **Difficulty:** {get_difficulty_badge(q['difficulty'])}")
            st.markdown(f"**Full Question:** {q['question']}")
            st.markdown("")
            for j, opt in enumerate(q["options"]):
                if j == q["correct"]:
                    st.markdown(f"✅ **{opt}** ← Correct answer")
                elif j == user_ans and not is_correct:
                    st.markdown(f"❌ ~~{opt}~~ ← Your answer")
                else:
                    st.markdown(f"&nbsp;&nbsp;&nbsp; {opt}")
            if user_ans is None:
                st.markdown("⚪ *Not answered*")
            st.markdown(f"\n💡 **Explanation:** {q['explanation']}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Take New Exam", type="primary", use_container_width=True):
            st.session_state.exam_questions = None
            st.session_state.exam_submitted = False
            st.session_state.exam_answers = {}
            st.rerun()
    with col2:
        if st.button("📖 Review Flashcards", use_container_width=True):
            st.session_state.active_tab = "Flashcards"
            st.rerun()


# ─────────────────────────────────────────────────────────────
# PAGE: STUDY GUIDE
# ─────────────────────────────────────────────────────────────

def render_study_guide():
    st.title("📚 Study Guide")

    tab_names = list(STUDY_GUIDE.keys())
    tabs = st.tabs(tab_names)
    for tab, name in zip(tabs, tab_names):
        with tab:
            st.markdown(STUDY_GUIDE[name])


# ─────────────────────────────────────────────────────────────
# PAGE: PROGRESS TRACKER
# ─────────────────────────────────────────────────────────────

def render_progress():
    st.title("📈 Progress Tracker")

    total_cards = sum(len(v) for v in FLASHCARD_DECKS.values())
    known = len(st.session_state.fc_known)
    pct = int((known / total_cards) * 100) if total_cards else 0

    # Overall progress
    st.subheader("🎯 Overall Flashcard Mastery")
    st.progress(pct / 100)
    st.markdown(f"**{known}/{total_cards} cards mastered ({pct}%)**")

    # Deck breakdown
    st.subheader("📚 Mastery by Deck")
    for deck_name, cards in FLASHCARD_DECKS.items():
        known_in_deck = sum(1 for i in range(len(cards)) if i in st.session_state.fc_known)
        d_pct = int((known_in_deck / len(cards)) * 100) if cards else 0
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{deck_name}**")
            st.progress(d_pct / 100)
        with col2:
            st.markdown(f"**{known_in_deck}/{len(cards)}** ({d_pct}%)")

    st.markdown("---")

    # Exam history
    st.subheader("📝 Last Exam Score")
    if st.session_state.exam_score:
        score = st.session_state.exam_score
        pct_e = score["percentage"]
        passed = pct_e >= 70
        color = "#2ecc71" if passed else "#e74c3c"
        st.markdown(f"""
        <div style='background:{color}15; border: 2px solid {color}; padding:20px; border-radius:12px;'>
            <h3 style='color:{color}; margin:0;'>{pct_e:.1f}% — {"PASS ✅" if passed else "Not Yet ❌"}</h3>
            <p>{score["correct"]}/{score["total"]} correct &nbsp;|&nbsp; Time: {score["time_taken"]}</p>
            <b>Domain breakdown:</b>
            <ul>
        """ + "".join(
            f"<li>{d}: {r['correct']}/{r['total']} ({int(r['correct']/r['total']*100)}%)</li>"
            for d, r in score["domain_results"].items()
        ) + "</ul></div>", unsafe_allow_html=True)
    else:
        st.info("No exam taken yet. Head to Practice Exam to test your knowledge!")

    st.markdown("---")
    if st.button("🔄 Reset All Progress", type="secondary"):
        st.session_state.fc_known = set()
        st.session_state.fc_unknown = set()
        st.session_state.exam_score = None
        st.session_state.exam_questions = None
        st.session_state.exam_submitted = False
        st.rerun()


# ─────────────────────────────────────────────────────────────
# PAGE: SERVICE CHEAT SHEET
# ─────────────────────────────────────────────────────────────

def render_cheat_sheet():
    st.title("🗒️ AWS Service Cheat Sheet")

    CHEAT_SHEET = {
        "Compute": [
            ("EC2", "Elastic Compute Cloud", "Virtual servers. Full control. Many instance types."),
            ("Lambda", "Serverless Functions", "Run code without servers. Event-driven. 15 min max."),
            ("ECS", "Elastic Container Service", "Docker container management."),
            ("EKS", "Elastic Kubernetes Service", "Managed Kubernetes."),
            ("Fargate", "Serverless Containers", "Run containers without managing EC2."),
            ("Elastic Beanstalk", "PaaS App Deployment", "Upload code, AWS handles the rest."),
            ("Lightsail", "Simple VPS", "Easy VMs for simple apps."),
            ("Batch", "Batch Computing", "Run batch jobs at scale."),
            ("Outposts", "On-premises AWS", "AWS rack in your data center."),
            ("Wavelength", "5G Edge Compute", "Ultra-low latency apps on 5G."),
        ],
        "Storage": [
            ("S3", "Simple Storage Service", "Object storage. 11 9s durability. Unlimited capacity."),
            ("EBS", "Elastic Block Store", "Block storage for EC2. Like a hard drive."),
            ("EFS", "Elastic File System", "Managed NFS file system. Linux. Auto-scales."),
            ("FSx", "Managed File Systems", "Windows File Server or Lustre (HPC)."),
            ("S3 Glacier", "Archive Storage", "Lowest cost. Retrieval takes minutes to hours."),
            ("Storage Gateway", "Hybrid Storage", "Connect on-prem to cloud storage."),
            ("Snowcone", "8TB Edge Device", "Physical transfer for remote locations."),
            ("Snowball Edge", "80TB Transfer", "Physical device for large migrations."),
            ("Snowmobile", "100PB Transfer", "Shipping container for massive data."),
        ],
        "Databases": [
            ("RDS", "Relational DB Service", "Managed MySQL, PostgreSQL, Oracle, SQL Server, MariaDB."),
            ("Aurora", "Cloud-native RDBMS", "MySQL/PostgreSQL compatible. 5x faster. Auto-scales."),
            ("DynamoDB", "NoSQL Key-Value", "Single-digit ms latency. Serverless. Any scale."),
            ("ElastiCache", "In-memory Cache", "Redis or Memcached. Sub-ms latency."),
            ("Redshift", "Data Warehouse", "Petabyte-scale analytics. BI queries."),
            ("DocumentDB", "Document DB", "MongoDB-compatible managed database."),
            ("Neptune", "Graph Database", "For highly connected data. Social networks, fraud."),
            ("QLDB", "Ledger Database", "Immutable, cryptographically verifiable transaction log."),
            ("Keyspaces", "Cassandra", "Managed Apache Cassandra."),
            ("Timestream", "Time Series DB", "IoT and operational applications time series data."),
        ],
        "Networking": [
            ("VPC", "Virtual Private Cloud", "Your isolated virtual network in AWS."),
            ("CloudFront", "CDN", "Global content delivery. 400+ edge locations."),
            ("Route 53", "DNS & Domain", "Scalable DNS. Domain registration. Health checks."),
            ("ALB", "Application Load Balancer", "HTTP/HTTPS load balancing. Layer 7."),
            ("NLB", "Network Load Balancer", "TCP/UDP load balancing. Layer 4. Ultra-high performance."),
            ("API Gateway", "API Management", "Create, publish, maintain REST and WebSocket APIs."),
            ("Direct Connect", "Dedicated Network", "Private, dedicated network connection to AWS."),
            ("Site-to-Site VPN", "Encrypted Tunnel", "Encrypted connection over internet to your VPC."),
            ("Transit Gateway", "Network Hub", "Connect VPCs and on-premises networks centrally."),
            ("Global Accelerator", "Performance", "Improve global app availability using AWS network."),
        ],
        "Security": [
            ("IAM", "Identity & Access Mgmt", "Users, groups, roles, policies. Authentication/Authorization."),
            ("Cognito", "User Identity", "User sign-up/sign-in for web/mobile apps."),
            ("Shield", "DDoS Protection", "Standard (free) or Advanced ($3K/mo) DDoS protection."),
            ("WAF", "Web App Firewall", "Block SQLi, XSS, bad bots. Works with CF, ALB, API GW."),
            ("GuardDuty", "Threat Detection", "ML-based threat detection from logs."),
            ("Inspector", "Vulnerability Scan", "EC2 and container image vulnerability assessment."),
            ("Macie", "Sensitive Data", "Discover PII and sensitive data in S3."),
            ("KMS", "Key Management", "Create and manage encryption keys."),
            ("Secrets Manager", "Secret Storage", "Store and rotate secrets, passwords, API keys."),
            ("CloudTrail", "API Audit Log", "Record all API calls. Compliance and auditing."),
            ("Config", "Config Compliance", "Track resource configurations. Compliance rules."),
            ("Security Hub", "Security Center", "Aggregated security findings from many services."),
        ],
        "Management & Governance": [
            ("CloudWatch", "Monitoring", "Metrics, logs, alarms, dashboards for AWS resources."),
            ("CloudFormation", "IaC", "Provision resources via JSON/YAML templates."),
            ("Systems Manager", "Operations", "Patch, run commands, parameter store. No SSH needed."),
            ("Organizations", "Multi-account", "Manage multiple accounts. SCPs. Consolidated billing."),
            ("Control Tower", "Landing Zone", "Automated multi-account governance with guardrails."),
            ("Trusted Advisor", "Best Practices", "Checks for cost, security, performance, reliability."),
            ("AWS Config", "Change Tracking", "Continuously monitor and record resource changes."),
            ("Service Catalog", "Self-Service", "Create and manage approved product portfolios."),
            ("License Manager", "License Tracking", "Manage software licenses."),
            ("OpsWorks", "Config Mgmt", "Managed Chef/Puppet for configuration management."),
        ],
        "Developer Tools": [
            ("CodeCommit", "Git Repos", "Managed private Git repositories."),
            ("CodeBuild", "Build Service", "Compile, test, and package code. CI."),
            ("CodeDeploy", "Deployment", "Automate deployments to EC2, Lambda, ECS."),
            ("CodePipeline", "CI/CD Pipeline", "Continuous integration and delivery pipelines."),
            ("CodeStar", "Dev Projects", "Unified UI for managing software development."),
            ("Cloud9", "IDE", "Cloud-based integrated development environment."),
            ("X-Ray", "Distributed Tracing", "Analyze and debug distributed applications."),
        ],
        "Analytics": [
            ("Athena", "SQL on S3", "Query S3 data with SQL. Serverless. Pay per query."),
            ("EMR", "Hadoop/Spark", "Managed big data processing. Hadoop, Spark, Hive."),
            ("Kinesis", "Real-time Streams", "Collect and process real-time streaming data."),
            ("Glue", "ETL Service", "Managed ETL. Data catalog. Serverless."),
            ("QuickSight", "BI Dashboards", "Business intelligence and visualization tool."),
            ("Data Pipeline", "Data Workflows", "Process and move data between AWS services."),
            ("Lake Formation", "Data Lake", "Build secure data lakes on S3."),
        ],
    }

    search = st.text_input("🔍 Search services...", placeholder="e.g., 'database', 'storage', 'Lambda'")

    for category, services in CHEAT_SHEET.items():
        filtered = services
        if search:
            filtered = [s for s in services if search.lower() in s[0].lower() or
                       search.lower() in s[1].lower() or search.lower() in s[2].lower() or
                       search.lower() in category.lower()]
        if not filtered:
            continue

        with st.expander(f"**{category}** ({len(filtered)} services)", expanded=bool(search)):
            cols = st.columns(3)
            for i, (name, full_name, desc) in enumerate(filtered):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style='background:#f8f9fa; border-radius:8px; padding:10px; margin-bottom:8px; border-left:3px solid #FF9900;'>
                        <div style='font-weight:700; color:#232F3E;'>{name}</div>
                        <div style='color:#FF9900; font-size:0.8em; margin:2px 0;'>{full_name}</div>
                        <div style='color:#555; font-size:0.85em;'>{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AWS Cloud Practitioner Study",
        page_icon="☁️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Global CSS
    st.markdown("""
    <style>
        .stApp { background-color: #f5f7fa; }
        .stButton > button { border-radius: 8px; font-weight: 600; }
        .stButton > button[kind="primary"] { background: #FF9900; border-color: #FF9900; color: #232F3E; }
        .stButton > button[kind="primary"]:hover { background: #e6880a; border-color: #e6880a; }
        .stSelectbox, .stMultiSelect { border-radius: 8px; }
        .stExpander { border-radius: 8px; }
        div[data-testid="metric-container"] {
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        h1, h2, h3 { color: #232F3E; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:10px 0;'>
            <div style='font-size:2.5em;'>☁️</div>
            <div style='font-weight:700; color:#FF9900; font-size:1.1em;'>AWS Cloud Practitioner</div>
            <div style='color:#666; font-size:0.8em;'>CLF-C02 Study App</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        pages = {
            "🏠 Home": "Home",
            "📖 Flashcards": "Flashcards",
            "📝 Practice Exam": "Practice Exam",
            "📚 Study Guide": "Study Guide",
            "🗒️ Service Cheat Sheet": "Cheat Sheet",
            "📈 Progress Tracker": "Progress",
        }

        for label, page_key in pages.items():
            is_active = st.session_state.active_tab == page_key
            if st.button(label, use_container_width=True,
                        type="primary" if is_active else "secondary"):
                st.session_state.active_tab = page_key
                st.rerun()

        st.markdown("---")

        # Quick stats in sidebar
        total_cards = sum(len(v) for v in FLASHCARD_DECKS.values())
        known = len(st.session_state.fc_known)
        pct = int((known / total_cards) * 100) if total_cards else 0
        st.markdown(f"**📊 Quick Stats**")
        st.progress(pct / 100, text=f"Cards: {known}/{total_cards} ({pct}%)")

        if st.session_state.exam_score:
            score = st.session_state.exam_score
            e_pct = score["percentage"]
            color = "🟢" if e_pct >= 70 else "🔴"
            st.markdown(f"{color} Last Exam: **{e_pct:.0f}%**")

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.75em; color:#888; text-align:center;'>
            CLF-C02 • 65 questions • 90 min<br>
            Passing score: 700/1000 (70%)
        </div>
        """, unsafe_allow_html=True)

    # Route to active page
    page = st.session_state.active_tab
    if page == "Home":
        render_home()
    elif page == "Flashcards":
        render_flashcards()
    elif page == "Practice Exam":
        render_practice_exam()
    elif page == "Study Guide":
        render_study_guide()
    elif page == "Cheat Sheet":
        render_cheat_sheet()
    elif page == "Progress":
        render_progress()


if __name__ == "__main__":
    main()
