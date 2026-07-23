"""
AWS Certified Cloud Practitioner (CLF-C02) — Exam Simulator
A timed, 65-question multiple-choice practice exam that mirrors the real
exam format: 90-minute countdown timer, single-question navigation with
a jump-to-question grid, mark-for-review, auto-submit on timeout, and a
scored results report broken down by exam domain.
"""

import copy
import os
import random
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import streamlit as st

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

EXAM_DURATION_SECONDS = 90 * 60
PASS_THRESHOLD_PERCENT = 70  # approximate; AWS's real cut score is a scaled 700/1000
RESULTS_RECIPIENT_EMAIL = "philjones820@gmail.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

DOMAIN_WEIGHTS = {
    "Cloud Concepts": 24,
    "Security and Compliance": 30,
    "Cloud Technology and Services": 34,
    "Billing, Pricing, and Support": 12,
}

# ─────────────────────────────────────────────────────────────
# QUESTION BANK — 65 questions, weighted to match the official
# AWS CLF-C02 exam guide domain breakdown
# ─────────────────────────────────────────────────────────────

QUESTIONS = [
    # ───────────── Cloud Concepts (16) ─────────────
    {
        "domain": "Cloud Concepts",
        "question": "What is the primary purpose of the AWS Well-Architected Framework?",
        "options": [
            "To provide a consistent approach for evaluating and improving cloud architectures across six pillars",
            "To automatically provision AWS resources based on a CloudFormation template",
            "To calculate the total cost of ownership for migrating to AWS",
            "To provide a centralized dashboard for monitoring AWS service health",
        ],
        "answer": 0,
        "explanation": "The AWS Well-Architected Framework provides a consistent set of best practices for building and evaluating cloud architectures across six pillars: Operational Excellence, Security, Reliability, Performance Efficiency, Cost Optimization, and Sustainability.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "Which cloud computing benefit describes replacing large upfront infrastructure investments with pay-as-you-go pricing?",
        "options": [
            "Trading capital expense for variable expense",
            "Economies of scale",
            "Increased speed and agility",
            "Going global in minutes",
        ],
        "answer": 0,
        "explanation": "This benefit lets organizations pay only for the IT resources they consume, rather than investing heavily in data centers and servers before knowing how they'll be used.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "A company notices that as more AWS customers use the cloud, AWS's operating costs per unit decrease, and those savings are passed on as lower prices. Which cloud benefit does this describe?",
        "options": [
            "Economies of scale",
            "Elasticity",
            "Fault tolerance",
            "High availability",
        ],
        "answer": 0,
        "explanation": "Because AWS aggregates usage from hundreds of thousands of customers, it achieves higher economies of scale, which translates into lower pay-as-you-go prices for customers.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "What does 'elasticity' mean in the context of AWS cloud computing?",
        "options": [
            "The ability to automatically scale compute resources up or down to match demand",
            "The ability of a system to continue operating despite component failures",
            "The physical distribution of AWS data centers across the globe",
            "The process of encrypting data both at rest and in transit",
        ],
        "answer": 0,
        "explanation": "Elasticity refers to the ability to automatically scale resources up or down as needed, so you only pay for what you use rather than over-provisioning for peak demand.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "Which AWS global infrastructure component is made up of one or more discrete data centers with redundant power, networking, and connectivity?",
        "options": [
            "Availability Zone",
            "Region",
            "Edge Location",
            "Local Zone",
        ],
        "answer": 0,
        "explanation": "An Availability Zone (AZ) consists of one or more discrete data centers, each with redundant power, networking, and connectivity, housed in separate facilities.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "A media company wants to cache and deliver video content to users worldwide with low latency. Which type of AWS infrastructure location is used for this purpose?",
        "options": [
            "Edge Location",
            "Availability Zone",
            "Region",
            "VPC",
        ],
        "answer": 0,
        "explanation": "Edge Locations are used by Amazon CloudFront to cache content closer to end users, reducing latency for content delivery.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "A company runs some workloads in its own data center and connects them to AWS resources over a VPN. Which cloud deployment model does this describe?",
        "options": [
            "Hybrid deployment",
            "Public cloud deployment",
            "Private cloud (on-premises) deployment",
            "Multi-cloud deployment",
        ],
        "answer": 0,
        "explanation": "A hybrid deployment model connects on-premises infrastructure with cloud resources, often used during a phased migration.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "Which statement best describes Infrastructure as a Service (IaaS)?",
        "options": [
            "The cloud provider manages the underlying hardware and virtualization while the customer manages the OS, middleware, and applications",
            "The cloud provider manages everything including the application, and the customer just uses the software",
            "The cloud provider manages the OS and runtime while the customer manages only application code and data",
            "The customer manages the physical hardware while the provider manages the software",
        ],
        "answer": 0,
        "explanation": "IaaS (e.g., Amazon EC2) gives customers the most control, letting them manage the OS, storage, and deployed applications, while AWS manages the underlying physical infrastructure.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "What does the cloud computing benefit 'go global in minutes' primarily allow a business to do?",
        "options": [
            "Deploy applications to multiple AWS Regions around the world with just a few clicks",
            "Reduce the cost of on-premises hardware maintenance",
            "Automatically encrypt data transmitted between Regions",
            "Guarantee 100% uptime for all deployed applications",
        ],
        "answer": 0,
        "explanation": "AWS's global infrastructure allows companies to deploy applications in multiple Regions worldwide with minimal effort, improving latency and redundancy for a global user base.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "Which pillar of the AWS Well-Architected Framework focuses on running and monitoring systems to deliver business value and continuously improve processes?",
        "options": [
            "Operational Excellence",
            "Performance Efficiency",
            "Reliability",
            "Cost Optimization",
        ],
        "answer": 0,
        "explanation": "The Operational Excellence pillar focuses on running and monitoring systems, and continually improving processes and procedures.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "A startup experiences highly unpredictable traffic spikes and does not want to pay for idle infrastructure during quiet periods. Which cloud characteristic best addresses this need?",
        "options": [
            "Elasticity",
            "Durability",
            "Consolidated billing",
            "Fault tolerance",
        ],
        "answer": 0,
        "explanation": "Elasticity allows resources to automatically scale out during spikes and scale back in during quiet periods, so the company only pays for what it actually uses.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "Which statement about AWS Regions is TRUE?",
        "options": [
            "Each Region is a separate geographic area that contains multiple, isolated Availability Zones",
            "A Region is a single physical data center",
            "All AWS Regions share the same set of available services",
            "Data automatically replicates between all Regions by default",
        ],
        "answer": 0,
        "explanation": "An AWS Region is a physical geographic location containing multiple isolated Availability Zones, which are connected through low-latency links.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "Which of the following is an example of Software as a Service (SaaS) that can be consumed on AWS?",
        "options": [
            "Amazon Chime for video conferencing",
            "Amazon EC2 for hosting virtual servers",
            "AWS Lambda for running custom application code",
            "Amazon VPC for network isolation",
        ],
        "answer": 0,
        "explanation": "Amazon Chime is a ready-to-use, fully managed application (SaaS) — the customer doesn't manage any underlying infrastructure, unlike EC2 (IaaS) or Lambda.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "What is the main purpose of AWS Trusted Advisor?",
        "options": [
            "To provide real-time recommendations across cost optimization, performance, security, fault tolerance, and service limits",
            "To record all API calls made within an AWS account",
            "To provide a marketplace for third-party software",
            "To manage encryption keys for data at rest",
        ],
        "answer": 0,
        "explanation": "AWS Trusted Advisor inspects your AWS environment and provides recommendations in categories including cost optimization, performance, security, fault tolerance, and service limits.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "Which term describes a system's ability to remain operational and accessible even when individual components fail?",
        "options": [
            "High availability",
            "Elasticity",
            "Economies of scale",
            "Consolidated billing",
        ],
        "answer": 0,
        "explanation": "High availability describes an architecture designed with redundancy (e.g., across multiple Availability Zones) so the system continues to function despite individual component failures.",
    },
    {
        "domain": "Cloud Concepts",
        "question": "A company is comparing the cost of running a data center for 5 years versus migrating to AWS. Which AWS resource would best help estimate cost savings by comparing on-premises costs to projected AWS costs?",
        "options": [
            "AWS Pricing Calculator and Total Cost of Ownership (TCO) comparison tools",
            "AWS Trusted Advisor",
            "AWS Config",
            "Amazon CloudWatch",
        ],
        "answer": 0,
        "explanation": "AWS provides TCO comparison resources and the AWS Pricing Calculator to help organizations estimate and compare the cost of on-premises infrastructure versus AWS cloud services.",
    },
    # ───────────── Security and Compliance (20) ─────────────
    {
        "domain": "Security and Compliance",
        "question": "Under the AWS Shared Responsibility Model, who is responsible for patching the guest operating system on an Amazon EC2 instance?",
        "options": [
            "The customer",
            "AWS",
            "Both AWS and the customer share this responsibility equally",
            "Neither; patching is not required for EC2",
        ],
        "answer": 0,
        "explanation": "For EC2, AWS is responsible for security OF the cloud (hardware, hypervisor), while the customer is responsible for security IN the cloud, which includes patching the guest OS, applications, and configuring security groups.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which AWS service allows you to create and centrally manage cryptographic keys used to encrypt data across AWS services?",
        "options": [
            "AWS Key Management Service (KMS)",
            "AWS IAM",
            "AWS Shield",
            "AWS Certificate Manager",
        ],
        "answer": 0,
        "explanation": "AWS KMS lets you create, manage, and control cryptographic keys used to encrypt data across a wide range of AWS services.",
    },
    {
        "domain": "Security and Compliance",
        "question": "What is the recommended best practice for securing the AWS account root user?",
        "options": [
            "Enable multi-factor authentication (MFA) and avoid using the root user for everyday tasks",
            "Share the root user credentials with all administrators for convenience",
            "Use the root user for all daily development and operations tasks",
            "Disable the root user account entirely so it can never be used",
        ],
        "answer": 0,
        "explanation": "Best practice is to enable MFA on the root account, lock away its credentials, and create individual IAM users with least-privilege permissions for day-to-day work.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which AWS service provides automatic DDoS protection at no additional cost to all AWS customers?",
        "options": [
            "AWS Shield Standard",
            "AWS Shield Advanced",
            "AWS WAF",
            "Amazon GuardDuty",
        ],
        "answer": 0,
        "explanation": "AWS Shield Standard is automatically enabled for all AWS customers at no additional cost and protects against common, most frequently occurring network and transport layer DDoS attacks.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which AWS service records and logs all API calls made within your AWS account, including who made the call and when?",
        "options": [
            "AWS CloudTrail",
            "Amazon CloudWatch",
            "AWS Config",
            "AWS X-Ray",
        ],
        "answer": 0,
        "explanation": "AWS CloudTrail records API activity in your AWS account for governance, compliance, and auditing purposes, capturing details such as the identity of the caller and the time of the call.",
    },
    {
        "domain": "Security and Compliance",
        "question": "What is the primary purpose of an IAM policy?",
        "options": [
            "To define permissions that specify what actions are allowed or denied on which AWS resources",
            "To encrypt data stored in Amazon S3 buckets",
            "To monitor network traffic for suspicious activity",
            "To create backups of EC2 instances",
        ],
        "answer": 0,
        "explanation": "IAM policies are JSON documents that define permissions, specifying which actions are allowed or denied on which AWS resources for a user, group, or role.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which AWS service uses machine learning to continuously analyze CloudTrail logs, VPC Flow Logs, and DNS logs to detect malicious or unauthorized activity?",
        "options": [
            "Amazon GuardDuty",
            "AWS Config",
            "Amazon Inspector",
            "AWS Artifact",
        ],
        "answer": 0,
        "explanation": "Amazon GuardDuty is an intelligent threat detection service that continuously monitors for malicious activity and unauthorized behavior using machine learning and threat intelligence.",
    },
    {
        "domain": "Security and Compliance",
        "question": "An application running on an EC2 instance needs temporary, automatically rotated credentials to access an S3 bucket. What should be used instead of embedding long-term access keys?",
        "options": [
            "An IAM role attached to the EC2 instance",
            "The AWS account root user credentials",
            "A hardcoded IAM user access key in the application code",
            "An S3 bucket policy granting public access",
        ],
        "answer": 0,
        "explanation": "IAM roles provide temporary security credentials that are automatically rotated, making them the secure best practice for granting AWS resources like EC2 instances access to other AWS services.",
    },
    {
        "domain": "Security and Compliance",
        "question": "What does the principle of least privilege mean in AWS Identity and Access Management?",
        "options": [
            "Granting only the minimum permissions necessary for a user or system to perform its required tasks",
            "Granting administrator access to all new IAM users by default",
            "Granting full access to all resources within a single AWS account",
            "Disabling all permissions until manually approved by AWS Support",
        ],
        "answer": 0,
        "explanation": "The principle of least privilege means granting only the permissions required to perform a specific task, reducing the potential impact of compromised credentials.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which AWS resource provides self-service, on-demand access to AWS security and compliance reports, such as SOC and PCI reports?",
        "options": [
            "AWS Artifact",
            "AWS Config",
            "AWS Trusted Advisor",
            "AWS Systems Manager",
        ],
        "answer": 0,
        "explanation": "AWS Artifact is a free, self-service portal for accessing AWS compliance reports and agreements, such as SOC, PCI, and ISO reports.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which AWS service performs automated security assessments to identify vulnerabilities and deviations from best practices on EC2 instances and container images?",
        "options": [
            "Amazon Inspector",
            "Amazon Macie",
            "AWS Shield",
            "AWS Firewall Manager",
        ],
        "answer": 0,
        "explanation": "Amazon Inspector automatically assesses applications for vulnerabilities and deviations from security best practices, including on EC2 instances and container images in Amazon ECR.",
    },
    {
        "domain": "Security and Compliance",
        "question": "What is the function of an Amazon VPC Security Group?",
        "options": [
            "A virtual, stateful firewall that controls inbound and outbound traffic at the instance level",
            "A physical firewall device installed in an AWS data center",
            "A service that encrypts all traffic entering a VPC",
            "A stateless firewall that operates at the subnet level",
        ],
        "answer": 0,
        "explanation": "A Security Group acts as a virtual firewall at the instance level and is stateful — return traffic is automatically allowed regardless of outbound/inbound rules.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which statement correctly distinguishes a Network ACL from a Security Group?",
        "options": [
            "A Network ACL is stateless and operates at the subnet level, while a Security Group is stateful and operates at the instance level",
            "A Network ACL is stateful and operates at the instance level, while a Security Group is stateless and operates at the subnet level",
            "Both Network ACLs and Security Groups are stateless and operate at the subnet level",
            "Both Network ACLs and Security Groups are stateful and operate at the instance level",
        ],
        "answer": 0,
        "explanation": "Network ACLs are stateless (return traffic must be explicitly allowed) and act as a firewall for subnets, while Security Groups are stateful and act as a firewall for individual instances.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which AWS service uses machine learning to automatically discover, classify, and protect sensitive data such as personally identifiable information (PII) stored in Amazon S3?",
        "options": [
            "Amazon Macie",
            "Amazon GuardDuty",
            "AWS Config",
            "AWS Secrets Manager",
        ],
        "answer": 0,
        "explanation": "Amazon Macie uses machine learning and pattern matching to discover, classify, and help protect sensitive data such as PII stored in Amazon S3.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which statement accurately describes the difference between an IAM user and an IAM role?",
        "options": [
            "An IAM user has permanent long-term credentials, while an IAM role provides temporary credentials that can be assumed by trusted entities",
            "An IAM role has permanent long-term credentials, while an IAM user provides temporary credentials",
            "IAM users and IAM roles are functionally identical with no differences",
            "IAM roles can only be used by AWS support staff, not by customers",
        ],
        "answer": 0,
        "explanation": "An IAM user represents a person or application with long-term credentials, while an IAM role is assumed temporarily and does not have long-term credentials attached to it.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which AWS service aggregates, organizes, and prioritizes security findings from services like GuardDuty, Inspector, and Macie into a single dashboard?",
        "options": [
            "AWS Security Hub",
            "AWS Config",
            "Amazon CloudWatch",
            "AWS CloudTrail",
        ],
        "answer": 0,
        "explanation": "AWS Security Hub provides a comprehensive view of your security posture by aggregating, organizing, and prioritizing security findings from multiple AWS services.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Why is Multi-Factor Authentication (MFA) recommended for AWS accounts?",
        "options": [
            "It adds an extra layer of protection beyond a username and password, requiring an additional authentication factor to sign in",
            "It automatically encrypts all data stored in S3 buckets",
            "It replaces the need for IAM policies entirely",
            "It is required to launch any EC2 instance",
        ],
        "answer": 0,
        "explanation": "MFA adds an additional layer of security by requiring users to provide a second form of authentication (such as a code from a virtual or hardware MFA device) in addition to their password.",
    },
    {
        "domain": "Security and Compliance",
        "question": "A company needs to ensure that all objects uploaded to an S3 bucket are automatically encrypted at rest without any custom application logic. Which feature should they use?",
        "options": [
            "Default encryption for the S3 bucket using SSE-S3 or SSE-KMS",
            "Amazon Macie sensitive data discovery",
            "AWS Shield Advanced",
            "VPC endpoint policies",
        ],
        "answer": 0,
        "explanation": "S3 default encryption (using SSE-S3, SSE-KMS, or SSE-C) automatically encrypts every object as it is stored, without requiring any changes to the uploading application.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Which AWS service continuously assesses, audits, and evaluates the configurations of your AWS resources against desired configurations over time?",
        "options": [
            "AWS Config",
            "AWS CloudTrail",
            "Amazon Inspector",
            "AWS Systems Manager",
        ],
        "answer": 0,
        "explanation": "AWS Config continuously monitors and records AWS resource configurations, allowing you to evaluate them against desired configurations for compliance auditing and change management.",
    },
    {
        "domain": "Security and Compliance",
        "question": "Under the Shared Responsibility Model, which of the following is the CUSTOMER's responsibility when using Amazon RDS (a managed database service)?",
        "options": [
            "Managing database user accounts, permissions, and data encryption settings",
            "Patching the underlying database engine software",
            "Maintaining the physical hardware hosting the database",
            "Managing the hypervisor layer",
        ],
        "answer": 0,
        "explanation": "For managed services like RDS, AWS handles the underlying infrastructure and database engine patching, while the customer remains responsible for managing database accounts/permissions, data, and configuring settings like encryption.",
    },
    # ───────────── Cloud Technology and Services (21) ─────────────
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS service is best suited for storing and retrieving any amount of unstructured data as objects, with 99.999999999% (11 nines) durability?",
        "options": [
            "Amazon S3",
            "Amazon EBS",
            "Amazon RDS",
            "Amazon EFS",
        ],
        "answer": 0,
        "explanation": "Amazon S3 (Simple Storage Service) is an object storage service designed for 99.999999999% durability, suitable for storing virtually unlimited amounts of unstructured data.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS compute service lets you run code in response to events without provisioning or managing any servers?",
        "options": [
            "AWS Lambda",
            "Amazon EC2",
            "Amazon Lightsail",
            "AWS Elastic Beanstalk",
        ],
        "answer": 0,
        "explanation": "AWS Lambda is a serverless compute service that runs your code in response to events, automatically managing the underlying compute resources and charging only for the compute time consumed.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "A company needs a relational database that guarantees ACID transactions and supports complex SQL joins. Which AWS service is most appropriate?",
        "options": [
            "Amazon RDS",
            "Amazon DynamoDB",
            "Amazon S3",
            "Amazon ElastiCache",
        ],
        "answer": 0,
        "explanation": "Amazon RDS is a managed relational database service supporting engines like MySQL, PostgreSQL, and Aurora, which provide ACID compliance and support complex SQL queries and joins.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS service should be used to decouple application components using a fully managed message queuing service?",
        "options": [
            "Amazon SQS",
            "Amazon SNS",
            "AWS Step Functions",
            "Amazon EventBridge",
        ],
        "answer": 0,
        "explanation": "Amazon Simple Queue Service (SQS) is a fully managed message queuing service that helps decouple and scale microservices, distributed systems, and serverless applications.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which S3 storage class offers the lowest storage cost for data that is rarely accessed and can tolerate a retrieval time ranging from minutes to hours?",
        "options": [
            "S3 Glacier (Flexible Retrieval or Deep Archive)",
            "S3 Standard",
            "S3 Standard-Infrequent Access",
            "S3 Intelligent-Tiering",
        ],
        "answer": 0,
        "explanation": "The S3 Glacier storage classes are designed for archival data and offer the lowest storage costs, with retrieval times ranging from minutes (Flexible Retrieval) to hours (Deep Archive).",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS networking service allows you to create a logically isolated virtual network where you define your own IP address range, subnets, and route tables?",
        "options": [
            "Amazon VPC",
            "Amazon Route 53",
            "AWS Direct Connect",
            "Elastic Load Balancing",
        ],
        "answer": 0,
        "explanation": "Amazon Virtual Private Cloud (VPC) lets you provision a logically isolated section of the AWS Cloud where you can launch resources in a virtual network you define.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS database service provides a fully managed NoSQL key-value and document database with single-digit millisecond performance at any scale?",
        "options": [
            "Amazon DynamoDB",
            "Amazon RDS",
            "Amazon Redshift",
            "Amazon Neptune",
        ],
        "answer": 0,
        "explanation": "Amazon DynamoDB is a fully managed, serverless NoSQL database that delivers consistent single-digit millisecond performance at any scale.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "What is the primary purpose of an EC2 Auto Scaling group?",
        "options": [
            "To automatically add or remove EC2 instances based on demand, maintaining application availability and cost efficiency",
            "To automatically encrypt data on EC2 instances",
            "To distribute incoming traffic across multiple EC2 instances",
            "To back up EC2 instance data to Amazon S3",
        ],
        "answer": 0,
        "explanation": "Auto Scaling groups automatically adjust the number of EC2 instances up or down according to conditions you define, helping ensure you have the right amount of capacity to handle the load.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which type of Elastic Load Balancer operates at Layer 7 (the application layer) and can route traffic based on URL path or hostname?",
        "options": [
            "Application Load Balancer (ALB)",
            "Network Load Balancer (NLB)",
            "Classic Load Balancer (CLB)",
            "Gateway Load Balancer (GWLB)",
        ],
        "answer": 0,
        "explanation": "The Application Load Balancer operates at Layer 7 and supports content-based routing, such as routing requests based on URL path, hostname, or HTTP headers.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS service allows you to deploy and scale containerized applications using Kubernetes without needing to install or operate the Kubernetes control plane?",
        "options": [
            "Amazon Elastic Kubernetes Service (EKS)",
            "Amazon Elastic Container Service (ECS)",
            "AWS Elastic Beanstalk",
            "AWS Fargate (standalone, without EKS)",
        ],
        "answer": 0,
        "explanation": "Amazon EKS is a managed Kubernetes service — AWS manages the Kubernetes control plane, so customers don't need to install, operate, or maintain it themselves.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS service delivers content to end users with low latency by caching copies of content at edge locations around the world?",
        "options": [
            "Amazon CloudFront",
            "Amazon Route 53",
            "AWS Global Accelerator",
            "AWS Direct Connect",
        ],
        "answer": 0,
        "explanation": "Amazon CloudFront is a content delivery network (CDN) service that caches content at edge locations globally to reduce latency for end users.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "What is Amazon Route 53 primarily used for?",
        "options": [
            "Scalable Domain Name System (DNS) and domain registration, routing users to endpoints",
            "Distributing incoming application traffic across multiple targets",
            "Caching static content close to end users",
            "Establishing a dedicated private network connection to AWS",
        ],
        "answer": 0,
        "explanation": "Amazon Route 53 is a highly available and scalable DNS web service that also supports domain registration and various routing policies, including health-check-based failover.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS storage service provides persistent block-level storage volumes that can be attached to a single EC2 instance?",
        "options": [
            "Amazon EBS (Elastic Block Store)",
            "Amazon S3",
            "Amazon EFS",
            "AWS Storage Gateway",
        ],
        "answer": 0,
        "explanation": "Amazon EBS provides persistent block storage volumes for use with EC2 instances, similar to an attachable hard drive, and data persists independently of the instance lifecycle.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS service allows you to provision and manage AWS resources using declarative templates written in JSON or YAML, enabling Infrastructure as Code?",
        "options": [
            "AWS CloudFormation",
            "AWS Config",
            "AWS Systems Manager",
            "AWS OpsWorks",
        ],
        "answer": 0,
        "explanation": "AWS CloudFormation lets you model, provision, and manage AWS resources using templates, enabling repeatable and version-controlled Infrastructure as Code deployments.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "A company needs to migrate a large on-premises relational database to AWS while keeping the source database operational and minimizing downtime during the cutover. Which service is best suited for this?",
        "options": [
            "AWS Database Migration Service (DMS)",
            "AWS Snowball",
            "AWS Direct Connect",
            "Amazon FSx",
        ],
        "answer": 0,
        "explanation": "AWS Database Migration Service (DMS) helps migrate databases to AWS quickly and securely while keeping the source database fully operational during the migration process.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "What differentiates an EC2 Spot Instance from an EC2 On-Demand Instance?",
        "options": [
            "Spot Instances use spare EC2 capacity at a steep discount but can be interrupted by AWS with short notice",
            "Spot Instances guarantee capacity is never interrupted, unlike On-Demand Instances",
            "Spot Instances require a 1- or 3-year commitment, unlike On-Demand Instances",
            "Spot Instances can only be used for storage workloads, not compute",
        ],
        "answer": 0,
        "explanation": "Spot Instances let you use unused EC2 capacity at up to a 90% discount compared to On-Demand pricing, but AWS can reclaim the capacity with a two-minute interruption notice.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS service provides a fully managed extract, transform, and load (ETL) service for preparing and processing data for analytics?",
        "options": [
            "AWS Glue",
            "Amazon Redshift",
            "Amazon Athena",
            "Amazon QuickSight",
        ],
        "answer": 0,
        "explanation": "AWS Glue is a fully managed ETL service that makes it easy to discover, prepare, and combine data for analytics, machine learning, and application development.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS storage option is best suited for hosting a simple static website consisting of HTML, CSS, and JavaScript files?",
        "options": [
            "Amazon S3 with static website hosting enabled",
            "Amazon EBS attached to an EC2 instance",
            "Amazon RDS",
            "AWS Storage Gateway",
        ],
        "answer": 0,
        "explanation": "Amazon S3 supports static website hosting directly from a bucket, making it a cost-effective and scalable option for serving static content without managing any servers.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "What is the primary purpose of Amazon CloudWatch?",
        "options": [
            "To monitor AWS resources and applications by collecting metrics and logs, and setting alarms",
            "To record all API calls made in an AWS account for auditing",
            "To provide DDoS protection for internet-facing applications",
            "To manage encryption keys for data at rest",
        ],
        "answer": 0,
        "explanation": "Amazon CloudWatch is a monitoring and observability service that collects metrics and logs, allowing you to set alarms and visualize the operational health of your AWS resources and applications.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "Which AWS service allows you to coordinate multiple AWS Lambda functions into serverless visual workflows for building distributed applications?",
        "options": [
            "AWS Step Functions",
            "Amazon SNS",
            "Amazon SQS",
            "AWS Batch",
        ],
        "answer": 0,
        "explanation": "AWS Step Functions lets you coordinate multiple AWS services, including Lambda functions, into visual serverless workflows to build and update distributed applications quickly.",
    },
    {
        "domain": "Cloud Technology and Services",
        "question": "A company needs to run a legacy application that requires full administrative (root) access to a customized operating system. Which AWS service is the most appropriate choice?",
        "options": [
            "Amazon EC2",
            "AWS Lambda",
            "AWS Fargate",
            "Amazon S3",
        ],
        "answer": 0,
        "explanation": "Amazon EC2 provides virtual servers where customers have full administrative control over the guest operating system, making it suitable for legacy applications requiring root-level OS access.",
    },
    # ───────────── Billing, Pricing, and Support (8) ─────────────
    {
        "domain": "Billing, Pricing, and Support",
        "question": "Which AWS Support plan provides 24/7 access to Cloud Support Engineers via phone, chat, and email, along with a guaranteed response time of less than 1 hour for business-critical system down issues?",
        "options": [
            "Business Support",
            "Basic Support",
            "Developer Support",
            "AWS Enterprise On-Ramp only",
        ],
        "answer": 0,
        "explanation": "The Business Support plan provides 24/7 access to Cloud Support Engineers via phone, chat, and email, with a response time of less than 1 hour for production system down cases.",
    },
    {
        "domain": "Billing, Pricing, and Support",
        "question": "Which free AWS tool allows you to estimate the monthly and annual cost of AWS services before you deploy them?",
        "options": [
            "AWS Pricing Calculator",
            "AWS Cost Explorer",
            "AWS Budgets",
            "AWS Cost and Usage Report",
        ],
        "answer": 0,
        "explanation": "The AWS Pricing Calculator lets you model your intended AWS usage and generate a cost estimate before you actually deploy any resources.",
    },
    {
        "domain": "Billing, Pricing, and Support",
        "question": "Which AWS Organizations billing feature allows a company to combine usage from multiple linked AWS accounts to potentially qualify for volume pricing discounts, while receiving one consolidated bill?",
        "options": [
            "Consolidated Billing",
            "AWS Budgets",
            "Cost Allocation Tags",
            "Reserved Instance Marketplace",
        ],
        "answer": 0,
        "explanation": "Consolidated Billing, a feature of AWS Organizations, combines usage across all linked accounts into a single bill and can help accounts collectively reach volume pricing discount thresholds.",
    },
    {
        "domain": "Billing, Pricing, and Support",
        "question": "What is a key benefit of AWS Savings Plans compared to On-Demand pricing?",
        "options": [
            "They offer lower prices in exchange for a commitment to a consistent amount of compute usage over a 1- or 3-year term",
            "They provide free access to premium AWS Support",
            "They guarantee EC2 instances will never be interrupted",
            "They eliminate the need for an AWS account",
        ],
        "answer": 0,
        "explanation": "AWS Savings Plans offer significant discounts compared to On-Demand pricing in exchange for a commitment to a consistent amount of usage (measured in $/hour) for a 1- or 3-year term.",
    },
    {
        "domain": "Billing, Pricing, and Support",
        "question": "Which free tool inspects your AWS environment and provides real-time recommendations across cost optimization, performance, security, fault tolerance, and service limits, with full check access available on Business and Enterprise Support plans?",
        "options": [
            "AWS Trusted Advisor",
            "AWS Config",
            "Amazon CloudWatch",
            "AWS Cost Explorer",
        ],
        "answer": 0,
        "explanation": "AWS Trusted Advisor provides best-practice recommendations across five categories; the Basic Support plan includes 7 core checks, while Business and Enterprise plans unlock the full set of checks.",
    },
    {
        "domain": "Billing, Pricing, and Support",
        "question": "Which AWS Support plan tier is intended for customers running production workloads and includes a dedicated Technical Account Manager (TAM) along with concierge-level support?",
        "options": [
            "Enterprise Support",
            "Developer Support",
            "Basic Support",
            "Business Support",
        ],
        "answer": 0,
        "explanation": "The Enterprise Support plan includes a dedicated Technical Account Manager (TAM), concierge support team access, and the fastest response times, designed for mission-critical production workloads.",
    },
    {
        "domain": "Billing, Pricing, and Support",
        "question": "Which billing report provides the most granular, line-item detail of AWS costs and usage, suitable for deep analysis with tools like Amazon Athena or Amazon QuickSight?",
        "options": [
            "AWS Cost and Usage Report (CUR)",
            "AWS Budgets Report",
            "AWS Trusted Advisor Report",
            "AWS Pricing Calculator Export",
        ],
        "answer": 0,
        "explanation": "The AWS Cost and Usage Report (CUR) provides the most comprehensive and granular set of cost and usage data available, and can be queried using services like Amazon Athena or visualized in Amazon QuickSight.",
    },
    {
        "domain": "Billing, Pricing, and Support",
        "question": "What is the primary purpose of AWS Organizations' Consolidated Billing feature?",
        "options": [
            "To combine billing for multiple AWS accounts into a single payment method while enabling shared volume discounts",
            "To automatically shut down unused EC2 instances to save costs",
            "To provide a dedicated Technical Account Manager for billing questions",
            "To convert On-Demand instances into Reserved Instances automatically",
        ],
        "answer": 0,
        "explanation": "Consolidated Billing combines usage from all accounts in an AWS Organization so they can be billed to a single, designated management account and may share in volume-based pricing discounts.",
    },
]

assert len(QUESTIONS) == 65, f"Expected 65 questions, found {len(QUESTIONS)}"

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────


def prepare_exam():
    """Return a shuffled copy of the question bank with shuffled answer options."""
    questions = copy.deepcopy(QUESTIONS)
    random.shuffle(questions)
    for q in questions:
        correct_text = q["options"][q["answer"]]
        random.shuffle(q["options"])
        q["answer"] = q["options"].index(correct_text)
    return questions


def format_mmss(total_seconds):
    total_seconds = max(0, int(total_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def init_state():
    defaults = {
        "stage": "start",
        "exam_questions": None,
        "answers": {},
        "marked": set(),
        "current_q": 0,
        "start_time": None,
        "confirm_submit": False,
        "attempts": 0,
        "email_status": None,
        "reveal_results": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_exam():
    st.session_state.exam_questions = prepare_exam()
    st.session_state.answers = {}
    st.session_state.marked = set()
    st.session_state.current_q = 0
    st.session_state.start_time = time.time()
    st.session_state.confirm_submit = False
    st.session_state.stage = "exam"


def submit_exam():
    st.session_state.stage = "results"
    st.session_state.attempts += 1


def retake_exam():
    st.session_state.stage = "start"
    st.session_state.exam_questions = None
    st.session_state.answers = {}
    st.session_state.marked = set()
    st.session_state.current_q = 0
    st.session_state.start_time = None
    st.session_state.confirm_submit = False
    st.session_state.email_status = None
    st.session_state.reveal_results = False


def elapsed_seconds():
    if st.session_state.start_time is None:
        return 0
    return time.time() - st.session_state.start_time


def remaining_seconds():
    return max(0, EXAM_DURATION_SECONDS - elapsed_seconds())


# ─────────────────────────────────────────────────────────────
# EMAIL DELIVERY
# ─────────────────────────────────────────────────────────────


def get_email_credentials():
    """Read sender credentials from st.secrets, falling back to env vars."""
    try:
        section = st.secrets.get("email", {})
    except Exception:
        section = {}
    sender = section.get("address") or os.environ.get("EMAIL_ADDRESS")
    app_password = section.get("app_password") or os.environ.get("EMAIL_APP_PASSWORD")
    return sender, app_password


def score_exam(questions, answers):
    total = len(questions)
    correct = sum(1 for i, q in enumerate(questions) if answers.get(i) == q["answer"])
    percent = round((correct / total) * 100, 1)
    passed = percent >= PASS_THRESHOLD_PERCENT

    domain_stats = {d: {"correct": 0, "total": 0} for d in DOMAIN_WEIGHTS}
    for i, q in enumerate(questions):
        d = q["domain"]
        domain_stats[d]["total"] += 1
        if answers.get(i) == q["answer"]:
            domain_stats[d]["correct"] += 1

    return {
        "total": total,
        "correct": correct,
        "percent": percent,
        "passed": passed,
        "domain_stats": domain_stats,
    }


def build_results_text(score, time_taken, questions, answers):
    lines = [
        "AWS CLOUD PRACTITIONER (CLF-C02) — EXAM RESULTS",
        "=" * 50,
        f"Result: {'PASS' if score['passed'] else 'FAIL'}",
        f"Score: {score['correct']}/{score['total']} ({score['percent']}%)",
        f"Time used: {format_mmss(time_taken)}",
        "",
        "Score by domain:",
    ]
    for domain, stats in score["domain_stats"].items():
        pct = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0
        lines.append(f"  - {domain}: {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    lines.append("")
    lines.append("Answer review:")
    for i, q in enumerate(questions):
        user_answer = answers.get(i)
        is_correct = user_answer == q["answer"]
        status = "CORRECT" if is_correct else ("UNANSWERED" if user_answer is None else "INCORRECT")
        lines.append("")
        lines.append(f"Q{i + 1} [{q['domain']}] — {status}")
        lines.append(q["question"])
        if user_answer is not None:
            lines.append(f"  Your answer: {q['options'][user_answer]}")
        lines.append(f"  Correct answer: {q['options'][q['answer']]}")
        lines.append(f"  Explanation: {q['explanation']}")

    return "\n".join(lines)


def build_results_html(score, time_taken, questions, answers):
    status_color = "#16a34a" if score["passed"] else "#dc2626"
    status_text = "PASS" if score["passed"] else "FAIL"

    domain_rows = ""
    for domain, stats in score["domain_stats"].items():
        pct = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0
        domain_rows += (
            f'<tr><td style="padding:6px 10px;border-bottom:1px solid #e5e7eb;">{domain}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:right;">'
            f"{stats['correct']}/{stats['total']} ({pct:.0f}%)</td></tr>"
        )

    review_rows = ""
    for i, q in enumerate(questions):
        user_answer = answers.get(i)
        is_correct = user_answer == q["answer"]
        is_unanswered = user_answer is None
        icon = "✅" if is_correct else ("⬜" if is_unanswered else "❌")
        your_answer_html = (
            f'<p style="margin:4px 0;color:#dc2626;"><strong>Your answer:</strong> {q["options"][user_answer]}</p>'
            if user_answer is not None and not is_correct
            else ""
        )
        if is_unanswered:
            your_answer_html = '<p style="margin:4px 0;color:#b45309;"><strong>You did not answer this question.</strong></p>'

        review_rows += f"""
        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:14px 16px;margin-bottom:10px;">
            <p style="margin:0 0 6px 0;font-weight:600;">{icon} Q{i + 1} · {q['domain']}</p>
            <p style="margin:0 0 8px 0;">{q['question']}</p>
            {your_answer_html}
            <p style="margin:4px 0;color:#16a34a;"><strong>Correct answer:</strong> {q['options'][q['answer']]}</p>
            <p style="margin:6px 0 0 0;color:#4b5563;font-size:0.9em;">💡 {q['explanation']}</p>
        </div>
        """

    return f"""
    <html>
    <body style="font-family:Arial,Helvetica,sans-serif;color:#111827;max-width:720px;margin:0 auto;">
        <div style="background:#232F3E;padding:20px 24px;border-radius:8px 8px 0 0;">
            <h1 style="color:#FF9900;margin:0;font-size:1.4em;">AWS Cloud Practitioner — Exam Results</h1>
        </div>
        <div style="padding:20px 24px;border:1px solid #e5e7eb;border-top:none;">
            <h2 style="color:{status_color};margin-top:0;">{status_text} — {score['correct']}/{score['total']} ({score['percent']}%)</h2>
            <p>Time used: {format_mmss(time_taken)}</p>

            <h3>Score by Domain</h3>
            <table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
                {domain_rows}
            </table>

            <h3>Answer Review</h3>
            {review_rows}

            <p style="margin-top:20px;color:#9ca3af;font-size:0.8em;">
                Generated by the AWS Cloud Practitioner Exam Simulator (Streamlit app).
            </p>
        </div>
    </body>
    </html>
    """


def send_results_email(score, time_taken, questions, answers):
    """Send the scored results to RESULTS_RECIPIENT_EMAIL. Returns (ok, message)."""
    sender, app_password = get_email_credentials()
    if not sender or not app_password:
        return False, (
            "Email credentials are not configured. Set `EMAIL_ADDRESS` / `EMAIL_APP_PASSWORD` "
            "environment variables, or add an `[email]` section with `address` and `app_password` "
            "to `.streamlit/secrets.toml`."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = (
        f"AWS Cloud Practitioner Exam Results — {score['percent']}% "
        f"({'PASS' if score['passed'] else 'FAIL'})"
    )
    msg["From"] = sender
    msg["To"] = RESULTS_RECIPIENT_EMAIL

    msg.attach(MIMEText(build_results_text(score, time_taken, questions, answers), "plain"))
    msg.attach(MIMEText(build_results_html(score, time_taken, questions, answers), "html"))

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.login(sender, app_password)
            server.sendmail(sender, RESULTS_RECIPIENT_EMAIL, msg.as_string())
        return True, f"Results emailed to {RESULTS_RECIPIENT_EMAIL}."
    except Exception as exc:
        return False, f"Failed to send results email: {exc}"


# ─────────────────────────────────────────────────────────────
# UI: PAGE CONFIG + STYLE
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AWS Cloud Practitioner Exam Simulator",
    page_icon="☁️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0f1620; }
    .exam-header {
        background: linear-gradient(90deg, #232F3E 0%, #131a22 100%);
        padding: 18px 24px;
        border-radius: 10px;
        border-left: 5px solid #FF9900;
        margin-bottom: 18px;
    }
    .exam-header h1 { color: #FF9900; margin: 0; font-size: 1.6em; }
    .exam-header p { color: #d5dbe2; margin: 4px 0 0 0; }
    .domain-badge {
        display: inline-block;
        background: #FF9900;
        color: #232F3E;
        font-weight: 700;
        font-size: 0.75em;
        padding: 3px 10px;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #1a2330;
        border: 1px solid #2c3b4f !important;
        border-radius: 10px;
    }
    .question-text { color: #f0f2f5; font-size: 1.15em; font-weight: 600; line-height: 1.5; }
    .stButton > button[kind="primary"] { background: #FF9900; border-color: #FF9900; color: #232F3E; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()

# ─────────────────────────────────────────────────────────────
# UI: START SCREEN
# ─────────────────────────────────────────────────────────────


def render_start():
    st.markdown(
        """
        <div class="exam-header">
            <h1>☁️ AWS Certified Cloud Practitioner (CLF-C02)</h1>
            <p>Full-length Exam Simulator — 65 questions · 90 minutes · multiple choice</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Before you begin")
        st.markdown(
            f"""
            - **65 questions**, each with 4 answer choices — this mirrors the real CLF-C02 exam length.
            - **90-minute countdown timer** shown on screen. The exam auto-submits the moment time runs out.
            - Use **Next / Previous** to move between questions, or jump directly using the question grid in the sidebar once the exam starts.
            - **Mark for Review** flags a question so you can revisit it before submitting.
            - You can submit early at any time once you're confident in your answers.
            - Questions and answer order are shuffled on every attempt.
            - **Your score is not shown on screen.** When you submit, results are scored and emailed to **{RESULTS_RECIPIENT_EMAIL}**.
            """
        )

        sender, app_password = get_email_credentials()
        with st.expander("📧 Email setup (required before your first exam)"):
            if sender and app_password:
                st.success(f"Email is configured to send from {sender}.")
            else:
                st.warning("Email credentials are not configured yet — results can't be sent until this is set up.")
            st.markdown(
                f"""
                Results are sent via Gmail SMTP from **philjones820@gmail.com**. To enable it, generate a
                [Gmail App Password](https://myaccount.google.com/apppasswords) for that account, then provide it
                to the app one of two ways:

                **Option A — Streamlit secrets** (create `.streamlit/secrets.toml`, keep it out of git):
                ```toml
                [email]
                address = "philjones820@gmail.com"
                app_password = "xxxx xxxx xxxx xxxx"
                ```

                **Option B — environment variables** before launching the app:
                ```bash
                export EMAIL_ADDRESS="philjones820@gmail.com"
                export EMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
                ```
                """
            )
        st.subheader("Exam domain breakdown")
        for domain, pct in DOMAIN_WEIGHTS.items():
            st.markdown(f"**{domain}** — {pct}%")
            st.progress(pct / 100)

    with col2:
        st.metric("Questions", "65")
        st.metric("Time Limit", "90 min")
        st.metric("Passing Score (practice)", f"{PASS_THRESHOLD_PERCENT}%")
        st.caption(
            "AWS's official passing score is a scaled 700 out of 1000. "
            "This simulator uses raw percentage as an approximate guide."
        )
        st.markdown("---")
        if st.button("🚀 Start Exam", type="primary", use_container_width=True):
            start_exam()
            st.rerun()


# ─────────────────────────────────────────────────────────────
# UI: EXAM IN PROGRESS
# ─────────────────────────────────────────────────────────────


def render_timer():
    remaining = int(remaining_seconds())
    st.components.v1.html(
        f"""
        <div style="font-family: -apple-system, sans-serif; text-align:center;
                    background:#1a2330; border:1px solid #2c3b4f; border-radius:10px; padding:10px;">
            <div style="color:#9aa7b5; font-size:0.75em; text-transform:uppercase;
                        letter-spacing:2px; margin-bottom:4px;">Time Remaining</div>
            <div id="timer" style="color:#4ade80; font-size:2.1em; font-weight:800;
                        font-variant-numeric: tabular-nums;">--:--</div>
        </div>
        <script>
            let remaining = {remaining};
            const el = document.getElementById('timer');
            function render() {{
                const m = Math.floor(remaining / 60);
                const s = remaining % 60;
                el.innerText = String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
                el.style.color = remaining <= 300 ? '#f87171' : (remaining <= 900 ? '#fbbf24' : '#4ade80');
            }}
            render();
            const interval = setInterval(function() {{
                remaining -= 1;
                if (remaining <= 0) {{
                    remaining = 0;
                    render();
                    clearInterval(interval);
                    setTimeout(function() {{ window.parent.location.reload(); }}, 400);
                }} else {{
                    render();
                }}
            }}, 1000);
        </script>
        """,
        height=90,
    )


def render_navigator():
    questions = st.session_state.exam_questions
    st.sidebar.markdown("### Question Navigator")
    answered = len(st.session_state.answers)
    marked = len(st.session_state.marked)
    st.sidebar.caption(f"Answered: {answered}/65 · Marked for review: {marked}")
    st.sidebar.caption(f"Time remaining (server): {format_mmss(remaining_seconds())}")

    cols_per_row = 5
    for row_start in range(0, len(questions), cols_per_row):
        cols = st.sidebar.columns(cols_per_row)
        for offset, col in enumerate(cols):
            idx = row_start + offset
            if idx >= len(questions):
                continue
            is_current = idx == st.session_state.current_q
            is_answered = idx in st.session_state.answers
            is_marked = idx in st.session_state.marked
            if is_marked:
                label = f"🚩{idx + 1}"
            elif is_answered:
                label = f"✅{idx + 1}"
            else:
                label = f"{idx + 1}"
            with col:
                if st.button(
                    label,
                    key=f"nav_{idx}",
                    type="primary" if is_current else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.current_q = idx
                    st.session_state.confirm_submit = False
                    st.rerun()

    st.sidebar.markdown("---")
    if not st.session_state.confirm_submit:
        if st.sidebar.button("🏁 Submit Exam", use_container_width=True):
            st.session_state.confirm_submit = True
            st.rerun()
    else:
        unanswered = 65 - answered
        if unanswered:
            st.sidebar.warning(f"{unanswered} question(s) unanswered.")
        st.sidebar.error("Submit final answers? This cannot be undone.")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("Yes, submit", type="primary", use_container_width=True):
                submit_exam()
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_submit = False
                st.rerun()


def render_question():
    idx = st.session_state.current_q
    q = st.session_state.exam_questions[idx]

    st.markdown(f'<span class="domain-badge">{q["domain"]}</span>', unsafe_allow_html=True)
    st.progress((idx + 1) / 65, text=f"Question {idx + 1} of 65")

    with st.container(border=True):
        st.markdown(f'<div class="question-text">{q["question"]}</div>', unsafe_allow_html=True)
        st.write("")

        current_answer = st.session_state.answers.get(idx)
        selected = st.radio(
            "Choose an answer:",
            options=list(range(len(q["options"]))),
            format_func=lambda i: q["options"][i],
            index=current_answer,
            key=f"radio_{idx}",
            label_visibility="collapsed",
        )
        if selected is not None and st.session_state.answers.get(idx) != selected:
            st.session_state.answers[idx] = selected

        marked = idx in st.session_state.marked
        new_marked = st.checkbox("🚩 Mark for review", value=marked, key=f"mark_{idx}")
        if new_marked and not marked:
            st.session_state.marked.add(idx)
        elif not new_marked and marked:
            st.session_state.marked.discard(idx)

    nav1, nav2, nav3 = st.columns([1, 1, 3])
    with nav1:
        if st.button("⬅ Previous", disabled=(idx == 0), use_container_width=True):
            st.session_state.current_q -= 1
            st.rerun()
    with nav2:
        if st.button("Next ➡", disabled=(idx == 64), use_container_width=True):
            st.session_state.current_q += 1
            st.rerun()


def render_exam():
    if remaining_seconds() <= 0:
        submit_exam()
        st.rerun()
        return

    st.markdown(
        """
        <div class="exam-header">
            <h1>☁️ AWS Cloud Practitioner — Exam In Progress</h1>
            <p>Answer all questions, then submit from the sidebar when you're ready.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_timer()
    render_question()
    render_navigator()


# ─────────────────────────────────────────────────────────────
# UI: RESULTS
# ─────────────────────────────────────────────────────────────


def render_results():
    questions = st.session_state.exam_questions
    answers = st.session_state.answers
    time_taken = min(elapsed_seconds(), EXAM_DURATION_SECONDS)
    score = score_exam(questions, answers)

    if st.session_state.email_status is None:
        with st.spinner("Scoring your exam and emailing your results..."):
            ok, message = send_results_email(score, time_taken, questions, answers)
        st.session_state.email_status = {"ok": ok, "message": message}

    status = st.session_state.email_status

    st.markdown(
        """
        <div class="exam-header">
            <h1>🏁 Exam Submitted</h1>
            <p>Your answers have been scored. Results are not shown on screen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if status["ok"]:
        st.success(f"✅ {status['message']}")
        st.caption(f"Time used: {format_mmss(time_taken)} · Questions answered: {len(answers)}/{len(questions)}")
    else:
        st.error(f"⚠️ {status['message']}")
        st.info("Fix the email configuration above, then click below to try sending again.")
        if st.button("📧 Retry sending email"):
            st.session_state.email_status = None
            st.rerun()
        with st.expander("Trouble with email delivery? View results on screen instead"):
            if st.button("Show my results now"):
                st.session_state.reveal_results = True
                st.rerun()
            if st.session_state.reveal_results:
                st.markdown(
                    f"**Score:** {score['correct']}/{score['total']} ({score['percent']}%) "
                    f"— {'PASS' if score['passed'] else 'FAIL'}"
                )
                for domain, stats in score["domain_stats"].items():
                    pct = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0
                    st.markdown(f"- **{domain}** — {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    st.markdown("---")
    if st.button("🔄 Retake Exam", type="primary"):
        retake_exam()
        st.rerun()


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────

if st.session_state.stage == "start":
    render_start()
elif st.session_state.stage == "exam":
    render_exam()
else:
    render_results()
