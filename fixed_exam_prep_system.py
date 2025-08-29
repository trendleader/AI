#!/usr/bin/env python3
"""
AWS ML Certification Exam Prep System - Windows Compatible Version
================================================================

Fixed version that handles Windows encoding issues and provides
a robust AWS ML exam preparation system.

Setup Instructions:
1. Ensure your test.env file contains: OPENAI_API_KEY=your_key_here
2. Run: pip install openai python-dotenv sentence-transformers streamlit
3. Run this script: python fixed_exam_prep_system.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# Fix Windows console encoding issues
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Check if we're in the correct directory
PROJECT_DIR = Path(r"C:\Users\pjone\OneDrive\Desktop\NewPython")
if not PROJECT_DIR.exists():
    PROJECT_DIR = Path.cwd()

# Configure logging without emoji characters for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_DIR / 'exam_prep.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = {
        'openai': 'openai',
        'dotenv': 'python-dotenv',
        'sentence_transformers': 'sentence-transformers',
        'streamlit': 'streamlit',
        'numpy': 'numpy'
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True

if not check_dependencies():
    sys.exit(1)

# Import after dependency check
try:
    import openai
    from dotenv import load_dotenv
    import streamlit as st
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class WindowsCompatibleExamPrep:
    """
    Windows-compatible AWS ML exam prep system with proper encoding handling
    """
    
    def __init__(self):
        self.project_dir = PROJECT_DIR
        self.setup_complete = False
        self.openai_available = False
        self.embedding_model = None
        self.study_chunks = []
        
        # Status indicators (Windows-compatible)
        self.status = {
            'success': '[SUCCESS]',
            'error': '[ERROR]', 
            'warning': '[WARNING]',
            'info': '[INFO]'
        }
        
        # Prompt templates
        self.prompts = {
            "comprehensive_answer": """
You are an expert AWS Machine Learning Engineer and certified instructor preparing students for the MLA-C01 exam.

CONTEXT FROM STUDY GUIDE:
{context}

STUDENT QUESTION: {question}

Please provide a comprehensive exam-focused answer that includes:

DIRECT ANSWER: Clear, concise response to the question
KEY CONCEPTS: Essential points students must understand
AWS SERVICES: Specific AWS services and their roles
EXAM TIPS: What to remember for the certification exam
COMMON MISTAKES: Typical misconceptions to avoid
RELATED TOPICS: Connected concepts to review

Structure your response for maximum learning impact and exam success.

ANSWER:
""",
            
            "question_generator": """
You are creating AWS ML certification exam questions. Based on the following content, generate {num_questions} realistic exam questions.

CONTENT: {content}

REQUIREMENTS:
- Multiple choice with 4 options (A, B, C, D)
- One clearly correct answer
- Realistic distractors that test understanding
- Explanations for all options
- Appropriate difficulty: {difficulty}

FORMAT AS JSON:
{{
  "questions": [
    {{
      "id": 1,
      "question": "Which AWS service is best for real-time streaming data ingestion for ML models?",
      "options": [
        "A) Amazon S3",
        "B) Amazon Kinesis Data Streams", 
        "C) Amazon RDS",
        "D) AWS Glue"
      ],
      "correct": "B",
      "explanation": "Amazon Kinesis Data Streams is designed for real-time data streaming and can feed ML models with continuous data.",
      "domain": "Data Engineering",
      "key_points": ["Real-time streaming", "Data ingestion", "ML model feeding"]
    }}
  ]
}}
""",
            
            "study_plan": """
Based on the student's performance data, create a personalized study plan.

PERFORMANCE DATA:
{performance_data}

WEAK AREAS IDENTIFIED:
{weak_areas}

Create a detailed study plan with:
1. Priority ranking of topics to study
2. Estimated hours needed per topic
3. Specific AWS services to focus on
4. Recommended study sequence
5. Practice question targets

STUDY PLAN:
"""
        }
        
    def setup_environment(self):
        """Setup environment with proper error handling"""
        try:
            # Load environment variables
            env_file = self.project_dir / "test.env"
            if env_file.exists():
                load_dotenv(env_file)
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key and len(api_key) > 20:  # Basic validation
                    openai.api_key = api_key
                    self.openai_available = True
                    logger.info(f"{self.status['success']} OpenAI API key loaded")
                else:
                    logger.warning(f"{self.status['warning']} OpenAI API key not found or invalid")
            else:
                logger.warning(f"{self.status['warning']} test.env file not found")
            
            # Initialize embedding model
            try:
                logger.info(f"{self.status['info']} Loading embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info(f"{self.status['success']} Embedding model loaded")
            except Exception as e:
                logger.error(f"{self.status['error']} Embedding model failed: {e}")
                return False
            
            # Load study chunks
            self.load_study_chunks()
            
            self.setup_complete = True
            logger.info(f"{self.status['success']} System ready")
            return True
            
        except Exception as e:
            logger.error(f"{self.status['error']} Setup failed: {e}")
            return False
    
    def load_study_chunks(self):
        """Load or create study content chunks"""
        chunks_file = self.project_dir / "aws_ml_study_guide_chunks.json"
        
        if chunks_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    self.study_chunks = json.load(f)
                logger.info(f"{self.status['success']} Loaded {len(self.study_chunks)} chunks")
                return
            except Exception as e:
                logger.warning(f"{self.status['warning']} Error loading chunks: {e}")
        
        # Create basic chunks
        logger.info(f"{self.status['info']} Creating study chunks...")
        self.study_chunks = self.create_study_content()
        
        # Save chunks
        try:
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(self.study_chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"{self.status['success']} Saved {len(self.study_chunks)} chunks")
        except Exception as e:
            logger.warning(f"{self.status['warning']} Could not save chunks: {e}")
    
    def create_study_content(self):
        """Create comprehensive study content for AWS ML certification"""
        return [
            {
                "id": 0,
                "title": "Amazon S3 for Machine Learning",
                "domain": "Domain 1: Data Engineering",
                "content": """Amazon S3 serves as the foundational storage service for ML workflows on AWS. It provides unlimited scalable storage for datasets, model artifacts, and results. Key ML features include:

Storage Classes: Choose from Standard, Intelligent-Tiering, Standard-IA, One Zone-IA, Glacier, and Glacier Deep Archive based on access patterns and cost requirements.

Versioning: Maintain multiple versions of datasets and models for reproducibility and rollback capabilities.

Lifecycle Policies: Automatically transition data between storage classes to optimize costs as data ages.

Integration: Seamless integration with SageMaker, EMR, Glue, and other ML services for data access and processing.

Security: Encryption at rest and in transit, IAM policies, bucket policies, and Access Control Lists (ACLs) for comprehensive security.

Performance: Transfer Acceleration for faster uploads, Multi-Part Upload for large files, and S3 Select for querying data without downloading entire objects.""",
                "key_points": [
                    "Unlimited scalable object storage",
                    "Multiple storage classes for cost optimization", 
                    "Versioning for data management",
                    "Native integration with AWS ML services",
                    "Comprehensive security features"
                ],
                "aws_services": ["Amazon S3", "S3 Transfer Acceleration", "S3 Select"],
                "exam_focus": "Understanding when to use different storage classes and S3 integration patterns"
            },
            
            {
                "id": 1,
                "title": "Amazon SageMaker Complete Platform",
                "domain": "Domain 3: Modeling", 
                "content": """Amazon SageMaker is AWS's comprehensive machine learning platform providing tools for the entire ML lifecycle:

Data Preparation:
- SageMaker Data Wrangler: Visual data preparation with 300+ built-in transformations
- SageMaker Processing: Run preprocessing and postprocessing workloads
- SageMaker Feature Store: Centralized repository for ML features

Model Development:
- SageMaker Studio: Integrated development environment for ML
- Built-in Algorithms: Pre-built algorithms for common ML tasks
- Custom Containers: Bring your own algorithms using Docker containers
- SageMaker Autopilot: Automated ML for model building

Training and Tuning:
- Distributed Training: Scale training across multiple instances
- Spot Training: Use Spot instances for cost savings up to 90%
- Automatic Model Tuning: Hyperparameter optimization
- SageMaker Experiments: Track and organize ML experiments

Deployment and Inference:
- Real-time Endpoints: Low-latency predictions
- Batch Transform: Process large datasets
- Multi-Model Endpoints: Host multiple models on single endpoint
- Serverless Inference: Automatically scale inference capacity

Monitoring and Management:
- SageMaker Model Monitor: Detect data drift and model performance degradation
- SageMaker Clarify: Bias detection and model explainability
- SageMaker Model Registry: Centralized model catalog""",
                "key_points": [
                    "End-to-end ML platform",
                    "Built-in algorithms and custom containers",
                    "Distributed training capabilities",
                    "Multiple deployment options",
                    "Automated monitoring and management"
                ],
                "aws_services": [
                    "SageMaker Studio", "SageMaker Data Wrangler", "SageMaker Processing",
                    "SageMaker Training", "SageMaker Endpoints", "SageMaker Model Monitor"
                ],
                "exam_focus": "Understanding SageMaker components and when to use each service"
            },
            
            {
                "id": 2,
                "title": "Data Preprocessing and Feature Engineering",
                "domain": "Domain 2: Exploratory Data Analysis",
                "content": """Data preprocessing and feature engineering are critical for ML success. AWS provides multiple tools and techniques:

Missing Data Handling:
- Simple Imputation: Mean, median, mode imputation
- Advanced Imputation: K-NN imputation, iterative imputation
- Deletion: Listwise or pairwise deletion when appropriate
- Indicator Variables: Flag missing values with binary indicators

Outlier Detection and Treatment:
- Statistical Methods: Z-score, IQR method, modified Z-score
- Visualization: Box plots, scatter plots, histograms
- Treatment: Removal, capping, transformation, or separate modeling

Feature Scaling and Normalization:
- Min-Max Scaling: Scale to [0,1] range
- Z-score Standardization: Mean 0, standard deviation 1
- Robust Scaling: Use median and IQR, less sensitive to outliers
- Unit Vector Scaling: Scale to unit norm

Categorical Encoding:
- One-Hot Encoding: Create binary columns for each category
- Label Encoding: Convert categories to integers
- Target Encoding: Use target variable statistics
- Binary Encoding: Reduce dimensionality compared to one-hot

Feature Engineering Techniques:
- Polynomial Features: Create interaction terms and polynomial combinations
- Binning: Convert continuous variables to categorical
- Text Features: TF-IDF, word embeddings, n-grams
- Time-based Features: Extract day, month, season from dates

AWS Tools:
- SageMaker Data Wrangler: Visual data preparation interface
- AWS Glue DataBrew: Visual data preparation service
- SageMaker Processing: Custom preprocessing jobs
- Built-in preprocessing in SageMaker algorithms""",
                "key_points": [
                    "Multiple imputation strategies available",
                    "Outlier detection requires domain knowledge",
                    "Feature scaling is algorithm-dependent",
                    "Categorical encoding affects model performance",
                    "AWS provides visual and programmatic tools"
                ],
                "aws_services": [
                    "SageMaker Data Wrangler", "AWS Glue DataBrew", 
                    "SageMaker Processing", "Amazon Athena"
                ],
                "exam_focus": "Choosing appropriate preprocessing techniques for different data types and ML algorithms"
            },
            
            {
                "id": 3,
                "title": "Model Training Strategies and Optimization",
                "domain": "Domain 3: Modeling",
                "content": """Effective model training requires understanding various strategies and optimization techniques:

Data Splitting Strategies:
- Train/Validation/Test: Typical 70/15/15 or 80/10/10 splits
- Cross-Validation: K-fold (k=5 or 10), stratified for imbalanced data
- Time Series: Forward chaining, expanding window, sliding window
- Stratified Sampling: Maintain class distribution in splits

Training Approaches:
- Batch Training: Process entire dataset at once
- Mini-batch Training: Process small batches iteratively
- Online Learning: Update model with each new example
- Distributed Training: Parallel training across multiple instances

Hyperparameter Optimization:
- Grid Search: Exhaustive search over parameter grid
- Random Search: Sample parameters randomly
- Bayesian Optimization: Use probability models to guide search
- Early Stopping: Prevent overfitting by monitoring validation performance

SageMaker Training Features:
- Automatic Model Tuning: Managed hyperparameter optimization
- Distributed Training: Data parallelism and model parallelism
- Spot Training: Use Spot instances for cost reduction
- Managed Spot Training: Automatically handle interruptions
- Checkpointing: Save training state for resumption

Optimization Techniques:
- Learning Rate Scheduling: Adaptive learning rates
- Regularization: L1, L2, dropout, early stopping
- Data Augmentation: Increase training data diversity
- Ensemble Methods: Combine multiple models for better performance

Performance Monitoring:
- Training Metrics: Loss, accuracy, precision, recall
- Validation Curves: Monitor for overfitting/underfitting
- Learning Curves: Assess if more data would help
- Resource Utilization: CPU, GPU, memory usage""",
                "key_points": [
                    "Proper data splitting prevents data leakage",
                    "Distributed training scales with data size",
                    "Hyperparameter tuning significantly impacts performance",
                    "Spot instances can reduce training costs by 90%",
                    "Monitoring prevents overfitting and resource waste"
                ],
                "aws_services": [
                    "SageMaker Training Jobs", "SageMaker Automatic Model Tuning",
                    "SageMaker Managed Spot Training", "Amazon EC2 Spot Instances"
                ],
                "exam_focus": "Understanding when to use different training strategies and optimization techniques"
            },
            
            {
                "id": 4,
                "title": "Model Deployment and Inference Patterns",
                "domain": "Domain 4: ML Implementation and Operations",
                "content": """Model deployment requires choosing appropriate inference patterns based on requirements:

Real-time Inference:
- SageMaker Endpoints: Hosted models with auto-scaling
- API Gateway + Lambda: Serverless inference for simple models
- EC2 Instances: Custom deployment with full control
- Container Services: ECS, EKS for containerized models

Batch Inference:
- SageMaker Batch Transform: Process large datasets offline
- AWS Batch: Managed batch processing for custom workloads
- EMR: Big data processing with Spark MLlib
- Lambda: Event-driven batch processing for small datasets

Serverless Inference:
- SageMaker Serverless Inference: Pay per invocation, auto-scaling
- AWS Lambda: For lightweight models under 500MB
- API Gateway: RESTful APIs for model access

Edge Deployment:
- SageMaker Edge Manager: Deploy models to edge devices
- AWS IoT Greengrass: Edge computing platform
- Amazon Lookout for Equipment: Anomaly detection on industrial equipment

Multi-Model Hosting:
- Multi-Model Endpoints: Host multiple models on single endpoint
- Model Variants: A/B test different model versions
- Auto Scaling: Automatically scale based on traffic

Deployment Strategies:
- Blue/Green Deployment: Switch traffic between environments
- Canary Deployment: Gradually shift traffic to new version
- Rolling Deployment: Update instances one by one
- Shadow Deployment: Test new version with production traffic

Performance Considerations:
- Cold Start Latency: Time to initialize model
- Throughput: Requests per second capability
- Concurrency: Simultaneous request handling
- Model Size: Affects memory and loading time""",
                "key_points": [
                    "Choose inference pattern based on latency and throughput requirements",
                    "Serverless options reduce operational overhead",
                    "Multi-model endpoints optimize resource utilization",
                    "Deployment strategies enable safe model updates",
                    "Edge deployment reduces latency for IoT use cases"
                ],
                "aws_services": [
                    "SageMaker Endpoints", "SageMaker Batch Transform", "AWS Lambda",
                    "API Gateway", "SageMaker Edge Manager", "AWS IoT Greengrass"
                ],
                "exam_focus": "Matching deployment patterns to use case requirements and understanding trade-offs"
            },
            
            {
                "id": 5,
                "title": "ML Security and Compliance",
                "domain": "Domain 4: ML Implementation and Operations",
                "content": """Security in ML involves protecting data, models, and infrastructure throughout the ML lifecycle:

Identity and Access Management:
- IAM Roles: Grant permissions to AWS services and users
- IAM Policies: Define fine-grained permissions
- Service-Linked Roles: Predefined roles for AWS services
- Cross-Account Access: Share resources across AWS accounts
- Temporary Credentials: Use STS for secure access

Data Protection:
- Encryption at Rest: S3, EBS, RDS encryption
- Encryption in Transit: HTTPS, TLS for data movement
- AWS KMS: Centralized key management
- Customer Managed Keys: Full control over encryption keys
- S3 Bucket Policies: Control access to training data

Network Security:
- VPC: Isolated network environment
- Security Groups: Instance-level firewalls
- NACLs: Subnet-level network controls
- Private Subnets: No direct internet access
- VPC Endpoints: Private connectivity to AWS services
- NAT Gateways: Secure outbound internet access

Model Security:
- Model Artifacts Encryption: Encrypt stored models
- Endpoint Security: HTTPS for model inference
- Model Access Logging: CloudTrail for API calls
- Resource-Based Policies: Control endpoint access

Data Governance:
- Data Classification: Identify sensitive data types
- Amazon Macie: Discover and classify sensitive data
- AWS Config: Monitor configuration compliance
- AWS CloudTrail: Audit API calls and user activity
- Data Loss Prevention: Prevent unauthorized data access

Compliance Frameworks:
- GDPR: General Data Protection Regulation
- HIPAA: Health Insurance Portability and Accountability Act
- SOC: Service Organization Control
- ISO 27001: Information security management
- PCI DSS: Payment Card Industry Data Security Standard

Privacy Techniques:
- Data Anonymization: Remove personally identifiable information
- Differential Privacy: Add noise to protect individual privacy
- Federated Learning: Train models without centralizing data
- Homomorphic Encryption: Compute on encrypted data""",
                "key_points": [
                    "IAM provides fine-grained access control",
                    "Encryption is required for data at rest and in transit",
                    "VPC provides network isolation for ML workloads",
                    "Compliance requirements vary by industry and region",
                    "Privacy-preserving techniques enable secure ML"
                ],
                "aws_services": [
                    "AWS IAM", "AWS KMS", "Amazon VPC", "AWS CloudTrail",
                    "Amazon Macie", "AWS Config", "AWS PrivateLink"
                ],
                "exam_focus": "Understanding security requirements and implementing appropriate controls for ML workloads"
            },
            
            {
                "id": 6,
                "title": "ML Monitoring and Operations",
                "domain": "Domain 4: ML Implementation and Operations",
                "content": """Operational excellence in ML requires comprehensive monitoring and management:

Model Performance Monitoring:
- Accuracy Metrics: Track prediction accuracy over time
- Business Metrics: Monitor impact on business KPIs
- A/B Testing: Compare model variants in production
- Champion-Challenger: Continuously evaluate new models
- Performance Degradation: Detect when models need retraining

Data Drift Detection:
- Statistical Tests: KS test, PSI (Population Stability Index)
- Distribution Monitoring: Compare training vs. production data
- Feature Drift: Monitor individual feature distributions
- Concept Drift: Changes in relationship between features and target
- SageMaker Model Monitor: Automated drift detection

Infrastructure Monitoring:
- CloudWatch Metrics: CPU, memory, network, custom metrics
- CloudWatch Alarms: Automated alerting on threshold breaches
- CloudWatch Dashboards: Visual monitoring interfaces
- VPC Flow Logs: Network traffic analysis
- AWS X-Ray: Distributed tracing for debugging

Logging and Auditing:
- CloudTrail: API call logging and governance
- CloudWatch Logs: Application and system logs
- VPC Flow Logs: Network traffic logs
- SageMaker Logs: Training and inference logs
- Log Analysis: Use CloudWatch Insights for log queries

Automated Operations:
- Auto Scaling: Automatically adjust capacity
- Lambda Functions: Event-driven automation
- Step Functions: Orchestrate complex workflows
- EventBridge: Event-driven architectures
- Systems Manager: Configuration management

Model Lifecycle Management:
- Model Registry: Centralized model catalog
- Model Versioning: Track model versions and metadata
- Model Approval: Workflow for model promotion
- Model Deployment: Automated deployment pipelines
- Model Retirement: Safely remove old models

Cost Optimization:
- Spot Instances: Reduce training costs by up to 90%
- Reserved Instances: Lower costs for predictable workloads
- Right-sizing: Match instance types to workload requirements
- Storage Optimization: Use appropriate S3 storage classes
- Monitoring Costs: CloudWatch billing alarms""",
                "key_points": [
                    "Model performance degrades over time without monitoring",
                    "Data drift is a common cause of model degradation",
                    "Infrastructure monitoring prevents service disruptions",
                    "Automated operations reduce manual effort and errors",
                    "Cost optimization requires ongoing monitoring and adjustment"
                ],
                "aws_services": [
                    "SageMaker Model Monitor", "Amazon CloudWatch", "AWS CloudTrail",
                    "AWS Lambda", "AWS Step Functions", "AWS Systems Manager"
                ],
                "exam_focus": "Implementing comprehensive monitoring and operational practices for production ML systems"
            }
        ]
    
    def search_relevant_content(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant study content using keyword matching"""
        if not self.study_chunks:
            return []
        
        query_terms = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.study_chunks:
            # Create searchable text
            searchable_content = (
                chunk.get('title', '') + ' ' +
                chunk.get('content', '') + ' ' +
                ' '.join(chunk.get('key_points', [])) + ' ' +
                ' '.join(chunk.get('aws_services', []))
            ).lower()
            
            # Calculate relevance score
            content_terms = set(searchable_content.split())
            intersection = query_terms.intersection(content_terms)
            base_score = len(intersection) / len(query_terms) if query_terms else 0
            
            # Boost for exact matches in title or AWS services
            title_boost = 0.3 if any(term in chunk.get('title', '').lower() for term in query_terms) else 0
            service_boost = 0.2 if any(term in ' '.join(chunk.get('aws_services', [])).lower() for term in query_terms) else 0
            
            final_score = base_score + title_boost + service_boost
            scored_chunks.append((final_score, chunk))
        
        # Sort by score and return top results
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k] if score > 0.1]
    
    def get_openai_response(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 800) -> str:
        """Get response from OpenAI API with error handling"""
        if not self.openai_available:
            return "OpenAI API not available. Please ensure your API key is set in test.env file."
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert AWS ML certification instructor providing clear, exam-focused guidance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"{self.status['error']} OpenAI API error: {str(e)}")
            return f"I encountered an error while generating the response. Please try again or check your API configuration."
    
    def get_comprehensive_answer(self, question: str) -> Dict[str, Any]:
        """Generate comprehensive exam-focused answer"""
        
        # Find relevant content
        relevant_chunks = self.search_relevant_content(question, top_k=3)
        
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I couldn't find specific information about your question in the study materials. Please try asking about AWS ML services like SageMaker, S3, or specific ML concepts like model training or deployment.",
                "sources": [],
                "confidence": "low"
            }
        
        # Prepare context
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"**{chunk['title']}** ({chunk['domain']})\n{chunk['content'][:1000]}...")
        
        context = "\n\n".join(context_parts)
        
        # Generate response using advanced prompting
        prompt = self.prompts["comprehensive_answer"].format(
            context=context,
            question=question
        )
        
        answer = self.get_openai_response(prompt, model="gpt-4" if self.openai_available else "gpt-3.5-turbo")
        
        return {
            "question": question,
            "answer": answer,
            "sources": [chunk['title'] for chunk in relevant_chunks],
            "context_chunks": len(relevant_chunks),
            "confidence": "high" if len(relevant_chunks) >= 2 else "medium"
        }
    
    def generate_practice_questions(self, topic: str, difficulty: str = "intermediate", num_questions: int = 3) -> List[Dict]:
        """Generate practice questions for specific topics"""
        
        # Find relevant content
        relevant_chunks = self.search_relevant_content(topic, top_k=2)
        
        if not relevant_chunks:
            return [{
                "id": 1,
                "question": f"Content not available for topic: {topic}. Please try topics like 'SageMaker', 'S3', 'Data Processing', 'Model Training', or 'Security'.",
                "options": [
                    "A) Try a more specific topic",
                    "B) Check your spelling", 
                    "C) Use AWS service names",
                    "D) All of the above"
                ],
                "correct": "D",
                "explanation": "The system contains information about specific AWS ML services and concepts. Try using more specific terms."
            }]
        
        # Prepare content for question generation
        content = "\n\n".join([chunk['content'][:800] for chunk in relevant_chunks])
        
        # Generate questions
        prompt = self.prompts["question_generator"].format(
            content=content,
            num_questions=num_questions,
            difficulty=difficulty
        )
        
        response = self.get_openai_response(prompt, model="gpt-4" if self.openai_available else "gpt-3.5-turbo", max_tokens=1500)
        
        # Parse JSON response
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                questions_data = json.loads(json_match.group())
                return questions_data.get("questions", [])
        except Exception as e:
            logger.warning(f"{self.status['warning']} Could not parse questions: {e}")
        
        # Fallback question if parsing fails
        return [{
            "id": 1,
            "question": f"Which AWS service would you primarily use for {topic.lower()} in machine learning workflows?",
            "options": [
                "A) Amazon S3 for data storage",
                "B) Amazon SageMaker for ML operations",
                "C) Amazon EC2 for compute resources", 
                "D) All of the above depending on the specific use case"
            ],
            "correct": "D",
            "explanation": f"For {topic}, multiple AWS services may be involved depending on the specific requirements, with SageMaker being the primary ML platform."
        }]
    
    def run_cli_session(self):
        """Run command-line interface session"""
        if not self.setup_complete:
            print(f"{self.status['error']} System not ready. Please check the logs.")
            return
        
        print("\n" + "="*60)
        print("AWS ML CERTIFICATION EXAM PREP - COMMAND LINE INTERFACE")
        print("="*60)
        print("Commands:")
        print("  ask [question]     - Get comprehensive answers")
        print("  practice [topic]   - Generate practice questions")
        