#!/usr/bin/env python3
"""
AWS ML Certification Exam Prep System - Windows Robust Version
============================================================

Fixed version that handles Windows console encoding issues more robustly
and is compatible with OpenAI API v1.0+.

Setup Instructions:
1. Ensure your test.env file contains: OPENAI_API_KEY=your_key_here
2. Run: pip install openai>=1.0.0 python-dotenv sentence-transformers streamlit
3. Run this script: python windows_robust_exam_prep.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# Safe Windows console encoding fix
def setup_windows_console():
    """Safely setup Windows console for UTF-8 if possible"""
    if os.name == 'nt':  # Windows
        try:
            # Try to set console to UTF-8 mode
            os.system('chcp 65001 > nul 2>&1')
        except:
            pass  # Ignore if it fails
        
        try:
            # Try to reconfigure stdout/stderr encoding
            import io
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass  # Ignore if it fails

# Apply Windows console fix safely
setup_windows_console()

# Check directory
PROJECT_DIR = Path(r"C:\Users\pjone\OneDrive\Desktop\NewPython")
if not PROJECT_DIR.exists():
    PROJECT_DIR = Path.cwd()

# Configure logging with safe encoding
def setup_logging():
    """Setup logging with safe encoding for Windows"""
    log_handlers = []
    
    # File handler with explicit UTF-8 encoding
    try:
        file_handler = logging.FileHandler(
            PROJECT_DIR / 'exam_prep.log', 
            encoding='utf-8',
            mode='a'
        )
        log_handlers.append(file_handler)
    except:
        pass  # Skip file logging if it fails
    
    # Stream handler with error handling
    try:
        stream_handler = logging.StreamHandler()
        log_handlers.append(stream_handler)
    except:
        pass
    
    if log_handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=log_handlers
        )
    
    return logging.getLogger(__name__)

logger = setup_logging()

def check_dependencies():
    """Check required dependencies"""
    required_packages = {
        'openai': 'openai>=1.0.0',
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
    input("Press Enter to exit...")
    sys.exit(1)

# Import after dependency check
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    import streamlit as st
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run: pip install openai>=1.0.0 python-dotenv sentence-transformers streamlit")
    input("Press Enter to exit...")
    sys.exit(1)

class WindowsRobustExamPrep:
    """
    AWS ML exam prep system with robust Windows support
    """
    
    def __init__(self):
        self.project_dir = PROJECT_DIR
        self.setup_complete = False
        self.openai_client = None
        self.embedding_model = None
        self.study_chunks = []
        
        # Simple status indicators (no special characters)
        self.status = {
            'success': '[OK]',
            'error': '[ERROR]', 
            'warning': '[WARN]',
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
        
    def safe_print(self, message):
        """Print message with safe encoding handling"""
        try:
            print(message)
        except UnicodeEncodeError:
            # Fallback to ASCII if UTF-8 fails
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            print(safe_message)
    
    def safe_log(self, level, message):
        """Log message with safe encoding handling"""
        try:
            if level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            elif level == 'error':
                logger.error(message)
        except:
            # Fallback to simple print if logging fails
            try:
                print(f"{level.upper()}: {message}")
            except:
                print(f"{level.upper()}: [Message encoding error]")
    
    def setup_environment(self):
        """Setup environment with robust error handling"""
        try:
            # Load environment variables
            env_file = self.project_dir / "test.env"
            if env_file.exists():
                load_dotenv(env_file)
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key and len(api_key) > 20:
                    self.openai_client = OpenAI(api_key=api_key)
                    self.safe_log('info', f"{self.status['success']} OpenAI client initialized")
                else:
                    self.safe_log('warning', f"{self.status['warning']} OpenAI API key not found or invalid")
            else:
                self.safe_log('warning', f"{self.status['warning']} test.env file not found")
            
            # Initialize embedding model
            try:
                self.safe_log('info', f"{self.status['info']} Loading embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.safe_log('info', f"{self.status['success']} Embedding model loaded")
            except Exception as e:
                self.safe_log('error', f"{self.status['error']} Embedding model failed: {e}")
                return False
            
            # Load study chunks
            self.load_study_chunks()
            
            self.setup_complete = True
            self.safe_log('info', f"{self.status['success']} System ready")
            return True
            
        except Exception as e:
            self.safe_log('error', f"{self.status['error']} Setup failed: {e}")
            return False
    
    def load_study_chunks(self):
        """Load or create study content"""
        chunks_file = self.project_dir / "aws_ml_study_guide_chunks.json"
        
        if chunks_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    self.study_chunks = json.load(f)
                self.safe_log('info', f"{self.status['success']} Loaded {len(self.study_chunks)} chunks")
                return
            except Exception as e:
                self.safe_log('warning', f"{self.status['warning']} Error loading chunks: {e}")
        
        # Create study chunks
        self.safe_log('info', f"{self.status['info']} Creating study chunks...")
        self.study_chunks = self.create_comprehensive_study_content()
        
        # Save chunks
        try:
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(self.study_chunks, f, indent=2, ensure_ascii=False)
            self.safe_log('info', f"{self.status['success']} Saved {len(self.study_chunks)} chunks")
        except Exception as e:
            self.safe_log('warning', f"{self.status['warning']} Could not save chunks: {e}")
    
    def create_comprehensive_study_content(self):
        """Create comprehensive study content for AWS ML certification"""
        return [
            {
                "id": 0,
                "title": "Amazon S3 for Machine Learning",
                "domain": "Domain 1: Data Engineering",
                "content": """Amazon S3 is the foundational storage service for ML workflows on AWS. Key capabilities include:

Storage Classes: Choose from Standard (frequent access), Intelligent-Tiering (automatic optimization), Standard-IA (infrequent access), One Zone-IA (single AZ), Glacier (archival), and Glacier Deep Archive (long-term archival) based on access patterns and cost requirements.

Versioning and Lifecycle: Enable versioning to maintain multiple versions of datasets and models for reproducibility. Configure lifecycle policies to automatically transition data between storage classes as it ages, optimizing costs.

ML Integration: S3 integrates seamlessly with SageMaker for training data access, EMR for big data processing, Glue for ETL operations, and Athena for querying data in place without moving it.

Performance Optimization: Use Transfer Acceleration for faster uploads from global locations, Multi-Part Upload for large files (>100MB recommended), and S3 Select to query specific data without downloading entire objects.

Security Features: Server-side encryption (SSE-S3, SSE-KMS, SSE-C), client-side encryption, bucket policies for access control, IAM policies for user permissions, and VPC endpoints for private connectivity.

Data Lake Architecture: S3 serves as the central data lake, storing raw data, processed data, and model artifacts in organized prefixes and folders.""",
                "key_points": [
                    "Unlimited scalable object storage for all ML data",
                    "Multiple storage classes optimize costs based on access patterns",
                    "Versioning enables data reproducibility and rollback",
                    "Native integration with all AWS ML and analytics services",
                    "Comprehensive security with multiple encryption options"
                ],
                "aws_services": ["Amazon S3", "S3 Transfer Acceleration", "S3 Select", "S3 Intelligent-Tiering"],
                "exam_focus": "Understanding storage class selection, S3 integration patterns, and cost optimization strategies"
            },
            
            {
                "id": 1,
                "title": "Amazon SageMaker Ecosystem",
                "domain": "Domain 3: Modeling",
                "content": """Amazon SageMaker provides a complete machine learning platform with integrated tools:

Data Preparation:
- SageMaker Data Wrangler: Visual interface with 300+ built-in transformations for data preparation
- SageMaker Processing: Run preprocessing and postprocessing workloads at scale
- SageMaker Feature Store: Online and offline feature repository with point-in-time correct retrieval
- SageMaker Clarify: Detect bias in datasets and explain model predictions

Model Development:
- SageMaker Studio: Web-based IDE for ML development with Jupyter notebooks
- Built-in Algorithms: Pre-optimized algorithms for classification, regression, clustering, and more
- Custom Containers: Bring your own algorithms using Docker containers
- SageMaker Autopilot: Automated machine learning with explainable results

Training Infrastructure:
- Managed Training: Fully managed training with automatic resource provisioning
- Distributed Training: Data parallelism and model parallelism for large models
- Spot Training: Use Spot instances with automatic checkpoint recovery
- Automatic Model Tuning: Bayesian optimization for hyperparameter tuning
- SageMaker Experiments: Track, organize, and compare ML experiments

Deployment Options:
- Real-time Endpoints: Auto-scaling inference with A/B testing capabilities
- Batch Transform: Process large datasets for offline predictions
- Multi-Model Endpoints: Host multiple models on a single endpoint
- Serverless Inference: Pay per request with automatic scaling
- Edge Deployment: Deploy models to edge devices with SageMaker Edge Manager""",
                "key_points": [
                    "Complete end-to-end ML platform reducing operational overhead",
                    "Built-in algorithms optimized for AWS infrastructure",
                    "Distributed training scales to handle large datasets and models",
                    "Multiple deployment options for different latency and cost requirements",
                    "Integrated monitoring and explainability tools"
                ],
                "aws_services": [
                    "SageMaker Studio", "SageMaker Data Wrangler", "SageMaker Processing",
                    "SageMaker Training", "SageMaker Endpoints", "SageMaker Autopilot"
                ],
                "exam_focus": "Understanding when to use different SageMaker components and their integration patterns"
            },
            
            {
                "id": 2,
                "title": "Data Preprocessing and Feature Engineering",
                "domain": "Domain 2: Exploratory Data Analysis",
                "content": """Data preprocessing is crucial for ML model success and involves multiple techniques:

Missing Data Strategies:
- Simple Imputation: Replace missing values with mean (numerical), mode (categorical), or median (skewed data)
- Advanced Imputation: K-Nearest Neighbors imputation, iterative imputation using other features
- Deletion: Listwise (remove entire rows) or pairwise (remove specific values) deletion
- Missing Indicators: Create binary flags indicating whether values were missing

Outlier Detection and Treatment:
- Statistical Methods: Z-score (values >3 standard deviations), Modified Z-score using median, IQR method (values outside 1.5*IQR from quartiles)
- Visualization: Box plots, scatter plots, histograms to identify unusual patterns
- Treatment Options: Remove outliers, cap at percentile values, apply transformations (log, sqrt), or model separately

Feature Scaling Techniques:
- Min-Max Normalization: Scale features to [0,1] range, sensitive to outliers
- Z-score Standardization: Transform to mean=0, std=1, assumes normal distribution
- Robust Scaling: Use median and IQR, less sensitive to outliers
- Unit Vector Scaling: Scale to unit norm, useful for text and sparse data

Categorical Variable Encoding:
- One-Hot Encoding: Create binary columns for each category, increases dimensionality
- Label Encoding: Convert categories to integers, implies ordinal relationship
- Target Encoding: Use target variable statistics, risk of data leakage
- Binary Encoding: Reduce dimensionality compared to one-hot encoding

Advanced Feature Engineering:
- Polynomial Features: Create interaction terms and higher-order combinations
- Binning: Convert continuous variables to categorical bins
- Text Processing: TF-IDF vectorization, word embeddings, n-gram features
- Time Series: Extract temporal features like day of week, seasonality, trends
- Domain-Specific: Create features based on business knowledge and domain expertise""",
                "key_points": [
                    "Missing data strategy affects model performance and bias",
                    "Outlier treatment depends on domain knowledge and model type",
                    "Feature scaling requirements vary by algorithm",
                    "Categorical encoding choice impacts model interpretation",
                    "Feature engineering often provides larger performance gains than algorithm tuning"
                ],
                "aws_services": [
                    "SageMaker Data Wrangler", "AWS Glue DataBrew", "SageMaker Processing", 
                    "Amazon Athena", "AWS Glue"
                ],
                "exam_focus": "Selecting appropriate preprocessing techniques for different data types and ML algorithms"
            },
            
            {
                "id": 3,
                "title": "Model Training and Optimization Strategies",
                "domain": "Domain 3: Modeling",
                "content": """Effective model training requires systematic approaches to data splitting, optimization, and resource management:

Data Splitting Best Practices:
- Train/Validation/Test: Common splits are 70/15/15 or 80/10/10 depending on dataset size
- Stratified Sampling: Maintain class distribution across splits for classification problems
- Time Series Splits: Use temporal splits, forward chaining, or expanding windows to prevent data leakage
- Cross-Validation: K-fold (typically k=5 or 10) for robust performance estimation

Training Methodologies:
- Batch Training: Process entire dataset, suitable for small to medium datasets
- Mini-batch Training: Process data in small batches, enables use of GPUs and large datasets
- Stochastic Training: Update model with individual samples, useful for online learning
- Distributed Training: Split training across multiple instances for scalability

Hyperparameter Optimization:
- Grid Search: Exhaustive search over parameter combinations, computationally expensive
- Random Search: Sample parameters randomly, often more efficient than grid search
- Bayesian Optimization: Use probabilistic models to guide search, more efficient for expensive evaluations
- Early Stopping: Monitor validation performance to prevent overfitting

SageMaker Training Capabilities:
- Automatic Model Tuning: Managed hyperparameter optimization using Bayesian methods
- Distributed Training: Supports both data parallelism and model parallelism
- Managed Spot Training: Automatically handle Spot instance interruptions with checkpointing
- Multi-Model Training: Train multiple models simultaneously with different configurations

Performance Monitoring and Debugging:
- Training Metrics: Track loss, accuracy, precision, recall throughout training
- Validation Curves: Monitor training vs validation performance to detect overfitting
- Learning Curves: Assess whether more data would improve performance
- Resource Utilization: Monitor CPU, GPU, memory usage to optimize instance selection
- Profiling: Use SageMaker Debugger to identify bottlenecks and optimize training""",
                "key_points": [
                    "Proper data splitting prevents optimistic bias in performance estimates",
                    "Distributed training enables handling of large datasets and models",
                    "Hyperparameter optimization significantly impacts final model performance",
                    "Spot instances can reduce training costs by up to 90%",
                    "Monitoring training progress prevents overfitting and resource waste"
                ],
                "aws_services": [
                    "SageMaker Training Jobs", "SageMaker Automatic Model Tuning",
                    "SageMaker Debugger", "SageMaker Experiments", "Amazon EC2 Spot Instances"
                ],
                "exam_focus": "Understanding training strategies, optimization techniques, and cost-effective resource utilization"
            },
            
            {
                "id": 4,
                "title": "Model Deployment and Inference Architecture",
                "domain": "Domain 4: ML Implementation and Operations",
                "content": """Model deployment architecture depends on latency, throughput, and cost requirements:

Real-Time Inference Options:
- SageMaker Real-Time Endpoints: Auto-scaling managed endpoints with built-in A/B testing
- API Gateway + Lambda: Serverless architecture for lightweight models under 500MB
- Amazon ECS/EKS: Container-based deployment for custom requirements and microservices
- EC2 Instances: Maximum control over environment and configuration

Batch Inference Patterns:
- SageMaker Batch Transform: Managed batch inference for large datasets
- AWS Batch: Custom batch processing with automatic resource provisioning
- Amazon EMR: Big data processing with Spark MLlib for distributed batch inference
- Lambda with S3 Events: Event-driven batch processing for smaller datasets

Serverless and Edge Options:
- SageMaker Serverless Inference: Pay per request with automatic scaling to zero
- AWS Lambda: For simple models with cold start considerations
- SageMaker Edge Manager: Deploy optimized models to edge devices
- AWS IoT Greengrass: Edge computing platform for local inference

Multi-Model Deployment Strategies:
- Multi-Model Endpoints: Host multiple models on single endpoint to reduce costs
- Model Variants: A/B test different model versions with traffic splitting
- Canary Deployments: Gradually shift traffic from old to new model versions
- Blue/Green Deployments: Switch traffic between two identical environments

Performance Considerations:
- Cold Start Latency: Time to initialize containers and load models
- Throughput Scaling: Requests per second and concurrent request handling
- Model Size Impact: Larger models require more memory and longer loading times
- Geographic Distribution: Use multiple regions for global low-latency access

Cost Optimization Strategies:
- Right-Sizing Instances: Match compute capacity to actual requirements
- Auto-Scaling Policies: Scale based on traffic patterns to minimize idle resources
- Spot Instances: Use for batch workloads and fault-tolerant applications
- Reserved Instances: Commit to longer terms for predictable workloads""",
                "key_points": [
                    "Deployment pattern selection based on latency and throughput requirements",
                    "Serverless options reduce operational overhead but have cold start considerations",
                    "Multi-model hosting strategies optimize resource utilization and costs",
                    "Deployment strategies enable safe production rollouts",
                    "Edge deployment reduces latency for IoT and mobile applications"
                ],
                "aws_services": [
                    "SageMaker Endpoints", "SageMaker Batch Transform", "AWS Lambda",
                    "Amazon API Gateway", "Amazon ECS", "SageMaker Edge Manager"
                ],
                "exam_focus": "Matching deployment architectures to requirements and understanding trade-offs between options"
            },
            
            {
                "id": 5,
                "title": "ML Security, Monitoring, and Operations",
                "domain": "Domain 4: ML Implementation and Operations",
                "content": """Production ML systems require comprehensive security, monitoring, and operational practices:

Security Framework:
- Identity and Access Management: IAM roles for services, least privilege policies, MFA for users
- Data Protection: Encryption at rest (S3, EBS) and in transit (HTTPS, TLS), AWS KMS for key management
- Network Isolation: VPC with private subnets, security groups, NACLs, VPC endpoints for AWS services
- Model Security: Encrypt model artifacts, secure endpoints with authentication, audit model access

Data Governance and Compliance:
- Data Classification: Identify and tag sensitive data using Amazon Macie
- Access Logging: CloudTrail for API calls, VPC Flow Logs for network traffic
- Compliance Frameworks: Support for GDPR, HIPAA, SOC, ISO 27001 requirements
- Data Lineage: Track data flow from source to model predictions

Model Performance Monitoring:
- Accuracy Tracking: Monitor prediction accuracy over time against ground truth
- Data Drift Detection: Statistical tests (KS test, PSI) to detect distribution changes
- Model Drift: Monitor for changes in model performance characteristics
- Business Metrics: Track impact on business KPIs and user satisfaction

Infrastructure Monitoring:
- System Metrics: CPU, memory, network utilization via CloudWatch
- Application Metrics: Custom metrics for model-specific performance indicators
- Alerting: CloudWatch alarms for threshold breaches and anomalies
- Dashboards: Real-time visualization of system and business metrics

Operational Excellence:
- Automated Scaling: Auto Scaling Groups for dynamic capacity management
- Cost Optimization: Regular rightsizing, Reserved Instance planning, Spot usage
- Disaster Recovery: Multi-AZ deployments, automated backups, recovery procedures
- Change Management: Infrastructure as Code, automated testing, gradual rollouts

Model Lifecycle Management:
- Model Registry: Centralized catalog with version control and metadata
- Approval Workflows: Automated testing and manual approval gates
- Deployment Automation: CI/CD pipelines for model deployment
- Model Retirement: Systematic sunset of outdated models with impact analysis""",
                "key_points": [
                    "Security requires layered approach from data to model to infrastructure",
                    "Model performance degrades over time requiring active monitoring",
                    "Data drift detection prevents silent model failures",
                    "Operational automation reduces manual effort and human error",
                    "Comprehensive monitoring enables proactive issue resolution"
                ],
                "aws_services": [
                    "AWS IAM", "AWS KMS", "Amazon VPC", "AWS CloudTrail", "Amazon CloudWatch",
                    "SageMaker Model Monitor", "Amazon Macie", "AWS Config"
                ],
                "exam_focus": "Implementing comprehensive security controls, monitoring strategies, and operational best practices"
            }
        ]
    
    def search_relevant_content(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant study content"""
        if not self.study_chunks:
            return []
        
        query_terms = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.study_chunks:
            searchable_content = (
                chunk.get('title', '') + ' ' +
                chunk.get('content', '') + ' ' +
                ' '.join(chunk.get('key_points', [])) + ' ' +
                ' '.join(chunk.get('aws_services', []))
            ).lower()
            
            content_terms = set(searchable_content.split())
            intersection = query_terms.intersection(content_terms)
            base_score = len(intersection) / len(query_terms) if query_terms else 0
            
            # Boost for matches
            title_boost = 0.3 if any(term in chunk.get('title', '').lower() for term in query_terms) else 0
            service_boost = 0.2 if any(term in ' '.join(chunk.get('aws_services', [])).lower() for term in query_terms) else 0
            
            final_score = base_score + title_boost + service_boost
            scored_chunks.append((final_score, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k] if score > 0.1]
    
    def get_openai_response(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 800) -> str:
        """Get response from OpenAI API"""
        if not self.openai_client:
            return "OpenAI API not available. Please ensure your API key is set in test.env file."
        
        try:
            response = self.openai_client.chat.completions.create(
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
            error_msg = f"OpenAI API error: {str(e)}"
            self.safe_log('error', error_msg)
            return f"I encountered an error generating the response. Please check your API configuration."
    
    def get_comprehensive_answer(self, question: str) -> Dict[str, Any]:
        """Generate comprehensive answer"""
        relevant_chunks = self.search_relevant_content(question, top_k=3)
        
        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I couldn't find specific information about your question. Try asking about AWS ML services like SageMaker, S3, or ML concepts like model training or deployment.",
                "sources": [],
                "confidence": "low"
            }
        
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"**{chunk['title']}**\n{chunk['content'][:1000]}...")
        
        context = "\n\n".join(context_parts)
        prompt = self.prompts["comprehensive_answer"].format(context=context, question=question)
        answer = self.get_openai_response(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [chunk['title'] for chunk in relevant_chunks],
            "context_chunks": len(relevant_chunks),
            "confidence": "high" if len(relevant_chunks) >= 2 else "medium"
        }
    
    def generate_practice_questions(self, topic: str, difficulty: str = "intermediate", num_questions: int = 3) -> List[Dict]:
        """Generate practice questions"""
        relevant_chunks = self.search_relevant_content(topic, top_k=2)
        
        if not relevant_chunks:
            return [{
                "id": 1,
                "question": f"No content found for '{topic}'. Try 'SageMaker', 'S3', 'Data Processing', 'Model Training', or 'Security'.",
                "options": ["A) Try a more specific topic", "B) Check spelling", "C) Use AWS service names", "D) All of the above"],
                "correct": "D",
                "explanation": "The system has information about specific AWS ML services. Try more specific terms."
            }]
        
        content = "\n\n".join([chunk['content'][:800] for chunk in relevant_chunks])
        prompt = self.prompts["question_generator"].format(content=content, num_questions=num_questions, difficulty=difficulty)
        response = self.get_openai_response(prompt, max_tokens=1500)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                questions_data = json.loads(json_match.group())
                return questions_data.get("questions", [])
        except Exception as e:
            self.safe_log('warning', f"Could not parse questions: {e}")
        
        return [{
            "id": 1,
            "question": f"Which AWS service is primarily used for {topic.lower()} in ML workflows?",
            "options": ["A) Amazon S3", "B) Amazon SageMaker", "C) Amazon EC2", "D) Depends on the specific use case"],
            "correct": "D",
            "explanation": f"For {topic}, multiple AWS services may be involved depending on requirements."
        }]
    
    def create_study_plan(self, weak_areas: List[str] = None) -> Dict[str, Any]:
        """Create personalized study plan"""
        if weak_areas is None:
            weak_areas = ["Data Engineering", "Modeling"]
        
        performance_data = {
            "weak_areas": weak_areas,
            "overall_score": 0.65,
            "domain_scores": {
                "Data Engineering": 0.60,
                "Exploratory Data Analysis": 0.75,
                "Modeling": 0.70,
                "ML Implementation and Operations": 0.55
            }
        }
        
        prompt = self.prompts["study_plan"].format(
            performance_data=json.dumps(performance_data, indent=2),
            weak_areas=", ".join(weak_areas)
        )
        
        study_plan = self.get_openai_response(prompt, max_tokens=1000)
        
        return {
            "study_plan": study_plan,
            "weak_areas": weak_areas,
            "recommended_focus_time": "2-3 hours per weak area per week",
            "exam_readiness": "65% based on simulated performance"
        }


def create_streamlit_app():
    """Create Streamlit web application"""
    st.set_page_config(
        page_title="AWS ML Certification Prep",
        page_icon="🎓",
        layout="wide"
    )
    
    # Initialize system
    @st.cache_resource
    def load_exam_prep():
        exam_prep = WindowsRobustExamPrep()
        if exam_prep.setup_environment():
            return exam_prep
        return None
    
    exam_prep = load_exam_prep()
    
    if exam_prep is None:
        st.error("Failed to initialize exam prep system. Please check your configuration.")
        st.stop()
    
    # Main interface
    st.title("AWS ML Certification Exam Prep")
    st.write("AI-powered study system for MLA-C01 certification")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["Study Q&A", "Practice Questions", "Study Plan", "System Status"]
    )
    
    if page == "Study Q&A":
        st.header("Study Q&A")
        st.write("Ask detailed questions about AWS ML concepts")
        
        # Quick topics
        st.subheader("Quick Topics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Amazon S3"):
                st.session_state.question = "How is Amazon S3 used in machine learning workflows?"
        with col2:
            if st.button("SageMaker"):
                st.session_state.question = "What are the key components of Amazon SageMaker?"
        with col3:
            if st.button("Model Deployment"):
                st.session_state.question = "What are the different options for deploying ML models on AWS?"
        
        # Question input
        question = st.text_input(
            "Ask your AWS ML question:",
            value=st.session_state.get('question', ''),
            key='question_input'
        )
        
        if st.button("Get Answer", type="primary"):
            if question:
                with st.spinner("Generating answer..."):
                    response = exam_prep.get_comprehensive_answer(question)
                
                st.success("Answer generated!")
                
                st.markdown("### Answer")
                st.write(response['answer'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"Confidence: {response['confidence']}")
                with col2:
                    st.info(f"Sources Used: {response['context_chunks']}")
                
                if response['sources']:
                    with st.expander("View Sources"):
                        for source in response['sources']:
                            st.write(f"- {source}")
    
    elif page == "Practice Questions":
        st.header("Practice Questions")
        st.write("Generate and answer exam-style questions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            topic = st.selectbox("Select Topic:", ["SageMaker", "S3", "Data Processing", "Model Training", "Security", "Monitoring"])
        with col2:
            difficulty = st.selectbox("Difficulty:", ["easy", "intermediate", "hard"])
        with col3:
            num_questions = st.slider("Number of Questions:", 1, 5, 2)
        
        if st.button("Generate Questions", type="primary"):
            with st.spinner("Generating practice questions..."):
                questions = exam_prep.generate_practice_questions(topic, difficulty, num_questions)
            
            st.session_state.questions = questions
            st.session_state.current_q = 0
            st.session_state.answers = {}
            st.session_state.show_results = False
        
        # Display questions
        if 'questions' in st.session_state and not st.session_state.get('show_results', False):
            questions = st.session_state.questions
            current = st.session_state.current_q
            
            if current < len(questions):
                q = questions[current]
                
                st.markdown(f"### Question {current + 1} of {len(questions)}")
                st.write(q['question'])
                
                answer = st.radio("Choose your answer:", q['options'], key=f"q_{current}")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    if st.button("Previous") and current > 0:
                        st.session_state.current_q -= 1
                        st.rerun()
                
                with col2:
                    if st.button("Submit Answer", type="primary"):
                        st.session_state.answers[current] = answer[0]
                        
                        is_correct = answer[0] == q['correct']
                        if is_correct:
                            st.success("Correct!")
                        else:
                            st.error(f"Incorrect. Correct answer: {q['correct']}")
                        
                        st.info(f"Explanation: {q['explanation']}")
                
                with col3:
                    if st.button("Next") and current < len(questions) - 1:
                        st.session_state.current_q += 1
                        st.rerun()
                    elif current == len(questions) - 1:
                        if st.button("Show Results"):
                            st.session_state.show_results = True
                            st.rerun()
        
        # Show results
        elif st.session_state.get('show_results', False):
            st.markdown("### Practice Results")
            
            questions = st.session_state.questions
            answers = st.session_state.answers
            
            correct_count = sum(1 for i, q in enumerate(questions) if answers.get(i) == q['correct'])
            score_percentage = (correct_count / len(questions)) * 100
            
            if score_percentage >= 80:
                st.success(f"Excellent! Score: {correct_count}/{len(questions)} ({score_percentage:.1f}%)")
            elif score_percentage >= 60:
                st.warning(f"Good job! Score: {correct_count}/{len(questions)} ({score_percentage:.1f}%)")
            else:
                st.error(f"Keep studying! Score: {correct_count}/{len(questions)} ({score_percentage:.1f}%)")
            
            if st.button("Generate New Questions"):
                for key in ['questions', 'current_q', 'answers', 'show_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    elif page == "Study Plan":
        st.header("Personalized Study Plan")
        st.write("Get a customized study plan based on your weak areas")
        
        weak_areas = st.multiselect(
            "Select your weak areas:",
            ["Data Engineering", "Exploratory Data Analysis", "Modeling", "ML Implementation and Operations"],
            default=["Data Engineering", "Modeling"]
        )
        
        if st.button("Generate Study Plan", type="primary"):
            with st.spinner("Creating personalized study plan..."):
                try:
                    study_plan_response = exam_prep.create_study_plan(weak_areas)
                    
                    st.success("Study plan created!")
                    
                    st.markdown("### Your Personalized Study Plan")
                    st.write(study_plan_response['study_plan'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"Focus Areas: {', '.join(study_plan_response['weak_areas'])}")
                    with col2:
                        st.info(f"Recommended Time: {study_plan_response['recommended_focus_time']}")
                    
                    st.warning(f"Current Exam Readiness: {study_plan_response['exam_readiness']}")
                    
                except Exception as e:
                    st.error(f"Error creating study plan: {str(e)}")
                    st.info("Please check your OpenAI API configuration.")
    
    elif page == "System Status":
        st.header("System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuration")
            st.write(f"Project Directory: {exam_prep.project_dir}")
            st.write(f"OpenAI Client: {'Available' if exam_prep.openai_client else 'Not Available'}")
            st.write(f"Embedding Model: {'Loaded' if exam_prep.embedding_model else 'Not Loaded'}")
            st.write(f"Study Chunks: {len(exam_prep.study_chunks)} loaded")
        
        with col2:
            st.subheader("System Health")
            if exam_prep.setup_complete:
                st.success("System Ready")
            else:
                st.error("System Not Ready")
            
            if st.button("Test OpenAI Connection"):
                if exam_prep.openai_client:
                    try:
                        test_response = exam_prep.get_openai_response("Test message", max_tokens=20)
                        if "error" not in test_response.lower():
                            st.success("OpenAI API working correctly")
                            st.write(f"Test response: {test_response[:100]}...")
                        else:
                            st.error(f"OpenAI API error: {test_response}")
                    except Exception as e:
                        st.error(f"Connection test failed: {str(e)}")
                else:
                    st.error("OpenAI client not configured")
        
        st.subheader("API Information")
        try:
            import openai
            st.info(f"OpenAI library version: {openai.__version__}")
            if exam_prep.openai_client:
                st.success("Using modern OpenAI API (v1.0+)")
            else:
                st.warning("OpenAI client not initialized")
        except Exception as e:
            st.error(f"Error checking OpenAI version: {e}")


def run_cli_session():
    """Run command-line interface"""
    exam_prep = WindowsRobustExamPrep()
    
    if not exam_prep.setup_environment():
        exam_prep.safe_print("Failed to setup environment. Check the logs.")
        return
    
    exam_prep.safe_print("\n" + "="*60)
    exam_prep.safe_print("AWS ML CERTIFICATION EXAM PREP - CLI")
    exam_prep.safe_print("="*60)
    exam_prep.safe_print("Commands:")
    exam_prep.safe_print("  ask [question]     - Get comprehensive answers")
    exam_prep.safe_print("  practice [topic]   - Generate practice questions")
    exam_prep.safe_print("  topics             - Show available study topics")
    exam_prep.safe_print("  status             - Show system status")
    exam_prep.safe_print("  quit               - Exit")
    exam_prep.safe_print("="*60)
    
    while True:
        try:
            user_input = input("\nEnter command: ").strip()
            
            if user_input.lower() == 'quit':
                exam_prep.safe_print("Good luck with your AWS ML certification!")
                break
            
            elif user_input.lower() == 'topics':
                exam_prep.safe_print("\nAvailable Topics:")
                for chunk in exam_prep.study_chunks:
                    exam_prep.safe_print(f"  - {chunk['title']}")
            
            elif user_input.lower() == 'status':
                exam_prep.safe_print(f"\nSystem Status:")
                exam_prep.safe_print(f"  Project Directory: {exam_prep.project_dir}")
                exam_prep.safe_print(f"  OpenAI Client: {'Available' if exam_prep.openai_client else 'Not Available'}")
                exam_prep.safe_print(f"  Study Content: {len(exam_prep.study_chunks)} chunks")
                exam_prep.safe_print(f"  Setup Complete: {'Yes' if exam_prep.setup_complete else 'No'}")
            
            elif user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                if question:
                    exam_prep.safe_print(f"\nSearching for: {question}")
                    response = exam_prep.get_comprehensive_answer(question)
                    
                    exam_prep.safe_print(f"\nANSWER:")
                    exam_prep.safe_print(response['answer'])
                    
                    if response['sources']:
                        exam_prep.safe_print(f"\nSOURCES: {', '.join(response['sources'])}")
                    exam_prep.safe_print(f"CONFIDENCE: {response['confidence']}")
                else:
                    exam_prep.safe_print("Please provide a question after 'ask'")
            
            elif user_input.lower().startswith('practice '):
                topic = user_input[9:].strip()
                if topic:
                    exam_prep.safe_print(f"\nGenerating practice questions for: {topic}")
                    questions = exam_prep.generate_practice_questions(topic, num_questions=2)
                    
                    for i, q in enumerate(questions, 1):
                        exam_prep.safe_print(f"\nQUESTION {i}: {q['question']}")
                        for option in q['options']:
                            exam_prep.safe_print(f"  {option}")
                        
                        user_answer = input("Your answer (A/B/C/D): ").strip().upper()
                        
                        if user_answer == q['correct']:
                            exam_prep.safe_print("CORRECT!")
                        else:
                            exam_prep.safe_print(f"INCORRECT. Correct answer: {q['correct']}")
                        
                        exam_prep.safe_print(f"EXPLANATION: {q['explanation']}")
                else:
                    exam_prep.safe_print("Please provide a topic after 'practice'")
            
            else:
                exam_prep.safe_print("Unknown command. Available: ask, practice, topics, status, quit")
        
        except KeyboardInterrupt:
            exam_prep.safe_print("\nSession interrupted. Good luck!")
            break
        except Exception as e:
            exam_prep.safe_print(f"Error: {e}")


def main():
    """Main function"""
    print("AWS ML Certification Exam Prep System")
    print("Windows-Compatible Version")
    print("="*50)
    
    print("\nChoose interface:")
    print("1. Web Interface (Streamlit) - Recommended")
    print("2. Command Line Interface")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\nStarting web interface...")
            create_streamlit_app()
        elif choice == "2":
            run_cli_session()
        else:
            print("Invalid choice. Starting web interface...")
            create_streamlit_app()
    
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    try:
        import streamlit as st
        create_streamlit_app()
    except:
        main()