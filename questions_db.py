"""
AWS Certification Questions Database
Extracted from ML Engineer and Data Engineer Master Cheat Sheets
"""

QUESTIONS_DATABASE = [
    # ML Engineer - Domain 1: Data Engineering
    {
        "certification": "ML Engineer",
        "domain": "Data Engineering",
        "question": "Which AWS service is most suitable for storing large datasets with high scalability, durability, and availability?",
        "options": [
            "Amazon RDS",
            "Amazon DynamoDB",
            "Amazon S3",
            "Amazon EBS"
        ],
        "correct_answer": 2,
        "explanation": "Amazon S3 is an object storage service that offers high scalability, durability, and availability. It's well-suited for storing large datasets, including raw data, processed data, and model artifacts.",
        "related_topic": "Storage Services"
    },
    {
        "certification": "ML Engineer",
        "domain": "Data Engineering",
        "question": "Which data ingestion approach is suitable for large datasets where real-time processing is not required?",
        "options": [
            "Streaming Processing",
            "Batch Processing",
            "Real-time Processing",
            "Event-driven Processing"
        ],
        "correct_answer": 1,
        "explanation": "Batch processing involves collecting data over a period and processing it in batches. It's cost-effective for large datasets where real-time processing is not required.",
        "related_topic": "Data Ingestion"
    },
    {
        "certification": "ML Engineer",
        "domain": "Data Engineering",
        "question": "What is the primary purpose of AWS Glue in ML data pipelines?",
        "options": [
            "Real-time data streaming",
            "Extract, Transform, Load (ETL) operations",
            "Model training",
            "Endpoint hosting"
        ],
        "correct_answer": 1,
        "explanation": "AWS Glue is a fully managed ETL service that makes it easy to prepare and load data for analytics and machine learning. It provides automated data discovery, code generation, and job scheduling.",
        "related_topic": "Data Transformation"
    },
    {
        "certification": "ML Engineer",
        "domain": "Data Engineering",
        "question": "Which AWS service should be used for real-time processing of streaming data at massive scale?",
        "options": [
            "AWS Batch",
            "Amazon S3",
            "Amazon Kinesis",
            "AWS Glue"
        ],
        "correct_answer": 2,
        "explanation": "Amazon Kinesis is a family of services for real-time processing of streaming data at massive scale. It includes Kinesis Data Streams for capturing data, Firehose for loading, and Analytics for processing.",
        "related_topic": "Streaming Data"
    },
    {
        "certification": "ML Engineer",
        "domain": "Data Engineering",
        "question": "What is the purpose of MapReduce in big data processing?",
        "options": [
            "For storing data in a distributed manner",
            "For processing and generating big datasets with a parallel, distributed algorithm",
            "For real-time analytics",
            "For model inference"
        ],
        "correct_answer": 1,
        "explanation": "MapReduce is a programming model for processing large datasets in parallel across a cluster. The Map function converts data into key-value pairs, and the Reduce function combines tuples into a smaller set.",
        "related_topic": "Big Data Processing"
    },
    {
        "certification": "ML Engineer",
        "domain": "Data Engineering",
        "question": "Which service is best for orchestrating complex data workflows with multiple dependencies?",
        "options": [
            "AWS Lambda",
            "Amazon EventBridge",
            "Amazon Managed Workflows for Apache Airflow (MWAA)",
            "AWS Batch"
        ],
        "correct_answer": 2,
        "explanation": "Amazon MWAA is a managed service for Apache Airflow that provides a visual interface for defining complex data pipelines with dependencies between tasks.",
        "related_topic": "Workflow Orchestration"
    },

    # ML Engineer - Domain 2: Exploratory Data Analysis
    {
        "certification": "ML Engineer",
        "domain": "EDA & Data Preparation",
        "question": "Which technique is used to fill in missing values in a dataset?",
        "options": [
            "Imputation",
            "Normalization",
            "Augmentation",
            "Scaling"
        ],
        "correct_answer": 0,
        "explanation": "Imputation is a technique for filling in missing values with estimated values. This can be done using simple methods like mean/median imputation or more sophisticated techniques like k-NN imputation.",
        "related_topic": "Missing Data Handling"
    },
    {
        "certification": "ML Engineer",
        "domain": "EDA & Data Preparation",
        "question": "What is the purpose of normalization in data preprocessing?",
        "options": [
            "Removing duplicate records",
            "Scaling data to a standard range (typically 0-1)",
            "Removing outliers",
            "Converting categorical variables"
        ],
        "correct_answer": 1,
        "explanation": "Normalization scales data to a standard range, typically between 0 and 1. This prevents features with larger values from dominating those with smaller values during model training.",
        "related_topic": "Feature Scaling"
    },
    {
        "certification": "ML Engineer",
        "domain": "EDA & Data Preparation",
        "question": "Which AWS service can automatically label data at scale using human workers and automated labeling?",
        "options": [
            "Amazon Mechanical Turk",
            "Amazon SageMaker Ground Truth",
            "AWS Glue",
            "Amazon Athena"
        ],
        "correct_answer": 1,
        "explanation": "Amazon SageMaker Ground Truth is a fully managed data labeling service that can use MTurk workers, private workforces, or vendor-managed workforces. It also supports automated labeling to reduce costs.",
        "related_topic": "Data Labeling"
    },
    {
        "certification": "ML Engineer",
        "domain": "EDA & Data Preparation",
        "question": "What is one-hot encoding used for?",
        "options": [
            "Converting continuous values into discrete bins",
            "Converting categorical variables into numerical representations",
            "Removing duplicate records",
            "Identifying and treating outliers"
        ],
        "correct_answer": 1,
        "explanation": "One-hot encoding converts categorical features into numerical features using binary vectors where only one element is 1 and the rest are 0. This prevents artificial ordinal relationships between categories.",
        "related_topic": "Categorical Encoding"
    },
    {
        "certification": "ML Engineer",
        "domain": "EDA & Data Preparation",
        "question": "Which technique reduces the number of features while retaining important information?",
        "options": [
            "Feature engineering",
            "Dimensionality reduction",
            "Data augmentation",
            "Outlier removal"
        ],
        "correct_answer": 1,
        "explanation": "Dimensionality reduction techniques like PCA reduce the number of features in a dataset while retaining important information. This improves model performance by reducing overfitting and computational complexity.",
        "related_topic": "Feature Engineering"
    },
    {
        "certification": "ML Engineer",
        "domain": "EDA & Data Preparation",
        "question": "What does correlation measure in descriptive statistics?",
        "options": [
            "The average value of data",
            "The spread or dispersion of data",
            "The strength and direction of a linear relationship between two variables",
            "The middle value when data is ordered"
        ],
        "correct_answer": 2,
        "explanation": "Correlation measures the strength and direction of a linear relationship between two variables, ranging from -1 (perfect negative) to +1 (perfect positive), with 0 indicating no linear correlation.",
        "related_topic": "Statistical Analysis"
    },

    # ML Engineer - Domain 3: Modeling
    {
        "certification": "ML Engineer",
        "domain": "Modeling",
        "question": "Which type of learning algorithm learns from labeled data with known outputs?",
        "options": [
            "Unsupervised learning",
            "Supervised learning",
            "Reinforcement learning",
            "Transfer learning"
        ],
        "correct_answer": 1,
        "explanation": "Supervised learning algorithms learn from labeled data where the input and desired output are provided. Examples include classification and regression tasks.",
        "related_topic": "ML Algorithms"
    },
    {
        "certification": "ML Engineer",
        "domain": "Modeling",
        "question": "Which algorithm is suitable for binary classification problems with interpretability emphasis?",
        "options": [
            "Neural Networks",
            "Decision Trees",
            "Logistic Regression",
            "XGBoost"
        ],
        "correct_answer": 2,
        "explanation": "Logistic Regression is used for binary classification and is simple and interpretable. It predicts the probability of belonging to a certain class using an S-shaped sigmoid curve.",
        "related_topic": "Classification Models"
    },
    {
        "certification": "ML Engineer",
        "domain": "Modeling",
        "question": "What is the purpose of cross-validation in model evaluation?",
        "options": [
            "To train models faster",
            "To assess how well a model generalizes to independent data",
            "To reduce model complexity",
            "To handle missing values"
        ],
        "correct_answer": 1,
        "explanation": "Cross-validation helps assess how well a model generalizes to unseen data by dividing the dataset into k folds and training the model k times, each time using a different fold as validation.",
        "related_topic": "Model Evaluation"
    },
    {
        "certification": "ML Engineer",
        "domain": "Modeling",
        "question": "Which regularization technique randomly 'drops out' neurons during training to prevent overfitting?",
        "options": [
            "L1 Regularization",
            "L2 Regularization",
            "Dropout",
            "Early Stopping"
        ],
        "correct_answer": 2,
        "explanation": "Dropout is a technique used in neural networks where a fraction of neurons are randomly ignored during training. This prevents the network from relying on specific neurons and encourages learning robust features.",
        "related_topic": "Regularization"
    },
    {
        "certification": "ML Engineer",
        "domain": "Modeling",
        "question": "What does the 'elbow point' represent in k-means clustering?",
        "options": [
            "The maximum cluster size",
            "The optimal number of clusters where the rate of decrease in WCSS sharply changes",
            "The center point of each cluster",
            "The number of outliers in the data"
        ],
        "correct_answer": 1,
        "explanation": "The elbow point in a k-means elbow plot indicates where the within-cluster sum of squares (WCSS) stops decreasing significantly, suggesting a good value for the number of clusters.",
        "related_topic": "Clustering"
    },
    {
        "certification": "ML Engineer",
        "domain": "Modeling",
        "question": "Which metric is most appropriate for imbalanced datasets in classification?",
        "options": [
            "Accuracy",
            "F1 Score",
            "Precision only",
            "Recall only"
        ],
        "correct_answer": 1,
        "explanation": "F1 Score is the harmonic mean of precision and recall, providing a balance between the two. It's especially useful for imbalanced datasets where accuracy can be misleading.",
        "related_topic": "Evaluation Metrics"
    },
    {
        "certification": "ML Engineer",
        "domain": "Modeling",
        "question": "What is the gradient descent algorithm used for?",
        "options": [
            "Scaling features",
            "Finding the minimum of a loss function",
            "Removing outliers",
            "Encoding categorical variables"
        ],
        "correct_answer": 1,
        "explanation": "Gradient descent is an iterative optimization algorithm that finds the minimum of the loss function by repeatedly adjusting model parameters in the direction of steepest descent.",
        "related_topic": "Optimization"
    },

    # Data Engineer - Domain 1: Data Ingestion
    {
        "certification": "Data Engineer",
        "domain": "Data Ingestion",
        "question": "Which AWS service is best for capturing real-time streaming data continuously?",
        "options": [
            "Amazon S3",
            "Amazon Kinesis",
            "AWS Glue",
            "Amazon Redshift"
        ],
        "correct_answer": 1,
        "explanation": "Amazon Kinesis is a managed service for real-time data streams that scales automatically to handle high volumes of continuous data. It captures and stores streams of data records.",
        "related_topic": "Streaming Sources"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Ingestion",
        "question": "Which service is ideal for loading streaming data into data lakes and data stores?",
        "options": [
            "Amazon Kinesis Data Streams",
            "Amazon Kinesis Data Firehose",
            "AWS Batch",
            "AWS Glue"
        ],
        "correct_answer": 1,
        "explanation": "Amazon Kinesis Data Firehose is the easiest way to load streaming data into data lakes and stores. It's fully managed, automatically scaling and can transform data formats.",
        "related_topic": "Data Loading"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Ingestion",
        "question": "What is the primary purpose of AWS EventBridge in data pipelines?",
        "options": [
            "Data transformation",
            "Event routing and scheduling",
            "Data storage",
            "Model training"
        ],
        "correct_answer": 1,
        "explanation": "AWS EventBridge is a serverless event bus that allows you to schedule events and trigger various AWS services based on pre-defined rules or events from other services.",
        "related_topic": "Scheduling"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Ingestion",
        "question": "Which technique allows managing the rate of data access to avoid exceeding source limits?",
        "options": [
            "Fan-in",
            "Fan-out",
            "Throttling",
            "Batching"
        ],
        "correct_answer": 2,
        "explanation": "Throttling involves managing data access to avoid exceeding rate limits of data sources. Techniques include exponential backoff for retries and batching to reduce the number of requests.",
        "related_topic": "Rate Limiting"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Ingestion",
        "question": "What is the difference between fan-in and fan-out in data streaming?",
        "options": [
            "Fan-in splits data; fan-out merges data",
            "Fan-in merges data streams; fan-out distributes a single stream to multiple destinations",
            "They are the same concept",
            "Fan-in is for batch; fan-out is for streaming"
        ],
        "correct_answer": 1,
        "explanation": "Fan-in merges data streams from multiple sources into a single stream. Fan-out distributes a single data stream to multiple destinations for parallel processing.",
        "related_topic": "Stream Distribution"
    },

    # Data Engineer - Domain 2: Data Store Management
    {
        "certification": "Data Engineer",
        "domain": "Data Storage",
        "question": "Which data store is optimized for analytical workloads on large historical datasets?",
        "options": [
            "Amazon DynamoDB",
            "Amazon RDS",
            "Amazon Redshift",
            "Amazon Kinesis"
        ],
        "correct_answer": 2,
        "explanation": "Amazon Redshift is a fully managed data warehouse optimized for analytical workloads on large datasets. It offers good cost-performance for running complex SQL queries on historical data.",
        "related_topic": "Data Warehouses"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Storage",
        "question": "Which AWS service is best for key-value and document data with high scalability?",
        "options": [
            "Amazon Redshift",
            "Amazon RDS",
            "Amazon DynamoDB",
            "Amazon EMR"
        ],
        "correct_answer": 2,
        "explanation": "Amazon DynamoDB is a NoSQL database that provides high performance and scalability for key-value and document data. It's ideal for applications requiring predictable low latency.",
        "related_topic": "NoSQL Databases"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Storage",
        "question": "What is the primary use case for AWS Lake Formation?",
        "options": [
            "Real-time streaming",
            "Setting up a secure data lake for raw data in various formats",
            "Model training",
            "API hosting"
        ],
        "correct_answer": 1,
        "explanation": "AWS Lake Formation is a service for setting up a secure data lake to store raw data in various formats. It simplifies data governance and cataloging.",
        "related_topic": "Data Lakes"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Storage",
        "question": "Which service is used for querying data in S3 without loading it into a data warehouse?",
        "options": [
            "Amazon Redshift",
            "Amazon Redshift Spectrum",
            "Amazon RDS",
            "Amazon DynamoDB"
        ],
        "correct_answer": 1,
        "explanation": "Amazon Redshift Spectrum enables direct querying of data stored in S3 without loading it into Redshift tables. It's ideal for analyzing infrequently accessed or very large datasets.",
        "related_topic": "Remote Querying"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Storage",
        "question": "What is a materialized view in Amazon Redshift?",
        "options": [
            "A temporary table that exists only during a query",
            "A pre-computed subset of data based on a query that can be accessed faster",
            "A virtual table that shows real-time data",
            "A backup copy of a table"
        ],
        "correct_answer": 1,
        "explanation": "Materialized views are pre-computed subsets of data based on a query that can be accessed faster than the original source. This improves query performance for frequently accessed data.",
        "related_topic": "Query Optimization"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Storage",
        "question": "Which S3 storage class is most cost-effective for rarely accessed data with retrieval times in hours?",
        "options": [
            "S3 Standard",
            "S3 Intelligent-Tiering",
            "S3 Glacier Deep Archive",
            "S3 Glacier"
        ],
        "correct_answer": 2,
        "explanation": "S3 Glacier Deep Archive is the lowest-cost storage class for rarely accessed data. It has retrieval times measured in hours and is suitable for long-term retention.",
        "related_topic": "Storage Tiers"
    },

    # Data Engineer - Domain 3: Data Transformation
    {
        "certification": "Data Engineer",
        "domain": "Data Transformation",
        "question": "Which service is best for large-scale distributed data processing using Spark or Hadoop?",
        "options": [
            "AWS Glue",
            "Amazon EMR",
            "AWS Lambda",
            "AWS Batch"
        ],
        "correct_answer": 1,
        "explanation": "Amazon EMR is a managed Hadoop framework for large-scale data processing. It supports Apache Spark and other big data tools for distributed processing.",
        "related_topic": "Big Data Processing"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Transformation",
        "question": "What does JDBC stand for and what is its purpose?",
        "options": [
            "Java Database Connection for connecting to relational databases",
            "JavaScript Database Connection",
            "Java Data Batch Connection",
            "Just Data Connection Interface"
        ],
        "correct_answer": 0,
        "explanation": "JDBC (Java Database Connectivity) is a standardized API that allows applications to connect to various relational databases. It provides a layer of abstraction between the application and specific database systems.",
        "related_topic": "Database Connectivity"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Transformation",
        "question": "Which format is most efficient for analytical workloads due to better compression and query performance?",
        "options": [
            "CSV",
            "JSON",
            "Apache Parquet",
            "XML"
        ],
        "correct_answer": 2,
        "explanation": "Apache Parquet is a columnar data format that offers better compression and faster query performance for analytical workloads compared to row-based formats like CSV.",
        "related_topic": "Data Formats"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Transformation",
        "question": "Which AWS service provides a visual interface for data preparation without writing code?",
        "options": [
            "AWS Glue",
            "AWS Glue DataBrew",
            "AWS Lambda",
            "Amazon EMR"
        ],
        "correct_answer": 1,
        "explanation": "AWS Glue DataBrew is a visual data preparation tool that simplifies data cleaning and transformation without requiring code. It includes profiling, cleaning, and recipe management capabilities.",
        "related_topic": "Data Preparation"
    },

    # Data Engineer - Domain 4: Data Quality
    {
        "certification": "Data Engineer",
        "domain": "Data Quality",
        "question": "What is referential integrity in data consistency?",
        "options": [
            "Data types match across columns",
            "Foreign keys in one table correspond to valid primary keys in another table",
            "All values are non-null",
            "Data is sorted correctly"
        ],
        "correct_answer": 1,
        "explanation": "Referential integrity ensures that foreign keys in one table reference valid primary keys in another table. This maintains consistency across related datasets.",
        "related_topic": "Data Consistency"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Quality",
        "question": "Which AWS service automatically discovers and classifies sensitive data like PII?",
        "options": [
            "AWS Glue",
            "Amazon Macie",
            "AWS Config",
            "Amazon Athena"
        ],
        "correct_answer": 1,
        "explanation": "Amazon Macie automatically discovers and classifies sensitive data like PII within S3 buckets using pre-defined patterns and machine learning.",
        "related_topic": "PII Protection"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Quality",
        "question": "What is a data quality rule?",
        "options": [
            "A guideline for how to organize files",
            "A specific condition that data must adhere to to be considered valid",
            "A type of encryption algorithm",
            "A backup strategy for data"
        ],
        "correct_answer": 1,
        "explanation": "Data quality rules are specific conditions that data must meet to be valid. Examples include valid email format, positive prices, or dates within a specific range.",
        "related_topic": "Data Validation"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Quality",
        "question": "Which technique helps detect outliers in numerical data?",
        "options": [
            "Normalization",
            "Encoding",
            "Interquartile Range (IQR)",
            "Sampling"
        ],
        "correct_answer": 2,
        "explanation": "The Interquartile Range (IQR) is a statistical method used to detect outliers. Values outside the range of Q1-1.5*IQR to Q3+1.5*IQR are typically considered outliers.",
        "related_topic": "Outlier Detection"
    },

    # Data Engineer - Data Warehousing & Analytics
    {
        "certification": "Data Engineer",
        "domain": "Data Warehousing",
        "question": "What is the COPY command in Amazon Redshift used for?",
        "options": [
            "Creating backup copies of tables",
            "Loading data from S3 into Redshift tables",
            "Duplicating database schemas",
            "Copying data between Redshift clusters"
        ],
        "correct_answer": 1,
        "explanation": "The COPY command is the primary way to load data from S3 into Redshift. It allows parallel processing for faster data ingestion and supports various file formats.",
        "related_topic": "Data Loading"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Warehousing",
        "question": "What is the UNLOAD command in Amazon Redshift used for?",
        "options": [
            "Removing data from tables",
            "Unloading data from Redshift to S3",
            "Deleting databases",
            "Stopping query execution"
        ],
        "correct_answer": 1,
        "explanation": "The UNLOAD command extracts data from Redshift and writes it to S3. It allows specifying destination location, file format, and compression options.",
        "related_topic": "Data Export"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Warehousing",
        "question": "Which Redshift feature allows querying other databases like RDS without copying data?",
        "options": [
            "Redshift Spectrum",
            "Redshift Materialized Views",
            "Redshift Federated Queries",
            "Redshift Cross-Database Access"
        ],
        "correct_answer": 2,
        "explanation": "Redshift Federated Queries allow querying data across multiple databases including RDS, Aurora, and other Redshift clusters. This eliminates the need for complex data movement.",
        "related_topic": "Multi-Source Querying"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Warehousing",
        "question": "What is a data catalog and why is it important?",
        "options": [
            "A list of all SQL queries run on the system",
            "A central registry of data assets with metadata for discovering and understanding data",
            "A backup log of all data changes",
            "A type of encryption for data"
        ],
        "correct_answer": 1,
        "explanation": "A data catalog is a central registry containing metadata about data assets including schemas, lineage, and descriptions. It helps teams discover and understand available data.",
        "related_topic": "Data Governance"
    },
    {
        "certification": "Data Engineer",
        "domain": "Data Warehousing",
        "question": "Which AWS service is used for automatic schema discovery and data cataloging?",
        "options": [
            "Amazon Athena",
            "AWS Glue Crawlers",
            "Amazon Redshift",
            "AWS Lambda"
        ],
        "correct_answer": 1,
        "explanation": "AWS Glue Crawlers automatically crawl through data sources and discover schemas. They update the Glue Data Catalog with metadata about columns, data types, and partitions.",
        "related_topic": "Schema Discovery"
    },

    # Security & Operations
    {
        "certification": "ML Engineer",
        "domain": "Security & Operations",
        "question": "Which AWS service provides centralized management of encryption keys?",
        "options": [
            "AWS IAM",
            "AWS KMS",
            "AWS Secrets Manager",
            "AWS CloudTrail"
        ],
        "correct_answer": 1,
        "explanation": "AWS Key Management Service (KMS) provides secure and centralized management of encryption keys used for protecting data in AWS services.",
        "related_topic": "Encryption"
    },
    {
        "certification": "Data Engineer",
        "domain": "Security & Operations",
        "question": "What is the purpose of AWS Secrets Manager?",
        "options": [
            "Creating security groups",
            "Managing IAM policies",
            "Storing and managing sensitive credentials like passwords and API keys",
            "Encrypting data at rest"
        ],
        "correct_answer": 2,
        "explanation": "AWS Secrets Manager is a service for securely storing and managing sensitive information like passwords, API keys, and database credentials. It supports automated secret rotation.",
        "related_topic": "Credential Management"
    },
    {
        "certification": "ML Engineer",
        "domain": "Security & Operations",
        "question": "Which principle should be followed when granting IAM permissions?",
        "options": [
            "Grant all permissions for convenience",
            "Principle of Least Privilege - grant only necessary permissions",
            "Grant permissions based on department",
            "Grant permissions only to administrators"
        ],
        "correct_answer": 1,
        "explanation": "The Principle of Least Privilege means granting only the minimum permissions necessary for a user or service to perform their intended tasks, reducing security risks.",
        "related_topic": "IAM Best Practices"
    },
    {
        "certification": "Data Engineer",
        "domain": "Security & Operations",
        "question": "Which AWS service tracks all API calls and user activity for audit purposes?",
        "options": [
            "Amazon CloudWatch",
            "AWS CloudTrail",
            "AWS Config",
            "Amazon GuardDuty"
        ],
        "correct_answer": 1,
        "explanation": "AWS CloudTrail records all API calls made to and from your AWS account, capturing details like who made the call, when, which service was called, and the resources involved.",
        "related_topic": "Audit Logging"
    },
]

TOPICS = {
    "ML Engineer": [
        "Data Engineering",
        "EDA & Data Preparation",
        "Modeling",
        "Security & Operations"
    ],
    "Data Engineer": [
        "Data Ingestion",
        "Data Storage",
        "Data Transformation",
        "Data Quality",
        "Data Warehousing",
        "Security & Operations"
    ]
}
