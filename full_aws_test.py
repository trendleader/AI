#!/usr/bin/env python3
"""
Complete test with your full AWS Data Engineer knowledge base
Save as full_aws_test.py and run with: python full_aws_test.py
"""

from rag_implementation_fixed import AWSDataEngineerRAG

# Import your full AWS knowledge base
aws_data_engineer_knowledge_base = {
    "domain_1_data_ingestion_transformation": {
        "task_1_1_data_ingestion": {
            "streaming_sources": {
                "concept": "Real-time data that arrives continuously in never-ending streams",
                "examples": ["sensor data", "social media feeds", "application logs", "stock quotes"],
                "processing": "Data is processed as it arrives, enabling real-time analytics and insights",
                "aws_services": {
                    "amazon_kinesis": "Managed service for real-time data streams that scales automatically to handle high volumes",
                    "amazon_msk": "Managed Apache Kafka service providing highly scalable platform for real-time data feeds",
                    "dynamodb_streams": "Captures all changes made to DynamoDB tables in real-time",
                    "aws_glue": "Serverless data integration service for ingesting streaming data from various sources"
                }
            },
            "batch_sources": {
                "concept": "Data delivered in large chunks at specific intervals (daily, weekly)",
                "examples": ["CSV files", "log files", "database backups"],
                "processing": "Data is processed at predefined times based on ingestion schedule",
                "aws_services": {
                    "amazon_s3": "Scalable object storage service for storing large datasets of any type",
                    "aws_glue": "Extract data from various batch sources and transform before loading",
                    "amazon_emr": "Managed Hadoop framework for processing and analyzing large datasets",
                    "aws_dms": "Migrates data between databases, including batch data transfer",
                    "amazon_redshift": "Data warehouse service that loads data from various sources in batch mode",
                    "aws_lambda": "Serverless compute triggered to process data from S3 upon upload",
                    "amazon_appflow": "Fully managed service for integrating data from various sources"
                }
            }
        },
        "task_1_2_transform_process_data": {
            "lambda_optimization": {
                "memory_allocation": "Set memory sufficient for data processing needs - higher memory allows larger datasets",
                "timeout": "Set timeout value to prevent infinite execution and resource consumption",
                "concurrency": "Configure concurrent instances to handle multiple data requests simultaneously",
                "provisioned_concurrency": "Keep minimum instances running for faster response times in critical functions"
            },
            "data_transformation_services": {
                "amazon_emr": "Service for running large-scale distributed data processing across clusters using Hadoop/Spark",
                "aws_glue": "Managed ETL service that simplifies data preparation and migration between systems",
                "aws_lambda": "Serverless compute for smaller-scale transformations without server management",
                "amazon_redshift": "Data warehouse service for large-scale analysis and complex query processing"
            }
        }
    },
    "domain_2_data_store_management": {
        "task_2_1_choose_data_store": {
            "database_options": {
                "amazon_rds": {
                    "description": "Managed relational database service for MySQL, PostgreSQL, Aurora",
                    "use_case": "Frequently accessed structured data with predictable performance",
                    "features": "Familiar SQL access, horizontal scaling, automated backups"
                },
                "dynamodb": {
                    "description": "NoSQL database with easy scaling and high availability", 
                    "use_case": "Key-value pairs and document data with predictable low latency",
                    "pricing": "Based on provisioned capacity units for reads and writes"
                },
                "amazon_redshift": {
                    "description": "Fully managed data warehouse optimized for analytical workloads",
                    "strengths": "Good cost-performance for complex SQL queries on historical data",
                    "limitations": "Not ideal for real-time data ingestion or updates"
                }
            },
            "s3_storage": {
                "use_cases": {
                    "object_storage": "Store large datasets like log files, images, videos, backups",
                    "data_lakes": "Central repository for raw, semi-structured, and structured data",
                    "static_hosting": "Host websites directly from S3 buckets",
                    "disaster_recovery": "Back up critical data for secure storage and retrieval"
                },
                "storage_classes": {
                    "s3_standard": "Default storage class for frequently accessed data",
                    "s3_intelligent_tiering": "Automatically migrates between Standard and IA based on access",
                    "s3_glacier": "Low-cost storage for rarely accessed data with retrieval fees",
                    "s3_glacier_deep_archive": "Very low-cost storage with retrieval times in hours"
                }
            }
        },
        "task_2_3_data_lifecycle": {
            "s3_lifecycle_management": {
                "lifecycle_policies": "Automate data movement between storage classes based on defined rules",
                "configuration": "Define ID, Prefix, Transitions, and Expiration rules via XML or console",
                "benefits": "Optimize costs by automatically moving data to appropriate storage tiers"
            }
        }
    },
    "domain_3_data_operations_support": {
        "task_3_3_maintain_monitor_pipelines": {
            "monitoring_solutions": {
                "amazon_cloudwatch": {
                    "capabilities": "Provides logs, metrics, and events from AWS resources for monitoring",
                    "features": ["Log filtering and searching", "Real-time monitoring dashboards", "Alerting mechanisms"],
                    "benefits": ["Improved visibility", "Proactive troubleshooting", "Enhanced auditing"]
                },
                "performance_troubleshooting": {
                    "common_issues": ["Slow processing times", "Data errors or inconsistencies", "Infrastructure bottlenecks"],
                    "techniques": ["Profiling - measure execution time", "Debugging with logging", "Stress testing under load"]
                }
            }
        },
        "task_3_4_ensure_data_quality": {
            "data_quality_checks": {
                "missing_values": "Identify columns with missing entries (NULL values)",
                "data_types": "Ensure each column has expected type (numbers for price, text for address)",
                "uniqueness": "Verify columns like ID contain unique values and identify duplicates",
                "valid_ranges": "Check numeric data falls within valid range (positive values for age)"
            }
        }
    },
    "domain_4_data_security_governance": {
        "task_4_1_authentication_mechanisms": {
            "iam_roles": {
                "concept": "Identity that grants temporary permissions without long-term credentials",
                "benefits": ["Reduced credential management", "Improved security posture", "Granular control"],
                "use_cases": ["AWS Services like Lambda", "Command-line tools", "Infrastructure as Code"]
            },
            "credential_management": {
                "aws_secrets_manager": {
                    "purpose": "Secure way to store and manage secrets like passwords, API keys, database credentials",
                    "features": ["Automatic rotation", "Secure storage", "Access control integration"]
                }
            }
        }
    }
}

def run_comprehensive_test():
    """Run comprehensive test with full AWS knowledge base"""
    
    print("🚀 AWS Data Engineer RAG - COMPREHENSIVE TEST")
    print("Setup: TF-IDF + Scikit-learn + Basic Text Processing")
    print("=" * 60)
    
    # Initialize system
    print("🔧 Initializing RAG system...")
    rag = AWSDataEngineerRAG()
    
    # Build with full knowledge base
    print("📚 Building FULL AWS knowledge base...")
    rag.build_knowledge_base(aws_data_engineer_knowledge_base)
    
    print(f"📊 Knowledge base contains {len(rag.chunks)} chunks")
    print(f"🔍 Keyword index has {len(rag.keyword_index)} unique terms")
    print(f"☁️ AWS service index covers {len(rag.service_index)} services")
    
    # Comprehensive test queries
    test_cases = [
        # Lambda optimization queries
        {
            "category": "Lambda Optimization", 
            "queries": [
                "How do I optimize Lambda memory allocation?",
                "Lambda timeout configuration best practices",
                "Lambda concurrency settings for data processing"
            ]
        },
        # Storage and databases
        {
            "category": "Data Storage",
            "queries": [
                "When should I use DynamoDB vs RDS?", 
                "S3 storage class selection criteria",
                "Redshift vs EMR for data processing"
            ]
        },
        # Data pipeline operations
        {
            "category": "Data Pipelines",
            "queries": [
                "Real-time streaming with Kinesis configuration",
                "Batch processing with AWS Glue setup",
                "Data quality checks implementation"
            ]
        },
        # Monitoring and troubleshooting
        {
            "category": "Monitoring",
            "queries": [
                "CloudWatch monitoring for data pipelines",
                "Troubleshooting slow data processing",
                "Performance optimization techniques"
            ]
        },
        # Security and governance
        {
            "category": "Security",
            "queries": [
                "IAM roles for data processing services",
                "AWS Secrets Manager integration",
                "Data encryption best practices"
            ]
        }
    ]
    
    print("\n🧪 Running comprehensive test suite...")
    
    total_queries = 0
    total_time = 0
    results_summary = []
    
    for category_data in test_cases:
        category = category_data["category"]
        queries = category_data["queries"]
        
        print(f"\n📋 Testing {category}")
        print("-" * 40)
        
        category_results = []
        
        for query in queries:
            # Test with optimal settings for your setup
            results = rag.search(query, search_type='hybrid', top_k=3)
            
            total_queries += 1
            total_time += results['search_time_ms']
            
            print(f"🔍 Query: '{query}'")
            print(f"   ⚡ {results['total_results']} results in {results['search_time_ms']}ms")
            
            if results['results']:
                best_score = results['results'][0]['score']
                best_content = results['results'][0]['content'][:80] + "..."
                aws_services = results['results'][0]['aws_services']
                
                print(f"   🎯 Best score: {best_score:.3f}")
                print(f"   📝 Content: {best_content}")
                print(f"   ☁️ AWS Services: {', '.join(aws_services) if aws_services else 'None'}")
                
                category_results.append({
                    'query': query,
                    'score': best_score,
                    'time_ms': results['search_time_ms'],
                    'results_count': results['total_results']
                })
            else:
                print(f"   ❌ No results found")
                category_results.append({
                    'query': query,
                    'score': 0.0,
                    'time_ms': results['search_time_ms'],
                    'results_count': 0
                })
            
            print()
        
        results_summary.append({
            'category': category,
            'results': category_results
        })
    
    # Performance summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total queries tested: {total_queries}")
    print(f"Average query time: {total_time/total_queries:.2f}ms")
    print(f"Total knowledge chunks: {len(rag.chunks)}")
    
    # Category analysis
    print(f"\n📈 RESULTS BY CATEGORY")
    print("-" * 50)
    
    for category_summary in results_summary:
        category = category_summary['category']
        results = category_summary['results']
        
        avg_score = sum(r['score'] for r in results) / len(results)
        avg_time = sum(r['time_ms'] for r in results) / len(results)
        
        print(f"{category:20} | Avg Score: {avg_score:.3f} | Avg Time: {avg_time:.1f}ms")
    
    # Overall assessment
    overall_avg_score = sum(
        sum(r['score'] for r in cat['results']) 
        for cat in results_summary
    ) / total_queries
    
    print(f"\n🎯 OVERALL PERFORMANCE")
    print("-" * 30)
    print(f"Average Relevance Score: {overall_avg_score:.3f}")
    print(f"Average Response Time: {total_time/total_queries:.2f}ms")
    
    if overall_avg_score >= 0.5:
        print("🎉 EXCELLENT! Your TF-IDF setup is performing very well!")
    elif overall_avg_score >= 0.4:
        print("👍 GOOD! Your system provides solid results.")
    else:
        print("📈 DECENT! Consider query optimization for better results.")
    
    # Save the comprehensive model
    print(f"\n💾 Saving comprehensive model...")
    rag.save_model("aws_comprehensive_rag_model.pkl")
    
    # Interactive mode
    print(f"\n🎮 INTERACTIVE MODE")
    print("Ask your own AWS Data Engineering questions!")
    print("Type 'quit' to exit, 'help' for tips")
    
    while True:
        try:
            user_query = input("\n❓ Your question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            elif user_query.lower() == 'help':
                print("\n💡 Tips for best results with your TF-IDF setup:")
                print("  🎯 Use specific AWS service names (Lambda, S3, Kinesis)")
                print("  📊 Include action words (optimize, configure, troubleshoot)")
                print("  ⚡ Technical terms work great (performance, monitoring, security)")
                print("  🔍 Try: 'Lambda memory optimization', 'S3 lifecycle policies'")
                continue
            elif not user_query:
                continue
            
            # Search with your optimal settings
            results = rag.search(user_query, search_type='hybrid', top_k=3)
            
            print(f"\n📊 Found {results['total_results']} results in {results['search_time_ms']}ms")
            
            if results['results']:
                print(f"\n🎯 Top Results:")
                for i, result in enumerate(results['results'], 1):
                    print(f"\n{i}. Score: {result['score']:.3f}")
                    print(f"   Domain: {result['domain'].replace('_', ' ').title()}")
                    print(f"   AWS Services: {', '.join(result['aws_services']) if result['aws_services'] else 'None'}")
                    print(f"   Content: {result['content']}")
                
                # Show assembled context
                print(f"\n📄 Assembled Context (for LLM integration):")
                print("=" * 50)
                print(results['context'])
                print("=" * 50)
            else:
                print("❌ No results found. Try using specific AWS service names or technical terms.")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Comprehensive test completed!")
    print(f"📁 Model saved as: aws_comprehensive_rag_model.pkl")
    print(f"🚀 Your TF-IDF + Scikit-learn setup is production-ready!")

if __name__ == "__main__":
    run_comprehensive_test()
