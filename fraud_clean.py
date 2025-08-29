"""
Pharmacy Claims Fraud Detection System
======================================
Advanced ML-based fraud detection for pharmacy billing claims with SQL Server integration
"""

import pandas as pd
import numpy as np
import pyodbc
import warnings
import logging
from datetime import datetime, timedelta
import csv
from typing import List, Dict, Tuple, Any
import os
from dataclasses import dataclass

# Machine Learning imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Statistical analysis
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Environment variables and OpenAI integration
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("python-dotenv package not installed. Install with: pip install python-dotenv")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not installed. Install with: pip install openai")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class FraudAlert:
    """Data class for fraud alerts"""
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    member_id: str
    description: str
    evidence: Dict[str, Any]
    confidence_score: float
    recommendation: str

class PharmacyFraudDetector:
    """Advanced fraud detection system for pharmacy claims"""
    
    def __init__(self, server_name: str = "JONESFAMILYPC3", database: str = "PRO_SSRS", env_file_path: str = None):
        self.server_name = server_name
        self.database = database
        self.connection_string = f"""
            DRIVER={{ODBC Driver 17 for SQL Server}};
            SERVER={server_name};
            DATABASE={database};
            Trusted_Connection=yes;
        """
        self.data = None
        self.fraud_alerts = []
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Load environment variables
        self._load_environment_variables(env_file_path)
        
        # Initialize OpenAI if available
        self._initialize_openai()
    
    def _load_environment_variables(self, env_file_path: str = None):
        """Load environment variables from .env file"""
        if DOTENV_AVAILABLE:
            if env_file_path:
                # Load from specific path
                load_dotenv(env_file_path)
                logging.info(f"Loaded environment variables from: {env_file_path}")
            else:
                # Try common locations
                possible_paths = [
                    "test.env",
                    ".env",
                    r"C:\Users\pjone\OneDrive\Desktop\NewPython\test.env",
                    os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "NewPython", "test.env")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        load_dotenv(path)
                        logging.info(f"Loaded environment variables from: {path}")
                        break
                else:
                    logging.warning("No .env file found in common locations")
        else:
            logging.warning("python-dotenv not available. Install with: pip install python-dotenv")
    
    def _initialize_openai(self):
        """Initialize OpenAI with API key from environment"""
        if OPENAI_AVAILABLE:
            # Try to get API key from environment variables
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    # For OpenAI v1.0+ (current version)
                    from openai import OpenAI
                    # The API key will be used when creating the client in generate_openai_analysis
                    logging.info("OpenAI v1.0+ detected - API key loaded successfully from environment")
                except ImportError:
                    # Fallback for older versions
                    openai.api_key = api_key
                    logging.info("OpenAI legacy version - API key loaded successfully from environment")
            else:
                logging.warning("OPENAI_API_KEY not found in environment variables")
        else:
            logging.warning("OpenAI package not available")
    
    def connect_to_database(self) -> pd.DataFrame:
        """Connect to SQL Server and retrieve pharmacy claims data"""
        try:
            logging.info("Connecting to SQL Server...")
            conn = pyodbc.connect(self.connection_string)
            
            query = """
            SELECT 
                prescription_id,
                claim_id,
                member_id,
                provider_id,
                pharmacy_id,
                ndc_code,
                drug_name,
                drug_strength,
                dosage_form,
                quantity,
                days_supply,
                refill_number,
                prescription_date,
                fill_date,
                billed_amount,
                paid_amount,
                copay_amount
            FROM [dbo].[pharmacy_claims]
            WHERE fill_date >= DATEADD(year, -2, GETDATE())
            ORDER BY fill_date DESC
            """
            
            self.data = pd.read_sql(query, conn)
            conn.close()
            
            logging.info(f"Successfully loaded {len(self.data)} records from database")
            return self.data
            
        except Exception as e:
            logging.error(f"Database connection failed: {str(e)}")
            raise
    
    def load_sample_data(self, csv_path: str = "medication.csv") -> pd.DataFrame:
        """Load sample data from CSV for testing"""
        try:
            self.data = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(self.data)} records from CSV file")
            return self.data
        except Exception as e:
            logging.error(f"Failed to load CSV: {str(e)}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """Clean and preprocess the data for analysis"""
        if self.data is None:
            raise ValueError("No data loaded. Call connect_to_database() or load_sample_data() first.")
        
        logging.info("Preprocessing data...")
        
        # Convert date columns
        self.data['prescription_date'] = pd.to_datetime(self.data['prescription_date'])
        self.data['fill_date'] = pd.to_datetime(self.data['fill_date'])
        
        # Create derived features
        self.data['days_to_fill'] = (self.data['fill_date'] - self.data['prescription_date']).dt.days
        self.data['paid_ratio'] = self.data['paid_amount'] / self.data['billed_amount']
        self.data['copay_ratio'] = self.data['copay_amount'] / self.data['billed_amount']
        self.data['fill_month'] = self.data['fill_date'].dt.to_period('M')
        self.data['fill_year'] = self.data['fill_date'].dt.year
        self.data['fill_weekday'] = self.data['fill_date'].dt.dayofweek
        
        # Create drug category based on common patterns
        self.data['is_controlled'] = self.data['drug_name'].str.contains(
            'Oxycodone|OxyContin|Hydrocodone|Vicodin|Fentanyl|Methadone|Tramadol|MS Contin',
            case=False, na=False
        )
        
        # Calculate per-unit costs
        self.data['cost_per_day'] = self.data['billed_amount'] / self.data['days_supply']
        self.data['cost_per_unit'] = self.data['billed_amount'] / self.data['quantity']
        
        logging.info("Data preprocessing completed")
        return self.data
    
    def detect_duplicate_prescriptions(self) -> List[FraudAlert]:
        """Detect duplicate prescriptions for the same member in the same month"""
        logging.info("Analyzing duplicate prescriptions...")
        alerts = []
        
        # Group by member, drug, strength, and month
        grouped = self.data.groupby([
            'member_id', 'drug_name', 'drug_strength', 'fill_month'
        ]).agg({
            'prescription_id': 'count',
            'billed_amount': ['sum', 'mean'],
            'paid_amount': 'sum',
            'pharmacy_id': lambda x: list(x.unique()),
            'fill_date': lambda x: list(x)
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() if col[1] else col[0] 
                          for col in grouped.columns.values]
        
        # Find duplicates (more than 1 prescription per month)
        duplicates = grouped[grouped['prescription_id_count'] > 1]
        
        for _, row in duplicates.iterrows():
            severity = "CRITICAL" if row['prescription_id_count'] > 2 else "HIGH"
            
            alert = FraudAlert(
                alert_type="DUPLICATE_PRESCRIPTION",
                severity=severity,
                member_id=str(row['member_id']),
                description=f"Member filled {row['prescription_id_count']} prescriptions for {row['drug_name']} {row['drug_strength']} in {row['fill_month']}",
                evidence={
                    'prescription_count': row['prescription_id_count'],
                    'total_billed': row['billed_amount_sum'],
                    'avg_billed': row['billed_amount_mean'],
                    'pharmacies': row['pharmacy_id_<lambda>'],
                    'fill_dates': row['fill_date_<lambda>']
                },
                confidence_score=0.95,
                recommendation="Investigate member for potential prescription fraud or doctor shopping"
            )
            alerts.append(alert)
        
        logging.info(f"Found {len(alerts)} duplicate prescription alerts")
        return alerts
    
    def detect_billing_outliers(self) -> List[FraudAlert]:
        """Detect unusual billing amounts using statistical methods"""
        logging.info("Analyzing billing outliers...")
        alerts = []
        
        # Calculate outliers by drug type
        for drug_group in self.data.groupby(['drug_name', 'drug_strength']):
            drug_name, drug_strength = drug_group[0]
            group_data = drug_group[1]
            
            if len(group_data) < 5:  # Skip if too few samples
                continue
            
            # Calculate Z-score for billed amounts
            mean_cost = group_data['billed_amount'].mean()
            std_cost = group_data['billed_amount'].std()
            
            if std_cost == 0:  # Skip if no variation
                continue
            
            # Find outliers (Z-score > 2.5)
            outliers = group_data[
                np.abs((group_data['billed_amount'] - mean_cost) / std_cost) > 2.5
            ]
            
            for _, row in outliers.iterrows():
                z_score = abs((row['billed_amount'] - mean_cost) / std_cost)
                severity = "CRITICAL" if z_score > 3.5 else "HIGH"
                
                alert = FraudAlert(
                    alert_type="BILLING_OUTLIER",
                    severity=severity,
                    member_id=str(row['member_id']),
                    description=f"Unusual billing amount for {drug_name} {drug_strength}: ${row['billed_amount']:.2f} (expected: ${mean_cost:.2f})",
                    evidence={
                        'billed_amount': row['billed_amount'],
                        'expected_amount': mean_cost,
                        'z_score': z_score,
                        'pharmacy_id': row['pharmacy_id'],
                        'provider_id': row['provider_id']
                    },
                    confidence_score=min(0.9, z_score / 4.0),
                    recommendation="Review pharmacy pricing and provider prescribing patterns"
                )
                alerts.append(alert)
        
        logging.info(f"Found {len(alerts)} billing outlier alerts")
        return alerts
    
    def detect_unusual_payment_ratios(self) -> List[FraudAlert]:
        """Detect unusual paid/billed ratios"""
        logging.info("Analyzing payment ratios...")
        alerts = []
        
        # Calculate normal payment ratio statistics
        mean_ratio = self.data['paid_ratio'].mean()
        std_ratio = self.data['paid_ratio'].std()
        
        # Find unusual ratios (more than 2 standard deviations from mean)
        unusual = self.data[
            np.abs(self.data['paid_ratio'] - mean_ratio) > 2 * std_ratio
        ]
        
        for _, row in unusual.iterrows():
            deviation = abs(row['paid_ratio'] - mean_ratio) / std_ratio
            severity = "HIGH" if deviation > 3 else "MEDIUM"
            
            alert = FraudAlert(
                alert_type="UNUSUAL_PAYMENT_RATIO",
                severity=severity,
                member_id=str(row['member_id']),
                description=f"Unusual payment ratio for {row['drug_name']}: {row['paid_ratio']:.1%} (expected: {mean_ratio:.1%})",
                evidence={
                    'paid_ratio': row['paid_ratio'],
                    'expected_ratio': mean_ratio,
                    'deviation_score': deviation,
                    'billed_amount': row['billed_amount'],
                    'paid_amount': row['paid_amount'],
                    'pharmacy_id': row['pharmacy_id']
                },
                confidence_score=min(0.8, deviation / 3.0),
                recommendation="Investigate payment processing and insurance coordination"
            )
            alerts.append(alert)
        
        logging.info(f"Found {len(alerts)} unusual payment ratio alerts")
        return alerts
    
    def detect_pharmacy_patterns(self) -> List[FraudAlert]:
        """Detect suspicious pharmacy patterns"""
        logging.info("Analyzing pharmacy patterns...")
        alerts = []
        
        # Analyze each pharmacy
        pharmacy_stats = self.data.groupby('pharmacy_id').agg({
            'billed_amount': ['mean', 'std', 'count'],
            'paid_ratio': 'mean',
            'is_controlled': 'sum',
            'member_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        pharmacy_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                 for col in pharmacy_stats.columns.values]
        
        # Calculate overall statistics for comparison
        overall_mean_cost = self.data['billed_amount'].mean()
        overall_mean_ratio = self.data['paid_ratio'].mean()
        overall_controlled_pct = self.data['is_controlled'].mean()
        
        for _, pharmacy in pharmacy_stats.iterrows():
            suspicious_indicators = []
            risk_score = 0
            
            # Check for high average billing
            if pharmacy['billed_amount_mean'] > overall_mean_cost * 1.5:
                suspicious_indicators.append("High average billing amounts")
                risk_score += 0.3
            
            # Check for unusual payment ratios
            if abs(pharmacy['paid_ratio_mean'] - overall_mean_ratio) > 0.1:
                suspicious_indicators.append("Unusual payment ratios")
                risk_score += 0.2
            
            # Check for high controlled substance percentage
            controlled_pct = pharmacy['is_controlled_sum'] / pharmacy['billed_amount_count']
            if controlled_pct > overall_controlled_pct * 2:
                suspicious_indicators.append("High controlled substance dispensing")
                risk_score += 0.4
            
            # Check for low patient diversity (pill mill indicator)
            avg_claims_per_patient = pharmacy['billed_amount_count'] / pharmacy['member_id_nunique']
            if avg_claims_per_patient > 5:
                suspicious_indicators.append("Low patient diversity (potential pill mill)")
                risk_score += 0.5
            
            if risk_score > 0.4:  # Threshold for alerting
                severity = "CRITICAL" if risk_score > 0.7 else "HIGH" if risk_score > 0.5 else "MEDIUM"
                
                alert = FraudAlert(
                    alert_type="SUSPICIOUS_PHARMACY",
                    severity=severity,
                    member_id="N/A",
                    description=f"Pharmacy {pharmacy['pharmacy_id']} shows suspicious patterns: {', '.join(suspicious_indicators)}",
                    evidence={
                        'pharmacy_id': pharmacy['pharmacy_id'],
                        'avg_billing': pharmacy['billed_amount_mean'],
                        'avg_paid_ratio': pharmacy['paid_ratio_mean'],
                        'controlled_substance_pct': controlled_pct,
                        'total_claims': pharmacy['billed_amount_count'],
                        'unique_patients': pharmacy['member_id_nunique'],
                        'risk_score': risk_score
                    },
                    confidence_score=risk_score,
                    recommendation="Conduct detailed audit of pharmacy operations and prescriber relationships"
                )
                alerts.append(alert)
        
        logging.info(f"Found {len(alerts)} suspicious pharmacy alerts")
        return alerts
    
    def train_ml_models(self) -> Dict[str, Any]:
        """Train machine learning models for fraud prediction"""
        logging.info("Training ML models...")
        
        # Prepare features for ML
        features = ['quantity', 'days_supply', 'billed_amount', 'paid_amount', 
                   'copay_amount', 'days_to_fill', 'paid_ratio', 'copay_ratio',
                   'cost_per_day', 'cost_per_unit', 'fill_weekday']
        
        # Create synthetic labels based on our fraud detection rules
        # In real scenario, you'd have labeled fraud cases
        fraud_labels = np.zeros(len(self.data))
        
        # Mark known fraud patterns as positive cases
        duplicate_masks = self.data.duplicated(subset=['member_id', 'drug_name', 'fill_month'], keep=False)
        billing_outliers = np.abs(stats.zscore(self.data['billed_amount'])) > 2.5
        ratio_outliers = np.abs(stats.zscore(self.data['paid_ratio'])) > 2
        
        fraud_labels[duplicate_masks | billing_outliers | ratio_outliers] = 1
        
        # Prepare data
        X = self.data[features].fillna(0)
        y = fraud_labels
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        models = {}
        
        # 1. Isolation Forest for anomaly detection
        models['isolation_forest'] = IsolationForest(
            contamination=0.1, random_state=42, n_estimators=100
        )
        models['isolation_forest'].fit(X_train)
        
        # 2. Random Forest for classification
        models['random_forest'] = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        models['random_forest'].fit(X_train, y_train)
        
        # 3. DBSCAN for clustering-based anomaly detection
        models['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
        models['dbscan'].fit(X_train)
        
        self.models = models
        
        # Evaluate Random Forest
        y_pred = models['random_forest'].predict(X_test)
        logging.info("Random Forest Classification Report:")
        logging.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Save models
        joblib.dump(models, 'fraud_detection_models.pkl')
        joblib.dump(self.scalers, 'fraud_detection_scalers.pkl')
        
        logging.info("ML models trained and saved successfully")
        return models
    
    def predict_fraud_risk(self, new_data: pd.DataFrame = None) -> List[FraudAlert]:
        """Use trained models to predict fraud risk for new claims"""
        if new_data is None:
            new_data = self.data
        
        logging.info("Predicting fraud risk using ML models...")
        alerts = []
        
        if not self.models:
            logging.warning("No trained models found. Training models first...")
            self.train_ml_models()
        
        features = ['quantity', 'days_supply', 'billed_amount', 'paid_amount', 
                   'copay_amount', 'days_to_fill', 'paid_ratio', 'copay_ratio',
                   'cost_per_day', 'cost_per_unit', 'fill_weekday']
        
        X = new_data[features].fillna(0)
        X_scaled = self.scalers['standard'].transform(X)
        
        # Get predictions from different models
        isolation_pred = self.models['isolation_forest'].predict(X_scaled)
        rf_pred = self.models['random_forest'].predict(X_scaled)
        rf_proba = self.models['random_forest'].predict_proba(X_scaled)
        
        # Combine predictions
        for i, (_, row) in enumerate(new_data.iterrows()):
            risk_score = 0
            risk_factors = []
            
            # Isolation Forest prediction
            if isolation_pred[i] == -1:  # Anomaly
                risk_score += 0.4
                risk_factors.append("Anomalous transaction pattern")
            
            # Random Forest prediction
            if rf_pred[i] == 1:  # Fraud
                fraud_probability = rf_proba[i][1]
                risk_score += fraud_probability * 0.6
                risk_factors.append(f"High fraud probability ({fraud_probability:.1%})")
            
            # Create alert if risk score is significant
            if risk_score > 0.5:
                severity = "CRITICAL" if risk_score > 0.8 else "HIGH" if risk_score > 0.6 else "MEDIUM"
                
                alert = FraudAlert(
                    alert_type="ML_FRAUD_PREDICTION",
                    severity=severity,
                    member_id=str(row['member_id']),
                    description=f"ML models predict high fraud risk: {', '.join(risk_factors)}",
                    evidence={
                        'risk_score': risk_score,
                        'isolation_forest_anomaly': isolation_pred[i] == -1,
                        'fraud_probability': rf_proba[i][1] if rf_pred[i] == 1 else 0,
                        'prescription_id': row['prescription_id'],
                        'drug_name': row['drug_name'],
                        'billed_amount': row['billed_amount']
                    },
                    confidence_score=risk_score,
                    recommendation="Priority review recommended based on ML analysis"
                )
                alerts.append(alert)
        
        logging.info(f"ML models generated {len(alerts)} fraud risk alerts")
        return alerts
    
    def generate_openai_analysis(self, alerts: List[FraudAlert]) -> str:
        """Use OpenAI to generate detailed fraud analysis and recommendations"""
        if not OPENAI_AVAILABLE:
            return "OpenAI integration not available. Install openai package and set API key."
        
        # Check for OpenAI client initialization
        try:
            # For OpenAI v1.0+ (current version)
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            # Fallback for older versions
            if not hasattr(openai, 'api_key') or not openai.api_key:
                return "OpenAI API key not configured. Set openai.api_key or OPENAI_API_KEY environment variable."
        
        logging.info("Generating OpenAI analysis...")
        
        # Prepare summary of alerts for OpenAI
        alert_summary = []
        for alert in alerts[:10]:  # Limit to top 10 alerts
            alert_summary.append({
                'type': alert.alert_type,
                'severity': alert.severity,
                'member': alert.member_id,
                'description': alert.description,
                'confidence': alert.confidence_score
            })
        
        prompt = f"""
        As a healthcare fraud expert, analyze the following pharmacy billing fraud alerts and provide:
        1. Risk assessment and prioritization
        2. Specific investigative steps
        3. Potential financial impact
        4. Prevention recommendations
        
        Fraud Alerts Summary:
        {alert_summary}
        
        Total alerts: {len(alerts)}
        
        Please provide a comprehensive analysis focusing on the most critical risks and actionable recommendations.
        """
        
        try:
            # Try new OpenAI v1.0+ API first
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert healthcare fraud investigator with deep knowledge of pharmacy billing patterns and fraud detection."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content
                logging.info("OpenAI analysis generated successfully (v1.0+ API)")
                return analysis
                
            except ImportError:
                # Fallback to legacy API
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert healthcare fraud investigator with deep knowledge of pharmacy billing patterns and fraud detection."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content
                logging.info("OpenAI analysis generated successfully (legacy API)")
                return analysis
            
        except Exception as e:
            logging.error(f"OpenAI analysis failed: {str(e)}")
            return f"OpenAI analysis failed: {str(e)}"
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete fraud detection analysis"""
        logging.info("Starting comprehensive fraud detection analysis...")
        
        # Load and preprocess data
        if self.data is None:
            try:
                self.connect_to_database()
            except:
                logging.warning("Database connection failed, using sample data...")
                self.load_sample_data()
        
        self.preprocess_data()
        
        # Run all fraud detection methods
        all_alerts = []
        
        # Traditional rule-based detection
        all_alerts.extend(self.detect_duplicate_prescriptions())
        all_alerts.extend(self.detect_billing_outliers())
        all_alerts.extend(self.detect_unusual_payment_ratios())
        all_alerts.extend(self.detect_pharmacy_patterns())
        
        # ML-based detection
        ml_alerts = self.predict_fraud_risk()
        all_alerts.extend(ml_alerts)
        
        # Store alerts
        self.fraud_alerts = all_alerts
        
        # Generate summary statistics
        summary = {
            'total_alerts': len(all_alerts),
            'critical_alerts': len([a for a in all_alerts if a.severity == "CRITICAL"]),
            'high_alerts': len([a for a in all_alerts if a.severity == "HIGH"]),
            'medium_alerts': len([a for a in all_alerts if a.severity == "MEDIUM"]),
            'alert_types': {}
        }
        
        # Count by alert type
        for alert in all_alerts:
            alert_type = alert.alert_type
            if alert_type not in summary['alert_types']:
                summary['alert_types'][alert_type] = 0
            summary['alert_types'][alert_type] += 1
        
        # Generate OpenAI analysis if available
        openai_analysis = self.generate_openai_analysis(all_alerts)
        
        results = {
            'summary': summary,
            'alerts': all_alerts,
            'openai_analysis': openai_analysis,
            'data_stats': {
                'total_claims': len(self.data),
                'date_range': f"{self.data['fill_date'].min()} to {self.data['fill_date'].max()}",
                'total_billed': self.data['billed_amount'].sum(),
                'total_paid': self.data['paid_amount'].sum(),
                'unique_members': self.data['member_id'].nunique(),
                'unique_pharmacies': self.data['pharmacy_id'].nunique()
            }
        }
        
        logging.info(f"Analysis complete. Found {len(all_alerts)} total alerts.")
        return results
    
    def export_results_to_csv(self, results: Dict[str, Any], filename: str = None) -> str:
        """Export fraud detection results to CSV files"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fraud_analysis_{timestamp}"
        
        # Export main alerts
        alerts_data = []
        for alert in results['alerts']:
            alerts_data.append({
                'Alert_Type': alert.alert_type,
                'Severity': alert.severity,
                'Member_ID': alert.member_id,
                'Description': alert.description,
                'Confidence_Score': alert.confidence_score,
                'Recommendation': alert.recommendation,
                'Evidence': str(alert.evidence)
            })
        
        alerts_df = pd.DataFrame(alerts_data)
        alerts_filename = f"{filename}_alerts.csv"
        alerts_df.to_csv(alerts_filename, index=False)
        
        # Export summary report
        summary_data = [
            ['Metric', 'Value'],
            ['Total Claims Analyzed', results['data_stats']['total_claims']],
            ['Date Range', results['data_stats']['date_range']],
            ['Total Billed Amount', f"${results['data_stats']['total_billed']:,.2f}"],
            ['Total Paid Amount', f"${results['data_stats']['total_paid']:,.2f}"],
            ['Unique Members', results['data_stats']['unique_members']],
            ['Unique Pharmacies', results['data_stats']['unique_pharmacies']],
            ['', ''],
            ['FRAUD ALERTS SUMMARY', ''],
            ['Total Alerts', results['summary']['total_alerts']],
            ['Critical Alerts', results['summary']['critical_alerts']],
            ['High Priority Alerts', results['summary']['high_alerts']],
            ['Medium Priority Alerts', results['summary']['medium_alerts']],
            ['', ''],
            ['ALERT TYPES', '']
        ]
        
        for alert_type, count in results['summary']['alert_types'].items():
            summary_data.append([alert_type, count])
        
        if results['openai_analysis']:
            summary_data.extend([
                ['', ''],
                ['AI ANALYSIS', ''],
                ['OpenAI Recommendation', results['openai_analysis']]
            ])
        
        summary_filename = f"{filename}_summary.csv"
        with open(summary_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(summary_data)
        
        # Export high-risk members for investigation
        high_risk_members = {}
        for alert in results['alerts']:
            if alert.severity in ['CRITICAL', 'HIGH']:
                member_id = alert.member_id
                if member_id not in high_risk_members:
                    high_risk_members[member_id] = {
                        'member_id': member_id,
                        'alert_count': 0,
                        'max_severity': '',
                        'alert_types': [],
                        'total_risk_score': 0
                    }
                
                high_risk_members[member_id]['alert_count'] += 1
                high_risk_members[member_id]['alert_types'].append(alert.alert_type)
                high_risk_members[member_id]['total_risk_score'] += alert.confidence_score
                
                if alert.severity == 'CRITICAL' or high_risk_members[member_id]['max_severity'] != 'CRITICAL':
                    high_risk_members[member_id]['max_severity'] = alert.severity
        
        if high_risk_members:
            members_df = pd.DataFrame(list(high_risk_members.values()))
            members_df['alert_types'] = members_df['alert_types'].apply(lambda x: ', '.join(set(x)))
            members_df = members_df.sort_values('total_risk_score', ascending=False)
            
            members_filename = f"{filename}_high_risk_members.csv"
            members_df.to_csv(members_filename, index=False)
        
        logging.info(f"Results exported to {alerts_filename}, {summary_filename}")
        return alerts_filename
    
    def generate_dashboard_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for fraud detection dashboard"""
        dashboard_data = {
            'kpis': {
                'total_alerts': results['summary']['total_alerts'],
                'fraud_rate': (results['summary']['total_alerts'] / results['data_stats']['total_claims']) * 100,
                'potential_loss': sum([
                    float(str(alert.evidence.get('billed_amount', 0)).replace(',', ''))
                    for alert in results['alerts'] 
                    if 'billed_amount' in alert.evidence
                ]),
                'high_risk_pharmacies': len([
                    alert for alert in results['alerts'] 
                    if alert.alert_type == 'SUSPICIOUS_PHARMACY'
                ])
            },
            'alert_distribution': results['summary']['alert_types'],
            'severity_distribution': {
                'CRITICAL': results['summary']['critical_alerts'],
                'HIGH': results['summary']['high_alerts'],
                'MEDIUM': results['summary']['medium_alerts']
            },
            'monthly_trends': self._calculate_monthly_trends(),
            'top_risks': [
                {
                    'member_id': alert.member_id,
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'description': alert.description,
                    'confidence': alert.confidence_score
                }
                for alert in sorted(results['alerts'], 
                                  key=lambda x: x.confidence_score, reverse=True)[:10]
            ]
        }
        return dashboard_data
    
    def _calculate_monthly_trends(self) -> Dict[str, List]:
        """Calculate monthly fraud trends"""
        if self.data is None:
            return {}
        
        monthly_stats = self.data.groupby('fill_month').agg({
            'billed_amount': ['count', 'sum', 'mean'],
            'paid_ratio': 'mean'
        }).reset_index()
        
        return {
            'months': [str(month) for month in monthly_stats['fill_month']],
            'claim_counts': monthly_stats['billed_amount']['count'].tolist(),
            'total_billed': monthly_stats['billed_amount']['sum'].tolist(),
            'avg_paid_ratio': monthly_stats['paid_ratio']['mean'].tolist()
        }


def main():
    """Main execution function"""
    # Initialize fraud detector with automatic .env loading
    detector = PharmacyFraudDetector(
        env_file_path=r"C:\Users\pjone\OneDrive\Desktop\NewPython\test.env"
    )
    
    # Run comprehensive analysis
    try:
        results = detector.run_comprehensive_analysis()
        
        # Print summary
        print("\n" + "="*60)
        print("PHARMACY FRAUD DETECTION ANALYSIS COMPLETE")
        print("="*60)
        
        print(f"\nDATA SUMMARY:")
        print(f"- Total Claims Analyzed: {results['data_stats']['total_claims']:,}")
        print(f"- Date Range: {results['data_stats']['date_range']}")
        print(f"- Total Billed: ${results['data_stats']['total_billed']:,.2f}")
        print(f"- Total Paid: ${results['data_stats']['total_paid']:,.2f}")
        print(f"- Unique Members: {results['data_stats']['unique_members']:,}")
        print(f"- Unique Pharmacies: {results['data_stats']['unique_pharmacies']:,}")
        
        print(f"\nFRAUD ALERTS:")
        print(f"- Total Alerts: {results['summary']['total_alerts']}")
        print(f"- Critical: {results['summary']['critical_alerts']}")
        print(f"- High Priority: {results['summary']['high_alerts']}")
        print(f"- Medium Priority: {results['summary']['medium_alerts']}")
        
        print(f"\nALERT BREAKDOWN:")
        for alert_type, count in results['summary']['alert_types'].items():
            print(f"- {alert_type}: {count}")
        
        # Export results
        csv_filename = detector.export_results_to_csv(results)
        print(f"\nResults exported to CSV files starting with: {csv_filename.replace('_alerts.csv', '')}")
        
        # Show top 5 critical alerts
        critical_alerts = [a for a in results['alerts'] if a.severity == 'CRITICAL']
        if critical_alerts:
            print(f"\nTOP CRITICAL ALERTS:")
            for i, alert in enumerate(critical_alerts[:5], 1):
                print(f"{i}. {alert.description}")
                print(f"   Member: {alert.member_id}, Confidence: {alert.confidence_score:.1%}")
                print(f"   Recommendation: {alert.recommendation}\n")
        
        # Show OpenAI analysis if available
        if results['openai_analysis'] and "not available" not in results['openai_analysis'].lower():
            print("\nAI ANALYSIS:")
            print("-" * 40)
            print(results['openai_analysis'])
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()


# Example usage for integration:

# Basic usage with automatic .env loading
detector = PharmacyFraudDetector(
    server_name="JONESFAMILYPC3", 
    database="PRO_SSRS",
    env_file_path=r"C:\Users\pjone\OneDrive\Desktop\NewPython\test.env"
)

# Or let it auto-detect the .env file
detector = PharmacyFraudDetector()

# Run analysis
results = detector.run_comprehensive_analysis()

# Export to CSV
detector.export_results_to_csv(results, "fraud_analysis_2024")

# Get dashboard data for visualization
dashboard_data = detector.generate_dashboard_data(results)
