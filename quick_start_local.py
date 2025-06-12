"""
Quick Start Script for Bank Churn MLOps Pipeline (Local-only mode)
This script demonstrates the complete pipeline without MLflow server dependency
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

# Import our modules
from src.data_preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_start_demo_local():
    """
    Run a complete demonstration of the MLOps pipeline without MLflow dependency
    """
    logger.info("Starting Bank Churn MLOps Pipeline Quick Demo (Local Mode)")
    logger.info("=" * 60)
    
    # Check if dataset exists
    data_path = 'data/Bank Customer Churn Prediction.csv'
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        logger.info("Please place your dataset in the data/ directory and try again.")
        return False
    
    try:
        # Step 1: Data Preprocessing
        logger.info("Step 1: Data Preprocessing")
        logger.info("-" * 30)
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            data_path, test_size=0.2, random_state=42
        )
        
        logger.info(f"Preprocessing completed:")
        logger.info(f"   Training set: {X_train.shape}")
        logger.info(f"   Test set: {X_test.shape}")
        logger.info(f"   Features: {len(X_train.columns)}")
        logger.info(f"   Churn rate (train): {y_train.mean():.2%}")
        logger.info(f"   Churn rate (test): {y_test.mean():.2%}")
        
        # Step 2: Model Training (without MLflow)
        logger.info("\nStep 2: Model Training (Local Mode)")
        logger.info("-" * 30)
        
        # Train Random Forest model directly
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_prob)
        }
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 5 Most Important Features:")
        for _, row in feature_importance.head(5).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Step 3: Model Evaluation
        logger.info("\nStep 3: Model Evaluation")
        logger.info("-" * 30)
        
        logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"   {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Step 4: Save Model
        logger.info("\nStep 4: Model Saving")
        logger.info("-" * 30)
        
        os.makedirs('models', exist_ok=True)
        model_path = 'models/quick_start_model_local.pkl'
        joblib.dump(model, model_path)
        
        # Also save preprocessor
        preprocessor_path = 'models/preprocessor_local.pkl'
        joblib.dump(preprocessor, preprocessor_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Step 5: Model Inference Demo
        logger.info("\nStep 5: Model Inference Demo")
        logger.info("-" * 30)
        
        # Create sample customers for prediction
        sample_customers = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'credit_score': [450, 750, 650],
            'age': [55, 25, 35],
            'tenure': [1, 8, 5],
            'balance': [0, 150000, 80000],
            'products_number': [1, 2, 1],
            'estimated_salary': [30000, 120000, 70000],
            'country': ['Germany', 'France', 'Spain'],
            'gender': ['Female', 'Male', 'Female'],
            'credit_card': [0, 1, 1],
            'active_member': [0, 1, 1]
        })
        
        logger.info("Creating predictions for sample customers...")
        
        # Preprocess sample data
        sample_processed = preprocessor.preprocess_inference_data(sample_customers)
        
        # Make predictions
        predictions = model.predict(sample_processed)
        probabilities = model.predict_proba(sample_processed)[:, 1]
        
        logger.info("\nPrediction Results:")
        for i in range(len(sample_customers)):
            customer_id = sample_customers.iloc[i]['customer_id']
            prediction = "Churn" if predictions[i] == 1 else "No Churn"
            probability = probabilities[i]
            
            if probability > 0.7:
                risk_level = "High Risk"
            elif probability > 0.3:
                risk_level = "Medium Risk"
            else:
                risk_level = "Low Risk"
            
            logger.info(f"   Customer {customer_id}: {prediction} (Risk: {risk_level}, Prob: {probability:.2%})")
        
        # Step 6: Create Summary Report
        logger.info("\nStep 6: Summary Report")
        logger.info("-" * 30)
        
        # Create results directory
        os.makedirs('results/reports', exist_ok=True)
        
        # Generate summary report
        report = {
            'demo_timestamp': datetime.now().isoformat(),
            'mode': 'local_only',
            'dataset_info': {
                'total_samples': len(X_train) + len(X_test),
                'features': len(X_train.columns),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'overall_churn_rate': (len(y_train[y_train == 1]) + len(y_test[y_test == 1])) / (len(y_train) + len(y_test))
            },
            'model_performance': metrics,
            'top_features': feature_importance.head(10).to_dict('records'),
            'sample_predictions': {
                'customer_ids': sample_customers['customer_id'].tolist(),
                'predictions': ["Churn" if p == 1 else "No Churn" for p in predictions],
                'probabilities': probabilities.tolist()
            }
        }
        
        # Save report
        import json
        report_path = f"results/reports/quick_start_local_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Demo report saved to {report_path}")
        
        # Final summary
        logger.info("\nQuick Start Demo Complete!")
        logger.info("=" * 60)
        logger.info("Summary of what was accomplished:")
        logger.info("- Data preprocessing and feature engineering")
        logger.info("- Random Forest model training")
        logger.info("- Model evaluation and performance metrics")
        logger.info("- Feature importance analysis")
        logger.info("- Model inference on sample customers")
        logger.info("- Comprehensive reporting")
        logger.info("\nNext steps:")
        logger.info("1. Set up Docker services for full MLOps pipeline")
        logger.info("2. Configure Airflow for automated model training")
        logger.info("3. Set up MLflow for experiment tracking")
        logger.info("4. Customize the model and features for your needs")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Bank Customer Churn Prediction - Quick Start Demo (Local Mode)")
    print("=" * 60)
    
    # Run the demo
    success = quick_start_demo_local()
    
    if success:
        print("\nDemo completed successfully!")
        print("Check the results/ directory for outputs.")
        print("Your ML pipeline is working perfectly!")
        print("\nReady for Docker setup with full Airflow + MLflow!")
    else:
        print("\nDemo failed. Check the logs above for details.")
        sys.exit(1)