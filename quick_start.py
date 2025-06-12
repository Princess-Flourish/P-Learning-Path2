"""
Quick Start Script for Bank Churn MLOps Pipeline
This script demonstrates the complete pipeline without Docker for quick testing
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

# Import our modules
from src.data_preprocessing import DataPreprocessor
from src.model_training import ChurnModelTrainer
from src.model_inference import ChurnModelInference

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_start_demo():
    """
    Run a complete demonstration of the MLOps pipeline
    """
    logger.info("Starting Bank Churn MLOps Pipeline Quick Demo")
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
        
        # Step 2: Model Training
        logger.info("\nStep 2: Model Training")
        logger.info("-" * 30)
        
        trainer = ChurnModelTrainer(experiment_name="quick_start_demo")
        
        # Train baseline model for speed
        logger.info("Training baseline Random Forest model...")
        model = trainer.train_baseline_model(X_train, y_train, X_test, y_test)
        
        # Get feature importance
        feature_importance = trainer.get_feature_importance(top_n=5)
        if feature_importance is not None:
            logger.info("\nTop 5 Most Important Features:")
            for _, row in feature_importance.iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Step 3: Model Evaluation
        logger.info("\nStep 3: Model Evaluation")
        logger.info("-" * 30)
        
        evaluation_metrics = trainer.evaluate_model(X_test, y_test, detailed=False)
        logger.info("Model Performance Metrics:")
        for metric, value in evaluation_metrics.items():
            logger.info(f"   {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Step 4: Save Model
        logger.info("\nStep 4: Model Saving")
        logger.info("-" * 30)
        
        os.makedirs('models', exist_ok=True)
        model_path = 'models/quick_start_model.pkl'
        trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Step 5: Model Inference Demo
        logger.info("\n🔮 Step 5: Model Inference Demo")
        logger.info("-" * 30)
        
        # Create sample customers for prediction
        sample_customers = pd.DataFrame({
            # Customer 1: High risk profile
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
        
        # Initialize inference
        inference = ChurnModelInference(model_path=model_path)
        
        # Make predictions
        results = inference.predict(sample_processed, return_probabilities=True)
        
        if results is not None:
            logger.info("\nPrediction Results:")
            for i in range(len(sample_customers)):
                customer_id = sample_customers.iloc[i]['customer_id']
                prediction = results['prediction_labels'][i]
                probability = results['churn_probability'][i]
                
                risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
                
                logger.info(f"   Customer {customer_id}: {prediction} (Risk: {risk_level}, Prob: {probability:.2%})")
        
        # Step 6: Create Summary Report
        logger.info("\nStep 6: Summary Report")
        logger.info("-" * 30)
        
        # Create results directory
        os.makedirs('results/reports', exist_ok=True)
        
        # Generate summary report
        report = {
            'demo_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(X_train) + len(X_test),
                'features': len(X_train.columns),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'overall_churn_rate': (len(y_train[y_train == 1]) + len(y_test[y_test == 1])) / (len(y_train) + len(y_test))
            },
            'model_performance': evaluation_metrics,
            'top_features': feature_importance.to_dict('records') if feature_importance is not None else None,
            'sample_predictions': {
                'customer_ids': sample_customers['customer_id'].tolist(),
                'predictions': results['prediction_labels'] if results else None,
                'probabilities': results['churn_probability'].tolist() if results else None
            }
        }
        
        # Save report
        import json
        report_path = f"results/reports/quick_start_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Demo report saved to {report_path}")
        
        # Final summary
        logger.info("\nQuick Start Demo Complete!")
        logger.info("=" * 60)
        logger.info("Summary of what was accomplished:")
        logger.info("Data preprocessing and feature engineering")
        logger.info("Random Forest model training")
        logger.info("Model evaluation and performance metrics")
        logger.info("Feature importance analysis")
        logger.info("Model inference on sample customers")
        logger.info("Comprehensive reporting")
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

def create_sample_dataset():
    """
    Create a sample dataset if the original is not available
    """
    logger.info("Creating sample dataset for demonstration...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data that resembles the bank churn dataset
    data = {
        'customer_id': range(1, n_samples + 1),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int),
        'country': np.random.choice(['France', 'Germany', 'Spain'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.randint(0, 11, n_samples),
        'balance': np.random.exponential(50000, n_samples),
        'products_number': np.random.randint(1, 5, n_samples),
        'credit_card': np.random.choice([0, 1], n_samples),
        'active_member': np.random.choice([0, 1], n_samples),
        'estimated_salary': np.random.normal(75000, 25000, n_samples),
    }
    
    # Create churn based on some logical rules
    churn_prob = (
        (data['age'] > 50).astype(int) * 0.2 +
        (data['balance'] == 0).astype(int) * 0.3 +
        (data['products_number'] == 1).astype(int) * 0.2 +
        (data['active_member'] == 0).astype(int) * 0.3 +
        np.random.random(n_samples) * 0.1
    )
    
    data['churn'] = (churn_prob > 0.5).astype(int)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/Bank Customer Churn Prediction.csv', index=False)
    
    logger.info(f"Sample dataset created with {n_samples} samples")
    logger.info(f"   Churn rate: {df['churn'].mean():.2%}")
    
    return True

if __name__ == "__main__":
    print("Bank Customer Churn Prediction - Quick Start Demo")
    print("=" * 60)
    
    # Check if dataset exists, create sample if not
    data_path = 'data/Bank Customer Churn Prediction.csv'
    if not os.path.exists(data_path):
        print(" Original dataset not found.")
        response = input("Would you like to create a sample dataset for demonstration? (y/n): ")
        if response.lower().startswith('y'):
            create_sample_dataset()
        else:
            print("Please place your dataset in the data/ directory and run again.")
            sys.exit(1)
    
    # Run the demo
    success = quick_start_demo()
    
    if success:
        print("\nDemo completed successfully!")
        print("Check the results/ directory for outputs.")
    else:
        print("\nDemo failed. Check the logs above for details.")
        sys.exit(1)