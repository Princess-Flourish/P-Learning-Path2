"""
Airflow DAG for Bank Customer Churn Prediction Pipeline
This DAG orchestrates the complete ML pipeline from data preprocessing to model training and evaluation
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
import sys
import os

# Add the src directory to Python path so we can import our modules
sys.path.append('/opt/airflow/src')

from src.data_preprocessing import DataPreprocessor
from src.model_training import ChurnModelTrainer
from src.model_inference import ChurnModelInference
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG configuration
dag = DAG(
    'bank_churn_ml_pipeline',
    default_args=default_args,
    description='Complete ML pipeline for bank customer churn prediction',
    schedule_interval='@weekly',  # Run weekly
    catchup=False,
    max_active_runs=1,
    tags=['machine_learning', 'churn_prediction', 'bank'],
)

# Configuration variables
DATA_PATH = '/opt/airflow/data/Bank Customer Churn Prediction.csv'
MODEL_PATH = '/opt/airflow/models/churn_model.pkl'
PREPROCESSOR_PATH = '/opt/airflow/models/preprocessor.pkl'
RESULTS_PATH = '/opt/airflow/results/'

def check_data_quality(**context):
    """
    Task to check data quality and validate input data
    """
    logger.info("Starting data quality check...")
    
    try:
        # Load data
        data = pd.read_csv(DATA_PATH)
        
        # Basic quality checks
        checks = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'churn_rate': data['churn'].mean() if 'churn' in data.columns else None
        }
        
        # Log quality metrics
        logger.info(f"Data Quality Report: {checks}")
        
        # Define quality thresholds
        if checks['row_count'] < 1000:
            raise ValueError(f"Insufficient data: {checks['row_count']} rows")
        
        if checks['missing_values'] > checks['row_count'] * 0.1:  # More than 10% missing
            logger.warning(f"High missing values: {checks['missing_values']}")
        
        if checks['churn_rate'] is None:
            raise ValueError("Target column 'churn' not found")
        
        if checks['churn_rate'] < 0.01 or checks['churn_rate'] > 0.99:
            logger.warning(f"Unusual churn rate: {checks['churn_rate']:.2%}")
        
        # Store quality metrics for downstream tasks
        context['task_instance'].xcom_push(key='data_quality', value=checks)
        
        logger.info("Data quality check completed successfully")
        return checks
        
    except Exception as e:
        logger.error(f"Data quality check failed: {str(e)}")
        raise

def preprocess_data(**context):
    """
    Task to preprocess the raw data
    """
    logger.info("Starting data preprocessing...")
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Run preprocessing pipeline
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            DATA_PATH, 
            test_size=0.2, 
            random_state=42
        )
        
        # Save preprocessed data
        os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
        
        # Save the preprocessor for later use
        import joblib
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        
        # Save preprocessed datasets
        train_data_path = '/opt/airflow/data/processed/train_data.csv'
        test_data_path = '/opt/airflow/data/processed/test_data.csv'
        os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
        
        # Combine features and target for saving
        train_data = X_train.copy()
        train_data['churn'] = y_train
        test_data = X_test.copy()
        test_data['churn'] = y_test
        
        train_data.to_csv(train_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)
        
        # Store preprocessing results
        preprocessing_results = {
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'feature_count': len(X_train.columns),
            'train_churn_rate': y_train.mean(),
            'test_churn_rate': y_test.mean()
        }
        
        context['task_instance'].xcom_push(key='preprocessing_results', value=preprocessing_results)
        
        logger.info(f"Data preprocessing completed: {preprocessing_results}")
        return preprocessing_results
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise

def train_model(**context):
    """
    Task to train the machine learning model with improved exception handling
    """
    logger.info("Starting model training...")
    
    training_results = None
    mlflow_run_failed = False
    
    try:
        # Set MLflow tracking URI
        import os
        os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow-server:5000'

        # Load preprocessed data
        train_data = pd.read_csv('/opt/airflow/data/processed/train_data.csv')
        test_data = pd.read_csv('/opt/airflow/data/processed/test_data.csv')
        
        # Separate features and target
        X_train = train_data.drop('churn', axis=1)
        y_train = train_data['churn']
        X_test = test_data.drop('churn', axis=1)
        y_test = test_data['churn']
        
        # Initialize model trainer
        trainer = ChurnModelTrainer(experiment_name="bank_churn_prediction_airflow")
        
        # Get hyperparameter tuning flag from Airflow Variables
        try:
            tune_hyperparameters = Variable.get("tune_hyperparameters", default_var=True, deserialize_json=True)
        except:
            tune_hyperparameters = False
        
        # Core training with detailed exception handling
        try:
            logger.info("Starting core model training...")
            
            # Run full training pipeline - this is where MLflow runs happen
            model = trainer.full_training_pipeline(
                X_train, y_train, X_test, y_test,
                tune_hyperparameters=tune_hyperparameters
            )
            
            logger.info("Core model training completed successfully")
            
        except Exception as training_error:
            logger.error(f"Core training failed: {str(training_error)}")
            mlflow_run_failed = True
            
            # Don't re-raise immediately - try to salvage what we can
            # Check if we have any partial results
            try:
                # Try to get the best model if hyperparameter tuning was attempted
                if hasattr(trainer, 'best_model') and trainer.best_model is not None:
                    model = trainer.best_model
                    logger.info("Retrieved best model from failed training attempt")
                else:
                    # If no model available, re-raise the original error
                    raise training_error
            except:
                raise training_error
        
        # Post-training evaluation with separate exception handling
        try:
            logger.info("Starting model evaluation...")
            evaluation_metrics = trainer.evaluate_model(X_test, y_test, detailed=False)
            logger.info("Model evaluation completed successfully")
            
        except Exception as eval_error:
            logger.warning(f"Model evaluation failed: {str(eval_error)}")
            # Provide default metrics if evaluation fails
            evaluation_metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'evaluation_status': 'failed'
            }
        
        # Feature importance extraction with exception handling
        feature_importance = None
        try:
            if hasattr(trainer, 'get_feature_importance') and callable(trainer.get_feature_importance):
                importance_df = trainer.get_feature_importance(top_n=5)
                if importance_df is not None:
                    feature_importance = importance_df.to_dict('records')
                    logger.info("Feature importance extracted successfully")
        except Exception as fi_error:
            logger.warning(f"Feature importance extraction failed: {str(fi_error)}")
        
        # Prepare training results
        training_results = {
            'model_type': type(model).__name__ if model is not None else 'Unknown',
            'training_completed': model is not None,
            'training_had_issues': mlflow_run_failed,
            'evaluation_metrics': evaluation_metrics,
            'feature_importance_top5': feature_importance
        }
        
        # Push results to XCom
        context['task_instance'].xcom_push(key='training_results', value=training_results)
        
        # Final status logging
        if mlflow_run_failed:
            logger.warning("Training completed with issues - check MLflow for details")
            logger.warning(f"Training results: {training_results}")
        else:
            logger.info(f"Model training completed successfully: {evaluation_metrics}")
        
        return training_results
        
    except Exception as e:
        logger.error(f"Model training task failed completely: {str(e)}")
        
        # Create failure results for downstream tasks
        failure_results = {
            'model_type': 'Failed',
            'training_completed': False,
            'training_had_issues': True,
            'evaluation_metrics': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'evaluation_status': 'task_failed'
            },
            'feature_importance_top5': None,
            'error_message': str(e)
        }
        
        # Still push results so downstream tasks can handle the failure gracefully
        context['task_instance'].xcom_push(key='training_results', value=failure_results)
        
        # Re-raise to fail the Airflow task (but MLflow runs should be cleaner now)
        raise

def validate_model(**context):
    """
    Task to validate the trained model meets performance thresholds
    """
    logger.info("Starting model validation...")
    
    try:
        # Get training results from previous task
        training_results = context['task_instance'].xcom_pull(key='training_results', task_ids='train_model')
        
        if not training_results or not training_results.get('training_completed'):
            raise ValueError("No valid training results found")
        
        evaluation_metrics = training_results['evaluation_metrics']
        
        # Define performance thresholds
        thresholds = {
            'roc_auc': 0.75,      # Minimum AUC score
            'precision': 0.60,     # Minimum precision
            'recall': 0.60,        # Minimum recall
            'f1_score': 0.60       # Minimum F1 score
        }
        
        # Check if model meets thresholds
        validation_results = {'model_approved': True, 'failed_checks': []}
        
        for metric, threshold in thresholds.items():
            if metric in evaluation_metrics:
                if evaluation_metrics[metric] < threshold:
                    validation_results['model_approved'] = False
                    validation_results['failed_checks'].append(
                        f"{metric}: {evaluation_metrics[metric]:.4f} < {threshold}"
                    )
        
        # Log validation results
        if validation_results['model_approved']:
            logger.info("Model validation PASSED - Model meets all performance thresholds")
        else:
            logger.warning(f"Model validation FAILED - {validation_results['failed_checks']}")
        
        validation_results['evaluation_metrics'] = evaluation_metrics
        validation_results['thresholds'] = thresholds
        
        context['task_instance'].xcom_push(key='validation_results', value=validation_results)
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        raise

def deploy_model(**context):
    """
    Task to deploy the validated model
    """
    logger.info("Starting model deployment...")
    
    try:
        # Get validation results
        validation_results = context['task_instance'].xcom_pull(key='validation_results', task_ids='validate_model')
        
        if not validation_results or not validation_results.get('model_approved'):
            logger.warning("Model not approved for deployment")
            return {'deployment_status': 'skipped', 'reason': 'model_not_approved'}
        
        # For this example, deployment means copying model to production directory
        production_model_path = '/opt/airflow/models/production/churn_model.pkl'
        os.makedirs(os.path.dirname(production_model_path), exist_ok=True)
        
        # Copy model to production directory
        import shutil
        shutil.copy2(MODEL_PATH, production_model_path)
        
        # Create deployment metadata
        deployment_info = {
            'deployment_timestamp': datetime.now().isoformat(),
            'model_path': production_model_path,
            'deployment_status': 'success',
            'model_metrics': validation_results['evaluation_metrics']
        }
        
        # Save deployment metadata
        deployment_metadata_path = '/opt/airflow/models/production/deployment_metadata.json'
        import json
        with open(deployment_metadata_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        context['task_instance'].xcom_push(key='deployment_results', value=deployment_info)
        
        logger.info("Model deployment completed successfully")
        return deployment_info
        
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}")
        raise

def run_batch_inference(**context):
    """
    Task to run batch inference on new data (simulation)
    """
    logger.info("Starting batch inference...")
    
    try:
        # For demonstration, we'll use a subset of the original data as "new" data
        original_data = pd.read_csv(DATA_PATH)
        
        # Take a sample as "new" data for inference
        new_data = original_data.sample(n=100, random_state=42).drop('churn', axis=1)
        
        # Load production model
        production_model_path = '/opt/airflow/models/production/churn_model.pkl'
        
        # Initialize inference class
        inference = ChurnModelInference(model_path=production_model_path)
        
        # Load preprocessor
        import joblib
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        # Preprocess new data
        processed_data = preprocessor.preprocess_inference_data(new_data)
        
        # Make predictions
        results = inference.predict(processed_data, return_probabilities=True)
        
        if results is None:
            raise ValueError("Inference failed")
        
        # Create results DataFrame
        inference_df = pd.DataFrame({
            'customer_id': new_data['customer_id'].values,
            'prediction': results['predictions'],
            'prediction_label': results['prediction_labels'],
            'churn_probability': results['churn_probability']
        })
        
        # Save inference results
        results_dir = '/opt/airflow/results/inference/'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = f"{results_dir}batch_inference_{timestamp}.csv"
        inference_df.to_csv(results_path, index=False)
        
        # Monitor predictions
        monitoring_stats = inference.monitor_predictions(
            inference_df, 
            save_path=f"{results_dir}monitoring_log.csv"
        )
        
        inference_results = {
            'inference_completed': True,
            'total_predictions': len(inference_df),
            'predicted_churn_rate': monitoring_stats['predicted_churn_rate'],
            'results_path': results_path,
            'monitoring_stats': monitoring_stats
        }
        
        context['task_instance'].xcom_push(key='inference_results', value=inference_results)
        
        logger.info(f"Batch inference completed: {inference_results}")
        return inference_results
        
    except Exception as e:
        logger.error(f"Batch inference failed: {str(e)}")
        raise

def generate_report(**context):
    """
    Task to generate a comprehensive pipeline report
    """
    logger.info("Generating pipeline report...")
    
    try:
        # Collect results from all previous tasks
        data_quality = context['task_instance'].xcom_pull(key='data_quality', task_ids='check_data_quality')
        preprocessing_results = context['task_instance'].xcom_pull(key='preprocessing_results', task_ids='preprocess_data')
        training_results = context['task_instance'].xcom_pull(key='training_results', task_ids='train_model')
        validation_results = context['task_instance'].xcom_pull(key='validation_results', task_ids='validate_model')
        deployment_results = context['task_instance'].xcom_pull(key='deployment_results', task_ids='deploy_model')
        inference_results = context['task_instance'].xcom_pull(key='inference_results', task_ids='run_batch_inference')
        
        # Create comprehensive report
        report = {
            'pipeline_run_date': datetime.now().isoformat(),
            'pipeline_status': 'completed',
            'data_quality_check': data_quality,
            'preprocessing': preprocessing_results,
            'model_training': training_results,
            'model_validation': validation_results,
            'model_deployment': deployment_results,
            'batch_inference': inference_results
        }
        
        # Save report
        reports_dir = '/opt/airflow/results/reports/'
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{reports_dir}pipeline_report_{timestamp}.json"
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Pipeline report generated: {report_path}")
        return report
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise

# Define tasks

# File sensor to check if data is available
data_sensor = FileSensor(
    task_id='wait_for_data',
    filepath=DATA_PATH,
    timeout=60 * 5,  # Wait up to 5 minutes
    poke_interval=30,  # Check every 30 seconds
    dag=dag
)

# Data quality check task
data_quality_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag
)

# Data preprocessing task
preprocessing_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

# Model training task
training_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

# Model validation task
validation_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

# Model deployment task
deployment_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# Batch inference task
inference_task = PythonOperator(
    task_id='run_batch_inference',
    python_callable=run_batch_inference,
    dag=dag
)

# Report generation task
report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    trigger_rule='all_done',  # Run even if some upstream tasks fail
    dag=dag
)

# MLflow server health check
mlflow_health_check = BashOperator(
    task_id='check_mlflow_server',
    bash_command='curl -f http://mlflow-server:5000/health || echo "MLflow server not responding"',
    dag=dag
)

# Define task dependencies
data_sensor >> data_quality_task >> preprocessing_task
mlflow_health_check >> training_task
preprocessing_task >> training_task >> validation_task >> deployment_task >> inference_task
inference_task >> report_task

# Parallel paths for monitoring
validation_task >> report_task  # Report can run even if deployment fails