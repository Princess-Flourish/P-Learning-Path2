"""
Model Training Module for Bank Customer Churn Prediction
This module handles model training, hyperparameter tuning, and MLflow tracking
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnModelTrainer:
    """
    A class to handle model training, evaluation, and MLflow tracking for churn prediction
    """
    
    def __init__(self, experiment_name="bank_churn_prediction"):
        """
        Initialize the model trainer with MLflow experiment setup
        
        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
        # Set up MLflow
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """
        Set up MLflow experiment and tracking URI
        """
        try:
            # Set MLflow tracking URI - use environment variable if available, otherwise localhost
            tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
            
            # Create or get existing experiment
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLflow experiment: {self.experiment_name}")
            except mlflow.exceptions.MlflowException:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
            
            # Set the experiment
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow setup completed successfully")
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {str(e)}. Continuing without MLflow tracking.")
    
    def train_baseline_model(self, X_train, y_train, X_test, y_test):
        """
        Train a baseline Random Forest model with default parameters
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            X_test (pandas.DataFrame): Test features
            y_test (pandas.Series): Test target
            
        Returns:
            sklearn.ensemble.RandomForestClassifier: Trained baseline model
        """
        logger.info("Training baseline Random Forest model...")
        
        # Initialize baseline model with some reasonable defaults
        baseline_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        # Train the model
        baseline_model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = baseline_model.predict(X_train)
        y_test_pred = baseline_model.predict(X_test)
        y_test_prob = baseline_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_prob)
        
        # Log results
        logger.info("Baseline Model Results:")
        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        
        # Track with MLflow
        try:
            with mlflow.start_run(run_name="baseline_random_forest"):
                # Log parameters
                mlflow.log_param("model_type", "RandomForestClassifier")
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("is_baseline", True)
                
                # Log metrics
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)
                
                # Log model
                mlflow.sklearn.log_model(baseline_model, "model")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
        
        return baseline_model
    
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=5):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            sklearn.model_selection.GridSearchCV: Best model from grid search
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform grid search
        logger.info(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} parameter combinations...")
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='roc_auc',  # Use ROC-AUC as the scoring metric
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params = grid_search.best_params_
        logger.info(f"Best parameters found: {self.best_params}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def train_optimized_model(self, X_train, y_train, X_test, y_test, params=None):
        """
        Train the final optimized model with best parameters
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            X_test (pandas.DataFrame): Test features
            y_test (pandas.Series): Test target
            params (dict, optional): Model parameters. If None, uses best_params
            
        Returns:
            sklearn.ensemble.RandomForestClassifier: Trained optimized model
        """
        logger.info("Training optimized Random Forest model...")
        
        # Use provided parameters or best parameters from tuning
        if params is None:
            if self.best_params is None:
                logger.warning("No parameters provided and no tuning performed. Using default parameters.")
                params = {'n_estimators': 200, 'random_state': 42}
            else:
                params = self.best_params.copy()
                params['random_state'] = 42
        
        # Initialize and train optimized model
        self.model = RandomForestClassifier(**params, n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, y_test_prob)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log comprehensive results
        self.log_model_results(metrics, params)
        
        # Track with MLflow
        self.track_with_mlflow(metrics, params, X_test, y_test)
        
        return self.model
    
    def calculate_metrics(self, y_train, y_train_pred, y_test, y_test_pred, y_test_prob):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_train (pandas.Series): Training target
            y_train_pred (numpy.array): Training predictions
            y_test (pandas.Series): Test target
            y_test_pred (numpy.array): Test predictions
            y_test_prob (numpy.array): Test prediction probabilities
            
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_prob)
        }
        
        return metrics
    
    def log_model_results(self, metrics, params):
        """
        Log model training results to console
        
        Args:
            metrics (dict): Dictionary of evaluation metrics
            params (dict): Model parameters used
        """
        logger.info("Optimized Model Results:")
        logger.info(f"Parameters: {params}")
        logger.info(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        if 'cv_mean' in metrics:
            logger.info(f"Cross-Validation ROC-AUC: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    def track_with_mlflow(self, metrics, params, X_test, y_test):
        """
        Track model training with MLflow
        
        Args:
            metrics (dict): Dictionary of evaluation metrics
            params (dict): Model parameters used
            X_test (pandas.DataFrame): Test features
            y_test (pandas.Series): Test target
        """
        try:
            with mlflow.start_run(run_name=f"optimized_random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("model_type", "RandomForestClassifier")
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
                # Log feature importance as artifact
                if self.feature_importance is not None:
                    importance_path = "feature_importance.csv"
                    self.feature_importance.to_csv(importance_path, index=False)
                    mlflow.log_artifact(importance_path)
                    os.remove(importance_path)  # Clean up temporary file
                
                # Create and log plots
                self.create_and_log_plots(X_test, y_test)
                
                logger.info("Model tracked successfully with MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow tracking failed: {str(e)}")
    
    def create_and_log_plots(self, X_test, y_test):
        """
        Create visualizations and log them as MLflow artifacts
        
        Args:
            X_test (pandas.DataFrame): Test features
            y_test (pandas.Series): Test target
        """
        try:
            # Create feature importance plot
            plt.figure(figsize=(10, 8))
            top_features = self.feature_importance.head(15)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Top 15 Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('feature_importance_plot.png')
            plt.close()
            
            # Create confusion matrix plot
            y_pred = self.model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('confusion_matrix.png')
            plt.close()
            
            # Clean up temporary files
            for file in ['feature_importance_plot.png', 'confusion_matrix.png']:
                if os.path.exists(file):
                    os.remove(file)
                    
        except Exception as e:
            logger.warning(f"Plot creation failed: {str(e)}")
    
    def save_model(self, model_path="models/churn_model.pkl"):
        """
        Save the trained model to disk
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            logger.error("No model to save. Please train a model first.")
            return
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path="models/churn_model.pkl"):
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            sklearn.ensemble.RandomForestClassifier: Loaded model
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
    
    def get_feature_importance(self, top_n=None):
        """
        Get feature importance from the trained model
        
        Args:
            top_n (int, optional): Number of top features to return
            
        Returns:
            pandas.DataFrame: Feature importance dataframe
        """
        if self.feature_importance is None:
            logger.error("No feature importance available. Please train a model first.")
            return None
        
        if top_n is not None:
            return self.feature_importance.head(top_n)
        
        return self.feature_importance
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X (pandas.DataFrame): Features for prediction
            
        Returns:
            numpy.array: Predictions
        """
        if self.model is None:
            logger.error("No model available for prediction. Please train or load a model first.")
            return None
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities using the trained model
        
        Args:
            X (pandas.DataFrame): Features for prediction
            
        Returns:
            numpy.array: Prediction probabilities
        """
        if self.model is None:
            logger.error("No model available for prediction. Please train or load a model first.")
            return None
        
        return self.model.predict_proba(X)
    
    def evaluate_model(self, X_test, y_test, detailed=True):
        """
        Evaluate the trained model on test data
        
        Args:
            X_test (pandas.DataFrame): Test features
            y_test (pandas.Series): Test target
            detailed (bool): Whether to return detailed classification report
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            logger.error("No model available for evaluation. Please train or load a model first.")
            return None
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        if detailed:
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print(f"\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        
        return metrics
    
    def full_training_pipeline(self, X_train, y_train, X_test, y_test, tune_hyperparameters=True):
        """
        Execute the complete model training pipeline
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            X_test (pandas.DataFrame): Test features
            y_test (pandas.Series): Test target
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            
        Returns:
            sklearn.ensemble.RandomForestClassifier: Final trained model
        """
        logger.info("Starting full training pipeline...")
        
        # Step 1: Train baseline model
        baseline_model = self.train_baseline_model(X_train, y_train, X_test, y_test)
        
        # Step 2: Hyperparameter tuning (optional)
        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            grid_search = self.hyperparameter_tuning(X_train, y_train)
        else:
            logger.info("Skipping hyperparameter tuning...")
        
        # Step 3: Train optimized model
        final_model = self.train_optimized_model(X_train, y_train, X_test, y_test)
        
        # Step 4: Save model
        self.save_model()
        
        logger.info("Training pipeline completed successfully!")
        
        return final_model