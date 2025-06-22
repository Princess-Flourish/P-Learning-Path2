"""
Model Inference Module for Bank Customer Churn Prediction
This module handles model loading and prediction for new data
"""

import pandas as pd
import numpy as np
import joblib
import mlflow.sklearn
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnModelInference:
    """
    A class to handle model inference for churn prediction
    """
    
    def __init__(self, model_path=None, mlflow_model_uri=None):
        """
        Initialize the inference class
        
        Args:
            model_path (str, optional): Path to saved model file
            mlflow_model_uri (str, optional): MLflow model URI for loading model
        """
        self.model = None
        self.model_path = model_path
        self.mlflow_model_uri = mlflow_model_uri
        self.feature_names = None
        
        # Try to load model during initialization
        if model_path:
            self.load_model_from_file(model_path)
        elif mlflow_model_uri:
            self.load_model_from_mlflow(mlflow_model_uri)
    
    def load_model_from_file(self, model_path):
        """
        Load model from a saved file
        
        Args:
            model_path (str): Path to the saved model file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            self.model_path = model_path
            
            # Try to get feature names if available
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from file: {str(e)}")
            return False
    
    def load_model_from_mlflow(self, model_uri):
        """
        Load model from MLflow model registry
        
        Args:
            model_uri (str): MLflow model URI (e.g., 'models:/model_name/version')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = mlflow.sklearn.load_model(model_uri)
            self.mlflow_model_uri = model_uri
            
            # Try to get feature names if available
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_
            
            logger.info(f"Model loaded successfully from MLflow: {model_uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {str(e)}")
            return False
    
    def validate_input_data(self, X):
        """
        Validate input data for prediction
        
        Args:
            X (pandas.DataFrame): Input features
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if self.model is None:
            logger.error("No model loaded. Please load a model first.")
            return False
        
        if not isinstance(X, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            return False
        
        if X.empty:
            logger.error("Input DataFrame is empty")
            return False
        
        # Check if feature names match (if available)
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            extra_features = set(X.columns) - set(self.feature_names)
            
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                return False
            
            if extra_features:
                logger.warning(f"Extra features found (will be ignored): {extra_features}")
                # Select only required features in correct order
                X = X[self.feature_names]
        
        # Check for missing values
        if X.isnull().sum().sum() > 0:
            logger.warning("Input data contains missing values")
            # For now, we'll proceed but log the warning
        
        return True
    
    def predict(self, X, return_probabilities=False):
        """
        Make churn predictions for input data
        
        Args:
            X (pandas.DataFrame): Input features
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            dict: Dictionary containing predictions and optionally probabilities
        """
        if not self.validate_input_data(X):
            return None
        
        try:
            # Ensure we use the correct feature order if feature names are available
            if self.feature_names is not None:
                X_ordered = X[self.feature_names]
            else:
                X_ordered = X
            
            # Make predictions
            predictions = self.model.predict(X_ordered)
            
            results = {
                'predictions': predictions,
                'prediction_labels': ['No Churn' if pred == 0 else 'Churn' for pred in predictions]
            }
            
            # Add probabilities if requested
            if return_probabilities:
                probabilities = self.model.predict_proba(X_ordered)
                results['probabilities'] = probabilities
                results['churn_probability'] = probabilities[:, 1]  # Probability of churn (class 1)
            
            logger.info(f"Predictions completed for {len(X)} samples")
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None
    
    def predict_single_customer(self, customer_data, return_probabilities=True):
        """
        Make prediction for a single customer
        
        Args:
            customer_data (dict or pandas.Series): Customer data
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            dict: Prediction result for the customer
        """
        try:
            # Convert single customer data to DataFrame
            if isinstance(customer_data, dict):
                X = pd.DataFrame([customer_data])
            elif isinstance(customer_data, pd.Series):
                X = pd.DataFrame([customer_data])
            else:
                logger.error("Customer data must be a dictionary or pandas Series")
                return None
            
            # Make prediction
            results = self.predict(X, return_probabilities=return_probabilities)
            
            if results is None:
                return None
            
            # Format results for single customer
            single_result = {
                'prediction': results['predictions'][0],
                'prediction_label': results['prediction_labels'][0]
            }
            
            if return_probabilities:
                single_result['churn_probability'] = results['churn_probability'][0]
                single_result['no_churn_probability'] = 1 - results['churn_probability'][0]
                
                # Add risk category based on probability
                churn_prob = results['churn_probability'][0]
                if churn_prob < 0.3:
                    risk_category = "Low Risk"
                elif churn_prob < 0.7:
                    risk_category = "Medium Risk"
                else:
                    risk_category = "High Risk"
                
                single_result['risk_category'] = risk_category
            
            return single_result
            
        except Exception as e:
            logger.error(f"Single customer prediction failed: {str(e)}")
            return None
    
    def batch_predict(self, data_path, output_path=None, chunk_size=1000):
        """
        Perform batch predictions on a large dataset
        
        Args:
            data_path (str): Path to input data CSV file
            output_path (str, optional): Path to save predictions
            chunk_size (int): Number of rows to process at once
            
        Returns:
            pandas.DataFrame: DataFrame with predictions
        """
        try:
            logger.info(f"Starting batch prediction for {data_path}")
            
            # Read data in chunks if file is large
            all_predictions = []
            chunk_count = 0
            
            for chunk in pd.read_csv(data_path, chunksize=chunk_size):
                chunk_count += 1
                logger.info(f"Processing chunk {chunk_count} with {len(chunk)} rows")
                
                # Make predictions for this chunk
                results = self.predict(chunk, return_probabilities=True)
                
                if results is None:
                    logger.error(f"Failed to process chunk {chunk_count}")
                    continue
                
                # Create prediction DataFrame for this chunk
                prediction_df = pd.DataFrame({
                    'prediction': results['predictions'],
                    'prediction_label': results['prediction_labels'],
                    'churn_probability': results['churn_probability']
                })
                
                # Add original data
                chunk_with_predictions = pd.concat([chunk.reset_index(drop=True), 
                                                  prediction_df.reset_index(drop=True)], axis=1)
                
                all_predictions.append(chunk_with_predictions)
            
            # Combine all chunks
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            
            # Save to file if output path is provided
            if output_path:
                final_predictions.to_csv(output_path, index=False)
                logger.info(f"Predictions saved to {output_path}")
            
            logger.info(f"Batch prediction completed for {len(final_predictions)} rows")
            return final_predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            return None
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "model_type": type(self.model).__name__,
            "model_loaded": True
        }
        
        # Add model-specific information
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth
        
        if hasattr(self.model, 'feature_importances_'):
            info['n_features'] = len(self.model.feature_importances_)
        
        if self.feature_names is not None:
            info['feature_names'] = list(self.feature_names)
        
        if self.model_path:
            info['loaded_from'] = 'file'
            info['model_path'] = self.model_path
        
        if self.mlflow_model_uri:
            info['loaded_from'] = 'mlflow'
            info['mlflow_uri'] = self.mlflow_model_uri
        
        return info
    
    def explain_prediction(self, customer_data, top_features=5):
        """
        Provide explanation for a prediction using feature importance
        
        Args:
            customer_data (dict or pandas.Series): Customer data
            top_features (int): Number of top features to include in explanation
            
        Returns:
            dict: Explanation of the prediction
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {"error": "Model does not support feature importance"}
        
        try:
            # Get prediction
            prediction_result = self.predict_single_customer(customer_data, return_probabilities=True)
            
            if prediction_result is None:
                return {"error": "Failed to make prediction"}
            
            # Convert customer data to DataFrame if needed
            if isinstance(customer_data, dict):
                X = pd.DataFrame([customer_data])
            elif isinstance(customer_data, pd.Series):
                X = pd.DataFrame([customer_data])
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns if self.feature_names is None else self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Get top features
            top_features_info = feature_importance.head(top_features)
            
            explanation = {
                'prediction': prediction_result,
                'top_influential_features': [
                    {
                        'feature': row['feature'],
                        'importance': row['importance'],
                        'customer_value': customer_data.get(row['feature'], 'N/A') if isinstance(customer_data, dict) else customer_data[row['feature']]
                    }
                    for _, row in top_features_info.iterrows()
                ]
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Prediction explanation failed: {str(e)}")
            return {"error": str(e)}
    
    def monitor_predictions(self, predictions_df, save_path=None):
        """
        Monitor and log prediction statistics
        
        Args:
            predictions_df (pandas.DataFrame): DataFrame with predictions
            save_path (str, optional): Path to save monitoring report
            
        Returns:
            dict: Monitoring statistics
        """
        try:
            if 'prediction' not in predictions_df.columns:
                logger.error("Predictions DataFrame must contain 'prediction' column")
                return None
            
            # Calculate monitoring statistics
            total_predictions = len(predictions_df)
            churn_predictions = (predictions_df['prediction'] == 1).sum()
            churn_rate = churn_predictions / total_predictions
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'total_predictions': total_predictions,
                'churn_predictions': churn_predictions,
                'no_churn_predictions': total_predictions - churn_predictions,
                'predicted_churn_rate': churn_rate
            }
            
            # Add probability statistics if available
            if 'churn_probability' in predictions_df.columns:
                stats.update({
                    'avg_churn_probability': predictions_df['churn_probability'].mean(),
                    'median_churn_probability': predictions_df['churn_probability'].median(),
                    'high_risk_customers': (predictions_df['churn_probability'] > 0.7).sum(),
                    'medium_risk_customers': ((predictions_df['churn_probability'] >= 0.3) & 
                                            (predictions_df['churn_probability'] <= 0.7)).sum(),
                    'low_risk_customers': (predictions_df['churn_probability'] < 0.3).sum()
                })
            
            # Save monitoring report if path provided
            if save_path:
                monitoring_df = pd.DataFrame([stats])
                
                # Append to existing file or create new one
                if os.path.exists(save_path):
                    existing_df = pd.read_csv(save_path)
                    combined_df = pd.concat([existing_df, monitoring_df], ignore_index=True)
                else:
                    combined_df = monitoring_df
                
                combined_df.to_csv(save_path, index=False)
                logger.info(f"Monitoring report saved to {save_path}")
            
            logger.info(f"Prediction monitoring completed: {churn_rate:.2%} churn rate")
            return stats
            
        except Exception as e:
            logger.error(f"Prediction monitoring failed: {str(e)}")
            return None