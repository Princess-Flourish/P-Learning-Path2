"""
Data Preprocessing Module for Bank Customer Churn Prediction
This module handles data loading, cleaning, feature engineering, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import os

# Set up logging to track preprocessing steps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class to handle all data preprocessing tasks for the churn prediction model
    """
    
    def __init__(self):
        """
        Initialize the data preprocessor with necessary encoders and scalers
        """
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'churn'
        
    def load_data(self, file_path):
        """
        Load the bank customer churn dataset from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        try:
            logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def explore_data(self, data):
        """
        Perform basic data exploration and print key statistics
        
        Args:
            data (pandas.DataFrame): The dataset to explore
        """
        logger.info("Starting data exploration...")
        
        # Basic information about the dataset
        print("Dataset Info:")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Check for missing values
        print("\nMissing Values:")
        print(data.isnull().sum())
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(data.describe())
        
        # Target variable distribution
        print(f"\nChurn Distribution:")
        print(data['churn'].value_counts())
        print(f"Churn Rate: {data['churn'].mean():.2%}")
        
        # Categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns
        print(f"\nCategorical Columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            print(f"\n{col} Distribution:")
            print(data[col].value_counts())
    
    def clean_data(self, data):
        """
        Clean the dataset by handling missing values and outliers
        
        Args:
            data (pandas.DataFrame): Raw dataset
            
        Returns:
            pandas.DataFrame: Cleaned dataset
        """
        logger.info("Starting data cleaning...")
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Check for missing values
        if cleaned_data.isnull().sum().sum() > 0:
            logger.warning("Found missing values, handling them...")
            # For this dataset, we'll fill numerical missing values with median
            # and categorical missing values with mode
            for column in cleaned_data.columns:
                if cleaned_data[column].dtype in ['int64', 'float64']:
                    cleaned_data[column].fillna(cleaned_data[column].median(), inplace=True)
                else:
                    cleaned_data[column].fillna(cleaned_data[column].mode()[0], inplace=True)
        
        # Remove duplicates if any
        initial_shape = cleaned_data.shape[0]
        cleaned_data = cleaned_data.drop_duplicates()
        final_shape = cleaned_data.shape[0]
        
        if initial_shape != final_shape:
            logger.info(f"Removed {initial_shape - final_shape} duplicate rows")
        
        # Handle outliers in numerical columns using IQR method
        numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']
        
        for col in numerical_cols:
            if col in cleaned_data.columns:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers before removal
                outliers_count = ((cleaned_data[col] < lower_bound) | 
                                (cleaned_data[col] > upper_bound)).sum()
                
                if outliers_count > 0:
                    logger.info(f"Found {outliers_count} outliers in {col}")
                    # Cap outliers instead of removing them to preserve data
                    cleaned_data[col] = np.clip(cleaned_data[col], lower_bound, upper_bound)
        
        logger.info(f"Data cleaning completed. Final shape: {cleaned_data.shape}")
        return cleaned_data
    
    def engineer_features(self, data):
        """
        Create new features from existing ones to improve model performance
        
        Args:
            data (pandas.DataFrame): Cleaned dataset
            
        Returns:
            pandas.DataFrame: Dataset with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Make a copy to avoid modifying original data
        featured_data = data.copy()
        
        # Create age groups
        featured_data['age_group'] = pd.cut(featured_data['age'], 
                                          bins=[0, 30, 40, 50, 60, 100], 
                                          labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
        
        # Create balance categories
        featured_data['balance_category'] = pd.cut(featured_data['balance'], 
                                                 bins=[-1, 0, 50000, 150000, float('inf')], 
                                                 labels=['Zero', 'Low', 'Medium', 'High'])
        
        # Create credit score categories
        featured_data['credit_score_category'] = pd.cut(featured_data['credit_score'], 
                                                      bins=[0, 580, 670, 740, 800, 850], 
                                                      labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        
        # Create interaction features
        featured_data['balance_per_product'] = featured_data['balance'] / (featured_data['products_number'] + 1)
        featured_data['salary_to_balance_ratio'] = featured_data['estimated_salary'] / (featured_data['balance'] + 1)
        
        # Create tenure-age interaction
        featured_data['tenure_age_ratio'] = featured_data['tenure'] / featured_data['age']
        
        # Binary feature: high value customer (high balance and salary)
        featured_data['high_value_customer'] = ((featured_data['balance'] > featured_data['balance'].quantile(0.75)) & 
                                              (featured_data['estimated_salary'] > featured_data['estimated_salary'].quantile(0.75))).astype(int)
        
        logger.info(f"Feature engineering completed. New shape: {featured_data.shape}")
        return featured_data
    
    def encode_categorical_features(self, data, fit=True):
        """
        Encode categorical features using Label Encoding
        
        Args:
            data (pandas.DataFrame): Dataset with categorical features
            fit (bool): Whether to fit the encoders (True for training, False for inference)
            
        Returns:
            pandas.DataFrame: Dataset with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        # Make a copy to avoid modifying original data
        encoded_data = data.copy()
        
        # Identify categorical columns (excluding the target variable)
        categorical_cols = encoded_data.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        # Encode each categorical column
        for col in categorical_cols:
            if fit:
                # Fit and transform for training data
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                encoded_data[col] = self.label_encoders[col].fit_transform(encoded_data[col].astype(str))
            else:
                # Transform only for inference data
                if col in self.label_encoders:
                    # Handle unseen categories by using the most frequent class
                    encoded_data[col] = encoded_data[col].astype(str)
                    mask = encoded_data[col].isin(self.label_encoders[col].classes_)
                    encoded_data.loc[~mask, col] = self.label_encoders[col].classes_[0]
                    encoded_data[col] = self.label_encoders[col].transform(encoded_data[col])
        
        logger.info(f"Categorical encoding completed for {len(categorical_cols)} columns")
        return encoded_data
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """
        Scale numerical features using StandardScaler
        
        Args:
            X_train (pandas.DataFrame): Training features
            X_test (pandas.DataFrame, optional): Test features
            fit (bool): Whether to fit the scaler
            
        Returns:
            tuple: Scaled training and test features
        """
        logger.info("Scaling numerical features...")
        
        if fit:
            # Fit and transform training data
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        else:
            # Transform only
            X_train_scaled = pd.DataFrame(
                self.scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled
    
    def prepare_features_and_target(self, data):
        """
        Separate features and target variable, and remove unnecessary columns
        
        Args:
            data (pandas.DataFrame): Processed dataset
            
        Returns:
            tuple: Features (X) and target (y)
        """
        logger.info("Preparing features and target...")
        
        # Remove customer_id as it's not useful for prediction
        features_data = data.drop(['customer_id'], axis=1, errors='ignore')
        
        # Split features and target
        if self.target_column in features_data.columns:
            X = features_data.drop([self.target_column], axis=1)
            y = features_data[self.target_column]
        else:
            # For inference data without target
            X = features_data
            y = None
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Features prepared. Shape: {X.shape}")
        if y is not None:
            logger.info(f"Target prepared. Shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets
        
        Args:
            X (pandas.DataFrame): Features
            y (pandas.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Splitting data into train and test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Train churn rate: {y_train.mean():.2%}")
        logger.info(f"Test churn rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline from raw data to model-ready data
        
        Args:
            file_path (str): Path to the raw data CSV file
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test (all preprocessed)
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Step 1: Load data
        raw_data = self.load_data(file_path)
        
        # Step 2: Explore data
        self.explore_data(raw_data)
        
        # Step 3: Clean data
        cleaned_data = self.clean_data(raw_data)
        
        # Step 4: Engineer features
        featured_data = self.engineer_features(cleaned_data)
        
        # Step 5: Encode categorical features
        encoded_data = self.encode_categorical_features(featured_data, fit=True)
        
        # Step 6: Prepare features and target
        X, y = self.prepare_features_and_target(encoded_data)
        
        # Step 7: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        
        # Step 8: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)
        
        logger.info("Preprocessing pipeline completed successfully!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def preprocess_inference_data(self, data):
        """
        Preprocess new data for inference using fitted preprocessors
        
        Args:
            data (pandas.DataFrame): New data for inference
            
        Returns:
            pandas.DataFrame: Preprocessed data ready for prediction
        """
        logger.info("Preprocessing inference data...")
        
        # Step 1: Clean data
        cleaned_data = self.clean_data(data)
        
        # Step 2: Engineer features
        featured_data = self.engineer_features(cleaned_data)
        
        # Step 3: Encode categorical features (using fitted encoders)
        encoded_data = self.encode_categorical_features(featured_data, fit=False)
        
        # Step 4: Prepare features
        X, _ = self.prepare_features_and_target(encoded_data)
        
        # Step 5: Scale features (using fitted scaler)
        X_scaled, _ = self.scale_features(X, fit=False)
        
        logger.info("Inference data preprocessing completed!")
        
        return X_scaled