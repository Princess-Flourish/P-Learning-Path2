"""
Simple test to verify the complete ML pipeline works
"""

import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add src to path
sys.path.append('src')
from src.data_preprocessing import DataPreprocessor

def simple_pipeline_test():
    """
    Test the complete pipeline with a simple approach
    """
    print("=" * 60)
    print("Bank Churn MLOps Pipeline - Simple Test")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("\n1. Data Preprocessing...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        'data/Bank Customer Churn Prediction.csv', 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Churn rate: {y_train.mean():.2%}")
    
    # Step 2: Train model
    print("\n2. Model Training...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Step 3: Evaluate model
    print("\n3. Model Evaluation...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Step 4: Feature importance
    print("\n4. Top 5 Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Step 5: Save models
    print("\n5. Saving Models...")
    joblib.dump(model, 'models/churn_model_simple.pkl')
    joblib.dump(preprocessor, 'models/preprocessor_simple.pkl')
    print("   Models saved successfully!")
    
    # Step 6: Simple inference test (using original training features)
    print("\n6. Inference Test...")
    
    # Use the first 3 rows from test set for inference demo
    sample_X = X_test.head(3)
    sample_y = y_test.head(3)
    
    predictions = model.predict(sample_X)
    probabilities = model.predict_proba(sample_X)[:, 1]
    
    print("   Sample Predictions:")
    for i in range(len(sample_X)):
        actual = "Churn" if sample_y.iloc[i] == 1 else "No Churn"
        predicted = "Churn" if predictions[i] == 1 else "No Churn"
        prob = probabilities[i]
        
        print(f"   Customer {i+1}: Actual={actual}, Predicted={predicted}, Probability={prob:.2%}")
    
    # Step 7: Summary
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("✓ Data preprocessing: PASSED")
    print("✓ Model training: PASSED")
    print("✓ Model evaluation: PASSED")
    print("✓ Feature importance: PASSED")
    print("✓ Model saving: PASSED")
    print("✓ Inference: PASSED")
    print("\nKey Results:")
    print(f"✓ Model Accuracy: {metrics['Accuracy']:.1%}")
    print(f"✓ ROC-AUC Score: {metrics['ROC-AUC']:.1%}")
    print(f"✓ Top Feature: {feature_importance.iloc[0]['feature']}")
    print("\nYour ML pipeline is ready for Docker deployment!")
    
    return True

if __name__ == "__main__":
    try:
        simple_pipeline_test()
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()