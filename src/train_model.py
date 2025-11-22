import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from data_loader import load_data
from preprocessing import preprocess_pipeline

def train_and_evaluate():
    # Load Data
    data_dir = os.path.join(os.path.dirname(__file__), '../data/crimedataset')
    train_df, _ = load_data(data_dir)
    
    # Preprocess
    print("Preprocessing data...")
    # Pass None for kmeans_model to trigger fitting
    df, kmeans_model = preprocess_pipeline(train_df, is_train=True, kmeans_model=None)
    
    # Feature Selection
    features = ['Hour', 'Day', 'Month', 'Year', 'DayOfWeek', 'IsWeekend', 'IsHoliday', 'LocationCluster', 'PdDistrict', 'Season']
    target = 'IsViolent'
    
    # Encoding Categorical Variables
    print("Encoding categorical features...")
    le_dict = {}
    for col in ['PdDistrict', 'Season']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    X = df[features]
    y = df[target]
    
    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec}
        print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
        
        if acc > best_score:
            best_score = acc
            best_model = model
            
    # Save Artifacts
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Saving best model: {best_model.__class__.__name__}")
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    joblib.dump(le_dict, os.path.join(models_dir, 'label_encoders.pkl'))
    joblib.dump(kmeans_model, os.path.join(models_dir, 'kmeans.pkl'))
    
    return results

if __name__ == "__main__":
    train_and_evaluate()
