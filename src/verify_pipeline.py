import pandas as pd
import numpy as np
import os
import joblib
from preprocessing import preprocess_pipeline

def verify_data_integrity():
    print("=== Starting Deep Verification ===")
    
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), '../data/crimedataset')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    # 1. Load Data
    print("\n[1] Loading Data...")
    if not os.path.exists(train_path):
        print("X Train file missing!")
        return
    
    df_train = pd.read_csv(train_path, parse_dates=['Dates'])
    print(f"OK Train Data Loaded: {df_train.shape}")
    
    # 2. Check for Duplicates
    print("\n[2] Checking for Duplicates...")
    duplicates = df_train.duplicated().sum()
    if duplicates > 0:
        print(f"! Warning: {duplicates} duplicate rows found in training data.")
    else:
        print("OK No duplicates found.")
        
    # 3. Class Balance
    print("\n[3] Checking Class Balance...")
    violent_categories = [
        'ASSAULT', 'ROBBERY', 'SEX OFFENSES FORCIBLE', 'KIDNAPPING', 'HOMICIDE', 'ARSON'
    ]
    df_train['IsViolent'] = df_train['Category'].apply(lambda x: 1 if x in violent_categories else 0)
    balance = df_train['IsViolent'].value_counts(normalize=True)
    print(f"Violent Crime Ratio: {balance.get(1, 0)*100:.2f}%")
    print(f"Non-Violent Crime Ratio: {balance.get(0, 0)*100:.2f}%")
    
    if balance.get(1, 0) < 0.1:
        print("! Severe Class Imbalance detected (<10% positive class). Model may struggle with Recall.")
        
    # 4. Check for Data Leakage (Train vs Test overlap)
    # Since test data might not have labels, we check for exact feature matches if test exists
    if os.path.exists(test_path):
        print("\n[4] Checking for Data Leakage (Train/Test Overlap)...")
        df_test = pd.read_csv(test_path, parse_dates=['Dates'])
        # Check intersection of Dates and Location
        # This is a heuristic; exact row match might be too slow for large data
        # We'll check a sample
        train_dates = set(df_train['Dates'].dt.date.unique())
        test_dates = set(df_test['Dates'].dt.date.unique())
        
        overlap = train_dates.intersection(test_dates)
        if len(overlap) > 0:
            print(f"! Warning: Found {len(overlap)} days present in BOTH Train and Test sets. Possible leakage if splitting by time.")
        else:
            print("OK No date overlap between Train and Test.")
            
    # 5. Verify Model Artifacts
    print("\n[5] Verifying Model Artifacts...")
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    required_files = ['best_model.pkl', 'label_encoders.pkl', 'kmeans.pkl']
    
    all_exist = True
    for f in required_files:
        fpath = os.path.join(models_dir, f)
        if os.path.exists(fpath):
            print(f"OK Found {f}")
            # Try loading
            try:
                joblib.load(fpath)
                print(f"   -> Successfully loaded {f}")
            except Exception as e:
                print(f"   X Failed to load {f}: {e}")
                all_exist = False
        else:
            print(f"X Missing {f}")
            all_exist = False
            
    if all_exist:
        print("\n=== Verification Complete: SYSTEM HEALTHY ===")
    else:
        print("\n=== Verification Complete: ISSUES DETECTED ===")

if __name__ == "__main__":
    verify_data_integrity()
