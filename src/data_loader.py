import pandas as pd
import os

def load_data(data_dir):
    """
    Loads train and test data from the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing 'train.csv' and 'test.csv'.
        
    Returns:
        tuple: (train_df, test_df)
    """
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")
        
    print("Loading training data...")
    train_df = pd.read_csv(train_path, parse_dates=['Dates'])
    print(f"Training data loaded: {train_df.shape}")
    
    print("Loading test data...")
    test_df = pd.read_csv(test_path, parse_dates=['Dates'])
    print(f"Test data loaded: {test_df.shape}")
    
    return train_df, test_df

if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(__file__), '../data/crimedataset')
    try:
        train, test = load_data(data_dir)
        print(train.head())
    except Exception as e:
        print(e)
