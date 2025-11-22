import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def define_target(df):
    """
    Creates the target variable 'IsViolent' based on crime category.
    """
    violent_categories = [
        'ASSAULT', 'ROBBERY', 'SEX OFFENSES FORCIBLE', 'KIDNAPPING', 'HOMICIDE', 'ARSON'
    ]
    
    df['IsViolent'] = df['Category'].apply(lambda x: 1 if x in violent_categories else 0)
    return df

def extract_temporal_features(df):
    """
    Extracts temporal features from the 'Dates' column.
    """
    df['Hour'] = df['Dates'].dt.hour
    df['Day'] = df['Dates'].dt.day
    df['Month'] = df['Dates'].dt.month
    df['Year'] = df['Dates'].dt.year
    df['DayOfWeek'] = df['Dates'].dt.dayofweek # 0=Monday, 6=Sunday
    
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Holidays
    cal = calendar()
    holidays = cal.holidays(start=df['Dates'].min(), end=df['Dates'].max())
    df['IsHoliday'] = df['Dates'].dt.date.astype('datetime64[ns]').isin(holidays).astype(int)
    
    return df

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def extract_contextual_features(df):
    """
    Extracts contextual features like Season.
    """
    df['Season'] = df['Month'].apply(get_season)
    return df

def extract_location_features(df, n_clusters=10, kmeans_model=None):
    """
    Extracts location features including K-Means clusters for high-crime zones.
    """
    if kmeans_model is None:
        # Fit mode
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['LocationCluster'] = kmeans.fit_predict(df[['X', 'Y']])
        return df, kmeans
    else:
        # Predict mode
        df['LocationCluster'] = kmeans_model.predict(df[['X', 'Y']])
        return df, kmeans_model

def preprocess_pipeline(df, is_train=True, kmeans_model=None):
    """
    Runs the full preprocessing pipeline.
    """
    df = extract_temporal_features(df)
    df = extract_contextual_features(df)
    
    # Location features (Clustering)
    df, kmeans_model = extract_location_features(df, kmeans_model=kmeans_model)
    
    if is_train:
        df = define_target(df)
        
    return df, kmeans_model

if __name__ == "__main__":
    # Test
    pass
