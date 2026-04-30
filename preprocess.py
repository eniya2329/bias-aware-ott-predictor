import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def bias_correction(df):
    df['device_weight'] = df['device_type'].apply(
        lambda x: 1.5 if x == 'smart_tv' else 1.0
    )

    df['corrected_watch_time'] = df['watch_time'] * df['device_weight']
    return df

def feature_engineering(df):
    df['engagement_score'] = df['corrected_watch_time'] * df['completion_rate']
    return df