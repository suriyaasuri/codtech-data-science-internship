import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    print("Loading data...")
    return pd.read_csv(filepath)

def preprocess_data(df):
    print("Preprocessing data...")
    # Example: Convert column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    return df

def transform_data(df):
    print("Transforming data...")
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y):
    print("Splitting data...")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def save_processed_data(X_train, X_test, y_train, y_test):
    print("Saving processed data...")
    pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

def run_pipeline():
    df = load_data('iris.csv')
    df = preprocess_data(df)
    X, y = transform_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_processed_data(X_train, X_test, y_train, y_test)
    print("Pipeline executed successfully.")

if __name__ == "__main__":
    run_pipeline()
