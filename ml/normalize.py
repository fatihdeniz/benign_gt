import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def normalize_data(df, label_encoder_path='label_encoder.pkl', scaler_path='scaler.pkl'):
    # Separate categorical and numerical columns
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    numerical_cols = [col for col in df.columns if col not in categorical_cols and col not in binary_cols]

    # One-hot encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])
        label_encoders[col] = label_encoder

    # Save label encoders to a file
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoders, f)

    # Standardize numerical and binary columns
    scalers = {}
    for col in numerical_cols + binary_cols:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        scalers[col] = scaler

    # Save scalers to a file
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)

    return df

def apply_normalization(df, label_encoder_path='label_encoder.pkl', scaler_path='scaler.pkl'):
    # Load label encoders from file
    with open(label_encoder_path, 'rb') as f:
        label_encoders = pickle.load(f)

    # Apply label encoders to categorical columns
    categorical_cols = [col for col in df.columns if col in label_encoders]
    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])

    # Load scalers from file
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)

    # Apply scalers to numerical and binary columns
    numerical_cols = [col for col in df.columns if col in scalers and col not in categorical_cols]
    binary_cols = [col for col in df.columns if col in scalers and col in categorical_cols]
    for col in numerical_cols + binary_cols:
        df[col] = scalers[col].transform(df[col].values.reshape(-1, 1))

    return df
