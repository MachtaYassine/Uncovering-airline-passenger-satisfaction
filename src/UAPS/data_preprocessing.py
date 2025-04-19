# Data preprocessing functions for the ML pipeline

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset by handling missing values and removing duplicates."""
    df = df.drop_duplicates()
    # Fill missing values in 'Arrival Delay in Minutes' with median if present
    if 'Arrival Delay in Minutes' in df.columns:
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())
    return df

def preprocess_data(df):
    """Improved preprocessing for airline passenger satisfaction dataset."""
    # Drop id column if present
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    # Drop Unnamed: 0 column if present (artifact from pandas CSV)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    # Define columns
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    ordinal_columns = [
        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
    ]
    numerical_columns = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    # Only keep columns that exist in df
    categorical_columns = [c for c in categorical_columns if c in df.columns]
    ordinal_columns = [c for c in ordinal_columns if c in df.columns]
    numerical_columns = [c for c in numerical_columns if c in df.columns]
    # Impute missing values
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in ordinal_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    # Encode categorical columns
    # Gender: Male=1, Female=0
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    # Customer Type: Loyal=1, disloyal=0
    if 'Customer Type' in df.columns:
        df['Customer Type'] = df['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
    # Type of Travel: Business=1, Personal=0
    if 'Type of Travel' in df.columns:
        df['Type of Travel'] = df['Type of Travel'].map({'Business travel': 1, 'Personal Travel': 0})
    # Class: One-hot encode
    if 'Class' in df.columns:
        df = pd.get_dummies(df, columns=['Class'], prefix='Class')
    # satisfaction: satisfied=1, neutral or dissatisfied=0
    if 'satisfaction' in df.columns:
        df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
    # Standardize numerical and ordinal columns
    for col in numerical_columns + ordinal_columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    # Convert all integer columns to float for MLflow compatibility
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = df[col].astype(float)
    return df

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Preprocess raw airline passenger data.")
    parser.add_argument('--input', type=str, default="data/raw/train.csv", help="Path to the raw input CSV file.")
    parser.add_argument('--output', type=str, default="data/processed/train_processed.csv", help="Path to save the processed CSV file.")
    parser.add_argument('--test_input', type=str, default="data/raw/test.csv", help="Path to the raw test CSV file.")
    parser.add_argument('--test_output', type=str, default="data/processed/test_processed.csv", help="Path to save the processed test CSV file.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode to see preprocessing steps.")
    parser.add_argument('--override', action='store_true', help="Override and recompute processed files even if they exist.")
    args = parser.parse_args()

    # Only process if output does not exist or override is set
    if not os.path.exists(args.output) or args.override:
        print(f"Loading data from {args.input} ...")
        df = load_data(args.input)
        print("Cleaning data ...")
        df = clean_data(df)
        print("Preprocessing data ...")
        try:
            if args.debug:
                processed = debug_preprocess_data(df)
            else:
                processed = preprocess_data(df)
            print("Saving processed data ...")
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            processed.to_csv(args.output, index=False)
            print(f"Processed data saved to {args.output}")
        except Exception as e:
            print(f"Error during preprocessing or saving: {e}")
    else:
        print(f"{args.output} already exists. Skipping train processing.")

    # Only process test if output does not exist or override is set
    if args.test_input and args.test_output and (not os.path.exists(args.test_output) or args.override):
        print(f"Loading test data from {args.test_input} ...")
        df_test = load_data(args.test_input)
        print("Cleaning test data ...")
        df_test = clean_data(df_test)
        print("Preprocessing test data ...")
        try:
            processed_test = preprocess_data(df_test)
            # Ensure columns match train processed (except possibly 'satisfaction')
            if os.path.exists(args.output):
                processed = load_data(args.output)
                missing_cols = set(processed.columns) - set(processed_test.columns)
                for col in missing_cols:
                    if col != 'satisfaction':
                        processed_test[col] = 0
                processed_test = processed_test[[c for c in processed.columns if c in processed_test.columns]]
            os.makedirs(os.path.dirname(args.test_output), exist_ok=True)
            processed_test.to_csv(args.test_output, index=False)
            print(f"Processed test data saved to {args.test_output}")
        except Exception as e:
            print(f"Error during test preprocessing or saving: {e}")
    elif os.path.exists(args.test_output) and not args.override:
        print(f"{args.test_output} already exists. Skipping test processing.")

if __name__ == "__main__":
    main()