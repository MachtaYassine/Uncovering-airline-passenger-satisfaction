import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Abstract Base Strategy ===
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# === Log Transformation Strategy ===
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info("Log transformation completed.")
        return df_transformed

# === Standard Scaling Strategy ===
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed

# === Min-Max Scaling Strategy ===
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed

# === One-Hot Encoding Strategy ===
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features)
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed

# === Binary Encoding Strategy ===
class BinaryEncoding(FeatureEngineeringStrategy):
    def __init__(self, binary_columns):
        self.binary_columns = binary_columns

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying binary encoding to columns: {self.binary_columns}")
        df_encoded = df.copy()
        for col in self.binary_columns:
            df_encoded[col] = df_encoded[col].astype("object")
            unique_vals = list(df_encoded[col].unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            df_encoded[col] = df_encoded[col].map(mapping)
        logging.info("Binary encoding completed.")
        return df_encoded

# === Feature Engineering Context ===
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)

# === Example Usage ===
if __name__ == "__main__":
    # Example DataFrame (replace this with your actual dataset)
    df = pd.read_csv("/Users/rouablel/Uncovering-airline-passenger-satisfaction/Data/train.csv")
    
    # Assume these are predefined
    categorical_columns = [c for c in df.columns if df[c].dtype.name == 'category']
    # list of your categorical columns
    data_describe = df[categorical_columns].nunique()
    
    binary_columns = [c for c in categorical_columns if data_describe[c] == 2]
    nonbinary_columns = [c for c in categorical_columns if data_describe[c] > 2]
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != "satisfaction"]

    # Binary Encoding
    binary_encoder = FeatureEngineer(BinaryEncoding(binary_columns))
    df = binary_encoder.apply_feature_engineering(df)

    # One-Hot Encoding
    onehot_encoder = FeatureEngineer(OneHotEncoding(nonbinary_columns))
    df = onehot_encoder.apply_feature_engineering(df)

    # Standard Scaling
    scaler = FeatureEngineer(LogTransformation(numerical_columns))
    df = scaler.apply_feature_engineering(df)

    print("Transformed dataset shape:", df.shape)
