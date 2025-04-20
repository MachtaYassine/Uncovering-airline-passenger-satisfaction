import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Abstract method to build and train a model.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        pass


# Concrete Strategy for Linear Regression using scikit-learn
class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and trains a linear regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear Regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Linear Regression model with scaling.")

        # Creating a pipeline with standard scaling and linear regression
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Feature scaling
                ("model", LinearRegression()),  # Linear regression model
            ]
        )

        logging.info("Training Linear Regression model.")
        pipeline.fit(X_train, y_train)  # Fit the pipeline to the training data

        logging.info("Model training completed.")
        return pipeline


# Context Class for Model Building
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specific model building strategy.

        Parameters:
        strategy (ModelBuildingStrategy): The strategy to be used for model building.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder.

        Parameters:
        strategy (ModelBuildingStrategy): The new strategy to be used for model building.
        """
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Executes the model building and training using the current strategy.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)


if __name__ == "__main__":
    from src import handle_missing_values,feature_engineering,preprocessing



    # Load dataset
    try:
        df = pd.read_csv("/Users/rouablel/Uncovering-airline-passenger-satisfaction/Data/train.csv")

        logging.info("Dataset loaded successfully.")
    except FileNotFoundError:
        logging.error("File 'train.csv' not found.")
        exit()
    # Apply the missing value strategy
    strategy = handle_missing_values.FillSpecificColumnWithMedianStrategy(column='Arrival Delay in Minutes')
    df = strategy.handle(df)
    prep = preprocessing.Preprocessor(drop_cols=['ID'], cat_cols=['Gender', 'Customer Type', 'Type of Travel',
                                                    'Class', 'Inflight wifi service', 'Departure/Arrival time convenient',
                                                      'Ease of Online booking', 'Gate location', 'Food and drink',
                                                        'Online boarding', 'Seat comfort', 'Inflight entertainment',
                                                          'On-board service', 'Leg room service', 'Baggage handling',
                                                            'Checkin service', 'Inflight service', 'Cleanliness',"satisfaction"])
    df = prep.transform(df)


    categorical_columns = [c for c in df.columns if df[c].dtype.name == 'category']
    # list of your categorical columns
    data_describe = df[categorical_columns].nunique()
    
    binary_columns = [c for c in categorical_columns if data_describe[c] == 2]
    nonbinary_columns = [c for c in categorical_columns if data_describe[c] > 2]
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != "satisfaction"]

    # Binary Encoding
    binary_encoder = feature_engineering.FeatureEngineer(feature_engineering.BinaryEncoding(binary_columns))
    df = binary_encoder.apply_feature_engineering(df)

    # One-Hot Encoding
    onehot_encoder = feature_engineering.FeatureEngineer(feature_engineering.OneHotEncoding(nonbinary_columns))
    df = onehot_encoder.apply_feature_engineering(df)

    # Standard Scaling
    scaler = feature_engineering.FeatureEngineer(feature_engineering.LogTransformation(numerical_columns))
    df = scaler.apply_feature_engineering(df)

    # Define the target column
    target_column = "satisfaction"


    # Split the dataset into features and target
    if target_column not in df.columns:
        #ogging.error(f"Target column '{target_column}' not found in the dataset.")
        exit()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    

    # Build and train the model
    model_builder = ModelBuilder(LinearRegressionStrategy())
    trained_model = model_builder.build_model(X, y)

    # Print model coefficients
    print("Model Coefficients:")
    print(trained_model.named_steps['model'].coef_)
