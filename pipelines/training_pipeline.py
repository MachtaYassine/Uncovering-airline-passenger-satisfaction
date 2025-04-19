from zenml import Model, pipeline

# Use your actual step implementations from `src/`
from src.handle_missing_values import handle_missing_values_step
from src.feature_engineering import feature_engineering_step
from src.data_splitter import data_splitter_step
from src.model_building import model_building_step
from src.model_evaluator import model_evaluator_step

@pipeline(
    model=Model(name="satisfaction_predictor")
)
def ml_pipeline():
    """End-to-end pipeline for predicting airline passenger satisfaction."""

    

    # Handle missing values (e.g., NaNs in satisfaction scores or delays)
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering: Apply transformations to numeric columns
    engineered_data = feature_engineering_step(
        filled_data,
        strategy="standard",  # could be "log" or "minmax" too
        features=[
            "Flight distance",
            "Departure delay in minutes",
            "Arrival delay in minutes"
        ]
    )

    # Outlier Removal: Based on satisfaction or flight distance
    clean_data = outlier_detection_step(
        engineered_data,
        column_name="Satisfaction"
    )

    # Data split
    X_train, X_test, y_train, y_test = data_splitter_step(
        clean_data,
        target_column="Satisfaction"
    )

    # Train model
    model = model_building_step(
        X_train=X_train,
        y_train=y_train
    )

    # Evaluate model
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model,
        X_test=X_test,
        y_test=y_test
    )

    return model
