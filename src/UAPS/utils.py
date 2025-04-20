# Utility functions for the ML pipeline

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


def save_model(model, file_path, meta: dict = None):
    """Save the trained model and metadata to a file."""
    save_dict = {"model": model}
    if meta:
        save_dict["meta"] = meta
    joblib.dump(save_dict, file_path)


def load_model(file_path):
    """Load a trained model and metadata from a file."""
    save_dict = joblib.load(file_path)
    model = save_dict["model"]
    meta = save_dict.get("meta", {})
    return model, meta


def evaluate_model(y_true, y_pred):
    """Evaluate the model performance."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, recall, f1
