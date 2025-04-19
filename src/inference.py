# Inference script for loading and using the trained model

import os
import argparse
import pandas as pd
import mlflow.pyfunc

class Inference:
    def __init__(self, model_path: str):
        self.model = mlflow.pyfunc.load_model(model_path)
        self.input_schema = self.model.metadata.get_input_schema()
      

    def predict(self, data: pd.DataFrame):
        
        preds = self.model.predict(data)
        # If output looks like logits/probs, convert to class labels
        if isinstance(preds, (pd.DataFrame, pd.DataFrame)) and preds.shape[1] > 1:
            # For logits, take argmax
            return preds.values.argmax(axis=1)
        return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for tabular MLflow models")
    parser.add_argument("--experiment_id", type=str, required=True, help="MLflow experiment ID")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV file with input data (must match training features)")
    args = parser.parse_args()

    model_path = os.path.join("mlruns", args.experiment_id, args.run_id, "artifacts", "model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder not found at {model_path}")

    data = pd.read_csv(args.input_csv)
    drop_cols = ['satisfaction']
    for col in drop_cols:
        if col in data.columns:
            data = data.drop(columns=[col])

    inference = Inference(model_path)
    preds = inference.predict(data)
    print(preds)