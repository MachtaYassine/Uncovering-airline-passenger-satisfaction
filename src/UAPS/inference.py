import os
import argparse
import pandas as pd
import mlflow.pyfunc

class Inference:
    def __init__(self, model_path: str):
        self.model = mlflow.pyfunc.load_model(model_path)
        # Detect model type from MLflow tags or flavor
        self.model_type = None
        try:
            # Try to get from MLflow tags
            tags = self.model.metadata.run_id and mlflow.get_run(self.model.metadata.run_id).data.tags
            if tags and "model_file" in tags:
                self.model_type = tags["model_file"]
                print(f"Detected model type from tags: {self.model_type}")
        except Exception:
            pass
        # Fallback: check flavors
        if self.model_type is None:
            flavors = self.model.metadata.flavors
            if "pytorch" in flavors:
                print("Detected PyTorch model")
                self.model_type = "mlflow_pytorch"
            elif "sklearn" in flavors:
                print("Detected sklearn model")
                self.model_type = "mlflow_sklearn"
        
        
    def predict(self, data: pd.DataFrame):
        # Handle dtype conversion based on model type
        if self.model_type == "mlflow_pytorch":
            data = data.astype("float32")
        preds = self.model.predict(data)
        # If output looks like logits/probs, convert to class labels
        if isinstance(preds, (pd.DataFrame, pd.DataFrame)) and preds.shape[1] > 1:
            return preds.values.argmax(axis=1)
        return preds

def main():
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

if __name__ == "__main__":
    main()