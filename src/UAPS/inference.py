import os
import argparse
import pandas as pd
import json
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
    parser.add_argument("--experiment_id", type=str, required=True, help="Training MLflow experiment ID")
    parser.add_argument("--run_id", type=str, required=True, help="Training MLflow run ID")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV file with input data")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save prediction results and metadata")
    parser.add_argument("--inference_experiment_name", type=str, default="inference", help="MLflow experiment name for logging inference run")
    args = parser.parse_args()

    # Set and create experiment for inference
    mlflow.set_experiment(args.inference_experiment_name)

    os.makedirs(args.output_dir, exist_ok=True)

    with mlflow.start_run(run_name="inference_run") as run:
        # Log training metadata
        mlflow.log_param("training_experiment_id", args.experiment_id)
        mlflow.log_param("training_run_id", args.run_id)
        mlflow.log_param("training_input_csv", args.input_csv)
        mlflow.log_param("output_dir", args.output_dir)

        # Save those parameters as a file
        params = {
            "training_experiment_id": args.experiment_id,
            "training_run_id": args.run_id,
            "input_csv": args.input_csv
        }
        params_file = os.path.join(args.output_dir, "inference_params.json")
        with open(params_file, "w") as f:
            json.dump(params, f, indent=4)
        mlflow.log_artifact(params_file)

        # Load model
        model_path = os.path.join("mlruns", args.experiment_id, args.run_id, "artifacts", "model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model folder not found at {model_path}")

        # Read input
        data = pd.read_csv(args.input_csv)
        if 'satisfaction' in data.columns:
            data = data.drop(columns=['satisfaction'])

        # Inference
        inference = Inference(model_path)
        preds = inference.predict(data)

        # Save and log predictions
        preds_df = pd.DataFrame(preds, columns=["prediction"])
        pred_file = os.path.join(args.output_dir, "predictions.csv")
        preds_df.to_csv(pred_file, index=False)
        mlflow.log_artifact(pred_file)

        print(f" Inference completed and logged in run {run.info.run_id}")

if __name__ == "__main__":
    main()