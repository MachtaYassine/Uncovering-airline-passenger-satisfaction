import os
import argparse
import pandas as pd
import json
import mlflow.pyfunc
from UAPS.data_preprocessing import preprocess_data

def get_best_model():
    import re
    models_dir = "mlruns/models"
    pattern = re.compile(r"best_model_(\d+\.\d+)")
    model_dirs = [name for name in os.listdir(models_dir) if pattern.match(name)]

    if not model_dirs:
        raise FileNotFoundError("No local fallback models found in 'mlruns/models'.")

    best_model_name = max(model_dirs, key=lambda name: float(pattern.match(name).group(1)))
    model_path = os.path.join(models_dir, best_model_name)
    return model_path, best_model_name

class Inference:
    def __init__(self, model_path: str, preprocess: bool = False):
        self.model = mlflow.pyfunc.load_model(model_path)
        self.model_type = None
        self.preprocess = preprocess
        try:
            tags = self.model.metadata.run_id and mlflow.get_run(self.model.metadata.run_id).data.tags
            if tags and "model_file" in tags:
                self.model_type = tags["model_file"]
                print(f"Detected model type from tags: {self.model_type}")
        except Exception:
            pass
        if self.model_type is None:
            flavors = self.model.metadata.flavors
            if "pytorch" in flavors:
                self.model_type = "mlflow_pytorch"
                print("Detected PyTorch model")
            elif "sklearn" in flavors:
                self.model_type = "mlflow_sklearn"
                print("Detected sklearn model")
        
    def predict(self, data: pd.DataFrame):
        if self.preprocess:
            data = preprocess_data(data)
        if self.model_type == "mlflow_pytorch":
            data = data.astype("float32")
        preds = self.model.predict(data)
        if isinstance(preds, (pd.DataFrame, pd.DataFrame)) and preds.shape[1] > 1:
            return preds.values.argmax(axis=1)
        return preds


def main():
    parser = argparse.ArgumentParser(description="Inference script for tabular MLflow models")
    parser.add_argument("--experiment_id", type=str, help="Training MLflow experiment ID")
    parser.add_argument("--run_id", type=str, help="Training MLflow run ID")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV file with input data")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save prediction results and metadata")
    parser.add_argument("--inference_experiment_name", type=str, default="inference", help="MLflow experiment name for logging inference run")
    parser.add_argument("--preprocess", action="store_true", help="Whether to preprocess the input data before inference.")
    args = parser.parse_args()

    # Set and create experiment for inference
    mlflow.set_experiment(args.inference_experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)

    with mlflow.start_run(run_name="inference_run") as run:
        # Log training metadata
        mlflow.log_param("training_experiment_id", args.experiment_id)
        mlflow.log_param("training_run_id", args.run_id)
        mlflow.log_param("input_csv", args.input_csv)
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

        # Try to load model from run
        if args.experiment_id and args.run_id:
            model_path = os.path.join("mlruns", args.experiment_id, args.run_id, "artifacts", "model")
            print(f"Loaded model from training run path: {model_path}")

        else:
            print(f"No Model provided, using best saved model instead.")
            # Fallback to best model
            model_path, best_model_name = get_best_model()
            mlflow.log_param("fallback_model_used", best_model_name)
            print(f"Using fallback model: {best_model_name}")

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

        print(f"Inference completed and logged in run {run.info.run_id}")

if __name__ == "__main__":
    main()