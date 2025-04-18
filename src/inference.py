# Inference script for loading and using the trained model

import mlflow.pyfunc
import pandas as pd

class Inference:
    def __init__(self, model_path: str):
        self.model = mlflow.pyfunc.load_model(model_path)

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference script for tabular MLflow models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the MLflow model directory (e.g. mlruns/.../artifacts/model)")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV file with input data (must match training features)")

    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)
    inference = Inference(args.model_path)
    preds = inference.predict(data)
    print(preds)