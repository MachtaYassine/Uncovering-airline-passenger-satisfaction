from fastapi import FastAPI, UploadFile, File
import pandas as pd
import mlflow.pyfunc
import os
import argparse
from typing import Optional
from UAPS.inference import Inference

app = FastAPI()

# Global inference object, will be set in startup event
inference: Optional[Inference] = None

@app.on_event("startup")
def load_model_on_startup():
    global inference
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", type=str, required=True, help="MLflow experiment ID")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    args, _ = parser.parse_known_args()
    model_path = os.path.join("mlruns", args.experiment_id, args.run_id, "artifacts", "model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder not found at {model_path}")
    inference = Inference(model_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global inference
    df = pd.read_csv(file.file)
    drop_cols = ['satisfaction']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    preds = inference.predict(df)
    return {"predictions": preds.tolist()}

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Airline Passenger Satisfaction API!",
        "usage": "POST a CSV file to /predict to get predictions.",
        "docs": "/docs for OpenAPI UI"
    }