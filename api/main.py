from fastapi import FastAPI, UploadFile, File, Query
import pandas as pd
import mlflow.pyfunc
import os
import argparse
from typing import Optional
from UAPS.inference import Inference
from UAPS.data_preprocessing import preprocess_data

app = FastAPI()

# Global inference object, will be set in startup event
inference: Optional[Inference] = None

@app.on_event("startup")
def load_model_on_startup():
    global inference
    import sys
    from UAPS.inference import get_best_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", type=str, required=False, help="MLflow experiment ID")
    parser.add_argument("--run_id", type=str, required=False, help="MLflow run ID")
    args, _ = parser.parse_known_args()
    if args.experiment_id and args.run_id:
        model_path = os.path.join("mlruns", args.experiment_id, args.run_id, "artifacts", "model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model folder not found at {model_path}")
        print(f"[API] Loaded model from experiment_id={args.experiment_id}, run_id={args.run_id}")
    else:
        model_path, best_model_name = get_best_model()
        print(f"[API] No experiment_id/run_id provided. Loaded best model: {best_model_name}")
    inference = Inference(model_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...), preprocess: bool = Query(False, description="Whether to preprocess the input data before inference.")):
    global inference
    df = pd.read_csv(file.file)
    drop_cols = ['satisfaction']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    if preprocess:
        df = preprocess_data(df)
    preds = inference.predict(df)
    # Save predictions to CSV in api_prediction folder
    os.makedirs("api_prediction", exist_ok=True)
    output_path = os.path.join("api_prediction", f"predictions.csv")
    result_df = df.copy()
    result_df['prediction'] = preds
    result_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return {"predictions": preds.tolist()}

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Airline Passenger Satisfaction API!",
        "usage": "POST a CSV file to /predict to get predictions.",
        "docs": "/docs for OpenAPI UI"
    }