# Training script for ML pipeline

import argparse
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from UAPS.model import TorchMLP
import mlflow.pytorch

def train_torch_model(X_train, y_train, X_val, y_val, epochs=10, lr=1e-3, hidden_dim=64, use_progress_bar=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = X_train.astype(float)
    X_val = X_val.astype(float)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long).to(device)
    model = TorchMLP(X_train.shape[1], hidden_dim=hidden_dim, num_classes=len(set(y_train)))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if use_progress_bar:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_val_tensor).argmax(dim=1).cpu().numpy()
        acc = (preds == y_val_tensor.cpu().numpy()).mean()
        evaluation_results = {"accuracy": acc}
    return model, evaluation_results

def get_model(args, input_dim=None, num_classes=2):
    if args.model_type == 'random_forest':
        return RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    elif args.model_type == 'logistic_regression':
        return LogisticRegression(max_iter=args.max_iter, random_state=42)
    elif args.model_type == 'mlp':
        args.model_type = 'mlp (sklearn)'
        return MLPClassifier(hidden_layer_sizes=(args.hidden_dim,), max_iter=args.max_iter, random_state=42)
    elif args.model_type == 'torch_nn':
        return TorchMLP(input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}, please choose from ['random_forest', 'logistic_regression', 'mlp', 'torch_nn']")

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data):
    # Add preprocessing steps here
    return data

def train_model(X_train, y_train, X_val, y_val, args, use_progress_bar=False):

    model = get_model(args, input_dim=X_train.shape[1], num_classes=len(set(y_train)))
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    if use_progress_bar:
        # Simulate progress bar for fitting (since sklearn fit is not incremental)
        for _ in tqdm(range(1), desc="Fitting RandomForest"):
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    evaluation_results = {"accuracy":acc}
    return model, evaluation_results

def cross_validate_model(model, X, y, args, cv=5, batch_size=64, lr=1e-3, epochs=10, device='cuda'):
    """
    Perform cross-validation on the model.
    Args:
        model: PyTorch model (instance of nn.Module)
        X: Features (input data)
        y: Target labels
        cv: Number of folds for cross-validation
        batch_size: Mini-batch size for training
        lr: Learning rate for the optimizer
        epochs: Number of training epochs
        device: Device to run the model on ('cpu' or 'cuda')
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    fold = 1
    total_accuracy = 0

    # Loop over each fold
    for train_idx, val_idx in kf.split(X):
        print(f"Fold {fold}/{cv}")
        
        # Split the data into training and validation sets for this fold
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Train the model on kth fold
        if args.model_type == 'torch_nn':
            model, evaluation_results = train_torch_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim)
        else:
            model, evaluation_results = train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, args, use_progress_bar=True)
        accuracy = evaluation_results["accuracy"]
        print(f"Validation Accuracy for Fold {fold}: {accuracy:.4f}")
        fold += 1
        total_accuracy += accuracy

    # Calculate average accuracy across all folds
    avg_accuracy = total_accuracy / cv
    print(f"Average Cross-Validation Accuracy: {avg_accuracy:.4f}")



def main():
    import argparse
    import mlflow
    from UAPS.data_preprocessing import load_data, clean_data, preprocess_data
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument("--train_path", type=str, default="data/raw/train.csv", help="Path to the training CSV file.")
    parser.add_argument("--test_path", type=str, default="data/raw/test.csv", help="Path to the test CSV file.")
    parser.add_argument("--target_column", type=str, default="satisfaction", help="Name of the target column.")
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "logistic_regression", "mlp", "torch_nn"], help="Type of model to train.")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators for RandomForest.")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations for LogisticRegression/MLP.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer size for MLP/TorchNN.")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for TorchNN.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for TorchNN.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking and run regular training.")
    args = parser.parse_args()

    print("Loading training data...")
    train_df = load_data(args.train_path)
    print("Cleaning training data...")
    train_df = clean_data(train_df)
    print("Preprocessing training data...")
    train_df = preprocess_data(train_df)
    X_train = train_df.drop(columns=[args.target_column])
    y_train = train_df[args.target_column]
    print("Loading test data...")
    test_df = load_data(args.test_path)
    print("Cleaning test data...")
    test_df = clean_data(test_df)
    print("Preprocessing test data...")
    test_df = preprocess_data(test_df)
    X_test = test_df.drop(columns=[args.target_column])
    y_test = test_df[args.target_column]

    if args.no_mlflow:
        print(f"Running training WITHOUT MLflow using model: {args.model_type}")
        if args.model_type == 'torch_nn':
            print("Training model...")
            model, evaluation_results = train_torch_model(X_train, y_train, X_test, y_test, epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim)
            acc = evaluation_results["accuracy"]
            print("Cross validation...")
            cross_validate_model(model, X_train, y_train, args)
            print(f"Torch NN Final accuracy on test set: {acc:.4f}")
        else:
            print("Training model...")
            model, evaluation_results = train_model(X_train, y_train, X_test, y_test, args, use_progress_bar=True)
            acc = evaluation_results["accuracy"]
            print("Cross validation...")
            cross_validate_model(model, X_train, y_train, args)
            print(f"Final accuracy on test set: {acc:.4f}")
        print("Experiment completed.")
    else:
        mlflow.set_experiment("Default")
        mlflow.start_run(run_name=f"{args.model_type}_run")
        print(f"Running training WITH MLflow using model: {args.model_type}")
        # Log model-specific parameters and metadata
        mlflow.log_param("model_type", args.model_type)
        mlflow.set_tag("mlflow.runName", f"{args.model_type}_run")
        mlflow.set_tag("model_name", args.model_type)
        mlflow.set_tag("input_dim", X_train.shape[1])
        mlflow.set_tag("num_classes", len(set(y_train)))
        mlflow.set_tag("train_rows", X_train.shape[0])
        mlflow.set_tag("test_rows", X_test.shape[0])
        mlflow.set_tag("feature_names", str(list(X_train.columns)))
        if args.model_type == 'random_forest':
            mlflow.log_param("n_estimators", args.n_estimators)
        if args.model_type in ['mlp', 'torch_nn']:
            mlflow.log_param("hidden_dim", args.hidden_dim)
        if args.model_type in ['mlp', 'logistic_regression']:
            mlflow.log_param("max_iter", args.max_iter)
        if args.model_type == 'torch_nn':
            mlflow.log_param("epochs", args.epochs)
            mlflow.log_param("lr", args.lr)
            print("Training model...")
            model, evaluation_results = train_torch_model(X_train, y_train, X_test, y_test, epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim)
            acc = evaluation_results["accuracy"]
            print("Cross validation...")
            cross_validate_model(model, X_train, y_train, args)
            print(f"Torch NN Final accuracy on test set: {acc:.4f}")
            model.cpu()
            input_example = X_test.astype('float32').iloc[:2]
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                input_example=input_example
            )
            mlflow.set_tag("model_file", "mlflow_pytorch")  
        else:
            print("Training model...")
            model, evaluation_results = train_model(X_train, y_train, X_test, y_test, args, use_progress_bar=True)
            acc = evaluation_results["accuracy"]
            print("Cross validation...")
            cross_validate_model(model, X_train, y_train, args)
            print(f"Final accuracy on test set: {acc:.4f}")
            mlflow.log_param("model_type", args.model_type)
            mlflow.log_metric("accuracy", acc)
            signature = infer_signature(X_train, model.predict(X_train))
            input_example = X_train.iloc[:1]
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)
            mlflow.set_tag("model_file", "mlflow_sklearn")
        active_run = mlflow.active_run()
        if active_run is not None:
            print(f"Experiment ID: {active_run.info.experiment_id}")
            print(f"Run ID: {active_run.info.run_id}")
        mlflow.end_run()
        print("Experiment completed.")

if __name__ == "__main__":
    main()