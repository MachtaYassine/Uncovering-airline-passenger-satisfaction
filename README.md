# Uncovering-airline-passenger-satisfaction

## Usage Instructions

### 1. Environment Setup

Install dependencies (using conda):

```bash
conda env create -f environment.yml
conda activate UAPS
```

Or with pip:

```bash
pip install -r requirements.txt
```

---

## Installation

Install the package in editable mode from the project root:

```bash
pip install -e .
```

---

## Usage

### Training

Train a model (example for PyTorch):

```bash
train --model-type torch_nn --hidden-dim 64 --epochs 20 --lr 0.001
```

See all options:
```bash
train -h
```

### Inference

Run inference on a trained model:

```bash
infer --experiment_id <EXPERIMENT_ID> --run_id <RUN_ID> --input_csv data/processed/test_processed.csv
```

See all options:
```bash
infer -h
```

### Clear MLflow Runs

Clear the mlruns directory (with confirmation):

```bash
clear_mlruns
```

---

### 2. Data Preprocessing

Preprocess the raw data (train and test):

```bash
python -m src.data_preprocessing
```

This will create `data/processed/train_processed.csv` and `data/processed/test_processed.csv` if they do not already exist.

---



### 4. MLflow Tracking

To compare and benchmark models, launch the MLflow UI:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

### 5. Inference

For tabular models (sklearn, torch), use your own inference logic or extend `src/inference.py` as needed.

For text models (example in `src/inference.py`):

```bash
python -m src.inference --model_path <path_to_model> --tokenizer_path <path_to_tokenizer> --text "Your text here"
```

---

### 6. Testing

Run tests with:

```bash
pytest tests/
```

---

**Note:**
- Adjust paths and parameters as needed for your use case.
- Make sure to preprocess data before training or inference.
- For custom models, update `src/model.py` and `src/inference.py` accordingly.
- You no longer need to use `python -m src.train` or `python -m src.inference`.
- All commands are available directly after `pip install -e .`.
- For more help on each command, use the `-h` flag (e.g., `train -h`).
- Make sure your data is preprocessed before training or inference.

---