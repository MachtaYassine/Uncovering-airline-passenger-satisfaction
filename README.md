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

### 2. Data Preprocessing

Preprocess the raw data (train and test):

```bash
python -m src.data_preprocessing
```

This will create `data/processed/train_processed.csv` and `data/processed/test_processed.csv` if they do not already exist.

---

### 3. Model Training & Benchmarking

Train a model and log results with MLflow. Example commands:

**Random Forest:**
```bash
python -m src.train --model-type random_forest --n-estimators 100
```

**Logistic Regression:**
```bash
python -m src.train --model-type logistic_regression --max-iter 200
```

**MLP (sklearn):**
```bash
python -m src.train --model-type mlp --hidden-dim 64 --max-iter 200
```

**PyTorch Neural Network:**
```bash
python -m src.train --model-type torch_nn --hidden-dim 64 --epochs 20 --lr 0.001
```

To disable MLflow logging, add `--no-mlflow`.

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