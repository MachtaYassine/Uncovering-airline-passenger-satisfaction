import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import pytest
from src.train import train_model, evaluate_model
from sklearn.ensemble import RandomForestClassifier

def test_train_model():
    X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [2, 3, 4, 5]})
    y = [0, 1, 0, 1]
    model = train_model(X, y, use_progress_bar=False)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    X = pd.DataFrame({'a': [1, 2], 'b': [2, 3]})
    y = [0, 1]
    model = RandomForestClassifier().fit(X, y)
    acc = evaluate_model(model, X, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
