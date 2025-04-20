import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import pytest
from src import data_preprocessing

def test_load_data():
    df = data_preprocessing.load_data('data/raw/train.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_clean_data():
    df = pd.DataFrame({'A': [1, 1, 2], 'Arrival Delay in Minutes': [10, None, 30]})
    cleaned = data_preprocessing.clean_data(df)
    assert cleaned.isnull().sum().sum() == 0 or 'Arrival Delay in Minutes' not in cleaned.columns
    assert cleaned.duplicated().sum() == 0

def test_preprocess_data():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z'],
        'satisfaction': [0, 1, 0]
    })
    processed = data_preprocessing.preprocess_data(df)
    assert isinstance(processed, pd.DataFrame)
    assert 'satisfaction' in processed.columns
