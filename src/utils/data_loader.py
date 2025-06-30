# src/utils/data_loader.py

import os
import json
import pandas as pd
from typing import Dict

def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV {file_path}: {e}")
        return pd.DataFrame()

def load_json(file_path: str) -> Dict:
    """Load a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON {file_path}: {e}")
        return {}