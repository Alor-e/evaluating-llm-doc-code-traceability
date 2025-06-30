# src/utils/metrics.py

from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Set, Dict
from sklearn.preprocessing import MultiLabelBinarizer

def calculate_metrics(y_true: List[Set[int]], y_pred: List[Set[int]]) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1-Score for multi-label classification.

    Args:
        y_true (List[Set[int]]): List of true artifact_id sets for each document.
        y_pred (List[Set[int]]): List of predicted artifact_id sets for each document.

    Returns:
        Dict[str, float]: Dictionary with precision, recall, and f1-score.
    """
    # Initialize the MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    # Calculate metrics using micro averaging suitable for multi-label
    precision = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }