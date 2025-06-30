#!/usr/bin/env python3
"""
bm25_baseline_all_artifacts_granularity_precision_recall.py

A BM25 baseline for software traceability that:
 - Loads dataset_name.json.
 - Builds document contexts (text snippet + location) and code artifact contexts (content + location).
 - Builds all document–artifact pairs (one-to-many).
 - Uses BM25 (via BM25Okapi) to compute similarity scores for each document (as a query)
   against all code artifacts (as the corpus).
 - Performs a grid search over thresholds (from 0.0 to max BM25 score) to find the best threshold
   that maximizes F1.
 - Computes overall and per-granularity precision, recall, and F1.
 - Aggregates results over multiple runs (with dataset shuffling between runs).
 - Outputs results in JSON format matching CodeBERT/TF–IDF.
 
Usage Example:
  python bm25_baseline_all_artifacts_granularity_precision_recall.py \
    --dataset data/unity_catalog.json \
    --n_runs 3 \
    --step 0.05
"""

import os
import json
import argparse
import numpy as np
import random
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

#                           LOADING & PAIR CREATION

def load_dataset(dataset_path: str) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_doc_and_artifact_contexts(data: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Builds contexts for documents and code artifacts.
    
    Documents: Each document context is built by combining the text snippet and its location.
    Code artifacts: Each artifact is represented as a dict with keys:
       "title", "content", "location", "traceability_granularity", and a combined "context" (content + location).
       
    Returns:
      doc_texts: List of document context strings.
      code_artifacts: List of artifact dictionaries.
    """
    doc_texts = []
    code_artifacts = {}
    for rec in data:
        doc_snippet = rec["document"].get("text", "")
        doc_location = rec["document"].get("location", "")
        doc_context = f"{doc_snippet} [Location: {doc_location}]"
        doc_texts.append(doc_context)
        
        for art in rec["artifacts"]:
            title = art.get("title", "").strip()
            content = art.get("content", "").strip()
            granularity = art.get("traceability_granularity", "").strip()
            if title and title not in code_artifacts:
                artifact_location = art.get("location", "").strip()
                combined_context = f"{content} [Location: {artifact_location}]"
                code_artifacts[title] = {
                    "title": title,
                    "content": content,
                    "location": artifact_location,
                    "traceability_granularity": granularity,
                    "context": combined_context
                }
    code_artifact_list = list(code_artifacts.values())
    return doc_texts, code_artifact_list

def build_all_pairs(data: List[Dict], code_titles: List[str]) -> List[Dict]:
    """
    Builds all document–artifact pairs with labels.
    
    Each pair is represented as a dictionary:
      {
        "doc_idx": int,
        "code_idx": int,
        "label": 0 or 1
      }
    
    Label is 1 if the artifact's title is in the document's "artifacts" list, else 0.
    """
    # Corrected: code_titles is a list of strings.
    title_to_idx = {t: i for i, t in enumerate(code_titles)}
    pairs = []
    for doc_idx, rec in enumerate(data):
        positive_titles = set(a["title"].strip() for a in rec["artifacts"] if a.get("title"))
        for title, code_idx in title_to_idx.items():
            label = 1 if title in positive_titles else 0
            pairs.append({
                "doc_idx": doc_idx,
                "code_idx": code_idx,
                "label": label
            })
    return pairs

#                    BM25 SIMILARITY

def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer.
    """
    return text.split()

def build_bm25_similarity_matrix(doc_texts: List[str], code_artifacts: List[Dict]) -> np.ndarray:
    """
    Constructs BM25 similarity scores.
    For code artifacts, tokenize each artifact's context and build a corpus.
    For each document, tokenize its context and use BM25 to score against the code corpus.
    
    Returns:
      similarity_matrix: numpy array of shape (n_docs, n_artifacts)
    """
    # Build corpus from code artifacts.
    corpus = [tokenize(art["context"]) for art in code_artifacts]
    bm25 = BM25Okapi(corpus)
    
    similarity_matrix = []
    for doc in doc_texts:
        query_tokens = tokenize(doc)
        scores = bm25.get_scores(query_tokens)
        similarity_matrix.append(scores)
    similarity_matrix = np.array(similarity_matrix)
    return similarity_matrix

#                   GRID SEARCH THRESHOLD & METRICS

def precision_recall_f1(preds: np.ndarray, labels: np.ndarray):
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def grid_search_threshold(similarity_scores: np.ndarray, labels: np.ndarray, step: float = 0.05):
    # Use a threshold range from 0.0 to the maximum BM25 score.
    max_score = similarity_scores.max()
    best_threshold = 0.0
    best_f1 = -1
    best_precision = 0.0
    best_recall = 0.0
    thresholds = np.arange(0.0, max_score + step, step)
    threshold_metrics = []
    for t in thresholds:
        preds = (similarity_scores >= t).astype(int)
        p, r, f1 = precision_recall_f1(preds, labels)
        threshold_metrics.append({
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1)
        })
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = p
            best_recall = r
    return best_threshold, best_f1, best_precision, best_recall, threshold_metrics

#            SINGLE RUN (ALL-ARTIFACTS) & MULTI-RUN

def run_bm25_baseline_once(dataset_path: str, step: float = 0.05):
    # Load data and shuffle to introduce randomness.
    data = load_dataset(dataset_path)
    random.shuffle(data)
    
    # Build contexts for documents and code artifacts.
    doc_texts, code_artifacts = build_doc_and_artifact_contexts(data)
    
    # Build all pairs.
    code_titles = [art["title"] for art in code_artifacts]
    pairs = build_all_pairs(data, code_titles)
    
    # Add granularity to each pair.
    # Create mapping from title to granularity.
    title_to_gran = {art["title"]: art.get("traceability_granularity", "Unknown") for art in code_artifacts}
    for pair in pairs:
        code_idx = pair["code_idx"]
        title = code_titles[code_idx]
        pair["granularity"] = title_to_gran.get(title, "Unknown")
    
    # Compute BM25 similarity matrix.
    similarity_matrix = build_bm25_similarity_matrix(doc_texts, code_artifacts)
    
    # Extract similarity scores and labels.
    similarity_scores = []
    labels_list = []
    for pair in pairs:
        d_i = pair["doc_idx"]
        c_i = pair["code_idx"]
        label = pair["label"]
        sim = similarity_matrix[d_i, c_i]
        similarity_scores.append(sim)
        labels_list.append(label)
    similarity_scores = np.array(similarity_scores)
    labels_array = np.array(labels_list)
    
    best_threshold, best_f1, best_precision, best_recall, threshold_metrics = grid_search_threshold(similarity_scores, labels_array, step)
    preds = (similarity_scores >= best_threshold).astype(int)
    total_predicted = int(np.sum(preds))
    
    run_result = {
        "best_threshold": float(best_threshold),
        "best_f1": float(best_f1),
        "best_precision": float(best_precision),
        "best_recall": float(best_recall),
        "threshold_metrics": threshold_metrics,
        "total_predicted_links": total_predicted
    }
    
    # Build predicted links counts by granularity.
    total_predicted_links, predicted_by_granularity = build_predicted_links(similarity_matrix, pairs, best_threshold, [art.get("traceability_granularity", "Unknown") for art in code_artifacts])
    run_result["total_predicted_links"] = total_predicted_links
    run_result["predicted_links_by_granularity"] = predicted_by_granularity
    
    # Compute per-granularity metrics.
    granularity_counts = {}
    # We iterate over pairs and corresponding similarity scores.
    for pair, sim in zip(pairs, similarity_matrix.flatten()):
        gran = pair.get("granularity", "Unknown")
        pred = 1 if sim >= best_threshold else 0
        if gran not in granularity_counts:
            granularity_counts[gran] = {"tp": 0, "fp": 0, "fn": 0, "predicted": 0}
        if pred == 1:
            granularity_counts[gran]["predicted"] += 1
        if pred == 1 and pair["label"] == 1:
            granularity_counts[gran]["tp"] += 1
        if pred == 1 and pair["label"] == 0:
            granularity_counts[gran]["fp"] += 1
        if pred == 0 and pair["label"] == 1:
            granularity_counts[gran]["fn"] += 1
    granularity_metrics = {}
    for gran, counts in granularity_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        if tp == 0:
            prec, rec, f1 = 0.0, 0.0, 0.0
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
        granularity_metrics[gran] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "predicted_links": counts["predicted"]
        }
    run_result["granularity_metrics"] = granularity_metrics
    return run_result

def build_predicted_links(similarity_matrix, pairs, best_threshold: float, code_granularities: List[str]):
    preds = []
    for pair in pairs:
        d_i = pair["doc_idx"]
        c_i = pair["code_idx"]
        sim = similarity_matrix[d_i, c_i]
        pred = 1 if sim >= best_threshold else 0
        preds.append(pred)
    preds = np.array(preds)
    total_predicted = int(np.sum(preds))
    
    predicted_by_granularity = {}
    for i, pair in enumerate(pairs):
        if preds[i] == 1:
            gran = pair.get("granularity", "Unknown")
            if gran not in predicted_by_granularity:
                predicted_by_granularity[gran] = 0
            predicted_by_granularity[gran] += 1
    return total_predicted, predicted_by_granularity

def multi_run_bm25_baseline_all_artifacts_granularity(dataset_path: str, n_runs: int = 1, step: float = 0.05):
    out_dir = "results/bm25_baseline_granularity"
    os.makedirs(out_dir, exist_ok=True)
    
    all_runs = []
    for run_idx in range(1, n_runs + 1):
        run_result = run_bm25_baseline_once(dataset_path, step=step)
        run_file = os.path.join(out_dir, f"bm25_run_{run_idx}.json")
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(run_result, f, indent=2)
        print(f"[BM25:Granularity] Run {run_idx}: best_threshold={run_result['best_threshold']:.3f}, "
              f"best_f1={run_result['best_f1']:.3f}, best_precision={run_result['best_precision']:.3f}, "
              f"best_recall={run_result['best_recall']:.3f}, total_predicted_links={run_result['total_predicted_links']}")
        print(f"    Predicted links by granularity: {run_result['predicted_links_by_granularity']}")
        all_runs.append(run_result)
    
    # Aggregate overall metrics.
    avg_f1 = np.mean([rr["best_f1"] for rr in all_runs])
    avg_precision = np.mean([rr["best_precision"] for rr in all_runs])
    avg_recall = np.mean([rr["best_recall"] for rr in all_runs])
    avg_threshold = np.mean([rr["best_threshold"] for rr in all_runs])
    avg_total_predicted = np.mean([rr["total_predicted_links"] for rr in all_runs])
    
    # Aggregate per-run granularity metrics.
    aggregated_granularity = {}
    for rr in all_runs:
        for gran, metrics in rr["granularity_metrics"].items():
            if gran not in aggregated_granularity:
                aggregated_granularity[gran] = {"precision": [], "recall": [], "f1": [], "predicted_links": []}
            aggregated_granularity[gran]["precision"].append(metrics["precision"])
            aggregated_granularity[gran]["recall"].append(metrics["recall"])
            aggregated_granularity[gran]["f1"].append(metrics["f1"])
            aggregated_granularity[gran]["predicted_links"].append(metrics["predicted_links"])
    avg_gran = {}
    for gran, lists in aggregated_granularity.items():
        avg_gran[gran] = {
            "precision": float(np.mean(lists["precision"])),
            "recall": float(np.mean(lists["recall"])),
            "f1": float(np.mean(lists["f1"])),
            "predicted_links": float(np.mean(lists["predicted_links"]))
        }
    
    aggregated = {
        "average_f1": float(avg_f1),
        "average_precision": float(avg_precision),
        "average_recall": float(avg_recall),
        "average_threshold": float(avg_threshold),
        "average_total_predicted_links": float(avg_total_predicted),
        "average_predicted_links_by_granularity": avg_gran,
        "n_runs": n_runs
    }
    
    agg_file = os.path.join(out_dir, "aggregated_bm25_granularity_precision_recall.json")
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"[BM25:Granularity] Aggregated => average_f1={avg_f1:.3f}, "
          f"average_precision={avg_precision:.3f}, average_recall={avg_recall:.3f}, "
          f"average_threshold={avg_threshold:.3f}, average_total_predicted_links={avg_total_predicted:.1f}")
    print(f"    Average predicted links by granularity: {aggregated['average_predicted_links_by_granularity']}")
    print(f"Saved aggregated results to {agg_file}")

def main():
    parser = argparse.ArgumentParser(description="BM25 Baseline with Granularity Counts and Precision/Recall for Software Traceability")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset_name.json (e.g., data/unity_catalog.json or data/crawl4ai.json)")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of runs (each run shuffles the data for randomness)")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Step size for threshold grid search (default: 0.05)")
    args = parser.parse_args()
    
    multi_run_bm25_baseline_all_artifacts_granularity(
        dataset_path=args.dataset,
        n_runs=args.n_runs,
        step=args.step
    )

if __name__ == "__main__":
    main()
