"""
tfidf_baseline_all_artifacts_granularity_precision_recall.py

A comprehensive TF–IDF baseline approach for software traceability that:
 - Loads dataset_name.json (e.g., unity_catalog.json or crawl4ai.json).
 - Adds contextual information by including both snippets and location.
 - Considers all artifact granularities (e.g., Class, Method, Statement).
 - Builds all doc–artifact pairs without negative sampling.
 - Assigns label=1 if artifact is in doc's "artifacts" list, else 0.
 - Computes TF–IDF + Cosine Similarity for each pair.
 - Performs a grid search over thresholds to find the best threshold maximizing F1.
 - Counts the number of links returned overall and by granularity.
 - Calculates and aggregates precision, recall, and F1 (overall and per granularity) across multiple runs.
 - Saves detailed results in structured JSON files.
 - Shuffles the dataset between runs.

Usage Example:
  python tfidf_baseline_all_artifacts_granularity_precision_recall.py \
    --dataset data/unity_catalog.json \
    --n_runs 3 \
    --step 0.05
"""

import os
import json
import argparse
import numpy as np
import random
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#                           LOADING & PAIR CREATION

def load_dataset(dataset_path: str) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_doc_and_artifact_lists(data: List[Dict]):
    """
    Builds lists of document texts and artifact details.
    
    Document texts: For each document, combine the text snippet and location.
    Code artifacts: For each artifact, combine its content and location.
    
    Returns:
      doc_texts: List of enriched document texts.
      code_titles: List of unique artifact titles.
      code_texts: List of artifact contents.
      code_granularities: List of artifact granularities corresponding to code_titles.
    """
    doc_texts = []
    code_map = {}  # title -> (content, granularity)
    for rec in data:
        doc_snippet = rec["document"].get("text", "")
        doc_location = rec["document"].get("location", "")
        enriched_doc = f"{doc_snippet} [Location: {doc_location}]"
        doc_texts.append(enriched_doc)
        
        for art in rec["artifacts"]:
            title = art.get("title", "").strip()
            content = art.get("content", "").strip()
            granularity = art.get("traceability_granularity", "").strip()
            if title and title not in code_map:
                # We do not read the full file content; we use provided content and add location.
                artifact_location = art.get("location", "").strip()
                combined_content = f"{content} [Location: {artifact_location}]"
                code_map[title] = (combined_content, granularity)
    code_titles = list(code_map.keys())
    code_texts = [code_map[t][0] for t in code_titles]
    code_granularities = [code_map[t][1] for t in code_titles]
    return doc_texts, code_titles, code_texts, code_granularities

def build_all_pairs(data: List[Dict], code_titles: List[str]) -> List[Dict]:
    """
    Builds all document-artifact pairs with labels.
    
    Each pair is represented as a dictionary:
      {
        "doc_idx": int,
        "code_idx": int,
        "label": 0 or 1
      }
    
    Label is 1 if the artifact's title is in the document's "artifacts" list, else 0.
    """
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

#                   TF–IDF SIMILARITY

def build_tfidf_vectors(doc_texts: List[str], code_texts: List[str]):
    """
    Builds TF–IDF vectors for documents and code artifacts.
    
    Returns:
      doc_vectors: TF–IDF vectors for documents.
      code_vectors: TF–IDF vectors for code artifacts.
      vectorizer: The fitted TfidfVectorizer instance.
    """
    all_texts = doc_texts + code_texts
    vectorizer = TfidfVectorizer().fit(all_texts)
    doc_vectors = vectorizer.transform(doc_texts)
    code_vectors = vectorizer.transform(code_texts)
    return doc_vectors, code_vectors, vectorizer

def compute_similarity(doc_vectors, code_vectors):
    """
    Computes cosine similarity between each document and each code artifact.
    
    Returns:
      similarity_matrix: numpy array of shape (n_docs, n_code)
    """
    similarity_matrix = cosine_similarity(doc_vectors, code_vectors)
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
    best_threshold = 0.0
    best_f1 = -1
    best_precision = 0.0
    best_recall = 0.0
    thresholds = np.arange(0.0, 1.0 + step, step)
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

def run_tfidf_baseline_once(similarity_matrix, pairs, step: float = 0.05):
    # Extract similarity scores and true labels from pairs.
    similarity_scores = []
    labels = []
    for pair in pairs:
        d_i = pair["doc_idx"]
        c_i = pair["code_idx"]
        label = pair["label"]
        sim = similarity_matrix[d_i, c_i]
        similarity_scores.append(sim)
        labels.append(label)
    similarity_scores = np.array(similarity_scores)
    labels = np.array(labels)
    
    best_threshold, best_f1, best_precision, best_recall, threshold_metrics = grid_search_threshold(similarity_scores, labels, step)
    
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
    return run_result, preds, labels

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
            granularity = pair.get("granularity", "Unknown")
            if granularity not in predicted_by_granularity:
                predicted_by_granularity[granularity] = 0
            predicted_by_granularity[granularity] += 1
    return total_predicted, predicted_by_granularity

def run_tfidf_baseline_all_artifacts_granularity_once(dataset_path: str, step: float = 0.05):
    # Load data and shuffle it to introduce randomness between runs.
    data = load_dataset(dataset_path)
    random.shuffle(data)
    
    # Build document and artifact lists.
    doc_texts, code_titles, code_texts, code_granularities = build_doc_and_artifact_lists(data)
    pairs = build_all_pairs(data, code_titles)
    
    # Add granularity to each pair.
    for pair in pairs:
        code_idx = pair["code_idx"]
        pair["granularity"] = code_granularities[code_idx]
    
    # Build TF–IDF vectors.
    doc_vectors, code_vectors, vectorizer = build_tfidf_vectors(doc_texts, code_texts)
    similarity_matrix = compute_similarity(doc_vectors, code_vectors)
    
    run_result, preds, labels = run_tfidf_baseline_once(similarity_matrix, pairs, step)
    
    total_predicted, predicted_by_granularity = build_predicted_links(similarity_matrix, pairs, run_result["best_threshold"], code_granularities)
    run_result["total_predicted_links"] = total_predicted
    run_result["predicted_links_by_granularity"] = predicted_by_granularity
    
    # Compute per-granularity metrics.
    granularity_counts = {}
    # Note: since similarity_matrix is 2D (n_docs x n_code), we flatten it in the loop using pairs order.
    for pair, sim in zip(pairs, similarity_matrix.flatten()):
        gran = pair.get("granularity", "Unknown")
        pred = 1 if sim >= run_result["best_threshold"] else 0
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

def multi_run_tfidf_baseline_all_artifacts_granularity(dataset_path: str, n_runs: int = 1, step: float = 0.05):
    out_dir = "results/tfidf_baseline_granularity"
    os.makedirs(out_dir, exist_ok=True)
    
    all_runs = []
    for run_idx in range(1, n_runs + 1):
        run_result = run_tfidf_baseline_all_artifacts_granularity_once(dataset_path, step=step)
        run_file = os.path.join(out_dir, f"tfidf_run_{run_idx}.json")
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(run_result, f, indent=2)
        print(f"[TF–IDF:Granularity] Run {run_idx}: best_threshold={run_result['best_threshold']:.3f}, "
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
    
    agg_file = os.path.join(out_dir, "aggregated_tfidf_granularity_precision_recall.json")
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"[TF–IDF:Granularity] Aggregated => average_f1={avg_f1:.3f}, "
          f"average_precision={avg_precision:.3f}, average_recall={avg_recall:.3f}, "
          f"average_threshold={avg_threshold:.3f}, average_total_predicted_links={avg_total_predicted:.1f}")
    print(f"    Average predicted links by granularity: {aggregated['average_predicted_links_by_granularity']}")
    print(f"Saved aggregated results to {agg_file}")

def main():
    parser = argparse.ArgumentParser(description="TF–IDF Baseline with Granularity Counts and Precision/Recall for Software Traceability")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset_name.json (e.g., data/unity_catalog.json or data/crawl4ai.json)")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of runs (each run shuffles the data for randomness)")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Step size for threshold grid search (default: 0.05)")
    
    args = parser.parse_args()
    
    multi_run_tfidf_baseline_all_artifacts_granularity(
        dataset_path=args.dataset,
        n_runs=args.n_runs,
        step=args.step
    )

if __name__ == "__main__":
    main()
