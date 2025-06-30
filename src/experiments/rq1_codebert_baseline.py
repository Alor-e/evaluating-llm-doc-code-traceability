#!/usr/bin/env python3
"""
bert_baseline_all_artifacts_sliding_window.py

A baseline approach for software traceability using BERT-based embeddings (CodeBERT)
that:
 - Loads dataset_name.json
 - Builds all doc–artifact pairs (no negative sampling)
 - On the document side: uses the text snippet and location.
 - On the code side: uses a list of code artifacts (each with title, content, location).
 - Uses CodeBERT embeddings with a sliding window to overcome the 512-token limit.
 - Performs cosine similarity comparisons and grid search for threshold to maximize F1.
 - Computes evaluation metrics both overall and per artifact granularity.
 - Outputs evaluation results in JSON format.
 - Prints out the first few document and artifact contexts for verification.

Usage Example:
  python bert_baseline_all_artifacts_sliding_window.py \
    --dataset data/crawl4ai/crawl4ai.json \
    --n_runs 1 \
    --step 0.05
"""

import os
import json
import argparse
import numpy as np
import random
import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

#                   LOADING & CONTEXT CONSTRUCTION

def load_dataset(dataset_path: str) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_doc_and_artifact_contexts(data: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Builds contexts for documents and code artifacts.
    
    Documents: Combine the text snippet and location.
    Code artifacts: For each artifact, create a dict with title, content, location,
    and a combined 'context' (content + location).
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
            art_location = art.get("location", "").strip()
            if title and title not in code_artifacts:
                combined_context = f"{content} [Location: {art_location}]"
                code_artifacts[title] = {
                    "title": title,
                    "content": content,
                    "location": art_location,
                    "traceability_granularity": art.get("traceability_granularity", "Unknown"),
                    "context": combined_context
                }
    code_artifact_list = list(code_artifacts.values())
    return doc_texts, code_artifact_list

def build_all_pairs(data: List[Dict], code_artifacts: List[Dict]) -> List[Tuple[int, int, int]]:
    """
    Builds all document–artifact pairs.
    Each pair: (document_index, code_artifact_index, label)
    Label = 1 if artifact's title is in document's artifacts, else 0.
    """
    title_to_idx = {art["title"]: idx for idx, art in enumerate(code_artifacts)}
    pairs = []
    for doc_idx, rec in enumerate(data):
        positive_titles = set(a["title"] for a in rec["artifacts"] if a.get("title"))
        for title, code_idx in title_to_idx.items():
            label = 1 if title in positive_titles else 0
            pairs.append((doc_idx, code_idx, label))
    return pairs

#              CODEBERT EMBEDDING with Sliding Window + SIMILARITY

class CodeBERTEmbedder:
    def __init__(self, model_name="microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.max_length = self.tokenizer.model_max_length  # typically 512

    def embed_text(self, text: str, window_overlap: int = 50) -> np.ndarray:
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) <= self.max_length:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(dim=0)
            return cls_emb.cpu().numpy()
        
        # Sliding window approach for long inputs
        chunk_size = self.max_length - 2  # reserve space for special tokens
        stride = chunk_size - window_overlap
        segments = []
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i+chunk_size]
            if not chunk_tokens:
                break
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            segments.append(chunk_text)
        
        embeddings = []
        for segment in segments:
            inputs = self.tokenizer(segment, return_tensors='pt', truncation=True, max_length=self.max_length)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(dim=0)
            embeddings.append(cls_emb.cpu().numpy())
        
        avg_embedding = np.mean(np.vstack(embeddings), axis=0)
        return avg_embedding

def build_codebert_embeddings(doc_texts: List[str], code_artifacts: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    embedder = CodeBERTEmbedder()
    doc_embs = []
    for d in doc_texts:
        emb = embedder.embed_text(d)
        doc_embs.append(emb)
    doc_embs = np.vstack(doc_embs)
    
    code_embs = []
    for art in code_artifacts:
        emb = embedder.embed_text(art["context"])
        code_embs.append(emb)
    code_embs = np.vstack(code_embs)
    
    return doc_embs, code_embs

def compute_similarity(doc_idx: int, code_idx: int, doc_embs: np.ndarray, code_embs: np.ndarray) -> float:
    doc_vec = doc_embs[doc_idx].reshape(1, -1)
    code_vec = code_embs[code_idx].reshape(1, -1)
    sim = cosine_similarity(doc_vec, code_vec)[0, 0]
    return float(sim)

#                GRID SEARCH + METRICS

def precision_recall_f1(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def grid_search_threshold(similarities: np.ndarray, labels: np.ndarray, step: float = 0.05) -> Tuple[float, float, List[Dict]]:
    best_threshold = 0.0
    best_f1 = -1
    thresholds = np.arange(0.0, 1.0 + step, step)
    threshold_metrics = []
    for t in thresholds:
        preds = (similarities >= t).astype(int)
        p, r, f1 = precision_recall_f1(preds, labels)
        threshold_metrics.append({
            "threshold": float(t),
            "precision": p,
            "recall": r,
            "f1": f1
        })
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold, best_f1, threshold_metrics

#            SINGLE RUN (ALL-ARTIFACTS) & MULTI-RUN

def run_codebert_all_artifacts_once(dataset_path: str, step: float = 0.05) -> Dict:
    data = load_dataset(dataset_path)
    
    # Shuffle the dataset to introduce randomness between runs.
    random.shuffle(data)
    
    doc_texts, code_artifacts = build_doc_and_artifact_contexts(data)
    print("First 3 Document Contexts:")
    for dt in doc_texts[:3]:
        print(dt)
    print("\nFirst 3 Code Artifacts:")
    for art in code_artifacts[:3]:
        print(art)
    
    pairs = build_all_pairs(data, code_artifacts)
    doc_embs, code_embs = build_codebert_embeddings(doc_texts, code_artifacts)
    
    sims_list = []
    labels_list = []
    for (d_i, c_i, label) in pairs:
        sim = compute_similarity(d_i, c_i, doc_embs, code_embs)
        sims_list.append(sim)
        labels_list.append(label)
    
    sims_array = np.array(sims_list)
    labs_array = np.array(labels_list)
    
    best_threshold, best_f1, threshold_metrics = grid_search_threshold(sims_array, labs_array, step)
    
    # Compute per-granularity metrics.
    granularity_counts = {}
    for (d_i, c_i, label), sim in zip(pairs, sims_list):
        granularity = code_artifacts[c_i].get("traceability_granularity", "Unknown")
        pred = 1 if sim >= best_threshold else 0
        if granularity not in granularity_counts:
            granularity_counts[granularity] = {"tp": 0, "fp": 0, "fn": 0, "predicted": 0}
        if pred == 1:
            granularity_counts[granularity]["predicted"] += 1
        if pred == 1 and label == 1:
            granularity_counts[granularity]["tp"] += 1
        if pred == 1 and label == 0:
            granularity_counts[granularity]["fp"] += 1
        if pred == 0 and label == 1:
            granularity_counts[granularity]["fn"] += 1
    
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
    
    total_predicted_links = int(np.sum((sims_array >= best_threshold).astype(int)))
    
    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "threshold_metrics": threshold_metrics,
        "average_total_predicted_links": total_predicted_links,
        "granularity_metrics": granularity_metrics
    }

def multi_run_codebert_all_artifacts(dataset_path: str, n_runs: int = 1, step: float = 0.05):
    out_dir = "results/codebert_all"
    os.makedirs(out_dir, exist_ok=True)
    
    all_runs = []
    for i in range(1, n_runs + 1):
        result = run_codebert_all_artifacts_once(dataset_path, step=step)
        run_file = os.path.join(out_dir, f"codebert_run_{i}.json")
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[CodeBERT:AllArt] Run {i} => best_threshold={result['best_threshold']:.3f}, best_f1={result['best_f1']:.3f}")
        all_runs.append(result)
    
    # Aggregate overall metrics from each run.
    f1_sum = sum(rr["best_f1"] for rr in all_runs)
    thresh_sum = sum(rr["best_threshold"] for rr in all_runs)
    total_predicted_links_sum = sum(rr["average_total_predicted_links"] for rr in all_runs)
    
    # Extract overall best precision and recall from each run's threshold_metrics.
    precisions = []
    recalls = []
    for rr in all_runs:
        best_thresh = rr["best_threshold"]
        best_metric = next((entry for entry in rr["threshold_metrics"] if abs(entry["threshold"] - best_thresh) < 1e-6), None)
        if best_metric:
            precisions.append(best_metric["precision"])
            recalls.append(best_metric["recall"])
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    
    avg_f1 = f1_sum / n_runs
    avg_thresh = thresh_sum / n_runs
    avg_total_predicted = total_predicted_links_sum / n_runs
    
    # Aggregate granularity metrics.
    aggregated_gran = {}
    for rr in all_runs:
        for gran, metrics in rr["granularity_metrics"].items():
            if gran not in aggregated_gran:
                aggregated_gran[gran] = {"precision": [], "recall": [], "f1": [], "predicted_links": []}
            aggregated_gran[gran]["precision"].append(metrics["precision"])
            aggregated_gran[gran]["recall"].append(metrics["recall"])
            aggregated_gran[gran]["f1"].append(metrics["f1"])
            aggregated_gran[gran]["predicted_links"].append(metrics["predicted_links"])
    avg_gran = {}
    for gran, lists in aggregated_gran.items():
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
        "average_threshold": float(avg_thresh),
        "average_total_predicted_links": float(avg_total_predicted),
        "average_predicted_links_by_granularity": avg_gran,
        "n_runs": n_runs
    }
    agg_file = os.path.join(out_dir, "aggregated_codebert_all.json")
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"[CodeBERT:AllArt] Aggregated => avg_f1={avg_f1:.3f}, avg_precision={avg_precision:.3f}, avg_recall={avg_recall:.3f}, avg_threshold={avg_thresh:.3f}, avg_total_predicted_links={avg_total_predicted:.1f}")
    print(f"Saved aggregated results to {agg_file}")

def main():
    parser = argparse.ArgumentParser(description="BERT-based (CodeBERT) Baseline with Sliding Window for Software Traceability")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset_name.json")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs (identical, unless randomness occurs)")
    parser.add_argument("--step", type=float, default=0.05, help="Threshold grid search step size")
    args = parser.parse_args()
    
    multi_run_codebert_all_artifacts(
        dataset_path=args.dataset,
        n_runs=args.n_runs,
        step=args.step
    )

if __name__ == "__main__":
    main()
