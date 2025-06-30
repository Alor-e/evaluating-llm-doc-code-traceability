
"""
trace_chain_analyzer.py ― RQ3 chain‑analysis utility
===================================================
• Reads  results/rq1_<model>_no_file/<dataset>/combined_results.json
• Writes results/rq3_<model>_no_file/<dataset>/{trace_chain_analyses.json, metrics.json}

Table 1 buckets  →  metrics["correctness"]
Table 2 buckets  →  metrics["patterns"]   (only EXTENDED/SHORTENED/REORDERED/SUBSTITUTED)
Percentages for Table 2 are computed over the *Partial‑Intermediate* subset,
so numbers align with the paper’s definition.
"""

from __future__ import annotations

import json
import os
import sys
from enum import Enum
from typing import Dict, List

class CorrectnessLevel(Enum):
    COMPLETE_MATCH = "Exact match with ground truth chain"
    PARTIAL_MATCH_INTERMEDIATE = "Endpoints correct – interior deviates"
    PARTIAL_MATCH_PREFIX = "Correct start – wrong end"
    PARTIAL_MATCH_SUFFIX = "Wrong start – correct end"
    INCORRECT = "No meaningful overlap with ground truth"


class InteriorPattern(Enum):
    EXTENDED = "Extra interior node(s) added"
    SHORTENED = "Missing interior node(s)"
    REORDERED = "Interior nodes present but wrong order"
    SUBSTITUTED = "At least one interior node replaced (mix add/remove)"


class TraceChainAnalyzer:
    """Analyse one <model, dataset> pair."""

    def __init__(self, model: str, dataset: str):
        self.model = model
        self.dataset = dataset
        self.src_dir = f"results/rq1_{model}_no_file/{dataset}"
        self.dst_dir = f"results/rq3_{model}_no_file/{dataset}"
        os.makedirs(self.dst_dir, exist_ok=True)

    @staticmethod
    def _parse(chain: str) -> List[str]:
        return [n.strip() for n in chain.split("->") if n.strip()] if chain else []

    @staticmethod
    def _is_subsequence(shorter: List[str], longer: List[str]) -> bool:
        """Return True iff *shorter* appears in *longer* preserving order."""
        it = iter(longer)
        return all(elem in it for elem in shorter)

    # ---------- interior pattern classifier ----------
    def _interior_pattern(self, pred: List[str], gt: List[str]) -> str | None:
        """Return pattern name for chains with correct endpoints; else None."""
        if not pred or not gt or pred == gt:
            return None  # identical or empty → no interior error
        if pred[0] != gt[0] or pred[-1] != gt[-1]:
            return None  # endpoint mismatch handled elsewhere

        # 1) EXTENDED: GT is an ordered subsequence of PRED & PRED longer
        if len(pred) > len(gt) and self._is_subsequence(gt, pred):
            return InteriorPattern.EXTENDED.name

        # 2) SHORTENED: PRED is an ordered subsequence of GT & shorter
        if len(pred) < len(gt) and self._is_subsequence(pred, gt):
            return InteriorPattern.SHORTENED.name

        # 3) REORDERED: exact same multiset, same length, order differs
        if len(pred) == len(gt) and set(pred) == set(gt):
            return InteriorPattern.REORDERED.name

        # 4) OTHERWISE: endpoints OK, interior differs by mix add/remove
        return InteriorPattern.SUBSTITUTED.name

    # ---------- correctness bucket ----------
    @staticmethod
    def _correctness(pred: List[str], gt: List[str]) -> CorrectnessLevel:
        if pred == gt:
            return CorrectnessLevel.COMPLETE_MATCH
        if pred and gt and pred[0] == gt[0] and pred[-1] == gt[-1]:
            return CorrectnessLevel.PARTIAL_MATCH_INTERMEDIATE
        if pred and gt and pred[0] == gt[0]:
            return CorrectnessLevel.PARTIAL_MATCH_PREFIX
        if pred and gt and pred[-1] == gt[-1]:
            return CorrectnessLevel.PARTIAL_MATCH_SUFFIX
        return CorrectnessLevel.INCORRECT

    # ---------- main driver ----------
    def run(self) -> Dict:
        src_file = os.path.join(self.src_dir, "combined_results.json")
        if not os.path.isfile(src_file):
            raise FileNotFoundError(src_file)

        with open(src_file) as f:
            combined = json.load(f)

        metrics = {
            "total": 0,
            "patterns": {p.name: 0 for p in InteriorPattern},
            "correctness": {c.name: 0 for c in CorrectnessLevel},
        }
        analyses = []

        for run in combined["runs"]:
            for rec in run["results"]:
                pred_raw = rec.get("predicted_trace_chain", "")
                gt_raw = rec.get("ground_truth_trace_chain", "")
                pred, gt = self._parse(pred_raw), self._parse(gt_raw)

                corr = self._correctness(pred, gt)
                pat = self._interior_pattern(pred, gt)

                metrics["total"] += 1
                metrics["correctness"][corr.name] += 1
                if pat:
                    metrics["patterns"][pat] += 1

                analyses.append({
                    "document_text": rec.get("sent_document_text"),
                    "artifact_title": rec.get("artifact_title"),
                    "predicted_chain": pred_raw,
                    "ground_truth_chain": gt_raw,
                    "interior_pattern": pat or "N/A",
                    "correctness": corr.name,
                    "traceability_granularity": rec.get("traceability_granularity"),
                })

        # ── percentage calculations ───────────────────────────────
        total_pi = metrics["correctness"]["PARTIAL_MATCH_INTERMEDIATE"] or 1  # avoid /0

        metrics["patterns_percentages"] = {
            k: (v / total_pi) * 100 for k, v in metrics["patterns"].items()
        }
        metrics["correctness_percentages"] = {
            k: (v / metrics["total"]) * 100 for k, v in metrics["correctness"].items()
        }

        # ── save artefacts ────────────────────────────────────────
        with open(os.path.join(self.dst_dir, "trace_chain_analyses.json"), "w") as f:
            json.dump(analyses, f, indent=2)
        with open(os.path.join(self.dst_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

# 3. CLI UTILITIES

def _discover_models() -> List[str]:
    return [d.removeprefix("rq1_").removesuffix("_no_file")
            for d in os.listdir("results")
            if d.startswith("rq1_") and d.endswith("_no_file")]


def main():
    args = sys.argv[1:]
    models = args[:1] or _discover_models()
    datasets = args[1:] or ["crawl4ai", "unity_catalog"]

    for model in models:
        for ds in datasets:
            try:
                print(f" {model}/{ds}")
                m = TraceChainAnalyzer(model, ds).run()
                pp = m["patterns_percentages"]
                print(f"   total={m['total']}  ext={pp['EXTENDED']:.1f}%  shr={pp['SHORTENED']:.1f}%  "
                      f"reo={pp['REORDERED']:.1f}%  sub={pp['SUBSTITUTED']:.1f}%")
            except FileNotFoundError:
                print(f"✘ {model}/{ds}  (missing combined_results.json)")


if __name__ == "__main__":
    main()
