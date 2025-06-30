import json
import os
from collections import Counter
from typing import Dict, List, Tuple
import re
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.stats import proportion

class TraceabilityAnalyzer:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.results_dir = f"results/rq1/{dataset_name}"
        self.fn_samples = self.load_fn_samples()
        self.tp_samples = self.load_tp_samples()

    def convert_to_serializable(self, obj):
        """Convert numpy/pandas types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        return obj

    def load_fn_samples(self) -> List[Dict]:
        """Load ALL False Negatives from negative results."""
        with open(os.path.join(self.results_dir, "negative_results.json"), 'r') as f:
            negative_results = json.load(f)
            # Pool all FNs from all runs
            all_fns = []
            for run in negative_results["runs"]:
                all_fns.extend([
                    result for result in run["negative_results"]
                    if result["confusion_metrics"] == "False Negative"
                ])
            return all_fns

    def load_tp_samples(self) -> List[Dict]:
        """Load ALL True Positives from combined results."""
        with open(os.path.join(self.results_dir, "combined_results.json"), 'r') as f:
            combined = json.load(f)
            # Pool all TPs from all runs
            all_tps = []
            for run in combined['runs']:
                all_tps.extend([
                    result for result in run['results']
                    if result['confusion_metrics'] == "True Positive"
                ])
            return all_tps

    def get_characteristics(self, samples: List[Dict], sample_type: str) -> Dict:
        """Extract characteristics for a sample set."""
        characteristics = []
        
        for sample in samples:
            doc_text = sample["sent_document_text"]
            artifact_title = sample["artifact_title"]
            base_name = artifact_title.split('.')[-1]
            
            char = {
                "sample_type": sample_type,
                "artifact_title": artifact_title,
                "granularity": sample.get("traceability_granularity"),
                "explicit_mention": base_name.lower() in doc_text.lower(),
                "doc_length": len(doc_text.split()),
                "has_code_example": "```" in doc_text,
                "num_code_examples": doc_text.count("```") // 2,
                "has_parameters_section": "### Parameters" in doc_text,
                "has_return_section": "### Return" in doc_text,
                "is_intro_section": "All URIs are relative to" in doc_text,
                "is_cli_artifact": "Cli" in artifact_title
            }
            characteristics.append(char)
        
        # Aggregate statistics
        stats = {
            "total_samples": len(characteristics),
            "explicit_mentions": sum(1 for c in characteristics if c["explicit_mention"]),
            "with_code_examples": sum(1 for c in characteristics if c["has_code_example"]),
            "granularity_distribution": Counter(c["granularity"] for c in characteristics),
            "intro_sections": sum(1 for c in characteristics if c["is_intro_section"]),
            "cli_artifacts": sum(1 for c in characteristics if c["is_cli_artifact"]),
            "avg_doc_length": sum(c["doc_length"] for c in characteristics) / len(characteristics)
        }
        
        return {"characteristics": characteristics, "statistics": stats}

    def analyze_patterns(self, samples: List[Dict], sample_type: str) -> Dict:
        """Analyze patterns in samples."""
        patterns = {
            "intro_sections": [],
            "cli_artifacts": [],
            "other_artifacts": []
        }
        
        granularity_patterns = {}
        
        for sample in samples:
            doc_text = sample["sent_document_text"]
            artifact_title = sample["artifact_title"]
            granularity = sample.get("traceability_granularity", "Unknown")
            
            record = {
                "artifact_title": artifact_title,
                "doc_text": doc_text,
                "granularity": granularity
            }
            
            # Pattern categorization
            if "All URIs are relative to" in doc_text:
                patterns["intro_sections"].append(record)
            elif "Cli" in artifact_title:
                patterns["cli_artifacts"].append(record)
            else:
                patterns["other_artifacts"].append(record)
            
            # Granularity tracking
            if granularity not in granularity_patterns:
                granularity_patterns[granularity] = []
            granularity_patterns[granularity].append(record)
        
        stats = {
            "total_analyzed": len(samples),
            "pattern_distribution": {
                "intro_sections": len(patterns["intro_sections"]),
                "cli_artifacts": len(patterns["cli_artifacts"]),
                "other_artifacts": len(patterns["other_artifacts"])
            },
            "granularity_distribution": {
                gran: len(items)
                for gran, items in granularity_patterns.items()
            }
        }
        
        return {
            "patterns": patterns,
            "granularity_patterns": granularity_patterns,
            "statistics": stats
        }

    def statistical_comparison(self, fn_chars: List[Dict], tp_chars: List[Dict]) -> Dict:
        """Perform statistical comparison between FN and TP characteristics."""
        comparisons = {}
        
        # Convert to pandas DataFrames for easier analysis
        fn_df = pd.DataFrame(fn_chars)
        tp_df = pd.DataFrame(tp_chars)
        
        # Numeric comparisons (Mann-Whitney U test)
        numeric_fields = ['doc_length', 'num_code_examples']
        for field in numeric_fields:
            stat, pval = stats.mannwhitneyu(
                fn_df[field], tp_df[field], 
                alternative='two-sided'
            )
            effect_size = stat / (len(fn_df) * len(tp_df))  # Common language effect size
            comparisons[f"{field}_comparison"] = {
                "test": "Mann-Whitney U",
                "statistic": float(stat),
                "p_value": float(pval),
                "effect_size": float(effect_size),
                "significant": pval < 0.05
            }
        
        # Proportion comparisons (Z-test)
        binary_fields = [
            'explicit_mention', 'has_code_example',
            'has_parameters_section', 'has_return_section',
            'is_intro_section', 'is_cli_artifact'
        ]
        for field in binary_fields:
            fn_count = fn_df[field].sum()
            tp_count = tp_df[field].sum()
            
            stat, pval = proportion.proportions_ztest(
                [fn_count, tp_count],
                [len(fn_df), len(tp_df)]
            )
            effect_size = abs(
                fn_count/len(fn_df) - tp_count/len(tp_df)
            )  # Difference in proportions
            
            comparisons[f"{field}_comparison"] = {
                "test": "Z-test",
                "statistic": float(stat),
                "p_value": float(pval),
                "effect_size": float(effect_size),
                "significant": pval < 0.05,
                "proportions": {
                    "false_negatives": float(fn_count/len(fn_df)),
                    "true_positives": float(tp_count/len(tp_df))
                }
            }
        
        # Chi-square test for categorical distributions
        granularity_fn = Counter(fn_df['granularity'])
        granularity_tp = Counter(tp_df['granularity'])
        
        # Ensure same categories in both
        all_categories = set(granularity_fn.keys()) | set(granularity_tp.keys())
        observed = np.array([
            [granularity_fn.get(cat, 0) for cat in all_categories],
            [granularity_tp.get(cat, 0) for cat in all_categories]
        ])
        
        chi2, pval = stats.chi2_contingency(observed)[:2]
        
        comparisons["granularity_distribution"] = {
            "test": "Chi-square",
            "statistic": float(chi2),
            "p_value": float(pval),
            "significant": pval < 0.05,
            "categories": list(all_categories),
            "distributions": {
                "false_negatives": {k: float(v/sum(granularity_fn.values())) 
                                  for k, v in granularity_fn.items()},
                "true_positives": {k: float(v/sum(granularity_tp.values())) 
                                 for k, v in granularity_tp.items()}
            }
        }
        
        return comparisons

    def analyze_all(self):
        """Perform complete analysis and save results."""
        # Analyze False Negatives - note we use self.fn_samples directly as it's now a list
        fn_characteristics = self.get_characteristics(
            self.fn_samples, "false_negative"
        )
        fn_patterns = self.analyze_patterns(
            self.fn_samples, "false_negative"
        )
        
        # Analyze True Positives - self.tp_samples is already a list
        tp_characteristics = self.get_characteristics(
            self.tp_samples, "true_positive"
        )
        tp_patterns = self.analyze_patterns(
            self.tp_samples, "true_positive"
        )
        
        # Statistical comparison
        statistical_analysis = self.statistical_comparison(
            fn_characteristics["characteristics"],
            tp_characteristics["characteristics"]
        )
        
        # Save results
        self.save_results(fn_characteristics, fn_patterns, "fn_analysis.json")
        self.save_results(tp_characteristics, tp_patterns, "tp_analysis.json")
        
        serializable_analysis = self.convert_to_serializable(statistical_analysis)
        with open(os.path.join(self.results_dir, "comparative_analysis.json"), 'w') as f:
            json.dump(serializable_analysis, f, indent=2)

    def save_results(self, characteristics: Dict, patterns: Dict, filename: str):
        """Save analysis results to file."""
        serializable_data = self.convert_to_serializable({
            "characteristics": characteristics,
            "patterns": patterns
        })
        with open(os.path.join(self.results_dir, filename), 'w') as f:
            json.dump(serializable_data, f, indent=2)

def main():
    print("\nAnalyzing Unity Catalog...")
    uc_analyzer = TraceabilityAnalyzer("unity_catalog")
    uc_analyzer.analyze_all()
    
    print("\nAnalyzing Crawl4AI...")
    c4_analyzer = TraceabilityAnalyzer("crawl4ai")
    c4_analyzer.analyze_all()

if __name__ == "__main__":
    main()