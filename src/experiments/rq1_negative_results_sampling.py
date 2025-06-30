import json
import random
import math
import os

def calculate_sample_size(population_size: int, confidence_level: float = 0.95, margin_error: float = 0.05) -> int:
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores[confidence_level]
    numerator = (z**2 * 0.25 * population_size)
    denominator = (margin_error**2 * (population_size - 1) + z**2 * 0.25)
    return max(math.ceil(numerator/denominator), 30)

def pool_and_stratify_negatives(dataset_name: str, confidence_level: float = 0.95) -> dict:
    results_dir = f"results/rq1/{dataset_name}"
    with open(os.path.join(results_dir, "negative_results.json"), 'r') as f:
        negative_results = json.load(f)

    fps, fns = [], []
    for run in negative_results["runs"]:
        for result in run["negative_results"]:
            if result["confusion_metrics"] == "False Positive":
                fps.append(result)
            else:
                fns.append(result)

    total_population = len(fps) + len(fns)
    fp_proportion = len(fps) / total_population
    fn_proportion = len(fns) / total_population
    total_sample_size = calculate_sample_size(total_population, confidence_level)

    fp_sample_size = min(max(30, math.ceil(total_sample_size * fp_proportion)), len(fps))
    fn_sample_size = min(max(30, math.ceil(total_sample_size * fn_proportion)), len(fns))

    return {
        "dataset": dataset_name,
        "statistics": {
            "total_population": total_population,
            "fp_total": len(fps),
            "fn_total": len(fns),
            "fp_proportion": fp_proportion,
            "fn_proportion": fn_proportion,
            "calculated_sample_size": total_sample_size,
            "fp_sample_size": fp_sample_size,
            "fn_sample_size": fn_sample_size
        },
        "false_positives": sorted(random.sample(fps, fp_sample_size), 
                                key=lambda x: x['artifact_title'].lower()),
        "false_negatives": sorted(random.sample(fns, fn_sample_size), 
                                key=lambda x: x['artifact_title'].lower())
    }

def save_stratified_samples(dataset_name: str, stratified_results: dict):
    output_path = f"results/rq1/{dataset_name}/stratified_negative_samples.json"
    with open(output_path, 'w') as f:
        json.dump(stratified_results, f, indent=2)
    
    # Restored original print format
    print(f"\nResults for {dataset_name}:")
    print(f"Total population: {stratified_results['statistics']['total_population']}")
    print(f"FP samples: {stratified_results['statistics']['fp_sample_size']} "
          f"({stratified_results['statistics']['fp_proportion']:.2%} of population)")
    print(f"FN samples: {stratified_results['statistics']['fn_sample_size']} "
          f"({stratified_results['statistics']['fn_proportion']:.2%} of population)")

def main():
    """Fully restored original main function with progress messages"""
    datasets = ["crawl4ai", "unity_catalog"]
    
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name}...")
        stratified_results = pool_and_stratify_negatives(dataset_name)
        save_stratified_samples(dataset_name, stratified_results)

if __name__ == "__main__":
    main()