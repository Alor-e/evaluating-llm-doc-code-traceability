import json
import os

# Define eight error groups.
def map_label_to_group(label: str) -> str:
    l = label.lower().strip()
    # Group 1: Missing Key Details: labels with "omission", "missing", or "omitted" (but not "incomplete")
    if ("omission" in l or "missing" in l or "omitted" in l) and "incomplete" not in l:
        return "Missing Key Details"
    # Group 2: Incomplete Explanation: anything with "incomplete"
    if "incomplete" in l:
        return "Incomplete Explanation"
    # Group 3: Misinterpretation: labels with "misunderstand", "misinterpret", "misrepresentation", "uncertain", "hedging", or "minor_"
    if ("misunderstand" in l or "misinterpret" in l or "misrepresentation" in l or
        "uncertain" in l or "hedging" in l or "minor_" in l):
        return "Misinterpretation"
    # Group 4: Unsupported Assumption: labels with "assumption", "unsubstantiated", or "overspecification"
    if "assumption" in l or "unsubstantiated" in l or "overspecification" in l:
        return "Unsupported Assumption"
    # Group 5: Implementation Detail: labels with "implementation_detail", "incorrect_implementation", "inaccurate_implementation", or "implementation_mismatch"
    if ("implementation_detail" in l or "incorrect_implementation" in l or
        "inaccurate_implementation" in l or "implementation_mismatch" in l):
        return "Implementation Detail"
    # Group 6: Architectural: labels with "architectural", "architecture", "component", "class_relationship", "layer", or "misattribution"
    if ("architectural" in l or "architecture" in l or "component" in l or
        "class_relationship" in l or "layer" in l or "misattribution" in l):
        return "Architectural"
    # Group 7: Scope Mismatch: labels with "focus", "scope", "wrong_focus", or "oversimplification"
    if "focus" in l or "scope" in l or "wrong_focus" in l or "oversimplification" in l:
        return "Scope Mismatch"
    # Group 8: Relationship-Specific: labels containing "relationship"
    if "relationship" in l:
        return "Relationship-Specific"
    return "Other / Unclassified"

# Read error types from a dataset's aggregated_metrics.json file in results/rq2.
def read_error_types(dataset: str) -> dict:
    """
    Reads the error_types from results/rq2/{dataset}/aggregated_metrics.json.
    """
    file_path = os.path.join("results", "rq2", dataset, "aggregated_metrics.json")
    if not os.path.exists(file_path):
        print(f"File not found for dataset '{dataset}': {file_path}")
        return {}
    with open(file_path, "r") as f:
        data = json.load(f)
    return data.get("error_types", {})

# Aggregate error types across multiple datasets.
def aggregate_all_error_types(datasets: list) -> dict:
    combined = {}
    for ds in datasets:
        et = read_error_types(ds)
        for label, count in et.items():
            combined[label] = combined.get(label, 0) + count
    return combined

# Group error types into eight categories.
def group_error_types(error_types: dict):
    groups = [
        "Missing Key Details",
        "Incomplete Explanation",
        "Misinterpretation",
        "Unsupported Assumption",
        "Implementation Detail",
        "Architectural",
        "Scope Mismatch",
        "Relationship-Specific",
        "Other / Unclassified"
    ]
    grouped_counts = {group: 0 for group in groups}
    miscellaneous_details = {}
    
    for label, count in error_types.items():
        group = map_label_to_group(label)
        grouped_counts[group] += count
        if group == "Other / Unclassified":
            miscellaneous_details[label] = count
    return grouped_counts, miscellaneous_details

# Process one dataset and save its aggregated error aggregation.
def process_dataset(dataset: str) -> dict:
    et = read_error_types(dataset)
    grouped_counts, miscellaneous_details = group_error_types(et)
    result = {
        "dataset": dataset,
        "grouped_error_types": grouped_counts,
        "miscellaneous_details": miscellaneous_details,
        "aggregated_error_types": et,
    }
    # Save per-dataset result in its folder (optional) or in a common location.
    output_file = os.path.join("results", "rq2", dataset, f"explanation_error_aggregation_{dataset}.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Aggregated explanation error aggregation for '{dataset}' saved to: {output_file}")
    return result

# Main function: process each dataset separately and then combine them.
def main():
    datasets = ["crawl4ai", "unity_catalog"]
    per_dataset_results = {}
    for ds in datasets:
        per_dataset_results[ds] = process_dataset(ds)
    
    # Optionally, produce a combined aggregation across datasets.
    combined_et = aggregate_all_error_types(datasets)
    combined_grouped, combined_misc = group_error_types(combined_et)
    combined_result = {
        "datasets": datasets,
        "aggregated_error_types": combined_et,
        "grouped_error_types": combined_grouped,
        "miscellaneous_details": combined_misc
    }
    output_combined = os.path.join("results", "rq2", "explanation_error_aggregation_combined.json")
    with open(output_combined, "w") as f:
        json.dump(combined_result, f, indent=2)
    print(f"Combined aggregated explanation error aggregation saved to: {output_combined}")

if __name__ == "__main__":
    main()
