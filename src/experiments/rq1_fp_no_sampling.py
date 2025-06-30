import json
import os
import glob

def load_artifact_details(dataset):
    """
    Loads artifact details for the given dataset from data/{dataset}/artifact_details.json.
    Returns a lookup dict mapping artifact_title to artifact_content.
    """
    details_path = os.path.join("data", dataset, "artifact_details.json")
    with open(details_path, "r") as f:
        artifacts = json.load(f)
    # Build a lookup dictionary using artifact_title as key.
    lookup = {artifact["artifact_title"]: artifact.get("artifact_content", "") for artifact in artifacts}
    return lookup

def process_negative_results_file(filepath):
    """
    Processes a negative_results.json file:
      - Loads the JSON data.
      - Loads the corresponding artifact_details.json based on the dataset.
      - Iterates over each run and each negative result.
      - Filters for objects with confusion_metrics equal to "False Positive".
      - Adds error_group (empty string) and artifact_code (from artifact details) to each.
      - Writes the flat list to a fp_analysis.json file in the same folder.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    
    dataset = data.get("dataset")
    if not dataset:
        print(f"Dataset not specified in {filepath}. Skipping file.")
        return

    artifact_lookup = load_artifact_details(dataset)
    filtered_results = []

    for run in data.get("runs", []):
        for result in run.get("negative_results", []):
            # Only consider results with "False Positive" confusion metrics.
            if result.get("confusion_metrics") == "False Positive":
                result["error_group"] = [""]  # Add error_group as an empty string in an array.
                artifact_title = result.get("artifact_title")
                # Lookup artifact_code using artifact_title (default to empty string if not found).
                result["artifact_code"] = artifact_lookup.get(artifact_title, "")
                filtered_results.append(result)

    # Write the filtered list to fp_analysis.json in the same folder.
    output_path = os.path.join(os.path.dirname(filepath), "fp_analysis.json")
    with open(output_path, "w") as f:
        json.dump(filtered_results, f, indent=2)
    print(f"Processed {filepath}. Wrote {len(filtered_results)} items to {output_path}.")

def main():
    """
    Finds all negative_results.json files in results/rq1_* folders (for both crawl4ai and unity_catalog)
    and processes each.
    """
    # Pattern matches any negative_results.json file under results/rq1_*/<dataset>/negative_results.json
    pattern = os.path.join("results", "rq1_*", "*", "negative_results.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print("No negative_results.json files found.")
    for filepath in files:
        process_negative_results_file(filepath)

if __name__ == "__main__":
    main()
