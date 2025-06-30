import os
import json
import random
import csv

DATASETS = ["crawl4ai"]
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_ground_truth_pairs(dataset):
    """
    Returns a set of (split_text, artifact_title) pairs that are ground truth links for the dataset.
    """
    dataset_path = os.path.join(BASE_PATH, dataset)
    main_json_path = os.path.join(dataset_path, f"{dataset}.json")
    if not os.path.exists(main_json_path):
        return set()
    gt_pairs = set()
    main_data = load_json(main_json_path)
    for entry in main_data:
        doc_text = entry.get("document", {}).get("text")
        for artifact in entry.get("artifacts", []):
            artifact_title = artifact.get("title")
            if doc_text and artifact_title:
                gt_pairs.add((doc_text, artifact_title))
    return gt_pairs

def generate_pairs(dataset):
    dataset_path = os.path.join(BASE_PATH, dataset)
    artifact_details_path = os.path.join(dataset_path, "artifact_details.json")
    split_documents_path = os.path.join(dataset_path, "split_documents.json")
    if not (os.path.exists(artifact_details_path) and os.path.exists(split_documents_path)):
        return []
    artifacts = load_json(artifact_details_path)
    splits = load_json(split_documents_path)
    gt_pairs = build_ground_truth_pairs(dataset)
    pairs = []
    for split in splits:
        split_text = split.get("split_text")
        file_location = split.get("file_location")
        for artifact in artifacts:
            artifact_title = artifact.get("artifact_title")
            pair = {
                "text": split_text,
                "location": file_location,
                "artifact_title": artifact_title,
                "artifact_location": artifact.get("artifact_location"),
                "artifact_content": artifact.get("artifact_content"),
                "artifact_type": artifact.get("artifact_type"),
                "Alor_label": "y" if (split_text, artifact_title) in gt_pairs else "n",
                "Hassan_label": ""
            }
            pairs.append(pair)
    return pairs

def save_as_csv(pairs, output_path):
    if not pairs:
        return
    keys = pairs[0].keys()
    with open(output_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(pairs)

def main(output_csv=False):
    for dataset in DATASETS:
        pairs = generate_pairs(dataset)
        random.shuffle(pairs)  # Shuffle the pairs for random order
        if output_csv:
            output_path = f"doc_code_pairs_{dataset}.csv"
            save_as_csv(pairs, output_path)
        else:
            output_path = f"doc_code_pairs_{dataset}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"Generated {len(pairs)} pairs and saved to {output_path}")

if __name__ == "__main__":
    main()
