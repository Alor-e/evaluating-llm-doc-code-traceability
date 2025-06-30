import json
from pathlib import Path


def extract_unique_concepts(dataset_name: str) -> dict:
    # Construct file paths
    input_path = Path(f"data/{dataset_name}/split_documents.json")
    output_path = Path(f"data/{dataset_name}/artifact_lists.json")
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read the input JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Extract concepts only from documents where met_criteria is True
    all_concepts = set()
    for doc in documents:
        if doc.get('met_criteria') == True:
            concepts = doc.get('concepts', [])
            all_concepts.update(concepts)
    
    # Create dictionary with empty strings as values
    concepts_dict = {concept: "" for concept in sorted(all_concepts)}
    
    # Save to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(concepts_dict, f, indent=2)
    
    return concepts_dict


def create_artifact_details(dataset_name: str) -> list:
    # Construct file paths
    input_path = Path(f"data/{dataset_name}/artifact_lists.json")
    output_path = Path(f"data/{dataset_name}/artifact_details.json")
    
    # Read the input JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        artifacts_dict = json.load(f)
    
    # Transform into detailed artifact list
    detailed_artifacts = []
    for artifact_name, artifact_content in artifacts_dict.items():
        # Check if content starts with 'Class' or 'class'
        is_class = (
            artifact_content.strip().lower().startswith('class')
            if artifact_content
            else False
        )
        
        artifact_detail = {
            "artifact_title": artifact_name,
            "artifact_location": "",
            "artifact_content": artifact_content,
            "artifact_type": "Class" if is_class else "",
            "traceability_granularity": "Class" if is_class else ""
        }
        detailed_artifacts.append(artifact_detail)
    
    # Save to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_artifacts, f, indent=2)
    
    return detailed_artifacts


# Usage
if __name__ == "__main__":
    dataset_name = "unity_cataloga"
    # concepts_dict = extract_unique_concepts(dataset_name)
    # print(f"Extracted {len(concepts_dict)} unique concepts")
    # print("First few concepts:", dict(list(concepts_dict.items())[:5]) if concepts_dict else "No concepts found")
    # detailed_artifacts = create_artifact_details(dataset_name)