import json
from pathlib import Path
from typing import List, Dict

def create_traceability_json_dataset(dataset_name: str):
    """
    Creates a JSON dataset showing one-to-many relationships between documents and code artifacts.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'crawl4ai')
        
    Returns:
        int: Number of document-to-artifacts mappings created
    """
    # Define paths
    split_docs_path = Path(f"data/{dataset_name}/split_documents.json")
    artifact_details_path = Path(f"data/{dataset_name}/artifact_details.json")
    output_json_path = Path(f"data/{dataset_name}/{dataset_name}.json")
    
    # Load source files
    with open(split_docs_path, 'r', encoding='utf-8') as f:
        split_documents = json.load(f)
    with open(artifact_details_path, 'r', encoding='utf-8') as f:
        artifact_details = json.load(f)
    
    # Create artifact lookup dictionary for faster access
    artifact_lookup = {item['artifact_title']: item for item in artifact_details}
    
    # Prepare the dataset
    traceability_dataset = []
    
    # Filter and process documents
    for doc in split_documents:
        if not doc.get('met_criteria', False):
            continue
            
        concepts = doc.get('concepts', [])
        explanations = doc.get('explanation', [])
        
        # Determine relationship mapping type
        exp_len = len(explanations)
        concept_len = len(concepts)
        
        # Prepare artifacts list for this document
        doc_artifacts = []
        
        for i, concept in enumerate(concepts):
            # Skip if concept not in artifact details
            if concept not in artifact_lookup:
                continue
                
            # Get artifact details
            artifact = artifact_lookup[concept]
            
            # Determine relationship
            relationship = ""
            if exp_len == concept_len:
                relationship = str(explanations[i]) if explanations[i] is not None else ""
            elif exp_len == 1:
                relationship = str(explanations[0]) if explanations[0] is not None else ""
            elif exp_len == 0:
                relationship = ""
            else:
                relationship = "error"
            
            # Create artifact entry
            artifact_entry = {
                "title": concept,  # Keep the original concept/title for reference
                "location": str(artifact['artifact_location']),
                "content": str(artifact['artifact_content']),
                "type": str(artifact['artifact_type']),
                "relationship": relationship,
                "traceability_granularity": str(artifact['traceability_granularity'])
            }
            
            doc_artifacts.append(artifact_entry)
        
        # Only create document entry if it has mapped artifacts
        if doc_artifacts:
            doc_entry = {
                "document": {
                    "text": str(doc['split_text']),
                    "location": str(doc['file_location']),
                    "type": ""
                },
                "artifacts": doc_artifacts
            }
            traceability_dataset.append(doc_entry)
    
    # Write to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(traceability_dataset, f, indent=2)
    
    return len(traceability_dataset)

if __name__ == "__main__":
    dataset_name = "crawl4ai"
    mappings_created = create_traceability_json_dataset(dataset_name)
    print(f"Created {mappings_created} document-to-artifacts mappings in the traceability dataset")