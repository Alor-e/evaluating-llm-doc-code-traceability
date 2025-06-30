import json
import csv
from pathlib import Path

def create_traceability_dataset(dataset_name: str):
    # Read both JSON files
    split_docs_path = Path(f"data/{dataset_name}/split_documents.json")
    artifact_details_path = Path(f"data/{dataset_name}/artifact_details.json")
    output_csv_path = Path(f"data/{dataset_name}/{dataset_name}.csv")
    
    # Load JSONs
    with open(split_docs_path, 'r', encoding='utf-8') as f:
        split_documents = json.load(f)
    with open(artifact_details_path, 'r', encoding='utf-8') as f:
        artifact_details = json.load(f)
    
    # Create artifact lookup dictionary for faster access
    artifact_lookup = {item['artifact_title']: item for item in artifact_details}
    
    # CSV headers
    headers = [
        'Document Text', 'Document Location', 'Document Type',
        'Artifact Location', 'Artifact Content Representation',
        'Artifact Content', 'Artifact Type', 'Relationship',
        'Traceability Pathway Depth', 'Traceability Granularity'
    ]
    
    # Prepare CSV rows
    csv_rows = []
    
    # Filter and process documents
    for doc in split_documents:
        if not doc.get('met_criteria', False):
            continue
            
        concepts = doc.get('concepts', [])
        explanations = doc.get('explanation', [])
        
        # Determine relationship mapping type
        exp_len = len(explanations)
        concept_len = len(concepts)
        
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
            
            # Create traceability pathway depth
            pathway_depth = f"{doc['file_location']} -> {artifact['artifact_location']}"
            
            # Create row with safe string conversions
            row = {
                'Document Text': str(doc['split_text']),
                'Document Location': str(doc['file_location']),
                'Document Type': '',
                'Artifact Location': str(artifact['artifact_location']),
                'Artifact Content Representation': '',
                'Artifact Content': str(artifact['artifact_content']),
                'Artifact Type': str(artifact['artifact_type']),
                'Relationship': relationship,
                'Traceability Pathway Depth': pathway_depth,
                'Traceability Granularity': str(artifact['traceability_granularity'])
            }
            
            csv_rows.append(row)
    
    # Write to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    return len(csv_rows)

if __name__ == "__main__":
    dataset_name = "crawl4ai"
    rows_created = create_traceability_dataset(dataset_name)
    print(f"Created {rows_created} rows in the traceability dataset")