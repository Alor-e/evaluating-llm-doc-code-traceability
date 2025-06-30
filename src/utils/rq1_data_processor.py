# src/utils/rq1_data_processor.py

import copy
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Optional

def load_dataset_files(dataset_name: str, data_dir: str = "data") -> tuple:
    """
    Loads all necessary dataset files for traceability analysis.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'crawl4ai')
        data_dir (str): Base directory containing dataset files
    
    Returns:
        tuple: Contains loaded ground truth data, all artifacts, directory tree, and full documents
    """
    base_path = Path(data_dir) / dataset_name
    
    # Load main traceability dataset (ground truth)
    with open(base_path / f"{dataset_name}.json", 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
        ground_truth = ground_truth # Limit to first 10 for testing
    
    # Load all available artifacts (from artifact_details.json)
    with open(base_path / "artifact_details.json", 'r', encoding='utf-8') as f:
        all_artifacts = json.load(f)
        
    # Load directory tree for context
    with open(base_path / "directory_tree.txt", 'r', encoding='utf-8') as f:
        directory_tree = f.read()
        
    # Load full documents for additional context
    with open(base_path / "full_documents.json", 'r', encoding='utf-8') as f:
        full_documents = json.load(f)
    
    return ground_truth, all_artifacts, directory_tree, full_documents

def get_document_content(doc_location: str, full_documents: Dict) -> str:
    """
    Retrieves full document content for added context.
    
    Args:
        doc_location (str): Document location/path
        full_documents (Dict): Dictionary of full document contents
    
    Returns:
        str: Full document content or empty string if not found
    """
    return full_documents.get(doc_location, "")

def get_all_documents_context(ground_truth: List[Dict]) -> List[Dict]:
    """
    Extracts a list of all documents and their locations for context.
    
    Args:
        ground_truth (List[Dict]): The ground truth dataset
        
    Returns:
        List[Dict]: List of documents with their locations
    """
    return [
        {
            "text": entry["document"]["text"],
            "location": entry["document"]["location"]
        }
        for entry in ground_truth
    ]

def prepare_rq1_data(dataset_name: str, data_dir: str = "data", include_doc_context: bool = False, run_id: int = 1) -> List[Dict]:
    """
    Prepares data for RQ1 traceability analysis by organizing documents and their potential artifacts.
    Each document becomes a separate "request" with all necessary context for LLM evaluation.
    Includes shuffling of both documents and artifacts for robust evaluation.
    
    Args:
        dataset_name (str): Name of the dataset
        data_dir (str): Base directory for datasets
        include_doc_context (bool): Whether to include full document context
        run_id (int): Run identifier for organizing shuffled datasets
    
    Returns:
        List[Dict]: List of prepared requests with shuffled ordering
    """
    # Create run directory
    run_dir = os.path.join(data_dir, dataset_name, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Load all necessary data
    ground_truth, all_artifacts, directory_tree, full_documents = load_dataset_files(dataset_name, data_dir)
    
    # Shuffle dataset while preserving relationships
    shuffled_dataset = copy.deepcopy(ground_truth)
    random.shuffle(shuffled_dataset)  # Shuffle documents
    
    for doc in shuffled_dataset:
        random.shuffle(doc['artifacts'])  # Shuffle artifacts within each document
    
    # Save shuffled dataset for reference
    with open(os.path.join(run_dir, "shuffled_dataset.json"), "w") as f:
        json.dump(shuffled_dataset, f, indent=2)
    
    # Get document context if requested
    doc_context = get_all_documents_context(shuffled_dataset) if include_doc_context else None
    
    prepared_requests = []
    
    # Process each document in shuffled dataset
    for doc_entry in shuffled_dataset:
        document = doc_entry['document']
        ground_truth_artifacts = doc_entry['artifacts']
        
        # Get full document content
        doc_content = get_document_content(document['location'], full_documents)
        
        # Create a lookup of ground truth artifact titles for easy reference
        ground_truth_titles = {artifact['title'] for artifact in ground_truth_artifacts}
        
        # Create the request object
        request = {
            "doc_text": document['text'],
            "doc_location": document['location'],
            "document_file": doc_content,
            "directory_tree": directory_tree,
            "document_context": doc_context,  # Will be None if not requested
            
            # Store ALL artifacts as available choices for LLM
            "artifact_list": [
                {
                    "artifact_id": idx,  # Preserve original indexing
                    "title": artifact['artifact_title'],
                    "location": artifact['artifact_location'],
                    "content": artifact['artifact_content']
                }
                for idx, artifact in enumerate(all_artifacts)
            ],
            
            # Store ground truth for metric calculation
            "ground_truth": {
                "artifacts": ground_truth_artifacts,
                "artifact_titles": list(ground_truth_titles)
            }
        }
        
        prepared_requests.append(request)
    
    return prepared_requests


def shuffle_dataset(dataset: List[Dict]) -> List[Dict]:
    """
    Shuffles both document order and artifact order while preserving relationships
    """
    shuffled = copy.deepcopy(dataset)
    
    # Shuffle documents
    random.shuffle(shuffled)
    
    # Shuffle artifacts within each document
    for doc in shuffled:
        random.shuffle(doc['artifacts'])
        
    return shuffled


if __name__ == "__main__":
   # Test the data preparation
   prepared_data = prepare_rq1_data('crawl4ai', include_doc_context=True, run_id=1)
   print(f"Prepared {len(prepared_data)} requests")
   
   # Print sample of first request
   if prepared_data:
       print("\nSample request structure:")
       sample = prepared_data[0]
       print(f"Document: {sample['doc_location']}")
       print(f"Total available artifacts: {len(sample['artifact_list'])}")
       print(f"Ground truth artifacts: {len(sample['ground_truth']['artifacts'])}")
       print(f"Document context available: {sample['document_context'] is not None}")