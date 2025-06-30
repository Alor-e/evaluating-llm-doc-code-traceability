# src/utils/document_extractor.py
from pathlib import Path
import json
import logging
import pandas as pd
from typing import Dict, Set

class DocumentExtractor:
    def __init__(self):
        self.base_path = Path.home() / "Desktop"
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def get_unique_doc_locations(self, dataset_name: str) -> Set[str]:
        """Get unique document locations from dataset"""
        df = pd.read_csv(f"data/{dataset_name}/{dataset_name}.csv")
        return set(df["Document Location"].unique())

    def read_document_content(self, repo_path: Path, doc_location: str) -> str:
        """Read document content handling different formats"""
        full_path = repo_path / doc_location
        
        try:
            if not full_path.exists():
                logging.error(f"Document not found: {full_path}")
                return ""

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logging.error(f"Error reading {full_path}: {str(e)}")
            return ""

    def extract_documents(self, dataset_name: str):
        """Extract and save full documents for a dataset"""
        logging.info(f"Processing {dataset_name}")
        
        # Get unique document locations
        doc_locations = self.get_unique_doc_locations(dataset_name)
        logging.info(f"Found {len(doc_locations)} unique documents")
        
        # Repository path
        repo_path = self.base_path / dataset_name
        if not repo_path.exists():
            logging.error(f"Repository not found: {repo_path}")
            return
        
        # Extract documents
        documents = {}
        for doc_location in doc_locations:
            content = self.read_document_content(repo_path, doc_location)
            if content:
                documents[doc_location] = content
                logging.info(f"Extracted: {doc_location}")
            else:
                logging.warning(f"Failed to extract: {doc_location}")
        
        # Save to JSON
        output_path = Path(f"data/{dataset_name}/full_documents.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(documents)} documents to {output_path}")
        return documents

    def extract_all_datasets(self):
        """Extract documents for all datasets"""
        datasets = ["unity_catalog", "athena_crisis", "crawl4ai"]
        
        for dataset in datasets:
            self.extract_documents(dataset)

if __name__ == "__main__":
    extractor = DocumentExtractor()
    extractor.extract_all_datasets()