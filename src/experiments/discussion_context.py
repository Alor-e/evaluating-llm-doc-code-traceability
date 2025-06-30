import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
from src.utils.rq1_data_processor import prepare_rq1_data
from src.utils.llm_interface import call_llm

def execute_many_to_many(dataset_name: str, run_id: int = 1, data_dir: str = "data", include_doc_context: bool = False) -> None:
    """
    Executes many-to-many traceability analysis while maintaining controlled experimental conditions.
    """
    # Use existing data preparation to maintain consistency
    prepared_requests = prepare_rq1_data(dataset_name, data_dir, include_doc_context, run_id)

    # System message remains identical to maintain consistency
    system_message = """You are a software traceability expert that maps documentation snippets to code artifacts (classes, methods, class-level attributes), identifying both explicit and implicit traces, explaining relationship types, and constructing precise traceability pathways. Always reply in JSON format."""

    # Create results directory
    results_dir = f"results/d2/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Pool all documents and organize full documents to avoid duplication
    all_docs = []
    full_documents = {}
    
    for request in prepared_requests:
        # Add snippet info
        all_docs.append({
            "text": request["doc_text"],
            "location": request["doc_location"]
        })
        
        # Store full document content only once per unique document
        if request["doc_location"] not in full_documents:
            full_documents[request["doc_location"]] = request["document_file"]

    # Maintain the same cached content structure but organize documents efficiently
    cached_content = {
        "available_artifacts": [
            {
                "artifact_id": artifact["artifact_id"],
                "title": artifact["title"],
                "location": artifact["location"],
                "content": artifact["content"]
            }
            for artifact in prepared_requests[0]["artifact_list"]  # Use first request's artifacts as they're same
        ],
        "directory_tree": prepared_requests[0]["directory_tree"],
        "full_documents": full_documents  # Add full documents without duplication
    }

    # Add document context if available (maintaining consistency with RQ1)
    if include_doc_context:
        cached_content["document_context"] = prepared_requests[0].get("document_context")

    # Construct prompt with same validation and requirements
    prompt_dict = {
        "task": "Documentation to Code Traceability Analysis",
        "description": """Analyze the documentation segments and identify all relevant code artifacts that implement, test, or configure the documented functionality. Document context is provided in the cached content if available.""",
        
        "data": {
            "documents": all_docs
        },
        
        "instructions": {
            "steps": [
                "1. Focus ONLY on the provided text fields when identifying traces",
                "2. The location and file information are provided for context only",
                "3. Analyze each text snippet to identify explicitly mentioned code elements",
                "4. Identify code elements demonstrated in usage examples",
                "5. For each identified relationship:",
                "   - Mark as 'explicit' if directly mentioned",
                "   - Mark as 'implicit' if from usage examples",
                "   - Explain the specific relationship nature",
                "   - Identify any trace chains/pathways",
                "6. Only trace to:",
                "   - Classes and their usage",
                "   - Methods and their usage",
                "   - Class-level attributes in public interface",
                "7. DO NOT trace to:",
                "   - Individual statements",
                "   - Implementation details",
                "   - Local variables or parameters",
                "8. For each relationship:",
                "   - Provide specific evidence",
                "   - Explain relationship in detail",
                "   - Identify intermediate steps",
                "9. Include all required fields",
                "10. Only trace to provided code snippets",
                "11. For traced classes, include base/implementing classes",
                "12. For traceability pathways:",
                "   - Start with document name",
                "   - Use exact artifact titles",
                "   - Follow 'A -> B -> C' format",
                "   - Must end at traced artifact",
                "   - Explain intermediate steps"
            ],
            "output_format": {
                "traces": [
                    {
                        "artifact_id": "integer",
                        "title": "string",
                        "traced_documents": [
                            {
                                "doc_location": "string",
                                "relationship": "explicit|implicit",
                                "relationship_type": "string",
                                "relationship_explanation": "string",
                                "trace_chain": "string",
                                "trace_chain_explanation": "string"
                            }
                        ]
                    }
                ]
            }
        }
    }

    try:
        # Use same LLM interface for consistency
        response = call_llm(
            prompt=json.dumps(prompt_dict, indent=2),
            system_message=system_message,
            results_dir=results_dir,
            data_entry={"dataset": dataset_name},
            cached_message=json.dumps(cached_content, indent=2)
        )

        if response:
    # Process results maintaining same metadata
            results, metrics = process_many_to_many_response(response, prepared_requests)
            save_run_results(dataset_name, run_id, results, metrics, results_dir)

            # Print metrics for monitoring
            print(f"\nRun {run_id} Metrics:")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1: {metrics['f1_score']:.3f}")

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        
        # Save error with same error handling as RQ1
        error_data = {
            "dataset": dataset_name,
            "run_id": run_id,
            "error": str(e)
        }
        
        error_file = os.path.join(results_dir, "errors.json")
        try:
            with open(error_file, 'r') as f:
                errors = json.load(f)
        except FileNotFoundError:
            errors = {"errors": []}
        
        errors["errors"].append(error_data)
        
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)

def process_many_to_many_response(response: Dict, prepared_requests: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Process LLM response maintaining same metrics calculation approach."""
    results = []
    
    # Initialize counters for metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Create complete set of ground truth pairs and tracking
    ground_truth_pairs = set()
    ground_truth_info = {}  # Store additional info for each pair
    processed_pairs = set()
    
    # Build complete ground truth set
    for request in prepared_requests:
        doc_location = request["doc_location"]
        for artifact in request["ground_truth"]["artifacts"]:
            pair = (doc_location, artifact["title"])
            ground_truth_pairs.add(pair)
            ground_truth_info[pair] = {
                "doc_text": request["doc_text"],
                "granularity": artifact.get("traceability_granularity")
            }

    # Process LLM response
    for trace in response.get("traces", []):
        artifact_title = trace["title"]
        
        for doc in trace.get("traced_documents", []):
            doc_location = doc["doc_location"]
            pair = (doc_location, artifact_title)
            processed_pairs.add(pair)
            
            # Determine if this is a correct trace
            is_correct = pair in ground_truth_pairs
            
            if is_correct:
                true_positives += 1
            else:
                false_positives += 1
            
            result = {
                "document_text": ground_truth_info.get(pair, {}).get("doc_text"),
                "document_location": doc_location,
                "artifact_id": trace["artifact_id"],
                "artifact_title": artifact_title,
                "predicted_relationship": doc.get("relationship"),
                "relationship_type": doc.get("relationship_type"),
                "relationship_explanation": doc.get("relationship_explanation"),
                "predicted_trace_chain": doc.get("trace_chain"),
                "predicted_trace_chain_explanation": doc.get("trace_chain_explanation"),
                "confusion_metrics": "True Positive" if is_correct else "False Positive",
                "traceability_granularity": ground_truth_info.get(pair, {}).get("granularity")
            }
            
            results.append(result)
    
    # Process ALL unmatched ground truth pairs as False Negatives
    for pair in ground_truth_pairs - processed_pairs:
        false_negatives += 1
        doc_location, artifact_title = pair
        
        results.append({
            "document_text": ground_truth_info[pair]["doc_text"],
            "document_location": doc_location,
            "artifact_title": artifact_title,
            "confusion_metrics": "False Negative",
            "traceability_granularity": ground_truth_info[pair]["granularity"]
        })
    
    # Calculate metrics
    total_predictions = true_positives + false_positives
    total_ground_truth = len(ground_truth_pairs)
    
    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Verify counts
    assert true_positives + false_negatives == total_ground_truth, \
        f"TP ({true_positives}) + FN ({false_negatives}) should equal ground truth ({total_ground_truth})"
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_predicted": total_predictions,
        "total_ground_truth": total_ground_truth
    }
    
    print(f"\nMetric verification:")
    print(f"Ground truth pairs: {total_ground_truth}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Verification: TP + FN = {true_positives + false_negatives} (should equal ground truth)")
    
    return results, metrics

def save_run_results(dataset_name: str, run_id: int, results: List[Dict], metrics: Dict, results_dir: str):
    """Save results maintaining same format as RQ1."""
    run_results = {
        "dataset": dataset_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "metrics": metrics
    }

    # Save individual run
    run_filename = os.path.join(results_dir, f"run_{run_id}.json")
    with open(run_filename, "w") as f:
        json.dump(run_results, f, indent=2)

    # Update combined results
    combined_filename = os.path.join(results_dir, "combined_results.json")
    try:
        with open(combined_filename, "r") as f:
            combined_results = json.load(f)
    except FileNotFoundError:
        combined_results = {"runs": []}
    
    combined_results["runs"].append(run_results)
    
    with open(combined_filename, "w") as f:
        json.dump(combined_results, f, indent=2)

    # After all runs, calculate and save averaged metrics
    if run_id == 5:  # Assuming 5 runs as in RQ1
        calculate_average_metrics(dataset_name, results_dir)

def calculate_average_metrics(dataset_name: str, results_dir: str):
    """Calculate and save averaged metrics across all runs."""
    with open(os.path.join(results_dir, "combined_results.json"), 'r') as f:
        combined_results = json.load(f)

    num_runs = len(combined_results["runs"])
    avg_metrics = {
        "precision": sum(run["metrics"]["precision"] for run in combined_results["runs"]) / num_runs,
        "recall": sum(run["metrics"]["recall"] for run in combined_results["runs"]) / num_runs,
        "f1_score": sum(run["metrics"]["f1_score"] for run in combined_results["runs"]) / num_runs,
        "true_positives": sum(run["metrics"]["true_positives"] for run in combined_results["runs"]) / num_runs,
        "false_positives": sum(run["metrics"]["false_positives"] for run in combined_results["runs"]) / num_runs,
        "false_negatives": sum(run["metrics"]["false_negatives"] for run in combined_results["runs"]) / num_runs,
        "total_predicted": sum(run["metrics"]["total_predicted"] for run in combined_results["runs"]) / num_runs,
        "total_ground_truth": sum(run["metrics"]["total_ground_truth"] for run in combined_results["runs"]) / num_runs
    }

    # Save averaged metrics
    avg_metrics_file = os.path.join(results_dir, "averaged_metrics.json")
    with open(avg_metrics_file, 'w') as f:
        json.dump({
            "dataset": dataset_name,
            "num_runs": num_runs,
            "averaged_metrics": avg_metrics
        }, f, indent=2)

def main():
    """Run multiple experiments maintaining same methodology as RQ1."""
    datasets = ["crawl4ai", "unity_catalog"]
    num_runs = 5  # Same as RQ1
    
    for dataset_name in datasets:
        for run_id in range(1, num_runs + 1):
            print(f"\nStarting Run {run_id}/{num_runs} for {dataset_name}")
            execute_many_to_many(
                dataset_name=dataset_name,
                run_id=run_id,
                include_doc_context=False  # Maintain RQ1 setting
            )

if __name__ == "__main__":
    main()