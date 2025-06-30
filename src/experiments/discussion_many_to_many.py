# d2_many_to_many.py

import os
import json
from datetime import datetime
from typing import List, Dict
from src.utils.rq1_data_processor import prepare_rq1_data
from src.utils.llm_interface import call_llm

# Core Many-to-Many Execution and Metrics

def execute_d2(
    dataset_name: str, 
    run_id: int = 1, 
    data_dir: str = "data", 
    include_doc_context: bool = False
) -> None:
    """
    Executes a many-to-many traceability analysis by aggregating all documentation snippets
    and all available code artifacts into one single LLM request.
    
    It then processes the aggregated LLM response to compute metrics (TP, FP, FN, etc.)
    in a manner analogous to RQ1.
    
    Args:
        dataset_name (str): Name of the dataset to process.
        run_id (int): Identifier for the current run.
        data_dir (str): Base directory where datasets are stored.
        include_doc_context (bool): Whether to include full document context (default: False).
    """
    # Prepare data using your existing data processor.
    # Each prepared request is expected to contain "doc_text", "doc_location", "artifact_list",
    # and "ground_truth" (with "artifact_titles" and "artifacts").
    prepared_requests: List[Dict] = prepare_rq1_data(dataset_name, data_dir, include_doc_context, run_id)

    # Aggregate all documentation snippets.
    # Assign a unique "doc_id" (since location is not unique)
    documents = []
    for idx, req in enumerate(prepared_requests, start=1):
        # We no longer include the full document text in the prompt output.
        # Instead, we provide a unique doc_id and the location.
        doc_entry = {
            "doc_id": f"doc_{idx}",
            "location": {
                "value": req["doc_location"],
                "description": "Document location (not unique)"
            }
            # Optionally, you can include a short snippet if needed, but we omit the full text.
        }
        # Optionally include document context if enabled.
        if include_doc_context and req.get("document_context"):
            doc_entry["context"] = req["document_context"]
        documents.append(doc_entry)
        # Also, store the doc_id in the request for later matching.
        req["doc_id"] = f"doc_{idx}"

    # Aggregate all code artifacts (deduplicated by artifact_id).
    artifacts_lookup = {}
    for req in prepared_requests:
        for artifact in req["artifact_list"]:
            art_id = artifact["artifact_id"]
            if art_id not in artifacts_lookup:
                artifacts_lookup[art_id] = {
                    "artifact_id": artifact["artifact_id"],
                    "title": artifact["title"],
                    "location": artifact["location"],
                    "content": artifact["content"]
                }
    available_artifacts = list(artifacts_lookup.values())

    # Build the aggregated prompt.
    # Note: We no longer instruct the LLM to return the full document text.
    prompt_dict = {
        "task": "Documentation to Code Traceability Analysis",
        "description": (
            "Analyze the complete set of documentation snippets and identify code artifacts "
            "that implement or relate to the documented functionality. Use only artifact names "
            "from the provided list. For each trace, return the unique 'doc_id' of the document along with "
            "the corresponding trace details."
        ),
        "data": {
            "documents": documents,
            "available_artifacts": available_artifacts
        },
        "instructions": {
            "steps": [
                "1. Focus ONLY on the provided document snippets when identifying traces.",
                "2. The document location is provided for context only; do not use it as a unique identifier.",
                "3. Analyze each document snippet to identify explicitly mentioned code elements.",
                "4. Identify any code elements demonstrated in usage examples within the snippets.",
                "5. For each identified artifact:",
                "   - If it is explicitly mentioned in the snippet, mark it as 'explicit'.",
                "   - If it appears in usage examples but isn’t directly discussed, mark it as 'implicit'.",
                "   - Explain the specific nature of its relation to the documentation.",
                "   - Identify if it is part of a chain/pathway to other artifacts.",
                "6. Only trace to:",
                "   - Classes and their usage.",
                "   - Methods and their usage.",
                "   - Class-level attributes that form part of the public interface.",
                "7. Do not trace to:",
                "   - Individual statements within methods.",
                "   - Implementation details, local variables, or parameters.",
                "   - Elements mentioned outside the provided snippets.",
                "8. For each relationship:",
                "   - Provide specific evidence from the snippet.",
                "   - Explain the nature of the relationship in detail.",
                "   - Identify any intermediate steps or dependencies.",
                "9. Ensure you only trace to artifacts from the provided list (using exact artifact titles).",
                "10. For any traced class, also trace to its base classes and implementing classes if available.",
                "11. For traceability pathways:",
                "   - Return the unique 'doc_id' of the document.",
                "   - Use exact artifact titles for code elements.",
                "   - Strictly follow the 'A -> B -> C' format with spaces around arrows.",
                "   - The pathway must end at the traced artifact.",
                "   - Explain why each intermediate step is necessary."
            ],
            "output_format": {
                "traces": [
                    {
                        "doc_id": {
                            "type": "string",
                            "description": "Unique identifier for the document snippet"
                        },
                        "traced_artifacts": [
                            {
                                "artifact_id": {
                                    "type": "integer",
                                    "description": "Unique identifier for the artifact"
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Exact artifact title from available_artifacts"
                                },
                                "relationship": {
                                    "type": "string",
                                    "enum": ["explicit", "implicit"],
                                    "description": "Whether the artifact is explicitly mentioned or implicitly used"
                                },
                                "relationship_type": {
                                    "type": "string",
                                    "description": "Nature of relationship (implements, extends, uses, etc.)"
                                },
                                "relationship_explanation": {
                                    "type": "string",
                                    "description": "Detailed explanation with evidence from the snippet"
                                },
                                "trace_chain": {
                                    "type": "string",
                                    "description": "Format: doc_id -> Artifact1 -> Artifact2"
                                },
                                "trace_chain_explanation": {
                                    "type": "string",
                                    "description": "Explanation of chain relationships"
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }

    system_message = (
        "You are a software traceability expert that maps documentation snippets to code artifacts. "
        "Given the complete set of documents (each with a unique doc_id) and available code artifacts, "
        "identify all trace links between them. Reply strictly in JSON format."
    )

    # Define the directory to store results.
    results_dir = f"results/d2/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Make the LLM call and save the raw output.
    try:
        raw_output = call_llm(
            prompt=json.dumps(prompt_dict, indent=2),
            system_message=system_message,
            results_dir=results_dir,
            data_entry={"documents": documents},  # for error logging
            cached_message=""
        )
        print("Raw LLM output:")
        print(raw_output)
    except Exception as e:
        print(f"Error during LLM call: {e}")
        raw_output = ""

    # Save the raw output as text.
    raw_response_filename = os.path.join(results_dir, f"raw_llm_response_run_{run_id}.txt")
    with open(raw_response_filename, "w") as f:
        # If raw_output is a dict, convert it to a JSON-formatted string.
        if isinstance(raw_output, dict):
            f.write(json.dumps(raw_output, indent=2))
        else:
            f.write(str(raw_output))
    print(f"Raw LLM response saved to: {raw_response_filename}")

    # Attempt to parse the raw output as JSON.
    try:
        llm_response = json.loads(json.dumps(raw_output)) if isinstance(raw_output, dict) else json.loads(raw_output)
    except Exception as e:
        print(f"Error parsing raw LLM output as JSON: {e}")
        llm_response = None

    # Process the LLM response to compute metrics.
    if llm_response:
        processed_results = process_many_to_many_response(llm_response, prepared_requests)
    else:
        processed_results = []

    # Save the run results (including processed metrics).
    save_run_results_d2(dataset_name, run_id, processed_results, results_dir)

# Processing & Metrics Functions

def process_many_to_many_response(aggregated_response: Dict, prepared_requests: List[Dict]) -> List[Dict]:
    """
    Processes the aggregated LLM response (which contains trace mappings for all documents)
    and computes metrics by comparing predicted trace links against the ground truth.
    
    This version loops over every document in prepared_requests.
    If a document is missing in the LLM response, all its ground truth artifacts are marked as false negatives.
    
    Returns:
        List[Dict]: A list of processed results containing detailed metrics.
    """
    results = []
    # Build a lookup from doc_id (returned by the LLM) to the trace.
    trace_lookup = {}
    for trace in aggregated_response.get("traces", []):
        doc_id = trace.get("doc_id")
        if doc_id:
            trace_lookup[doc_id] = trace

    # Process every document in the prepared_requests.
    for req in prepared_requests:
        doc_id = req["doc_id"]
        doc_text = req["doc_text"]  # We'll later append the original text if needed.
        ground_truth_titles = set(req["ground_truth"]["artifact_titles"])
        ground_truth_artifacts = {
            artifact["title"]: artifact for artifact in req["ground_truth"]["artifacts"]
        }
        predicted_titles = set()

        if doc_id in trace_lookup:
            trace = trace_lookup[doc_id]
            predicted_artifacts = trace.get("traced_artifacts", [])
            for pred_artifact in predicted_artifacts:
                artifact_title = pred_artifact.get("title")
                is_correct = artifact_title in ground_truth_titles
                if is_correct:
                    predicted_titles.add(artifact_title)
                result = {
                    "sent_document_text": doc_text,
                    "doc_id": doc_id,
                    "artifact_id": pred_artifact.get("artifact_id"),
                    "artifact_title": artifact_title,
                    "predicted_relationship": pred_artifact.get("relationship"),
                    "relationship_type": pred_artifact.get("relationship_type"),
                    "relationship_explanation": pred_artifact.get("relationship_explanation"),
                    "predicted_trace_chain": pred_artifact.get("trace_chain"),
                    "predicted_trace_chain_explanation": pred_artifact.get("trace_chain_explanation"),
                    "ground_truth_relationship": ground_truth_artifacts[artifact_title].get("relationship") if is_correct else None,
                    "ground_truth_trace_chain": ground_truth_artifacts[artifact_title].get("trace_chain") if is_correct else None,
                    "traceability_granularity": ground_truth_artifacts[artifact_title].get("traceability_granularity") if is_correct else None,
                    "confusion_metrics": "True Positive" if is_correct else "False Positive",
                    "prediction_details": {
                        "matches_ground_truth": is_correct,
                        "relationship_match": (is_correct and pred_artifact.get("relationship_type") == ground_truth_artifacts[artifact_title].get("relationship")) if is_correct else False
                    }
                }
                results.append(result)
        # Mark missing ground truth artifacts as false negatives.
        missed_titles = ground_truth_titles - predicted_titles
        for title in missed_titles:
            ground_truth_artifact = ground_truth_artifacts.get(title)
            if ground_truth_artifact:
                result = {
                    "sent_document_text": doc_text,
                    "doc_id": doc_id,
                    "artifact_title": title,
                    "ground_truth_relationship": ground_truth_artifact.get("relationship"),
                    "ground_truth_trace_chain": ground_truth_artifact.get("trace_chain"),
                    "traceability_granularity": ground_truth_artifact.get("traceability_granularity"),
                    "confusion_metrics": "False Negative",
                    "predicted_relationship": None,
                    "relationship_type": None,
                    "relationship_explanation": None,
                    "predicted_trace_chain": None,
                    "predicted_trace_chain_explanation": None,
                    "prediction_details": {
                        "matches_ground_truth": False,
                        "relationship_match": False,
                        "missed_by_llm": True
                    }
                }
                results.append(result)
    return results

def save_run_results_d2(dataset_name: str, run_id: int, results: List[Dict], results_dir: str):
    """
    Saves the results of a single run and updates the combined results file.
    
    Args:
        dataset_name (str): Name of the dataset.
        run_id (int): Run identifier.
        results (List[Dict]): Processed results with metrics.
        results_dir (str): Directory to save results.
    """
    run_results = {
        "dataset": dataset_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }

    # Save individual run.
    run_filename = os.path.join(results_dir, f"run_{run_id}.json")
    with open(run_filename, "w") as f:
        json.dump(run_results, f, indent=2)
    print(f"Run {run_id} results saved to: {run_filename}")

    # Update combined results.
    combined_filename = os.path.join(results_dir, "combined_results.json")
    try:
        with open(combined_filename, "r") as f:
            combined_results = json.load(f)
    except FileNotFoundError:
        combined_results = {"runs": []}
    
    combined_results["runs"].append(run_results)
    with open(combined_filename, "w") as f:
        json.dump(combined_results, f, indent=2)

def calculate_averages_d2(dataset_name: str):
    """
    Calculates and saves averaged metrics across runs for the many-to-many analysis.
    
    Args:
        dataset_name (str): Name of the dataset.
    """
    combined_filename = os.path.join("results", "d2", dataset_name, "combined_results.json")
    with open(combined_filename, "r") as f:
        combined_results = json.load(f)

    run_metrics = []
    granularity_metrics_per_run = {}
    negative_results = {"dataset": dataset_name, "runs": []}
    num_runs = len(combined_results["runs"])

    for run in combined_results["runs"]:
        results = run["results"]
        run_negatives = []
        run_counts = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
        run_granularity_counts = {}

        for artifact in results:
            if artifact["confusion_metrics"] == "True Positive":
                run_counts["true_positives"] += 1
            elif artifact["confusion_metrics"] == "False Positive":
                run_counts["false_positives"] += 1
                run_negatives.append(artifact)
            elif artifact["confusion_metrics"] == "False Negative":
                run_counts["false_negatives"] += 1
                run_negatives.append(artifact)

            granularity = artifact.get("traceability_granularity")
            if granularity:
                if granularity not in run_granularity_counts:
                    run_granularity_counts[granularity] = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
                if artifact["confusion_metrics"] == "True Positive":
                    run_granularity_counts[granularity]["true_positives"] += 1
                elif artifact["confusion_metrics"] == "False Positive":
                    run_granularity_counts[granularity]["false_positives"] += 1
                elif artifact["confusion_metrics"] == "False Negative":
                    run_granularity_counts[granularity]["false_negatives"] += 1

        tp = run_counts["true_positives"]
        fp = run_counts["false_positives"]
        fn = run_counts["false_negatives"]
        run_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        run_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        run_f1 = 2 * (run_precision * run_recall) / (run_precision + run_recall) if (run_precision + run_recall) > 0 else 0

        run_metrics.append({
            "run_id": run["run_id"],
            "metrics": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": run_precision,
                "recall": run_recall,
                "f1_score": run_f1
            }
        })

        for granularity, counts in run_granularity_counts.items():
            if granularity not in granularity_metrics_per_run:
                granularity_metrics_per_run[granularity] = []
            g_tp = counts["true_positives"]
            g_fp = counts["false_positives"]
            g_fn = counts["false_negatives"]
            g_precision = g_tp / (g_tp + g_fp) if (g_tp + g_fp) > 0 else 0
            g_recall = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0
            g_f1 = 2 * (g_precision * g_recall) / (g_precision + g_recall) if (g_precision + g_recall) > 0 else 0
            granularity_metrics_per_run[granularity].append({
                "run_id": run["run_id"],
                "metrics": {
                    "true_positives": g_tp,
                    "false_positives": g_fp,
                    "false_negatives": g_fn,
                    "precision": g_precision,
                    "recall": g_recall,
                    "f1_score": g_f1
                }
            })

        negative_results["runs"].append({
            "run_id": run["run_id"],
            "timestamp": run["timestamp"],
            "negative_results": run_negatives
        })

    avg_metrics = {
        "true_positives": sum(r["metrics"]["true_positives"] for r in run_metrics) / num_runs,
        "false_positives": sum(r["metrics"]["false_positives"] for r in run_metrics) / num_runs,
        "false_negatives": sum(r["metrics"]["false_negatives"] for r in run_metrics) / num_runs,
        "precision": sum(r["metrics"]["precision"] for r in run_metrics) / num_runs,
        "recall": sum(r["metrics"]["recall"] for r in run_metrics) / num_runs,
        "f1_score": sum(r["metrics"]["f1_score"] for r in run_metrics) / num_runs,
        "per_run": run_metrics
    }

    granularity_averages = {}
    for granularity, run_metrics_list in granularity_metrics_per_run.items():
        granularity_averages[granularity] = {
            "precision": sum(r["metrics"]["precision"] for r in run_metrics_list) / num_runs,
            "recall": sum(r["metrics"]["recall"] for r in run_metrics_list) / num_runs,
            "f1_score": sum(r["metrics"]["f1_score"] for r in run_metrics_list) / num_runs,
            "true_positives": sum(r["metrics"]["true_positives"] for r in run_metrics_list) / num_runs,
            "false_positives": sum(r["metrics"]["false_positives"] for r in run_metrics_list) / num_runs,
            "false_negatives": sum(r["metrics"]["false_negatives"] for r in run_metrics_list) / num_runs,
            "per_run": run_metrics_list
        }
    avg_metrics["per_granularity"] = granularity_averages

    averages_filename = os.path.join("results", "d2", dataset_name, "averaged_metrics.json")
    with open(averages_filename, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "num_runs": num_runs,
            "averaged_metrics": avg_metrics
        }, f, indent=2)
    print(f"\nAveraged metrics saved to: {averages_filename}")

    negatives_filename = os.path.join("results", "d2", dataset_name, "negative_results.json")
    with open(negatives_filename, "w") as f:
        json.dump(negative_results, f, indent=2)
    print(f"Negative results saved to: {negatives_filename}")

# Multiple Experiments Entry Point

def run_multiple_d2_experiments(
    dataset_names: List[str],
    num_runs: int = 1,
    data_dir: str = "data",
    include_doc_context: bool = False
) -> None:
    """
    Executes multiple runs of the many-to-many traceability analysis for each dataset provided.
    
    Args:
        dataset_names (List[str]): List of dataset names.
        num_runs (int): Number of runs per dataset.
        data_dir (str): Base directory for datasets.
        include_doc_context (bool): Whether to include document context.
    """
    for dataset_name in dataset_names:
        for run_id in range(1, num_runs + 1):
            print(f"\nStarting run {run_id} for dataset '{dataset_name}'")
            execute_d2(dataset_name, run_id, data_dir, include_doc_context)
        # After all runs, calculate averaged metrics.
        calculate_averages_d2(dataset_name)

def main():
    # Example usage: specify an array of datasets and number of runs per dataset.
    datasets = ["unity_catalog", "crawl4ai"]
    num_runs = 5
    run_multiple_d2_experiments(dataset_names=datasets, num_runs=num_runs, include_doc_context=False)

if __name__ == "__main__":
    main()
