import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict
from src.utils.rq1_data_processor import prepare_rq1_data
from src.utils.llm_interface import call_llm


def process_llm_response(llm_response: Dict, request: Dict) -> List[Dict]:
    """
    Process LLM response and calculate metrics by comparing with ground truth.
    Stores all LLM output fields and calculates detailed metrics.
    
    Args:
        llm_response (Dict): LLM's artifact predictions containing full trace analysis
        request (Dict): Original request containing ground truth
        
    Returns:
        List[Dict]: Processed results with metrics and complete trace information
    """
    results = []
    
    # Get predicted artifacts from LLM response
    predicted_artifacts = llm_response.get("artifacts", [])
    # Store original document text from LLM response
    doc_text = request["doc_text"]
    
    # Get ground truth artifacts and create efficient lookup
    ground_truth_titles = set(request["ground_truth"]["artifact_titles"])
    ground_truth_artifacts = {
        artifact['title']: artifact 
        for artifact in request["ground_truth"]["artifacts"]
    }
    
    # Track which predictions matched ground truth
    predicted_titles = set()
    
    # Process each predicted artifact
    for pred_artifact in predicted_artifacts:
        artifact_title = pred_artifact.get("title")
        is_correct = artifact_title in ground_truth_titles
        
        if is_correct:
            predicted_titles.add(artifact_title)
            ground_truth_artifact = ground_truth_artifacts[artifact_title]
        
        # Store complete LLM analysis with metrics
        # For predicted artifacts
        result = {
            # Document Information
            "sent_document_text": doc_text,
            "document_location": request["doc_location"],
            
            # Artifact Identification
            "artifact_id": pred_artifact.get("artifact_id"),
            "artifact_title": artifact_title,
            
            # LLM's Analysis
            "predicted_relationship": pred_artifact.get("relationship"),
            "relationship_type": pred_artifact.get("relationship_type"),
            "relationship_explanation": pred_artifact.get("relationship_explanation"),
            "predicted_trace_chain": pred_artifact.get("trace_chain"),
            "predicted_trace_chain_explanation": pred_artifact.get("trace_chain_explanation"),
            
            # Ground Truth Information
            "ground_truth_relationship": ground_truth_artifact.get("relationship") if is_correct else None,
            "ground_truth_trace_chain": ground_truth_artifact.get("trace_chain") if is_correct else None,
            "traceability_granularity": ground_truth_artifact.get("traceability_granularity"),
            
            # Metric Classification
            "confusion_metrics": "True Positive" if is_correct else "False Positive",
            
            # Analysis Details
            "prediction_details": {
                "matches_ground_truth": is_correct,
                "relationship_match": (
                    is_correct and 
                    pred_artifact.get("relationship_type") == ground_truth_artifact.get("relationship")
                ) if is_correct else False
            }
        }
        
        results.append(result)
    
    # Add False Negatives for missed ground truth artifacts
    missed_titles = ground_truth_titles - predicted_titles
    for title in missed_titles:
        ground_truth_artifact = ground_truth_artifacts.get(title)
        if ground_truth_artifact:
            result = {
                # Document Information
                "sent_document_text": doc_text,
                "document_location": request["doc_location"],
                
                # Artifact Information
                "artifact_title": title,
                
                # Ground Truth Information
                "ground_truth_relationship": ground_truth_artifact.get("relationship"),
                "ground_truth_trace_chain": ground_truth_artifact.get("trace_chain"),
                "traceability_granularity": ground_truth_artifact.get("traceability_granularity"),
                
                # Metrics
                "confusion_metrics": "False Negative",
                
                # Empty fields for LLM predictions since this was missed
                "predicted_relationship": None,
                "relationship_type": None,
                "relationship_explanation": None,
                "predicted_trace_chain": None,
                "predicted_trace_chain_explanation": None,
                
                # Analysis Details
                "prediction_details": {
                    "matches_ground_truth": False,
                    "relationship_match": False,
                    "missed_by_llm": True
                }
            }
            results.append(result)
    
    return results


def save_run_results(dataset_name: str, run_id: int, results: List[Dict], results_dir: str):
    """
    Save the results of a single run.
    
    Args:
        dataset_name (str): Name of the dataset
        run_id (int): Run identifier
        results (List[Dict]): Run results
        results_dir (str): Directory to save results
    """
    run_results = {
        "dataset": dataset_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }

    # Save individual run
    run_filename = f"{results_dir}/run_{run_id}.json"
    with open(run_filename, "w") as f:
        json.dump(run_results, f, indent=2)
    print(f"Run {run_id} results saved to: {run_filename}")

    # Update combined results
    combined_filename = f"{results_dir}/combined_results.json"
    try:
        with open(combined_filename, "r") as f:
            combined_results = json.load(f)
    except FileNotFoundError:
        combined_results = {"runs": []}
    
    combined_results["runs"].append(run_results)
    
    with open(combined_filename, "w") as f:
        json.dump(combined_results, f, indent=2)


def calculate_averages(dataset_name: str):
    """
    Calculates metrics for each run and then averages them for more rigorous evaluation.
    
    Args:
        dataset_name (str): Name of the dataset to analyze
    """
    combined_filename = f"results/rq1/{dataset_name}/combined_results.json"
    with open(combined_filename, "r") as f:
        combined_results = json.load(f)
    
    # Track metrics for each run
    run_metrics = []
    granularity_metrics_per_run = {}
    
    # Collect negative results across all runs
    negative_results = {
        "dataset": dataset_name,
        "runs": []
    }
    
    num_runs = len(combined_results["runs"])
    
    # Process each run
    for run in combined_results["runs"]:
        results = run["results"]
        run_negatives = []
        
        # Initialize counters for this run
        run_counts = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
        
        # Initialize granularity counters for this run
        run_granularity_counts = {}
        
        # Process artifacts
        for artifact in results:
            # Update overall counts
            if artifact["confusion_metrics"] == "True Positive":
                run_counts["true_positives"] += 1
            elif artifact["confusion_metrics"] == "False Positive":
                run_counts["false_positives"] += 1
                run_negatives.append(artifact)
            elif artifact["confusion_metrics"] == "False Negative":
                run_counts["false_negatives"] += 1
                run_negatives.append(artifact)
            
            # Update granularity counts
            granularity = artifact.get("traceability_granularity")
            if granularity:
                if granularity not in run_granularity_counts:
                    run_granularity_counts[granularity] = {
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": 0
                    }
                
                if artifact["confusion_metrics"] == "True Positive":
                    run_granularity_counts[granularity]["true_positives"] += 1
                elif artifact["confusion_metrics"] == "False Positive":
                    run_granularity_counts[granularity]["false_positives"] += 1
                elif artifact["confusion_metrics"] == "False Negative":
                    run_granularity_counts[granularity]["false_negatives"] += 1
        
        # Calculate metrics for this run
        tp = run_counts["true_positives"]
        fp = run_counts["false_positives"]
        fn = run_counts["false_negatives"]
        
        run_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        run_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        run_f1 = 2 * (run_precision * run_recall) / (run_precision + run_recall) if (run_precision + run_recall) > 0 else 0
        
        # Store metrics for this run
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
        
        # Calculate and store granularity metrics for this run
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
        
        # Store negative results
        negative_results["runs"].append({
            "run_id": run["run_id"],
            "timestamp": run["timestamp"],
            "negative_results": run_negatives
        })
    
    # Calculate final averaged metrics
    avg_metrics = {
        "true_positives": sum(r["metrics"]["true_positives"] for r in run_metrics) / num_runs,
        "false_positives": sum(r["metrics"]["false_positives"] for r in run_metrics) / num_runs,
        "false_negatives": sum(r["metrics"]["false_negatives"] for r in run_metrics) / num_runs,
        "precision": sum(r["metrics"]["precision"] for r in run_metrics) / num_runs,
        "recall": sum(r["metrics"]["recall"] for r in run_metrics) / num_runs,
        "f1_score": sum(r["metrics"]["f1_score"] for r in run_metrics) / num_runs,
        "per_run": run_metrics  # Store individual run metrics for analysis
    }
    
    # Calculate averaged granularity metrics
    granularity_averages = {}
    for granularity, run_metrics_list in granularity_metrics_per_run.items():
        granularity_averages[granularity] = {
            "precision": sum(r["metrics"]["precision"] for r in run_metrics_list) / num_runs,
            "recall": sum(r["metrics"]["recall"] for r in run_metrics_list) / num_runs,
            "f1_score": sum(r["metrics"]["f1_score"] for r in run_metrics_list) / num_runs,
            "true_positives": sum(r["metrics"]["true_positives"] for r in run_metrics_list) / num_runs,
            "false_positives": sum(r["metrics"]["false_positives"] for r in run_metrics_list) / num_runs,
            "false_negatives": sum(r["metrics"]["false_negatives"] for r in run_metrics_list) / num_runs,
            "per_run": run_metrics_list  # Store individual run metrics for granularity
        }
    
    avg_metrics["per_granularity"] = granularity_averages
    
    # Save averages
    averages_filename = f"results/rq1/{dataset_name}/averaged_metrics.json"
    with open(averages_filename, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "num_runs": num_runs,
            "averaged_metrics": avg_metrics
        }, f, indent=2)
    print(f"\nAveraged metrics saved to: {averages_filename}")
    
    # Save negative results
    negatives_filename = f"results/rq1/{dataset_name}/negative_results.json"
    with open(negatives_filename, "w") as f:
        json.dump(negative_results, f, indent=2)
    print(f"Negative results saved to: {negatives_filename}")


# === Asynchronous LLM calls ===

async def process_document(idx: int, request: Dict, total: int, system_message: str, results_dir: str) -> List[Dict]:
    """
    Process a single document request asynchronously.
    Prints status messages, builds the prompt, calls the LLM concurrently,
    and processes the response.
    """
    print(f"\nProcessing document {idx}/{total}")

    # Prepare cached content with all available artifacts and optional document context
    cached_content = {
        "available_artifacts": [
            {
                "artifact_id": artifact["artifact_id"],
                "title": artifact["title"],
                "location": artifact["location"],
                "content": artifact["content"]
            }
            for artifact in request["artifact_list"]
        ],
        "directory_tree": request["directory_tree"]
    }

    if request.get("document_context"):
        cached_content["document_context"] = request["document_context"]

    cached_message = json.dumps(cached_content, indent=2)

    # Construct prompt
    prompt_dict = {
        "task": "Documentation to Code Traceability Analysis",
        "description": "Analyze the documentation snippet and identify code artifacts that implement or relate to the documented functionality. Use only artifact names from the provided list.",
        "data": {
            "document": {
                "text": {
                    "value": request["doc_text"],
                    "description": "Documentation text snippet to analyze for trace links"
                },
                "location": {
                    "value": request["doc_location"],
                    "description": "Location information for context"
                },
                "file": {
                    "value": "",  # request["document_file"] omitted to run without file
                    "description": "File from which the document.text snippet was extracted, for context, the file may or may not be included"
                }
            }
        },
        "instructions": {
            "steps": [
                "1. Focus ONLY on the provided 'text' field when identifying traces",
                "2. The location and file information are provided for context only - DO NOT trace to elements mentioned elsewhere in the file",
                "3. Analyze the specific text snippet to identify explicitly mentioned code elements",
                "4. Identify any code elements demonstrated in usage examples within this text",
                "5. For each identified artifact:",
                "   - If it's explicitly mentioned in the text snippet, mark as 'explicit'",
                "   - If it appears in usage examples within this text but isn't directly discussed, mark as 'implicit'",
                "   - Explain the specific nature of how it relates to the documentation",
                "   - Identify if it's part of a chain/pathway to other artifacts",
                "6. Only trace to:",
                "   - Classes and their usage",
                "   - Methods and their usage",
                "   - Class-level attributes that form part of the public interface and their usage",
                "7. DO NOT trace to:",
                "   - Individual statements within methods",
                "   - Implementation details",
                "   - Local variables or parameters",
                "   - Elements mentioned outside the provided text snippet",
                "8. For each relationship:",
                "   - Provide specific evidence from the text",
                "   - Explain the nature of the relationship in detail",
                "   - Identify any intermediate steps or dependencies",
                "9. Make sure to include all required fields in your response",
                "10. Ensure you only trace to the included list of code snippets returning only titles within that list given",
                "11. For any traced class, also trace to its base classes and implementing classes if they are in the available artifacts list",
                "12. For traceability pathways:",
                "   - Always start with the document name (extracted from location)",
                "   - Use exact artifact titles for code elements",
                "   - Strictly follow 'A -> B -> C' format with spaces around arrows",
                "   - Must end at the traced artifact",
                "   - Explain why each intermediate step is necessary"
            ],
            "output_format": {
                "artifacts": [
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
                            "description": "Whether artifact is explicitly mentioned or implicitly used"
                        },
                        "relationship_type": {
                            "type": "string",
                            "description": "Nature of relationship (implements, extends, uses, etc.)"
                        },
                        "relationship_explanation": {
                            "type": "string",
                            "description": "Detailed explanation with evidence from text"
                        },
                        "trace_chain": {
                            "type": "string",
                            "description": "Format: doc_name.md -> Artifact1 -> Artifact2"
                        },
                        "trace_chain_explanation": {
                            "type": "string",
                            "description": "Explanation of chain relationships"
                        }
                    }
                ]
            }
        }
    }

    try:
        # Call the LLM asynchronously by offloading the blocking call to a thread
        response = await asyncio.to_thread(
            call_llm,
            prompt=json.dumps(prompt_dict, indent=2),
            system_message=system_message,
            results_dir=results_dir,
            data_entry=request,
            cached_message=cached_message
        )

        if response:
            print(response)
            result = process_llm_response(response, request)
            return result

    except Exception as e:
        print(f"Error processing document {idx}: {str(e)}")
    
    return []


async def async_execute_rq1(dataset_name: str, run_id: int = 1, data_dir: str = "data", include_doc_context: bool = False) -> None:
    """
    Asynchronously executes RQ1 by processing the dataset, sending traceability prompts to the LLM concurrently,
    and calculating metrics.
    
    Args:
        dataset_name (str): Name of the dataset to process
        run_id (int): Identifier for the current run
        data_dir (str): Base directory where datasets are stored
        include_doc_context (bool): Whether to include full document context
    """
    # Prepare the data using the rq1_data_processor (synchronously)
    prepared_requests: List[Dict] = prepare_rq1_data(dataset_name, data_dir, include_doc_context, run_id)

    # System Message remains focused on traceability
    system_message = (
        "You are a software traceability expert that maps documentation snippets to code artifacts "
        "(classes, methods, class-level attributes), identifying both explicit and implicit traces, "
        "explaining relationship types, and constructing precise traceability pathways. Always reply in JSON format."
    )

    # Store results for this run
    rq1_results = []

    # Create results directory structure
    results_dir = f"results/rq1/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Create and schedule tasks for each document request concurrently
    tasks = []
    total_requests = len(prepared_requests)
    for idx, request in enumerate(prepared_requests, start=1):
        task = asyncio.create_task(process_document(idx, request, total_requests, system_message, results_dir))
        tasks.append(task)

    # Wait for all LLM calls to finish
    task_results = await asyncio.gather(*tasks)
    for result in task_results:
        rq1_results.extend(result)

    # Save run results (synchronous I/O)
    save_run_results(dataset_name, run_id, rq1_results, results_dir)


async def async_run_multiple_experiments(dataset_names: list, num_runs: int = 5):
    """
    Asynchronously executes multiple runs of the RQ1 experiment.
    
    Args:
        dataset_names (list): List of dataset names to process
        num_runs (int): Number of runs to perform
    """
    for dataset_name in dataset_names:
        for run_id in range(3, num_runs + 1):
            print(f"\nStarting Run {run_id}/{num_runs} for dataset '{dataset_name}'")
            await async_execute_rq1(dataset_name=dataset_name, run_id=run_id, include_doc_context=False)
        
        # Optionally calculate and save averages here (synchronous)
        calculate_averages(dataset_name)


if __name__ == "__main__":
    # Example usage:
    # Replace 'unity_catalog' with 'athena_crisis' or 'crawl4ai' as needed
    # For example, to run multiple experiments concurrently:
    asyncio.run(async_run_multiple_experiments(dataset_names=['crawl4ai', 'unity_catalog'], num_runs=5))
    # To run a single experiment (if desired), you could also do:
    # asyncio.run(async_execute_rq1(dataset_name='unity_catalog', run_id=1))