import json
import os
import asyncio
from typing import Dict, List, Optional
from src.utils.llm_base_interface import call_base_llm

class RelationshipEvaluator:
    def __init__(self, dataset_name: str, data_dir: str = "data"):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.results_dir = f"results/rq2/{dataset_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.artifact_lookup = self.load_artifact_details()
        
    def load_artifact_details(self) -> dict:
        """Load artifact details and create lookup by title"""
        artifact_details_path = os.path.join(self.data_dir, self.dataset_name, "artifact_details.json")
        with open(artifact_details_path, 'r') as f:
            artifacts = json.load(f)
        return {artifact['artifact_title']: artifact['artifact_content'] for artifact in artifacts}
    
    def load_combined_results(self) -> dict:
        """Load combined results from RQ1"""
        results_path = f"results/rq1/{self.dataset_name}/combined_results.json"
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def evaluate_relationship(
        self, doc_text: str, code_content: str, 
        predicted: str, ground_truth: str
    ) -> dict:
        """
        (No changes here; your prompt remains the same)
        """
        prompt = {
            "task": "Judge Relationship Description (3-category)",
            "context": {
                "code": code_content,
                "documentation": doc_text
            },
            "descriptions": {
                "predicted": predicted,
                "ground_truth": ground_truth
            },
            "instructions": """
            You are an expert judge comparing two relationship descriptions:
            (1) predicted vs. (2) ground_truth.

            Focus on whether the predicted description gets the high-level 
            relationship right. Do NOT penalize minor omissions 
            about code details or parameters. Only label as partially_correct if 
            the predicted text contradicts or significantly misunderstands a 
            major aspect of the ground truth. Everything else can be correct.

            Label as:
            - "correct" if the predicted description basically captures the 
                same high-level relationship described by the ground truth, 
                even if some minor details are omitted.
            - "partially_correct" if it covers some of the core idea but 
                introduces a misunderstanding or omits a crucial aspect that 
                changes the overall meaning.
            - "incorrect" if it is largely or completely inconsistent with 
                the ground truth (e.g., describing a different class relationship 
                or functionality).

            Provide:
            - alignment_label: correct, partially_correct, or incorrect
            - justification: short reason
            - error_type: if not correct (e.g., "major_contradiction")

            IMPORTANT:
            - Do NOT generate new content; only evaluate alignment. 
            - Treat missing details or code specifics as unimportant, 
                as long as the main relationship is correct.
            """,
            "output_format": {
                "alignment_label": "string", 
                "justification": "string",
                "error_type": "string"
            }
        }

        system_message = (
            "You are an impartial evaluator. Determine whether the predicted relationship "
            "is correct, partially_correct, or incorrect relative to the ground_truth. "
            "Do not generate a new explanation. Ensure your response is in JSON only."
        )

        try:
            response = call_base_llm(
                prompt=json.dumps(prompt),
                system_message=system_message
            )
            print(response)
            return response
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None

    def process_results(self, run_id: int = None):
        """
        Optionally accepts run_id to differentiate outputs for multiple runs.
        This is the original synchronous version.
        """
        combined_results = self.load_combined_results()
        evaluations = []
        
        # Tracking metrics for 3-class labeling
        metrics = {
            "total_evaluated": 0,
            "correct": 0,
            "partially_correct": 0,
            "incorrect": 0,
            "error_types": {}
        }
        
        for run in combined_results['runs']:
            for result in run['results']:
                # Only evaluate if the result is 'True Positive' from RQ1
                if result['confusion_metrics'] == "True Positive":
                    code_content = self.artifact_lookup.get(result['artifact_title'])
                    if not code_content:
                        print(f"Warning: No code content found for {result['artifact_title']}")
                        continue

                    evaluation = self.evaluate_relationship(
                        doc_text=result['sent_document_text'],
                        code_content=code_content,
                        predicted=result['relationship_explanation'],
                        ground_truth=result['ground_truth_relationship']
                    )
                    
                    if evaluation:
                        # Record evaluation
                        evaluation_record = {
                            "document_text": result['sent_document_text'],
                            "artifact_title": result['artifact_title'],
                            "predicted_relationship": result['relationship_explanation'],
                            "ground_truth_relationship": result['ground_truth_relationship'],
                            "traceability_granularity": result.get('traceability_granularity'),
                            "evaluation": evaluation
                        }
                        evaluations.append(evaluation_record)
                        
                        # Update metrics
                        label = evaluation["alignment_label"]
                        metrics["total_evaluated"] += 1
                        
                        if label == "correct":
                            metrics["correct"] += 1
                        elif label == "partially_correct":
                            metrics["partially_correct"] += 1
                        elif label == "incorrect":
                            metrics["incorrect"] += 1
                        
                        # Track error types if not correct
                        if label != "correct":
                            error_type = evaluation.get("error_type", "unknown")
                            metrics["error_types"][error_type] = (
                                metrics["error_types"].get(error_type, 0) + 1
                            )
        
        # Calculate final metrics
        total = metrics["total_evaluated"]
        if total > 0:
            metrics["strict_accuracy"] = metrics["correct"] / total
            metrics["relaxed_accuracy"] = (
                (metrics["correct"] + metrics["partially_correct"]) / total
            )
        
        # Save results (optionally with run_id in filename)
        self.save_results(evaluations, metrics, run_id)
        
        return metrics
    
    def save_results(self, evaluations: List[Dict], metrics: Dict, run_id: int = None):
        """Save all results and metrics. If run_id is provided, store them in separate files."""
        
        # If run_id is not None, save to different filenames
        if run_id is not None:
            eval_filename = f"relationship_evaluations_run_{run_id}.json"
            metrics_filename = f"metrics_run_{run_id}.json"
        else:
            eval_filename = "relationship_evaluations.json"
            metrics_filename = "metrics.json"
        
        # Save raw evaluations
        with open(os.path.join(self.results_dir, eval_filename), 'w') as f:
            json.dump(evaluations, f, indent=2)
        
        # Save metrics
        with open(os.path.join(self.results_dir, metrics_filename), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def process_results_multiple_runs(self, n_runs: int = 1) -> dict:
        """
        Run process_results(run_id=i) multiple times, 
        store each run's results, then average them.
        """
        all_runs = []
        for i in range(1, n_runs + 1):
            # Each run is identified by run_id=i
            run_metrics = self.process_results(run_id=i)
            all_runs.append(run_metrics)
        
        # Initialize an aggregated dictionary
        aggregated = {
            "total_evaluated": 0,
            "correct": 0,
            "partially_correct": 0,
            "incorrect": 0,
            "error_types": {},
            "strict_accuracy": 0.0,
            "relaxed_accuracy": 0.0
        }
        
        # Sum up metrics over all runs
        for metrics in all_runs:
            aggregated["total_evaluated"] += metrics.get("total_evaluated", 0)
            aggregated["correct"] += metrics.get("correct", 0)
            aggregated["partially_correct"] += metrics.get("partially_correct", 0)
            aggregated["incorrect"] += metrics.get("incorrect", 0)
            
            # Merge error types
            for etype, count in metrics.get("error_types", {}).items():
                aggregated["error_types"][etype] = \
                    aggregated["error_types"].get(etype, 0) + count
            
            aggregated["strict_accuracy"] += metrics.get("strict_accuracy", 0.0)
            aggregated["relaxed_accuracy"] += metrics.get("relaxed_accuracy", 0.0)
        
        # Average the metrics
        if n_runs > 0:
            aggregated["total_evaluated"] = int(aggregated["total_evaluated"] / n_runs)
            aggregated["correct"] = int(aggregated["correct"] / n_runs)
            aggregated["partially_correct"] = int(aggregated["partially_correct"] / n_runs)
            aggregated["incorrect"] = int(aggregated["incorrect"] / n_runs)
            aggregated["strict_accuracy"] /= n_runs
            aggregated["relaxed_accuracy"] /= n_runs
        
        # (Optional) Save an aggregated metrics file
        agg_path = os.path.join(self.results_dir, "aggregated_metrics.json")
        with open(agg_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        return aggregated

    # === Asynchronous Functions (Non-Substantive, for Concurrency) ===

    async def async_evaluate_relationship(
        self, doc_text: str, code_content: str, 
        predicted: str, ground_truth: str
    ) -> dict:
        """
        Asynchronous version of evaluate_relationship.
        Wraps the blocking LLM call in asyncio.to_thread so that evaluations run concurrently.
        """
        prompt = {
            "task": "Judge Relationship Description (3-category)",
            "context": {
                "code": code_content,
                "documentation": doc_text
            },
            "descriptions": {
                "predicted": predicted,
                "ground_truth": ground_truth
            },
            "instructions": """
            You are an expert judge comparing two relationship descriptions:
            (1) predicted vs. (2) ground_truth.

            Focus on whether the predicted description gets the high-level 
            relationship right. Do NOT penalize minor omissions 
            about code details or parameters. Only label as partially_correct if 
            the predicted text contradicts or significantly misunderstands a 
            major aspect of the ground truth. Everything else can be correct.

            Label as:
            - "correct" if the predicted description basically captures the 
                same high-level relationship described by the ground truth, 
                even if some minor details are omitted.
            - "partially_correct" if it covers some of the core idea but 
                introduces a misunderstanding or omits a crucial aspect that 
                changes the overall meaning.
            - "incorrect" if it is largely or completely inconsistent with 
                the ground truth (e.g., describing a different class relationship 
                or functionality).

            Provide:
            - alignment_label: correct, partially_correct, or incorrect
            - justification: short reason
            - error_type: if not correct (e.g., "major_contradiction")

            IMPORTANT:
            - Do NOT generate new content; only evaluate alignment. 
            - Treat missing details or code specifics as unimportant, 
                as long as the main relationship is correct.
            """,
            "output_format": {
                "alignment_label": "string", 
                "justification": "string",
                "error_type": "string"
            }
        }

        system_message = (
            "You are an impartial evaluator. Determine whether the predicted relationship "
            "is correct, partially_correct, or incorrect relative to the ground_truth. "
            "Do not generate a new explanation. Ensure your response is in JSON only."
        )

        try:
            response = await asyncio.to_thread(
                call_base_llm,
                prompt=json.dumps(prompt),
                system_message=system_message
            )
            print(response)
            return response
        except Exception as e:
            print(f"Error during async evaluation: {str(e)}")
            return None

    async def async_process_results(self, run_id: int = None) -> dict:
        """
        Asynchronously process results by concurrently evaluating each 'True Positive' entry.
        """
        combined_results = self.load_combined_results()
        evaluations = []
        
        # Tracking metrics for 3-class labeling
        metrics = {
            "total_evaluated": 0,
            "correct": 0,
            "partially_correct": 0,
            "incorrect": 0,
            "error_types": {}
        }
        
        tasks = []  # Each element: (result_entry, async_task)
        
        for run in combined_results['runs']:
            for result in run['results']:
                # Only evaluate if the result is 'True Positive' from RQ1
                if result['confusion_metrics'] == "True Positive":
                    code_content = self.artifact_lookup.get(result['artifact_title'])
                    if not code_content:
                        print(f"Warning: No code content found for {result['artifact_title']}")
                        continue

                    task = asyncio.create_task(
                        self.async_evaluate_relationship(
                            doc_text=result['sent_document_text'],
                            code_content=code_content,
                            predicted=result['relationship_explanation'],
                            ground_truth=result['ground_truth_relationship']
                        )
                    )
                    tasks.append((result, task))
        
        # Wait for all async evaluations to complete
        responses = await asyncio.gather(*(t for (_, t) in tasks))
        
        # Process each evaluation response
        for ((result, _), evaluation) in zip(tasks, responses):
            if evaluation:
                evaluation_record = {
                    "document_text": result['sent_document_text'],
                    "artifact_title": result['artifact_title'],
                    "predicted_relationship": result['relationship_explanation'],
                    "ground_truth_relationship": result['ground_truth_relationship'],
                    "traceability_granularity": result.get('traceability_granularity'),
                    "evaluation": evaluation
                }
                evaluations.append(evaluation_record)
                
                label = evaluation["alignment_label"]
                metrics["total_evaluated"] += 1
                
                if label == "correct":
                    metrics["correct"] += 1
                elif label == "partially_correct":
                    metrics["partially_correct"] += 1
                elif label == "incorrect":
                    metrics["incorrect"] += 1
                
                if label != "correct":
                    error_type = evaluation.get("error_type", "unknown")
                    metrics["error_types"][error_type] = metrics["error_types"].get(error_type, 0) + 1
        
        if metrics["total_evaluated"] > 0:
            metrics["strict_accuracy"] = metrics["correct"] / metrics["total_evaluated"]
            metrics["relaxed_accuracy"] = (metrics["correct"] + metrics["partially_correct"]) / metrics["total_evaluated"]
        
        self.save_results(evaluations, metrics, run_id)
        return metrics

    async def async_process_results_multiple_runs(self, n_runs: int = 1) -> dict:
        """
        Asynchronously run process_results (with run_id) multiple times and then average the metrics.
        """
        all_runs = []
        for i in range(1, n_runs + 1):
            run_metrics = await self.async_process_results(run_id=i)
            all_runs.append(run_metrics)
        
        aggregated = {
            "total_evaluated": 0,
            "correct": 0,
            "partially_correct": 0,
            "incorrect": 0,
            "error_types": {},
            "strict_accuracy": 0.0,
            "relaxed_accuracy": 0.0
        }
        
        for m in all_runs:
            aggregated["total_evaluated"] += m.get("total_evaluated", 0)
            aggregated["correct"] += m.get("correct", 0)
            aggregated["partially_correct"] += m.get("partially_correct", 0)
            aggregated["incorrect"] += m.get("incorrect", 0)
            
            for etype, count in m.get("error_types", {}).items():
                aggregated["error_types"][etype] = aggregated["error_types"].get(etype, 0) + count
            
            aggregated["strict_accuracy"] += m.get("strict_accuracy", 0.0)
            aggregated["relaxed_accuracy"] += m.get("relaxed_accuracy", 0.0)
        
        if n_runs > 0:
            aggregated["total_evaluated"] = int(aggregated["total_evaluated"] / n_runs)
            aggregated["correct"] = int(aggregated["correct"] / n_runs)
            aggregated["partially_correct"] = int(aggregated["partially_correct"] / n_runs)
            aggregated["incorrect"] = int(aggregated["incorrect"] / n_runs)
            aggregated["strict_accuracy"] /= n_runs
            aggregated["relaxed_accuracy"] /= n_runs
        
        agg_path = os.path.join(self.results_dir, "aggregated_metrics.json")
        with open(agg_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        return aggregated

# === Asynchronous Main Entry Point ===

async def async_main():
    """Asynchronous main execution function"""
    dataset_names = ["crawl4ai", "unity_catalog"]  # Multiple dataset names

    for ds_name in dataset_names:
        print(f"\n=== Running evaluation for dataset: {ds_name} ===")
        evaluator = RelationshipEvaluator(ds_name)

        # Example: run 1 time and average (adjust n_runs as needed)
        aggregated_metrics = await evaluator.async_process_results_multiple_runs(n_runs=1)

        print("\nEvaluation (Averaged over Multiple Runs) Complete!")
        print(f"Total Evaluated (avg): {aggregated_metrics['total_evaluated']}")
        print(f"Correct (avg): {aggregated_metrics['correct']}")
        print(f"Partially Correct (avg): {aggregated_metrics['partially_correct']}")
        print(f"Incorrect (avg): {aggregated_metrics['incorrect']}")
        print(f"\nStrict Accuracy (avg): {aggregated_metrics['strict_accuracy']:.3f}")
        print(f"Relaxed Accuracy (avg): {aggregated_metrics['relaxed_accuracy']:.3f}")
        print("\nError Types (sum across runs):")
        for error_type, count in aggregated_metrics.get('error_types', {}).items():
            print(f"  {error_type}: {count}")

if __name__ == "__main__":
    asyncio.run(async_main())
