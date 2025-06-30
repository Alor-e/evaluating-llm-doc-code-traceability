
ERROR_GROUPS = {
    "CBV": "Context Boundary Violation - Trace extends beyond document segment",
    "IAE": "Implicit Assumption Error - Unsupported inferred relationship",
    "PAL": "Phantom Artifact Link - Non-existent code artifact reference",
    "APB": "Architectural Pattern Bias - Assumed undocumented pattern",
    "IOL": "Implementation Overlink - Internal implementation detail leak",
    "CCE": "Chain Construction Error - Invalid trace pathway steps",
    "CUS": "Custom Unseen Scenario - Requires manual investigation"
}

def calculate_metrics(data):
    # Initialize counters for all error groups
    group_counts = {group: 0 for group in data['error_groups']}
    total_cases = len(data['cases'])
    
    # Count occurrences in cases
    for case in data['cases']:
        for group in case['groups']:
            group_counts[group] += 1
    
    # Prepare groups with percentages
    groups = {}
    for group in data['error_groups']:
        count = group_counts[group]
        percentage = round((count / total_cases) * 100, 1)
        groups[group] = {
            "count": count,
            "percentage": percentage
        }
    
    # Create cleaned cases without original_analysis
    cleaned_cases = [
        {k: v for k, v in case.items() if k != 'original_analysis'}
        for case in data['cases']
    ]
    
    return {
        "metrics": {
            "total_false_positives": total_cases,
            "groups": groups
        },
        "error_groups": data['error_groups'],
        "cases": cleaned_cases
    }

# Usage example with your data:
if __name__ == "__main__":
    import json
    
    # Load your original JSON data
    with open('results/rq1/unity_catalog/grouped_false_positives.json') as f:
        original_data = json.load(f)
    
    # Calculate metrics and clean cases
    result = calculate_metrics(original_data)
    
    # Print the metrics section
    print(json.dumps(result['metrics'], indent=2))
    
    # Optional: Save full cleaned data
    with open('cleaned_data.json', 'w') as f:
        json.dump(result, f, indent=2)