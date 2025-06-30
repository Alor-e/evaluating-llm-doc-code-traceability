# src/config/config.py

from typing import List

# ---------------------------
# Experiment Configuration
# ---------------------------

# Number of runs to execute for each RQ
# Example: If NUM_RUNS is set to 3, each RQ will be executed three times (run_0, run_1, run_2)
NUM_RUNS: int = 3

# List of Research Questions (RQs) to run
# Available RQs: 'rq1', 'rq2', 'rq3'
# You can modify this list to include only the RQs you want to execute
RUN_RQS: List[str] = ['rq1', 'rq2', 'rq3']

# Starting run index
# Useful if you want to resume experiments from a specific run number
# Example: If START_RUN_INDEX is set to 2, the runs will start from run_2
START_RUN_INDEX: int = 0

# ---------------------------
# Directory Structure Configuration
# ---------------------------

# Base directory where all results will be saved
RESULTS_DIR: str = 'results'

# Subdirectory within RESULTS_DIR for RQs
RQS_DIR: str = 'rqs'

# Function to generate run directory names based on run index
def get_run_directory(run_index: int) -> str:
    """
    Generates the run directory name based on the run index.

    Args:
        run_index (int): The index number of the run.

    Returns:
        str: The formatted run directory name (e.g., 'run_0').
    """
    return f"run_{run_index}"

# ---------------------------
# Example Usage
# ---------------------------
# In your experiment scripts (e.g., rq1_traceability.py), you can import these configurations as follows:

# from config.config import NUM_RUNS, RUN_RQS, START_RUN_INDEX, RESULTS_DIR, RQS_DIR, get_run_directory

# Then, use these configurations to control the flow of your experiments and organize results.

