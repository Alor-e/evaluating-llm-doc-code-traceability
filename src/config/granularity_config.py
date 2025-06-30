# granularity_config.py
from typing import Dict, List

# Revised granularity groups based on logical levels and distribution
GRANULARITY_MAPPING = {
    "statement_level": [  # Low-level code constructs
        "Statement",
        "Statement-level",
        "Block",
        "Block-level",
        "Parameter-level",
        "Property-level"
    ],
    "function_level": [  # Core executable units
        "Method",
        "Method-level",
        "Function",
        "Function-level"
    ],
    "class_level": [  # Design/organizational units
        "Class",
        "Class-level",
        "Component",
        "Component-level"
    ],
    "file_level": [  # System organization
        "File",
        "File-level",
        "Directory",
        "Directory-level"
    ]
}

def get_standardized_granularity(granularity: str) -> str:
    """Convert various granularity terms to one of the three main levels"""
    granularity = granularity.lower().strip()
    for standard, variations in GRANULARITY_MAPPING.items():
        if any(variation.lower() in granularity for variation in variations):
            return standard
    return "other"

