"""
Model utilities.

Provides helpers for:
- serializing and deserializing model weights
- averaging weights across workers (parameter server role)
"""

import numpy as np
from typing import Dict, Any, List


def serialize_weights(weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Convert numpy arrays to list for JSON transport."""
    return {k: v.tolist() for k, v in weights.items()}


def deserialize_weights(weights: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert lists back to numpy arrays."""
    return {k: np.array(v) for k, v in weights.items()}


def average_weights(all_weights: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Average weights across multiple workers."""
    if not all_weights:
        return {}

    # Initialize average with zeros of the same shape
    avg_weights = {k: np.zeros_like(v) for k, v in all_weights[0].items()}

    for weights in all_weights:
        for k, v in weights.items():
            avg_weights[k] += v

    for k in avg_weights:
        avg_weights[k] /= len(all_weights)

    return avg_weights
