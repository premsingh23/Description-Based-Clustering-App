import numpy as np
from typing import List, Dict


def apply_simple_weighting(embeddings: List[List[float]], importance: List[Dict], top_n: int = 10, factor: float = 2.0) -> List[List[float]]:
    arr = np.array(embeddings)
    weights = np.ones(arr.shape[1])
    top_dims = [d["dimension"] for d in importance[:top_n]]
    for dim in top_dims:
        weights[dim] = factor
    return (arr * weights).tolist()
