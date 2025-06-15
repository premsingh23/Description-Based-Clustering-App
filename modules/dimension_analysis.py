import numpy as np
import pandas as pd
from typing import List, Dict


def compute_dimension_variance(df: pd.DataFrame, embeddings: List[List[float]], label_col: str) -> List[Dict]:
    """Return dimensions ranked by variance across label means."""
    arr = np.array(embeddings)
    labels = df[label_col].unique()
    variances = []
    for i in range(arr.shape[1]):
        means = [arr[df[label_col] == lbl, i].mean() for lbl in labels]
        variances.append({"dimension": i, "variance": float(np.var(means))})
    return sorted(variances, key=lambda x: x["variance"], reverse=True)
