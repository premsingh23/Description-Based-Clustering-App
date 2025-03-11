import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import pairwise_distances
from scipy import stats


def analyze_dimensions(
    df: pd.DataFrame,
    embeddings: List[List[float]],
    label_column: str,
    reference_label_1: str,
    reference_label_2: str
) -> List[Dict[str, Any]]:
    """
    Analyze embedding dimensions to identify which are most important for distinguishing
    between two reference labels.
    
    Args:
        df: DataFrame containing the data
        embeddings: List of embedding vectors
        label_column: Name of the column containing labels
        reference_label_1: First reference label
        reference_label_2: Second reference label
        
    Returns:
        List of dictionaries with dimension importance information, sorted by importance
    """
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Get indices for each reference label
    ref1_indices = df[df[label_column] == reference_label_1].index.tolist()
    ref2_indices = df[df[label_column] == reference_label_2].index.tolist()
    
    if not ref1_indices or not ref2_indices:
        raise ValueError(f"One or both reference labels not found in the data")
    
    # Get embeddings for each reference label
    ref1_embeddings = embeddings_array[ref1_indices]
    ref2_embeddings = embeddings_array[ref2_indices]
    
    # Calculate mean embeddings for each label
    ref1_mean = np.mean(ref1_embeddings, axis=0)
    ref2_mean = np.mean(ref2_embeddings, axis=0)
    
    # Calculate difference between means
    mean_diff = np.abs(ref1_mean - ref2_mean)
    
    # Calculate statistical significance for each dimension
    p_values = []
    effect_sizes = []
    
    for dim in range(embeddings_array.shape[1]):
        ref1_dim_values = ref1_embeddings[:, dim]
        ref2_dim_values = ref2_embeddings[:, dim]
        
        # Statistical test (t-test)
        t_stat, p_value = stats.ttest_ind(ref1_dim_values, ref2_dim_values, equal_var=False)
        p_values.append(p_value)
        
        # Effect size (Cohen's d)
        mean1, mean2 = np.mean(ref1_dim_values), np.mean(ref2_dim_values)
        std1, std2 = np.std(ref1_dim_values), np.std(ref2_dim_values)
        pooled_std = np.sqrt(((len(ref1_dim_values) - 1) * std1**2 + 
                             (len(ref2_dim_values) - 1) * std2**2) / 
                            (len(ref1_dim_values) + len(ref2_dim_values) - 2))
        
        if pooled_std == 0:
            effect_size = 0
        else:
            effect_size = abs(mean1 - mean2) / pooled_std
            
        effect_sizes.append(effect_size)
    
    # Combine statistics into a composite importance score
    # Higher difference, lower p-value, and higher effect size indicate greater importance
    importance_scores = []
    for i in range(embeddings_array.shape[1]):
        # Convert p-value to importance (1 - p, with minimum to avoid division by zero)
        p_importance = 1 - min(p_values[i], 0.9999)
        
        # Combine mean difference, p-value importance, and effect size
        importance = (mean_diff[i] * p_importance * (1 + effect_sizes[i]))
        importance_scores.append(importance)
    
    # Create dimension info list
    dimension_info = []
    for i in range(embeddings_array.shape[1]):
        dimension_info.append({
            "dimension": i,
            "importance": importance_scores[i],
            "mean_difference": float(mean_diff[i]),
            "p_value": float(p_values[i]),
            "effect_size": float(effect_sizes[i]),
            "ref1_mean": float(ref1_mean[i]),
            "ref2_mean": float(ref2_mean[i])
        })
    
    # Sort by importance (descending)
    dimension_info.sort(key=lambda x: x["importance"], reverse=True)
    
    return dimension_info


def select_reference_labels(
    df: pd.DataFrame,
    embeddings: List[List[float]],
    label_column: str,
    min_count: int = 5,
    max_candidate_labels: int = 20
) -> List[Tuple[str, str, float]]:
    """
    Suggest pairs of reference labels that are well-separated in the embedding space.
    
    Args:
        df: DataFrame containing the data
        embeddings: List of embedding vectors
        label_column: Name of the column containing labels
        min_count: Minimum count of samples per label to consider
        max_candidate_labels: Maximum number of candidate labels to consider
        
    Returns:
        List of tuples (label1, label2, distance) representing good reference label pairs
    """
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Get label counts
    label_counts = df[label_column].value_counts()
    
    # Filter labels with sufficient samples
    candidate_labels = label_counts[label_counts >= min_count].index.tolist()
    
    # Limit number of candidate labels
    candidate_labels = candidate_labels[:max_candidate_labels]
    
    # Calculate mean embedding for each label
    label_mean_embeddings = {}
    
    for label in candidate_labels:
        label_indices = df[df[label_column] == label].index.tolist()
        label_embeddings = embeddings_array[label_indices]
        label_mean = np.mean(label_embeddings, axis=0)
        label_mean_embeddings[label] = label_mean
    
    # Calculate distances between all pairs of label means
    label_pairs = []
    
    for i, label1 in enumerate(candidate_labels):
        for label2 in candidate_labels[i+1:]:
            distance = np.linalg.norm(
                label_mean_embeddings[label1] - label_mean_embeddings[label2]
            )
            label_pairs.append((label1, label2, float(distance)))
    
    # Sort by distance (descending)
    label_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return label_pairs


def calculate_dimension_clusters(dimension_importance: List[Dict[str, Any]], n_clusters: int = 3) -> Dict[str, Any]:
    """
    Cluster dimensions based on their importance to identify groups of related dimensions.
    
    Args:
        dimension_importance: List of dimension importance information
        n_clusters: Number of clusters to form
        
    Returns:
        Dictionary with cluster information
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn is required for dimension clustering")
    
    # Extract dimension features
    dimension_features = []
    for dim_info in dimension_importance:
        features = [
            dim_info["importance"],
            dim_info["mean_difference"],
            dim_info["effect_size"]
        ]
        dimension_features.append(features)
    
    # Convert to numpy array
    features_array = np.array(dimension_features)
    
    # Normalize features
    features_mean = np.mean(features_array, axis=0)
    features_std = np.std(features_array, axis=0)
    features_normalized = (features_array - features_mean) / (features_std + 1e-10)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_normalized)
    
    # Add cluster information to dimensions
    clustered_dimensions = []
    for i, dim_info in enumerate(dimension_importance):
        dim_info_with_cluster = dim_info.copy()
        dim_info_with_cluster["cluster"] = int(cluster_labels[i])
        clustered_dimensions.append(dim_info_with_cluster)
    
    # Group dimensions by cluster
    cluster_groups = {}
    for i in range(n_clusters):
        cluster_dims = [d for d in clustered_dimensions if d["cluster"] == i]
        cluster_groups[f"cluster_{i}"] = {
            "dimensions": [d["dimension"] for d in cluster_dims],
            "avg_importance": np.mean([d["importance"] for d in cluster_dims]),
            "count": len(cluster_dims)
        }
    
    # Sort clusters by average importance
    sorted_clusters = dict(sorted(
        cluster_groups.items(), 
        key=lambda item: item[1]["avg_importance"], 
        reverse=True
    ))
    
    return {
        "clustered_dimensions": clustered_dimensions,
        "cluster_groups": sorted_clusters
    } 