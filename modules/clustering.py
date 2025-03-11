import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def get_clustering_algorithms() -> Dict[str, Dict[str, Any]]:
    """
    Get available clustering algorithms with descriptions and default parameters.
    
    Returns:
        Dictionary mapping algorithm names to metadata
    """
    return {
        "K-Means": {
            "description": "Fast, general-purpose clustering algorithm",
            "parameters": {
                "n_clusters": {
                    "description": "Number of clusters to form",
                    "default": 5,
                    "type": "int",
                    "min": 2,
                    "max": 50
                },
                "random_state": {
                    "description": "Random seed for reproducibility",
                    "default": 42,
                    "type": "int"
                }
            }
        },
        "Hierarchical": {
            "description": "Builds nested clusters by merging or splitting them",
            "parameters": {
                "n_clusters": {
                    "description": "Number of clusters to form",
                    "default": 5,
                    "type": "int",
                    "min": 2,
                    "max": 50
                },
                "affinity": {
                    "description": "Metric used to compute linkage",
                    "default": "euclidean",
                    "type": "categorical",
                    "options": ["euclidean", "l1", "l2", "manhattan", "cosine"]
                },
                "linkage": {
                    "description": "Linkage criteria",
                    "default": "ward",
                    "type": "categorical",
                    "options": ["ward", "complete", "average", "single"]
                }
            }
        },
        "DBSCAN": {
            "description": "Density-based clustering for discovering clusters of arbitrary shape",
            "parameters": {
                "eps": {
                    "description": "Maximum distance between samples for one to be considered as in the neighborhood of the other",
                    "default": 0.5,
                    "type": "float",
                    "min": 0.1,
                    "max": 5.0
                },
                "min_samples": {
                    "description": "Number of samples in a neighborhood for a point to be considered as a core point",
                    "default": 5,
                    "type": "int",
                    "min": 2,
                    "max": 100
                }
            }
        },
        "HDBSCAN": {
            "description": "Hierarchical DBSCAN with varying density clusters",
            "parameters": {
                "min_cluster_size": {
                    "description": "Minimum size of clusters",
                    "default": 5,
                    "type": "int",
                    "min": 2,
                    "max": 100
                },
                "min_samples": {
                    "description": "Number of samples in a neighborhood for a point to be considered as a core point",
                    "default": None,
                    "type": "int",
                    "min": 1,
                    "max": 100
                }
            }
        },
        "Gaussian Mixture": {
            "description": "Probabilistic model for representing normally distributed subpopulations",
            "parameters": {
                "n_components": {
                    "description": "Number of mixture components",
                    "default": 5,
                    "type": "int",
                    "min": 2,
                    "max": 50
                },
                "covariance_type": {
                    "description": "Type of covariance parameters",
                    "default": "full",
                    "type": "categorical",
                    "options": ["full", "tied", "diag", "spherical"]
                }
            }
        }
    }


def cluster_data(
    embeddings: List[List[float]],
    algorithm: str = "K-Means",
    params: Optional[Dict[str, Any]] = None
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Cluster data using the specified algorithm.
    
    Args:
        embeddings: List of embedding vectors
        algorithm: Name of the clustering algorithm to use
        params: Optional parameters for the clustering algorithm
        
    Returns:
        Tuple containing:
        - List of cluster labels for each embedding
        - Dictionary with clustering metrics
    """
    if params is None:
        params = {}
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Apply clustering based on selected algorithm
    if algorithm == "K-Means":
        cluster_labels, metrics = _cluster_kmeans(embeddings_array, params)
    
    elif algorithm == "Hierarchical":
        cluster_labels, metrics = _cluster_hierarchical(embeddings_array, params)
    
    elif algorithm == "DBSCAN":
        cluster_labels, metrics = _cluster_dbscan(embeddings_array, params)
    
    elif algorithm == "HDBSCAN":
        cluster_labels, metrics = _cluster_hdbscan(embeddings_array, params)
    
    elif algorithm == "Gaussian Mixture":
        cluster_labels, metrics = _cluster_gaussian_mixture(embeddings_array, params)
    
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    # Convert labels to Python list
    cluster_labels_list = cluster_labels.tolist()
    
    # Add general clustering metrics if applicable
    if len(set(cluster_labels_list)) > 1 and -1 not in cluster_labels:  # Ensure valid clustering for metrics
        try:
            silhouette = silhouette_score(embeddings_array, cluster_labels)
            metrics["silhouette_score"] = float(silhouette)
        except Exception:
            metrics["silhouette_score"] = None
            
        try:
            ch_score = calinski_harabasz_score(embeddings_array, cluster_labels)
            metrics["calinski_harabasz_score"] = float(ch_score)
        except Exception:
            metrics["calinski_harabasz_score"] = None
            
        try:
            db_score = davies_bouldin_score(embeddings_array, cluster_labels)
            metrics["davies_bouldin_score"] = float(db_score)
        except Exception:
            metrics["davies_bouldin_score"] = None
    
    return cluster_labels_list, metrics


def _cluster_kmeans(embeddings_array: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply K-Means clustering."""
    from sklearn.cluster import KMeans
    
    # Get parameters
    n_clusters = params.get("n_clusters", 5)
    random_state = params.get("random_state", 42)
    
    # Initialize and fit model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Collect metrics
    metrics = {
        "inertia": float(kmeans.inertia_),
        "n_clusters": n_clusters,
        "n_iter": int(kmeans.n_iter_),
        "algorithm_name": "K-Means"
    }
    
    return cluster_labels, metrics


def _cluster_hierarchical(embeddings_array: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply Hierarchical clustering."""
    from sklearn.cluster import AgglomerativeClustering
    
    # Get parameters
    n_clusters = params.get("n_clusters", 5)
    affinity = params.get("affinity", "euclidean")
    linkage = params.get("linkage", "ward")
    
    # Initialize and fit model
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        linkage=linkage
    )
    cluster_labels = hierarchical.fit_predict(embeddings_array)
    
    # Collect metrics
    metrics = {
        "n_clusters": n_clusters,
        "affinity": affinity,
        "linkage": linkage,
        "algorithm_name": "Hierarchical"
    }
    
    # Calculate additional metrics if distance matrix is available
    if hasattr(hierarchical, "distances_"):
        metrics["max_distance"] = float(np.max(hierarchical.distances_))
        metrics["mean_distance"] = float(np.mean(hierarchical.distances_))
    
    return cluster_labels, metrics


def _cluster_dbscan(embeddings_array: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply DBSCAN clustering."""
    from sklearn.cluster import DBSCAN
    
    # Get parameters
    eps = params.get("eps", 0.5)
    min_samples = params.get("min_samples", 5)
    
    # Initialize and fit model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings_array)
    
    # Count number of clusters and noise points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    # Collect metrics
    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "eps": eps,
        "min_samples": min_samples,
        "algorithm_name": "DBSCAN"
    }
    
    return cluster_labels, metrics


def _cluster_hdbscan(embeddings_array: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply HDBSCAN clustering."""
    try:
        import hdbscan
    except ImportError:
        raise ImportError("HDBSCAN is not installed. Install with 'pip install hdbscan'")
    
    # Get parameters
    min_cluster_size = params.get("min_cluster_size", 5)
    min_samples = params.get("min_samples", None)
    
    # Initialize and fit model
    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    cluster_labels = hdbscan_clusterer.fit_predict(embeddings_array)
    
    # Count number of clusters and noise points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    # Collect metrics
    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "algorithm_name": "HDBSCAN"
    }
    
    # Add HDBSCAN-specific metrics if available
    if hasattr(hdbscan_clusterer, "outlier_scores_"):
        metrics["mean_outlier_score"] = float(np.mean(hdbscan_clusterer.outlier_scores_))
        metrics["max_outlier_score"] = float(np.max(hdbscan_clusterer.outlier_scores_))
    
    if hasattr(hdbscan_clusterer, "probabilities_"):
        metrics["mean_cluster_probability"] = float(np.mean(hdbscan_clusterer.probabilities_))
        metrics["min_cluster_probability"] = float(np.min(hdbscan_clusterer.probabilities_))
    
    return cluster_labels, metrics


def _cluster_gaussian_mixture(embeddings_array: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply Gaussian Mixture Model clustering."""
    from sklearn.mixture import GaussianMixture
    
    # Get parameters
    n_components = params.get("n_components", 5)
    covariance_type = params.get("covariance_type", "full")
    random_state = params.get("random_state", 42)
    
    # Initialize and fit model
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )
    gmm.fit(embeddings_array)
    cluster_labels = gmm.predict(embeddings_array)
    
    # Collect metrics
    metrics = {
        "n_components": n_components,
        "covariance_type": covariance_type,
        "aic": float(gmm.aic(embeddings_array)),
        "bic": float(gmm.bic(embeddings_array)),
        "n_iter": int(gmm.n_iter_),
        "algorithm_name": "Gaussian Mixture"
    }
    
    return cluster_labels, metrics


def optimize_clustering(
    embeddings: List[List[float]],
    algorithm: str = "K-Means",
    param_ranges: Optional[Dict[str, List[Any]]] = None
) -> Dict[str, Any]:
    """
    Find optimal clustering parameters by testing multiple configurations.
    
    Args:
        embeddings: List of embedding vectors
        algorithm: Clustering algorithm to optimize
        param_ranges: Dictionary mapping parameter names to lists of values to try
        
    Returns:
        Dictionary with optimal parameters and metrics
    """
    # Define default parameter ranges if not provided
    if param_ranges is None:
        if algorithm == "K-Means" or algorithm == "Hierarchical" or algorithm == "Gaussian Mixture":
            param_ranges = {
                "n_clusters": list(range(2, 21))  # Test from 2 to 20 clusters
            }
        elif algorithm == "DBSCAN":
            param_ranges = {
                "eps": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
                "min_samples": [3, 5, 10, 15, 20]
            }
        elif algorithm == "HDBSCAN":
            param_ranges = {
                "min_cluster_size": [3, 5, 10, 15, 20, 30],
                "min_samples": [None, 3, 5, 10, 15]
            }
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Generate all parameter combinations
    import itertools
    param_keys = list(param_ranges.keys())
    param_values = [param_ranges[key] for key in param_keys]
    
    # Track best parameters and metrics
    best_score = -np.inf
    best_params = {}
    best_labels = []
    best_metrics = {}
    
    # Try each parameter combination
    for param_combination in itertools.product(*param_values):
        params = {key: value for key, value in zip(param_keys, param_combination)}
        
        # Run clustering with these parameters
        try:
            labels, metrics = cluster_data(embeddings, algorithm, params)
            
            # Use silhouette score as the optimization metric (if available)
            score = metrics.get("silhouette_score")
            
            # For DBSCAN and HDBSCAN, also consider number of clusters
            if algorithm in ["DBSCAN", "HDBSCAN"]:
                # Penalize having too many noise points
                n_noise = metrics.get("n_noise", 0)
                noise_ratio = n_noise / len(embeddings)
                
                # Penalize having too few clusters (1 or 0)
                n_clusters = metrics.get("n_clusters", 0)
                
                # Only consider valid clusterings (with reasonable noise level and more than one cluster)
                if score is not None and noise_ratio < 0.5 and n_clusters > 1:
                    # Adjust score based on noise ratio
                    score = score * (1 - noise_ratio)
                else:
                    score = None
            
            # Update best parameters if this is the best score
            if score is not None and score > best_score:
                best_score = score
                best_params = params
                best_labels = labels
                best_metrics = metrics
                
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            continue
    
    # Return results
    return {
        "best_params": best_params,
        "best_labels": best_labels,
        "best_metrics": best_metrics,
        "algorithm": algorithm
    } 