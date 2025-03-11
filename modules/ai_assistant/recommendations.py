import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import string


def recommend_preprocessing(df: pd.DataFrame, desc_col: str) -> Dict[str, Any]:
    """
    Recommend preprocessing options based on text data.
    
    Args:
        df: DataFrame containing the data
        desc_col: Column containing descriptions
        
    Returns:
        Dictionary with recommendations
    """
    # Sample text to analyze
    sample_texts = df[desc_col].dropna().sample(min(5, len(df))).astype(str).tolist()
    
    # Analyze text characteristics
    has_uppercase = any(any(c.isupper() for c in text) for text in sample_texts)
    has_punctuation = any(any(c in string.punctuation for c in text) for text in sample_texts)
    has_numbers = any(any(c.isdigit() for c in text) for text in sample_texts)
    
    # Detect language characteristics
    avg_word_count = np.mean([len(text.split()) for text in sample_texts])
    avg_char_count = np.mean([len(text) for text in sample_texts])
    
    # Check for technical content
    technical_indicators = [
        "algorithm", "data", "function", "method", "system", 
        "analysis", "protocol", "parameter", "value", "code"
    ]
    
    technical_score = 0
    for text in sample_texts:
        text_lower = text.lower()
        for indicator in technical_indicators:
            if indicator in text_lower:
                technical_score += 1
    
    is_technical = technical_score > (len(sample_texts) * 0.3)
    
    # Generate recommendations
    recommended_options = {
        "lowercase": has_uppercase,
        "remove_punctuation": has_punctuation and not is_technical,
        "remove_numbers": has_numbers and not is_technical,
        "remove_whitespace": True,
        "remove_stopwords": True,
        "lemmatize": avg_word_count > 10,  # Lemmatize longer texts
        "stem": False,  # Typically lemmatization is preferred over stemming
        "remove_short_words": True
    }
    
    # Generate message
    characteristics = []
    if has_uppercase:
        characteristics.append("mixed case text")
    if has_punctuation:
        characteristics.append("punctuation")
    if has_numbers:
        characteristics.append("numeric values")
    if is_technical:
        characteristics.append("technical content")
        
    char_text = ", ".join(characteristics) if characteristics else "simple text"
    
    message = f"Based on the sample descriptions, your data contains {char_text}.\n\n"
    message += f"Average description length: {avg_word_count:.1f} words ({avg_char_count:.1f} characters)\n\n"
    
    message += "Recommended preprocessing steps:\n"
    for option, enabled in recommended_options.items():
        if enabled:
            message += f"• {option.replace('_', ' ').title()}\n"
    
    if is_technical:
        message += "\nNote: Technical content detected. Preserving numbers and punctuation is recommended."
        
    return {
        "message": message,
        "recommended_options": recommended_options,
        "avg_word_count": float(avg_word_count),
        "avg_char_count": float(avg_char_count),
        "is_technical": is_technical
    }


def recommend_reference_labels(
    df: pd.DataFrame,
    embeddings: List[List[float]],
    label_column: str,
    min_count: int = 5
) -> List[Tuple[str, str, float]]:
    """
    Recommend pairs of reference labels that are well-separated in the embedding space.
    
    Args:
        df: DataFrame containing the data
        embeddings: List of embedding vectors
        label_column: Name of the column containing labels
        min_count: Minimum count of samples per label to consider
        
    Returns:
        List of tuples (label1, label2, distance) representing good reference label pairs
    """
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Get label counts
    label_counts = df[label_column].value_counts()
    
    # Filter labels with sufficient samples
    candidate_labels = label_counts[label_counts >= min_count].index.tolist()
    
    # Limit number of candidate labels if there are too many
    max_candidates = 20
    if len(candidate_labels) > max_candidates:
        candidate_labels = candidate_labels[:max_candidates]
    
    if len(candidate_labels) < 2:
        return []  # Not enough labels
    
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


def recommend_weighting(dimension_importance: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recommend weighting scheme based on dimension importance.
    
    Args:
        dimension_importance: List of dimension importance information
        
    Returns:
        Dictionary with weighting recommendations
    """
    # Calculate statistics about dimension importance
    importances = [d["importance"] for d in dimension_importance]
    importance_mean = np.mean(importances)
    importance_std = np.std(importances)
    importance_range = max(importances) - min(importances)
    
    # Analyze the importance distribution
    sorted_importances = sorted(importances, reverse=True)
    
    # Calculate rate of importance drop-off
    if len(sorted_importances) >= 10:
        top_10_drop = (sorted_importances[0] - sorted_importances[9]) / sorted_importances[0]
    else:
        top_10_drop = 0
    
    # Get p-values and effect sizes if available
    has_stats = all(["p_value" in d and "effect_size" in d for d in dimension_importance])
    
    if has_stats:
        p_values = [d["p_value"] for d in dimension_importance]
        effect_sizes = [d["effect_size"] for d in dimension_importance]
        
        # Count significant dimensions
        sig_dims = sum(1 for p in p_values if p < 0.05)
        high_effect_dims = sum(1 for e in effect_sizes if e > 0.8)
    else:
        sig_dims = 0
        high_effect_dims = 0
    
    # Determine recommended scheme
    if has_stats and high_effect_dims > 0:
        scheme = "statistical"
        explanation = "Statistical weighting is recommended because there are dimensions with strong effect sizes."
    elif top_10_drop > 0.8:
        scheme = "exponential"
        explanation = "Exponential weighting is recommended due to the steep drop-off in importance scores."
    elif top_10_drop > 0.5:
        scheme = "linear"
        explanation = "Linear weighting is recommended based on the moderate drop-off in importance scores."
    elif importance_std < 0.1 * importance_mean:
        scheme = "binary"
        explanation = "Binary weighting is recommended since dimensions have similar importance scores."
    else:
        scheme = "logarithmic"
        explanation = "Logarithmic weighting is recommended for balanced emphasis across dimensions."
    
    # Determine number of dimensions to emphasize
    if scheme == "exponential":
        # For exponential, focus on fewer dimensions
        top_n = max(5, int(len(dimension_importance) * 0.1))
    elif scheme == "binary":
        # For binary, include more dimensions
        top_n = max(10, int(len(dimension_importance) * 0.3))
    elif has_stats:
        # If we have statistical data, use significant dimensions
        top_n = max(5, sig_dims)
    else:
        # Default: use about 20% of dimensions
        top_n = max(5, int(len(dimension_importance) * 0.2))
    
    # Cap at reasonable maximum
    top_n = min(top_n, 50)
    
    # Determine weight factor
    if scheme == "exponential":
        weight_factor = 3.0  # Higher for exponential to emphasize top dims
    elif scheme == "binary":
        weight_factor = 5.0  # Higher for binary (on/off effect)
    else:
        weight_factor = 2.0  # Moderate for other schemes
    
    # Generate message
    message = f"Based on the dimension analysis, I recommend using the {scheme} weighting scheme "
    message += f"with the top {top_n} dimensions.\n\n"
    message += explanation + "\n\n"
    
    if has_stats:
        message += f"Found {sig_dims} statistically significant dimensions "
        message += f"and {high_effect_dims} dimensions with large effect sizes.\n\n"
    
    message += f"A weight factor of {weight_factor} is recommended for this scheme."
    
    return {
        "scheme": scheme,
        "top_n": top_n,
        "weight_factor": weight_factor,
        "explanation": explanation,
        "message": message,
        "importance_stats": {
            "mean": float(importance_mean),
            "std": float(importance_std),
            "range": float(importance_range),
            "top_10_drop": float(top_10_drop)
        }
    }


def recommend_clustering(embeddings: List[List[float]]) -> Dict[str, Any]:
    """
    Recommend clustering algorithm and parameters.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Dictionary with clustering recommendations
    """
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Get basic properties
    n_samples = len(embeddings_array)
    n_dimensions = embeddings_array.shape[1]
    
    # Calculate embedding statistics
    from sklearn.metrics.pairwise import euclidean_distances
    
    # Sample points for large datasets
    if n_samples > 1000:
        idx = np.random.choice(n_samples, 1000, replace=False)
        sample_array = embeddings_array[idx]
    else:
        sample_array = embeddings_array
    
    # Calculate distances
    distances = euclidean_distances(sample_array)
    np.fill_diagonal(distances, np.nan)
    
    mean_distance = float(np.nanmean(distances))
    std_distance = float(np.nanstd(distances))
    
    # Detect natural cluster count using silhouette analysis
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    best_score = -1
    best_n_clusters = 2
    
    # Try a range of cluster counts
    max_clusters = min(15, n_samples // 10)  # Cap at reasonable maximum
    
    for n_clusters in range(2, max_clusters + 1):
        # Skip if too few samples per cluster on average
        if n_samples / n_clusters < 5:
            continue
            
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(sample_array)
            
            score = silhouette_score(sample_array, labels)
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
        except:
            # Skip if clustering fails
            continue
    
    # Check for potential outliers
    distance_threshold = mean_distance + 2 * std_distance
    potential_outliers = np.sum(np.nanmean(distances > distance_threshold, axis=1))
    outlier_percentage = 100 * potential_outliers / len(sample_array)
    
    # Determine recommended algorithm
    if outlier_percentage > 15:
        # Many outliers - use density-based clustering
        algorithm = "DBSCAN"
        explanation = f"DBSCAN is recommended because approximately {outlier_percentage:.1f}% of points appear to be outliers."
        
        # Estimate DBSCAN parameters
        from sklearn.neighbors import NearestNeighbors
        
        # Find average distance to k nearest neighbors
        k = min(10, n_samples - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(sample_array)
        distances, _ = nn.kneighbors(sample_array)
        
        # Use average distance to kth neighbor as eps
        eps = float(np.mean(distances[:, -1]))
        min_samples = max(3, int(np.log(n_samples)))
        
        params = {
            "eps": eps,
            "min_samples": min_samples
        }
        
    elif n_samples > 1000 and outlier_percentage > 5:
        # Large dataset with some outliers - use HDBSCAN
        algorithm = "HDBSCAN"
        explanation = f"HDBSCAN is recommended for your large dataset with varying density regions and {outlier_percentage:.1f}% potential outliers."
        
        min_cluster_size = max(5, int(np.log(n_samples)))
        min_samples = None  # Let HDBSCAN determine automatically
        
        params = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples
        }
        
    elif n_dimensions > 50:
        # High-dimensional data - use Gaussian Mixture
        algorithm = "Gaussian Mixture"
        explanation = f"Gaussian Mixture is recommended for your high-dimensional data ({n_dimensions} dimensions), as it handles high-dimensional spaces well."
        
        params = {
            "n_components": best_n_clusters,
            "covariance_type": "full"
        }
        
    else:
        # Default - use K-Means
        algorithm = "K-Means"
        explanation = f"K-Means is recommended as a general-purpose clustering algorithm for your data."
        
        params = {
            "n_clusters": best_n_clusters,
            "random_state": 42
        }
    
    # Generate message
    message = f"Based on your data characteristics, I recommend using {algorithm} clustering"
    
    if algorithm == "K-Means" or algorithm == "Gaussian Mixture":
        message += f" with {best_n_clusters} clusters.\n\n"
    else:
        message += ".\n\n"
        
    message += explanation + "\n\n"
    
    if algorithm == "DBSCAN":
        message += f"Recommended parameters: eps={eps:.2f}, min_samples={min_samples}"
    elif algorithm == "HDBSCAN":
        message += f"Recommended parameters: min_cluster_size={min_cluster_size}"
    elif algorithm == "K-Means":
        message += f"Analysis suggests {best_n_clusters} natural clusters in your data (silhouette score: {best_score:.2f})."
    elif algorithm == "Gaussian Mixture":
        message += f"Analysis suggests {best_n_clusters} components for the Gaussian Mixture model."
    
    return {
        "algorithm": algorithm,
        "explanation": explanation,
        "message": message,
        "params": params,
        "n_clusters": best_n_clusters,
        "silhouette_score": float(best_score),
        "data_stats": {
            "n_samples": n_samples,
            "n_dimensions": n_dimensions,
            "mean_distance": mean_distance,
            "std_distance": std_distance,
            "outlier_percentage": float(outlier_percentage)
        }
    } 