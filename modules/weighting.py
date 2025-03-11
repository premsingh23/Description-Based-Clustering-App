import numpy as np
from typing import List, Dict, Any, Optional


def get_weighting_schemes() -> Dict[str, str]:
    """
    Get available weighting schemes with descriptions.
    
    Returns:
        Dictionary mapping scheme names to descriptions
    """
    return {
        "linear": "Linear weighting based on importance rank",
        "exponential": "Exponential weighting with higher emphasis on top dimensions",
        "binary": "Binary weighting (include/exclude dimensions)",
        "logarithmic": "Logarithmic weighting for smoother importance scaling",
        "sigmoid": "Sigmoid weighting for smooth transition between important/unimportant",
        "statistical": "Statistical weighting based on p-values and effect sizes"
    }


def apply_weights(
    embeddings: List[List[float]],
    dimension_importance: List[Dict[str, Any]],
    scheme: str = "linear",
    top_n: int = 20,
    weight_factor: float = 2.0
) -> List[List[float]]:
    """
    Apply weighting to embeddings based on dimension importance.
    
    Args:
        embeddings: List of embedding vectors
        dimension_importance: List of dimension importance information
        scheme: Weighting scheme to apply
        top_n: Number of top dimensions to consider
        weight_factor: Factor to control weight intensity
        
    Returns:
        List of weighted embedding vectors
    """
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Get total dimensions
    total_dims = embeddings_array.shape[1]
    
    # Limit top_n to total dimensions
    top_n = min(top_n, total_dims)
    
    # Initialize weights (default: all 1.0 - no change)
    weights = np.ones(total_dims)
    
    # Get dimension indices sorted by importance
    important_dimensions = [d["dimension"] for d in dimension_importance[:top_n]]
    
    # Apply weighting based on selected scheme
    if scheme == "linear":
        for i, dim in enumerate(important_dimensions):
            # Linear decreasing weight (top dimension gets highest weight)
            rank_weight = 1.0 + weight_factor * (1.0 - (i / top_n))
            weights[dim] = rank_weight
            
    elif scheme == "exponential":
        for i, dim in enumerate(important_dimensions):
            # Exponential decreasing weight
            rank_weight = 1.0 + weight_factor * np.exp(-3.0 * (i / top_n))
            weights[dim] = rank_weight
            
    elif scheme == "binary":
        # Binary weighting: only consider top dimensions
        for i in range(total_dims):
            weights[i] = weight_factor if i in important_dimensions else 1.0
            
    elif scheme == "logarithmic":
        for i, dim in enumerate(important_dimensions):
            # Logarithmic decreasing weight
            rank_weight = 1.0 + weight_factor * (1.0 - np.log(i + 1) / np.log(top_n + 1))
            weights[dim] = rank_weight
            
    elif scheme == "sigmoid":
        middle_point = top_n / 2
        for i, dim in enumerate(important_dimensions):
            # Sigmoid weight transition
            x = (i - middle_point) / (middle_point / 3)  # Scale for reasonable sigmoid
            rank_weight = 1.0 + weight_factor / (1.0 + np.exp(x))
            weights[dim] = rank_weight
            
    elif scheme == "statistical":
        # Use actual statistical values from dimension analysis
        for dim_info in dimension_importance[:top_n]:
            dim = dim_info["dimension"]
            # Combine p-value and effect size
            p_importance = 1.0 - min(dim_info["p_value"], 0.9999)  # Convert p-value to importance
            stat_weight = 1.0 + weight_factor * p_importance * (1.0 + dim_info["effect_size"])
            weights[dim] = stat_weight
    
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    
    # Apply weights to embeddings
    weighted_embeddings = embeddings_array * weights
    
    # Convert back to list
    return weighted_embeddings.tolist()


def create_custom_weight_profile(
    dimension_importance: List[Dict[str, Any]],
    important_terms: List[str],
    reference_embeddings: Dict[str, List[float]],
    weight_factor: float = 2.0
) -> np.ndarray:
    """
    Create a custom weight profile based on important terms.
    
    Args:
        dimension_importance: List of dimension importance information
        important_terms: List of important terms to emphasize
        reference_embeddings: Dictionary mapping terms to embeddings
        weight_factor: Factor to control weight intensity
        
    Returns:
        Custom weight profile as numpy array
    """
    # Get total dimensions
    total_dims = len(dimension_importance)
    
    # Initialize weights (default: all 1.0 - no change)
    weights = np.ones(total_dims)
    
    # Check if reference embeddings are available for terms
    available_terms = [term for term in important_terms if term in reference_embeddings]
    
    if not available_terms:
        return weights
    
    # Get embeddings for important terms
    term_embeddings = np.array([reference_embeddings[term] for term in available_terms])
    
    # Find dimensions with high variance across terms
    term_variances = np.var(term_embeddings, axis=0)
    
    # Sort dimensions by variance
    variance_sorted_dims = np.argsort(term_variances)[::-1]  # Descending
    
    # Apply weights to top variance dimensions
    top_variance_dims = variance_sorted_dims[:min(50, total_dims)]
    
    for i, dim in enumerate(top_variance_dims):
        # Weights decrease with rank
        rank_factor = 1.0 - (i / len(top_variance_dims))
        weights[dim] = 1.0 + weight_factor * rank_factor * term_variances[dim]
    
    return weights


def get_dimension_correlations(
    embeddings: List[List[float]],
    dimension_importance: List[Dict[str, Any]],
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Calculate correlations between important dimensions.
    
    Args:
        embeddings: List of embedding vectors
        dimension_importance: List of dimension importance information
        top_n: Number of top dimensions to consider
        
    Returns:
        Dictionary with correlation information
    """
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Get top important dimensions
    top_dims = [d["dimension"] for d in dimension_importance[:top_n]]
    
    # Extract values for these dimensions
    top_dim_values = embeddings_array[:, top_dims]
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(top_dim_values, rowvar=False)
    
    # Create mappings between matrix indices and dimension numbers
    index_to_dim = {i: dim for i, dim in enumerate(top_dims)}
    
    # Find highly correlated pairs
    correlated_pairs = []
    
    for i in range(len(top_dims)):
        for j in range(i+1, len(top_dims)):
            corr = correlation_matrix[i, j]
            if abs(corr) > 0.5:  # Threshold for "high" correlation
                correlated_pairs.append({
                    "dim1": index_to_dim[i],
                    "dim2": index_to_dim[j],
                    "correlation": float(corr)
                })
    
    # Sort by absolute correlation (descending)
    correlated_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    return {
        "correlation_matrix": correlation_matrix.tolist(),
        "dimension_mapping": index_to_dim,
        "correlated_pairs": correlated_pairs
    } 