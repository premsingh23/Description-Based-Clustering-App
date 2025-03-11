import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter


def analyze_data_quality(df: pd.DataFrame, label_col: str, desc_col: str) -> Dict[str, Any]:
    """
    Analyze the quality of input data and provide feedback.
    
    Args:
        df: Input DataFrame
        label_col: Column containing labels
        desc_col: Column containing descriptions
        
    Returns:
        Dictionary with analysis results and suggestions
    """
    issues = []
    suggestions = []
    
    # Check basic data properties
    row_count = len(df)
    label_count = df[label_col].nunique()
    
    # Check for missing values
    missing_labels = df[label_col].isna().sum()
    missing_descs = df[desc_col].isna().sum()
    
    if missing_labels > 0:
        pct = (missing_labels / row_count) * 100
        issues.append(f"Missing values in label column: {missing_labels} rows ({pct:.1f}%)")
        
        suggestions.append({
            "id": "drop_missing_labels",
            "title": "Drop rows with missing labels",
            "apply_func": lambda df: df.dropna(subset=[label_col])
        })
    
    if missing_descs > 0:
        pct = (missing_descs / row_count) * 100
        issues.append(f"Missing values in description column: {missing_descs} rows ({pct:.1f}%)")
        
        if missing_descs / row_count < 0.1:  # Less than 10% missing
            suggestions.append({
                "id": "drop_missing_descs",
                "title": "Drop rows with missing descriptions",
                "apply_func": lambda df: df.dropna(subset=[desc_col])
            })
        else:
            suggestions.append({
                "id": "fill_missing_descs",
                "title": "Fill missing descriptions with placeholder",
                "apply_func": lambda df: df.fillna({desc_col: "No description provided"})
            })
    
    # Check for empty descriptions
    empty_descs = (df[desc_col] == "").sum()
    if empty_descs > 0:
        pct = (empty_descs / row_count) * 100
        issues.append(f"Empty descriptions: {empty_descs} rows ({pct:.1f}%)")
        
        suggestions.append({
            "id": "drop_empty_descs",
            "title": "Drop rows with empty descriptions",
            "apply_func": lambda df: df[df[desc_col] != ""]
        })
    
    # Check description length
    desc_lens = df[desc_col].astype(str).str.len()
    short_descs = (desc_lens < 10).sum()
    
    if short_descs > 0:
        pct = (short_descs / row_count) * 100
        issues.append(f"Very short descriptions (<10 chars): {short_descs} rows ({pct:.1f}%)")
        
        if pct > 5:
            suggestions.append({
                "id": "filter_short_descs",
                "title": "Filter out rows with very short descriptions",
                "apply_func": lambda df: df[df[desc_col].astype(str).str.len() >= 10]
            })
    
    # Check label distribution
    label_counts = df[label_col].value_counts()
    single_label_count = (label_counts == 1).sum()
    
    if single_label_count > 0:
        pct = (single_label_count / label_count) * 100
        issues.append(f"Labels with only one instance: {single_label_count} ({pct:.1f}% of all labels)")
        
        if pct > 20:
            suggestions.append({
                "id": "filter_rare_labels",
                "title": "Filter out labels with only one instance",
                "apply_func": lambda df: df.groupby(label_col).filter(lambda x: len(x) > 1)
            })
    
    # Generate message
    if not issues:
        message = f"Your data looks good! {row_count} rows with {label_count} unique labels."
    else:
        message = f"Data analysis found {len(issues)} potential issues:\n\n" + "\n".join(f"• {issue}" for issue in issues)
        
        if suggestions:
            message += "\n\nConsider addressing these issues before proceeding."
    
    return {
        "message": message,
        "issues": issues,
        "has_suggestions": len(suggestions) > 0,
        "suggestions": suggestions,
        "row_count": row_count,
        "label_count": label_count
    }


def analyze_embeddings(embeddings: List[List[float]], model_info: Dict[str, Any], labels: List[str]) -> Dict[str, Any]:
    """
    Analyze generated embeddings.
    
    Args:
        embeddings: List of embedding vectors
        model_info: Information about the embedding model
        labels: List of corresponding labels
        
    Returns:
        Dictionary with analysis results
    """
    embeddings_array = np.array(embeddings)
    
    # Basic stats
    dimensions = len(embeddings[0]) if embeddings else 0
    sample_count = len(embeddings)
    
    # Calculate embedding norms
    norms = np.linalg.norm(embeddings_array, axis=1)
    avg_norm = float(np.mean(norms))
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))
    
    # Check for zero vectors
    zero_vectors = np.all(embeddings_array == 0, axis=1).sum()
    
    # Calculate variance across dimensions
    dimension_variances = np.var(embeddings_array, axis=0)
    avg_variance = float(np.mean(dimension_variances))
    min_variance = float(np.min(dimension_variances))
    max_variance = float(np.max(dimension_variances))
    
    # Get top dimensions by variance
    top_dims = np.argsort(dimension_variances)[::-1][:10].tolist()
    
    # Calculate average distance between vectors
    # (for large sets, sample a subset)
    if sample_count > 1000:
        idx = np.random.choice(sample_count, 1000, replace=False)
        sampled_embeddings = embeddings_array[idx]
    else:
        sampled_embeddings = embeddings_array
        
    from sklearn.metrics.pairwise import cosine_distances
    distances = cosine_distances(sampled_embeddings)
    np.fill_diagonal(distances, np.nan)  # Ignore self-distances
    avg_distance = float(np.nanmean(distances))
    
    # Generate explanation
    if zero_vectors > 0:
        quality = "poor"
        explanation = f"The embeddings may have issues. {zero_vectors} vectors are zero vectors, indicating potential problems with text processing or the embedding model."
    elif avg_variance < 0.001:
        quality = "concerning"
        explanation = "The embeddings show very low variance across dimensions, which might indicate issues with the embedding model or insufficient diversity in the text data."
    elif avg_distance < 0.1:
        quality = "low diversity"
        explanation = "The embeddings are very similar to each other, which might make clustering difficult. Consider using more diverse text descriptions."
    elif avg_distance > 0.8:
        quality = "high diversity"
        explanation = "The embeddings show high diversity, which is generally good for clustering. The text descriptions appear to be semantically distinct."
    else:
        quality = "good"
        explanation = "The embeddings look good with reasonable variance and semantic distribution. They should work well for clustering."
    
    explanation += f"\n\nThe embeddings have {dimensions} dimensions with average vector norm of {avg_norm:.3f}."
    
    return {
        "dimensions": dimensions,
        "sample_count": sample_count,
        "avg_norm": avg_norm,
        "min_norm": min_norm,
        "max_norm": max_norm,
        "zero_vectors": int(zero_vectors),
        "avg_variance": avg_variance,
        "min_variance": min_variance,
        "max_variance": max_variance,
        "top_variance_dimensions": top_dims,
        "avg_distance": avg_distance,
        "quality": quality,
        "explanation": explanation,
        "model_info": model_info
    }


def analyze_clusters(
    df_with_clusters: pd.DataFrame,
    embeddings: List[List[float]],
    clusters: List[int],
    dimension_importance: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze clustering results.
    
    Args:
        df_with_clusters: DataFrame with cluster assignments
        embeddings: List of embedding vectors
        clusters: List of cluster labels
        dimension_importance: List of dimension importance information
        
    Returns:
        Dictionary with analysis results
    """
    # Basic stats
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    n_clusters = len(cluster_counts)
    min_cluster_size = int(cluster_counts.min())
    max_cluster_size = int(cluster_counts.max())
    size_ratio = max_cluster_size / min_cluster_size if min_cluster_size > 0 else float('inf')
    
    # Check for outliers/noise
    noise_count = list(clusters).count(-1) if -1 in clusters else 0
    noise_pct = (noise_count / len(clusters)) * 100 if clusters else 0
    
    # Calculate cluster cohesion (within-cluster distances)
    embeddings_array = np.array(embeddings)
    clusters_array = np.array(clusters)
    
    from sklearn.metrics.pairwise import cosine_distances
    
    cohesion_scores = {}
    for cluster_id in set(clusters):
        if cluster_id == -1:  # Skip noise cluster
            continue
            
        cluster_mask = (clusters_array == cluster_id)
        if sum(cluster_mask) <= 1:  # Skip single-item clusters
            cohesion_scores[cluster_id] = 0
            continue
            
        cluster_embeddings = embeddings_array[cluster_mask]
        distances = cosine_distances(cluster_embeddings)
        np.fill_diagonal(distances, np.nan)
        cohesion_scores[cluster_id] = float(np.nanmean(distances))
    
    avg_cohesion = np.mean(list(cohesion_scores.values())) if cohesion_scores else 0
    
    # Calculate cluster separation (between-cluster distances)
    separation_scores = {}
    cluster_centers = {}
    
    for cluster_id in set(clusters):
        if cluster_id == -1:  # Skip noise cluster
            continue
            
        cluster_mask = (clusters_array == cluster_id)
        cluster_embeddings = embeddings_array[cluster_mask]
        cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
    
    for cluster_id1 in cluster_centers:
        for cluster_id2 in cluster_centers:
            if cluster_id1 >= cluster_id2:
                continue
                
            center1 = cluster_centers[cluster_id1]
            center2 = cluster_centers[cluster_id2]
            distance = cosine_distances([center1], [center2])[0][0]
            separation_scores[(cluster_id1, cluster_id2)] = float(distance)
    
    avg_separation = np.mean(list(separation_scores.values())) if separation_scores else 0
    
    # Quality assessment
    quality_score = 0
    quality_issues = []
    
    # Check cluster count
    if n_clusters == 1:
        quality_issues.append("Only one cluster was found. Consider different clustering parameters.")
        quality_score += 10
    elif n_clusters == 2:
        quality_score += 50
    elif 3 <= n_clusters <= 10:
        quality_score += 80
    else:
        quality_issues.append(f"Large number of clusters ({n_clusters}). Consider simplifying with fewer clusters.")
        quality_score += 40
    
    # Check cluster sizes
    if size_ratio > 10:
        quality_issues.append(f"Very imbalanced cluster sizes (largest is {size_ratio:.1f}x the smallest).")
        quality_score = max(quality_score - 20, 10)
    elif size_ratio > 5:
        quality_issues.append(f"Somewhat imbalanced cluster sizes (ratio: {size_ratio:.1f}x).")
        quality_score = max(quality_score - 10, 10)
    else:
        quality_score += 10
    
    # Check noise percentage
    if noise_pct > 50:
        quality_issues.append(f"Very high percentage of noise points ({noise_pct:.1f}%).")
        quality_score = max(quality_score - 30, 10)
    elif noise_pct > 20:
        quality_issues.append(f"High percentage of noise points ({noise_pct:.1f}%).")
        quality_score = max(quality_score - 15, 10)
    elif noise_pct > 0:
        quality_score += 5
    else:
        quality_score += 10
    
    # Check cohesion and separation
    if avg_cohesion < 0.3 and avg_separation > 0.7:
        quality_score += 20
    elif avg_cohesion > 0.7 and avg_separation < 0.3:
        quality_issues.append("Clusters are not well-separated and have low internal cohesion.")
        quality_score = max(quality_score - 20, 10)
    
    # Cap quality score
    quality_score = min(quality_score, 100)
    
    # Generate quality assessment text
    if quality_score >= 80:
        quality_assessment = "Excellent clustering! The clusters are well-defined and separated."
    elif quality_score >= 60:
        quality_assessment = "Good clustering with reasonable separation between clusters."
    elif quality_score >= 40:
        quality_assessment = "Fair clustering, but some issues might affect interpretation."
    else:
        quality_assessment = "Poor clustering. Consider trying different parameters or preprocessing steps."
    
    # Generate recommendations
    recommendations = []
    if quality_issues:
        recommendations.extend(quality_issues)
        
    if avg_cohesion > 0.5:
        recommendations.append("Try increasing the weight on top dimensions to improve cluster separation.")
    
    if size_ratio > 8:
        recommendations.append("Consider using a different clustering algorithm like HDBSCAN to handle varying cluster densities.")
    
    # Generate message
    message = f"Analysis found {n_clusters} clusters with sizes ranging from {min_cluster_size} to {max_cluster_size} items.\n\n"
    
    if quality_issues:
        message += "Issues detected:\n"
        for issue in quality_issues:
            message += f"• {issue}\n"
        message += "\n"
    
    message += f"Cluster quality: {quality_assessment}\n\n"
    
    if recommendations:
        message += "Recommendations:\n"
        for rec in recommendations:
            message += f"• {rec}\n"
    
    return {
        "n_clusters": n_clusters,
        "cluster_counts": cluster_counts.to_dict(),
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "size_ratio": size_ratio,
        "noise_count": noise_count,
        "noise_pct": noise_pct,
        "cohesion_scores": cohesion_scores,
        "avg_cohesion": float(avg_cohesion),
        "separation_scores": {f"{k[0]}-{k[1]}": v for k, v in separation_scores.items()},
        "avg_separation": float(avg_separation),
        "quality_score": quality_score,
        "quality_issues": quality_issues,
        "quality_assessment": quality_assessment,
        "recommendations": recommendations,
        "message": message
    } 