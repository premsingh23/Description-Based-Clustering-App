import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple


def ensure_directory(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to check/create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_model_state(state: Dict[str, Any], name: str, directory: str = "saved_states") -> str:
    """
    Save the model state to disk.
    
    Args:
        state: Dictionary containing state to save
        name: Name for the saved state
        directory: Directory to save state in
        
    Returns:
        Path where state was saved
    """
    # Ensure directory exists
    ensure_directory(directory)
    
    # Create timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{name}_{timestamp}.pkl"
    filepath = os.path.join(directory, filename)
    
    # Save state
    with open(filepath, "wb") as f:
        pickle.dump(state, f)
    
    return filepath


def load_model_state(filepath: str) -> Dict[str, Any]:
    """
    Load model state from disk.
    
    Args:
        filepath: Path to the saved state file
        
    Returns:
        Loaded state dictionary
    """
    with open(filepath, "rb") as f:
        state = pickle.load(f)
    
    return state


def get_saved_states(directory: str = "saved_states") -> List[Dict[str, Any]]:
    """
    Get information about all saved states.
    
    Args:
        directory: Directory containing saved states
        
    Returns:
        List of dictionaries with state information
    """
    # Check if directory exists
    if not os.path.exists(directory):
        return []
    
    # Get all pickle files
    files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    
    # Get information for each file
    states = []
    for file in files:
        filepath = os.path.join(directory, file)
        
        # Get file metadata
        file_stats = os.stat(filepath)
        created = file_stats.st_ctime
        size = file_stats.st_size
        
        # Parse name and timestamp from filename
        name_parts = file.replace(".pkl", "").split("_")
        if len(name_parts) >= 3:
            # Format: name_YYYYMMDD_HHMMSS.pkl
            timestamp_parts = name_parts[-2:]
            name = "_".join(name_parts[:-2])
            timestamp = "_".join(timestamp_parts)
        else:
            name = name_parts[0]
            timestamp = ""
        
        states.append({
            "filename": file,
            "filepath": filepath,
            "name": name,
            "timestamp": timestamp,
            "created": created,
            "size": size,
        })
    
    # Sort by creation time (newest first)
    states.sort(key=lambda x: x["created"], reverse=True)
    
    return states


def convert_to_streamlit_format(figure: Any) -> Any:
    """
    Convert various figure types to formats compatible with Streamlit.
    
    Args:
        figure: Figure object to convert
        
    Returns:
        Converted figure compatible with Streamlit
    """
    # Check if it's already a plotly figure
    if 'plotly' in str(type(figure)):
        return figure
    
    # Check if it's a matplotlib figure
    if 'matplotlib' in str(type(figure)):
        return figure  # Streamlit can directly handle matplotlib figures
    
    # For other types, just return as is and let Streamlit handle
    return figure


def detect_file_type(file) -> str:
    """
    Detect file type from a file object.
    
    Args:
        file: File object to check
        
    Returns:
        String describing the file type
    """
    # Get file name
    filename = file.name.lower()
    
    if filename.endswith('.csv'):
        return 'csv'
    elif filename.endswith(('.xls', '.xlsx')):
        return 'excel'
    elif filename.endswith('.json'):
        return 'json'
    elif filename.endswith('.txt'):
        return 'text'
    elif filename.endswith('.pkl'):
        return 'pickle'
    else:
        return 'unknown'


def find_optimal_clusters(
    data: np.ndarray,
    max_clusters: int = 10,
    method: str = "silhouette"
) -> Tuple[int, Dict[str, List[float]]]:
    """
    Find the optimal number of clusters using various metrics.
    
    Args:
        data: Data array to cluster
        max_clusters: Maximum number of clusters to check
        method: Primary method to use ('silhouette', 'elbow', or 'gap')
        
    Returns:
        Tuple containing:
        - Optimal number of clusters
        - Dictionary with metrics for each cluster count
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Initialize metrics
    metrics = {
        "silhouette": [],
        "inertia": [],
        "calinski_harabasz": [],
        "davies_bouldin": []
    }
    
    # Try different cluster counts
    for n_clusters in range(2, max_clusters + 1):
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Calculate metrics
        metrics["inertia"].append(kmeans.inertia_)
        
        try:
            silhouette = silhouette_score(data, labels)
            metrics["silhouette"].append(silhouette)
        except:
            metrics["silhouette"].append(0)
            
        try:
            ch_score = calinski_harabasz_score(data, labels)
            metrics["calinski_harabasz"].append(ch_score)
        except:
            metrics["calinski_harabasz"].append(0)
            
        try:
            db_score = davies_bouldin_score(data, labels)
            metrics["davies_bouldin"].append(db_score)
        except:
            metrics["davies_bouldin"].append(0)
    
    # Determine optimal number of clusters based on method
    if method == "silhouette":
        # Higher silhouette is better
        optimal_clusters = np.argmax(metrics["silhouette"]) + 2  # +2 because we start at 2 clusters
    elif method == "elbow":
        # Use elbow method on inertia
        from scipy.signal import argrelextrema
        from kneed import KneeLocator
        try:
            kl = KneeLocator(
                range(2, max_clusters + 1), 
                metrics["inertia"], 
                curve="convex", 
                direction="decreasing"
            )
            optimal_clusters = kl.elbow
            if optimal_clusters is None:
                optimal_clusters = 2
        except:
            # Find point of maximum curvature
            inertia_curve = np.array(metrics["inertia"])
            curvature = np.gradient(np.gradient(inertia_curve))
            optimal_clusters = np.argmax(curvature) + 2
    else:  # Default to calinski_harabasz
        # Higher calinski_harabasz is better
        optimal_clusters = np.argmax(metrics["calinski_harabasz"]) + 2
    
    return optimal_clusters, metrics


def generate_color_palette(n_colors: int) -> List[str]:
    """
    Generate a pleasing color palette with n colors.
    
    Args:
        n_colors: Number of colors to generate
        
    Returns:
        List of hex color strings
    """
    import colorsys
    
    # Use HSV color space for even distribution
    hsv_tuples = [(i / n_colors, 0.8, 0.9) for i in range(n_colors)]
    rgb_tuples = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_tuples]
    
    # Convert to hex format
    hex_colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in rgb_tuples]
    
    return hex_colors 