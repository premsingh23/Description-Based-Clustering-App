import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Union
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_clusters(
    embeddings: List[List[float]],
    clusters: List[int],
    labels: Optional[List[str]] = None,
    method: str = "PCA",
    dimensions: int = 2,
    max_points: int = 3000,
    custom_colors: Optional[Dict[int, str]] = None
) -> go.Figure:
    """
    Visualize clusters in 2D or 3D using dimension reduction techniques.
    
    Args:
        embeddings: List of embedding vectors
        clusters: List of cluster labels for each embedding
        labels: Optional list of original data labels for hover information
        method: Dimension reduction method ('PCA', 't-SNE', or 'UMAP')
        dimensions: Number of dimensions for visualization (2 or 3)
        max_points: Maximum number of points to visualize (for performance)
        custom_colors: Optional custom color mapping for clusters
        
    Returns:
        Plotly figure object
    """
    # Convert to numpy arrays
    embeddings_array = np.array(embeddings)
    clusters_array = np.array(clusters)
    
    # Sample points if too many (for performance)
    if len(embeddings_array) > max_points:
        indices = np.random.choice(len(embeddings_array), max_points, replace=False)
        embeddings_array = embeddings_array[indices]
        clusters_array = clusters_array[indices]
        if labels is not None:
            labels = [labels[i] for i in indices]
    
    # Reduce dimensions
    if method.lower() == "pca":
        reducer = PCA(n_components=dimensions, random_state=42)
        reduced_data = reducer.fit_transform(embeddings_array)
        
    elif method.lower() == "t-sne" or method.lower() == "tsne":
        reducer = TSNE(n_components=dimensions, random_state=42)
        reduced_data = reducer.fit_transform(embeddings_array)
        
    elif method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=dimensions, random_state=42)
            reduced_data = reducer.fit_transform(embeddings_array)
        except ImportError:
            raise ImportError("UMAP is not installed. Install with 'pip install umap-learn'")
            
    else:
        raise ValueError(f"Unknown dimension reduction method: {method}")
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame()
    
    if dimensions == 2:
        plot_df["x"] = reduced_data[:, 0]
        plot_df["y"] = reduced_data[:, 1]
        
        # Create plotly figure
        if labels is not None:
            plot_df["label"] = labels
            fig = px.scatter(
                plot_df, x="x", y="y", color=clusters_array.astype(str),
                hover_data=["label"], title=f"Cluster Visualization ({method} 2D)"
            )
        else:
            fig = px.scatter(
                plot_df, x="x", y="y", color=clusters_array.astype(str),
                title=f"Cluster Visualization ({method} 2D)"
            )
            
    elif dimensions == 3:
        plot_df["x"] = reduced_data[:, 0]
        plot_df["y"] = reduced_data[:, 1]
        plot_df["z"] = reduced_data[:, 2]
        
        # Create plotly figure
        if labels is not None:
            plot_df["label"] = labels
            fig = px.scatter_3d(
                plot_df, x="x", y="y", z="z", color=clusters_array.astype(str),
                hover_data=["label"], title=f"Cluster Visualization ({method} 3D)"
            )
        else:
            fig = px.scatter_3d(
                plot_df, x="x", y="y", z="z", color=clusters_array.astype(str),
                title=f"Cluster Visualization ({method} 3D)"
            )
    
    else:
        raise ValueError("Dimensions must be 2 or 3")
    
    # Customize layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend_title_text="Cluster",
        template="plotly_white"
    )
    
    # Apply custom colors if provided
    if custom_colors is not None:
        for i, color in custom_colors.items():
            fig.update_traces(
                selector=dict(name=str(i)),
                marker=dict(color=color)
            )
    
    return fig


def visualize_dimensions(
    dimension_importance: List[Dict[str, Any]],
    top_n: int = 20,
    include_stats: bool = True
) -> go.Figure:
    """
    Visualize dimension importance.
    
    Args:
        dimension_importance: List of dimension importance information
        top_n: Number of top dimensions to visualize
        include_stats: Whether to include statistical metrics
        
    Returns:
        Plotly figure object
    """
    # Limit to top n dimensions
    top_dimensions = dimension_importance[:min(top_n, len(dimension_importance))]
    
    # Create DataFrame for plotting
    df = pd.DataFrame(top_dimensions)
    
    # Create basic bar chart of importance
    fig = px.bar(
        df, 
        x="dimension", 
        y="importance",
        title=f"Top {top_n} Important Dimensions",
        labels={"dimension": "Dimension", "importance": "Importance Score"}
    )
    
    # Add additional statistical metrics if requested
    if include_stats and "p_value" in df.columns and "effect_size" in df.columns:
        # Add p-value as hover information
        hover_text = []
        for _, row in df.iterrows():
            text = f"Dimension: {row['dimension']}<br>"
            text += f"Importance: {row['importance']:.4f}<br>"
            text += f"p-value: {row['p_value']:.4f}<br>"
            text += f"Effect size: {row['effect_size']:.4f}<br>"
            text += f"Mean diff: {row['mean_difference']:.4f}"
            hover_text.append(text)
        
        fig.update_traces(hovertext=hover_text, hoverinfo="text")
        
        # Add effect size as marker size
        fig.update_traces(
            marker=dict(
                size=df["effect_size"] * 50,  # Scale effect size for visibility
                line=dict(width=1, color="DarkSlateGrey")
            )
        )
    
    # Improve layout
    fig.update_layout(
        xaxis_title="Dimension",
        yaxis_title="Importance Score",
        template="plotly_white",
        xaxis=dict(tickmode="linear")
    )
    
    return fig


def create_cluster_report(
    cluster_data: pd.DataFrame,
    label_column: str,
    description_column: str,
    max_terms: int = 20,
    max_examples: int = 5
) -> Dict[str, Any]:
    """
    Create a detailed report about a cluster.
    
    Args:
        cluster_data: DataFrame containing data for a specific cluster
        label_column: Name of the column containing labels
        description_column: Name of the column containing descriptions
        max_terms: Maximum number of common terms to include
        max_examples: Maximum number of examples to include
        
    Returns:
        Dictionary with cluster information
    """
    # Count label distribution
    label_counts = cluster_data[label_column].value_counts().head(max_terms)
    top_labels = pd.DataFrame({
        "label": label_counts.index,
        "count": label_counts.values,
        "percentage": (label_counts.values / len(cluster_data) * 100).round(2)
    })
    
    # Extract common terms from descriptions
    all_text = " ".join(cluster_data[description_column].astype(str))
    
    # Simple tokenization and counting
    words = all_text.lower().split()
    
    # Remove very short words and punctuation
    words = [word.strip(".,!?()[]{}\"'") for word in words if len(word) > 2]
    
    # Count words
    word_counts = Counter(words).most_common(max_terms)
    common_terms = pd.DataFrame({
        "term": [word for word, count in word_counts],
        "count": [count for word, count in word_counts],
        "percentage": [(count / len(words) * 100).round(2) for word, count in word_counts]
    })
    
    # Sample examples from cluster
    examples = cluster_data.sample(min(max_examples, len(cluster_data)))
    
    # Compile report
    report = {
        "cluster_size": len(cluster_data),
        "top_labels": top_labels,
        "common_terms": common_terms,
        "examples": examples[[label_column, description_column]].to_dict('records')
    }
    
    return report


def plot_cluster_comparison(
    df: pd.DataFrame,
    cluster_column: str,
    metric_columns: List[str],
    title: str = "Cluster Comparison"
) -> go.Figure:
    """
    Create a radar plot comparing clusters based on various metrics.
    
    Args:
        df: DataFrame containing cluster data
        cluster_column: Name of the column containing cluster labels
        metric_columns: List of columns to use as metrics for comparison
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Group by cluster and calculate mean of metrics
    cluster_means = df.groupby(cluster_column)[metric_columns].mean()
    
    # Normalize data for radar chart
    normalized_means = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    # Create radar chart
    fig = go.Figure()
    
    for cluster in normalized_means.index:
        fig.add_trace(go.Scatterpolar(
            r=normalized_means.loc[cluster].values,
            theta=metric_columns,
            fill='toself',
            name=f'Cluster {cluster}'
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=title,
        showlegend=True
    )
    
    return fig


def create_dimension_heatmap(
    embeddings: List[List[float]],
    clusters: List[int],
    dimension_importance: List[Dict[str, Any]],
    top_n: int = 10
) -> go.Figure:
    """
    Create a heatmap showing how top dimensions vary across clusters.
    
    Args:
        embeddings: List of embedding vectors
        clusters: List of cluster labels for each embedding
        dimension_importance: List of dimension importance information
        top_n: Number of top dimensions to include
        
    Returns:
        Plotly figure object
    """
    # Convert to numpy arrays
    embeddings_array = np.array(embeddings)
    clusters_array = np.array(clusters)
    
    # Get top dimensions
    top_dims = [d["dimension"] for d in dimension_importance[:top_n]]
    
    # Calculate mean values of top dimensions for each cluster
    unique_clusters = sorted(set(clusters))
    
    cluster_dim_means = np.zeros((len(unique_clusters), len(top_dims)))
    
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = (clusters_array == cluster)
        cluster_embeddings = embeddings_array[cluster_mask]
        
        # Calculate mean for each dimension
        for j, dim in enumerate(top_dims):
            cluster_dim_means[i, j] = cluster_embeddings[:, dim].mean()
    
    # Normalize for better visualization
    cluster_dim_means = (cluster_dim_means - cluster_dim_means.min(axis=0)) / \
                      (cluster_dim_means.max(axis=0) - cluster_dim_means.min(axis=0) + 1e-10)
    
    # Create heatmap
    fig = px.imshow(
        cluster_dim_means,
        labels=dict(x="Dimension", y="Cluster", color="Normalized Value"),
        x=[f"Dim {dim}" for dim in top_dims],
        y=[f"Cluster {cluster}" for cluster in unique_clusters],
        title=f"Top {top_n} Dimensions Across Clusters",
        color_continuous_scale="Viridis"
    )
    
    # Customize layout
    fig.update_layout(
        xaxis=dict(tickangle=45),
        coloraxis_colorbar=dict(title="Value")
    )
    
    return fig


def plot_cluster_sizes(clusters: List[int]) -> go.Figure:
    """
    Create a bar chart showing the size of each cluster.
    
    Args:
        clusters: List of cluster labels
        
    Returns:
        Plotly figure object
    """
    # Count cluster sizes
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    
    # Create bar chart
    fig = px.bar(
        x=cluster_counts.index.astype(str),
        y=cluster_counts.values,
        title="Cluster Sizes",
        labels={"x": "Cluster", "y": "Number of Items"}
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Number of Items",
        template="plotly_white"
    )
    
    return fig 