import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from typing import List


def cluster_scatter(embeddings: List[List[float]], labels: List[int], df: pd.DataFrame, label_col: str):
    arr = np.array(embeddings)
    reduced = PCA(n_components=2).fit_transform(arr)
    plot_df = pd.DataFrame({"x": reduced[:,0], "y": reduced[:,1], "cluster": labels, "label": df[label_col]})
    fig = px.scatter(plot_df, x="x", y="y", color=plot_df["cluster"].astype(str), hover_data=["label"], title="Cluster Visualization")
    return fig


def importance_bar(importance: List[dict]):
    df = pd.DataFrame(importance)
    fig = px.bar(df.head(10), x="dimension", y="variance", title="Top Dimensions")
    return fig
