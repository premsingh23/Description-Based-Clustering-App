from typing import List
from sklearn.cluster import KMeans


def kmeans_cluster(embeddings: List[List[float]], n_clusters: int = 5, random_state: int = 42) -> List[int]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(embeddings)
    return labels.tolist()
