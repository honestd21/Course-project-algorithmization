import numpy as np
from sklearn.cluster import KMeans


def cluster_targets(cells, weed, k=3, seed=0):
    """Группирует цели в очаги по координатам. Возвращает метки и центры."""
    pts = np.array(cells)
    km = KMeans(n_clusters=min(k, len(pts)), n_init=10, random_state=seed).fit(pts)
    return km.labels_, km.cluster_centers_