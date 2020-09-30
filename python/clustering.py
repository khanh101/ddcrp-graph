from typing import Union, List, Set

import numpy as np
from sklearn.cluster import KMeans

def label_to_comm(label_list: Union[List[int], np.ndarray]) -> List[Set[int]]:
    min_label = min(label_list)
    max_label = max(label_list)
    communities = [[] for _ in range(min_label, max_label + 1)]
    for idx, label in enumerate(label_list):
        communities[label].append(idx)
    communities = [set(comm) for comm in communities]
    return communities

def spectral_clustering(A: np.ndarray, dim: int, num_clusters: int) -> List[Set[int]]:
    A = A * (1.0 - np.identity(A.shape[0]))
    D = np.diag(A.sum(axis=0))
    L = D - A
    v, w = np.linalg.eigh(L)
    w = w[:, range(1, dim+1)]

    clustering = KMeans(n_clusters=num_clusters).fit(w)
    pred = clustering.labels_
    comm = label_to_comm(pred)
    return comm

if __name__ == "__main__":
    A = np.random.random(size=(100, 100))
    print(spectral_clustering(A, 50, 5))