import numpy as np
import sklearn
import sklearn.cluster
from typing import List, Set, Union

from src.util import label_to_comm


def kmeans_improve(embedding: np.ndarray, init_comm: Union[int, List[Set[int]]]):
    num_clusters: int
    init: Union[str, np.ndarray]
    if isinstance(init_comm, int):
        num_clusters = init_comm
        init = "k-means++"
    else:
        num_clusters = len(init_comm)
        cluster_emb = []
        for cluster in init_comm:
            emb = np.zeros(embedding.shape[1])
            for node in cluster:
                emb += embedding[node]
            emb /= len(cluster)
            cluster_emb.append(emb)
        init = np.array(cluster_emb)

    clustering = sklearn.cluster.KMeans(n_clusters=num_clusters, init=init).fit(embedding)
    pred = clustering.labels_
    comm = label_to_comm(pred)
    return comm
