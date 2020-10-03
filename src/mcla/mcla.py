from typing import List, Set
import numpy as np
import sklearn
import sklearn.cluster
from src.mcla.util import jaccard, jaccard_single
from src.util import label_to_comm


def mcla(comm: List[Set[int]], num_clusters: int) -> List[Set[int]]:
    num_metanodes = len(comm)
    a = np.zeros((num_metanodes, num_metanodes), dtype=np.float64)
    for i in range(num_metanodes):
        for j in range(i+1, num_metanodes):
            a[i][j] = jaccard(comm[i], comm[j])
            a[j][i] = a[i][j]
    #d = np.diag(a.sum(axis=0))
    #l = d - a
    metacomm = label_to_comm(sklearn.cluster.SpectralClustering(
        n_clusters= num_clusters,
        assign_labels="kmeans",
        affinity="precomputed",
    ).fit_predict(a))

    metacomm_nodes = []
    for metac in metacomm:
        nodes = []
        for c in metac:
            nodes.extend(list(comm[c]))
        metacomm_nodes.append(nodes)


    nodes = []
    for c in comm:
        nodes.extend(list(c))
    num_nodes = len(set(nodes))
    comm_out = {}
    for node in range(num_nodes):
        weight = []
        for metacomm_node in metacomm_nodes:
            weight.append(jaccard_single(node, metacomm_node))
        label = weight.index(max(weight))
        if label not in comm_out:
            comm_out[label] = set()
        comm_out[label].add(node)
    return list(comm_out.values())


    pass