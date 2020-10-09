from typing import List, Set

import networkx as nx
import numpy as np
import sklearn
import sklearn.cluster
from src.mcla.util import jaccard, jaccard_single
from src.util import label_to_comm
import community as community_louvain

def mcla(comm: List[Set[int]], num_clusters: int) -> List[Set[int]]:
    num_metanodes = len(comm)
    print(f"num metanodes: {num_metanodes}")
    meta_graph = nx.Graph()
    for meta_node in range(num_metanodes):
        meta_graph.add_node(meta_node)
    for meta_node_1 in range(num_metanodes):
        for meta_node_2 in range(meta_node_1+1, num_metanodes):
            weight = jaccard(comm[meta_node_1], comm[meta_node_2])
            if weight > 0.0:
                meta_graph.add_edge(meta_node_1, meta_node_2, weight=weight)


    parition = community_louvain.best_partition(meta_graph)
    label = np.empty((num_metanodes,))
    for meta_node in range(num_metanodes):
        label[meta_node] = parition[meta_node]

    metacomm = label_to_comm(label)

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