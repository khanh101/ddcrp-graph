import community as community_louvain
import networkx as nx
import numpy as np
from typing import List, Set, Tuple

from src.mcla.util import jaccard, jaccard_single
from src.util import label_to_comm


def mcla(next_comm: List[Set[int]], ref: List[Set[int]] = None) -> Tuple[List[Set[int]], List[Set[int]]]:
    if ref is None:
        ref = []
    comm: List[Set[int]] = [*ref, *next_comm]
    num_metanodes = len(comm)
    print(f"num metanodes: {num_metanodes}")
    # create nx graph
    meta_graph = nx.Graph()
    for meta_node in range(num_metanodes):
        meta_graph.add_node(meta_node)
    for meta_node_1 in range(num_metanodes):
        for meta_node_2 in range(meta_node_1 + 1, num_metanodes):
            weight: float = jaccard(comm[meta_node_1], comm[meta_node_2])
            if weight > 0.0:
                meta_graph.add_edge(meta_node_1, meta_node_2, weight=weight)

    # run louvain
    parition = community_louvain.best_partition(meta_graph)
    label_list = np.empty((num_metanodes,), dtype=np.int)
    for meta_node in range(num_metanodes):
        label_list[meta_node] = parition[meta_node]

    metacomm: List[Set[int]] = label_to_comm(label_list)

    # comm voting
    metacomm_nodes: List[List[int]] = []
    for metac in metacomm:
        nodes = []
        for c in metac:
            nodes.extend(list(comm[c]))
        metacomm_nodes.append(nodes)

    nodes = []
    for c in comm:
        nodes.extend(list(c))
    num_nodes = len(set(nodes))
    label_list = np.empty((num_nodes,), dtype=np.int)
    for node in range(num_nodes):
        weight: List[float] = []
        for metacomm_node in metacomm_nodes:
            weight.append(jaccard_single(node, metacomm_node))
        label = weight.index(max(weight))
        label_list[node] = label
    new_comm = label_to_comm(label_list)
    # return mapping between new clusters and old clusters
    mapping: List[Set[int]] = []
    if len(ref) > 0:  # meta comm label to ref label
        for metac in metacomm:
            mapping.append(set([c for c in metac if c < len(ref)]))
    return new_comm, mapping
