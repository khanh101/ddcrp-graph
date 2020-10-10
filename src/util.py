from typing import List, Set

import networkx as nx
import numpy as np

def comm_to_label(comm: List[Set[int]]) -> np.ndarray:
    num_nodes = sum([len(c) for c in comm])
    label_list = np.empty(shape=(num_nodes,), dtype=np.int)
    for label, c in enumerate(comm):
        for node in c:
            label_list[node] = label
    return label_list


def label_to_comm(label_list: np.ndarray) -> List[Set[int]]:
    communities = {}
    for node, label in enumerate(label_list):
        if label not in communities:
            communities[label] = set()
        communities[label].add(node)
    return list(communities.values())

def subgraph_by_timestamp(mg: nx.MultiGraph, start: int, end: int) -> nx.Graph:
    edges = filter(
        lambda edge: start <= edge[2]["timestamp"] and edge[2]["timestamp"] < end,
        mg.edges(data=True),
    )
    g = nx.Graph()
    for u, v, data in edges:
        if g.has_edge(u, v):
            g[u][v]["weight"] += data["weight"]
        else:
            g.add_edge(u, v, weight=data["weight"])
    return g
