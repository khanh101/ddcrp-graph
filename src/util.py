from typing import List, Set, Tuple

import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import sklearn
import sklearn.linear_model


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
    return [v for _, v in sorted(communities.items(), key= lambda item: item[0])] # sort values by keys then return

def subgraph_by_timestamp(mg: nx.MultiGraph, start: int, end: int) -> nx.Graph:
    edges = filter(
        lambda edge: start <= edge[2]["timestamp"] and edge[2]["timestamp"] < end,
        mg.edges(data=True),
    )
    g = nx.Graph()
    for node in mg.nodes():
        g.add_node(node)
    for u, v, data in edges:
        if g.has_edge(u, v):
            g[u][v]["weight"] += data["weight"]
        else:
            g.add_edge(u, v, weight=data["weight"])
    return g

def similarity_matrix(cluster_label_list: np.ndarray) -> np.ndarray:
    num_points = cluster_label_list.shape[1]
    count: np.ndarray = np.zeros((num_points, num_points), dtype=np.int)
    for label_list in cluster_label_list:
        comm = label_to_comm(label_list)
        for c in comm:
            for i in c:
                for j in c:
                    count[i][j] += 1
    count = count / len(cluster_label_list)
    return count

def receptive_field(g: nx.Graph, hop: int=1) -> sp.sparse.coo_matrix:
    a1 = nx.adjacency_matrix(g) + sp.sparse.identity(g.number_of_nodes())
    out = a1
    while hop > 1:
        hop -= 1
        out = out.__matmul__(a1)
    out = sp.sparse.coo_matrix(out)
    out.data = out.data.astype(np.bool)
    return out

def linear_regression(X: np.ndarray, y: np.array) -> Tuple[np.ndarray, np.ndarray]:
    """
    y = X coef + intercept
    :param X: (n x feature)
    :param y: (n x target)
    :return:
    :coef: (feature x target)
    :intercept: (target)
    """
    model: sklearn.linear_model.LinearRegression = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    return model.coef_.T, model.intercept_

def set_union(s: List[Set[int]]):
    out = set()
    for ss in s:
        for i in ss:
            out.add(i)
    return out
def set_intersection(s: List[Set[int]]):
    out = set()
    for i in s[0]:
        intersection = True
        for ss in s[1:]:
            if i not in ss:
                intersection = False
                break
        if intersection:
            out.add(i)
    return out
def set_difference(a: Set[int], b: Set[int]):
    out = set()
    for i in a:
        if i not in b:
            out.add(i)
    return out
