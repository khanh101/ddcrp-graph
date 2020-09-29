from typing import List, Set, Dict

import numpy as np
import scipy as sp
import scipy.sparse

from ddcrp import DDCRP
from prior import NIW, marginal_loglikelihood


def graph_clustering(adjacency: sp.sparse.coo_matrix, embedding: np.ndarray, num_iterations: int) -> List[Set[int]]:
    num_nodes = adjacency.shape[0]
    adjacency_list: List[Dict[int, float]] = [{} for _ in range(num_nodes)]
    num_edges = len(adjacency.data)
    dimension = embedding.shape[1]
    for i in range(num_edges):
        source = adjacency.col[i]
        target = adjacency.row[i]
        weight = adjacency.data[i]
        adjacency_list[source][target] = weight

    def decay(d1: int, d2: int) -> float:
        if d2 in adjacency_list[d1]:
            return 1 / adjacency_list[d1][d2]
        else:
            return float("inf")

    prior = NIW(dimension)
    loglikelihood_dict: Dict[Set[int], float] = {}
    def set2str(s: Set[int]) -> str:
        return "#".join([str(i) for i in sorted(list(s))])
    def loglikelihood(s: Set[int]) -> float:
        key = set2str(s)
        if key not in loglikelihood_dict:
            loglikelihood_dict[key] = marginal_loglikelihood(prior=prior, data=embedding[list(s), :])
        return loglikelihood_dict[key]


    ddcrp = DDCRP(num_nodes, 0.1, decay, loglikelihood)

    for i in range(num_iterations):
        ddcrp.iterate()

    return list(ddcrp.assignment.table2customer.values())

pass
