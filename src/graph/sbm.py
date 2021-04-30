import networkx as nx
import numpy as np
from typing import List, Set, Tuple


def preferential_attachment_cluster(num_clusters: int, gamma: float) -> np.ndarray:
    """
    cluster size follows power law of gamma
    density ~ size ^ (-gamma)
    :param num_clusters:
    :param gamma:
    :return:
    """
    cluster_size = ((0.5 + np.arange(0, num_clusters)) / num_clusters) ** (1 / (-gamma + 1))
    cluster_size /= cluster_size.sum()
    return cluster_size


def sbm(cluster_size: np.ndarray, appox_num_nodes: float, approx_avg_degree: float) -> Tuple[nx.Graph, List[Set[int]]]:
    num_clusters = len(cluster_size)
    cluster_size /= cluster_size.sum()
    cluster_size *= appox_num_nodes
    cluster_size = np.rint(cluster_size).astype(np.int)
    cluster_size[cluster_size <= 0] = 1
    num_nodes = cluster_size.sum()
    sum_of_edges = approx_avg_degree * num_nodes / 2
    alpha = 10.0  # p_in / p_out
    p_out = 2 * sum_of_edges / (num_nodes ** 2 + sum([(alpha - 1) * size ** 2 for size in cluster_size]))
    p_inn = alpha * p_out
    p = p_out * np.ones((num_clusters, num_clusters)) + (p_inn - p_out) * np.identity(num_clusters)

    start = 0
    comm = []
    for size in cluster_size:
        comm.append(set(range(start, start + size)))
        start += size

    return nx.generators.community.stochastic_block_model(
        sizes=cluster_size,
        p=p,
    ), comm
