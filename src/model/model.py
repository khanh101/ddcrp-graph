import time
from typing import List, Set, Any, Dict, Tuple, Union

import numpy as np
import networkx as nx

from src.ddcrp.interface.DDCRP import DDCRP
from src.deepwalk.deepwalk import DeepWalk
from src.deepwalk.walk import random_walk
from src.draw import draw_size
from src.kmeans.kmeans import kmeans_improve
from src.logger import log
from src.mcla.mcla import mcla
from src.util import receptive_field, label_to_comm


class Model(object):
    _deepwalk: DeepWalk
    _ddcrp: DDCRP
    _context: int
    def __init__(self, seed: int, num_nodes: int, dim: int, context: int = 5):
        super(Model, self).__init__()
        self._deepwalk = DeepWalk(dim, context)
        self._ddcrp = DDCRP(seed, num_nodes, dim)
        self._context = context

    def deepwalk(
            self,
            g: nx.Graph,
            deepwalk_epochs: int=10,
    ) -> np.ndarray:
        # deepwalk
        walks_per_node: int = int(2 * g.number_of_edges() / g.number_of_nodes())
        walk_length: int = 3 * self._context
        walks = random_walk(g, walks_per_node, walk_length)
        t0 = time.time()
        embedding = self._deepwalk.train(walks, deepwalk_epochs)
        print(f"deepwalk time: {time.time() - t0}s")
        embedding -= embedding.mean(axis=0)  # normalized
        embedding /= embedding.std(axis=0).mean()  # normalized
        return embedding

    def ddcrp(
            self,
            g: nx.Graph,
            embedding: np.ndarray,
            ddcrp_iterations: int=10,
            ddcrp_logalpha: float= -float("inf"),
            receptive_hop: int = 1,
            ddcrp_scale: float = 5000,
    ) -> List[List[Set[int]]]:
        # ddcrp
        adj = receptive_field(g, receptive_hop)

        def distance(scale: float = 5000):
            data = np.empty((len(adj.data),), dtype=np.float64)
            for e in range(len(adj.data)):
                u, v = adj.col[e], adj.row[e]
                data[e] = - scale * ((embedding[u] - embedding[v]) ** 2).sum()
            return data

        adj.data = distance(ddcrp_scale)
        t0 = time.time()
        label_list = self._ddcrp.iterate(
            ddcrp_iterations,
            embedding,
            ddcrp_logalpha,
            adj,
        )
        log.write_log(f"ddcrp time: {time.time() - t0}")
        comm_list = []
        for label in label_list:
            comm_list.append(label_to_comm(label))
        return comm_list

    @staticmethod
    def mcla(comm_list: List[List[Set[int]]], reference: List[Set[int]]) -> Tuple[List[Set[int]], List[Set[int]]]:
        # mcla
        next_comm: List[Set[int]] = []
        for comm in comm_list:
            next_comm.extend(comm)
        comm, mapping = mcla(next_comm, reference)
        return comm, mapping

    @staticmethod
    def kmeans(embedding: np.ndarray, init_comm: Union[int, List[Set[int]]]) -> List[Set[int]]:
        return kmeans_improve(embedding, init_comm)
