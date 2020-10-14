import time
from typing import List, Set, Any

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
    deepwalk: DeepWalk
    ddcrp: DDCRP
    context: int
    def __init__(self, seed: int, num_nodes: int, dim: int, context: int = 5):
        super(Model, self).__init__()
        self.deepwalk = DeepWalk(dim, context)
        self.ddcrp = DDCRP(seed, num_nodes, dim)
        self.context = context

    def deepwalk_embedding(
            self,
            g: nx.Graph,
            deepwalk_epochs: int=10,
    ) -> np.ndarray:
        # deepwalk
        walks_per_node: int = int(2 * g.number_of_edges() / g.number_of_nodes())
        walk_length: int = 3 * self.context
        walks = random_walk(g, walks_per_node, walk_length)
        t0 = time.time()
        embedding = self.deepwalk.train(walks, deepwalk_epochs)
        print(f"deepwalk time: {time.time() - t0}s")
        embedding -= embedding.mean(axis=0)  # normalized
        embedding /= embedding.std(axis=0).mean()  # normalized
        return embedding

    def ddcrp_iterate(
            self,
            g: nx.Graph,
            embedding: np.ndarray,
            ddcrp_iterations: int=10,
            ddcrp_logalpha: float= -float("inf"),
            receptive_hop: int = 1,
            ddcrp_scale: float = 5000,
    ) -> Any:
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
        label_list = self.ddcrp.iterate(
            ddcrp_iterations,
            embedding,
            ddcrp_logalpha,
            adj,
        )
        log.write_log(f"ddcrp time: {time.time() - t0}")
        comm_list = []
        for label in label_list:
            comm_list.extend(label_to_comm(label))
        # mcla
        comm = mcla(comm_list)
        kmeans_improved_comm = kmeans_improve(embedding, comm)
        kmeans_comm = kmeans_improve(embedding, len(comm))
        return comm, kmeans_improved_comm, kmeans_comm