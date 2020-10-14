from typing import List, Tuple, Dict, Any

import networkx as nx

from src.graph.data import load_data
from src.logger import log
from src.model.model import Model
from src.util import subgraph_by_timestamp

seed = 1234

mg = load_data(dataset_name="email")


def timestamp(edge: Tuple[int, int, Dict[str, Any]]) -> int:
    return edge[2]["timestamp"]


edge_list = list(mg.edges(data=True))
edge_list.sort(key=lambda x: timestamp(x))

dim = 50
num_folds = 1000
fold_size = int(len(edge_list) / num_folds)
for hop in [1, 2]:
    for window in range(10, 51, 10):
        for scale in range(1000, 10001, 1000):
            start = 0
            model = Model(seed, mg.number_of_nodes(), dim)
            while True:
                end = start + window * fold_size
                #####
                g = subgraph_by_timestamp(
                    mg,
                    timestamp(edge_list[start]),
                    timestamp(edge_list[end]),
                )
                embedding = model.deepwalk_embedding(g)
                comm, kmeans_improved_comm, kmeans_comm = model.ddcrp_iterate(g, embedding, ddcrp_scale=scale,
                                                                          receptive_hop=hop)
                log.write_log(f"range: {start} -> {end}")
                log.write_log(f"window: {window}")
                log.write_log(f"scale {scale}")
                log.write_log(f"hop {hop}")
                log.write_log(f"cluster size {len(kmeans_improved_comm)} kmeans improved modularity: {nx.algorithms.community.quality.modularity(g, kmeans_improved_comm)}")
                log.write_log(f"cluster size {len(kmeans_comm)} kmeans naive    modularity: {nx.algorithms.community.quality.modularity(g, kmeans_comm)}")

            #####
            start += fold_size

pass
