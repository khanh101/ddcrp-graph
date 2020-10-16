from typing import List, Tuple, Dict, Any, Set

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
            end_loop = False
            comm: List[Set[int]] = []
            while True:
                if end_loop:
                    break
                end = start + window * fold_size
                if end >= len(edge_list):
                    end = len(edge_list) - 1
                    end_loop = True
                #####
                g = subgraph_by_timestamp(
                    mg,
                    timestamp(edge_list[start]),
                    timestamp(edge_list[end]),
                )
                embedding = model.deepwalk(g)
                comm_list = model.ddcrp(g, embedding, ddcrp_scale=scale, receptive_hop=hop)
                comm, mapping = model.mcla(comm_list, comm)
                log.write_log(f"range: {start} -> {end}")
                log.write_log(f"window: {window}")
                log.write_log(f"scale {scale}")
                log.write_log(f"hop {hop}")
                log.write_log(f"{mapping}")

                #####
                start += fold_size

pass
