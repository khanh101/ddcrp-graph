import networkx as nx
import time
from typing import List, Tuple, Dict, Any, Set

from src.graph.data import load_data
from src.logger import log
from src.model.model import Model
from src.util import subgraph_by_timestamp, set_intersection, set_difference, set_union

seed = 1234

mg = load_data(dataset_name="email")


def timestamp(edge: Tuple[int, int, Dict[str, Any]]) -> int:
    return edge[2]["timestamp"]


edge_list = list(mg.edges(data=True))
edge_list.sort(key=lambda x: timestamp(x))

dim = 50
num_folds = 1000
fold_size = int(len(edge_list) / num_folds)
ddcrp_cutoff = 5


def log_filename(hop: int, window: int, scale: float) -> str:
    return f"hop_{hop}_window_{window}_scale_{scale}"


def write_first_line(hop: int, window: int, scale: float):
    log.write_csv(
        data=["from_timestamp", "to_timestamp", "average degree", "predicted cluster size", "modularity", "performance",
              "improved modularity", "improved performance", "naive modularity", "naive performance", "ddcrp time",
              "response"],
        name=log_filename(hop, window, scale),
    )


def write_line(hop: int, window: int, scale: float, from_timestamp: int, to_timestamp: int, average_degree: float,
               predicted_cluster_size: int, modularity: float, performance: float, improved_modularity: float,
               improved_performance: float, naive_modularity: float, naive_performance: float, ddcrp_time: float,
               response: Any):
    log.write_csv(
        data=[from_timestamp, to_timestamp, average_degree, predicted_cluster_size, modularity, performance,
              improved_modularity, improved_performance, naive_modularity, naive_performance, ddcrp_time, response],
        name=log_filename(hop, window, scale),
    )


hop = 1
window = 10
scale = 3000

write_first_line(hop, window, scale)
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
    from_timestamp = timestamp(edge_list[start])
    to_timestamp = timestamp(edge_list[end])
    g = subgraph_by_timestamp(
        mg,
        from_timestamp,
        to_timestamp,
    )
    average_degree = 2 * g.number_of_edges() / g.number_of_nodes()
    embedding = model.deepwalk(g)
    t0 = time.time()
    comm_list = model.ddcrp(g, embedding, ddcrp_scale=scale, receptive_hop=hop)
    ddcrp_time = time.time() - t0
    comm_list = comm_list[ddcrp_cutoff:]
    init_comm, mapping = model.mcla(comm_list, comm)
    predicted_cluster_size = len(init_comm)
    new_comm = model.kmeans(embedding, init_comm)


    def response(new_comm: List[Set[int]]) -> Any:
        out = []
        for i, c in enumerate(mapping):
            if len(c) == 0:
                new = new_comm[i]
                out.append({
                    "type": "new",
                    "join": list(new),
                })
            elif len(c) == 1:
                old = comm[i]
                new = new_comm[i]
                out.append({
                    "type": "old",
                    "remain": list(set_intersection([old, new])),
                    "leave": list(set_difference(old, new)),
                    "join": list(set_difference(new, old)),
                })
            else:
                old = set_union([comm[i] for i in c])
                new = new_comm[i]
                out.append({
                    "type": "join",
                    "remain": list(set_intersection([old, new])),
                    "leave": list(set_difference(old, new)),
                    "join": list(set_difference(new, old)),
                })
        return out


    modularity = nx.algorithms.community.quality.modularity(g, init_comm)
    performance = nx.algorithms.community.quality.performance(g, init_comm)
    improved_modularity = nx.algorithms.community.quality.modularity(g, new_comm)
    improved_performance = nx.algorithms.community.quality.performance(g, new_comm)

    naive_comm = model.kmeans(embedding, len(init_comm))
    naive_modularity = nx.algorithms.community.quality.modularity(g, naive_comm)
    naive_performance = nx.algorithms.community.quality.performance(g, naive_comm)

    write_line(hop, window, scale, from_timestamp, to_timestamp, average_degree, predicted_cluster_size, modularity,
               performance, improved_modularity, improved_performance, naive_modularity, naive_performance, ddcrp_time,
               response(new_comm))

    #####
    start += fold_size
    comm = new_comm

pass
