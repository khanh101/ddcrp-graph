import time

import networkx as nx
import numpy as np

from src.draw import draw_size
from src.graph.sbm import sbm, preferential_attachment_cluster
from src.logger import log
from src.model.model import Model

seed = 1234
np.random.seed(seed)

# graph
num_clusters = 50
gamma = 2.5
# model
dim = 50
deepwalk_epochs = 10
ddcrp_iterations = 10
ddcrp_cutoff = 5
log.write_csv(["graph size", "average degree", "cluster size", "max modularity", "max performance", "scale", "predicted cluster size", "modularity", "performance", "improved modularity", "improved performance", "naive modularity", "naive performance", "ddcrp time"])
def write_line(graph_size: int, average_degree: float, cluster_size: int, max_modularity: float, max_performance: float, scale: float, predicted_cluster_size: int, modularity: float, performance: float, improved_modularity: float, improved_performance: float, naive_modularity: float, naive_performance: float, ddcrp_time: float):
    log.write_csv([graph_size, average_degree, cluster_size, max_modularity, max_performance, scale, predicted_cluster_size, modularity, performance, improved_modularity, improved_performance, naive_modularity, naive_performance, ddcrp_time])
for approx_avg_degree in range(10, 51, 10):
    for approx_num_nodes in range(500, 2001, 500):
        g, actual_comm = sbm(preferential_attachment_cluster(num_clusters, gamma), approx_num_nodes, approx_avg_degree)
        graph_size = g.number_of_nodes()
        average_degree = 2 * g.number_of_edges() / g.number_of_nodes()
        cluster_size = len(actual_comm)
        max_modularity = nx.algorithms.community.quality.modularity(g, actual_comm)
        max_performance = nx.algorithms.community.quality.performance(g, actual_comm)
        embedding = Model(seed, g.number_of_nodes(), dim).deepwalk(g, deepwalk_epochs)
        for scale in range(1000, 30000, 1000):
            t0 = time.time()
            comm_list = Model(seed, g.number_of_nodes(), dim).ddcrp(g, embedding, ddcrp_scale=scale, ddcrp_iterations=ddcrp_iterations)
            ddcrp_time = time.time() - t0
            comm_list = comm_list[ddcrp_cutoff:]
            comm, _ = Model.mcla(comm_list)
            predicted_cluster_size = len(comm)
            modularity = nx.algorithms.community.quality.modularity(g, comm)
            performance = nx.algorithms.community.quality.performance(g, comm)
            improved_comm = Model.kmeans(embedding, comm)
            improved_modularity = nx.algorithms.community.quality.modularity(g, improved_comm)
            improved_performance = nx.algorithms.community.quality.performance(g, improved_comm)
            naive_comm = Model.kmeans(embedding, len(comm))
            naive_modularity = nx.algorithms.community.quality.modularity(g, naive_comm)
            naive_performance = nx.algorithms.community.quality.performance(g, naive_comm)
            write_line(graph_size, average_degree, cluster_size, max_modularity, max_performance, scale, predicted_cluster_size, modularity, performance, improved_modularity, improved_performance, naive_modularity, naive_performance, ddcrp_time)
