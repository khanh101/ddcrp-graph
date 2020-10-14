
import networkx as nx
import numpy as np

from src.draw import draw_size, draw_mat
from graph.sbm import sbm, preferential_attachment_cluster
from src.logger import log
from src.model.model import Model
from src.util import comm_to_label, similarity_matrix

seed = 1234
np.random.seed(seed)

# graph
num_clusters = 50
gamma = 2.5
approx_avg_degree = 50
# model
dim = 50

for approx_num_nodes in range(500, 2001, 500):
    g, actual_comm = sbm(preferential_attachment_cluster(num_clusters, gamma), approx_num_nodes, approx_avg_degree)
    log.write_log(f"generated graph: size {g.number_of_nodes()}, cluster size {len(actual_comm)} average degree: {2 * g.number_of_edges() / g.number_of_nodes()} max modularity: {nx.algorithms.community.quality.modularity(g, actual_comm)}")
    draw_size([len(c) for c in actual_comm], name="actual_size", log=True)

    embedding = Model(seed, g.number_of_nodes(), dim).deepwalk_embedding(g)
    for scale in range(1000, 30000, 1000):
        log.write_log(f"scale {scale}")
        comm, kmeans_improved_comm, kmeans_comm = Model(seed, g.number_of_nodes(), dim).ddcrp_iterate(g, embedding, ddcrp_scale=scale)
        diff = abs(len(kmeans_improved_comm) - len(actual_comm)) / len(actual_comm)
        log.write_log(f"scale {scale} diff {diff}")
        log.write_log(f"cluster size {len(kmeans_improved_comm)} kmeans improved modularity: {nx.algorithms.community.quality.modularity(g, kmeans_improved_comm)}")
        log.write_log(f"cluster size {len(kmeans_comm)} kmeans naive    modularity: {nx.algorithms.community.quality.modularity(g, kmeans_comm)}")
