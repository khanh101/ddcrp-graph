import time

import networkx as nx
import numpy as np

from draw import draw_size, draw_mat
from graph.sbm import sbm
from src.ddcrp.interface.DDCRP import DDCRP
from src.deepwalk.deepwalk import DeepWalk
from src.deepwalk.walk import random_walk
from src.kmeans.kmeans import kmeans_improve
from src.mcla.mcla import mcla
from util import comm_to_label, similarity_matrix, receptive_field, label_to_comm

seed = 1234
np.random.seed(seed)

# graph
num_clusters = 50
gamma = 2.5
approx_num_nodes = 1000
approx_avg_degree = 50

g, comm = sbm(num_clusters, gamma, approx_num_nodes, approx_avg_degree)
print(f"num clusters: {len(comm)}")
print(f"max modularity: {nx.algorithms.community.quality.modularity(g, comm)}")
num_nodes = g.number_of_nodes()
draw_size([len(c) for c in comm], name="actual_size")

label = comm_to_label(comm)
label_list = np.array([label])

sim = similarity_matrix(label_list)
draw_mat(sim)
# deepwalk
dim = 50
context = 5
walks_per_node = 20
walk_length = 10
epochs = 10
deepwalk = DeepWalk(dim, context)
t0 = time.time()
embedding = deepwalk.train(random_walk(g, walks_per_node, walk_length), epochs)
print(f"deepwalk time: {time.time() - t0}")
embedding -= embedding.mean(axis=0)  # normalized
embedding /= embedding.std(axis=0).mean()  # normalized
# ddcrp
num_iterations = 10
logalpha = -float("inf")
hop = 1
adj = receptive_field(g, hop)


def distance(scale: int = 10000):
    data = np.empty((len(adj.data),), dtype=np.float64)
    for e in range(len(adj.data)):
        u, v = adj.col[e], adj.row[e]
        data[e] = - scale * ((embedding[u] - embedding[v]) ** 2).sum()
    return data


scale = 10000
adj.data = distance(scale)

ddcrp = DDCRP(seed, num_nodes, dim)
t0 = time.time()
label_list = ddcrp.iterate(num_iterations, embedding, logalpha, adj)
print(f"ddcrp time: {time.time() - t0}")

comm_list = []
for label in label_list:
    comm_list.extend(label_to_comm(label))
# mcla
comm = mcla(comm_list)
draw_size([len(c) for c in comm], name="predicted_size")
# evaluate
print(f"predicted num clusters")
print(f"initial modularity: {nx.algorithms.community.quality.modularity(g, comm)}")
print(f"kmeans improved modularity: {nx.algorithms.community.quality.modularity(g, kmeans_improve(embedding, comm))}")
print(f"kmeans naive modularity: {nx.algorithms.community.quality.modularity(g, kmeans_improve(embedding, len(comm)))}")

pass
