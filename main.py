
import networkx as nx
import numpy as np

from src.draw import draw_size, draw_mat
from graph.sbm import sbm
from src.model.model import Model
from src.util import comm_to_label, similarity_matrix

seed = 1234
np.random.seed(seed)

# graph
num_clusters = 50
gamma = 2.5
approx_num_nodes = 1000
approx_avg_degree = 50

g, comm = sbm(num_clusters, gamma, approx_num_nodes, approx_avg_degree)
print(f"num clusters: {len(comm)}")
print(f"average degree: {2 * g.number_of_edges() / g.number_of_nodes()}")
print(f"max modularity: {nx.algorithms.community.quality.modularity(g, comm)}")
num_nodes = g.number_of_nodes()
draw_size([len(c) for c in comm], name="actual_size")

label = comm_to_label(comm)
label_list = np.array([label])

sim = similarity_matrix(label_list)
draw_mat(sim)
# model
dim = 50
model = Model(seed, num_nodes, dim)
model.iterate(g)

