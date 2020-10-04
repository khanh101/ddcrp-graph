import time

import scipy as sp
import scipy.sparse
from src.ddcrp.interface.clustering import clustering
import networkx as nx
import numpy as np

from src.deepwalk.walk import Walk
from src.deepwalk.word2vec import Word2Vec
from draw import draw_size, draw_mat
from src.mcla.mcla import mcla
from src.util import label_to_comm

seed = 1234
np.random.seed(seed)

def deepwalk(a: sp.sparse.coo_matrix, dim: int, nonbacktracking: bool = False) -> np.ndarray:
    num_nodes = a.shape[0]
    walks_per_node = 20
    walk_length = 10
    context = 5
    num_iterations = 10
    degree_normalization = False
    w = Walk(a)
    walks = w.walk(
        degree_normalization=degree_normalization,
        nonbacktracking=nonbacktracking,
        walks_per_node=walks_per_node,
        walk_length=walk_length,
    )
    embedding_list = Word2Vec(
        num_nodes=num_nodes,
        walks=walks,
        dim=dim,
        context=context,
        num_iterations=num_iterations,
    )

    emb = np.zeros(shape=(num_nodes, dim))
    for node, emb_vec in enumerate(embedding_list):
        emb[node, :] = emb_vec

    return emb


def similarity_matrix(cluster_label_list: np.ndarray) -> np.ndarray:
    num_points = cluster_label_list.shape[1]
    count: np.ndarray = np.zeros((num_points, num_points), dtype=np.int)
    for label_list in cluster_label_list:
        comm = label_to_comm(label_list)
        for c in comm:
            for i in c:
                for j in c:
                    count[i][j] += 1
    count = count / len(cluster_label_list)
    return count

def average_degree(a: sp.sparse.coo_matrix) -> float:
    return a.sum() / a.shape[0]


dim = 50
num_clusters = 10 # 3
prior_scale = 5
cluster_scale = 1
gamma = 2.5
num_points = 1000 # 10
cluster_size = np.random.random(size=(num_clusters,)) ** 2.5
cluster_size /= cluster_size.sum()
cluster_size *= num_points
cluster_size = 1 + cluster_size.astype(np.int)
num_points = sum(cluster_size)

draw_size(cluster_size)
p_in = 50 / num_points
p_out = p_in / 10
p = (p_in * np.identity(len(cluster_size))) + p_out



g = nx.generators.community.stochastic_block_model(
    sizes=cluster_size,
    p=p,
    seed=seed,
)
a = nx.adjacency_matrix(g)
# a = a.dot(a)
a = sp.sparse.coo_matrix(a)
print(f"average degree: {average_degree(a)}")
a.data = a.data.astype(np.float64)
print(f"num edges: {len(a.data)}")
cluster_list = [set() for _ in range(num_clusters)]
count = 0
for cluster, size in enumerate(cluster_size):
    for _ in range(size):
        cluster_list[cluster].add(count)
        count += 1

print(f"max modularity: {nx.algorithms.community.quality.modularity(g, cluster_list)}")

t0 = time.time()
data = deepwalk(a, dim)

data -= data.mean(axis=0)
data /= data.std(axis=0).mean()
t1 = time.time()
print(f"deepwalk time: {t1-t0}")

for e in range(len(a.data)):
    a.data[e] = - 100000 * ((data[a.col[e]] - data[a.row[e]])**2).sum()
t0 = time.time()
cluster_label_list = clustering(seed, 1+int(17000 / num_points), data, -float("inf"), a)
t1 = time.time()
print(f"ddcrp time: {t1-t0}")

sim = similarity_matrix(cluster_label_list)
draw_mat(sim)

num_clusters = None
comm = []
for cluster_label in cluster_label_list:
    c = label_to_comm(cluster_label)
    num_clusters = len(c)
    comm.extend(c)

cluster_list = mcla(comm, num_clusters)

print(len(cluster_list))
print(cluster_list)
draw_size(sorted([len(cluster) for cluster in cluster_list]))
print(f"run modularity: {nx.algorithms.community.quality.modularity(g, cluster_list)}")

