import time

import scipy as sp
import scipy.sparse
from cpp.clustering import clustering

import numpy as np

from python.draw import draw_data, draw_size
seed = 1234
np.random.seed(seed)

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
num_edges = 100 * num_points
draw_size(cluster_size)

data_list = [
    np.random.multivariate_normal(
        np.random.multivariate_normal([0, 0], prior_scale * np.identity(2)),
        cluster_scale * np.identity(2),
        size=size,
        )
    for size in cluster_size
]

cluster_list = [set() for _ in range(num_clusters)]
count = 0
for cluster, size in enumerate(cluster_size):
    for _ in range(size):
        cluster_list[cluster].add(count)
        count += 1

data = np.concatenate(data_list, axis=0)

data -= data.mean(axis=0)
data /= data.std(axis=0)

draw_data(data, cluster_list)

adj = np.empty((num_points, num_points))
for i in range(num_points):
    for j in range(num_points):
        adj[i][j] = - 100000 * ((data[i] - data[j])**2).sum()

edge_list = list(adj.reshape((num_points*num_points)))
edge_list.sort()
edge_list.reverse()
max_edge = edge_list[0]
print(f"max_edge: {max_edge}")
cutoff = edge_list[num_edges]

row = []
col = []
edge = []
for i in range(num_points):
    for j in range(num_points):
        if adj[i][j] >= cutoff:
            row.append(i)
            col.append(j)
            edge.append(adj[i][j])
row = np.array(row, dtype= np.uint64)
col = np.array(col, dtype=np.uint64)
edge = np.array(edge, dtype=np.double)
adj = sp.sparse.coo_matrix((edge, (row, col)), shape=(num_points, num_points))

print(f"num_edges: {len(adj.data)}")
t0 = time.time()
cluster_list = clustering(seed, 10, data, -float("inf"), adj)
t1 = time.time()
print(f"elapsed time: {t1-t0}")

draw_size([len(cluster) for cluster in cluster_list])

draw_data(data, cluster_list)


