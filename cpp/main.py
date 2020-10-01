import scipy as sp
import scipy.sparse
from cpp.clustering import clustering

import numpy as np

from python.draw import draw_data, draw_size

num_clusters = 10 # 3
prior_scale = 5
cluster_scale = 1
gamma = 2.5
num_points = 500 # 10
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
        adj[i][j] = - 10000 * ((data[i] - data[j])**2).sum()

edge_list = list(adj.reshape((num_points*num_points)))
edge_list.sort()
edge_list.reverse()
cutoff = edge_list[num_edges]
adj[adj < cutoff] = 0

adj = sp.sparse.coo_matrix(adj)
adj.eliminate_zeros()
print(len(adj.data))
cluster_list = clustering(1234, 10, data, -float("inf"), adj)

draw_size([len(cluster) for cluster in cluster_list])

draw_data(data, cluster_list)


