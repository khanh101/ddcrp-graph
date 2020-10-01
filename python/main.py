from typing import Set, Dict

import numpy as np

from python.clustering import spectral_clustering
from python.ddcrp.ddcrp import DDCRP
from python.ddcrp.prior import NIW, marginal_loglikelihood

from python.draw import draw_data, draw_size, draw_mat
from python.ensemble import Ensemble
from python.util import set2str

num_clusters = 10
prior_scale = 5
cluster_scale = 1
gamma = 2.5
num_points = 300
cluster_size = np.random.random(size=(num_clusters,)) ** 2.5
cluster_size /= cluster_size.sum()
cluster_size *= num_points
cluster_size = 1 + cluster_size.astype(np.int)
num_points = sum(cluster_size)
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


def logdecay(d1: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for d2 in range(num_points):
        out[d2] = -(abs(data[d1] - data[d2]) ** (0.5)).sum()
    return out


prior = NIW(2)
loglikelihood_dict: Dict[str, float] = {}


def loglikelihood(s: Set[int]) -> float:
    key = set2str(s)
    if key not in loglikelihood_dict:
        loglikelihood_dict[key] = marginal_loglikelihood(prior=prior, data=data[list(s), :])
    return loglikelihood_dict[key]


ddcrp = DDCRP(len(data))

ens = Ensemble(num_points)

num_clusters: int = 0
num_iterations: int = 10
for i in range(num_iterations):
    ddcrp.iterate(-float("inf"), logdecay, loglikelihood)
    table = ddcrp.assignment.table();
    ens.add(table)
    max_misclustering_rate: float = ens.misclustering_rate().mean()
    print(f"iter {i} num clusters {len(table)} misclustering_rate {max_misclustering_rate}")
    num_clusters += len(table)

num_clusters = int(num_clusters / num_iterations)

draw_mat(ens.similarity())
draw_mat(ens.misclustering_rate())

A = ens.similarity() * (1.0 - np.identity(num_points))
cluster_list = spectral_clustering(A, int(np.log(num_points)), num_clusters)

draw_size([len(cluster) for cluster in cluster_list])

draw_data(data, cluster_list)

