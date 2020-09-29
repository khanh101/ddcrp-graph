from typing import Set, Dict

import numpy as np
from matplotlib import pyplot as plt

from ddcrp.ddcrp import DDCRP
from ddcrp.prior import NIW, marginal_loglikelihood

import scipy as sp
import scipy.spatial

from draw import draw_data

num_clusters = 100
prior_scale = 5
cluster_scale = 1
gamma = 2.5
num_nodes = 300
size = np.random.random(size=(num_clusters,)) ** 2.5
size /= size.sum()
size *= num_nodes
size = 1 + size.astype(np.int)


cluster_list = [
    np.random.multivariate_normal(
        np.random.multivariate_normal([0, 0], prior_scale * np.identity(2)),
        cluster_scale * np.identity(2),
        size=s,
    )
    for s in size
]

data = np.concatenate(cluster_list, axis=0)

data -= data.mean(axis=0)
data /= data.std(axis=0)

draw_data(data)


def logdecay(d1: int, d2: int) -> float:
    return -(abs(data[d1] - data[d2]) ** (0.5)).sum()


prior = NIW(1)
loglikelihood_dict: Dict[Set[int], float] = {}


def set2str(s: Set[int]) -> str:
    return "#".join([str(i) for i in sorted(list(s))])


def loglikelihood(s: Set[int]) -> float:
    key = set2str(s)
    if key not in loglikelihood_dict:
        loglikelihood_dict[key] = marginal_loglikelihood(prior=prior, data=data[list(s), :])
    return loglikelihood_dict[key]


ddcrp = DDCRP(len(data))

for i in range(10):
    ddcrp.iterate(-float("inf"), logdecay, loglikelihood)
    print(f"iter {i} num clusters {len(ddcrp.assignment.table2customer)}")

cluster_list = list(ddcrp.assignment.table2customer.values())

cluster_size = []
for table in ddcrp.assignment.table2customer.values():
    cluster_size.append(len(table))

cluster_size.sort()
plt.scatter(range(len(cluster_size)), cluster_size)
plt.show()

draw_data(data, cluster_list)

pass
