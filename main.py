from typing import Set, Dict

import numpy as np

from ddcrp.ddcrp import DDCRP
from ddcrp.prior import NIW, marginal_loglikelihood

from draw import draw_data, draw_size

num_clusters = 10
prior_scale = 5
cluster_scale = 1
gamma = 2.5
num_nodes = 300
cluster_size = np.random.random(size=(num_clusters,)) ** 2.5
cluster_size /= cluster_size.sum()
cluster_size *= num_nodes
cluster_size = 1 + cluster_size.astype(np.int)

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


def logdecay(d1: int, d2: int) -> float:
    return -(abs(data[d1] - data[d2]) ** (0.5)).sum()


prior = NIW(2)
loglikelihood_dict: Dict[Set[int], float] = {}


def set2str(s: Set[int]) -> str:
    return "#".join([str(i) for i in sorted(list(s))])


def loglikelihood(s: Set[int]) -> float:
    #return 0
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

draw_size(cluster_size)

draw_data(data, cluster_list)

pass
