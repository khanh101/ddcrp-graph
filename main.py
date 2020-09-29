from typing import Set, Dict

import numpy as np
from matplotlib import pyplot as plt

from ddcrp.ddcrp import DDCRP
from ddcrp.prior import NIW, marginal_loglikelihood

import scipy as sp
import scipy.spatial

identity = [
    [0.5, 0],
    [0, 0.5],
]
cluster1 = np.random.multivariate_normal([-2, -2], identity, size=(200,))
cluster2 = np.random.multivariate_normal([+2, +2], identity, size=(200,))

data = np.concatenate((cluster1, cluster2), axis=0)

plt.scatter(data[:, 0], data[:, 1])
plt.show()


def logdecay(d1: int, d2: int) -> float:
    return -(abs(data[d1] - data[d2])**(2.0)).sum()


prior = NIW(1)
loglikelihood_dict: Dict[Set[int], float] = {}
def set2str(s: Set[int]) -> str:
    return "#".join([str(i) for i in sorted(list(s))])
def loglikelihood(s: Set[int]) -> float:
    key = set2str(s)
    if key not in loglikelihood_dict:
        loglikelihood_dict[key] = marginal_loglikelihood(prior=prior, data=data[list(s), :])
    return loglikelihood_dict[key]


ddcrp = DDCRP(len(data), -float("inf"), logdecay, loglikelihood)

for i in range(10):
    ddcrp.iterate()
    print(f"iter {i} num clusters {len(ddcrp.assignment.table2customer)}")
cluster_size = []
for table in ddcrp.assignment.table2customer.values():
    cluster_size.append(len(table))
    cluster = data[list(table), :]
    if len(table) < 3:
        plt.plot(cluster[:, 0], cluster[:, 1])
    else:
        hull = sp.spatial.ConvexHull(cluster)
        vertices = list(hull.vertices) + [hull.vertices[0]]
        plt.plot(cluster[vertices, 0], cluster[vertices, 1])

plt.show()

cluster_size.sort()
plt.scatter(range(len(cluster_size)), cluster_size)
plt.show()

pass
