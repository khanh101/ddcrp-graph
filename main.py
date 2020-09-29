from typing import Set, Dict

import numpy as np
from matplotlib import pyplot as plt

from ddcrp import DDCRP
from prior import NIW, marginal_loglikelihood

cluster1 = np.random.normal(loc=-2, scale=1, size=(50,))
cluster2 = np.random.normal(loc=+2, scale=1, size=(50,))

data = np.concatenate((cluster1, cluster2), axis=0)
data = np.sort(data)

plt.scatter(range(len(data)), data)
plt.show()


hist, bin_edge = np.histogram(data, bins=20)
plt.bar(bin_edge[:len(hist)], hist)
plt.show()


def decay(d1: int, d2: int) -> float:
    return np.exp(- (data[d1] - data[d2])**2)


prior = NIW(1)
loglikelihood_dict: Dict[Set[int], float] = {}
def set2str(s: Set[int]) -> str:
    return "#".join([str(i) for i in sorted(list(s))])
def loglikelihood(s: Set[int]) -> float:
    key = set2str(s)
    if key not in loglikelihood_dict:
        loglikelihood_dict[key] = marginal_loglikelihood(prior=prior, data=data[list(s)].reshape((len(s), 1)))
    return loglikelihood_dict[key]


ddcrp = DDCRP(len(data), 0.1, decay, loglikelihood)

for i in range(100):
    ddcrp.iterate()
    print(f"num clusters {len(ddcrp.assignment.table2customer)}")
    for table in ddcrp.assignment.table2customer.values():
        plt.scatter(list(table), data[list(table)])
    plt.show()


pass
