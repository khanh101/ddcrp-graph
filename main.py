from typing import Set

import numpy as np
from matplotlib import pyplot as plt

from ddcrp_gibbs.ddcrp import DDCRP

cluster1 = np.random.normal(loc=-2, scale=1, size=(100,))
cluster2 = np.random.normal(loc=+2, scale=1, size=(100,))

data = np.concatenate((cluster1, cluster2), axis=0)

hist, bin_edge = np.histogram(data, bins=20)
plt.bar(bin_edge[:len(hist)], hist)
plt.show()


def decay(d1: int, d2: int) -> float:
    return 1 / (0.0001 + abs(data[d1] - data[d2]) ** (-0.5))


def loglikelihood(s: Set[int]) -> float:
    return 0


ddcrp = DDCRP(len(data), 0.01, decay, loglikelihood)

for i in range(100):
    ddcrp.iterate()
    print(ddcrp.assignment.table2customer)


pass
