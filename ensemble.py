from typing import List, Set

import numpy as np

class Ensemble(object):
    num_items: int
    count: int
    similarity_count: np.ndarray
    def __init__(self, num_items: int):
        super(Ensemble, self).__init__()
        self.num_items = num_items
        self.count = 0
        self.similarity_count = np.zeros((num_items, num_items))

    def add(self, cluster_list: List[Set[int]]):
        self.count += 1
        for cluster in cluster_list:
            for a in cluster:
                for b in cluster:
                    self.similarity_count[a][b] += 1

    def similarity(self) -> np.ndarray:
        return self.similarity_count / self.count

