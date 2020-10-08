import random
from typing import List

import scipy as sp
import scipy.sparse

class Walk(object):
    def __init__(self, a: sp.sparse.coo_matrix):
        self.edge_list: List[int, List[int]] = []
        self.weight_list: List[int, List[float]] = []
        for node in range(a.shape[0]):
            self.edge_list.append([])
            self.weight_list.append([])
        for edge in range(len(a.data)):
            source = a.col[edge]
            target = a.row[edge]
            weight = a.data[edge]
            self.edge_list[source].append(target)
            self.weight_list[source].append(weight)

    def _walk_from_node(
            self,
            start: int,
            length: int,
    ) -> List[int]:
        path: List[int] = [-1, start, ]
        for _ in range(length):
            neighbour: List[int] = [*self.edge_list[path[-1]]]
            weight: List[int] = [*self.weight_list[path[-1]]]
            if len(neighbour) == 0:
                break

            target = random.choices(neighbour, weights= weight, k=1)[0]
            path.append(target)
        return path[1:]

    def walk(
            self,
            walks_per_node: int = 1,
            walk_length: int = 1,
    ) -> List[List[int]]:
        walks: List[List[int]] = []
        for node in range(len(self.edge_list)):
            for _ in range(walks_per_node):
                walks.append(self._walk_from_node(node, walk_length))
        return walks
