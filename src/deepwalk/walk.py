import random
from typing import Dict, List

import scipy as sp
import scipy.sparse

class Walk(object):
    def __init__(self, a: sp.sparse.coo_matrix):
        self._edges: Dict[int, List[int]] = {}
        self._degree_in: Dict[int, int] = {}
        for node in range(a.shape[0]):
            self._edges[node]: List[int] = []
            self._degree_in[node]: int = 0
        for edge in range(len(a.data)):
            source = a.col[edge]
            target = a.row[edge]
            self._edges[source].append(target)
            self._degree_in[target] += 1

    def _walk_from_node(
            self,
            start: int,
            length: int,
            degree_normalization: bool,
            nonbacktracking: bool = False,
    ) -> List[int]:
        path: List[int] = [-1, start, ]
        for _ in range(length):
            seq: List[int] = [*self._edges[path[-1]]]
            if nonbacktracking:
                for backtrack in path:
                    try:
                        seq.remove(backtrack)
                    except ValueError as e:
                        pass
            if len(seq) == 0:
                break

            degree_likelihood: List[float] = None
            if degree_normalization:
                # degree normalization
                degree_likelihood = [1 / self._degree_in[node] for node in seq]
            else:
                # uniform distribution
                degree_likelihood = [1 for _ in seq]

            target = seq[random.choices(range(len(degree_likelihood)), weights= degree_likelihood, k=1)[0]]
            path.append(target)
        return path[1:]

    def walk(
            self,
            degree_normalization: bool = False,
            nonbacktracking: bool = False,
            walks_per_node: int = 1,
            walk_length: int = 1,
    ) -> List[List[int]]:
        walks: List[List[int]] = []
        for node in range(len(self._degree_in)):
            for _ in range(walks_per_node):
                walks.append(self._walk_from_node(node, walk_length, degree_normalization, nonbacktracking))
        return walks
