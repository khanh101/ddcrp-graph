from typing import Tuple, Iterator, Any

import numpy as np
import scipy.sparse as sparse


class Node(object):
    def __init__(self, index: int, name: str):
        self.index: int = index
        self.name: str = name

    def __str__(self) -> str:
        return f"{{{self.index}: {self.name}}}"


class Edge(object):
    def __init__(self, index: int, source: int, destination: int, timestamp: int, weight: float):
        self.index: int = index
        self.source: int = source
        self.destination: int = destination
        self.timestamp: int = timestamp
        self.weight: float = weight

    def __str__(self) -> str:
        return f"{{{self.index}: {self.source}->{self.destination}: {self.weight}}}"


class Graph(object):
    _adjacency_matrix: sparse.coo_matrix

    def __init__(self):
        self._adjacency_matrix = None

    def num_nodes(self) -> int:
        raise Exception("Abstract Class")

    def num_edges(self) -> int:
        raise Exception("Abstract Class")

    def nodes(self) -> Iterator[Node]:
        raise Exception("Abstract Class")

    def edges(self) -> Iterator[Edge]:
        raise Exception("Abstract Class")

    def __getitem__(self, item) -> Any:
        return SliceGraph(self, item)

    def degree(self) -> Tuple[np.ndarray, np.ndarray]:
        adj = self.adjacency_matrix()
        degree_in = np.asarray(adj.sum(axis=1)).flatten()
        degree_out = np.asarray(adj.sum(axis=0)).flatten()
        return degree_in, degree_out

    def adjacency_matrix(self) -> sparse.coo_matrix:
        if self._adjacency_matrix is None:
            row = np.zeros(shape=(self.num_edges(),), dtype=np.int)
            col = np.zeros(shape=(self.num_edges(),), dtype=np.int)
            data = np.zeros(shape=(self.num_edges(),), dtype=np.float)

            for i, edge in enumerate(self.edges()):
                col[i] = edge.source
                row[i] = edge.destination
                data[i] = edge.weight

            self._adjacency_matrix = sparse.coo_matrix((data, (row, col)), shape=(self.num_nodes(), self.num_nodes()))

        return self._adjacency_matrix


class SliceGraph(Graph):
    _g: Graph
    _slice: slice
    _num_edges: int

    def __init__(self, g: Graph, slice_item: slice):
        super(SliceGraph, self).__init__()
        if slice_item.step is not None and slice_item != 1:
            raise Exception("Slice step must be 1")
        self._slice = slice_item
        self._g = g
        num_edges = 0
        for _ in self.edges():
            num_edges += 1
        self._num_edges = num_edges

    def num_nodes(self) -> int:
        return self._g.num_nodes()

    def num_edges(self) -> int:
        return self._num_edges

    def nodes(self) -> Iterator[Node]:
        return self._g.nodes()

    def edges(self) -> Iterator[Edge]:
        for index, edge in enumerate(self._g.edges()):
            if self._slice.start is not None and index < self._slice.start:
                continue
            if self._slice.stop is not None and self._slice.stop <= index:
                continue
            yield edge


class AdjGraph(Graph):
    _adjacency_matrix: sparse.coo_matrix

    def __init__(self, adjacency_matrix: sparse.coo_matrix):
        super(AdjGraph, self).__init__()
        self._adjacency_matrix = sparse.coo_matrix(adjacency_matrix)

    def num_nodes(self) -> int:
        return self._adjacency_matrix.shape[0]

    def num_edges(self) -> int:
        return self._adjacency_matrix.data.shape[0]

    def nodes(self) -> Iterator[Node]:
        for i in range(self.num_nodes()):
            yield Node(index=i, name=f"{i}")

    def edges(self) -> Iterator[Edge]:
        for i in range(self.num_edges()):
            u = self._adjacency_matrix.row[i]
            v = self._adjacency_matrix.col[i]
            w = self._adjacency_matrix.data[i]
            yield Edge(index=i, source=u, destination=v, timestamp=i, weight=w)

    def adjacency_matrix(self) -> sparse.coo_matrix:
        return self._adjacency_matrix
