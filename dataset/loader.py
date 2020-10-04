import io
from typing import Iterator

from dataset import Graph, Node, Edge


class Loader(Graph):
    def __init__(self, dataset_name: str = "email"):
        super(Loader, self).__init__()
        """
        dataset_name:
          - email
          - reddit
          - dblp
        """
        self._dataset_name: str = dataset_name
        self._num_nodes: int = None
        self._num_edges: int = None
        self._nodes: io.TextIOWrapper = None
        self._edges: io.TextIOWrapper = None
        # OPEN FILE
        self._nodes = open(f"dataset/{self._dataset_name}-nodes.csv")
        self._edges = open(f"dataset/{self._dataset_name}-edges.csv")
        self._num_nodes = -1
        self._num_edges = -1
        for _ in self._nodes:
            self._num_nodes += 1
        for _ in self._edges:
            self._num_edges += 1

    def close(self):
        self._nodes.close()
        self._edges.close()

    def seek0(self):
        self._nodes.seek(0, 0)
        self._edges.seek(0, 0)

    def num_nodes(self) -> int:
        self.seek0()
        return self._num_nodes

    def num_edges(self) -> int:
        self.seek0()
        return self._num_edges

    def nodes(self) -> Iterator[Node]:
        self.seek0()
        for i, line in enumerate(self._nodes):
            if i == 0:
                continue
            if line[len(line) - 1] == "\n":
                line = line[:len(line) - 1]
            id, name = line.split(",")
            id = int(id)
            yield Node(index=id, name=name)

    def edges(self) -> Iterator[Edge]:
        self.seek0()
        for i, line in enumerate(self._edges):
            if i == 0:
                continue
            if line[len(line) - 1] == "\n":
                line = line[:len(line) - 1]
            source, destination, timestamp, weight = line.split(",")
            source = int(source)
            destination = int(destination)
            timestamp = int(timestamp)
            weight = float(weight)
            yield Edge(index=i - 1, source=source, destination=destination, timestamp=timestamp, weight=weight)
