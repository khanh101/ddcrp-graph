import random
from typing import List, Generator, Iterator, Set, Union

import networkx as nx
import numpy as np

from algorithm.clustering.clustering import label_to_comm
from dataset import Graph, Node, Edge


class Generator(Graph):
    _nodes: List[Node]
    _edges: List[Edge]
    _communities: List[Set[int]]

    def __init__(self, cluster_size: Union[List[int], np.ndarray], cluster_prob: Union[List[List[float]], np.ndarray],
                 gamma: float = 1, directed: bool = True, weakly_connected: bool = False):
        super(Generator, self).__init__()
        self._nodes = []
        self._edges = []
        # NODE
        num_nodes = sum(cluster_size)
        for node in range(num_nodes):
            self._nodes.append(Node(
                index=node,
                name=f"node-{node}",
            ))

        # PREPARATION
        cluster_label = np.empty(shape=(num_nodes,), dtype=np.int)
        node_idx = 0
        for label, size in enumerate(cluster_size):
            for _ in range(size):
                cluster_label[node_idx] = label
                node_idx += 1

        self._communities = label_to_comm(cluster_label)

        degree_prob = np.random.uniform(size=(num_nodes,)) ** gamma
        degree_prob /= np.mean(degree_prob)

        # NX GRAPH
        if directed:
            g = nx.DiGraph()
            for i in range(num_nodes):
                g.add_node(i)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # ignore self-loop
                        if degree_prob[i] * degree_prob[j] * cluster_prob[cluster_label[i]][
                            cluster_label[j]] > np.random.uniform():
                            g.add_edge(i, j)

            if weakly_connected:
                components = list(map(lambda x: list(x), nx.weakly_connected_components(g)))
                components.sort(key=len)
                giant_components = components[-1]

                for i in range(len(components) - 1):
                    component = components[i]
                    node1 = random.choice(component)
                    node2 = np.random.choice(giant_components)
                    if np.random.random() < 0.5:  # 50% swap
                        node1, node2 = node2, node1
                    g.add_edge(node1, node2)

            for i, j in g.edges():
                self._edges.append(Edge(
                    index=None,
                    source=i,
                    destination=j,
                    timestamp=None,
                    weight=1,
                ))
        else:
            g = nx.Graph()
            for i in range(num_nodes):
                g.add_node(i)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if degree_prob[i] * degree_prob[j] * cluster_prob[cluster_label[i]][
                        cluster_label[j]] > np.random.uniform():
                        g.add_edge(i, j)

            if weakly_connected:
                components = list(map(lambda x: list(x), nx.connected_components(g)))
                components.sort(key=len)
                giant_components = components[-1]

                for i in range(len(components) - 1):
                    component = components[i]
                    node1 = random.choice(component)
                    node2 = np.random.choice(giant_components)
                    g.add_edge(node1, node2)

            for i, j in g.edges():
                self._edges.append(Edge(
                    index=None,
                    source=i,
                    destination=j,
                    timestamp=None,
                    weight=1,
                ))
                self._edges.append(Edge(
                    index=None,
                    source=j,
                    destination=i,
                    timestamp=None,
                    weight=1,
                ))

        random.shuffle(self._edges)
        for i in range(len(self._edges)):
            self._edges[i].index = i
            self._edges[i].timestamp = i

    def num_nodes(self) -> int:
        return len(self._nodes)

    def num_edges(self) -> int:
        return len(self._edges)

    def nodes(self) -> Iterator[Node]:
        for node in self._nodes:
            yield node

    def edges(self) -> Iterator[Edge]:
        for edge in self._edges:
            yield edge

    def communities(self) -> List[Set[int]]:
        return self._communities
