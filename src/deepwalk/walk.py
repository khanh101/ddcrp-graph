import random
from typing import List
import networkx as nx

def random_walk(g: nx.Graph, walks_per_node: int = 1, walk_length: int = 1) -> List[List[int]]:
    walks: List[List[int]] = []
    for start in g.nodes():
        for _ in range(walks_per_node):
            path: List[int] = [start,]
            for _ in range(walk_length):
                next = random.choice(list(g.neighbors(path[-1])))
                path.append(next)
            walks.append(path)
    return walks

