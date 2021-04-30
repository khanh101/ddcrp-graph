import networkx as nx

from dataset.loader import Loader


def load_data(dataset_name: str = "email") -> nx.MultiGraph:
    loader = Loader(dataset_name=dataset_name)
    g: nx.MultiGraph = nx.MultiGraph()
    for node in loader.nodes():
        g.add_node(node.index, name=node.name)
    for edge in loader.edges():
        g.add_edge(edge.source, edge.target, weight=edge.weight, timestamp=edge.timestamp)
    return g
