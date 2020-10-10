import os
from typing import List

import networkx as nx
import numpy as np

from src.deepwalk.walk import random_walk
from gensim.models import Word2Vec

workers: int = int(os.popen('grep -c cores /proc/cpuinfo').read())

class DeepWalk(object):
    word2vec: Word2Vec
    vocab: bool
    def __init__(self, dim: int, context: int):
        super(DeepWalk, self).__init__()
        self.word2vec = Word2Vec(
            size=dim,
            window=context,
            min_count=1, # minimal frequency count
            sg=1, # skip gram
            workers=workers,
            iter=1,
        )
        self.vocab = False

    def train(self, walks: List[List[int]], epochs: int) -> np.ndarray:
        nodes = []
        for path in walks:
            nodes.extend(path)
        num_nodes = len(set(nodes))
        sentences: List[List[str]] = [[str(node) for node in walk] for walk in walks]
        if not self.vocab:
            self.word2vec.build_vocab(sentences)
            self.vocab = True
        self.word2vec.train(
            sentences=sentences,
            total_examples=len(sentences),
            epochs=epochs,
        )
        vector_list: List[np.ndarray] = [self.word2vec.wv[str(node)] for node in range(num_nodes)]
        return np.array(vector_list)


def deepwalk(g: nx.Graph, dim: int) -> np.ndarray:
    num_nodes = a.shape[0]
    walks_per_node = 20
    walk_length = 10
    context = 5
    num_iterations = 10

    walks = random_walk(g, walks_per_node, walk_length)

    embedding_list = Word2Vec(
        num_nodes=num_nodes,
        walks=walks,
        dim=dim,
        context=context,
        num_iterations=num_iterations,
    )

    emb = np.zeros(shape=(num_nodes, dim))
    for node, emb_vec in enumerate(embedding_list):
        emb[node, :] = emb_vec

    return emb
