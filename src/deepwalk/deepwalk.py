import os
from typing import List

import numpy as np

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
        print(f"deepwalk created with {workers} workers")

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

