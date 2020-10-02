from typing import List

import numpy as np
from gensim.models import Word2Vec as Word2VecModel


def Word2Vec(num_nodes: int, walks: List[List[int]], dim: int = 1, context: int = 1, num_iterations: int = 1) -> List[
    np.ndarray]:
    walks: List[List[str]] = [[str(node) for node in walk] for walk in walks]
    model = Word2VecModel(
        sentences=walks,
        size=dim,
        window=context,
        min_count=0,
        sg=1,
        workers=4,
        iter=num_iterations,
    )
    vector_list: List[np.ndarray] = [model.wv[str(node)] for node in range(num_nodes)]
    return vector_list
