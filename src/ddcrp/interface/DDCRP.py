import numpy as np
import scipy as sp
import scipy.sparse

from src.ddcrp.interface.clustering import new_state, del_state, iterate_state


class DDCRP(object):
    state: int

    def __init__(self, seed: int, num_nodes: int, dimension: int):
        super(DDCRP, self).__init__()
        self.state = new_state(seed, num_nodes, dimension)

    def __del__(self):
        del_state(self.state)

    def iterate(self,
                num_iterations: int,
                embedding: np.ndarray,
                logalpha: float,
                adj: sp.sparse.coo_matrix
                ) -> np.ndarray:
        return iterate_state(
            self.state,
            num_iterations,
            embedding,
            logalpha,
            adj,
        )
