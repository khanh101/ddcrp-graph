import math

import numpy as np
import scipy as sp
import scipy.special

class NIW(object):
    dim: int
    m: np.ndarray
    k: float
    v: int
    S: np.ndarray
    def __init__(self, dim: int):
        super(NIW, self).__init__()
        self.dim = dim
        self.m = np.zeros(shape=(1, dim))
        self.k = 0.01
        self.v = dim + 2
        #self.S = np.zeros(shape=(dim, dim))
        self.S = np.identity(dim)

    def posterior(self, data: np.ndarray): # shape (n x d)
        n, d = data.shape
        data_mean = data.mean(axis=0).reshape((1, d))
        out = NIW(self.dim)
        out.m = (self.k / (self.k + n)) * self.m + (n / (self.k + n)) * data_mean
        out.k = self.k + n
        out.v = self.v + n
        S = sum([data[i:i+1, :].T.__matmul__(data[i:i+1, :]) for i in range(n)])
        out.S = self.S + S + self.k * self.m.T.__matmul__(self.m) - out.k * out.m.T.__matmul__(out.m)
        return out

def marginal_loglikelihood(prior: NIW, data: np.ndarray) -> float:
    n, d = data.shape
    posterior = prior.posterior(data)
    out: float = - (n * d / 2) * np.log(np.pi)
    out += sp.special.multigammaln(posterior.v / 2, d) - sp.special.multigammaln(prior.v / 2, d)
    out += (prior.v / 2) * np.linalg.slogdet(prior.S)[1] - (posterior.v / 2) * np.linalg.slogdet(posterior.S)[1]
    out += (d / 2) * (np.log(prior.k) - np.log(posterior.k))
    return out


if __name__ == "__main__":
    data = np.random.random(size=(200, 50))
    prior = NIW(50)
    while True:
        print(marginal_loglikelihood(prior, data))
        prior = prior.posterior(data)
