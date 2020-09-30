from typing import Any

import numpy as np
import scipy as sp
import scipy.special

class NIW(object):
    """
    Normal-Inverse-Wishart Prior Hyperparameters
    """
    dim: int
    m: np.ndarray
    k: float
    v: int
    S: np.ndarray
    # precompute
    __gamma_d_v_2: float
    __logdet_S: float
    __log_k: float
    def gamma_d_v_2(self) -> float:
        if self.__gamma_d_v_2 is None:
            self.__gamma_d_v_2 = sp.special.multigammaln(self.v / 2, self.dim)
        return self.__gamma_d_v_2

    def logdet_S(self) -> float:
        if self.__logdet_S is None:
            self.__logdet_S = np.linalg.slogdet(self.S)[1]
        return self.__logdet_S

    def log_k(self) -> float:
        if self.__log_k is None:
            self.__log_k = np.log(self.k)
        return self.__log_k

    def __init__(self, dim: int):
        """
        init the param
        Murphy: Machine learning - a probabilistic perspective
        Sect. 4.6.3.2
        :param dim: dimension
        """
        super(NIW, self).__init__()
        self.dim = dim
        self.m = np.zeros(shape=(1, dim))
        self.k = 0.01
        self.v = dim + 2
        self.S = np.identity(dim)
        # precompute
        self.__gamma_d_v_2 = None
        self.__logdet_S = None
        self.__log_k = None

    def posterior(self, data: np.ndarray) -> Any:
        """
        Murphy: Machine learning - a probabilistic perspective
        Sect. 4.6.3.3
        :param data: (n x d) array
        :return:
        """
        n, d = data.shape
        data_mean = data.mean(axis=0).reshape((1, d))
        out = NIW(self.dim)
        out.m = (self.k / (self.k + n)) * self.m + (n / (self.k + n)) * data_mean
        out.k = self.k + n
        out.v = self.v + n
        S = sum([data[i:i+1, :].T.__matmul__(data[i:i+1, :]) for i in range(n)])
        out.S = self.S + S + self.k * self.m.T.__matmul__(self.m) - out.k * out.m.T.__matmul__(out.m)
        return out


log_pi: float = np.log(np.pi)
def marginal_loglikelihood(prior: NIW, data: np.ndarray) -> float:
    """
    Ref. https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf page 21: Marginal likelihood
    :param prior: NIW prior
    :param data: (n x d) array
    :return: marginal loglikelihood log(P(data))
    """
    n, d = data.shape
    posterior = prior.posterior(data)
    out: float = - (n * d / 2) * log_pi
    #out += sp.special.multigammaln(posterior.v / 2, d) - sp.special.multigammaln(prior.v / 2, d)
    out += posterior.gamma_d_v_2() - prior.gamma_d_v_2()
    #out += (prior.v / 2) * np.linalg.slogdet(prior.S)[1] - (posterior.v / 2) * np.linalg.slogdet(posterior.S)[1]
    out += (prior.v / 2) * prior.logdet_S() - (posterior.v / 2) * posterior.logdet_S()
    #out += (d / 2) * (np.log(prior.k) - np.log(posterior.k))
    out += (d / 2) * (prior.log_k() - posterior.log_k())
    return out


if __name__ == "__main__":
    data = np.random.random(size=(200, 50))
    prior = NIW(50)
    while True:
        print(marginal_loglikelihood(prior, data))
        prior = prior.posterior(data)
