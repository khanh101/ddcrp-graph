import numpy as np

from python.ddcrp.prior import NIW, marginal_loglikelihood


niw = NIW(2)

print(marginal_loglikelihood(niw, np.array([[1, 2], [4, 5]], dtype= np.float)))
