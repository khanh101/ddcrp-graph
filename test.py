from dataset.loader import Loader
from draw import draw_size
import numpy as np
import scipy as sp
import scipy.sparse
g = Loader()
num_nodes = g.num_nodes()
row = []
col = []
data = []

timestamp_list = []
for edge in g.edges():
    timestamp_list.append(edge.timestamp)
    col.append(edge.source)
    row.append(edge.target)
    data.append(edge.weight)

row = np.array(row)
col = np.array(col)
data = np.array(data)

a = sp.sparse.coo_matrix((data, (row, col)), shape= num_nodes)

pass