import ctypes
import os
from typing import List, Set

import numpy as np
import scipy as sp
import scipy.sparse

lib: ctypes.CDLL = None
def loadLibrary(path: str = "../cmake-build-debug/libddcrp_gibbs.so"):
    global lib
    lib = ctypes.cdll.LoadLibrary(path)

'''
void clustering_c(
    uint64 seed, // random seed
    uint64 num_iterations, // num_iterations
    uint64 num_nodes, uint64 dimension, // num_nodes, embedding dimension
    const float64* embedding, // (d x n) col major matrix
    float64 logalpha, // logalpha
    uint64 num_edges, // num edges
    const uint64* adj_row, // coo_matrix for logdecay
    const uint64* adj_col, //
    const float* adj_logdecay, //
    uint64* cluster_assignment // output assignment: preallocated
);
'''
def clustering(
        seed: int,
        num_iterations: int,
        embedding: np.ndarray,
        logalpha: float,
        adj: sp.sparse.coo_matrix
) -> List[Set[int]]:
    if lib is None:
        loadLibrary()
    """
    :param seed: random seed
    :param num_iterations: num_iterations
    :param embedding: embedding matrix (n x d)
    :param logalpha: log(alpha)
    :param adj: adjacency matrix
    :return: List[Set[int]] clustering
    """
    embedding = embedding.astype(np.float64)
    if not embedding.flags.c_contiguous:
        embedding = np.ascontiguousarray(embedding) # (n x d) row major == (d x n) col major
    adj_row: np.ndarray = adj.row.astype(np.uint64)
    if not adj.row.flags.c_contiguous:
        adj_row = np.ascontiguousarray(adj_row)
    adj_col: np.ndarray = adj.col.astype(np.uint64)
    if not adj.col.flags.c_contiguous:
        adj_col = np.ascontiguousarray(adj_col)
    adj_logdecay: np.ndarray = adj.data.astype(np.float64)
    if not adj.data.flags.c_contiguous:
        adj_logdecay = np.ascontiguousarray(adj_logdecay)
    cluster_assignment: np.ndarray = np.ascontiguousarray(np.zeros(embedding.shape[0]))
    # prepare
    seed: ctypes.c_uint64 = ctypes.c_uint64(seed)
    num_iterations: ctypes.c_uint64 = ctypes.c_uint64(num_iterations)
    num_nodes: ctypes.c_uint64 = ctypes.c_uint64(embedding.shape[0])
    dimension: ctypes.c_uint64 = ctypes.c_uint64(embedding.shape[1])
    embedding_ptr: ctypes.POINTER(ctypes.c_double) = embedding.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    logalpha: ctypes.c_double = ctypes.c_double(logalpha)
    num_edges: ctypes.c_uint64 = ctypes.c_uint64(len(adj_logdecay))
    adj_row_ptr: ctypes.POINTER(ctypes.c_uint64) = adj_row.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    adj_col_ptr: ctypes.POINTER(ctypes.c_uint64) = adj_col.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    adj_logdecay_ptr: ctypes.POINTER(ctypes.c_double) = adj_logdecay.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cluster_assignment_ptr: ctypes.POINTER(ctypes.c_uint64) = cluster_assignment.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    # call
    lib.clustering_c(seed, num_iterations, num_nodes, dimension, embedding_ptr, logalpha, num_edges, adj_row_ptr, adj_col_ptr, adj_logdecay_ptr, cluster_assignment_ptr)

    # get reuslt
    out: List[Set[int]] = []
    for label in np.unique(cluster_assignment):
        s = set()
        for i in range(len(cluster_assignment)):
            if cluster_assignment[i] == label:
                s.add(i)
        out.append(s)

    return out