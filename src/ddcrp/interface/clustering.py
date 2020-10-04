import ctypes
import os
from typing import List, Set

import numpy as np
import scipy as sp
import scipy.sparse

lib: ctypes.CDLL = None
def loadLibrary(path: str = "./cmake-build-release/libddcrp.so"):
    global lib
    lib = ctypes.cdll.LoadLibrary(path)
'''
uint64 new_state(
        uint64 seed,
        uint64 num_nodes,
        uint64 dimension
);
'''
def new_state(seed: int, num_nodes: int, dimension: int) -> int:
    if lib is None:
        loadLibrary()
    seed: ctypes.c_uint64 = ctypes.c_uint64(seed)
    num_nodes: ctypes.c_uint64 = ctypes.c_uint64(num_nodes)
    dimension: ctypes.c_uint64 = ctypes.c_uint64(dimension)
    state = int(lib.new_state(seed, num_nodes, dimension))
    return state
'''
    void del_state(uint64 state_ptr);
'''
def del_state(state: int):
    state: ctypes.c_uint64 = ctypes.c_uint64(state)
    lib.del_state(state)
'''
void iterate_state(
        uint64 state_ptr,
        uint64 num_iterations,
        const float64* embedding,
        float64 logalpha,
        uint64 num_edges, // num edges
        const uint64* adj_row, // coo_matrix for logdecay
        const uint64* adj_col, //
        const float64* adj_logdecay, //
        uint64* cluster_assignment // output assignment: preallocated
);
'''
def iterate_state(
            state: int,
            num_iterations: int,
            embedding: np.ndarray,
            logalpha: float,
            adj: sp.sparse.coo_matrix
    ) -> np.ndarray:
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
    cluster_assignment: np.ndarray = np.ascontiguousarray(np.zeros((num_iterations, embedding.shape[0]), dtype=np.uint64))
    # prepare
    state: ctypes.c_uint64 = ctypes.c_uint64(state)
    num_iterations: ctypes.c_uint64 = ctypes.c_uint64(num_iterations)
    embedding_ptr: ctypes.POINTER(ctypes.c_double) = embedding.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    logalpha: ctypes.c_double = ctypes.c_double(logalpha)
    num_edges: ctypes.c_uint64 = ctypes.c_uint64(len(adj_logdecay))
    adj_row_ptr: ctypes.POINTER(ctypes.c_uint64) = adj_row.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    adj_col_ptr: ctypes.POINTER(ctypes.c_uint64) = adj_col.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    adj_logdecay_ptr: ctypes.POINTER(ctypes.c_double) = adj_logdecay.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cluster_assignment_ptr: ctypes.POINTER(ctypes.c_uint64) = cluster_assignment.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    lib.iterate_state(state, num_iterations, embedding_ptr, logalpha, num_edges, adj_row_ptr, adj_col_ptr, adj_logdecay_ptr, cluster_assignment_ptr)

    return cluster_assignment






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
) -> np.ndarray:
    if lib is None:
        loadLibrary()
    """
    :param seed: random seed
    :param num_iterations: num_iterations
    :param embedding: embedding matrix (n x d)
    :param logalpha: log(alpha)
    :param adj: adjacency matrix
    :return: clustering
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
    cluster_assignment: np.ndarray = np.ascontiguousarray(np.zeros((num_iterations, embedding.shape[0]), dtype=np.uint64))
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
    #lib.clustering_c(seed, num_iterations, num_nodes, dimension, embedding_ptr, logalpha, num_edges, adj_row_ptr, adj_col_ptr, adj_logdecay_ptr, cluster_assignment_ptr)
    state = lib.new_state(seed, num_nodes, dimension)
    state_ptr = ctypes.c_uint64(state)
    lib.iterate_state(state_ptr, num_iterations, embedding_ptr, logalpha, num_edges, adj_row_ptr, adj_col_ptr, adj_logdecay_ptr, cluster_assignment_ptr)
    lib.del_state(state_ptr)

    return cluster_assignment
