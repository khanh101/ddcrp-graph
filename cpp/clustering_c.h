#ifndef DDCRP_GIBBS_CLUSTERING_C_H
#define DDCRP_GIBBS_CLUSTERING_C_H

#include "ddcrp/common.h"

extern "C" {
    void clustering_c(
            uint64 seed, // random seed
            uint64 num_iterations, // num_iterations
            uint64 num_nodes, uint64 dimension, // num_nodes, embedding dimension
            const float64* embedding, // (d x n) col major matrix
            float64 logalpha, // logalpha
            uint64 num_edges, // num edges
            const uint64* adj_rows, // coo_matrix for logdecay
            const uint64* adj_cols, //
            const uint64* adj_logdecay, //
            uint64* cluster_assignment // output assignment: preallocated
            );
};



#endif //DDCRP_GIBBS_CLUSTERING_C_H
