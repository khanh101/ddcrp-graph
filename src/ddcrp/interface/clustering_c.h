#ifndef DDCRP_CLUSTERING_C_H
#define DDCRP_CLUSTERING_C_H

#include "../core/common.h"

extern "C" {
    uint64 new_state(
            uint64 seed,
            uint64 num_nodes,
            uint64 dimension
    );
    void del_state(uint64 state_ptr);

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


    void clustering_c(
            uint64 seed, // random seed
            uint64 num_iterations, // num_iterations
            uint64 num_nodes, uint64 dimension, // num_nodes, embedding dimension
            const float64* embedding, // (d x n) col major matrix
            float64 logalpha, // logalpha
            uint64 num_edges, // num edges
            const uint64* adj_row, // coo_matrix for logdecay
            const uint64* adj_col, //
            const float64* adj_logdecay, //
            uint64* cluster_assignment // output assignment: preallocated
    );
};



#endif //DDCRP_CLUSTERING_C_H
