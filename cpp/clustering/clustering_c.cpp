#include <map>
#include <iostream>
#include "clustering_c.h"
#include "ddcrp/prior.h"
#include "ddcrp/ddcrp.h"

void clustering_c(uint64 seed, uint64 num_iterations, uint64 num_nodes, uint64 dimension, const float64 *embedding,
                  float64 logalpha, uint64 num_edges, const uint64 *adj_row, const uint64 *adj_col, const float64 *adj_logdecay,
                  uint64 *cluster_assignment) {
    auto data = load_data(dimension, num_nodes, embedding);
    auto adj_list = std::vector<std::map<uint64, float64>>(num_nodes);
    for (uint64 e=0; e<num_edges; e++) {
        auto source = adj_col[e];
        auto target = adj_row[e];
        auto logdecay = adj_logdecay[e];
        adj_list[source].insert(std::make_pair(target, logdecay));
    }
    auto niw = NIW(dimension);
    auto ddcrp = Assignment(num_nodes);


    auto logdecay = [&](uint64 customer) -> std::map<uint64, float64> {
        return adj_list[customer];
    };
    auto loglikelihood = [&](const std::vector<uint64>& customer_list) -> float64 {
        return niw.marginal_loglikelihood(data, customer_list);
    };

    auto result = std::vector<std::vector<uint64>>();
    for (auto iter = 0; iter < num_iterations; iter++) {
        ddcrp_iterate(
                math::UnitRNG(seed),
                ddcrp,
                logalpha,
                logdecay,
                loglikelihood
        );
        result.push_back(ddcrp.table_assignment());
        std::cout << "iter: " << iter << "/" << num_iterations << std:: endl;
    }
    // process result
    auto out = result.back();
    //
    std::memcpy(cluster_assignment, out.data(), num_nodes * sizeof(uint64));
}
