#include <map>
#include "clustering_c.h"
#include "ddcrp/prior.h"
#include "ddcrp/ddcrp.h"
void clustering_c(uint64 seed, uint64 num_iterations, uint64 num_nodes, uint64 dimension, const float64 *embedding,
                  float64 logalpha, uint64 num_edges, const uint64 *adj_rows, const uint64 *adj_cols, const uint64 *adj_logdecay,
                  uint64 *cluster_assignment) {
    auto data = load_data(dimension, num_nodes, embedding);
    auto adj_list = std::vector<std::map<uint64, float64>>(num_nodes);
    for (uint64 e=0; e<num_edges; e++) {
        auto source = adj_cols[e];
        auto target = adj_rows[e];
        auto logdecay = adj_logdecay[e];
        adj_list[source].insert(std::make_pair(target, logdecay));
    }
    auto niw = NIW(dimension);
    auto ddcrp = Assignment(num_nodes);

    auto logdecay = [&](uint64 customer) -> std::map<uint64, float64> {
        return adj_list[customer];
    };
    auto loglikelihood = [&](const std::set<uint64>& customer_list) -> float64 {
        return niw.marginal_loglikelihood(data, std::vector<uint64>(customer_list.begin(), customer_list.end()));
    };

    auto result = std::vector<std::vector<std::set<uint64>>>();
    for (auto iter = 0; iter < num_iterations; iter++) {
        ddcrp_iterate(
                math::UnitRNG(seed),
                ddcrp,
                logalpha,
                logdecay,
                loglikelihood
        );
        result.push_back(ddcrp.table_list());
    }
    // process result
    auto out = result.back();
    //
    for (uint64 label = 0; label < out.size(); label++) {
        for (uint64 node: out[label]) {
            cluster_assignment[node] = label;
        }
    }


}
