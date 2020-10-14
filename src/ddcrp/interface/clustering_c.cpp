#include <map>
#include <iostream>
#include "clustering_c.h"
#include "../core/prior.h"
#include "../core/ddcrp.h"

struct State {
    State(uint64 seed, uint64 num_nodes, uint64 dimension);
    ~State() = default;
    math::UnitRNG m_rng;
    Assignment m_assignment;
    NIW m_prior;
};

State::State(uint64 seed, uint64 num_nodes, uint64 dimension):
        m_rng(seed), m_assignment(num_nodes), m_prior(dimension)
{}

uint64 state_count = 0;
std::map<uint64, State*> state_mapping;

uint64 new_state(uint64 seed, uint64 num_nodes, uint64 dimension) {
    auto state =  new State(seed, num_nodes, dimension);
    auto state_ptr = state_count;
    state_mapping[state_ptr] = state;
    state_count++;
    std::cout <<"state "<< state_ptr<<" created"<<std::endl;
    return state_ptr;
}

void del_state(uint64 state_ptr) {
    auto state = state_mapping[state_ptr];
    delete state;
    state_mapping.erase(state_ptr);
    std::cout <<"state "<< state_ptr<<" deleted"<<std::endl;
}

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
) {
    auto& state = *state_mapping[state_ptr];
    std::cout <<"state "<< state_ptr<<" running"<<std::endl;
    auto& rng = state.m_rng;
    auto& niw = state.m_prior;
    auto& ddcrp = state.m_assignment;
    auto dimension = niw.dimension();
    auto num_nodes = ddcrp.num_customers();

    auto data = load_data(dimension, num_nodes, embedding);
    auto adj_list = std::vector<std::map<uint64, float64>>(num_nodes);
    for (uint64 e=0; e<num_edges; e++) {
        auto source = adj_col[e];
        auto target = adj_row[e];
        auto logdecay = adj_logdecay[e];
        adj_list[source].insert(std::make_pair(target, logdecay));
    }


    auto logdecay = [&](uint64 customer) -> std::map<uint64, float64> {
        return adj_list[customer];
    };
    auto loglikelihood = [&](const std::vector<uint64>& customer_list) -> float64 {
        return niw.marginal_loglikelihood(data, customer_list);
    };
    auto gen = [&]() -> float64 {
        return rng();
    };

    auto result = std::vector<uint64>();
    for (uint64 iter = 0; iter < num_iterations; iter++) {
        std::cout << "iter: " << iter+1 << "/" << num_iterations <<": number of tables " <<  << std:: endl;
        ddcrp_iterate(
                gen,
                ddcrp,
                logalpha,
                logdecay,
                loglikelihood
        );
        auto table = ddcrp.table_assignment();
        result.insert(result.end(), table.begin(), table.end());
    }
    // result
    std::memcpy(cluster_assignment, result.data(), result.size() * sizeof(uint64));

}
