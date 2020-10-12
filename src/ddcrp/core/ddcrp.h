//
// Created by khanh on 1/10/20.
//

#ifndef DDCRP_DDCRP_H
#define DDCRP_DDCRP_H

#include <map>
#include <vector>
#include <set>
#include <random>
#include <list>
#include "common.h"
#include "math.h"
#include "ds.h"
#include "assignment.h"

//template<typename UnitRNG>
void ddcrp_iterate(
        std::function<float64()> gen,
        Assignment &assignment,
        float64 logalpha, // log(alpha)
        const std::function<std::map<Customer, float64>(
                Customer)> &logdecay_func, // logdecay = logdecay_func[customer1][customer2]
        const std::function<float64(const std::vector<Customer> &)> &loglikelihood_func // loglikelihood of a compoentn
);

void ddcrp_iterate(
        std::function<float64()> gen,
        Assignment &assignment,
        float64 logalpha, // log(alpha)
        const std::function<std::map<Customer, float64>(
                Customer)> &logdecay_func, // logdecay = logdecay_func[customer1][customer2]
        const std::function<float64(const std::vector<Customer> &)> &loglikelihood_func // loglikelihood of a compoentn
) {
    auto target_list = std::vector<Customer>();
    auto logweight_list = std::vector<float64>();
    target_list.reserve(assignment.num_customers());
    logweight_list.reserve(assignment.num_customers());
    for (Customer source = 0; source < assignment.num_customers(); source++) {
        auto logdecay_map = logdecay_func(source);
        target_list.clear();
        for (auto it = logdecay_map.begin(); it != logdecay_map.end(); ++it) {
            target_list.push_back(it->first);
        }
        if (not (logdecay_map.find(source) != logdecay_map.end())) {
            target_list.push_back(source);
        }
        logweight_list.clear();
        logweight_list.resize(target_list.size(), -std::numeric_limits<float64>::infinity());
        assignment.unlink(source);
        auto source_component = assignment.component(source);
        #pragma omp parallel for
        for (uint64 i = 0; i < target_list.size(); i++) {
            Customer target = target_list[i];
            if (target == source) {
                // self loop
                logweight_list[i] = logalpha;
                continue;
            }
            auto logdecay = logdecay_map.find(target)->second;
            if (assignment.table(source) == assignment.table(target)) {
                // no table join
                logweight_list[i] = logdecay;
                continue;
            }
            // table join
            auto target_component = assignment.component(target);
            auto source_loglikehood = loglikelihood_func(source_component);
            auto target_loglikehood = loglikelihood_func(target_component);
            auto join_component = std::vector<Customer>();
            join_component.reserve(source_component.size() + target_component.size());
            join_component.insert(join_component.end(), source_component.begin(), source_component.end());
            join_component.insert(join_component.end(), target_component.begin(), target_component.end());
            auto join_loglikehood = loglikelihood_func(join_component);
            //std::cout << join_loglikehood << " " << source_loglikehood << " " << target_loglikehood << std::endl;
            logweight_list[i] = logdecay + join_loglikehood - source_loglikehood - target_loglikehood;
            // update source component
        }
        float64 max_logweight = logweight_list[0];
        for (auto logweight: logweight_list) {
            if (logweight > max_logweight) {
                max_logweight = logweight;
            }
        }
        auto &weight_list = logweight_list;
        for (uint64 i = 0; i < logweight_list.size(); i++) {
            weight_list[i] = math::exp(logweight_list[i] - max_logweight);
        }

        Customer target = target_list[math::discrete_sampling(gen, weight_list)];
        assignment.link(source, target);
    }
}

#endif //DDCRP_DDCRP_H
