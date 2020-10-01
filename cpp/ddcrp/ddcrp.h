//
// Created by khanh on 1/10/20.
//

#ifndef DDCRP_GIBBS_DDCRP_H
#define DDCRP_GIBBS_DDCRP_H

#include <map>
#include <vector>
#include <set>
#include <random>
#include "common.h"
#include "math.h"

using Customer = uint64;
using Table = uint64;

const Customer customer_nil = -1;

class Assignment {
public:
    explicit Assignment(Customer num_customers);

    ~Assignment() = default;

    std::set<Customer> weakly_connected_component(Customer customer);

    void unlink(Customer customer);

    void link(Customer source, Customer target);

public:
    struct Node {
        Customer m_parent;
        std::set<Customer> m_children;

        Node();
    };
    uint64 m_num_customers;
    std::vector<Node> graph;
    //std::map<Table, std::set<Customer>> table2customer;
    //std::vector<Table> customer2table;
    Table table_count;
};

Assignment::Node::Node() :
        m_parent(customer_nil), m_children() {}

Assignment::Assignment(Customer num_customers) :
//init default, each customer sits in one table
//:param num_customers: number of customers
        m_num_customers(num_customers),
        graph(),
        //table2customer(),
        //customer2table(),
        table_count(num_customers) {
    for (Customer customer = 0; customer < num_customers; customer++) {
        graph.emplace_back();
    }
    /*
    for (Customer customer = 0; customer < num_customers; customer++) {
        Table table = customer;
        table2customer.insert(std::make_pair(table, std::set({customer})));
    }
    for (Customer customer = 0; customer < num_customers; customer++) {
        Table table = customer;
        customer2table.push_back(table);
    }
     */
}

std::set<Customer> Assignment::weakly_connected_component(Customer customer) {
    std::set<Customer> visited;
    std::vector<Customer> frontier({customer});

    auto is_in_list = [](const std::vector<Customer> &list, Customer customer) -> bool {
        return std::ranges::any_of(list.begin(), list.end(), [=](Customer item) -> bool {
            return item == customer;
        });
    };
    while (not frontier.empty()) {
        auto current = frontier.back();
        frontier.pop_back();
        visited.insert(current);
        auto node = graph[current];
        std::set<Customer> adding = node.m_children;
        adding.insert(node.m_parent);
        for (Customer new_customer: adding) {
            if (new_customer != customer_nil and not is_in_list(frontier, new_customer) and
                not visited.contains(new_customer)) {
                frontier.push_back(new_customer);
            }
        }
    }
    return visited;
}

void Assignment::unlink(Customer customer) {
    if (customer == customer_nil) {
        return;
    }
    // remove link
    graph[graph[customer].m_parent].m_children.erase(customer);
    graph[customer].m_parent = customer_nil;
    /*
    // find weakly connected component
    auto component = weakly_connected_component(customer);
    // remove component from prev table
    Table prev_table = customer2table[customer];
    for (auto c_customer: component) {
        table2customer[prev_table].erase(c_customer);
    }
    if (table2customer[prev_table].empty()) {
        table2customer.erase(prev_table);
    }
    // add component to next table
    Table next_table = table_count++;
    table2customer.insert(std::make_pair(next_table, component));
    // update table label
    for (auto c_customer: component) {
        customer2table[c_customer] = next_table;
    }
     */
}

void Assignment::link(Customer source, Customer target) {
    graph[source].m_parent = target;
    graph[target].m_children.insert(source);
    /*
    auto source_component = weakly_connected_component(source);
    if (not source_component.contains(target)) {
        auto target_component = weakly_connected_component(target);
        // remove two old tables
        Table source_table = customer2table[source];
        Table target_table = customer2table[target];
        table2customer.erase(source_table);
        table2customer.erase(target_table);
        Table new_table = table_count++;
        auto &new_component = source_component;
        new_component.insert(target_component.begin(), target_component.end());
        for (auto c_customer: new_component) {
            customer2table[c_customer] = new_table;
        }
        table2customer.insert(std::make_pair(new_table, new_component));
    }
    */
}
template <typename UniformRandomNumberGenerator>
void ddcrp_iterate(
        UniformRandomNumberGenerator& gen,
        Assignment &assignment,
        float64 logalpha, // log(alpha)
        const std::vector<std::map<Customer, float64>> &logdecay_func, // logdecay = logdecay_func[customer1][customer2]
        const std::function<float64(const std::set<Customer> &customer_list)> &loglikelihood_func // loglikelihood of a compoentn
) {
    for (Customer source=0; source < assignment.m_num_customers; source++) {
        auto& logdecay_map = logdecay_func[source];
        std::vector<Customer> target_list({source});
        for (auto it=logdecay_map.begin(); it != logdecay_map.end(); it++) {
            target_list.push_back(it->first);
        }
        std::vector<float64> logweight_list(target_list.size(), 0);
        assignment.unlink(source);
        for (uint64 i=0; i<target_list.size(); i++) {
            Customer target = target_list[i];
            if (target == source) {
                // self loop
                logweight_list[i] = logalpha;
                continue;
            }
            auto source_component = assignment.weakly_connected_component(source);
            auto logdecay = logdecay_map[target];
            if (source_component.contains(target)) {
                // no table join
                logweight_list[i] = logdecay;
                continue;
            }
            // table join
            auto target_component = assignment.weakly_connected_component(target);
            auto source_loglikehood = loglikelihood_func(source_component);
            auto target_loglikehood = loglikelihood_func(target_component);
            auto join_loglikehood = loglikelihood_func(source_component + target_component);

        };
        float64 max_logweight = logweight_list[0];
        for (auto logweight: logweight_list) {
            if (logweight > max_logweight) {
                max_logweight = logweight;
            }
        }
        auto weight_list = std::vector<Customer>(logweight_list.size(), 0);
        for (uint64 i=0; i<logweight_list.size(); i++) {
            weight_list[i] = math::exp(logweight_list[i] - max_logweight);
        }
        auto dist = std::discrete_distribution<uint64>(weight_list.begin(), weight_list.end());
        Customer target = target_list[dist(gen)];
        assignment.link(source, target);
    }
}


#endif //DDCRP_GIBBS_DDCRP_H
