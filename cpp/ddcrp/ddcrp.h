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
const Table table_nil = -1;

class Assignment {
public:
    explicit Assignment(Customer num_customers);

    ~Assignment() = default;

    void unlink(Customer source);

    void link(Customer source, Customer target);

    [[nodiscard]] Customer num_customers() const;

    [[nodiscard]] std::set<Customer> table(Customer customer) const;

private:
    struct Node {
        Customer m_parent;
        std::set<Customer> m_children;

        Node();

        ~Node() = default;
    };

    uint64 m_num_customers;
    std::vector<Node> m_adjacency_list;
    std::map<Table, std::set<Customer>> m_table_list;
    std::vector<Table> m_table_assignment;
    uint64 m_table_count;

    std::set<Customer> weakly_connected_component(Customer customer);

};

Assignment::Node::Node() :
        m_parent(customer_nil), m_children() {}

Assignment::Assignment(Customer num_customers) :
//init default, each customer sits in one table
//:param num_customers: number of customers
        m_num_customers(num_customers),
        m_adjacency_list(),
        m_table_list(),
        m_table_assignment(),
        m_table_count(num_customers) {
    for (Customer customer = 0; customer < num_customers; customer++) {
        m_adjacency_list.emplace_back();
        m_table_list.emplace(customer, std::set<Customer>({customer}));
        m_table_assignment.push_back(customer);
    }
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
        auto node = m_adjacency_list[current];
        auto adding = node.m_children;
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

void Assignment::unlink(Customer source) {
    if (source == customer_nil) {
        return;
    }
    auto target = m_adjacency_list[source].m_parent;
    if (target == customer_nil) {
        return;
    }
    // remove link
    m_adjacency_list[target].m_children.erase(source);
    m_adjacency_list[source].m_parent = customer_nil;
    // update table
    auto new_source_component = weakly_connected_component(source);
    if (not new_source_component.contains(target)) { // split
        auto new_target_component = weakly_connected_component(target);
        // add new two compoents
        auto new_target_table = m_table_count;
        auto new_source_table = m_table_count + 1;
        m_table_count += 2;
        m_table_list.insert(std::make_pair(new_source_table, new_source_component));
        m_table_list.insert(std::make_pair(new_target_table, new_target_component));
        // remove old compoenent
        auto old_join_table = m_table_assignment[source];
        m_table_list.erase(old_join_table);
        // update assignment
        for (auto new_source_customer: new_source_component) {
            m_table_assignment[new_source_customer] = new_source_table;
        }
        for (auto new_target_customer: new_target_component) {
            m_table_assignment[new_target_customer] = new_target_table;
        }
    }
}

void Assignment::link(Customer source, Customer target) {
    // add link
    m_adjacency_list[source].m_parent = target;
    m_adjacency_list[target].m_children.insert(source);
    // update table
    auto old_source_table = m_table_assignment[source];
    auto old_source_component = m_table_list[old_source_table];
    if (not old_source_component.contains(target)) { // join
        auto old_target_table = m_table_assignment[target];
        auto old_target_component = m_table_list[old_target_table];
        // add new join component
        auto new_join_component = std::set<Customer>();
        new_join_component.insert(old_source_component.begin(), old_source_component.end());
        new_join_component.insert(old_target_component.begin(), old_target_component.end());
        auto new_join_table = m_table_count;
        m_table_count += 1;
        m_table_list.insert(std::make_pair(new_join_table, new_join_component));
        // remove old component
        m_table_list.erase(old_source_table);
        m_table_list.erase(old_target_table);
        // update assignment
        for (auto new_join_customer: new_join_component) {
            m_table_assignment[new_join_customer] = new_join_table;
        }
    }
}

Customer Assignment::num_customers() const {
    return m_num_customers;
}

std::set<Customer> Assignment::table(Customer customer) const {
    return m_table_list.find(m_table_assignment[customer])->second;
}

template<typename UnitRNG>
void ddcrp_iterate(
        UnitRNG gen,
        Assignment &assignment,
        float64 logalpha, // log(alpha)
        const std::vector<std::map<Customer, float64>> &logdecay_func, // logdecay = logdecay_func[customer1][customer2]
        const std::function<float64(
                const std::set<Customer> &customer_list)> &loglikelihood_func // loglikelihood of a compoentn
) {
    auto target_list = std::vector<Customer>();
    auto logweight_list = std::vector<float64>();
    target_list.reserve(assignment.num_customers());
    logweight_list.reserve(assignment.num_customers());
    for (Customer source = 0; source < assignment.num_customers(); source++) {
        auto &logdecay_map = logdecay_func[source];
        target_list.clear();
        for (auto it = logdecay_map.begin(); it != logdecay_map.end(); ++it) {
            target_list.push_back(it->first);
        }
        if (not logdecay_map.contains(source)) {
            target_list.push_back(source);
        }
        logweight_list.clear();
        logweight_list.resize(target_list.size(), -std::numeric_limits<float64>::infinity());
        assignment.unlink(source);
        auto source_component = assignment.table(source);
        for (uint64 i = 0; i < target_list.size(); i++) {
            Customer target = target_list[i];
            if (target == source) {
                // self loop
                logweight_list[i] = logalpha;
                continue;
            }
            auto logdecay = logdecay_map.find(target)->second;
            if (source_component.contains(target)) {
                // no table join
                logweight_list[i] = logdecay;
                continue;
            }
            // table join
            auto target_component = assignment.table(target);
            auto source_loglikehood = loglikelihood_func(source_component);
            auto target_loglikehood = loglikelihood_func(target_component);
            auto join_component = std::set<Customer>();
            join_component.insert(source_component.begin(), source_component.end());
            join_component.insert(target_component.begin(), target_component.end());
            auto join_loglikehood = loglikelihood_func(join_component);
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


#endif //DDCRP_GIBBS_DDCRP_H
