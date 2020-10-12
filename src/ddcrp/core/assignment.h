//
// Created by khanh on 13/10/20.
//

#ifndef FYP_ASSIGNMENT_H
#define FYP_ASSIGNMENT_H

#include <set>
#include "common.h"

using Customer = uint64;
using Table = uint64;

const Customer customer_nil = uint64_nil;
const Table table_nil = uint64_nil;

class Assignment {
public:
    explicit Assignment(Customer num_customers);

    ~Assignment() = default;

    void unlink(Customer source);

    void link(Customer source, Customer target);

    [[nodiscard]] Customer num_customers() const;

    [[nodiscard]] Table table(Customer customer) const;

    [[nodiscard]] std::vector<Customer> component(Customer customer) const;

    [[nodiscard]] std::vector<Table> table_assignment() const;

private:
    struct Node {
        Customer m_parent;
        std::set<Customer> m_children;

        Node();

        ~Node() = default;
    };

    uint64 m_num_customers;
    std::vector<Node> m_adjacency_list;
    std::vector<Table> m_customer2table;
    std::map<Table, std::set<Customer>> m_table2customer;
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
        m_customer2table(),
        m_table2customer(),
        m_table_count(num_customers) {
    for (Customer customer = 0; customer < num_customers; customer++) {
        m_adjacency_list.emplace_back();
        m_customer2table.push_back(customer);
        m_table2customer.emplace(customer, std::set<Customer>({customer}));
    }
}

std::set<Customer> Assignment::weakly_connected_component(Customer customer) {
    std::set<Customer> visited;
    std::vector<Customer> frontier({customer});

    auto is_in_list = [](const std::vector<Customer> &list, Customer customer) -> bool {
        for (Customer item: list) {
            if (item == customer) {
                return true;
            }
        }
        return false;
    };
    while (not frontier.empty()) {
        auto current = frontier.back();
        frontier.pop_back();
        visited.insert(current);
        auto node = m_adjacency_list[current];
        auto adding = node.m_children;
        adding.insert(node.m_parent);
        for (auto new_customer: adding) {
            if (new_customer != customer_nil and not is_in_list(frontier, new_customer) and
                not(visited.find(new_customer) != visited.end())) {
                frontier.push_back(new_customer);
            }
        };
    }
    return visited;
}

void Assignment::unlink(Customer source) {
    auto target = m_adjacency_list[source].m_parent;
    if (target == customer_nil) {
        return;
    }
    // remove link
    m_adjacency_list[target].m_children.erase(source);
    m_adjacency_list[source].m_parent = customer_nil;
    auto new_source_component = weakly_connected_component(source);
    if (new_source_component.find(target) != new_source_component.end()) {
        return;
    }
    // split
    // update assingment
    auto new_source_table = m_table_count;
    m_table_count++;
    for (auto new_source_customer: new_source_component) {
        m_customer2table[new_source_customer] = new_source_table;
    }
    // update table
    auto new_target_table = m_customer2table[target];
    auto& new_target_component = m_table2customer[new_target_table];
    for (auto new_source_customer: new_source_component) {
        new_target_component.erase(new_source_customer);
    }
    m_table2customer.emplace(new_source_table, new_source_component);
}

void Assignment::link(Customer source, Customer target) {
    // add link
    m_adjacency_list[source].m_parent = target;
    m_adjacency_list[target].m_children.insert(source);
    if (m_customer2table[source] == m_customer2table[target]) {
        return;
    }
    // join
    // update assingment
    auto new_join_table = m_customer2table[source];
    auto old_target_table = m_customer2table[target];
    auto& old_target_component = m_table2customer[old_target_table];
    for (auto old_target_customer: old_target_component) {
        m_customer2table[old_target_customer] = new_join_table;
    }
    // update table
    auto& new_join_component = m_table2customer[new_join_table];
    new_join_component.insert(old_target_component.begin(), old_target_component.end());
    // delete
    m_table2customer.erase(old_target_table);
}

Customer Assignment::num_customers() const {
    return m_num_customers;
}

Table Assignment::table(Customer customer) const {
    return m_customer2table[customer];
}

std::vector<Customer> Assignment::component(Customer customer) const {
    auto component = m_table2customer.find(m_customer2table[customer])->second;
    return std::vector<Customer>(component.begin(), component.end());
}

std::vector<Table> Assignment::table_assignment() const {
    return std::vector<Table>(m_customer2table);
}


#endif //FYP_ASSIGNMENT_H
