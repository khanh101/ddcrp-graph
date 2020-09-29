import math
import random
from typing import Set, List, Dict, Callable

Customer = int
Table = int

customer_nil: Customer = -1


class Assignment(object):
    class Node(object):
        parent: Customer
        children: Set[Customer]

        def __init__(self):
            super(Assignment.Node, self).__init__()
            self.parent = customer_nil
            self.children = set()

    num_customers: Customer
    graph: List[Node]
    table2customer: Dict[Table, Set[Customer]]
    customer2table: List[Table]
    table_count: Table

    def __init__(self, num_customers: Customer):
        super(Assignment, self).__init__()
        self.num_customers = num_customers
        self.graph = [Assignment.Node() for _ in range(num_customers)]
        self.customer2table = [customer for customer in range(num_customers)]
        self.table2customer = {}
        for customer in range(num_customers):
            self.table2customer[customer] = {customer}
        self.table_count = num_customers

    def weakly_connected_component(self, customer: Customer) -> Set[Customer]:
        subtree_node_set: Set[Customer] = set()
        frontier: List[Customer] = [customer]
        while len(frontier) > 0:
            current, frontier = frontier[-1], frontier[:-1]
            subtree_node_set.add(current)
            adding = [self.graph[current].parent] + list(self.graph[current].children)
            for node in adding:
                if node != customer_nil and node not in frontier and node not in subtree_node_set:
                    frontier.append(node)
        return subtree_node_set


    def unlink(self, customer: Customer):
        if self.graph[customer].parent != customer_nil:
            # remove link
            self.graph[self.graph[customer].parent].children.remove(customer)
            self.graph[customer].parent = customer_nil
            # find weakly connected component of customer
            component: Set[Customer] = self.weakly_connected_component(customer)
            # remove from prev table
            prev_table: Table = self.customer2table[customer]
            for node in component:
                self.table2customer[prev_table].remove(node)
            if len(self.table2customer[prev_table]) == 0:
                self.table2customer.pop(prev_table)
            # add to next table
            next_table: Table = self.table_count
            self.table_count += 1
            self.table2customer[next_table] = component
            # update table label
            for node in component:
                self.customer2table[node] = next_table


    def link(self, source: Customer, target: Customer):
        source_component: Set[Customer] = self.weakly_connected_component(source)
        if target in source_component:
            self.graph[source].parent = target
            self.graph[target].children.add(source)
        else: # join
            target_component: Set[Customer] = self.weakly_connected_component(target)
            self.graph[source].parent = target
            self.graph[target].children.add(source)
            # remove 2 old tables
            source_table: Table = self.customer2table[source]
            target_table: Table = self.customer2table[target]
            self.table2customer.pop(source_table)
            self.table2customer.pop(target_table)
            new_table: Table = self.table_count
            self.table_count += 1
            new_component: Set[Customer] = set(list(source_component) + list(target_component))
            for customer in new_component:
                self.customer2table[customer] = new_table
            self.table2customer[new_table] = new_component



class DDCRP(object):
    num_customers: Customer
    assignment: Assignment
    logalpha: float
    logdecay_func: Callable[[Customer, Customer], float]
    loglikelihood_func: Callable[[Set[Customer]], float]
    def __init__(self,
                 num_customers: Customer,
                 logalpha: float,
                 logdecay_func: Callable[[Customer, Customer], float],
                 loglikelihood_func: Callable[[Set[Customer]], float],
        ):
        super(DDCRP, self).__init__()
        self.num_customers = num_customers
        self.assignment = Assignment(num_customers=num_customers)
        self.logalpha = logalpha
        self.logdecay_func = logdecay_func
        self.loglikelihood_func = loglikelihood_func

    def iterate(self):
        for source in range(self.num_customers):
            self.assignment.unlink(source)
            logweight: List[float] = []
            for target in range(self.num_customers):
                if source == target: # self loop
                    logweight.append(self.logalpha)
                else:
                    source_component: Set[Customer] = self.assignment.weakly_connected_component(source)
                    logdecay: float = self.logdecay_func(source, target)
                    if target in source_component: # no join
                        logweight.append(logdecay)
                    else: # join
                        target_component: Set[Customer] = self.assignment.weakly_connected_component(target)
                        source_loglikelihood: float = self.loglikelihood_func(source_component)
                        target_loglikelihood: float = self.loglikelihood_func(target_component)
                        join_loglikelihood: float = self.loglikelihood_func(set(list(source_component) + list(target_component)))
                        logweight.append(
                            logdecay + join_loglikelihood - source_loglikelihood - target_loglikelihood,
                        )
            # sample
            max_lw: float = max(logweight)
            logweight: List[float] = [lw - max_lw for lw in logweight] # divide weight by a constant
            weight: List[float] = [math.exp(lw) for lw in logweight]
            target: Customer = random.choices(range(self.num_customers), weights=weight, k=1)[0]
            self.assignment.link(source, target)


if __name__ == "__main__":
    # test unlink
    a = Assignment(num_customers=5)
    a.graph[1].parent = 0
    a.graph[2].parent = 1
    a.graph[3].parent = 1
    a.graph[0].children.add(1)
    a.graph[1].children.add(2)
    a.graph[1].children.add(3)
    a.table_count = 2
    a.table2customer = {
        0: {0, 1, 2, 3},
        1: {4},
    }
    a.customer2table = [0, 0, 0, 0, 1]


    a.unlink(1)
    a.link(1, 4)
    a.unlink(1)
    a.link(1, 3)

    pass
