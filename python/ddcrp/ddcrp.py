import math
import random
from typing import Set, List, Dict, Callable

Customer = int
Table = int

customer_nil: Customer = -1


class Assignment(object):
    """
    Assignment: Contains the graph, table assignment and customer assignment
    """
    class Node(object):
        parent: Customer
        children: Set[Customer]

        def __init__(self):
            super(Assignment.Node, self).__init__()
            self.parent = customer_nil
            self.children = set()

    num_customers: Customer
    graph: List[Node]
    '''
    table2customer: Dict[Table, Set[Customer]]
    customer2table: List[Table]
    table_count: Table
    '''
    def __init__(self, num_customers: Customer):
        """
        init default, each customer sits in one table
        :param num_customers: number of customers
        """
        super(Assignment, self).__init__()
        self.num_customers = num_customers
        self.graph = [Assignment.Node() for _ in range(num_customers)]
        '''
        self.customer2table = [customer for customer in range(num_customers)]
        self.table2customer = {}
        for customer in range(num_customers):
            self.table2customer[customer] = {customer}
        self.table_count = num_customers
        '''

    def weakly_connected_component(self, customer: Customer) -> Set[Customer]:
        """
        :param customer: a customer
        :return: weakly connected component links to that customer
        """
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
        """
        unlink a customer from its parent
        :param customer:
        :return:
        """
        if self.graph[customer].parent != customer_nil:
            # remove link
            self.graph[self.graph[customer].parent].children.remove(customer)
            self.graph[customer].parent = customer_nil
            '''
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
            '''

    def link(self, source: Customer, target: Customer):
        """
        link source customer to target customer
        :param source: source
        :param target: target
        :return:
        """
        self.graph[source].parent = target
        self.graph[target].children.add(source)
        '''
        source_component: Set[Customer] = self.weakly_connected_component(source)
        if target not in source_component:
            target_component: Set[Customer] = self.weakly_connected_component(target)
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
        '''
    def table(self) -> List[Set[Customer]]:
        visited: Set[Customer] = set()
        out: List[Set[Customer]] = []
        for customer in range(self.num_customers):
            if customer not in visited:
                component = self.weakly_connected_component(customer)
                out.append(component)
                for customer in component:
                    visited.add(customer)
        return out


class DDCRP(object):
    """
    DDCRP : Distance Dependent Chinese Restaurant Process
    """
    num_customers: Customer
    assignment: Assignment

    def __init__(self, num_customers: Customer):
        """
        :param num_customers: number of customers
        """
        super(DDCRP, self).__init__()
        self.num_customers = num_customers
        self.assignment = Assignment(num_customers=num_customers)

    def iterate(self,
                logalpha: float,
                logdecaydict_func: Callable[[Customer], Dict[Customer, float]],
                loglikelihood_func: Callable[[Set[Customer]], float],
                ):
        """
        :param logalpha: log(alpha), alpha
        :param logdecaydict_func: a function returns a dict of target and the associate non(-inf) logdecay
        :param loglikelihood_func: a function returns marginal log likelihood of a table
        :return:
        """
        for source in range(self.num_customers):
            logdecay_dict: Dict[Customer, float] = logdecaydict_func(source)
            self.assignment.unlink(source)
            target_list: List[Customer] = [source] + list(logdecay_dict.keys())
            logweight_list: List[float] = []
            # only calculate probablities on reachable customers (defined logdecay)
            for target in target_list:
                if source == target:  # self loop
                    logweight_list.append(logalpha)
                else:
                    source_component: Set[Customer] = self.assignment.weakly_connected_component(source)
                    logdecay: float = logdecay_dict[target]
                    if target in source_component:  # no table join
                        logweight_list.append(logdecay)
                    else:  # table join
                        target_component: Set[Customer] = self.assignment.weakly_connected_component(target)
                        source_loglikelihood: float = loglikelihood_func(source_component)
                        target_loglikelihood: float = loglikelihood_func(target_component)
                        join_loglikelihood: float = loglikelihood_func(
                            set(list(source_component) + list(target_component))
                        )
                        logweight_list.append(
                            logdecay + join_loglikelihood - source_loglikelihood - target_loglikelihood,
                        )
            # sample
            max_lw: float = max(logweight_list)
            # divide weight by a constant to avoid overflow
            logweight_list: List[float] = [lw - max_lw for lw in logweight_list]
            weight_list: List[float] = [math.exp(lw) for lw in logweight_list]
            target: Customer = random.choices(target_list, weights=weight_list, k=1)[0]
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
