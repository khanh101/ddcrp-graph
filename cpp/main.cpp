//
// Created by khanh on 30/9/20.
//

#include <iostream>
#include <chrono>
#include <set>
#include "clustering_c.h"
#include "ddcrp/prior.h"
#include "ddcrp/ddcrp.h"

void test(uint64 n, uint64 d) {

    Eigen::MatrixXd data = Eigen::MatrixXd::Random(d, n);

    auto niw = NIW(d);
    auto ddcrp = Assignment(n);
    //
    auto logdecay = [&](Customer customer) -> std::map<uint64, float64>{
        auto ldc = std::map<uint64, float64>();
        for (uint64 target = 0; target < n; target++) {
            ldc.insert(std::make_pair(
                    target,
                    (data.col(customer) - data.col(target)).squaredNorm()
            ));
        }
        return ldc;
    };


    auto loglikelihood = [&](const std::set<uint64> &point_list) -> float64 {
        return 0;
    };
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        ddcrp_iterate(
                math::UnitRNG(1234),
                ddcrp,
                -std::numeric_limits<float64>::infinity(),
                logdecay,
                loglikelihood
        );
    }
    std::cout << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "(n, d): ("<< n << ", "<< d << ")" << " duration: " << 1e-9 * float64(duration.count()) << std::endl;
}

void eigen(uint64 rows) {
    std::vector<float64> values = {1, 2, 3, 4, 5, 6};
    auto map = Eigen::Map<const Eigen::Matrix<Eigen::MatrixXd::Scalar, Eigen::MatrixXd::RowsAtCompileTime, Eigen::MatrixXd::ColsAtCompileTime, Eigen::ColMajor>>(
            values.data(), rows, values.size() / rows);
    std::cout << map << std::endl;
}

template<class Iter>
void foo(Iter begin, Iter end) {
    for (auto it = begin; it != end; ++it) {
        int x = *it;
        std::cout << x << " ";
    }
    std::cout<<std::endl;
}

void initializer_list() {
    std::set<int> s{1, 2, 3};
    foo(s.begin(), s.end());
}


int main(int argc, char **argv) {
    initializer_list();
}
