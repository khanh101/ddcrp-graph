//
// Created by khanh on 30/9/20.
//

#include <iostream>
#include <chrono>
#include "ddcrp_c.h"

void test() {
    uint64 n = 10;
    uint64 d = 2;

    Eigen::MatrixXd data = Eigen::MatrixXd::Random(d, n);

    auto niw = NIW(d);
    auto ddcrp = Assignment(n);
    //
    auto logdecay = std::vector<std::map<uint64, float64>>();

    for (uint64 i = 0; i < n; i++) {
        auto ldc = std::map<uint64, float64>();
        for (uint64 j = 0; j < n; j++) {
            ldc.insert(std::make_pair(
                    j,
                    (data.col(i) - data.col(j)).squaredNorm()
                    ));
        }
        logdecay.push_back(ldc);
    }


    auto loglikelihood = [&](const std::set<uint64> &point_list) -> float64 {
        return 0;
    };
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        std::cout<< i << " ";
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
    std::cout << "duration: " << 1e-9 * float64(duration.count()) << std::endl;
}



int main(int argc, char **argv) {
    test();
}
