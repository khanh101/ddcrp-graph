//
// Created by khanh on 30/9/20.
//

#include <iostream>
#include "ddcrp_c.h"

int main(int argc, char** argv) {
    uint64 n = 300;
    uint64 d = 2;

    Eigen::MatrixXd data = Eigen::MatrixXd::Random(d, n);

    auto niw = NIW(d);
    auto ddcrp = Assignment(n);
    //
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float64> dist(0, 1);
    auto gen = [&]() -> float64 {
        return dist(e);
    };
    auto logdecay = std::vector<std::map<uint64, float64>>();

    for (uint64 i=0; i<n; i++) {
        auto ldc = std::map<uint64, float64>();
        for (uint64 j=0; j<n; j++) {
            ldc.insert(std::make_pair(
                    j,
                    (data.col(i) - data.col(j)).squaredNorm()
                    ));
        }
        logdecay.push_back(ldc);
    }


    auto loglikelihood = [&](const std::set<uint64>& point_list) -> float64 {
        return 0;
    };

    for (int i=0; i<10; i++) {
        std::cout << i << std::endl;
        ddcrp_iterate(
            gen,
            ddcrp,
            -std::numeric_limits<float64>::infinity(),
            logdecay,
            loglikelihood
            );
    }


    return 0;
}