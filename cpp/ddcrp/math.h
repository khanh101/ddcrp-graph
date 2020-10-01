//
// Created by khanh on 30/9/20.
//
#ifndef DDCRP_GIBBS_MATH_H
#define DDCRP_GIBBS_MATH_H

#include<numbers>
#include <cmath>
#include "common.h"

namespace math {
    float64 log(float64 x) {
        return std::log(x);
    }

    float64 exp(float64 x) {
        return std::exp(x);
    }

    const float64 log_pi = std::log(std::numbers::pi_v<double>);

    float64 multi_lgamma(float64 a, uint64 d) {
        if (a <= 0.5 * float64(d - 1)) {
            return 0.0;
        }

        double res = 0.25 * float64(d * (d - 1)) * log_pi;
        for (uint64 i = 1; i <= d; i++) {
            res += std::lgamma(a - float64(i - 1) / 2);
        }
        return res;
    }

    class UnitRNG{
    public:
        explicit UnitRNG(uint64 seed): m_engine(seed), m_dist(0.0, 1.0) {}
        ~UnitRNG() = default;
        float64 operator()(){
            return m_dist(m_engine);
        }
    private:
        std::default_random_engine m_engine;
        std::uniform_real_distribution<float64> m_dist;
    };

    template<typename UnitRNG>
    uint64 discrete_sampling(UnitRNG gen, const std::vector<float64> &weight) {
        float64 scale = 0.0;
        for (auto w: weight) {
            scale += w;
        }
        float64 r = gen();
        r *= scale;
        for (uint64 i = 0; i < weight.size(); i++) {
            if (r < weight[i]) {
                return i;
            }
            r -= weight[i];
        }
        return weight.size();
    }
}

#endif //DDCRP_GIBBS_MATH_H
