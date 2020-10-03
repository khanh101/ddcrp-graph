//
// Created by khanh on 3/10/20.
//

#include <cmath>
#include <random>
#include "common.h"

#include "math.h"

float64 math::log(float64 x) {
    return std::log(x);
}

float64 math::exp(float64 x) {
    return std::exp(x);
}

float64 math::multi_lgamma(float64 a, uint64 d) {
    if (a <= 0.5 * float64(d - 1)) {
        return 0.0;
    }

    double res = 0.25 * float64(d * (d - 1)) * log_pi;
    for (uint64 i = 1; i <= d; i++) {
        res += std::lgamma(a - float64(i - 1) / 2);
    }
    return res;
}

template<typename UnitRNG>
uint64 math::discrete_sampling(UnitRNG gen, const std::vector<float64> &weight) {
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

math::UnitRNG::UnitRNG(uint64 seed) : m_engine(seed), m_dist(0.0, 1.0) {}

float64 math::UnitRNG::operator()() {
    return m_dist(m_engine);
}

template uint64 math::discrete_sampling(math::UnitRNG gen, const std::vector<float64> &weight);
