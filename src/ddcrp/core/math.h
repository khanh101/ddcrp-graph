//
// Created by khanh on 30/9/20.
//
#ifndef DDCRP_GIBBS_MATH_H
#define DDCRP_GIBBS_MATH_H

#include<numbers>
#include<random>
namespace math {
    float64 log(float64 x);

    float64 exp(float64 x);

    const float64 log_pi = std::log(std::numbers::pi_v<double>);

    float64 multi_lgamma(float64 a, uint64 d);
    class UnitRNG{
    public:
        explicit UnitRNG(uint64 seed);

        ~UnitRNG() = default;

        float64 operator()();
    private:
        std::default_random_engine m_engine;
        std::uniform_real_distribution<float64> m_dist;
    };

    template<typename UnitRNG>
    uint64 discrete_sampling(UnitRNG gen, const std::vector<float64> &weight);


}


#endif //DDCRP_GIBBS_MATH_H
