//
// Created by khanh on 3/10/20.
//

#ifndef FYP_NIW_H
#define FYP_NIW_H

#include "prior.h"
#include "math.h"

class NIW: Prior {
    // Nornal Inverse Wishart
public:
    explicit NIW(uint64 dim);

    ~NIW() override = default;

    [[nodiscard]] float64 marginal_loglikelihood(const matrix &data, const std::vector<uint64> &index_list) const override;

private:
    NIW();

    static vector sample_mean(const matrix &data, const std::vector<uint64> &index_list);

    static matrix uncentered_sum_of_squares(const matrix &data, const std::vector<uint64> &index_list);

    [[nodiscard]] NIW posterior(const matrix &data, const std::vector<uint64> &index_list) const;


    void precompute();

public:
    uint64 m_dim;
    float64 m_k;
    float64 m_v;
    vector m_m;
    matrix m_S;
    // precompute
    float64 m_log_k;
    float64 m_logdet_S;
    float64 m_loggamma_d_v_2;
};


#endif //FYP_NIW_H
