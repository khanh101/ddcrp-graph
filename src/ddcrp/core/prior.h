//
// Created by khanh on 30/9/20.
//

#ifndef DDCRP_PRIOR_H
#define DDCRP_PRIOR_H

#include "common.h"
#include "math.h"

class NIW {
    // Nornal Inverse Wishart
public:
    explicit NIW(uint64 dim);

    ~NIW() = default;

    uint64 dimension() const;

    [[nodiscard]] float64 marginal_loglikelihood(const matrix &data, const std::vector<uint64> &index_list) const;

private:
    NIW();

    static vector sample_mean(const matrix &data, const std::vector<uint64> &index_list);

    static matrix uncentered_sum_of_squares(const matrix &data, const std::vector<uint64> &index_list);

    [[nodiscard]] NIW posterior(const matrix &data, const std::vector<uint64> &index_list) const;


    void precompute() {
        // precompute
        m_log_k = math::log(m_k);
        m_logdet_S = math::log(m_S.determinant());
        m_loggamma_d_v_2 = math::multi_lgamma(0.5 * m_v, m_dim);
    }

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

NIW::NIW() :
// init the param
// Murphy: Machine learning - a probabilistic perspective
// Sect. 4.6.3.2
// param dim: dimension
        m_dim(),
        m_k(),
        m_v(),
        m_m(),
        m_S(),
        m_log_k(), m_logdet_S(), m_loggamma_d_v_2() {
}

NIW::NIW(uint64 dim) :
// init the param
// Murphy: Machine learning - a probabilistic perspective
// Sect. 4.6.3.2
// param dim: dimension
        m_dim(dim),
        m_k(0.01),
        m_v(dim + 2),
        m_m(vector::Zero(dim, 1)),
        m_S(matrix::Identity(dim, dim)),
        m_log_k(), m_logdet_S(), m_loggamma_d_v_2() {
    precompute();
}

NIW NIW::posterior(const matrix &data, const std::vector<uint64> &index_list) const {
    // Murphy: Machine learning - a probabilistic perspective
    // Sect. 4.6.3.3
    // param data: (d x n) matrix
    NIW posterior;
    auto n = float64(index_list.size());
    auto mean = vector(sample_mean(data, index_list));
    auto S = uncentered_sum_of_squares(data, index_list);
    posterior.m_dim = m_dim;
    posterior.m_k = m_k + n;
    posterior.m_v = m_v + n;
    posterior.m_m = (m_k * m_m + n * mean) / posterior.m_k;
    posterior.m_S = m_S + S + m_k * m_m * m_m.transpose() - posterior.m_k * posterior.m_m * posterior.m_m.transpose();
    posterior.precompute();
    return posterior;
}

vector NIW::sample_mean(const matrix &data, const std::vector<uint64> &index_list) {
    // param data: (d x n) matrix
    vector mean;
    mean.resize(data.rows(), 1);
    mean.setZero();
    for (auto col: index_list) {
        mean += data.col(col);
    }
    mean /= index_list.size();
    return mean;
}

matrix NIW::uncentered_sum_of_squares(const matrix &data, const std::vector<uint64> &index_list) {
    auto dim = uint64(data.rows());
    matrix S = matrix::Zero(dim, dim);
    for (auto col: index_list) {
        S += data.col(col) * data.col(col).transpose();
    }
    return S;
}

float64 NIW::marginal_loglikelihood(const matrix &data, const std::vector<uint64> &index_list) const {
    auto posterior = this->posterior(data, index_list);
    auto n = float64(data.cols());
    auto d = float64(data.rows());
    float64 llh = 0;
    llh += -0.5 * n * d * math::log_pi;
    llh += posterior.m_loggamma_d_v_2 - m_loggamma_d_v_2;
    llh += 0.5 * m_v * m_logdet_S - 0.5 * posterior.m_v * posterior.m_logdet_S;
    llh += 0.5 * d * (m_log_k - posterior.m_log_k);
    return llh;
}

uint64 NIW::dimension() const {
    return m_dim;
}

#endif //DDCRP_PRIOR_H
