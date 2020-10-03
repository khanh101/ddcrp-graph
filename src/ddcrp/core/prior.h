//
// Created by khanh on 30/9/20.
//

#ifndef DDCRP_GIBBS_PRIOR_H
#define DDCRP_GIBBS_PRIOR_H

#include "common.h"

class Prior {
public:
    virtual ~Prior() = default;

    [[nodiscard]] virtual float64 marginal_loglikelihood(const matrix &data, const std::vector<uint64> &index_list) const = 0;
};
#endif //DDCRP_GIBBS_PRIOR_H
