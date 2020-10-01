//
// Created by khanh on 30/9/20.
// Common type
//

#ifndef DDCRP_GIBBS_TYPE_H
#define DDCRP_GIBBS_TYPE_H
#include <eigen3/Eigen/Dense>

static_assert(sizeof(double) == 8, "double size must be 8 bytes");
static_assert(sizeof(unsigned long) == 8, "unsigned long size must be 8 bytes");
static_assert(sizeof(long) == 8, "long size must be 8 bytes");

using float64 = double;
using uint64 = unsigned long;
using int64 = long;

using vector = Eigen::Matrix<float64, Eigen::Dynamic, 1>;
using matrix = Eigen::Matrix<float64, Eigen::Dynamic, Eigen::Dynamic>;

const bool debug = true;

#endif //DDCRP_GIBBS_TYPE_H
