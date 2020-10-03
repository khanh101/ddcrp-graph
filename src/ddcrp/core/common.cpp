//
// Created by khanh on 3/10/20.
//
#include "common.h"

matrix load_data(uint64 rows, uint64 cols, const float64 *data) {
    auto map = Eigen::Map<const Eigen::Matrix<Eigen::MatrixXd::Scalar, Eigen::MatrixXd::RowsAtCompileTime, Eigen::MatrixXd::ColsAtCompileTime, Eigen::ColMajor>>(data, rows, cols);
    return matrix(map);
}
