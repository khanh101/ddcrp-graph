//
// Created by khanh on 1/10/20.
//

#ifndef DDCRP_GIBBS_COLLECTION_H
#define DDCRP_GIBBS_COLLECTION_H

#include <initializer_list>
#include <vector>
#include "common.h"

template <typename ctype, typename dtype>
class collection {
public:
    explicit collection(const std::initializer_list<ctype>& container_list): m_container_list(container_list) {
    };
    ~collection() = default;
private:
    std::vector<std::reference_wrapper<ctype>> m_container_list;
};

#endif //DDCRP_GIBBS_COLLECTION_H
