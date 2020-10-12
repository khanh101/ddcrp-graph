//
// Created by khanh on 13/10/20.
//

#ifndef FYP_DS_H
#define FYP_DS_H
#include <vector>
#include "common.h"

namespace ds {
    template <typename dtype, dtype element_nil>
    class small_set {
    public:
        small_set() = default;
        ~small_set() = default;
        void insert(dtype the_element);
        void erase(dtype the_element);
        void foreach(std::function<void(dtype)> function);
    private:
        std::vector<dtype> m_element;
    };

    template<typename dtype, dtype element_nil>
    void small_set<dtype, element_nil>::insert(dtype the_element) {
        // insert if does not exist
        // skip if exists
        uint64 first_nil = uint64_nil;
        for (uint64 i=0; i < m_element.size(); i++) {
            auto element = m_element[i];
            if (element == the_element) {
                // exists then return
                return;
            }
            if (first_nil == uint64_nil and element == element_nil) {
                // record the first index with nil
                first_nil = i;
            }
        }
        // does not exist
        if (first_nil == uint64_nil) {
            m_element.push_back(the_element);
        } else {
            m_element[first_nil] = the_element;
        }
    }

    template<typename dtype, dtype element_nil>
    void small_set<dtype, element_nil>::erase(dtype the_element) {
        // erase once if exists
        // skip if does not exists
        for (uint64 i=0; i < m_element.size(); i++) {
            if (m_element[i] == the_element) {
                m_element[i] = element_nil;
                return;
            }
        }
    }

    template<typename dtype, dtype element_nil>
    void small_set<dtype, element_nil>::foreach(std::function<void(dtype)> function) {
        for (uint64 i=0; i < m_element.size(); i++) {
            if (m_element[i] != element_nil) {
                function(m_element[i]);
            }

        }
    }
}
#endif //FYP_DS_H
