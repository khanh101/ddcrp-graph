cmake_minimum_required(VERSION 3.17)
project(ddcrp_gibbs)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

add_library(ddcrp_gibbs SHARED
        cpp/clustering/clustering_c.cpp
        cpp/clustering/clustering_c.h
        cpp/clustering/ddcrp/math.h
        cpp/clustering/ddcrp/common.h cpp/clustering/ddcrp/prior.h cpp/clustering/ddcrp/ddcrp.h)
add_executable(ddcrp_gibbs_main cpp/main.cpp)
target_link_libraries(ddcrp_gibbs_main ddcrp_gibbs)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    message(" ")
    message("CMAKE IN DEBUG MODE")
    message(" ")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(" ")
    message("CMAKE IN RELEASE MODE")
    message(" ")
endif ()