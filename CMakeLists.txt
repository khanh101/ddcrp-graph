cmake_minimum_required(VERSION 3.17)
project(ddcrp_gibbs)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

add_library(ddcrp_gibbs SHARED
        cpp/clustering_c.cpp
        cpp/clustering_c.h
        cpp/ddcrp/math.h
        cpp/ddcrp/common.h cpp/ddcrp/prior.h cpp/ddcrp/ddcrp.h)
add_executable(ddcrp_gibbs_main cpp/main.cpp)
target_link_libraries(ddcrp_gibbs_main ddcrp_gibbs)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
