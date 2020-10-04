cmake_minimum_required(VERSION 3.17)

project(fyp)
set(CMAKE_CXX_STANDARD 20)
add_library(ddcrp SHARED
        src/ddcrp/interface/clustering_c.cpp
        src/ddcrp/interface/clustering_c.h
        src/ddcrp/core/math.h
        src/ddcrp/core/common.h
        src/ddcrp/core/prior.h
        src/ddcrp/core/ddcrp.h
        )

add_executable(main main.cpp)

target_link_libraries(main ddcrp)

find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(ddcrp PUBLIC OpenMP::OpenMP_CXX)
endif ()
if (CMAKE_BUILD_TYPE MATCHES Debug)
    message(" ")
    message("CMAKE IN DEBUG MODE")
    message(" ")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    message(" ")
    message("CMAKE IN RELEASE MODE")
    message(" ")
endif ()