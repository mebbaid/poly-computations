cmake_minimum_required(VERSION 3.21)
project(polycomp LANGUAGES CXX)

option(POLYCOMP_WITH_GMP "Exact arithmetic (Boost.Multiprecision + GMP)" ON)

find_package(Eigen3 CONFIG REQUIRED)

add_library(polycomp INTERFACE)
target_include_directories(polycomp INTERFACE include)
target_compile_features(polycomp INTERFACE cxx_std_20)
target_link_libraries(polycomp INTERFACE Eigen3::Eigen)

if(POLYCOMP_WITH_GMP)
  find_path(BOOST_MP_INCLUDE_DIR boost/multiprecision/gmp.hpp REQUIRED)
  find_path(GMP_INCLUDE_DIR gmp.h REQUIRED)
  find_library(GMP_LIB gmp REQUIRED)
  target_include_directories(polycomp INTERFACE ${BOOST_MP_INCLUDE_DIR} ${GMP_INCLUDE_DIR})
  target_link_libraries(polycomp INTERFACE ${GMP_LIB})
  target_compile_definitions(polycomp INTERFACE POLYCOMP_HAS_GMP)
endif()

if(PROJECT_IS_TOP_LEVEL)
  enable_testing()

  add_executable(polycomp_tests tests/tests.cpp)
  target_link_libraries(polycomp_tests PRIVATE polycomp)
  target_compile_options(polycomp_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME polycomp_tests COMMAND polycomp_tests)

  if(POLYCOMP_WITH_GMP)   # the comparison with Caron's cone is exact: needs Rational
    add_executable(polycomp_test_cwc tests/cwc.cpp)
    target_link_libraries(polycomp_test_cwc PRIVATE polycomp)
    target_compile_options(polycomp_test_cwc PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME polycomp_test_cwc COMMAND polycomp_test_cwc)
  endif()

  find_package(highs CONFIG)
  if(highs_FOUND)
    add_executable(polycomp_test_lp tests/test_lp.cpp)
    target_link_libraries(polycomp_test_lp PRIVATE polycomp highs::highs)
    target_compile_options(polycomp_test_lp PRIVATE -Wall)
    add_test(NAME polycomp_test_lp COMMAND polycomp_test_lp)
  endif()

  add_executable(polycomp_bench bench/bench_cwc.cpp)
  target_link_libraries(polycomp_bench PRIVATE polycomp)
endif()