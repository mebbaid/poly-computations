// bench/bench_conversion.cpp
#include <chrono>
#include <iostream>
#include <random>

#include "polycomp/conversion.hpp"

using namespace polycomp;

int main()
{
  using T = Rational;
  std::mt19937 gen(1);
  std::uniform_int_distribution<int> coord(-20, 20);

  for (int n : {30, 50, 80, 120}) {
    RowMat<T> V(n, 3);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < 3; ++j) V(i, j) = coord(gen);

    auto P = Polyhedron<T>::from_V(V);
    std::vector<Eigen::Index> trace;
    const auto t0 = std::chrono::steady_clock::now();
    compute_h_representation(P, &trace);
    const auto t1 = std::chrono::steady_clock::now();

    std::cout << "n = " << n << ": max rows " << *std::max_element(trace.begin(), trace.end())
              << ", " << P.h->A.rows() << " inequalities, "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
  }
}