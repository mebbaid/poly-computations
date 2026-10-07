// Contact wrench cone: cost of the projection, three ways (not a test; the
// correctness check is tests/test_cwc.cpp).
//
//   pairs        each equality as two inequalities, 12 Fourier-Motzkin steps
//   substituted  equalities solved by hand, 6 Fourier-Motzkin steps
//   pivot_out    the library's internal path (what project() runs)
//   project()    the public call, including double description and
//                redundancy removal of the result
//
// For each, the rows straight out of elimination (before any redundancy
// removal) are compared with Caron's 16.

#include <algorithm>
#include <chrono>
#include <iostream>

#include "polycomp/conversion.hpp"

using namespace polycomp;
using T = Rational;
using C = Integer;

struct Foot { T X, Y, mu; };
constexpr Eigen::Index W = 6, NV = 18;

void corner_system(const Foot& ft, RowMat<T>& E, RowMat<T>& A)
{
  const T px[4] = {ft.X, ft.X, -ft.X, -ft.X}, py[4] = {ft.Y, -ft.Y, ft.Y, -ft.Y};
  E = RowMat<T>::Zero(6, NV);
  E.leftCols(6).setIdentity();
  A = RowMat<T>::Zero(16, NV);
  for (int i = 0; i < 4; ++i) {
    const Eigen::Index fx = 6 + 3 * i, fy = fx + 1, fz = fx + 2, r = 4 * i;
    E(0, fx) = -1; E(1, fy) = -1; E(2, fz) = -1;
    E(3, fz) = -py[i]; E(4, fz) = px[i]; E(5, fy) = -px[i]; E(5, fx) = py[i];
    A(r + 0, fx) = 1;  A(r + 0, fz) = -ft.mu;
    A(r + 1, fx) = -1; A(r + 1, fz) = -ft.mu;
    A(r + 2, fy) = 1;  A(r + 2, fz) = -ft.mu;
    A(r + 3, fy) = -1; A(r + 3, fz) = -ft.mu;
  }
}

RowMat<T> caron_cwc(const Foot& ft)
{
  const T X = ft.X, Y = ft.Y, mu = ft.mu, s = (X + Y) * mu;
  RowMat<T> U(16, 6);
  U << -1, 0, -mu, 0, 0, 0,     1, 0, -mu, 0, 0, 0,     0, -1, -mu, 0, 0, 0,    0, 1, -mu, 0, 0, 0,
        0, 0, -Y, -1, 0, 0,     0, 0, -Y, 1, 0, 0,      0, 0, -X, 0, -1, 0,     0, 0, -X, 0, 1, 0,
       -Y, -X, -s, mu, mu, -1, -Y, X, -s, mu, -mu, -1,  Y, -X, -s, -mu, mu, -1, Y, X, -s, -mu, -mu, -1,
        Y, X, -s, mu, mu, 1,    Y, -X, -s, mu, -mu, 1, -Y, X, -s, -mu, mu, 1,  -Y, -X, -s, -mu, -mu, 1;
  return U;
}

RowMat<C> to_integer_rows(const RowMat<T>& M)
{
  RowMat<C> K(M.rows(), M.cols());
  Eigen::Index n = 0;
  for (Eigen::Index i = 0; i < M.rows(); ++i) {
    Eigen::Matrix<T, 1, Eigen::Dynamic> r = M.row(i);
    if (internal::lift(r(0), r.tail(r.size() - 1), K.row(n))) ++n;
  }
  K.conservativeResize(n, M.cols());
  return K;
}

bool same_row_set(const RowMat<C>& A, const RowMat<C>& B)
{
  auto contains = [](const RowMat<C>& M, const RowMat<C>& N) {
    for (Eigen::Index i = 0; i < N.rows(); ++i) {
      bool found = false;
      for (Eigen::Index j = 0; j < M.rows() && !found; ++j) found = M.row(j) == N.row(i);
      if (!found) return false;
    }
    return true;
  };
  return contains(A, B) && contains(B, A);
}

// Route "substituted": solve E for 6 of the f-columns, substitute into A.
RowMat<T> substitute_equalities(RowMat<T> E, const RowMat<T>& A, std::vector<Eigen::Index>& free_cols)
{
  std::vector<Eigen::Index> pivot_col;
  Eigen::Index r = 0;
  for (Eigen::Index c = W; c < NV && r < E.rows(); ++c) {
    Eigen::Index p = -1;
    for (Eigen::Index i = r; i < E.rows(); ++i) if (E(i, c) != 0) { p = i; break; }
    if (p < 0) continue;
    E.row(r).swap(E.row(p));
    E.row(r) /= E(r, c);
    for (Eigen::Index i = 0; i < E.rows(); ++i) if (i != r && E(i, c) != 0) E.row(i) -= E(i, c) * E.row(r);
    pivot_col.push_back(c);
    ++r;
  }
  RowMat<T> S = A;
  for (Eigen::Index k = 0; k < r; ++k)
    for (Eigen::Index i = 0; i < S.rows(); ++i)
      if (S(i, pivot_col[k]) != 0) S.row(i) -= S(i, pivot_col[k]) * E.row(k);
  free_cols.clear();
  for (Eigen::Index c = W; c < NV; ++c)
    if (std::find(pivot_col.begin(), pivot_col.end(), c) == pivot_col.end()) free_cols.push_back(c);
  return S;
}

using Route = RowMat<C> (*)(const RowMat<T>&, const RowMat<T>&, std::vector<Eigen::Index>&, Pruning::Minimality);

RowMat<C> project_pairs(const RowMat<T>& E, const RowMat<T>& A, std::vector<Eigen::Index>& trace,
                        Pruning::Minimality mode)
{
  RowMat<T> M(2 * E.rows() + A.rows(), NV);
  M << E, -E, A;
  const RowMat<C> K = eliminate_columns(to_integer_rows(M), W, NV, &trace, mode);
  return internal::unique_rows(RowMat<C>(K.leftCols(W)));
}

RowMat<C> project_substituted(const RowMat<T>& E, const RowMat<T>& A, std::vector<Eigen::Index>& trace,
                              Pruning::Minimality mode)
{
  std::vector<Eigen::Index> free_cols;
  const RowMat<T> S = substitute_equalities(E, A, free_cols);
  RowMat<T> Sp(S.rows(), W + Eigen::Index(free_cols.size()));
  Sp.leftCols(W) = S.leftCols(W);
  for (std::size_t k = 0; k < free_cols.size(); ++k) Sp.col(W + Eigen::Index(k)) = S.col(free_cols[k]);
  const RowMat<C> K = eliminate_columns(to_integer_rows(Sp), W, Sp.cols(), &trace, mode);
  return internal::unique_rows(RowMat<C>(K.leftCols(W)));
}

RowMat<C> project_pivot_out(const RowMat<T>& E, const RowMat<T>& A, std::vector<Eigen::Index>& trace,
                            Pruning::Minimality mode)
{
  std::vector<Eigen::Index> cols, remaining;
  for (Eigen::Index c = W; c < NV; ++c) cols.push_back(c);
  RowMat<C> M = to_integer_rows(A), Eq = to_integer_rows(E);
  pivot_out(M, Eq, cols, remaining);
  M = eliminate_columns(std::move(M), remaining, &trace, mode);
  return internal::unique_rows(RowMat<C>(M.leftCols(W)));
}

template <class F>
double ms(F&& f, int reps = 20)
{
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) f();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

int main()
{
  for (const Foot& ft : {Foot{T(2), T(1), T(1, 2)}, Foot{T(1), T(1), T(1)}, Foot{T(3), T(1, 2), T(7, 10)}}) {
    std::cout << "--- foot X=" << ft.X << " Y=" << ft.Y << " mu=" << ft.mu << " ---\n";
    RowMat<T> E, A;
    corner_system(ft, E, A);
    const RowMat<C> U = to_integer_rows(caron_cwc(ft));

    const struct { const char* name; Route fn; } routes[] = {
        {"pairs       (12 FM)", project_pairs},
        {"substituted ( 6 FM)", project_substituted},
        {"pivot_out   ( 6 FM)", project_pivot_out}};
    for (const auto& route : routes)
      for (auto mode : {Pruning::Minimality::adjacency, Pruning::Minimality::kohler}) {
        std::vector<Eigen::Index> trace;
        const RowMat<C> K = route.fn(E, A, trace, mode);
        const double t = ms([&] { std::vector<Eigen::Index> tr; route.fn(E, A, tr, mode); });
        std::cout << route.name << (mode == Pruning::Minimality::kohler ? " [kohler]   " : " [adjacency]")
                  << " trace";
        for (auto x : trace) std::cout << ' ' << x;
        std::cout << " | " << K.rows() << " distinct rows"
                  << (same_row_set(K, U) ? " = Caron's 16" : " (redundant rows left)") << ", " << t << " ms\n";
      }

    auto P = Polyhedron<T>::from_H(A, Vec<T>::Zero(16), E, Vec<T>::Zero(6));
    std::cout << "project() incl. DD + redundancy removal: " << ms([&] { project(P, W); }) << " ms\n";
  }
}