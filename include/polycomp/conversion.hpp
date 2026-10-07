#pragma once
// Representation conversion on the homogenized cones, following Ziegler §1.2.
//
//   fourier_motzkin(P):     v_cone -> h_cone   V -> H: C is a projection   (statement I)
//   double_description(P):  h_cone -> v_cone   H -> V: C is an intersection (statement II)
//
//   compute_h_representation(P), compute_v_representation(P):
//       homogenize, convert, remove redundancy, dehomogenize.
//   project(P, k): H-polyhedron onto its first k coordinates, minimal output.
//
// Both lift the cone to a higher-dimensional one with extra columns, and
// then remove those columns with the same routine, project_block:
//   1. pivot_out: columns that some equality (H side) or line (V side) can
//      pivot on are substituted away by Gaussian elimination, at no cost;
//   2. eliminate_columns: the remaining columns go through Fourier-Motzkin
//      with the history rules (see eliminate.hpp).
// The two directions differ only in how the rows are set up and read:
// inequalities with equalities on the H side, rays with lines on the V side.
//
// Equalities and lines come out explicitly (h_cone.E, v_cone.L): the
// pivot rows that are never used are zero on the eliminated columns and
// are exactly the equalities (lines) of the result.
//
// The history rules keep the intermediate systems small but remove
// redundancy in multiplier space only; compute_* finish with
// remove_redundancy (redundancy.hpp), which uses both cones.
//
// If `trace` is given, the number of rows after each elimination is appended.

#include <stdexcept>
#include <utility>
#include <vector>

#include "polycomp/eliminate.hpp"
#include "polycomp/normalize.hpp"
#include "polycomp/redundancy.hpp"

namespace polycomp
{

  namespace internal
  {
    // Removes the columns `cols` (in that order: pivots first where
    // possible, then Fourier-Motzkin) from the system (M, E) and keeps the
    // first `keep` columns of the result.
    template <class S>
    std::pair<RowMat<S>, RowMat<S>> project_block(RowMat<S> M, RowMat<S> E, const std::vector<Eigen::Index> &cols,
                                                  Eigen::Index keep, std::vector<Eigen::Index> *trace,
                                                  Pruning::Minimality minimality)
    {
      std::vector<Eigen::Index> remaining;
      pivot_out(M, E, cols, remaining);
      M = eliminate_columns(std::move(M), remaining, trace, minimality);
      return {RowMat<S>(M.leftCols(keep)), RowMat<S>(E.leftCols(keep))};
    }
  } // namespace internal

  // V -> H by projection.
  //
  // C = cone(rows of R) + span(rows of L) is the projection onto y of
  //
  //   { (y, lambda, mu) :  y - R^T lambda - L^T mu = 0,  lambda >= 0 }     (mu free)
  //
  // Columns [ y (D) | lambda (n) | mu (l) ]. The D equalities pivot on the
  // mu columns first (a free variable costs nothing) and then on lambda
  // columns; the lambda columns left over are eliminated by Fourier-Motzkin
  // from the n rows -lambda_j <= 0. Equalities left after pivoting involve y
  // only: the affine hull of C.
  template <class T>
  void fourier_motzkin(Polyhedron<T> &P, std::vector<Eigen::Index> *trace = nullptr,
                       Pruning::Minimality minimality = Pruning::Minimality::adjacency)
  {
    using C = cone_number_t<T>;
    if (!P.v_cone)
      throw std::logic_error("fourier_motzkin: no v_cone (call homogenize first)");
    const auto &G = *P.v_cone;
    const Eigen::Index D = P.d + 1, n = G.R.rows(), l = G.L.rows(), W = D + n + l;

    RowMat<C> E = RowMat<C>::Zero(D, W);
    E.leftCols(D).setIdentity();
    E.middleCols(D, n) = -G.R.transpose();
    E.rightCols(l) = -G.L.transpose();

    RowMat<C> M = RowMat<C>::Zero(n, W);
    for (Eigen::Index j = 0; j < n; ++j) M(j, D + j) = C(-1);

    std::vector<Eigen::Index> cols;
    for (Eigen::Index c = D + n; c < W; ++c) cols.push_back(c); // mu first
    for (Eigen::Index c = D; c < D + n; ++c) cols.push_back(c); // then lambda

    auto [A, Eq] = internal::project_block(std::move(M), std::move(E), cols, D, trace, minimality);
    P.h_cone = typename Polyhedron<T>::HCone{std::move(A), std::move(Eq)};
  }

  // H -> V by intersection (Ziegler's C0 construction).
  //
  // C = { y : A y <= 0, E y = 0 } is the slice w = 0, v = 0 of
  //
  //   C0 = { (y, w, v) : A y <= w,  E y = v }
  //      = span{ (e_i, A e_i, E e_i) : i < D }  +  cone{ (0, e_j, 0) : j < m },
  //
  // since (y, w, v) = sum_i y_i (e_i, A e_i, E e_i) + sum_j (w - A y)_j (0, e_j, 0).
  // Columns [ y (D) | w (m) | v (e) ]. The lines pivot on the v columns
  // first (that is Gaussian elimination of E) and on w columns where they
  // can (up to rank A of them); the remaining w columns are intersected away
  // by Fourier-Motzkin on the rays. Lines left over are the lineality space.
  template <class T>
  void double_description(Polyhedron<T> &P, std::vector<Eigen::Index> *trace = nullptr,
                          Pruning::Minimality minimality = Pruning::Minimality::adjacency)
  {
    using C = cone_number_t<T>;
    if (!P.h_cone)
      throw std::logic_error("double_description: no h_cone (call homogenize first)");
    const auto &H = *P.h_cone;
    const Eigen::Index D = P.d + 1, m = H.A.rows(), e = H.E.rows(), W = D + m + e;

    RowMat<C> L = RowMat<C>::Zero(D, W);
    L.leftCols(D).setIdentity();
    L.middleCols(D, m) = H.A.transpose();
    L.rightCols(e) = H.E.transpose();

    RowMat<C> R = RowMat<C>::Zero(m, W);
    for (Eigen::Index j = 0; j < m; ++j) R(j, D + j) = C(1);

    std::vector<Eigen::Index> cols;
    for (Eigen::Index c = D + m; c < W; ++c) cols.push_back(c); // v first
    for (Eigen::Index c = D; c < D + m; ++c) cols.push_back(c); // then w

    auto [Rays, Lines] = internal::project_block(std::move(R), std::move(L), cols, D, trace, minimality);
    P.v_cone = typename Polyhedron<T>::VCone{std::move(Rays), std::move(Lines)};
  }

  // Projection of an H-polyhedron onto its first k coordinates (Ziegler §1.2,
  // statement I read on the H side: eliminate x_{k+1}, ..., x_d).
  //
  //   pi(P) = { x' in R^k : exists x'' with (x', x'') in P }
  //
  // On the homogenized cone the columns are [ x0 | x' | x'' ]; the x''
  // columns are removed by project_block (equalities pivot first, the rest
  // by Fourier-Motzkin). The result is completed exactly like the
  // conversions: double description for its generators, redundancy removal
  // with both cones, dehomogenization. The returned polyhedron is minimal on
  // both sides: facets with explicit equalities, and points, extreme rays and
  // lines.
  //
  // (Projecting a V-polyhedron is just dropping columns of V, Y, L.)
  template <class T>
  Polyhedron<T> project(Polyhedron<T> P, Eigen::Index k, std::vector<Eigen::Index> *trace = nullptr,
                        Pruning::Minimality minimality = Pruning::Minimality::adjacency)
  {
    if (!P.h) throw std::logic_error("project: P has no H-representation");
    if (k < 0 || k > P.d) throw std::out_of_range("project: k must be in [0, d]");
    homogenize(P);
    auto &H = *P.h_cone;

    std::vector<Eigen::Index> cols;
    for (Eigen::Index c = k + 1; c <= P.d; ++c) cols.push_back(c);
    auto [A, E] = internal::project_block(std::move(H.A), std::move(H.E), cols, k + 1, trace, minimality);

    Polyhedron<T> Q;
    Q.d = k;
    Q.h_cone = typename Polyhedron<T>::HCone{std::move(A), std::move(E)};
    double_description(Q);
    remove_redundancy(Q);
    dehomogenize(Q);
    return Q;
  }

  template <class T>
  void compute_h_representation(Polyhedron<T> &P, std::vector<Eigen::Index> *trace = nullptr)
  {
    if (P.h) return;
    if (!P.v) throw std::logic_error("compute_h_representation: P has no representation");
    homogenize(P);
    fourier_motzkin(P, trace);
    remove_redundancy(P);
    dehomogenize(P);
  }

  template <class T>
  void compute_v_representation(Polyhedron<T> &P, std::vector<Eigen::Index> *trace = nullptr)
  {
    if (P.v) return;
    if (!P.h) throw std::logic_error("compute_v_representation: P has no representation");
    homogenize(P);
    double_description(P, trace);
    remove_redundancy(P);
    dehomogenize(P);
  }

} // namespace polycomp