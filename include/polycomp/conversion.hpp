#pragma once
// Representation conversion on the homogenized cones, following Ziegler §1.2.
//
//   fourier_motzkin(P):     v_cone -> h_cone   V -> H: C is a projection   (statement I)
//   double_description(P):  h_cone -> v_cone   H -> V: C is an intersection (statement II)
//
//   compute_h_representation(P), compute_v_representation(P):
//       homogenize, convert, dehomogenize.
//
// Both lift the cone to a higher-dimensional one (extra columns lambda, mu
// or w) and then remove the extra columns one at a time with eliminate_columns.
// They differ only in how the rows are set up and in how the rows are read
// (inequalities vs generators).
//
// Redundancy: Chernikov's and Kohler's rules (see eliminate.hpp) are applied
// after every elimination. They keep the intermediate systems small but do
// not guarantee a minimal output: fourier_motzkin may return redundant
// inequalities, and equalities appear as pairs of opposite inequalities in
// h_cone->A; double_description returns lines as pairs of opposite rays in
// v_cone->R.
//
// If `trace` is given, the number of rows after each elimination is appended.

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "polycomp/eliminate.hpp"
#include "polycomp/normalize.hpp"

namespace polycomp
{

  // V -> H by projection.
  //
  // C = cone(rows of R) + span(rows of L) is the projection onto y of
  //
  //   { (y, lambda, mu) :  y - R^T lambda - L^T mu = 0,  lambda >= 0 }     (mu free)
  //

  template <class T>
  void fourier_motzkin(Polyhedron<T> &P, std::vector<Eigen::Index> *trace = nullptr)
  {
    using C = cone_number_t<T>;
    if (!P.v_cone)
      throw std::logic_error("fourier_motzkin: no v_cone (call homogenize first)");
    const auto &G = *P.v_cone;
    const Eigen::Index D = P.d + 1, n = G.R.rows(), l = G.L.rows(), W = D + n + l;

    // Columns: [ y (D) | lambda (n) | mu (l) ].
    // Rows 0..D-1:     coordinate c of  y - R^T lambda - L^T mu <= 0
    // Rows D..2D-1:    the same rows negated (together: the equality)
    // Rows 2D..2D+n-1: -lambda_j <= 0
    RowMat<C> M = RowMat<C>::Zero(2 * D + n, W);
    for (Eigen::Index c = 0; c < D; ++c)
    {
      M(c, c) = C(1);
      M.row(c).segment(D, n) = -G.R.col(c).transpose();
      M.row(c).segment(D + n, l) = -G.L.col(c).transpose();
      M.row(D + c) = -M.row(c);
    }
    for (Eigen::Index j = 0; j < n; ++j)
      M(2 * D + j, D + j) = C(-1);

    M = eliminate_columns(std::move(M), D, W, trace);
    P.h_cone = typename Polyhedron<T>::HCone{M.leftCols(D), RowMat<C>(0, D)};
  }

  // H -> V by intersection (Ziegler's C0 construction).
  //
  // With M = [A; E; -E] (each equality as two inequalities),
  //
  //   C0 = { (y, w) : M y <= w }  =  cone of the rows   +-(e_i, M e_i)   i < D
  //                                                      (0, e_j)         j < m
  //
  // since (y, w) = sum_i y_i (e_i, M e_i) + sum_j (w - M y)_j (0, e_j). 
  // Intersecting C0 with the hyperplanes w_j = 0 leaves { (y, 0) : M y <= 0 },
  // i.e. the cone we want; each intersection is one elimination.

  template <class T>
  void double_description(Polyhedron<T> &P, std::vector<Eigen::Index> *trace = nullptr)
  {
    using C = cone_number_t<T>;
    if (!P.h_cone)
      throw std::logic_error("double_description: no h_cone (call homogenize first)");
    const auto &H = *P.h_cone;
    const Eigen::Index D = P.d + 1;

    RowMat<C> M(H.A.rows() + 2 * H.E.rows(), D);
    M << H.A, H.E, -H.E;
    const Eigen::Index m = M.rows(), W = D + m;

    // Columns: [ y (D) | w (m) ].
    // Rows 0..D-1:     (e_i, M e_i)      and rows D..2D-1 their negatives (lines)
    // Rows 2D..2D+m-1: (0, e_j)          slack generators
    RowMat<C> Gen = RowMat<C>::Zero(2 * D + m, W);
    for (Eigen::Index i = 0; i < D; ++i)
    {
      Gen(i, i) = C(1);
      Gen.row(i).tail(m) = M.col(i).transpose();
      Gen.row(D + i) = -Gen.row(i);
    }
    for (Eigen::Index j = 0; j < m; ++j)
      Gen(2 * D + j, D + j) = C(1);

    Gen = eliminate_columns(std::move(Gen), D, W, trace);
    P.v_cone = typename Polyhedron<T>::VCone{Gen.leftCols(D), RowMat<C>(0, D)};
  }

  template <class T>
  void compute_h_representation(Polyhedron<T> &P, std::vector<Eigen::Index> *trace = nullptr)
  {
    if (P.h)
      return;
    if (!P.v)
      throw std::logic_error("compute_h_representation: P has no representation");
    homogenize(P);
    fourier_motzkin(P, trace);
    dehomogenize(P);
  }

  template <class T>
  void compute_v_representation(Polyhedron<T> &P, std::vector<Eigen::Index> *trace = nullptr)
  {
    if (P.v)
      return;
    if (!P.h)
      throw std::logic_error("compute_v_representation: P has no representation");
    homogenize(P);
    double_description(P, trace);
    dehomogenize(P);
  }

} // namespace polycomp