#pragma once
// Redundancy by linear programming: the H side alone, no generators needed.
// Same output as facets() in redundancy.hpp; use this when the V side is
// out of reach (e.g. F_tau, whose vertex count is exponential in dim).
//
// Cone C = { y : A y <= 0,  E y = 0 } in R^D, rows a_i of A. By Farkas'
// lemma (Ziegler §1.4) every question is "is a vector in a cone of rows":
//
//   a_i is redundant            iff   a_i  in cone(a_j : j != i) + span(E)
//   a_i is an implicit equality iff  -a_i  in cone(a_j)          + span(E)
//
// Both are decided by ONE LP model with D rows (so the simplex basis is
// D x D, however many rows A has):
//
//   min  sum(u + v)   s.t.   A^T lambda + E^T mu + u - v = c,
//                            lambda >= 0,  mu free,  u, v >= 0.
//
// The optimum is the L1 distance from c to cone(A) + span(E): 0 iff c is
// in it. (It is the LP dual of  max { c y : y in C, -1 <= y <= 1 }.)
// Each test changes only the right-hand side c and one bound, so the
// previous basis stays dual feasible and HiGHS warm-starts from it.
//
//   1. Implicit equalities (Gordan's alternative): with u = v = 0, c = 0,
//      maximize sum(lambda), 0 <= lambda <= 1. Any lambda_i > 0 in the
//      solution is a certificate (A^T lambda in span E, so a_i y = 0 on C).
//      Those rows become equalities (lambda_i free) and the LP is solved
//      again, until no new row shows up.
//   2. Cheap facet certificates (no LP): the LP dual of step 1 is a point
//      y0 strictly inside every inequality; if its projection onto row i's
//      hyperplane is strictly inside every other row, row i is a facet.
//   3. Redundancy: for each remaining row, lambda_i := 0 and c := a_i. If the
//      distance is 0 the row is redundant and lambda_i STAYS 0 (so of two
//      equal rows exactly one is kept); otherwise lambda_i is freed again.
//
// Floating point only; tolerance lp_eps (cone rows are scaled to
// max |entry| = 1 by homogenize, so the distances are O(1)).

#include <Highs.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "polycomp/redundancy.hpp"

namespace polycomp
{

  inline double lp_eps = 1e-7;

  inline Facets<double> lp_facets(const RowMat<double> &A, const RowMat<double> &E)
  {
    const HighsInt m = HighsInt(A.rows()), e = HighsInt(E.rows()), D = HighsInt(A.cols());
    const HighsInt U = m + e;  // columns: lambda [0, m), mu [m, U), then (u_r, v_r) at U + 2r, U + 2r + 1

    Highs lp;
    lp.setOptionValue("output_flag", false);
    lp.setOptionValue("presolve", "off");  // re-solves start from the previous basis
    {
      std::vector<double> zero(D, 0.0);
      lp.addRows(D, zero.data(), zero.data(), 0, nullptr, nullptr, nullptr);
      std::vector<HighsInt> idx(D);
      for (HighsInt r = 0; r < D; ++r) idx[r] = r;
      std::vector<double> col(D);
      for (HighsInt i = 0; i < m; ++i)  // lambda_i: column a_i
      {
        for (HighsInt r = 0; r < D; ++r) col[r] = A(i, r);
        lp.addCol(0.0, 0.0, kHighsInf, D, idx.data(), col.data());
      }
      for (HighsInt k = 0; k < e; ++k)  // mu_k: column e_k
      {
        for (HighsInt r = 0; r < D; ++r) col[r] = E(k, r);
        lp.addCol(0.0, -kHighsInf, kHighsInf, D, idx.data(), col.data());
      }
      for (HighsInt r = 0; r < D; ++r)  // u_r, v_r: +-unit columns
      {
        const double one = 1.0, minus = -1.0;
        lp.addCol(1.0, 0.0, kHighsInf, 1, &idx[r], &one);
        lp.addCol(1.0, 0.0, kHighsInf, 1, &idx[r], &minus);
      }
    }

    auto solve = [&]() {
      lp.run();
      if (lp.getModelStatus() != HighsModelStatus::kOptimal)
        throw std::runtime_error("lp_facets: LP not solved to optimality");
      return lp.getObjectiveValue();
    };
    auto set_rhs = [&](const auto &c) {
      for (HighsInt r = 0; r < D; ++r) lp.changeRowBounds(r, c(r), c(r));
    };
    auto set_uv = [&](double hi) {  // allow (hi = inf) or forbid (hi = 0) the residual
      for (HighsInt r = 0; r < D; ++r)
      {
        lp.changeColBounds(U + 2 * r, 0.0, hi);
        lp.changeColBounds(U + 2 * r + 1, 0.0, hi);
      }
    };

    // 1. Implicit equalities: A^T lambda in span(E), lambda >= 0, as large as possible.
    std::vector<char> implicit(m, 0);
    set_uv(0.0);
    for (HighsInt i = 0; i < m; ++i)
    {
      lp.changeColCost(i, -1.0);
      lp.changeColBounds(i, 0.0, 1.0);
    }
    for (bool found = true; found;)
    {
      found = false;
      solve();  // rhs is 0
      const auto &lambda = lp.getSolution().col_value;
      for (HighsInt i = 0; i < m; ++i)
        if (!implicit[i] && lambda[i] > lp_eps)
        {
          implicit[i] = 1;
          found = true;
          lp.changeColCost(i, 0.0);
          lp.changeColBounds(i, -kHighsInf, kHighsInf);  // now an equality row
        }
    }
    const Eigen::VectorXd y0 = Eigen::Map<const Eigen::VectorXd>(lp.getSolution().row_dual.data(), D);  // see 2.
    for (HighsInt i = 0; i < m; ++i)
      if (!implicit[i])
      {
        lp.changeColCost(i, 0.0);
        lp.changeColBounds(i, 0.0, kHighsInf);
      }
    set_uv(kHighsInf);

    // 2. Facets certified without LP. The LP dual of step 1 is a point y0 of
    //    C with a_i y0 <= -1 on every other row (reduced cost -1 - a_i y0 >= 0)
    //    and = 0 on the equalities. Move it onto row i's hyperplane inside
    //    the subspace S = { E y = 0, implicit rows y = 0 }:
    //        p = y0 - (a_i y0 / a_i.n) n,   n = a_i projected onto S.
    //    If a_j p < 0 for every other row j, p is in the relative interior
    //    of the face cut out by a_i: a facet.
    std::vector<char> facet(m, 0);
    {
      const Eigen::VectorXd s0 = A * y0;
      RowMat<double> Z(e + std::count(implicit.begin(), implicit.end(), 1), D);
      Z.topRows(e) = E;
      for (HighsInt i = 0, k = e; i < m; ++i)
        if (implicit[i]) Z.row(k++) = A.row(i);
      Eigen::MatrixXd N = Eigen::MatrixXd::Identity(D, D);  // orthonormal basis of S
      if (Z.rows() > 0)
      {
        const Eigen::MatrixXd K = Eigen::FullPivLU<Eigen::MatrixXd>(Z).kernel();
        const Eigen::MatrixXd Q = Eigen::HouseholderQR<Eigen::MatrixXd>(K).householderQ();
        N = Q.leftCols(K.cols());
      }
      bool interior = true;
      for (HighsInt i = 0; i < m; ++i) interior = interior && (implicit[i] || s0(i) <= -0.5);
      for (HighsInt i = 0; interior && i < m; ++i)
      {
        if (implicit[i]) continue;
        const Eigen::VectorXd n = N * (N.transpose() * A.row(i).transpose());
        const double an = A.row(i).dot(n);
        if (an <= lp_eps) continue;
        const Eigen::VectorXd p = y0 - (s0(i) / an) * n;
        bool inside = true;
        for (HighsInt j = 0; j < m && inside; ++j)
          if (j != i && !implicit[j]) inside = A.row(j).dot(p) < -lp_eps;
        facet[i] = inside;
      }
    }

    // 3. Redundancy of the rest: is a_i in the cone of the rows still in play?
    Facets<double> out{RowMat<double>(m, D), RowMat<double>(m, D)};
    Eigen::Index nk = 0, ni = 0;
    for (HighsInt i = 0; i < m; ++i)
    {
      if (implicit[i]) { out.implicit.row(ni++) = A.row(i); continue; }
      if (facet[i]) { out.kept.row(nk++) = A.row(i); continue; }
      lp.changeColBounds(i, 0.0, 0.0);
      set_rhs(A.row(i));
      if (solve() > lp_eps)
      {
        lp.changeColBounds(i, 0.0, kHighsInf);
        out.kept.row(nk++) = A.row(i);
      }
    }
    out.kept.conservativeResize(nk, D);
    out.implicit.conservativeResize(ni, D);
    return out;
  }

  // Minimal H-representation of P by LP: facets in P.h->A, implicit
  // equalities moved to P.h->A_eq. P.v is not needed and not touched.
  inline void remove_redundancy_lp(Polyhedron<double> &P)
  {
    if (!P.h && !P.h_cone) throw std::logic_error("remove_redundancy_lp: P has no H-representation");
    homogenize(P);
    auto &H = *P.h_cone;
    const Facets<double> f = lp_facets(internal::unique_rows(H.A), H.E);
    RowMat<double> E(H.E.rows() + f.implicit.rows(), H.E.cols());
    E << H.E, f.implicit;
    H.A = f.kept;
    H.E = internal::independent_rows(E);
    P.h.reset();
    dehomogenize(P);
  }

} // namespace polycomp