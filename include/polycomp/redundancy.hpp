#pragma once
// Redundancy in y-space, decided with both descriptions of the same cone.
//
// Given inequalities A (rows a, a y <= 0) and generators R (rays r) with
// lines L of the SAME cone C in R^D, the face of C cut out by a is
//     F_a = cone{ r : a . r = 0 } + span(L),
// and a is a facet iff dim F_a = dim C - 1, where dim C = rank([R; L]).
// If a is tight on every generator, F_a = C: a is an implicit equality.
//
// The same test with the roles swapped decides extremality of a ray:
//     r is extreme iff rank({ a : a . r = 0 } u E) = rank([A; E]) - 1,
// and a ray tight on every inequality is a line (r and -r are in C).
//
// So one function, facets(rows, gens, gen_lines), serves both sides, and
// remove_redundancy(P) applies it to h_cone against v_cone and to v_cone
// against h_cone. Nothing here needs histories; it needs both cones.

#include <stdexcept>
#include <utility>
#include <vector>

#include "polycomp/normalize.hpp"

namespace polycomp
{

  namespace internal
  {
    // Rank by Gaussian elimination: fraction-free (p_c r - r_c p, then
    // canonical scaling) for exact numbers, partial pivoting for doubles.
    template <class S>
    Eigen::Index rank(RowMat<S> M)
    {
      using RowV = Eigen::Matrix<S, 1, Eigen::Dynamic>;
      Eigen::Index r = 0;
      for (Eigen::Index c = 0; c < M.cols() && r < M.rows(); ++c)
      {
        Eigen::Index p = -1;
        for (Eigen::Index i = r; i < M.rows(); ++i)
        {
          if (sign(M(i, c)) == 0) continue;
          if constexpr (std::is_floating_point_v<S>)
          {
            if (p < 0 || std::abs(M(i, c)) > std::abs(M(p, c))) p = i;
          }
          else { p = i; break; }
        }
        if (p < 0) continue;
        if (p != r) M.row(r).swap(M.row(p));
        const RowV pivot = M.row(r);
        for (Eigen::Index i = r + 1; i < M.rows(); ++i)
        {
          if (sign(M(i, c)) == 0) continue;
          M.row(i) = pivot(c) * M.row(i) - M(i, c) * pivot;
          M(i, c) = S(0);
          scale(M.row(i), true);
        }
        ++r;
      }
      return r;
    }

    // Rows of M that are linearly independent (greedy, in order).
    template <class S>
    RowMat<S> independent_rows(const RowMat<S> &M)
    {
      RowMat<S> B(M.rows(), M.cols());
      Eigen::Index n = 0;
      for (Eigen::Index i = 0; i < M.rows(); ++i)
      {
        B.row(n) = M.row(i);
        if (rank(RowMat<S>(B.topRows(n + 1))) == n + 1) ++n;
      }
      B.conservativeResize(n, M.cols());
      return B;
    }

    // Exact duplicates removed (rows are canonical, so equality is exact).
    template <class S>
    RowMat<S> unique_rows(const RowMat<S> &M)
    {
      RowMat<S> U(M.rows(), M.cols());
      Eigen::Index n = 0;
      for (Eigen::Index i = 0; i < M.rows(); ++i)
      {
        bool seen = false;
        for (Eigen::Index j = 0; j < n && !seen; ++j) seen = U.row(j) == M.row(i);
        if (!seen) U.row(n++) = M.row(i);
      }
      U.conservativeResize(n, M.cols());
      return U;
    }
  } // namespace internal

  template <class S>
  struct Facets
  {
    RowMat<S> kept;     // rows that define facets (resp. extreme rays)
    RowMat<S> implicit; // rows tight on every generator: equalities (resp. lines)
  };

  // Facet test by generator incidence; see the header comment.
  template <class S>
  Facets<S> facets(const RowMat<S> &rows, const RowMat<S> &gens, const RowMat<S> &gen_lines)
  {
    const Eigen::Index D = rows.cols(), nl = gen_lines.rows();
    RowMat<S> all(gens.rows() + nl, D);
    all << gens, gen_lines;
    const Eigen::Index dim = internal::rank(all);

    Facets<S> out{RowMat<S>(rows.rows(), D), RowMat<S>(rows.rows(), D)};
    Eigen::Index nk = 0, ni = 0;
    RowMat<S> tight(gens.rows() + nl, D);
    for (Eigen::Index i = 0; i < rows.rows(); ++i)
    {
      Eigen::Index nt = 0;
      for (Eigen::Index g = 0; g < gens.rows(); ++g)
        if (sign(S(rows.row(i).dot(gens.row(g)))) == 0) tight.row(nt++) = gens.row(g);
      if (nt == gens.rows()) { out.implicit.row(ni++) = rows.row(i); continue; }
      tight.middleRows(nt, nl) = gen_lines;
      if (internal::rank(RowMat<S>(tight.topRows(nt + nl))) == dim - 1) out.kept.row(nk++) = rows.row(i);
    }
    out.kept.conservativeResize(nk, D);
    out.implicit.conservativeResize(ni, D);
    return out;
  }

  // Makes both cones of P minimal: h_cone.A keeps its facets only, with
  // implicit equalities moved to E; v_cone.R keeps its extreme rays only,
  // with implicit lines moved to L. Requires both cones.
  template <class T>
  void remove_redundancy(Polyhedron<T> &P)
  {
    using C = cone_number_t<T>;
    if (!P.h_cone || !P.v_cone)
      throw std::logic_error("remove_redundancy: both h_cone and v_cone are required");
    auto &H = *P.h_cone;
    auto &V = *P.v_cone;

    const Facets<C> fa = facets(internal::unique_rows(H.A), V.R, V.L);
    const Facets<C> fr = facets(internal::unique_rows(V.R), H.A, H.E);

    RowMat<C> E(H.E.rows() + fa.implicit.rows(), H.E.cols());
    E << H.E, fa.implicit;
    RowMat<C> L(V.L.rows() + fr.implicit.rows(), V.L.cols());
    L << V.L, fr.implicit;

    H.A = fa.kept;
    H.E = internal::independent_rows(E);
    V.R = fr.kept;
    V.L = internal::independent_rows(L);
  }

} // namespace polycomp