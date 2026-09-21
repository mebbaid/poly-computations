#pragma once
//  Lectures
//   FM (Thm 1.4):  rows are inequalities; the result describes elim_k, i.e.
//                  the projection along e_k.
//   DD (V^{/k}):   rows are generators; the result generates the
//                  intersection with the hyperplane y_k = 0.
//
// two redundancy rules in multipliers
//   Chernikov: an extreme ray has at most t + 1 elements in its history.
//   Kohler:    an extreme ray's history contains no other row's history.
// Rows violating either rule are redundant. Both rules look at histories
// only (no arithmetic). Both rules remove redundancy in multiplier space not in y space.

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "polycomp/normalize.hpp"

namespace polycomp
{

  // Ziegler Theorem 1.4

  template <class S>
  RowMat<S> eliminate_k(const RowMat<S> &M, Eigen::Index k)
  {
    if (k < 0 || k >= M.cols())
      throw std::out_of_range("eliminate_k: k out of range");

    // Split the rows by the sign of their entry in column k.
    std::vector<Eigen::Index> Z, P, N;
    for (Eigen::Index i = 0; i < M.rows(); ++i)
    {
      const int s = sign(M(i, k));
      (s == 0 ? Z : s > 0 ? P
                          : N)
          .push_back(i);
    }

    // Allocate once for the largest possible result.
    const auto nz = static_cast<Eigen::Index>(Z.size());
    const auto np = static_cast<Eigen::Index>(P.size());
    const auto nn = static_cast<Eigen::Index>(N.size());
    RowMat<S> out(nz + np * nn, M.cols());
    Eigen::Index n = 0;

    // Rows not involving column k pass through.
    for (Eigen::Index i : Z)
    {
      out.row(n) = M.row(i);
      out(n++, k) = S(0); // exact zero, even if M(i,k) was only within eps
    }

    // Every (positive, negative) pair gives one combination with column k = 0.
    for (Eigen::Index i : P)
      for (Eigen::Index j : N)
      {
        out.row(n) = M(i, k) * M.row(j) - M(j, k) * M.row(i);
        out(n, k) = S(0);
        if (internal::scale(out.row(n)))
          ++n; // canonical form; zero rows dropped
      }

    out.conservativeResize(n, M.cols());
    return out;
  }

  // history[b] == true  <=>  original row b was used to build this row.
  using History = std::vector<bool>;

  // Row b of the original matrix has history {b}.
  inline std::vector<History> initial_histories(Eigen::Index rows)
  {
    const auto r = static_cast<std::size_t>(rows);
    std::vector<History> H(r, History(r, false));
    for (std::size_t b = 0; b < r; ++b)
      H[b][b] = true;
    return H;
  }

  inline std::size_t history_size(const History &h)
  {
    return static_cast<std::size_t>(std::count(h.begin(), h.end(), true));
  }

  // a is a subset of b.
  inline bool history_subset(const History &a, const History &b)
  {
    for (std::size_t x = 0; x < a.size(); ++x)
      if (a[x] && !b[x])
        return false;
    return true;
  }

  // Chernikov's rule
  // Same as eliminate_k(M, k), plus:
  //   - H[i] is the history of row i of M; on return H holds the histories of
  //     the returned rows (the history of a combination is the union);
  //   - a pair (i, j) whose union has more than max_history elements is
  //     skipped before any arithmetic

  template <class S>
  RowMat<S> eliminate_k(const RowMat<S> &M, Eigen::Index k, std::vector<History> &H,
                        std::size_t max_history)
  {
    if (k < 0 || k >= M.cols())
      throw std::out_of_range("eliminate_k: k out of range");
    if (static_cast<Eigen::Index>(H.size()) != M.rows())
      throw std::invalid_argument("eliminate_k: one history per row required");

    std::vector<Eigen::Index> Z, P, N;
    for (Eigen::Index i = 0; i < M.rows(); ++i)
    {
      const int s = sign(M(i, k));
      (s == 0 ? Z : s > 0 ? P
                          : N)
          .push_back(i);
    }

    const auto nz = static_cast<Eigen::Index>(Z.size());
    const auto np = static_cast<Eigen::Index>(P.size());
    const auto nn = static_cast<Eigen::Index>(N.size());
    RowMat<S> out(nz + np * nn, M.cols());
    std::vector<History> out_H;
    Eigen::Index n = 0;

    for (Eigen::Index i : Z)
    {
      out.row(n) = M.row(i);
      out(n++, k) = S(0);
      out_H.push_back(H[static_cast<std::size_t>(i)]); // history unchanged
    }

    for (Eigen::Index i : P)
      for (Eigen::Index j : N)
      {
        const History &hi = H[static_cast<std::size_t>(i)];
        const History &hj = H[static_cast<std::size_t>(j)];
        History h(hi.size());
        for (std::size_t b = 0; b < h.size(); ++b)
          h[b] = hi[b] || hj[b]; // union
        if (history_size(h) > max_history)
          continue; // Chernikov

        out.row(n) = M(i, k) * M.row(j) - M(j, k) * M.row(i);
        out(n, k) = S(0);
        if (!internal::scale(out.row(n)))
          continue;
        ++n;
        out_H.push_back(std::move(h));
      }

    out.conservativeResize(n, M.cols());
    H = std::move(out_H);
    return out;
  }

  // Kohler's rule
  // Drops every row whose history strictly contains another row's history .
  // Rows with identical histories are the same ray, so only the first is kept.
  template <class S>
  RowMat<S> remove_non_minimal(const RowMat<S> &M, std::vector<History> &H)
  {
    const auto r = static_cast<std::size_t>(M.rows());
    if (H.size() != r)
      throw std::invalid_argument("remove_non_minimal: one history per row required");

    std::vector<bool> keep(r, true);
    for (std::size_t a = 0; a < r; ++a)
      for (std::size_t b = 0; b < r && keep[a]; ++b)
      {
        if (a == b || !keep[b] || !history_subset(H[b], H[a]))
          continue;
        if (H[b] != H[a] || b < a)
          keep[a] = false; // strict subset, or duplicate
      }

    RowMat<S> out(M.rows(), M.cols());
    std::vector<History> out_H;
    Eigen::Index n = 0;
    for (std::size_t a = 0; a < r; ++a)
      if (keep[a])
      {
        out.row(n++) = M.row(static_cast<Eigen::Index>(a));
        out_H.push_back(std::move(H[a]));
      }
    out.conservativeResize(n, M.cols());
    H = std::move(out_H);
    return out;
  }

  // eliminate columns first, ..., last - 1 in order (Fourier-Motzkin elimination). 
  // The rows of M are the "original" rows for the histories, and t
  // counts the eliminations done in this call, so after the t-th step only
  // combinations of at most t + 1 original rows are formed (Chernikov) and
  // rows with non-minimal histories are dropped (Kohler).
  // Eliminated columns are left as zeros; the caller drops them.

  template <class S>
  RowMat<S> eliminate_columns(RowMat<S> M, Eigen::Index first, Eigen::Index last,
                              std::vector<Eigen::Index> *trace)
  {
    auto H = initial_histories(M.rows());
    for (Eigen::Index k = first; k < last; ++k)
    {
      const auto t = static_cast<std::size_t>(k - first + 1);
      M = eliminate_k(M, k, H, t + 1);
      M = remove_non_minimal(M, H);
      if (trace)
        trace->push_back(M.rows());
    }
    return M;
  }

} // namespace polycomp