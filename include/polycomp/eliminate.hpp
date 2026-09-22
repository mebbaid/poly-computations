#pragma once
// Ziegler, Lectures on Polytopes, Theorem 1.4 and the V^{/k} construction.
//
// Two redundancy rules that look at histories only (no arithmetic):
//   Chernikov: an extreme ray has at most t + 1 elements in its history.
//   Kohler:    an extreme ray's history contains no other row's history.
// Both remove redundancy in multiplier space, not in y space.
//

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "polycomp/normalize.hpp"

namespace polycomp
{

  // Bit b of row i is set  <=>  original row b was used to build row i.
  class Histories
  {
  public:
    Histories() = default;

    Histories(std::size_t rows, std::size_t originals)
        : rows_(rows), words_((originals + 63) / 64), bits_(rows * words_, 0) {}

    // Row b has history {b}: the start of every elimination.
    static Histories singletons(std::size_t rows)
    {
      Histories H(rows, rows);
      for (std::size_t b = 0; b < rows; ++b) H.set(b, b);
      return H;
    }

    std::size_t rows() const { return rows_; }
    std::size_t words() const { return words_; }
    std::uint64_t *operator[](std::size_t i) { return bits_.data() + i * words_; }
    const std::uint64_t *operator[](std::size_t i) const { return bits_.data() + i * words_; }

    void set(std::size_t i, std::size_t b) { (*this)[i][b / 64] |= std::uint64_t{1} << (b % 64); }
    bool test(std::size_t i, std::size_t b) const { return ((*this)[i][b / 64] >> (b % 64)) & 1u; }

    std::size_t count(std::size_t i) const
    {
      std::size_t c = 0;
      for (std::size_t w = 0; w < words_; ++w) c += static_cast<std::size_t>(std::popcount((*this)[i][w]));
      return c;
    }

    std::uint64_t *append()
    {
      bits_.resize(++rows_ * words_, 0);
      return (*this)[rows_ - 1];
    }

    void truncate(std::size_t rows)
    {
      rows_ = rows;
      bits_.resize(rows * words_);
    }

  private:
    std::size_t rows_ = 0, words_ = 0;
    std::vector<std::uint64_t> bits_;
  };

  namespace internal
  {
    // |a u b| <= limit, stopping as soon as the limit is exceeded.
    inline bool union_within(const std::uint64_t *a, const std::uint64_t *b, std::size_t words,
                             std::size_t limit)
    {
      std::size_t c = 0;
      for (std::size_t w = 0; w < words; ++w)
        if ((c += static_cast<std::size_t>(std::popcount(a[w] | b[w]))) > limit) return false;
      return true;
    }

    // a is a subset of b.
    inline bool subset(const std::uint64_t *a, const std::uint64_t *b, std::size_t words)
    {
      for (std::size_t w = 0; w < words; ++w)
        if (a[w] & ~b[w]) return false;
      return true;
    }
  } // namespace internal

  // Vanilla Theorem 1.4. And with the redundancy rules: 
  // (Chernikov: max_history = t + 1 at the t-th step), and if `minimal`,
  // rows whose history strictly contains another's, or equals an earlier
  // one's, are dropped (Kohler).
  template <class S>
  RowMat<S> eliminate_k(const RowMat<S> &M, Eigen::Index k, Histories *H = nullptr,
                        std::size_t max_history = std::numeric_limits<std::size_t>::max(),
                        bool minimal = false)
  {
    if (k < 0 || k >= M.cols())
      throw std::out_of_range("eliminate_k: k out of range");
    if (H && H->rows() != static_cast<std::size_t>(M.rows()))
      throw std::invalid_argument("eliminate_k: one history per row required");

    // 1. Split the rows by the sign of their entry in column k.
    std::vector<Eigen::Index> Z, P, N;
    for (Eigen::Index i = 0; i < M.rows(); ++i)
    {
      const int s = sign(M(i, k));
      (s == 0 ? Z : s > 0 ? P : N).push_back(i);
    }

    // 2. Candidates: (i, -1) passes row i through, (i, j) combines i in P
    //    with j in N. Their histories are written into Hc as they are listed.
    struct Candidate { Eigen::Index i, j; };
    std::vector<Candidate> cand;
    Histories Hc;
    const std::size_t words = H ? H->words() : 0;
    auto push_history = [&](const std::uint64_t *a, const std::uint64_t *b) {
      std::uint64_t *h = Hc.append();
      for (std::size_t w = 0; w < words; ++w) h[w] = b ? (a[w] | b[w]) : a[w];
    };
    if (H) Hc = Histories(0, words * 64);

    cand.reserve(Z.size() + (H ? 0 : P.size() * N.size()));
    for (Eigen::Index i : Z)
    {
      cand.push_back({i, -1});
      if (H) push_history((*H)[static_cast<std::size_t>(i)], nullptr);
    }
    for (Eigen::Index i : P)
      for (Eigen::Index j : N)
      {
        if (H)
        {
          const auto *hi = (*H)[static_cast<std::size_t>(i)];
          const auto *hj = (*H)[static_cast<std::size_t>(j)];
          if (!internal::union_within(hi, hj, words, max_history)) continue; // Chernikov
          push_history(hi, hj);
        }
        cand.push_back({i, j});
      }

    // 3. Kohler.
    std::vector<char> keep(cand.size(), 1);
    if (H && minimal)
    {
      std::vector<std::size_t> size(cand.size());
      for (std::size_t c = 0; c < cand.size(); ++c) size[c] = Hc.count(c);
      for (std::size_t a = 0; a < cand.size(); ++a)
        for (std::size_t b = 0; b < cand.size() && keep[a]; ++b)
        {
          if (a == b || !keep[b] || size[b] > size[a]) continue;
          if (!internal::subset(Hc[b], Hc[a], words)) continue;
          if (size[b] < size[a] || b < a) keep[a] = 0; // strict subset, or duplicate
        }
    }

    // 4. Arithmetic on remaining rows.
    std::size_t survivors = 0;
    for (char c : keep) survivors += c;
    RowMat<S> out(static_cast<Eigen::Index>(survivors), M.cols());
    Eigen::Index n = 0;
    for (std::size_t c = 0; c < cand.size(); ++c)
    {
      if (!keep[c]) continue;
      const auto [i, j] = cand[c];
      if (j < 0)
        out.row(n) = M.row(i);
      else
      {
        out.row(n) = M(i, k) * M.row(j) - M(j, k) * M.row(i);
        if (!internal::scale(out.row(n))) continue; // zero row: drop
      }
      out(n, k) = S(0); // exact zero, even if M(i,k) was only within eps
      if (H)            // compact histories in place (n <= c)
        for (std::size_t w = 0; w < words; ++w) Hc[static_cast<std::size_t>(n)][w] = Hc[c][w];
      ++n;
    }

    out.conservativeResize(n, M.cols());
    if (H)
    {
      Hc.truncate(static_cast<std::size_t>(n));
      *H = std::move(Hc);
    }
    return out;
  }

  // Eliminate columns first, ..., last - 1 in order - iterated Theorem 1.4.
  template <class S>
  RowMat<S> eliminate_columns(RowMat<S> M, Eigen::Index first, Eigen::Index last,
                              std::vector<Eigen::Index> *trace = nullptr)
  {
    if (first < 0 || first > last || last > M.cols())
      throw std::out_of_range("eliminate_columns: invalid column range");
    auto H = Histories::singletons(static_cast<std::size_t>(M.rows()));
    for (Eigen::Index k = first; k < last; ++k)
    {
      const auto t = static_cast<std::size_t>(k - first + 1);
      M = eliminate_k(M, k, &H, t + 1, true);
      if (trace)
        trace->push_back(M.rows());
    }
    return M;
  }

} // namespace polycomp