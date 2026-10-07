#pragma once
// Ziegler, Lectures on Polytopes, Theorem 1.4 and the V^{/k} construction.
//
// eliminate_k(M, k): rows of M are inequalities (Fourier-Motzkin) or
// generators (double description). Rows with M(i,k) = 0 pass through, and
// every pair with M(i,k) > 0 > M(j,k) gives the positive combination
// M(i,k) M_j - M(j,k) M_i, whose k-th entry is 0. Nothing else.
//
// Histories. Every row is sum_b lambda_b m_b over the ORIGINAL rows m_b,
// lambda >= 0, and its history is supp(lambda). After t eliminations the
// rows needed are the extreme rays of the multiplier cone
//     Lambda_t = { lambda >= 0 : (M^T lambda)_k = 0 for the t eliminated k },
// which is pointed (it lies in the orthant). Three rules look at histories
// only, never at numbers, and drop rows that are not extreme rays:
//
//   Chernikov  an extreme ray of Lambda_t has at most t + 1 elements in its
//              history.
//   Kohler     an extreme ray's history contains no other candidate's
//              history. Compares the candidates of this step with each other,
//              so all candidates must be listed first.
//   Adjacency  (Bastrakov-Zolotykh 2015; the combinatorial adjacency test of
//              the double description method). If the current rows are the
//              extreme rays of Lambda_{t-1}, the combination of u and v is an
//              extreme ray of Lambda_t iff no other CURRENT row w has
//              history(w) within history(u) | history(v). Compares a pair with
//              the current rows only, so it runs before anything is generated
//              and every pair is decided independently.
//
// On input that is itself minimal, Kohler and Adjacency keep exactly the same
// rows. All three remove redundancy in multiplier space, not in y space.

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

    // Appends an empty history and returns it.
    std::uint64_t *append()
    {
      bits_.resize(++rows_ * words_, 0);
      return (*this)[rows_ - 1];
    }

    // Keeps the first `rows` histories.
    void truncate(std::size_t rows)
    {
      rows_ = rows;
      bits_.resize(rows * words_);
    }

  private:
    std::size_t rows_ = 0, words_ = 0;
    std::vector<std::uint64_t> bits_;
  };

  // Which history rules an elimination step applies.
  struct Pruning
  {
    enum class Minimality { none, kohler, adjacency };

    std::size_t max_history = std::numeric_limits<std::size_t>::max(); // Chernikov: t + 1
    Minimality minimality = Minimality::none;
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

    struct SignSplit { std::vector<Eigen::Index> Z, P, N; };

    template <class S>
    SignSplit split_by_sign(const RowMat<S> &M, Eigen::Index k)
    {
      SignSplit s;
      for (Eigen::Index i = 0; i < M.rows(); ++i)
      {
        const int sg = sign(M(i, k));
        (sg == 0 ? s.Z : sg > 0 ? s.P : s.N).push_back(i);
      }
      return s;
    }

    // (i, -1) passes row i through; (i, j) combines i in P with j in N.
    struct Candidate { Eigen::Index i, j; };

    // Kohler: keep[a] = 0 if some other kept candidate's history is a strict
    // subset of a's, or equal to it and listed earlier.
    inline void kohler(const Histories &Hc, std::vector<char> &keep)
    {
      const std::size_t c = Hc.rows(), words = Hc.words();
      std::vector<std::size_t> size(c);
      for (std::size_t a = 0; a < c; ++a) size[a] = Hc.count(a);
      for (std::size_t a = 0; a < c; ++a)
        for (std::size_t b = 0; b < c && keep[a]; ++b)
        {
          if (a == b || !keep[b] || size[b] > size[a]) continue;
          if (!subset(Hc[b], Hc[a], words)) continue;
          if (size[b] < size[a] || b < a) keep[a] = 0;
        }
    }
  } // namespace internal

  // One elimination step.
  //
  // Without histories (H == nullptr) this is exactly Theorem 1.4, and `rules`
  // is ignored. With histories, `rules` decides which candidates are kept:
  //   rules.max_history  Chernikov bound, t + 1 at the t-th step;
  //   rules.minimality   none, kohler (candidates vs candidates), or adjacency
  //                      (pairs vs current rows; exact if M is itself minimal).
  // Rows that come out zero are dropped in every mode.
  template <class S>
  RowMat<S> eliminate_k(const RowMat<S> &M, Eigen::Index k, Histories *H = nullptr,
                        Pruning rules = {})
  {
    using internal::Candidate;
    using Minimality = Pruning::Minimality;
    if (k < 0 || k >= M.cols())
      throw std::out_of_range("eliminate_k: k out of range");
    if (H && H->rows() != static_cast<std::size_t>(M.rows()))
      throw std::invalid_argument("eliminate_k: one history per row required");

    // 1. Split the rows by the sign of their entry in column k.
    const auto [Z, P, N] = internal::split_by_sign(M, k);

    // 2. List the candidates. Without histories: all of Z and all of P x N.
    //    With histories: each pair is tested before it is listed, and its
    //    history (the union) is written into Hc.
    std::vector<Candidate> cand;
    Histories Hc;
    const std::size_t words = H ? H->words() : 0;
    if (H) Hc = Histories(0, words * 64);

    cand.reserve(Z.size() + (H ? 0 : P.size() * N.size()));
    for (Eigen::Index i : Z)
    {
      cand.push_back({i, -1});
      if (H)
      {
        auto *h = Hc.append();
        const auto *hi = (*H)[static_cast<std::size_t>(i)];
        for (std::size_t w = 0; w < words; ++w) h[w] = hi[w];
      }
    }

    if (!H)
    {
      for (Eigen::Index i : P)
        for (Eigen::Index j : N) cand.push_back({i, j});
    }
    else
    {
      const bool adjacency = rules.minimality == Minimality::adjacency;
      const std::size_t m = H->rows();

      // Sizes of the current histories, for the adjacency test's size filter.
      std::vector<std::size_t> size(adjacency ? m : 0);
      for (std::size_t w = 0; w < size.size(); ++w) size[w] = H->count(w);

      std::vector<std::size_t> neighbours; // rows that could refute pairs (i, .)
      std::vector<std::uint64_t> q(words);  // history of the current pair

      for (Eigen::Index i : P)
      {
        const auto ui = static_cast<std::size_t>(i);
        const auto *hi = (*H)[ui];

        // A refuter w of a pair (i, j) has history(w) within the pair's
        // union, so |history(w) u history(i)| <= |union| <= max_history:
        // it is a neighbour of i in the Chernikov graph. Collect those once
        // per i (Bastrakov-Zolotykh); every pair (i, j) scans only them.
        if (adjacency)
        {
          neighbours.clear();
          for (std::size_t w = 0; w < m; ++w)
            if (w != ui && internal::union_within(hi, (*H)[w], words, rules.max_history))
              neighbours.push_back(w);
        }

        for (Eigen::Index j : N)
        {
          const auto uj = static_cast<std::size_t>(j);
          const auto *hj = (*H)[uj];

          // Chernikov.
          if (!internal::union_within(hi, hj, words, rules.max_history)) continue;

          std::size_t qsize = 0;
          for (std::size_t w = 0; w < words; ++w)
            qsize += static_cast<std::size_t>(std::popcount(q[w] = hi[w] | hj[w]));

          // Adjacency: refuted by any other current row inside the union.
          if (adjacency)
          {
            bool refuted = false;
            for (std::size_t w : neighbours)
              if (w != uj && size[w] <= qsize && internal::subset((*H)[w], q.data(), words))
              {
                refuted = true;
                break;
              }
            if (refuted) continue;
          }

          auto *h = Hc.append();
          for (std::size_t w = 0; w < words; ++w) h[w] = q[w];
          cand.push_back({i, j});
        }
      }
    }

    // 3. Kohler, if asked for: needs the complete candidate list.
    std::vector<char> keep(cand.size(), 1);
    if (H && rules.minimality == Minimality::kohler) internal::kohler(Hc, keep);

    // 4. Arithmetic, only for the kept candidates.
    std::size_t kept = 0;
    for (char c : keep) kept += c;
    RowMat<S> out(static_cast<Eigen::Index>(kept), M.cols());
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

  // Eliminate columns first, ..., last - 1 in order (iterated Theorem 1.4).
  // The rows of M are the original rows for the histories and t counts the
  // eliminations done in this call; each step applies Chernikov (t + 1) and
  // the adjacency test. Eliminated columns are left as zeros; the caller
  // drops them.
  template <class S>
  RowMat<S> eliminate_columns(RowMat<S> M, Eigen::Index first, Eigen::Index last,
                              std::vector<Eigen::Index> *trace = nullptr,
                              Pruning::Minimality minimality = Pruning::Minimality::adjacency)
  {
    if (first < 0 || first > last || last > M.cols())
      throw std::out_of_range("eliminate_columns: invalid column range");
    auto H = Histories::singletons(static_cast<std::size_t>(M.rows()));
    for (Eigen::Index k = first; k < last; ++k)
    {
      const auto t = static_cast<std::size_t>(k - first + 1);
      M = eliminate_k(M, k, &H, Pruning{t + 1, minimality});
      if (trace)
        trace->push_back(M.rows());
    }
    return M;
  }

} // namespace polycomp