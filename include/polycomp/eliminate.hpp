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
//
// Lines / equalities. A row that holds with equality (an equality a y = 0 on
// the H side, a line on the V side) needs no pairing: if its entry in column
// k is nonzero it is a pivot, and every other row r is replaced by
//     |p_k| r - sign(p_k) r_k p,
// which has r_k = 0, keeps the direction of an inequality (positive factor
// |p_k|) and stays in the cone (any multiple of a line may be added). The
// pivot row is then consumed. This is Gaussian elimination; see pivot_out.

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

    // Keeps the first `rows` histories (or grows with empty ones).
    void resize(std::size_t rows)
    {
      rows_ = rows;
      bits_.resize(rows * words_, 0);
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

    // out = a | b, returning |out|.
    inline std::size_t set_union(const std::uint64_t *a, const std::uint64_t *b, std::uint64_t *out, std::size_t words)
    {
      std::size_t c = 0;
      for (std::size_t w = 0; w < words; ++w) c += static_cast<std::size_t>(std::popcount(out[w] = a[w] | b[w]));
      return c;
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

    // The candidates of one step, decided from signs and histories only.
    // Steps 2 and 3 of an elimination: never touches the numbers.
    inline std::vector<Candidate> select(const SignSplit &split, const Histories *H, const Pruning &rules)
    {
      using Minimality = Pruning::Minimality;
      const auto &[Z, P, N] = split;
      std::vector<Candidate> cand;
      cand.reserve(Z.size() + (H ? 0 : P.size() * N.size()));
      for (Eigen::Index i : Z) cand.push_back({i, -1});

      if (!H)
      {
        for (Eigen::Index i : P)
          for (Eigen::Index j : N) cand.push_back({i, j});
        return cand;
      }

      const std::size_t words = H->words(), m = H->rows();
      const bool adjacency = rules.minimality == Minimality::adjacency;

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
            if (w != ui && union_within(hi, (*H)[w], words, rules.max_history)) neighbours.push_back(w);
        }

        for (Eigen::Index j : N)
        {
          const auto uj = static_cast<std::size_t>(j);
          const auto *hj = (*H)[uj];

          if (!union_within(hi, hj, words, rules.max_history)) continue; // Chernikov

          if (adjacency)
          {
            const std::size_t qsize = set_union(hi, hj, q.data(), words);
            bool refuted = false;
            for (std::size_t w : neighbours)
              if (w != uj && size[w] <= qsize && subset((*H)[w], q.data(), words)) { refuted = true; break; }
            if (refuted) continue;
          }
          cand.push_back({i, j});
        }
      }

      // Kohler: compare the candidates' histories with each other.
      if (rules.minimality == Minimality::kohler)
      {
        const std::size_t c = cand.size();
        Histories Hc(c, words * 64);
        std::vector<std::size_t> csize(c);
        for (std::size_t a = 0; a < c; ++a)
        {
          const auto [i, j] = cand[a];
          const auto *hi = (*H)[static_cast<std::size_t>(i)];
          csize[a] = j < 0 ? set_union(hi, hi, Hc[a], words)
                           : set_union(hi, (*H)[static_cast<std::size_t>(j)], Hc[a], words);
        }
        std::vector<char> keep(c, 1);
        for (std::size_t a = 0; a < c; ++a)
          for (std::size_t b = 0; b < c && keep[a]; ++b)
          {
            if (a == b || !keep[b] || csize[b] > csize[a] || !subset(Hc[b], Hc[a], words)) continue;
            if (csize[b] < csize[a] || b < a) keep[a] = 0; // strict subset, or duplicate
          }
        std::size_t n = 0;
        for (std::size_t a = 0; a < c; ++a)
          if (keep[a]) cand[n++] = cand[a];
        cand.resize(n);
      }
      return cand;
    }
  } // namespace internal

  // One elimination step.
  //
  // Without histories (H == nullptr) this is exactly Theorem 1.4, and `rules`
  // is ignored. With histories (one per row of M, replaced on return by those
  // of the result), `rules` decides which candidates are kept:
  //   rules.max_history  Chernikov bound, t + 1 at the t-th step;
  //   rules.minimality   none, kohler (candidates vs candidates), or adjacency
  //                      (pairs vs current rows; exact if M is itself minimal).
  // Rows that come out zero are dropped in every mode.
  template <class S>
  RowMat<S> eliminate_k(const RowMat<S> &M, Eigen::Index k, Histories *H = nullptr, Pruning rules = {})
  {
    if (k < 0 || k >= M.cols())
      throw std::out_of_range("eliminate_k: k out of range");
    if (H && H->rows() != static_cast<std::size_t>(M.rows()))
      throw std::invalid_argument("eliminate_k: one history per row required");

    // 1-3. Split by sign, list the candidates, apply the rules.
    const auto cand = internal::select(internal::split_by_sign(M, k), H, rules);

    // 4. Arithmetic, only for the kept candidates; histories alongside.
    const std::size_t words = H ? H->words() : 0;
    Histories Hn;
    if (H) Hn = Histories(cand.size(), words * 64);
    RowMat<S> out(static_cast<Eigen::Index>(cand.size()), M.cols());
    Eigen::Index n = 0;
    for (const auto [i, j] : cand)
    {
      if (j < 0)
        out.row(n) = M.row(i);
      else
      {
        out.row(n) = M(i, k) * M.row(j) - M(j, k) * M.row(i);
        if (!internal::scale(out.row(n))) continue; // zero row: drop
      }
      out(n, k) = S(0); // exact zero, even if M(i,k) was only within eps
      if (H)
      {
        const auto *hi = (*H)[static_cast<std::size_t>(i)];
        internal::set_union(hi, j < 0 ? hi : (*H)[static_cast<std::size_t>(j)], Hn[static_cast<std::size_t>(n)], words);
      }
      ++n;
    }
    out.conservativeResize(n, M.cols());
    if (H)
    {
      Hn.resize(static_cast<std::size_t>(n));
      *H = std::move(Hn);
    }
    return out;
  }

  // Eliminate the given columns in order (iterated Theorem 1.4). The rows of
  // M are the original rows for the histories and t counts the eliminations
  // done in this call; each step applies Chernikov (t + 1) and `minimality`.
  // Eliminated columns are left as zeros; the caller drops them.
  template <class S>
  RowMat<S> eliminate_columns(RowMat<S> M, const std::vector<Eigen::Index> &cols,
                              std::vector<Eigen::Index> *trace = nullptr,
                              Pruning::Minimality minimality = Pruning::Minimality::adjacency)
  {
    for (Eigen::Index k : cols)
      if (k < 0 || k >= M.cols()) throw std::out_of_range("eliminate_columns: column out of range");
    auto H = Histories::singletons(static_cast<std::size_t>(M.rows()));
    std::size_t t = 0;
    for (Eigen::Index k : cols)
    {
      M = eliminate_k(M, k, &H, Pruning{++t + 1, minimality});
      if (trace) trace->push_back(M.rows());
    }
    return M;
  }

  // Columns first, ..., last - 1.
  template <class S>
  RowMat<S> eliminate_columns(RowMat<S> M, Eigen::Index first, Eigen::Index last,
                              std::vector<Eigen::Index> *trace = nullptr,
                              Pruning::Minimality minimality = Pruning::Minimality::adjacency)
  {
    if (first < 0 || first > last || last > M.cols())
      throw std::out_of_range("eliminate_columns: invalid column range");
    std::vector<Eigen::Index> cols;
    for (Eigen::Index k = first; k < last; ++k) cols.push_back(k);
    return eliminate_columns(std::move(M), cols, trace, minimality);
  }

  // Gaussian elimination of the columns in `cols`, in order, using the rows
  // of E as pivots (equalities on the H side, lines on the V side).
  //
  // For each column c that some remaining row of E has nonzero, that row p
  // becomes the pivot: every other row r of M and of E is replaced by
  //     |p_c| r - sign(p_c) r_c p,
  // so column c becomes zero everywhere, and p is consumed. Columns with no
  // pivot are left for Fourier-Motzkin and returned in `remaining`. The rows
  // of E that were never used as pivots are returned in E: they are zero on
  // all of `cols` and are equalities (lines) of the projection.
  // Every row is rescaled to canonical form after each pivot.
  template <class S>
  void pivot_out(RowMat<S> &M, RowMat<S> &E, const std::vector<Eigen::Index> &cols,
                 std::vector<Eigen::Index> &remaining)
  {
    remaining.clear();
    std::vector<char> used(static_cast<std::size_t>(E.rows()), 0);
    for (Eigen::Index c : cols)
    {
      Eigen::Index p = -1;
      for (Eigen::Index i = 0; i < E.rows() && p < 0; ++i)
        if (!used[static_cast<std::size_t>(i)] && sign(E(i, c)) != 0) p = i;
      if (p < 0) { remaining.push_back(c); continue; }
      used[static_cast<std::size_t>(p)] = 1;

      const S pc = E(p, c);
      const S apc = sign(pc) > 0 ? pc : S(-pc);
      auto reduce = [&](auto &&row, bool any_sign) {
        if (sign(row(c)) == 0) return;
        row = apc * row - (sign(pc) > 0 ? row(c) : S(-row(c))) * E.row(p);
        row(c) = S(0);
        internal::scale(row, any_sign);
      };
      for (Eigen::Index i = 0; i < M.rows(); ++i) reduce(M.row(i), false);
      for (Eigen::Index i = 0; i < E.rows(); ++i)
        if (i != p) reduce(E.row(i), true);
    }

    // Drop the consumed pivot rows.
    Eigen::Index n = 0;
    for (Eigen::Index i = 0; i < E.rows(); ++i)
      if (!used[static_cast<std::size_t>(i)]) E.row(n++) = E.row(i);
    E.conservativeResize(n, E.cols());
  }

} // namespace polycomp