// All tests in one executable: exit code != 0 on failure.
// One function per header, each run for every scalar type.
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "polycomp/conversion.hpp"
#include "polycomp/eliminate.hpp"
#include "polycomp/normalize.hpp"

using namespace polycomp;

static int failures = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ++failures;                                                              \
      std::cerr << __FILE__ << ":" << __LINE__ << ": " #cond "\n";             \
    }                                                                          \
  } while (0)

// ============================================================================
// Helpers
// ============================================================================

template <class F>
bool throws(F&& f)
{
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}

// Canonical cone form of a row: lift(row(0), rest) as used by homogenize.
// any_sign = true for equalities and lines (defined up to any nonzero factor).
template <class R>
auto canon(const R& a, bool any_sign)
{
  using S = typename R::Scalar;
  const Eigen::Matrix<S, 1, Eigen::Dynamic> row = a;
  Eigen::Matrix<cone_number_t<S>, 1, Eigen::Dynamic> x(row.size());
  internal::lift(row(0), row.tail(row.size() - 1), x, any_sign);
  return x;
}

// a = c * b for some c > 0 (or c != 0 with any_sign).
template <class A, class B>
bool same_direction(const A& a, const B& b, bool any_sign = false)
{
  const auto x = canon(a, any_sign), y = canon(b, any_sign);
  if (x.size() != y.size()) return false;
  for (Eigen::Index i = 0; i < x.size(); ++i)
    if (sign(x(i) - y(i)) != 0) return false;  // tolerance for double, exact otherwise
  return true;
}

// Row [a | b] of an H-representation, for comparing up to scaling.
template <class T, class Arow>
Eigen::Matrix<T, 1, Eigen::Dynamic> hrow(const Arow& a, const T& b)
{
  Eigen::Matrix<T, 1, Eigen::Dynamic> r(a.size() + 1);
  r << a, b;
  return r;
}

// ============================================================================
// normalize.hpp: homogenize / dehomogenize
// ============================================================================

template <class T>
void test_normalize()
{
  using C = cone_number_t<T>;
  using Row = Eigen::Matrix<C, 1, Eigen::Dynamic>;

  // --- H: lifting and round trip (Ziegler p. 33) ---------------------------
  RowMat<T> A(7, 2);
  A << -1, -4, -2, -1, 1, -2, 1, 0, 2, 1, -2, 6, -6, -1;
  Vec<T> z(7);
  z << -9, -4, 0, 4, 11, 17, -6;

  auto P = Polyhedron<T>::from_H(A, z);
  CHECK(P.d == 2 && P.h->A_eq.rows() == 0 && P.h->A_eq.cols() == 2);
  homogenize(P);
  const auto& K = *P.h_cone;
  CHECK(K.A.rows() == 8 && K.A.cols() == 3);
  CHECK(K.E.rows() == 0 && K.E.cols() == 3);
  Row r0(3), x0(3);
  r0 << 9, -1, -4;  // -x1 - 4x2 <= -9
  x0 << -1, 0, 0;   // x0 >= 0
  CHECK(same_direction(K.A.row(0), r0));
  CHECK(same_direction(K.A.row(7), x0));

  P.h.reset();
  dehomogenize(P);
  CHECK(P.h->A.rows() == 7);  // x0 >= 0 dropped
  for (Eigen::Index i = 0; i < 7; ++i)
    CHECK(same_direction(hrow<T>(A.row(i), z(i)), hrow<T>(P.h->A.row(i), P.h->z(i))));

  // --- H: trivial and infeasible inequalities ------------------------------
  {
    RowMat<T> A2(2, 2);
    A2 << 0, 0, 0, 0;
    Vec<T> z2(2);
    z2 << 5, -1;  // 0 <= 5 dropped, 0 <= -1 kept
    auto P2 = Polyhedron<T>::from_H(A2, z2);
    homogenize(P2);
    P2.h.reset();
    dehomogenize(P2);
    CHECK(P2.h->A.rows() == 1 && sign(P2.h->z(0) + T(1)) == 0);
  }

  // --- H: equalities --------------------------------------------------------
  {
    RowMat<T> A3(1, 2), Aeq(4, 2);
    A3 << 1, 1;
    Aeq << -2, 2,   // -2x1 + 2x2 = -1
            2, -2,  //  2x1 - 2x2 =  1  (same hyperplane, opposite sign)
            0, 0,   //  0 = 0  dropped
            0, 0;   //  0 = 3  infeasible, kept as 0 x = 1
    Vec<T> z3(1), zeq(4);
    z3 << 2;
    zeq << -1, 1, 0, 3;
    auto P3 = Polyhedron<T>::from_H(A3, z3, Aeq, zeq);
    homogenize(P3);
    const auto& E = P3.h_cone->E;
    CHECK(E.rows() == 3);
    Row e0(3);
    e0 << 1, -2, 2;  // first nonzero entry positive
    CHECK(same_direction(E.row(0), e0));
    CHECK(same_direction(E.row(1), e0));  // sign normalized: identical to row 0

    P3.h.reset();
    dehomogenize(P3);
    CHECK(P3.h->A_eq.rows() == 3);
    CHECK(same_direction(hrow<T>(Aeq.row(0), zeq(0)), hrow<T>(P3.h->A_eq.row(0), P3.h->z_eq(0)), true));
    CHECK(internal::is_zero(P3.h->A_eq.row(2)) && sign(P3.h->z_eq(2) - T(1)) == 0);
  }

  // --- V: points, rays, lines round trip ------------------------------------
  {
    RowMat<T> V(3, 2), Y(1, 2), L(1, 2);
    V << 0, 0, T(1) / T(2), T(3) / T(2), 1, 2;
    Y << 1, 1;
    L << 0, -3;
    auto Q = Polyhedron<T>::from_V(V, Y, L);
    homogenize(Q);
    CHECK(Q.v_cone->R.rows() == 4 && Q.v_cone->L.rows() == 1);
    Row l0(3);
    l0 << 0, 0, 1;  // (0, 0, -3) canonicalized with positive leading entry
    CHECK(same_direction(Q.v_cone->L.row(0), l0));

    Q.v.reset();
    dehomogenize(Q);
    CHECK(Q.v->V.rows() == 3 && Q.v->Y.rows() == 1 && Q.v->L.rows() == 1);
    for (Eigen::Index i = 0; i < 3; ++i)
      for (Eigen::Index j = 0; j < 2; ++j) CHECK(sign(Q.v->V(i, j) - V(i, j)) == 0);
    CHECK(same_direction(Q.v->Y.row(0), Y.row(0)));
    CHECK(same_direction(Q.v->L.row(0), L.row(0), true));
  }

  // --- V: no points means P is empty; rays and lines are dropped ------------
  {
    RowMat<T> Y(1, 2), L(1, 2);
    Y << 1, 0;
    L << 0, 1;
    auto Q = Polyhedron<T>::from_V(RowMat<T>(0, 2), Y, L);
    CHECK(Q.d == 2);
    homogenize(Q);
    Q.v.reset();
    dehomogenize(Q);
    CHECK(Q.v->V.rows() == 0 && Q.v->Y.rows() == 0 && Q.v->L.rows() == 0);
  }

  // --- invalid input --------------------------------------------------------
  {
    Polyhedron<T> bad;
    bad.d = 2;
    bad.v_cone = typename Polyhedron<T>::VCone{RowMat<C>(1, 3), RowMat<C>(0, 3)};
    bad.v_cone->R << -1, 0, 0;  // x0 < 0
    CHECK(throws([&] { dehomogenize(bad); }));

    Polyhedron<T> bad_line;
    bad_line.d = 2;
    bad_line.v_cone = typename Polyhedron<T>::VCone{RowMat<C>(0, 3), RowMat<C>(1, 3)};
    bad_line.v_cone->L << 1, 0, 0;  // line with x0 != 0
    CHECK(throws([&] { dehomogenize(bad_line); }));

    RowMat<T> A1(1, 2), Aeq3(1, 3), Y3(1, 3);
    A1 << 1, 0;
    Aeq3 << 1, 0, 0;
    Y3 << 1, 0, 0;
    Vec<T> one(1);
    one << 1;
    CHECK(throws([&] { Polyhedron<T>::from_H(A1, Vec<T>(2)); }));      // z too long
    CHECK(throws([&] { Polyhedron<T>::from_H(A1, one, Aeq3, one); }));  // A_eq width
    CHECK(throws([&] { Polyhedron<T>::from_V(A1, Y3); }));              // Y width
  }
}

// ============================================================================
// eliminate.hpp: eliminate_k, histories, Chernikov, Kohler, eliminate_columns
// ============================================================================

// The Fourier-Motzkin system for V -> H of the hexagon (Ziegler p. 33):
// columns [ y0 y1 y2 | lambda_1 .. lambda_6 ], 12 rows.
template <class T>
RowMat<cone_number_t<T>> hexagon_fm_system()
{
  using C = cone_number_t<T>;
  const T half = T(1) / T(2);
  RowMat<T> V(6, 2);
  V << 1, 2, 3, 3 * half, half, 3, 4, 2, 4, 3, 7 * half, 4;
  auto Q = Polyhedron<T>::from_V(V);
  homogenize(Q);
  const auto& R = Q.v_cone->R;
  const Eigen::Index D = 3, n = 6;
  RowMat<C> M = RowMat<C>::Zero(2 * D + n, D + n);
  for (Eigen::Index c = 0; c < D; ++c) {
    M(c, c) = C(1);
    M.row(c).segment(D, n) = -R.col(c).transpose();
    M.row(D + c) = -M.row(c);
  }
  for (Eigen::Index j = 0; j < n; ++j) M(2 * D + j, D + j) = C(-1);
  return M;
}

template <class T>
void test_eliminate()
{
  using C = cone_number_t<T>;

  // --- One step, FM reading: project Ziegler's p. 33 polygon onto x1 -------
  // Homogenized columns are (x0, x1, x2). Rows 5, 6 have x2 > 0 (upper bounds),
  // rows 1, 2, 3, 7 have x2 < 0 (lower bounds), row 4 and x0 >= 0 have x2 = 0.
  RowMat<T> A(7, 2);
  A << -1, -4, -2, -1, 1, -2, 1, 0, 2, 1, -2, 6, -6, -1;
  Vec<T> z(7);
  z << -9, -4, 0, 4, 11, 17, -6;
  auto P = Polyhedron<T>::from_H(A, z);
  homogenize(P);

  const RowMat<C> K = eliminate_k(P.h_cone->A, 2);
  CHECK(K.rows() == 2 + 2 * 4);           // |Z| + |P||N|
  CHECK((K.col(2).array() == 0).all());   // column 2 eliminated

  // Read each row (b0, c, 0) at x0 = 1 as  b0 + c x1 <= 0.
  T lo = T(-1000), hi = T(1000);
  int at_half = 0;
  for (Eigen::Index i = 0; i < K.rows(); ++i) {
    const T b0 = T(K(i, 0)), c = T(K(i, 1));
    if (sign(c) > 0) hi = std::min(hi, T(-b0 / c));
    if (sign(c) < 0) {
      const T bound = T(-b0 / c);
      lo = std::max(lo, bound);
      at_half += sign(T(bound - T(1) / T(2))) == 0;
    }
    if (sign(c) == 0) CHECK(sign(b0) <= 0);  // no contradiction: P is nonempty
  }
  CHECK(sign(T(lo - T(1) / T(2))) == 0);  // proj_2(P) = [1/2, 4]
  CHECK(sign(T(hi - T(4))) == 0);
  CHECK(at_half == 2);  // from pairs (6,2) and (6,7): row 7 is redundant

  // --- One step, DD reading: square conv{(+-1, +-1)} cut by x1 = 0 ---------
  // Eliminating column 1 gives the crossing points of all 2 x 2 pairs,
  // including the diagonals.
  RowMat<T> V(4, 2);
  V << 1, 1, 1, -1, -1, 1, -1, -1;
  auto Q = Polyhedron<T>::from_V(V);
  homogenize(Q);

  const RowMat<C> G = eliminate_k(Q.v_cone->R, 1);
  CHECK(G.rows() == 4);
  int top = 0, bottom = 0, center = 0;
  for (Eigen::Index i = 0; i < G.rows(); ++i) {
    CHECK(sign(G(i, 0)) > 0);  // all are points (x0 > 0)
    const T x2 = T(G(i, 2)) / T(G(i, 0));
    top += sign(T(x2 - T(1))) == 0;
    bottom += sign(T(x2 + T(1))) == 0;
    center += sign(x2) == 0;
  }
  // The segment [-1, 1] on x1 = 0, plus its midpoint twice (from the two
  // diagonals): correct but redundant, the price of pairing non-adjacent points.
  CHECK(top == 1 && bottom == 1 && center == 2);
  CHECK(throws([&] { eliminate_k(G, 3); }));

  // --- Histories: with no bound, the step is exactly the plain kernel ------
  {
    const RowMat<C> M0 = hexagon_fm_system<T>();
    RowMat<C> plain = M0, tracked = M0;
    auto H = Histories::singletons(static_cast<std::size_t>(M0.rows()));
    CHECK(H.rows() == 12 && H.count(5) == 1 && H.test(5, 5));
    for (Eigen::Index k = 3; k < 7; ++k) {  // 4 steps: 12 -> ... -> 2112 rows
      plain = eliminate_k(plain, k);
      tracked = eliminate_k(tracked, k, &H);
      CHECK(plain == tracked);
      CHECK(H.rows() == static_cast<std::size_t>(tracked.rows()));
    }
    CHECK(plain.rows() == 2112);
    // Most of those rows combine more than t + 1 = 5 original rows:
    // exactly the ones Chernikov's rule would never have formed.
    std::size_t over = 0;
    for (std::size_t r = 0; r < H.rows(); ++r) over += H.count(r) > 5;
    CHECK(over > 2000);
  }

  // --- Chernikov: the same 4 steps with bound t + 1 stay small --------------
  {
    RowMat<C> M = hexagon_fm_system<T>();
    auto H = Histories::singletons(static_cast<std::size_t>(M.rows()));
    for (Eigen::Index k = 3; k < 7; ++k) {
      const auto t = static_cast<std::size_t>(k - 3 + 1);
      M = eliminate_k(M, k, &H, Pruning{t + 1});
      for (std::size_t r = 0; r < H.rows(); ++r) CHECK(H.count(r) <= t + 1);
    }
    CHECK(M.rows() < 20);
  }

  // --- Kohler and Adjacency on a hand-built case -----------------------------
  {
    // Rows (1,1), (-1,1), (0,1); eliminating column 0 gives the Z row (0,1)
    // and the combination of rows 0 and 1. With (artificial) histories
    // {0}, {1}, {0}, the combination has history {0,1}, which contains the
    // Z row's history {0}: Kohler drops it as a candidate, Adjacency because
    // the current row 2 lies inside the pair's union.
    RowMat<C> M(3, 2);
    M << 1, 1, -1, 1, 0, 1;
    Histories H0(3, 3);
    H0.set(0, 0);
    H0.set(1, 1);
    H0.set(2, 0);
    Histories Hn = H0, Hk = H0, Ha = H0;
    CHECK(eliminate_k(M, 0, &Hn).rows() == 2);  // no rule
    const RowMat<C> Kk = eliminate_k(M, 0, &Hk, Pruning{3, Pruning::Minimality::kohler});
    const RowMat<C> Ka = eliminate_k(M, 0, &Ha, Pruning{3, Pruning::Minimality::adjacency});
    CHECK(Kk.rows() == 1 && Kk.row(0) == M.row(2));
    CHECK(Ka.rows() == 1 && Ka.row(0) == M.row(2));
  }

  // --- Kohler and Adjacency agree step by step on minimal input --------------
  {
    auto run = [](RowMat<C> M, Eigen::Index first, Eigen::Index last, Pruning::Minimality mode) {
      std::vector<RowMat<C>> steps;
      auto H = Histories::singletons(static_cast<std::size_t>(M.rows()));
      for (Eigen::Index k = first; k < last; ++k) {
        M = eliminate_k(M, k, &H, Pruning{static_cast<std::size_t>(k - first + 2), mode});
        steps.push_back(M);
      }
      return steps;
    };
    // The hexagon's V -> H system, and a random 3-D V -> H system.
    std::vector<RowMat<C>> systems{hexagon_fm_system<T>()};
    {
      std::mt19937 gen(7);
      std::uniform_int_distribution<int> coord(-9, 9);
      RowMat<T> V(14, 3);
      for (Eigen::Index i = 0; i < V.rows(); ++i)
        for (Eigen::Index j = 0; j < 3; ++j) V(i, j) = T(coord(gen));
      auto Q = Polyhedron<T>::from_V(V);
      homogenize(Q);
      const auto& R = Q.v_cone->R;
      const Eigen::Index D = 4, n = R.rows();
      RowMat<C> M = RowMat<C>::Zero(2 * D + n, D + n);
      for (Eigen::Index c = 0; c < D; ++c) {
        M(c, c) = C(1);
        M.row(c).segment(D, n) = -R.col(c).transpose();
        M.row(D + c) = -M.row(c);
      }
      for (Eigen::Index j = 0; j < n; ++j) M(2 * D + j, D + j) = C(-1);
      systems.push_back(M);
    }
    for (const auto& M : systems) {
      // rows = 2D + n and cols = D + n, so D = rows - cols; lambdas follow.
      const Eigen::Index D = M.rows() - M.cols();
      const auto a = run(M, D, M.cols(), Pruning::Minimality::kohler);
      const auto b = run(M, D, M.cols(), Pruning::Minimality::adjacency);
      CHECK(a.size() == b.size());
      for (std::size_t s = 0; s < a.size() && s < b.size(); ++s) CHECK(a[s] == b[s]);
    }

    // eliminate_columns (row pool, shrinking width) computes exactly the same
    // matrix as one eliminate_k per column, for both rules.
    for (const auto& M : systems) {
      const Eigen::Index D = M.rows() - M.cols();
      for (auto mode : {Pruning::Minimality::kohler, Pruning::Minimality::adjacency}) {
        std::vector<Eigen::Index> trace;
        const RowMat<C> pooled = eliminate_columns(M, D, M.cols(), &trace, mode);
        const auto steps = run(M, D, M.cols(), mode);
        CHECK(pooled == steps.back());
        CHECK(trace.size() == steps.size());
        for (std::size_t s = 0; s < trace.size() && s < steps.size(); ++s)
          CHECK(trace[s] == steps[s].rows());
      }
    }
    // ... also when the eliminated range is not at the end: eliminate the
    // first two columns of the hexagon system (y0, y1).
    {
      const RowMat<C> M = hexagon_fm_system<T>();
      const RowMat<C> pooled = eliminate_columns(M, 0, 2, nullptr, Pruning::Minimality::kohler);
      auto H = Histories::singletons(static_cast<std::size_t>(M.rows()));
      RowMat<C> X = eliminate_k(M, 0, &H, Pruning{2, Pruning::Minimality::kohler});
      X = eliminate_k(X, 1, &H, Pruning{3, Pruning::Minimality::kohler});
      CHECK(pooled == X);
      CHECK((pooled.leftCols(2).array() == 0).all());
    }
  }

  // --- eliminate_columns: the whole V -> H elimination of the hexagon -------
  {
    std::vector<Eigen::Index> trace;
    const RowMat<C> M = eliminate_columns(hexagon_fm_system<T>(), 3, 9, &trace);
    CHECK(trace.size() == 6);
    CHECK(*std::max_element(trace.begin(), trace.end()) < 20);
    CHECK((M.rightCols(6).array() == 0).all());  // all lambdas eliminated

    // Ziegler's projection onto x1, done with eliminate_columns instead.
    const RowMat<C> K3 = eliminate_columns(RowMat<C>(P.h_cone->A), 2, 3, nullptr);
    CHECK((K3.col(2).array() == 0).all());
  }
}

// ============================================================================
// conversion.hpp: fourier_motzkin / double_description
// ============================================================================

// x satisfies A x <= z and A_eq x = z_eq.
template <class T, class X>
bool satisfies(const typename Polyhedron<T>::HRep& H, const X& x)
{
  for (Eigen::Index i = 0; i < H.A.rows(); ++i)
    if (sign(T(H.A.row(i).dot(x) - H.z(i))) > 0) return false;
  for (Eigen::Index i = 0; i < H.A_eq.rows(); ++i)
    if (sign(T(H.A_eq.row(i).dot(x) - H.z_eq(i))) != 0) return false;
  return true;
}

// Some row of M equals x (entrywise, up to tolerance).
template <class T, class X>
bool has_row(const RowMat<T>& M, const X& x)
{
  for (Eigen::Index i = 0; i < M.rows(); ++i) {
    bool eq = true;
    for (Eigen::Index j = 0; j < M.cols(); ++j) eq = eq && sign(T(M(i, j) - x(j))) == 0;
    if (eq) return true;
  }
  return false;
}

// Some row of M points in the direction of x (up to positive scaling).
template <class T, class X>
bool has_direction(const RowMat<T>& M, const X& x)
{
  for (Eigen::Index i = 0; i < M.rows(); ++i)
    if (same_direction(M.row(i), x)) return true;
  return false;
}

template <class T>
void test_conversion()
{
  using C = cone_number_t<T>;
  using RowT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
  auto row = [](std::initializer_list<T> v) {
    RowT r(static_cast<Eigen::Index>(v.size()));
    Eigen::Index i = 0;
    for (const T& x : v) r(i++) = x;
    return r;
  };
  const T half = T(1) / T(2);

  // Ziegler p. 33: seven inequalities (row 7 redundant), six vertices.
  RowMat<T> A(7, 2);
  A << -1, -4, -2, -1, 1, -2, 1, 0, 2, 1, -2, 6, -6, -1;
  Vec<T> z(7);
  z << -9, -4, 0, 4, 11, 17, -6;
  RowMat<T> hexagon(6, 2);
  hexagon << 1, 2, 3, 3 * half, half, 3, 4, 2, 4, 3, 7 * half, 4;

  // --- H -> V: exactly the six vertices, no rays -----------------------------
  {
    auto P = Polyhedron<T>::from_H(A, z);
    compute_v_representation(P);
    CHECK(P.v->V.rows() == 6 && P.v->Y.rows() == 0);
    for (Eigen::Index i = 0; i < 6; ++i) CHECK(has_row(P.v->V, hexagon.row(i)));
  }

  // --- V -> H: exactly the six facets (row 7 of Ziegler's system is not one) ---
  {
    auto P = Polyhedron<T>::from_V(hexagon);
    compute_h_representation(P);
    const auto& H = *P.h;
    CHECK(H.A.rows() == 6 && H.A_eq.rows() == 0);
    for (Eigen::Index i = 0; i < 6; ++i) CHECK(satisfies<T>(H, hexagon.row(i).transpose()));
    RowMat<T> Az(H.A.rows(), 3);
    Az << H.A, H.z;
    for (Eigen::Index i = 0; i < 6; ++i) CHECK(has_direction(Az, hrow<T>(A.row(i), z(i))));

    // Round trip V -> H -> V: redundant inequalities do not create extra points.
    auto Q = Polyhedron<T>::from_H(H.A, H.z);
    compute_v_representation(Q);
    CHECK(Q.v->V.rows() == 6);
    for (Eigen::Index i = 0; i < 6; ++i) CHECK(has_row(Q.v->V, hexagon.row(i)));
  }

  // --- H -> V, non-pointed: the half-plane x2 >= 0 ---------------------------
  // One point on the boundary, the ray (0,1), and the x1-axis as a line.
  {
    RowMat<T> A1(1, 2);
    A1 << 0, -1;
    Vec<T> z1(1);
    z1 << 0;
    auto P = Polyhedron<T>::from_H(A1, z1);
    compute_v_representation(P);
    const auto& V = *P.v;
    CHECK(V.V.rows() == 1 && sign(V.V(0, 1)) == 0);
    CHECK(V.Y.rows() == 1 && same_direction(V.Y.row(0), row({0, 1})));
    CHECK(V.L.rows() == 1 && same_direction(V.L.row(0), row({1, 0}), true));
  }

  // --- V -> H, non-pointed: point (0,0), ray (0,1), line (1,0) --------------
  {
    RowMat<T> V(1, 2), Y(1, 2), L(1, 2);
    V << 0, 0;
    Y << 0, 1;
    L << 1, 0;
    auto P = Polyhedron<T>::from_V(V, Y, L);
    compute_h_representation(P);
    const auto& H = *P.h;
    for (Eigen::Index i = 0; i < H.A.rows(); ++i) {
      CHECK(sign(T(-H.z(i))) <= 0);         // point (0,0) satisfies a x <= z
      CHECK(sign(H.A(i, 1)) <= 0);          // ray (0,1): a . y <= 0
      CHECK(sign(H.A(i, 0)) == 0);          // line (1,0): a . l = 0
    }
    RowMat<T> Az(H.A.rows(), 3);
    Az << H.A, H.z;
    CHECK(H.A.rows() == 1 && has_direction(Az, row({0, -1, 0})));  // exactly -x2 <= 0
    CHECK(H.A_eq.rows() == 0);                                     // full-dimensional
  }

  // --- Linearity comes out explicitly on both sides --------------------------
  {
    // V -> H of a segment: the equality x1 = x2 and two bounds, nothing else.
    RowMat<T> V(2, 2);
    V << 0, 0, 1, 1;
    auto P = Polyhedron<T>::from_V(V);
    compute_h_representation(P);
    CHECK(P.h->A.rows() == 2 && P.h->A_eq.rows() == 1);
    CHECK(same_direction(hrow<T>(P.h->A_eq.row(0), P.h->z_eq(0)), row({1, -1, 0}), true));

    // H -> V of a line given only by an equality: one point and one line.
    RowMat<T> Aeq(1, 2);
    Aeq << 1, -1;
    Vec<T> zeq(1);
    zeq << 0;
    auto Q = Polyhedron<T>::from_H(RowMat<T>(0, 2), Vec<T>(0), Aeq, zeq);
    compute_v_representation(Q);
    CHECK(Q.v->V.rows() == 1 && Q.v->Y.rows() == 0 && Q.v->L.rows() == 1);
    CHECK(same_direction(Q.v->L.row(0), row({1, 1}), true));
    CHECK(sign(T(Q.v->V(0, 0) - Q.v->V(0, 1))) == 0);

    // Round trip with a line: V -> H -> V gives one point back, not two.
    RowMat<T> V2(1, 2), Y2(0, 2), L2(1, 2);
    V2 << 1, 0;
    L2 << 1, 1;
    auto R = Polyhedron<T>::from_V(V2, Y2, L2);
    compute_h_representation(R);
    auto R2 = Polyhedron<T>::from_H(R.h->A, R.h->z, R.h->A_eq, R.h->z_eq);
    compute_v_representation(R2);
    CHECK(R2.v->V.rows() == 1 && R2.v->Y.rows() == 0 && R2.v->L.rows() == 1);
  }

  // --- pivot_out: Gaussian substitution of equalities ------------------------
  {
    // x0 + x1 - 2 x2 = 0 and x1 - x2 = 0 pivot on x2 then x1; the inequality
    // x1 + x2 <= 0 (row (0,1,1)) becomes a bound on x0 alone.
    RowMat<C> E(2, 3), M(1, 3);
    E << 1, 1, -2, 0, 1, -1;
    M << 0, 1, 1;
    std::vector<Eigen::Index> remaining;
    pivot_out(M, E, std::vector<Eigen::Index>{2, 1}, remaining);
    CHECK(remaining.empty() && E.rows() == 0);
    CHECK(sign(M(0, 1)) == 0 && sign(M(0, 2)) == 0 && sign(M(0, 0)) > 0);  // 2 x0 <= 0, scaled
    // A column no equality can pivot on stays, and an unused equality survives.
    RowMat<C> E2(1, 3), M2(1, 3);
    E2 << 1, 0, 0;
    M2 << 1, 1, 1;
    pivot_out(M2, E2, std::vector<Eigen::Index>{1}, remaining);
    CHECK(remaining.size() == 1 && remaining[0] == 1 && E2.rows() == 1);
  }

  // --- facets: incidence test, and rank ---------------------------------------
  {
    // The square cone {y : +-y1 <= y0, +-y2 <= y0} with its four rays, plus
    // two redundant rows: y1 + y2 <= 2 y0 (implied) and 0 <= y0 (implied).
    RowMat<C> A(6, 3), R(4, 3), L(0, 3);
    A << -1, 1, 0, -1, -1, 0, -1, 0, 1, -1, 0, -1, -2, 1, 1, -1, 0, 0;
    R << 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1;
    CHECK(internal::rank(R) == 3 && internal::rank(A) == 3);
    const auto f = facets(A, R, L);
    CHECK(f.kept.rows() == 4 && f.implicit.rows() == 0);
    const auto g = facets(R, A, RowMat<C>(0, 3));
    CHECK(g.kept.rows() == 4);  // all four rays are extreme
    // A row tight on every ray is an implicit equality.
    RowMat<C> A2(1, 3);
    A2 << 0, 0, 1;
    RowMat<C> R2(2, 3);
    R2 << 1, 1, 0, 1, -1, 0;
    CHECK(facets(A2, R2, L).implicit.rows() == 1);
  }

  // --- H -> V with an equality: the segment x1 + x2 = 1, x >= 0 -------------
  {
    RowMat<T> A1(2, 2), Aeq(1, 2);
    A1 << -1, 0, 0, -1;
    Aeq << 1, 1;
    Vec<T> z1(2), zeq(1);
    z1 << 0, 0;
    zeq << 1;
    auto P = Polyhedron<T>::from_H(A1, z1, Aeq, zeq);
    const auto H = *P.h;
    compute_v_representation(P);
    CHECK(P.v->Y.rows() == 0);
    CHECK(has_row(P.v->V, row({1, 0})) && has_row(P.v->V, row({0, 1})));
    for (Eigen::Index i = 0; i < P.v->V.rows(); ++i)
      CHECK(satisfies<T>(H, P.v->V.row(i).transpose()));
  }

  // --- H -> V, infeasible: x1 <= -1 and x1 >= 0 ------------------------------
  {
    RowMat<T> A1(2, 1);
    A1 << 1, -1;
    Vec<T> z1(2);
    z1 << -1, 0;
    auto P = Polyhedron<T>::from_H(A1, z1);
    compute_v_representation(P);
    CHECK(P.v->V.rows() == 0);
  }

  // --- project: unit cube onto (x1, x2) is the unit square ------------------
  {
    RowMat<T> A1(6, 3);
    A1 << 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Vec<T> z1(6);
    z1 << 1, 0, 1, 0, 1, 0;
    const auto Q = project(Polyhedron<T>::from_H(A1, z1), 2);
    CHECK(Q.d == 2 && Q.h->A.rows() == 4 && Q.h->A_eq.rows() == 0);
    CHECK(Q.v->V.rows() == 4 && Q.v->Y.rows() == 0 && Q.v->L.rows() == 0);
    for (auto p : {row({0, 0}), row({1, 0}), row({0, 1}), row({1, 1})}) CHECK(has_row(Q.v->V, p));
  }

  // --- project through an equality: x1 = x2 + x3, x2, x3 in [0, 1] -> [0, 2] -
  {
    RowMat<T> A1(4, 3), Aeq(1, 3);
    A1 << 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1;
    Aeq << 1, -1, -1;
    Vec<T> z1(4), zeq(1);
    z1 << 1, 0, 1, 0;
    zeq << 0;
    std::vector<Eigen::Index> trace;
    const auto Q = project(Polyhedron<T>::from_H(A1, z1, Aeq, zeq), 1, &trace);
    CHECK(trace.size() == 1);  // x2 pivoted out by the equality, x3 by Fourier-Motzkin
    CHECK(Q.h->A.rows() == 2 && Q.h->A_eq.rows() == 0);
    CHECK(Q.v->V.rows() == 2 && has_row(Q.v->V, row({0})) && has_row(Q.v->V, row({2})));
  }

  // --- project keeps equalities that do not involve the dropped columns -----
  {
    // x1 = x2, x3 in [0, 1]  ->  the line x1 = x2 in R^2
    RowMat<T> A1(2, 3), Aeq(1, 3);
    A1 << 0, 0, 1, 0, 0, -1;
    Aeq << 1, -1, 0;
    Vec<T> z1(2), zeq(1);
    z1 << 1, 0;
    zeq << 0;
    const auto Q = project(Polyhedron<T>::from_H(A1, z1, Aeq, zeq), 2);
    CHECK(Q.h->A.rows() == 0 && Q.h->A_eq.rows() == 1);
    CHECK(same_direction(hrow<T>(Q.h->A_eq.row(0), Q.h->z_eq(0)), row({1, -1, 0}), true));
    CHECK(Q.v->V.rows() == 1 && Q.v->L.rows() == 1);
    CHECK(same_direction(Q.v->L.row(0), row({1, 1}), true));
  }

  // --- The trace shows Chernikov's rule at work ------------------------------
  {
    auto P = Polyhedron<T>::from_V(hexagon);
    homogenize(P);
    std::vector<Eigen::Index> trace;
    fourier_motzkin(P, &trace);
    // D = 3 of the 6 lambdas are substituted away by the equalities; the
    // other 3 are eliminated by Fourier-Motzkin, each step staying at 6 rows.
    CHECK(trace.size() == 3);
    CHECK(*std::max_element(trace.begin(), trace.end()) <= 6);
  }
}

// ============================================================================

template <class T>
void run_all()
{
  test_normalize<T>();
  test_eliminate<T>();
  test_conversion<T>();
}

int main()
{
  run_all<double>();
#ifdef POLYCOMP_HAS_GMP
  run_all<Rational>();

  // Exact rows become primitive integers: point (1/2, 1/3) -> (1, 1/2, 1/3) -> (6, 3, 2).
  RowMat<Rational> V(1, 2);
  V << Rational(1, 2), Rational(1, 3);
  auto P = Polyhedron<Rational>::from_V(V);
  homogenize(P);
  CHECK(P.v_cone->R(0, 0) == 6 && P.v_cone->R(0, 1) == 3 && P.v_cone->R(0, 2) == 2);
#endif
  std::cout << (failures ? "FAILED\n" : "all tests passed\n");
  return failures != 0;
}