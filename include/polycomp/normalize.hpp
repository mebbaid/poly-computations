#pragma once
// Ziegler §1.1: polyhedron in R^d  <->  cone in R^{d+1} (column 0 is x0).
//
//   homogenize(P):   h -> h_cone:  a x <= z  ->  A row (-z, a),  plus (-1, 0..0) for x0 >= 0
//                                  a x  = z  ->  E row (-z, a)
//                    v -> v_cone:  point v   ->  R row (1, v)
//                                  ray y     ->  R row (0, y)
//                                  line l    ->  L row (0, l)
//   dehomogenize(P): slice the cones at x0 = 1 to fill h / v if missing.
//
// Cone rows are put in canonical form via scaling

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "polycomp/polyhedron.hpp"

namespace polycomp {

inline double eps = 1e-9;  // sign tolerance for floating point

template <class S>
int sign(const S& x)
{
  if constexpr (std::is_floating_point_v<S>) return (x > eps) - (x < -eps);
  else return (x > 0) - (x < 0);
}

namespace internal {

// True if every entry of r has sign 0.
template <class Row>
bool is_zero(const Row& r)
{
  for (Eigen::Index i = 0; i < r.size(); ++i)
    if (sign(r(i)) != 0) return false;
  return true;
}

// Canonical form of r up to positive scaling or, with any_sign, up to any
// nonzero scaling (then the first nonzero entry is made positive).
template <class Row>
bool scale(Row&& r, bool any_sign = false)
{
  using S = typename std::decay_t<Row>::Scalar;
  constexpr bool fp = std::is_floating_point_v<S>;

  // Positive normalizer: max |entry| (floating point) or gcd of entries (exact).
  S m = 0;
  if constexpr (fp) m = r.cwiseAbs().maxCoeff();
  else for (Eigen::Index i = 0; i < r.size(); ++i) m = gcd(m, r(i));  // boost gcd, >= 0
  if (sign(m) == 0) return false;

  // For lines/equalities, divide by -m instead if the first nonzero entry is negative.
  if (any_sign)
    for (Eigen::Index i = 0; i < r.size(); ++i) {
      bool nonzero;
      if constexpr (fp) nonzero = std::abs(r(i)) > eps * m;  // relative to the row
      else nonzero = r(i) != 0;
      if (nonzero) {
        if (r(i) < 0) m = -m;
        break;
      }
    }

  r /= m;
  if constexpr (fp) r = r.unaryExpr([](S x) { return std::abs(x) < eps ? S(0) : x; });
  return true;
}

// out = canonical (lead, tail) in cone numbers. Returns false if the row is zero.
template <class T, class Tail, class Out>
bool lift(const T& lead, const Eigen::MatrixBase<Tail>& tail, Out&& out, bool any_sign = false)
{
  const Eigen::Index n = tail.size();
  if constexpr (std::is_same_v<T, cone_number_t<T>>) {  // double, Integer: copy
    out(0) = lead;
    out.tail(n) = tail;
  }
#ifdef POLYCOMP_HAS_GMP
  else {  // Rational -> Integer: clear denominators
    Integer l = denominator(lead);
    for (Eigen::Index i = 0; i < n; ++i) l = lcm(l, denominator(tail(i)));
    auto to_int = [&](const Rational& q) { return Integer(numerator(q) * (l / denominator(q))); };
    out(0) = to_int(lead);
    for (Eigen::Index i = 0; i < n; ++i) out(i + 1) = to_int(tail(i));
  }
#endif
  return scale(out, any_sign);
}

}  // namespace internal

template <class T>
void homogenize(Polyhedron<T>& P)
{
  using C = cone_number_t<T>;
  const Eigen::Index D = P.d + 1;

  if (P.h && !P.h_cone) {
    const auto& H = *P.h;
    typename Polyhedron<T>::HCone K{RowMat<C>(H.A.rows() + 1, D), RowMat<C>(H.A_eq.rows(), D)};

    Eigen::Index k = 0;
    for (Eigen::Index i = 0; i < H.A.rows(); ++i)
      if (internal::lift(T(-H.z(i)), H.A.row(i), K.A.row(k))) ++k;  // 0 <= 0 dropped
    K.A.row(k).setZero();
    K.A(k++, 0) = C(-1);                                             // x0 >= 0
    K.A.conservativeResize(k, D);

    Eigen::Index e = 0;
    for (Eigen::Index i = 0; i < H.A_eq.rows(); ++i)
      if (internal::lift(T(-H.z_eq(i)), H.A_eq.row(i), K.E.row(e), true)) ++e;  // 0 = 0 dropped
    K.E.conservativeResize(e, D);

    P.h_cone = std::move(K);
  }

  if (P.v && !P.v_cone) {
    const auto& Vr = *P.v;
    typename Polyhedron<T>::VCone K{RowMat<C>(Vr.V.rows() + Vr.Y.rows(), D), RowMat<C>(Vr.L.rows(), D)};

    Eigen::Index k = 0;
    for (Eigen::Index i = 0; i < Vr.V.rows(); ++i)
      if (internal::lift(T(1), Vr.V.row(i), K.R.row(k))) ++k;
    for (Eigen::Index i = 0; i < Vr.Y.rows(); ++i)
      if (internal::lift(T(0), Vr.Y.row(i), K.R.row(k))) ++k;        // zero rays dropped
    K.R.conservativeResize(k, D);

    Eigen::Index l = 0;
    for (Eigen::Index i = 0; i < Vr.L.rows(); ++i)
      if (internal::lift(T(0), Vr.L.row(i), K.L.row(l), true)) ++l;   // zero lines dropped
    K.L.conservativeResize(l, D);

    P.v_cone = std::move(K);
  }
}

template <class T>
void dehomogenize(Polyhedron<T>& P)
{
  const Eigen::Index d = P.d;

  // A row (b0, a) -> a x <= -b0. If a == 0 the row is trivially true or infeasible, kept as 0 x <= -1.
  // E row (b0, a) -> a x  = -b0. If a == 0 it is 0 = 0 (dropped) or infeasible, kept as 0 x = 1.
  if (P.h_cone && !P.h) {
    const auto& K = *P.h_cone;
    typename Polyhedron<T>::HRep H{RowMat<T>(K.A.rows(), d), Vec<T>(K.A.rows()),
                                   RowMat<T>(K.E.rows(), d), Vec<T>(K.E.rows())};

    Eigen::Index k = 0;
    for (Eigen::Index i = 0; i < K.A.rows(); ++i) {
      const bool a_zero = internal::is_zero(K.A.row(i).tail(d));
      if (a_zero && sign(K.A(i, 0)) <= 0) continue;
      H.A.row(k) = K.A.row(i).tail(d).template cast<T>();
      H.z(k++) = a_zero ? T(-1) : T(0) - T(K.A(i, 0));  // 0 - b0, not -b0: avoids -0.0
    }
    H.A.conservativeResize(k, d);
    H.z.conservativeResize(k);

    Eigen::Index e = 0;
    for (Eigen::Index i = 0; i < K.E.rows(); ++i) {
      const bool a_zero = internal::is_zero(K.E.row(i).tail(d));
      if (a_zero && sign(K.E(i, 0)) == 0) continue;
      H.A_eq.row(e) = K.E.row(i).tail(d).template cast<T>();
      H.z_eq(e++) = a_zero ? T(1) : T(0) - T(K.E(i, 0));
    }
    H.A_eq.conservativeResize(e, d);
    H.z_eq.conservativeResize(e);

    P.h = std::move(H);
  }

  // R row (x0, x): x0 > 0 -> point x / x0;  x0 = 0 -> ray x.
  // L row (0, l) -> line l. x0 < 0 in R, or x0 != 0 in L, cannot come from
  // a polyhedron (x0 >= 0 always holds) and is an error.
  if (P.v_cone && !P.v) {
    const auto& K = *P.v_cone;

    Eigen::Index np = 0;
    for (Eigen::Index i = 0; i < K.R.rows(); ++i) {
      if (sign(K.R(i, 0)) < 0) throw std::domain_error("dehomogenize: generator with x0 < 0");
      np += sign(K.R(i, 0)) > 0;
    }
    for (Eigen::Index i = 0; i < K.L.rows(); ++i)
      if (sign(K.L(i, 0)) != 0) throw std::domain_error("dehomogenize: line with x0 != 0");

    typename Polyhedron<T>::VRep Vr{RowMat<T>(np, d), RowMat<T>(K.R.rows() - np, d),
                                    RowMat<T>(K.L.rows(), d)};
    Eigen::Index p = 0, r = 0;
    for (Eigen::Index i = 0; i < K.R.rows(); ++i) {
      if (sign(K.R(i, 0)) > 0)
        Vr.V.row(p++) = K.R.row(i).tail(d).template cast<T>() / T(K.R(i, 0));
      else
        Vr.Y.row(r++) = K.R.row(i).tail(d).template cast<T>();
    }
    for (Eigen::Index i = 0; i < K.L.rows(); ++i)
      Vr.L.row(i) = K.L.row(i).tail(d).template cast<T>();

    if (np == 0) {  // no points: P is empty, rays and lines are meaningless
      Vr.Y.resize(0, d);
      Vr.L.resize(0, d);
    }
    P.v = std::move(Vr);
  }
}

}  // namespace polycomp