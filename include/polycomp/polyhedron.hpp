#pragma once
// A polyhedron P in R^d:
//   H:  P = { x : A x <= z,  A_eq x = z_eq }
//   V:  P = conv(rows of V) + cone(rows of Y) + span(rows of L)
// and the homogenized cones in R^{d+1} (column 0 is x0, P is the slice x0 = 1):
//   h_cone:  { y : A y <= 0,  E y = 0 }      rows (-z, a) and (-z_eq, a_eq), plus x0 >= 0
//   v_cone:  cone(rows of R) + span(rows of L)   rows (1, v), (0, y) and lines (0, l)
//
// points, rays, lines and generators are ROWS (Ziegler uses columns).
// cone rows are defined up to scaling (positive for A and R, any nonzero for E and L).


#include <Eigen/Dense>
#include <cstddef>
#include <optional>
#include <stdexcept>

#include <utility>
#include <string>

#ifdef POLYCOMP_HAS_GMP
#include <boost/multiprecision/eigen.hpp>
#include <boost/multiprecision/gmp.hpp>
#endif

namespace polycomp
{

#ifdef POLYCOMP_HAS_GMP
  namespace mp = boost::multiprecision;
  using Rational = mp::number<mp::gmp_rational, mp::et_off>;
  using Integer = mp::number<mp::gmp_int, mp::et_off>;
#endif

  template <class T>
  using RowMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  template <class T>
  using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  // Scalar type of the cones. Floating point stays floating point; exact
  // (Rational) input becomes fraction-free Integer rows.
  template <class T>
  struct cone_number
  {
    using type = T;
  };
#ifdef POLYCOMP_HAS_GMP
  template <>
  struct cone_number<Rational>
  {
    using type = Integer;
  };
#endif
  template <class T>
  using cone_number_t = typename cone_number<T>::type;

  template <class T>
  struct Polyhedron
  {
    using C = cone_number_t<T>;
    struct HRep  { RowMat<T> A; Vec<T> z; RowMat<T> A_eq; Vec<T> z_eq; };
    struct VRep  { RowMat<T> V; RowMat<T> Y; RowMat<T> L; };
    struct HCone { RowMat<C> A; RowMat<C> E; };  // A y <= 0,  E y = 0
    struct VCone { RowMat<C> R; RowMat<C> L; };  // cone(R) + span(L)

    Eigen::Index d = 0;
    std::optional<HRep> h;
    std::optional<VRep> v;
    std::optional<HCone> h_cone;
    std::optional<VCone> v_cone;

    static Polyhedron from_H(RowMat<T> A, Vec<T> z, RowMat<T> A_eq = {}, Vec<T> z_eq = {})
    {
      if (A.rows() != z.size())
        throw std::invalid_argument("from_H: A and z row counts differ");
      if (A_eq.rows() != z_eq.size())
        throw std::invalid_argument("from_H: A_eq and z_eq row counts differ");
      Polyhedron P;
      P.d = A.cols() ? A.cols() : A_eq.cols();
      fit_width(A, P.d, "from_H: A");
      fit_width(A_eq, P.d, "from_H: A_eq");
      P.h = HRep{std::move(A), std::move(z), std::move(A_eq), std::move(z_eq)};
      return P;
    }

    // Any of V, Y, L may be empty. If V has no rows, P is empty.
    static Polyhedron from_V(RowMat<T> V, RowMat<T> Y = {}, RowMat<T> L = {})
    {
      Polyhedron P;
      P.d = V.cols() ? V.cols() : (Y.cols() ? Y.cols() : L.cols());
      fit_width(V, P.d, "from_V: V");
      fit_width(Y, P.d, "from_V: Y");
      fit_width(L, P.d, "from_V: L");
      P.v = VRep{std::move(V), std::move(Y), std::move(L)};
      return P;
    }

  private:
    // Empty matrices get width d; non-empty ones must already have it.
    static void fit_width(RowMat<T>& M, Eigen::Index d, const char* what)
    {
      if (M.rows() == 0) M.resize(0, d);
      else if (M.cols() != d)
        throw std::invalid_argument(std::string(what) + " has the wrong number of columns");
    }
  };

} // namespace polycomp
