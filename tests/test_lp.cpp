// redundancy_lp.hpp against the incidence test of redundancy.hpp (which
// needs both cones) and against hand-known answers.
#include <chrono>
#include <iostream>
#include <random>

#include "polycomp/conversion.hpp"
#include "polycomp/redundancy_lp.hpp"

using namespace polycomp;
using RowD = Eigen::Matrix<double, 1, Eigen::Dynamic>;

static int failures = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ++failures;                                                              \
      std::cerr << __FILE__ << ":" << __LINE__ << ": " #cond "\n";             \
    }                                                                          \
  } while (0)

// Rows (a, z) of P.h as canonical cone rows (-z, a), for comparison up to scaling.
RowMat<double> cone_rows(const Polyhedron<double>::HRep& H)
{
  RowMat<double> K(H.A.rows(), H.A.cols() + 1);
  for (Eigen::Index i = 0; i < H.A.rows(); ++i) internal::lift(-H.z(i), H.A.row(i), K.row(i));
  return K;
}

bool same_rows(const RowMat<double>& A, const RowMat<double>& B)
{
  if (A.rows() != B.rows()) return false;
  for (Eigen::Index i = 0; i < A.rows(); ++i) {
    bool found = false;
    for (Eigen::Index j = 0; j < B.rows() && !found; ++j) found = (A.row(i) - B.row(j)).cwiseAbs().maxCoeff() < 1e-6;
    if (!found) return false;
  }
  return true;
}

int main()
{
  // --- Unit square with planted redundant rows -----------------------------
  {
    RowMat<double> A(8, 2);
    Vec<double> z(8);
    A << 1, 0, -1, 0, 0, 1, 0, -1,  // the square [0, 1]^2
         1, 1,                      // x1 + x2 <= 3      redundant
         2, 0,                      // 2 x1 <= 2         duplicate of x1 <= 1
         1, 0,                      // x1 <= 1           exact duplicate
         1, 0;                      // x1 <= 5           weaker
    z << 1, 0, 1, 0, 3, 2, 1, 5;
    auto P = Polyhedron<double>::from_H(A, z);
    remove_redundancy_lp(P);
    CHECK(P.h->A.rows() == 4 && P.h->A_eq.rows() == 0);
  }

  // --- Implicit equality: x1 + x2 <= 1, x1 + x2 >= 1, x >= 0 ---------------
  {
    RowMat<double> A(4, 2);
    Vec<double> z(4);
    A << 1, 1, -1, -1, -1, 0, 0, -1;
    z << 1, -1, 0, 0;
    auto P = Polyhedron<double>::from_H(A, z);
    remove_redundancy_lp(P);
    CHECK(P.h->A.rows() == 2 && P.h->A_eq.rows() == 1);
  }

  // --- Empty: x1 <= -1, x1 >= 0 ---------------------------------------------
  {
    RowMat<double> A(2, 1);
    Vec<double> z(2);
    A << 1, -1;
    z << -1, 0;
    auto P = Polyhedron<double>::from_H(A, z);
    remove_redundancy_lp(P);
    // C = {0}: every row is an implicit equality; what is left is the
    // inconsistent pair x1 = -1, x1 = 0.
    CHECK(P.h->A.rows() == 0 && P.h->A_eq.rows() == 2);
  }

  // --- Random polytopes: LP test == incidence test ---------------------------
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1, 1), W(0, 1);
  for (int d : {3, 4, 5}) {
    // Facets of the hull of 40 random points, by FM + incidence.
    RowMat<double> V(40, d);
    for (Eigen::Index i = 0; i < V.rows(); ++i)
      for (int j = 0; j < d; ++j) V(i, j) = U(rng);
    auto P = Polyhedron<double>::from_V(V);
    compute_h_representation(P);
    const RowMat<double> F = cone_rows(*P.h);

    // The same facets mixed with 3x as many redundant rows: positive
    // combinations of two facets, loosened by a random slack.
    const Eigen::Index f = P.h->A.rows(), m = 4 * f;
    RowMat<double> A(m, d);
    Vec<double> z(m);
    A.topRows(f) = P.h->A;
    z.head(f) = P.h->z;
    for (Eigen::Index i = f; i < m; ++i) {
      const Eigen::Index p = rng() % f, q = rng() % f;
      const double s = W(rng), r = W(rng);
      A.row(i) = s * P.h->A.row(p) + r * P.h->A.row(q);
      z(i) = s * P.h->z(p) + r * P.h->z(q) + 0.1 * W(rng);
    }
    std::vector<Eigen::Index> perm(static_cast<size_t>(m));
    for (Eigen::Index i = 0; i < m; ++i) perm[size_t(i)] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    RowMat<double> As(m, d);
    Vec<double> zs(m);
    for (Eigen::Index i = 0; i < m; ++i) { As.row(i) = A.row(perm[size_t(i)]); zs(i) = z(perm[size_t(i)]); }

    auto Q = Polyhedron<double>::from_H(As, zs);
    const auto t0 = std::chrono::steady_clock::now();
    remove_redundancy_lp(Q);
    const auto t1 = std::chrono::steady_clock::now();
    CHECK(Q.h->A_eq.rows() == 0);
    CHECK(same_rows(cone_rows(*Q.h), F));
    std::cout << "d=" << d << ": " << m << " rows -> " << Q.h->A.rows() << " facets (incidence: " << f << "), "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
  }

  // --- Contact wrench cone in double: 16 rows of U plus redundant ones -------
  {
    const double X = 2, Y = 1, mu = 0.5, s = (X + Y) * mu;
    RowMat<double> Uc(16, 6);
    Uc << -1, 0, -mu, 0, 0, 0,     1, 0, -mu, 0, 0, 0,     0, -1, -mu, 0, 0, 0,    0, 1, -mu, 0, 0, 0,
           0, 0, -Y, -1, 0, 0,     0, 0, -Y, 1, 0, 0,      0, 0, -X, 0, -1, 0,     0, 0, -X, 0, 1, 0,
          -Y, -X, -s, mu, mu, -1, -Y, X, -s, mu, -mu, -1,  Y, -X, -s, -mu, mu, -1, Y, X, -s, -mu, -mu, -1,
           Y, X, -s, mu, mu, 1,    Y, -X, -s, mu, -mu, 1, -Y, X, -s, -mu, mu, 1,  -Y, -X, -s, -mu, -mu, 1;
    RowMat<double> A(32, 6);
    A.topRows(16) = Uc;
    for (Eigen::Index i = 0; i < 16; ++i) A.row(16 + i) = Uc.row(i) + Uc.row((i + 5) % 16);  // implied
    auto P = Polyhedron<double>::from_H(A, Vec<double>::Zero(32));
    remove_redundancy_lp(P);
    CHECK(P.h->A.rows() == 16 && P.h->A_eq.rows() == 0);
  }

  std::cout << (failures ? "FAILED\n" : "all tests passed\n");
  return failures != 0;
}