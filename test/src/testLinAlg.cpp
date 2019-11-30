#define BOOST_TEST_MODULE testLinAlg
#define BOOST_TEST_DYN_LINK
#include <iostream>
#include <string>
#include <utility>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <duals/duals.hpp>
#include <duals/dualMP.hpp>
#include <duals/eigen.hpp>
#include "../../include/linalg.hpp"

/*
 * Test and timing module for the functions in include/linalg.hpp.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/29/2019
 */
std::mt19937 rng(1234567890);

using namespace Eigen;
using namespace boost::unit_test;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::et_on;
using boost::multiprecision::et_off;
typedef number<mpfr_float_backend<30> > mpfr_30;
typedef number<mpfr_float_backend<30>, et_off> mpfr_30_noet;
using Duals::DualNumber;
using Duals::DualMP;
using Duals::DualMatrix;

BOOST_AUTO_TEST_CASE(testPermuteRowsAndColumns)
{
    /*
     * Test that permuteRowsAndColumns() correctly permutes the input matrix.
     */
    MatrixXd A(3, 3);
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    std::pair<MatrixXd, PermutationMatrix<Dynamic, Dynamic> > perm = permuteRowsAndColumns<double>(A, rng);
    MatrixXd B = perm.first;
    PermutationMatrix<Dynamic, Dynamic> P = perm.second;
    BOOST_TEST(A == P.transpose() * B * P.transpose());

    Matrix<mpfr_30, Dynamic, Dynamic> C(3, 3);
    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < 3; ++j)
        {
            mpfr_30 x(i*3.0 + j + 1.0);
            C(i,j) = x;
        }
    }
    std::pair<Matrix<mpfr_30, Dynamic, Dynamic>, PermutationMatrix<Dynamic, Dynamic> >
        perm2 = permuteRowsAndColumns<mpfr_30>(C, rng);
    Matrix<mpfr_30, Dynamic, Dynamic> D = perm2.first;
    PermutationMatrix<Dynamic, Dynamic> Q = perm2.second;
    BOOST_TEST(C == Q.transpose() * D * Q.transpose());
}

BOOST_AUTO_TEST_CASE(testConvert)
{
    /*
     * Test that conversion yields very small differences in values (less
     * than on the order of 1e-17).
     */
    using boost::multiprecision::abs;

    MatrixXd A(3, 3);
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    Matrix<mpfr_30, Dynamic, Dynamic> B = linalg_internal::convert<double, mpfr_30>(A);
    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < 3; ++j)
        {
            std::stringstream ss;
            ss << A(i,j);
            mpfr_30 Aij(ss.str());
            BOOST_TEST(abs(Aij - B(i,j)) < 1e-17);
        }
    }
}

BOOST_AUTO_TEST_CASE(testConvertDual)
{
    /*
     * Test that conversion yields very small differences in values (less
     * than on the order of 1e-17).
     */
    using boost::multiprecision::abs;

    MatrixXDual A(3, 3);
    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < 3; ++j)
        {
            DualNumber v(3*i + j + 1, i * j);
            A(i,j) = v;
        }
    }
    Matrix<DualMP<30, et_off>, Dynamic, Dynamic> B = linalg_internal::convertDual<30, et_off>(A);
    for (unsigned i = 0; i < 3; ++i)
    {
        for (unsigned j = 0; j < 3; ++j)
        {
            std::stringstream ssx, ssd;
            ssx << A(i,j).x();
            ssd << A(i,j).d();
            mpfr_30 Aijx(ssx.str());
            mpfr_30 Aijd(ssd.str());
            BOOST_TEST(abs(Aijx - B(i,j).x()) < 1e-17);
            BOOST_TEST(abs(Aijd - B(i,j).d()) < 1e-17);
        }
    }
}

BOOST_AUTO_TEST_CASE(testSpanningTreeWeightVector)
{
    /*
     * Given a Laplacian matrix, compute the spanning tree weight
     * vector with the two functions. 
     */
    // Define a Laplacian matrix for a 3-state graph (nullspace: [4, 2, 1])
    MatrixXd laplacian(3, 3);
    laplacian <<  1, -1,  0,
                 -2,  3, -1,
                  0, -2,  2;

    // Compute the spanning tree weight vector with doubles
    std::pair<VectorXd, double> r1 = linalg_internal::spanningTreeWeightVector<double>(laplacian);
    BOOST_TEST(r1.first(0) == 4);
    BOOST_TEST(r1.first(1) == 2);
    BOOST_TEST(r1.first(2) == 1);

    // Compute the spanning tree weight vector with DualNumbers
    MatrixXDual laplacian_dual(3, 3);
    laplacian_dual(0, 0) = DualNumber(1, 1);
    laplacian_dual(0, 1) = DualNumber(-1, -1);
    laplacian_dual(0, 2) = DualNumber(0, 0);
    laplacian_dual(1, 0) = DualNumber(-2, 0);
    laplacian_dual(1, 1) = DualNumber(3, 0);
    laplacian_dual(1, 2) = DualNumber(-1, 0);
    laplacian_dual(2, 0) = DualNumber(0, 0);
    laplacian_dual(2, 1) = DualNumber(-2, 0);
    laplacian_dual(2, 2) = DualNumber(2, 0);
    std::pair<VectorXDual, DualNumber> r2 = linalg_internal::spanningTreeWeightVector<DualNumber>(laplacian_dual);
    BOOST_TEST(r2.first(0).x() == 4);
    BOOST_TEST(r2.first(1).x() == 2);
    BOOST_TEST(r2.first(2).x() == 1);
    BOOST_TEST(r2.first(0).d() == 0);
    BOOST_TEST(r2.first(1).d() == 2);
    BOOST_TEST(r2.first(2).d() == 1);

    // Compute the spanning tree weight vector with DualMatrix
    MatrixXd derivative(3, 3);
    derivative << 1, -1, 0,
                  0,  0, 0,
                  0,  0, 0;
    DualMatrix laplacian_dualm(laplacian, derivative);
    std::pair<DualMatrix, double> r3 = linalg_internal::spanningTreeWeightVector(laplacian_dualm);
    BOOST_TEST(r3.first.X()(0) == 4);
    BOOST_TEST(r3.first.X()(1) == 2);
    BOOST_TEST(r3.first.X()(2) == 1);
    BOOST_TEST(r3.first.D()(0) == 0);
    BOOST_TEST(r3.first.D()(1) == 2);
    BOOST_TEST(r3.first.D()(2) == 1);
}

BOOST_DATA_TEST_CASE(timeMultiply, data::xrange(2, 7))
{
    /*
     * Time matrix multiplication with double, boost::multiprecision, 
     * DualNumber, and DualMP types.  
     */
    using std::pow;
    unsigned dim = pow(2, sample);

    // Start with double matrices
    MatrixXd A = MatrixXd::Random(dim, dim);
    MatrixXd B = MatrixXd::Random(dim, dim);
    auto start = std::chrono::high_resolution_clock::now();
    MatrixXd C = A * B;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Double " << dim << "-by-" << dim << " matrix multiplication: "
              << elapsed.count() << " seconds" << std::endl;

    // Do the same with boost::multiprecision::number types
    Matrix<mpfr_30, Dynamic, Dynamic> D = linalg_internal::convert<double, mpfr_30>(A);
    Matrix<mpfr_30, Dynamic, Dynamic> E = linalg_internal::convert<double, mpfr_30>(B);
    start = std::chrono::high_resolution_clock::now();
    Matrix<mpfr_30, Dynamic, Dynamic> F = D * E;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "mpfr_30 " << dim << "-by-" << dim << " matrix multiplication: "
              << elapsed.count() << " seconds" << std::endl;

    // Do the same with boost::multiprecision::number types without expression templates
    Matrix<mpfr_30_noet, Dynamic, Dynamic> D2 = linalg_internal::convert<double, mpfr_30_noet>(A);
    Matrix<mpfr_30_noet, Dynamic, Dynamic> E2 = linalg_internal::convert<double, mpfr_30_noet>(B);
    start = std::chrono::high_resolution_clock::now();
    Matrix<mpfr_30_noet, Dynamic, Dynamic> F2 = D2 * E2;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "mpfr_30_noet " << dim << "-by-" << dim << " matrix multiplication: "
              << elapsed.count() << " seconds" << std::endl;

    // Do the same with DualNumber types
    MatrixXDual G = MatrixXDual::Random(dim, dim);
    MatrixXDual H = MatrixXDual::Random(dim, dim);
    start = std::chrono::high_resolution_clock::now();
    MatrixXDual I = G * H;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DualNumber " << dim << "-by-" << dim << " matrix multiplication: "
              << elapsed.count() << " seconds" << std::endl;

    // Do the same with DualMP<30, et_on> types
    Matrix<DualMP<30, et_on>, Dynamic, Dynamic> J = linalg_internal::convertDual<30, et_on>(G);
    Matrix<DualMP<30, et_on>, Dynamic, Dynamic> K = linalg_internal::convertDual<30, et_on>(H);
    start = std::chrono::high_resolution_clock::now();
    Matrix<DualMP<30, et_on>, Dynamic, Dynamic> L = J * K;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DualMP<30, et_on> " << dim << "-by-" << dim << " matrix multiplication: "
              << elapsed.count() << " seconds" << std::endl;

    // Do the same with DualMP<30, et_off> types
    Matrix<DualMP<30, et_off>, Dynamic, Dynamic> J2 = linalg_internal::convertDual<30, et_off>(G);
    Matrix<DualMP<30, et_off>, Dynamic, Dynamic> K2 = linalg_internal::convertDual<30, et_off>(H);
    start = std::chrono::high_resolution_clock::now();
    Matrix<DualMP<30, et_off>, Dynamic, Dynamic> L2 = J2 * K2;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DualMP<30, et_off> " << dim << "-by-" << dim << " matrix multiplication: "
              << elapsed.count() << " seconds" << std::endl;

    // Do the same with DualMatrix types
    DualMatrix M(A, B);
    DualMatrix N(B, A);
    start = std::chrono::high_resolution_clock::now();
    DualMatrix O = M * N;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DualMatrix " << dim << "-by-" << dim << " matrix multiplication: "
              << elapsed.count() << " seconds" << std::endl;

    // Do the same with DualMatrixMP<30, et_on> types
    DualMatrixMP<30, et_on> P(D, E);
    DualMatrixMP<30, et_on> Q(E, D);
    start = std::chrono::high_resolution_clock::now();
    DualMatrixMP<30, et_on> R = P * Q;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DualMatrixMP<30, et_on> " << dim << "-by-" << dim << " matrix multiplication: "
              << elapsed.count() << " seconds" << std::endl;

    // Do the same with DualMatrixMP<30, et_off> types
    DualMatrixMP<30, et_off> P2(D2, E2);
    DualMatrixMP<30, et_off> Q2(E2, D2);
    start = std::chrono::high_resolution_clock::now();
    DualMatrixMP<30, et_off> R2 = P2 * Q2;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DualMatrixMP<30, et_off> " << dim << "-by-" << dim << " matrix multiplication: "
              << elapsed.count() << " seconds" << std::endl << std::endl;

}

