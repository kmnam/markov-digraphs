#define BOOST_TEST_MODULE testLinAlg
#define BOOST_TEST_DYN_LINK
#include <iostream>
#include <string>
#include <utility>
#include <random>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <boost/test/included/unit_test.hpp>
#include <duals.hpp>
#include <dualMP.hpp>
#include <duals-eigen/eigen.hpp>
#include "../../include/digraph.hpp"

/*
 * Test module for the functions in include/linalg.hpp.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/22/2019
 */
std::mt19937 rng(1234567890);

using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
typedef number<mpfr_float_backend<30> > mpfr_30;
using Duals::DualNumber;
using Duals::DualMP;

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
    Matrix<DualMP<30>, Dynamic, Dynamic> B = linalg_internal::convertDual<30>(A);
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
