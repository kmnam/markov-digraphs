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
