#ifndef LINALG_HPP
#define LINALG_HPP

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <iterator>
#include <Eigen/Dense>

/*
 * Various linear algebra functions.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/22/2019
 */
using namespace Eigen;

template <typename T>
bool isclose(T a, T b, T tol)
{
    /*
     * Return true if abs(a - b) < tol.
     */
    T c = a - b;
    return ((c >= 0.0 && c < tol) || (c < 0.0 && -c < tol));
}

template <typename T>
Matrix<T, Dynamic, Dynamic> signedStirlingNumbersOfFirstKindByFactorial(unsigned n)
{
    /*
     * Return a lower-triangular matrix whose (i,j)-th entry is the
     * signed Stirling number of the first kind divided by i!,
     * s(i,j) / i!, for i >= j.
     */
    Matrix<T, Dynamic, Dynamic> S = Matrix<T, Dynamic, Dynamic>::Zero(n+1, n+1);

    // Initialize by setting S(0,0) = 1, S(k,0) = S(0,k) = 0
    S.row(0) = Matrix<T, 1, Dynamic>::Zero(1, n+1);
    S.col(0) = Matrix<T, Dynamic, 1>::Zero(n+1, 1);
    S(0,0) = 1.0;

    // Apply recurrence relation:
    // S(i,j) = s(i,j) / i! = -(i-1) * s(i-1,j) / i! + s(i-1,j-1) / i!
    //        = -((i-1) / i) * s(i-1,j) / (i-1)! + (1 / i) * s(i-1,j-1) / i!
    //        = -((i-1) / i) * S(i-1,j) + (1 / i) * S(i-1,j-1)
    for (unsigned i = 1; i < n+1; i++)
    {
        for (unsigned j = 1; j < i+1; j++)
            S(i,j) = -((i - 1.0) / i) * S(i-1,j) + (1.0 / i) * S(i-1,j-1);
    }

    // Return with row 0 and column 0 excised
    return S.block(1, 1, n, n);
}

namespace linalg_internal {

template <typename StdType, typename BoostMPFRType>
Matrix<BoostMPFRType, Dynamic, Dynamic> convert(const Ref<const Matrix<StdType, Dynamic, Dynamic> >& A)
{
    /*
     * Convert a matrix with double scalars into a matrix with the given Boost
     * multiprecision floating point types. 
     */
    Matrix<BoostMPFRType, Dynamic, Dynamic> B(A.rows(), A.cols());
    for (unsigned i = 0; i < A.rows(); ++i)
    {
        for (unsigned j = 0; j < A.cols(); ++j)
        {
            std::stringstream ss;
            ss << std::setprecision(std::numeric_limits<StdType>::max_digits10);
            ss << A(i,j);
            BoostMPFRType x(ss.str());
            B(i,j) = x;
        }
    }
    return B;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> nullspaceSVD(const Ref<const Matrix<T, Dynamic, Dynamic> >& A, T sv_tol)
{
    /*
     * Return a matrix whose columns form a basis for the nullspace of A.
     */
    // Perform a full singular value decomposition of A
    Eigen::BDCSVD<Matrix<T, Dynamic, Dynamic> > sv_decomp(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Initialize nullspace basis matrix
    Matrix<T, Dynamic, Dynamic> nullmat;
    unsigned ncols = 0;
    unsigned nrows = A.cols();

    // Run through the singular values of A (in ascending order) ...
    Matrix<T, Dynamic, Dynamic> S = sv_decomp.singularValues();
    Matrix<T, Dynamic, Dynamic> V = sv_decomp.matrixV();
    unsigned nsingvals = S.rows();
    unsigned j = nsingvals - 1;
    while (isclose<T>(S(j), 0.0, sv_tol) && j >= 0)
    {
        // ... and grab the columns of V that correspond to the zero
        // singular values
        ncols += 1;
        nullmat.resize(nrows, ncols);
        nullmat.col(ncols - 1) = V.col(j);
        j--;
    }

    return nullmat;
}

template <typename T>
std::pair<Matrix<T, Dynamic, 1>, T> spanningTreeWeightVector(const Ref<const Matrix<T, Dynamic, Dynamic> >& laplacian)
{
    /*
     * Use the recurrence of Chebotarev & Agaev (Lin Alg Appl, 2002, Eqs. 17-18)
     * for the spanning tree weight vector of the given Laplacian matrix.
     *
     * This function does not check that the given matrix is indeed a
     * valid row Laplacian matrix (zero row sums, positive diagonal,
     * negative off-diagonal). 
     */
    unsigned dim = laplacian.rows();
    Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(dim, dim);
    Matrix<T, Dynamic, Dynamic> weights = Matrix<T, Dynamic, Dynamic>::Identity(dim, dim);
    for (unsigned k = 1; k < dim; ++k)
    {
        T sigma = (laplacian * weights).trace() / k;
        weights = -laplacian * weights + sigma * identity;
    }

    // Return the row of the weight matrix whose product with the (negative transpose of)
    // Laplacian matrix has the smallest norm
    MatrixXd::Index min_i;
    Matrix<T, Dynamic, 1> sqnorm = (weights * (-laplacian)).rowwise().squaredNorm();
    sqnorm.template cast<double>().minCoeff(&min_i);
    return std::make_pair(weights.row(min_i), sqnorm(min_i));
}

}   // namespace linalg_internal

#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
typedef number<mpfr_float_backend<30> > mpfr_30;
typedef number<mpfr_float_backend<60> > mpfr_60;
typedef number<mpfr_float_backend<100> > mpfr_100;
typedef number<mpfr_float_backend<200> > mpfr_200;

template <typename T>
Matrix<T, Dynamic, Dynamic> nullspaceSVD(const Ref<const Matrix<T, Dynamic, Dynamic> >& A, T sv_tol)
{
    /*
     * Return a matrix whose columns form a basis for the nullspace of A.
     */
    // Get the maximum precision of the given scalar type
    unsigned prec = std::numeric_limits<T>::max_digits10;

    // Try obtaining the nullspace at the given precision
    Matrix<T, Dynamic, Dynamic> nullspace = linalg_internal::nullspaceSVD<T>(A, sv_tol);

    // While the nullspace was not successfully computed ...
    while (nullspace.cols() == 0)
    {
        // Update the precision of the type and re-compute the nullspace
        if (prec <= std::numeric_limits<double>::max_digits10)
        {
            prec = 30;
            Matrix<mpfr_30, Dynamic, Dynamic> B = linalg_internal::convert<T, mpfr_30>(A);
            nullspace = linalg_internal::nullspaceSVD<mpfr_30>(B, sv_tol).template cast<T>();
        }
        else if (prec <= 30)
        {
            prec = 60;
            Matrix<mpfr_60, Dynamic, Dynamic> B = linalg_internal::convert<T, mpfr_60>(A);
            nullspace = linalg_internal::nullspaceSVD<mpfr_60>(B, sv_tol).template cast<T>();
        }
        else if (prec <= 60)
        {
            prec = 100;
            Matrix<mpfr_100, Dynamic, Dynamic> B = linalg_internal::convert<T, mpfr_100>(A);
            nullspace = linalg_internal::nullspaceSVD<mpfr_100>(B, sv_tol).template cast<T>();
        }
        else if (prec <= 100)
        {
            prec = 200;
            Matrix<mpfr_200, Dynamic, Dynamic> B = linalg_internal::convert<T, mpfr_200>(A);
            nullspace = linalg_internal::nullspaceSVD<mpfr_200>(B, sv_tol).template cast<T>();
        }
        else
        {
            throw std::runtime_error("Nullspace was not successfully computed with 200-digit floats");
        }
    }
    return nullspace;
}

template <typename T>
Matrix<T, Dynamic, 1> spanningTreeWeightVector(const Ref<const Matrix<T, Dynamic, Dynamic> >& laplacian, T ztol)
{
    /*
     * Use the recurrence of Chebotarev & Agaev (Lin Alg Appl, 2002, Eqs. 17-18)
     * for the spanning tree weight vector of the given Laplacian matrix.
     *
     * This function does not check that the given matrix is indeed a
     * valid row Laplacian matrix (zero row sums, positive diagonal,
     * negative off-diagonal). 
     */
    // Get the maximum precision of the given scalar type
    unsigned prec = std::numeric_limits<T>::max_digits10;

    // Try obtaining the nullspace at the given precision
    Matrix<T, Dynamic, 1> weights;
    T min_sqnorm;
    std::tie(weights, min_sqnorm) = linalg_internal::spanningTreeWeightVector<T>(laplacian);

    // While the nullspace was not successfully computed ...
    while (min_sqnorm >= ztol)
    {
        // Update the precision of the type and re-compute the nullspace
        if (prec <= std::numeric_limits<double>::max_digits10)
        {
            prec = 30;
            Matrix<mpfr_30, Dynamic, Dynamic> B = linalg_internal::convert<T, mpfr_30>(laplacian);
            std::pair<Matrix<mpfr_30, Dynamic, 1>, mpfr_30> result = linalg_internal::spanningTreeWeightVector<mpfr_30>(B);
            weights = result.first.template cast<T>();
            min_sqnorm = result.second.convert_to<T>();
        }
        else if (prec <= 30)
        {
            prec = 60;
            Matrix<mpfr_60, Dynamic, Dynamic> B = linalg_internal::convert<T, mpfr_60>(laplacian);
            std::pair<Matrix<mpfr_60, Dynamic, 1>, mpfr_60> result = linalg_internal::spanningTreeWeightVector<mpfr_60>(B);
            weights = result.first.template cast<T>();
            min_sqnorm = result.second.convert_to<T>();
        }
        else if (prec <= 60)
        {
            prec = 100;
            Matrix<mpfr_100, Dynamic, Dynamic> B = linalg_internal::convert<T, mpfr_100>(laplacian);
            std::pair<Matrix<mpfr_100, Dynamic, 1>, mpfr_100> result = linalg_internal::spanningTreeWeightVector<mpfr_100>(B);
            weights = result.first.template cast<T>();
            min_sqnorm = result.second.convert_to<T>();
        }
        else if (prec <= 100)
        {
            prec = 200;
            Matrix<mpfr_200, Dynamic, Dynamic> B = linalg_internal::convert<T, mpfr_200>(laplacian);
            std::pair<Matrix<mpfr_200, Dynamic, 1>, mpfr_200> result = linalg_internal::spanningTreeWeightVector<mpfr_200>(B);
            weights = result.first.template cast<T>();
            min_sqnorm = result.second.convert_to<T>();
        }
        else
        {
            throw std::runtime_error("Spanning tree weights were not successfully computed with 200-digit floats");
        }
    }
    return weights;
}

#endif
