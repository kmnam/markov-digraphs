#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <complex>
#include <Eigen/Dense>

/*
 * Various utility functions.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/21/2019
 */
const double two_pi = 2 * std::acos(-1);

namespace utils {

// ------------------------------------------------------ //
//                    CUSTOM EXCEPTIONS                   //
// ------------------------------------------------------ //
namespace except {

struct NullspaceDimError : public std::runtime_error
{
    /*
     * Custom exception to be thrown when a matrix has an unexpected
     * nullspace dimension (number of singular values).
     */
    unsigned dim;
    NullspaceDimError(std::string const& msg, unsigned dim_) : std::runtime_error(msg)
    {
        dim = dim_;
    }
};

}   // namespace except

// ------------------------------------------------------ //
//                 BASIC HELPER FUNCTIONS                 //
// ------------------------------------------------------ //
namespace basic {

std::vector<std::vector<unsigned> > combinations(unsigned n, unsigned k)
{
    /*
     * Return a vector of integer vectors encoding all k-combinations of
     * indices up to n, i.e., {0, 1, ..., n-1}.
     */
    if (k > n)
        throw std::invalid_argument("k-combinations of n items undefined for k > n");

    std::vector<std::vector<unsigned> > combinations;    // Vector of combinations
    std::vector<bool> range(n);                          // Binary indicators for each index
    std::fill(range.end() - k, range.end(), true);
    do
    {
        std::vector<unsigned> c;
        for (unsigned i = 0; i < n; i++)
        {
            if (range[i]) c.push_back(i);
        }
        combinations.push_back(c);
    } while (std::next_permutation(range.begin(), range.end()));

    return combinations; 
}

std::vector<std::vector<unsigned> > powerset(unsigned n)
{
    /*
     * Return a vector of integer vectors encoding the power set of 
     * indices up to n, i.e., {0, 1, ..., n-1}. 
     */
    // Start with the empty set
    std::vector<std::vector<unsigned> > powerset;
    powerset.emplace_back(std::vector<unsigned>());

    // Run through all k-combinations for increasing k
    std::vector<std::vector<unsigned> > k_combinations;
    for (unsigned k = 1; k <= n; k++)
    {
        k_combinations = combinations(n, k);
        powerset.insert(
            powerset.end(),
            std::make_move_iterator(k_combinations.begin()),
            std::make_move_iterator(k_combinations.end())
        );
    }

    return powerset;
}

template <typename T>
std::vector<std::vector<T> > combinations(std::vector<T> data, unsigned k)
{
    /*
     * Return a vector of integer vectors encoding all k-combinations of
     * an input vector of arbitrary-type elements.
     */
    unsigned n = data.size();
    if (k > n)
        throw std::invalid_argument("k-combinations of n items undefined for k > n");

    std::vector<std::vector<T> > combinations;    // Vector of combinations
    std::vector<bool> range(n);                   // Binary indicators for each index
    std::fill(range.end() - k, range.end(), true);

    do
    {
        std::vector<T> c;
        for (unsigned i = 0; i < n; i++)
        {
            if (range[i]) c.push_back(data[i]);
        }
        combinations.push_back(c);
    } while (std::next_permutation(range.begin(), range.end()));

    return combinations; 
}

template <typename T>
std::vector<std::vector<T> > powerset(std::vector<T> data)
{
    /*
     * Return a vector of vectors encoding the power set of an 
     * input vector of arbitrary-type elements. 
     */
    // Start with the empty set
    std::vector<std::vector<T> > powerset;
    powerset.emplace_back(std::vector<unsigned>());

    // Run through all k-combinations for increasing k
    std::vector<std::vector<T> > k_combinations;
    for (unsigned k = 1; k <= data.size(); k++)
    {
        k_combinations = combinations(data, k);
        powerset.insert(
            powerset.end(),
            std::make_move_iterator(k_combinations.begin()),
            std::make_move_iterator(k_combinations.end())
        );
    }

    return powerset;
}

std::string vectorToString(const std::vector<unsigned>& nums)
{
    /*
     * Given a vector of (unsigned) integers, return a comma-delimited
     * string containing the integers. 
     */
    std::ostringstream os;
    if (!nums.empty())
    {
        std::copy(nums.begin(), nums.end() - 1, std::ostream_iterator<unsigned>(os, ","));
        os << nums.back();
    }
    return os.str();    
}

std::vector<unsigned> stringToVector(const std::string& nums)
{
    /*
     * Given a string of comma-delimited (unsigned) integers, return
     * a vector of these integers.
     */
    std::vector<unsigned> out;
    std::string num;
    std::istringstream is(nums);
    while (std::getline(is, num, ',')) out.push_back(std::stoul(num));
    return out;
}

}   // namespace basic

// ------------------------------------------------------ //
//            MATH AND LINEAR ALGEBRA FUNCTIONS           //
// ------------------------------------------------------ //
namespace math {

using Eigen::Array;
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::Dynamic;

// ------------------------------------------------------ //
//                INTERNAL HELPER FUNCTIONS               //
// ------------------------------------------------------ //
namespace internal {

template <typename T>
bool isclose(T a, T b, T tol)
{
    /*
     * Return true if abs(a - b) < tol.
     */
    T c = a - b;
    return ((c >= 0.0 && c < tol) || (c < 0.0 && -c < tol));
}

}   // namespace internal

// ------------------------------------------------------ //
//                   EXTERNAL FUNCTIONS                   //
// ------------------------------------------------------ //
template <typename T>
std::vector<std::complex<T> > rootsOfUnity(unsigned n)
{
    /*
     * Return the n-th roots of unity. 
     */
    using std::cos;
    using std::sin;
    std::vector<std::complex<T> > roots;
    for (unsigned i = 0; i < n; i++)
    {
        T pi_times_i_by_n = two_pi * i / n;
        T real = cos(pi_times_i_by_n);
        T imag = sin(pi_times_i_by_n);
        roots.emplace_back(std::complex<T>(real, imag));
    }
    return roots;
}

template <typename T>
Matrix<T, Dynamic, 1> logSpaceVector(T lower, T upper, unsigned size)
{
    /*
     * Return a logarithmically spaced vector from 10 ^ lower to 10 ^ upper.
     */
    return Eigen::pow(10.0, Array<T, Dynamic, 1>::LinSpaced(size, lower, upper)).matrix();
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

template <typename Derived1, typename Derived2>
Matrix<typename Derived1::Scalar, Dynamic, Dynamic> solve(const MatrixBase<Derived1>& A,
                                                          const MatrixBase<Derived2>& b)
{
    /*
     * Return a vector x which satisfies A * x = b.
     */
    return A.colPivHouseholderQr().solve(b);
}

template <typename Derived>
Matrix<typename Derived::Scalar, Dynamic, Dynamic> nullspace(const MatrixBase<Derived>& A,
                                                             typename Derived::Scalar zero_tol)
{
    /*
     * Return a matrix whose columns form a basis for the nullspace of A.
     */
    using MatrixType = Matrix<typename Derived::Scalar, Dynamic, Dynamic>;

    // Perform a full singular value decomposition of A
    Eigen::BDCSVD<MatrixType> sv_decomp(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Initialize nullspace basis matrix
    MatrixType nullmat;
    unsigned ncols = 0;
    unsigned nrows = A.cols();

    // Run through the singular values of A (in ascending order) ...
    MatrixType S = sv_decomp.singularValues();
    MatrixType V = sv_decomp.matrixV();
    unsigned nsingvals = S.rows();
    unsigned j = nsingvals - 1;
    while (internal::isclose<typename Derived::Scalar>(S(j), 0.0, zero_tol) && j >= 0)
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

template <typename Derived1, typename Derived2>
std::pair<Matrix<typename Derived1::Scalar, Dynamic, Dynamic>, Matrix<typename Derived1::Scalar, Dynamic, Dynamic> >
    nullspace_diff(const MatrixBase<Derived1>& A, const MatrixBase<Derived2>& X, typename Derived1::Scalar zero_tol)
{
    /*
     * Given values of a matrix-valued analytic function A(t) and its 
     * derivative A'(t) at a value of t0, compute:
     *
     * 1) a matrix N with columns forming a basis for the nullspace of A = A(t0), and
     * 2) the derivative of the nullspace N (viewed as the value of a nullspace
     *    function, evaluated at t0) at X = A'(t0). 
     */
    using MatrixType = Matrix<typename Derived1::Scalar, Dynamic, Dynamic>;

    // Perform a full singular value decomposition of A
    Eigen::BDCSVD<MatrixType> sv_decomp(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Grab U, S, V
    MatrixType S = sv_decomp.singularValues();
    unsigned nsingvals = S.rows();
    MatrixType U = sv_decomp.matrixU();
    MatrixType V = sv_decomp.matrixV();

    // Run through the singular values of A (in ascending order, i.e., 
    // in descending order of index) ...
    MatrixType nullmat;
    unsigned ncols = 0;
    unsigned nrows = A.cols();
    std::vector<unsigned> nullidx;
    unsigned j = nsingvals - 1;
    while (internal::isclose<typename Derived1::Scalar>(S(j), 0.0, zero_tol) && j >= 0)
    {
        // ... and grab the columns of V that correspond to the zero
        // singular values
        ncols += 1;
        nullmat.resize(nrows, ncols);
        nullmat.col(ncols - 1) = V.col(j);
        nullidx.push_back(j);
        j--;
    }

    // Compute the differential of the singular value matrix
    MatrixType Ut = U.transpose();
    MatrixType Vt = V.transpose();
    MatrixType Q = Ut * X * V;
    MatrixType dS = Q.diagonal();

    // Compute the differential of V (differential of U is not necessary)
    MatrixType W;    // Define W = Vt * dV, which is antisymmetric
                     // (Z = Ut * dU is not necessary)
    W.resize(A.cols(), A.cols());
    for (unsigned i = 0; i < nsingvals; i++)
    {
        for (unsigned j = 0; j < nsingvals; j++)
        {
            if (i > j)
            {
                // Solve the 2-by-2 system given by
                //
                // ( S(j)   S(i) ) ( T(0) = Z(i,j) ) = (  Q(i,j) =  (Ut * X * V)(i,j) )
                // ( S(i)   S(j) ) ( T(1) = W(j,i) ) = ( -Q(j,i) = -(Ut * X * V)(j,i) ),
                //
                // which has a unique solution if and only if S(i)^2 != S(j)^2
                // 
                // Check that S(i)^2 != S(j)^2 for all i, j
                if (abs(S(j) * S(j) - S(i) * S(i)) < zero_tol)
                {
                    std::cerr << "Degenerate singular values; derivative may "
                              << "be unreliable\n";
                }
                // Compute only W(i,j) and W(j,i); Z is not necessary
                W(j,i) = (-S(i) * Q(i,j) - S(j) * Q(j,i)) / (S(j) * S(j) - S(i) * S(i));
                W(i,j) = -W(j,i);
            }
        }
    }
    MatrixType dV = V * W;    // Compute dV (no need to compute dU)

    // Compute nullspace derivative matrix 
    MatrixType nulldiff(nullmat.rows(), nullmat.cols());
    for (unsigned j = 0; j < nullidx.size(); j++)
        nulldiff.col(j) = dV.col(nullidx[j]);

    return std::make_pair(nullmat, nulldiff);
}

}   // namespace linalg

}   // namespace utils

#endif
