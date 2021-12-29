/**
 *  Suite of mathematical functions of the `LabeledDigraph` class. 
 *
 *  **Author:**
 *      Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 *  **Last updated:** 
 *      12/29/2021
 */

#ifndef MATH_HELPER_FUNCTIONS_FOR_LABELED_DIGRAPH_HPP
#define MATH_HELPER_FUNCTIONS_FOR_LABELED_DIGRAPH_HPP

#include <cmath>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "kahan.hpp"
#include "KBNSum.hpp"

using namespace Eigen;

/** All non-homogeneous linear system solver methods. */
enum SolverMethod {  
    LUDecomposition,
    QRDecomposition
};

/** All supported summation methods. */  
enum SummationMethod {
    NaiveSummation, 
    KahanSummation, 
    PairwiseSummation,
    KBNSummation
};

/** 
 * Return true if abs(a - b) < tol.
 *
 * @param a   first scalar. 
 * @param b   second scalar. 
 * @param tol tolerance.
 * @returns   true if abs(a - b) < tol, false otherwise.  
 */
template <typename T>
bool isclose(T a, T b, T tol)
{
    T c = a - b;
    return ((c >= 0 && c < tol) || (c < 0 && -c < tol));
}

/**
 * Compute the nullspace of A by performing a singular value decomposition.
 * 
 * This function returns the column of V in the SVD of A = USV corresponding
 * to the least singular value (recall that singular values are always
 * non-negative). It therefore effectively assumes that the A has a 
 * nullspace of dimension one.
 * 
 * @param A Input matrix to be decomposed. 
 * @returns The column of V in the singular value decomposition A = USV 
 *          corresponding to the least singular value. 
 */
template <typename T>
Matrix<T, Dynamic, 1> getOneDimNullspaceFromSVD(const Ref<const Matrix<T, Dynamic, Dynamic> >& A)
{
    // Perform a singular value decomposition of A, only computing V in full
    Eigen::BDCSVD<Matrix<T, Dynamic, Dynamic> > svd(A, Eigen::ComputeFullV);

    // Return the column of V corresponding to the least singular value of A
    // (always given last in the decomposition) 
    Matrix<T, Dynamic, 1> singular = svd.singularValues(); 
    Matrix<T, Dynamic, Dynamic> V = svd.matrixV();
    return V.col(singular.rows() - 1); 
}

/**
 * Compute the nullspace of A by performing a singular value decomposition.
 * 
 * This function returns the column(s) of V in the SVD of A = USV
 * corresponding to singular values with absolute value < tol.
 *
 * @param A   Input matrix to be decomposed. 
 * @param tol Tolerance for singular value to be treated as sufficiently 
 *            close to zero.
 * @returns   The matrix of columns in V in the singular value decomposition
 *            A = USV corresponding to singular values less than tol.  
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> getNullspaceFromSVD(const Ref<const Matrix<T, Dynamic, Dynamic> >& A, const T tol)
{
    // Perform a singular value decomposition of A, only computing V in full
    Eigen::BDCSVD<Matrix<T, Dynamic, Dynamic> > svd(A, Eigen::ComputeFullV);

    // Initialize nullspace basis matrix
    Matrix<T, Dynamic, Dynamic> nullmat;
    unsigned ncols = 0;
    unsigned nrows = A.cols();

    // Run through the singular values of A (in ascending, i.e., reverse order) ...
    Matrix<T, Dynamic, 1> singular = svd.singularValues();
    Matrix<T, Dynamic, Dynamic> V = svd.matrixV();
    unsigned ns = singular.rows();
    unsigned j = ns - 1;
    while (isclose<T>(singular(j), 0.0, tol) && j >= 0)
    {
        // ... and grab the columns of V that correspond to the zero
        // singular values
        ncols++;
        nullmat.resize(nrows, ncols);
        nullmat.col(ncols - 1) = V.col(j);
        j--;
    }

    return nullmat;
}

/**
 * Solve the non-homogeneous linear system Ax = b by obtaining an LU 
 * decomposition of a matrix A. 
 * 
 * @param A Input matrix. 
 * @param b Input vector.
 * @returns Solution vector to Ax = b.  
 */
template <typename T>
Matrix<T, Dynamic, 1> solveByLUD(const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                                 const Ref<const Matrix<T, Dynamic, 1> >& b)
{
    // Obtain a full-pivot LU decomposition of A 
    FullPivLU<Matrix<T, Dynamic, Dynamic> > lud(A); 

    // Get and return the solution to Ax = b
    return lud.solve(b); 
}

/**
 * Solve the non-homogeneous linear system Ax = b by obtaining a QR 
 * decomposition of A. 
 *
 * @param A Input matrix. 
 * @param b Input vector. 
 * @returns Solution vector to Ax = b. 
 */
template <typename T>
Matrix<T, Dynamic, 1> solveByQRD(const Ref<const Matrix<T, Dynamic, Dynamic> >& A, 
                                 const Ref<const Matrix<T, Dynamic, 1> >& b)
{
    // Obtain a QR decomposition of A 
    ColPivHouseholderQR<Matrix<T, Dynamic, Dynamic> > qrd(A);

    // Get and return the solution to Ax = b
    return qrd.solve(b); 
}

/**
 * Apply one iteration of the recurrence of Chebotarev & Agaev (Lin Alg
 * Appl, 2002, Eqs.\ 17-18) for the spanning forest matrices of the graph,
 * using a *dense* Laplacian matrix.
 *
 * @param k         Index of current iteration. 
 * @param laplacian Input Laplacian matrix. 
 * @param curr      k-th spanning forest matrix obtained from previous
 *                  applications of the Chebotarev-Agaev recurrence. 
 * @returns         (k+1)-th spanning forest matrix.  
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> chebotarevAgaevRecurrence(const int k, 
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& laplacian,
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& curr)
{
    // First compute the following:
    //
    //                 tr(L(G) * Q_k(G))
    // \phi_{k+1}(G) = -----------------
    //                       k + 1
    //
    // Then compute: 
    //
    // Q_{k+1}(G) = -L(G) * Q_k(G) + \phi_{k+1}(G) * I
    //
    T denom = static_cast<T>(k + 1);
    Matrix<T, Dynamic, Dynamic> product = laplacian * curr;
    T phi = product.trace() / denom;  
    Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(laplacian.rows(), laplacian.cols());

    return (phi * identity) - product;
}

/**
 * Apply one iteration of the recurrence of Chebotarev & Agaev (Lin Alg
 * Appl, 2002, Eqs.\ 17-18) for the spanning forest matrices of the graph,
 * using a *compressed sparse row-major* Laplacian matrix.
 *
 * @param k         Index of current iteration. 
 * @param laplacian Input Laplacian matrix (compressed sparse row-major).
 * @param curr      k-th spanning forest matrix obtained from previous
 *                  applications of the Chebotarev-Agaev recurrence. 
 * @returns         (k+1)-th spanning forest matrix. 
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> chebotarevAgaevRecurrence(const int k,
                                                      const SparseMatrix<T, RowMajor>& laplacian, 
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& curr)
{
    // First compute the following:
    //
    //                 tr(L(G) * Q_k(G))
    // \phi_{k+1}(G) = -----------------
    //                       k + 1
    //
    // Then compute: 
    //
    // Q_{k+1}(G) = -L(G) * Q_k(G) + \phi_{k+1}(G) * I
    //
    T denom = static_cast<T>(k + 1);
    Matrix<T, Dynamic, Dynamic> product = laplacian * curr;
    T phi = product.trace() / denom; 
    Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(laplacian.rows(), laplacian.cols());

    return (phi * identity) - product;
}

// -------------------------------------------------------------------- //
//          OVERLOADED FUNCTIONS WITH CUSTOM SUMMATION METHODS          //
// -------------------------------------------------------------------- //

/**
 * Apply one iteration of the recurrence of Chebotarev & Agaev (Lin Alg
 * Appl, 2002, Eqs.\ 17-18) for the spanning forest matrices of the graph,
 * using a *dense* Laplacian matrix.
 *
 * @param k         Index of current iteration. 
 * @param laplacian Input Laplacian matrix. 
 * @param curr      k-th spanning forest matrix obtained from previous
 *                  applications of the Chebotarev-Agaev recurrence. 
 * @param method    Summation method.
 * @returns         (k+1)-th spanning forest matrix.  
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> chebotarevAgaevRecurrence(const int k, 
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& laplacian,
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& curr,
                                                      const SummationMethod method) 
{
    // First compute the following:
    //
    //                 tr(L(G) * Q_k(G))
    // \phi_{k+1}(G) = -----------------
    //                       k + 1
    //
    // Then compute: 
    //
    // Q_{k+1}(G) = -L(G) * Q_k(G) + \phi_{k+1}(G) * I
    //
    T denom = static_cast<T>(k + 1);
    Matrix<T, Dynamic, Dynamic> product;
    T phi; 
    switch (method)
    {
        case KahanSummation:
            product = KahanSum::multiply(laplacian, curr);
            phi = KahanSum::trace(product) / denom;
            break;

        case KBNSummation:
            product = KBNSum::multiply(laplacian, curr); 
            phi = KBNSum::trace(product) / denom;
            break;

        default:
            std::stringstream ss; 
            ss << "Unrecognized summation method: " << method; 
            throw std::invalid_argument(ss.str());
            break; 
    }
    Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(laplacian.rows(), laplacian.cols());

    return (phi * identity) - product;
}

/**
 * Apply one iteration of the recurrence of Chebotarev & Agaev (Lin Alg
 * Appl, 2002, Eqs.\ 17-18) for the spanning forest matrices of the graph,
 * using a *compressed sparse row-major* Laplacian matrix.
 *
 * @param k         Index of current iteration. 
 * @param laplacian Input Laplacian matrix (compressed sparse row-major).
 * @param curr      k-th spanning forest matrix obtained from previous
 *                  applications of the Chebotarev-Agaev recurrence. 
 * @param method    Summation method.
 * @returns         (k+1)-th spanning forest matrix. 
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> chebotarevAgaevRecurrence(const int k,
                                                      const SparseMatrix<T, RowMajor>& laplacian, 
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& curr,
                                                      const SummationMethod method)
{
    // First compute the following:
    //
    //                 tr(L(G) * Q_k(G))
    // \phi_{k+1}(G) = -----------------
    //                       k + 1
    //
    // Then compute: 
    //
    // Q_{k+1}(G) = -L(G) * Q_k(G) + \phi_{k+1}(G) * I
    //
    T denom = static_cast<T>(k + 1);
    Matrix<T, Dynamic, Dynamic> product;
    T phi; 
    switch (method)
    {
        case KahanSummation:
            product = KahanSum::multiply(laplacian, curr);
            phi = KahanSum::trace(product) / denom;
            break;

        case KBNSummation:
            product = KBNSum::multiply(laplacian, curr); 
            phi = KBNSum::trace(product) / denom;
            break;

        default:
            std::stringstream ss; 
            ss << "Unrecognized summation method: " << method; 
            throw std::invalid_argument(ss.str());
            break; 
    }
    Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(laplacian.rows(), laplacian.cols());

    return (phi * identity) - product;
}

#endif 
