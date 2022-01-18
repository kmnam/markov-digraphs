/**
 * \line include/graphs/grid.hpp
 *
 * An implementation of the grid graph. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     1/18/2022
 */

#ifndef GRID_LABELED_DIGRAPH_HPP
#define GRID_LABELED_DIGRAPH_HPP

#include <sstream>
#include <array>
#include <vector>
#include <utility>
#include <stdexcept>
#include <Eigen/Dense>
#include "../digraph.hpp"

using namespace Eigen;

/**
 * Apply operator Alpha onto the given vector `v`, with the given sextet
 * of edge labels.
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 3). 
 * @returns           Output vector (length 3). 
 */
template <typename T>
Matrix<T, 3, 1> alpha(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 3, 1> >& v)
{
    Matrix<T, 3, 3> A = Matrix<T, 3, 3>::Zero(); 
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = fA * rB; 
    T tmp1 = fB * rA; 
    A(0, 0) = rA * (rB + d) + rB * c;    // d * rA + c * rB + rA * rB
    A(0, 1) = tmp0 * c;                  // fA * c * rB
    A(0, 2) = tmp1 * d;                  // fB * d * rA
    A(1, 0) = rB;
    A(1, 1) = tmp0;                      // fA * rB
    A(2, 0) = rA;
    A(2, 2) = tmp1;                      // fB * rA 
    // A(1, 2) == A(2, 1) == 0

    return A * v; 
}

/**
 * Apply operator Beta onto the given vector `v`, with the given sextet
 * of edge labels.
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 3). 
 * @returns           Output vector (length 3). 
 */
template <typename T>
Matrix<T, 3, 1> beta(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 3, 1> >& v)
{
    Matrix<T, 3, 3> A = Matrix<T, 3, 3>::Zero(); 
    T fA = rung_labels[0]; 
    T rA = rung_labels[1]; 
    T fB = rung_labels[2]; 
    T rB = rung_labels[3]; 
    T c = rung_labels[4]; 
    T d = rung_labels[5]; 
    T tmp0 = fA * fB;
    A(0, 0) = fA * (d + rB); 
    A(0, 1) = fB * d; 
    A(0, 2) = tmp0 * d;        // fA * fB * d 
    A(1, 0) = fA * c; 
    A(1, 1) = fB * (c + rA); 
    A(1, 2) = tmp0 * c;        // fA * fB * c
    A(2, 0) = fA; 
    A(2, 1) = fB; 
    A(2, 2) = tmp0;            // fA * fB

    return A * v; 
}

/**
 * Apply operator Beta1 onto the given vector `v`, with the given sextet
 * of edge labels.
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 4). 
 * @returns           Output vector (length 3). 
 */
template <typename T>
Matrix<T, 3, 1> beta1(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 4, 1> >& v)
{
    Matrix<T, 3, 4> A = Matrix<T, 3, 4>::Zero(); 
    T fA = rung_labels[0]; 
    T rA = rung_labels[1]; 
    T fB = rung_labels[2]; 
    T rB = rung_labels[3]; 
    T c = rung_labels[4]; 
    T d = rung_labels[5]; 
    T tmp0 = c + rA;
    T tmp1 = d + rB; 
    T tmp2 = fA * fB; 
    A(0, 0) = tmp1;            // d + rB 
    A(0, 1) = fA * tmp1;       // fA * (d + rB)
    A(0, 2) = fB * d;
    A(0, 3) = tmp2 * d;        // fA * fB * d 
    A(1, 0) = tmp0;            // c + rA 
    A(1, 1) = fA * c; 
    A(1, 2) = fB * tmp0;       // fB * (c + rA) 
    A(1, 3) = tmp2 * c;        // fA * fB * c
    A(2, 0) = 1; 
    A(2, 1) = fA; 
    A(2, 2) = fB; 
    A(2, 3) = tmp2;            // fA * fB

    return A * v; 
}

/**
 * Apply operator Gamma1 onto the given vector `v`, with the given sextet
 * of edge labels.
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 2). 
 * @returns           Output vector (length 3). 
 */
template <typename T>
Matrix<T, 3, 1> gamma1(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 2, 1> >& v)
{
    Matrix<T, 3, 2> A = Matrix<T, 3, 2>::Zero(); 
    T fA = rung_labels[0]; 
    T rA = rung_labels[1]; 
    T fB = rung_labels[2]; 
    T rB = rung_labels[3]; 
    T c = rung_labels[4]; 
    T d = rung_labels[5]; 
    T tmp0 = fB * rA; 
    A(0, 0) = rA * (rB + d) + rB * c;    // d * rA + c * rB + rA * rB
    A(0, 1) = tmp0 * d;                  // fB * d * rA
    A(1, 0) = rB;
    A(2, 0) = rA;
    A(2, 1) = tmp0;                      // fB * rA 
    // A(1, 1) == 0

    return A * v; 
}

/**
 * Apply operator Gamma2 onto the given vector `v`, with the given sextet
 * of edge labels.
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 2). 
 * @returns           Output vector (length 3). 
 */
template <typename T>
Matrix<T, 3, 1> gamma2(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 2, 1> >& v)
{
    Matrix<T, 3, 2> A = Matrix<T, 3, 2>::Zero(); 
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = fA * rB; 
    A(0, 0) = rA * (rB + d) + rB * c;    // d * rA + c * rB + rA * rB
    A(0, 1) = tmp0 * c;                  // fA * c * rB
    A(1, 0) = rB;
    A(1, 1) = tmp0;                      // fA * rB
    A(2, 0) = rA;
    // A(2, 1) == 0

    return A * v; 
}

/**
 * Apply operator Delta1 onto the given vector `v`, with the given sextet 
 * of edge labels. 
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 2). 
 * @returns           Output scalar.
 */
template <typename T>
T delta1(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 2, 1> >& v)
{
    return rung_labels[0] * v(0) + (rung_labels[0] * rung_labels[2]) * v(1); 
}

/**
 * Apply operator Delta2 onto the given vector `v`, with the given sextet 
 * of edge labels. 
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 2). 
 * @returns           Output scalar. 
 */
template <typename T>
T delta2(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 2, 1> >& v)
{
    return rung_labels[2] * v(0) + (rung_labels[0] * rung_labels[2]) * v(1); 
}

/**
 * Apply operator Epsilon1 onto the given vector `v`, with the given sextet 
 * of edge labels. 
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 2). 
 * @returns           Output vector (length 3). 
 */
template <typename T>
Matrix<T, 3, 1> epsilon1(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 2, 1> >& v)
{
    Matrix<T, 3, 2> A; 
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = c + rA; 
    A(0, 0) = d + rB; 
    A(0, 1) = fB * d; 
    A(1, 0) = tmp0; 
    A(1, 1) = fB * tmp0; 
    A(2, 0) = 1; 
    A(2, 1) = fB; 

    return A * v; 
}

/**
 * Apply operator Epsilon2 onto the given vector `v`, with the given sextet 
 * of edge labels. 
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 2). 
 * @returns           Output vector (length 3).
 */
template <typename T>
Matrix<T, 3, 1> epsilon2(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 2, 1> >& v)
{
    Matrix<T, 3, 2> A; 
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = d + rB; 
    A(0, 0) = tmp0; 
    A(0, 1) = fA * tmp0;
    A(1, 0) = c + rA; 
    A(1, 1) = fA * c; 
    A(2, 0) = 1; 
    A(2, 1) = fA;  

    return A * v; 
}

/**
 * Apply operator Theta1 onto the given vector `v`, with the given sextet 
 * of edge labels. 
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph. 
 * @param v           Input vector (length 4). 
 * @returns           Output vector (length 4).
 */
template <typename T>
Matrix<T, 4, 1> theta1(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 4, 1> >& v)
{
    Matrix<T, 4, 4> A = Matrix<T, 4, 4>::Zero(); 
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = fA * fB; 
    A(0, 0) = fA * (d + rB); 
    A(0, 1) = fB * d;
    A(0, 2) = tmp0 * d;        // fA * fB * d
    A(0, 3) = tmp0 * d;        // fA * fB * d
    A(1, 0) = fA * c;
    A(1, 1) = fB * (c + rA); 
    A(1, 2) = tmp0 * c;        // fA * fB * c
    A(1, 3) = tmp0 * c;        // fA * fB * c
    A(2, 1) = fB; 
    A(2, 2) = tmp0;            // fA * fB
    A(3, 0) = fA; 
    A(3, 3) = tmp0;            // fA * fB

    return A * v;  
}

/**
 * Apply operator Theta2 onto the given vector `v`, with the given sextet 
 * of edge labels. 
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph. 
 * @param v           Input vector (length 3). 
 * @returns           Output vector (length 4).
 */
template <typename T>
Matrix<T, 4, 1> theta2(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 3, 1> >& v)
{
    Matrix<T, 4, 3> A = Matrix<T, 4, 3>::Zero(); 
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = fA * fB; 
    A(0, 0) = fA * (d + rB); 
    A(0, 1) = fB * d;
    A(0, 2) = tmp0 * d;        // fA * fB * d
    A(1, 0) = fA * c;
    A(1, 1) = fB * (c + rA);
    A(1, 2) = tmp0 * c;        // fA * fB * c 
    A(2, 1) = fB; 
    A(3, 0) = fA; 
    A(3, 2) = tmp0;            // fA * fB

    return A * v;  
}

/**
 * Apply operator Theta3 onto the given vector `v`, with the given sextet 
 * of edge labels. 
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph. 
 * @param v           Input vector (length 3). 
 * @returns           Output vector (length 4).
 */
template <typename T>
Matrix<T, 4, 1> theta3(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 3, 1> >& v)
{
    Matrix<T, 4, 3> A = Matrix<T, 4, 3>::Zero(); 
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = fA * fB; 
    A(0, 0) = fA * (d + rB); 
    A(0, 1) = fB * d;
    A(0, 2) = tmp0 * d;        // fA * fB * d 
    A(1, 0) = fA * c;
    A(1, 1) = fB * (c + rA);
    A(1, 2) = tmp0 * c;        // fA * fB * c 
    A(2, 1) = fB;
    A(2, 2) = tmp0;            // fA * fB 
    A(3, 0) = fA; 

    return A * v;  
}

/**
 * Apply operator Theta4 onto the given vector `v`, with the given sextet 
 * of edge labels. 
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph. 
 * @param v           Input vector (length 6). 
 * @returns           Output vector (length 2).
 */
template <typename T>
Matrix<T, 2, 1> theta4(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 6, 1> >& v)
{
    Matrix<T, 2, 6> A;
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = fA * fB; 
    A(0, 0) = c;
    A(0, 1) = fA * c; 
    A(0, 2) = fB * c; 
    A(0, 3) = tmp0 * c;        // fA * fB * c
    A(0, 4) = rA * fB; 
    A(0, 5) = 0; 
    A(1, 0) = d; 
    A(1, 1) = fA * d; 
    A(1, 2) = fB * d; 
    A(1, 3) = tmp0 * d;        // fA * fB * c
    A(1, 4) = 0; 
    A(1, 5) = fA * rB; 

    return A * v; 
}

/**
 * An implementation of the grid graph.
 *
 * The nodes in the grid graph are denoted by `"A0"`, `"B0"`, ...,
 * `"A{this->N}"`, `"B{this->N}"`, and therefore we have
 * `this->numnodes == 2 * this->N + 2`.
 *
 * Additional methods implemented for this graph include: 
 * - `getUpperExitProb()`: compute the *splitting probability* of exiting
 *   the graph through `"B{this->N}"` (reaching an auxiliary upper exit node),
 *   and not through `"A0"` (reaching an auxiliary lower exit node). 
 * - `getUpperExitRate(IOType)`: compute the reciprocal of the *unconditional
 *   mean first-passage time* to exiting the graph through `"B{this->N}"`,
 *   given that the exit rate from `"A0"` is zero. 
 * - `getLowerExitRate()`: compute the reciprocal of the *unconditional mean
 *   first-passage time* to exiting the graph through `"A0"`, given that the 
 *   exit rate from `"B{this->N}"` is zero. 
 * - `getUpperExitRate(IOType, IOType)`: compute the reciprocal of the
 *   *conditional mean first-passage time* to exiting the graph through
 *   `"B{this->N}"`, given that (1) both exit rates are nonzero and (2) exit
 *   through `"B{this->N}"` does occur. 
 */
template <typename InternalType, typename IOType>
class GridGraph : public LabeledDigraph<InternalType, IOType>
{
    /**
     * An implementation of the grid graph. 
     */
    protected:
        /** Index of the last pair of nodes. */
        unsigned N;

        /** Labels on the edges `A0 -> B0` and `B0 -> A0`. */
        std::pair<InternalType, InternalType> start;

        /**
         * Vector of edge labels that grows with the length of the graph. 
         * The i-th array stores the labels for the edges `A{i} -> A{i+1}`,
         * `A{i+1} -> A{i}`, `B{i} -> B{i+1}`, `B{i+1} -> B{i}`, `A{i} -> B{i}`, 
         * and `B{i} -> A{i}`. 
         */
        std::vector<std::array<InternalType, 6> > rung_labels;

        /**
         * Add a node to the graph with the given ID, and return a pointer
         * to the new node.
         *
         * A convenient private version of the parent method
         * `LabeledDigraph<...>::addNode()`, so that any call to
         * `GridGraph<...>::addNode()` throws an exception (see below).  
         *
         * @param id ID for new node. 
         * @returns  A pointer to new node. 
         * @throws std::runtime_error if node already exists with the given ID. 
         */
        Node* __addNode(std::string id)
        {
            // Check that a node with the given id doesn't exist already
            if (this->nodes.find(id) != this->nodes.end())
                throw std::runtime_error("Node already exists"); 

            Node* node = new Node(id);
            this->order.push_back(node); 
            this->nodes[id] = node;
            this->edges.emplace(node, std::unordered_map<Node*, InternalType>());
            this->numnodes++;
            return node;
        }

        /**
         * Add an edge between two nodes.
         *
         * If either ID does not correspond to a node in the graph, this function
         * instantiates these nodes.
         *
         * A convenient private version of the parent method
         * `LabeledDigraph<...>::addEdge()`, so that any call to
         * `GridGraph<...>::addEdge()` throws an exception (see below).  
         *
         * @param source_id ID of source node of new edge. 
         * @param target_id ID of target node of new edge. 
         * @param label     Label on new edge.
         * @throws std::runtime_error if the edge already exists. 
         */
        void __addEdge(std::string source_id, std::string target_id, IOType label = 1)
        {
            // Look for the two nodes
            Node* source = this->getNode(source_id);
            Node* target = this->getNode(target_id);

            // Then see if the edge already exists (in which case the two nodes
            // obviously already exist) 
            if (source != nullptr && target != nullptr &&
                this->edges[source].find(target) != this->edges[source].end())
            {
                throw std::runtime_error("Edge already exists; use LabeledDigraph<...>::setEdgeLabel() to change label");
            }

            // If the nodes do not exist, then add the nodes  
            if (source == nullptr)
                source = this->__addNode(source_id);
            if (target == nullptr)
                target = this->__addNode(target_id);

            // Then define the edge
            this->edges[source][target] = static_cast<InternalType>(label);
        }

        /**
         * Compute and return six matrices of spanning forest weights related 
         * to the exit statistics of the grid graph, named as follows: 
         *
         * 1) `v_alpha` (size `(3, 2 * this->N + 2)`)
         * 2) `v_beta1` (size `(3, this->N)`)
         * 3) `v_beta2` (size `(3, 2 * this->N + 2)`)
         * 4) `v_beta3` (size `(3, 2 * this->N + 2)`)
         * 5) `v_delta1` (size `(2, this->N)`)
         * 6) `v_delta2` (size `(2, this->N)`)
         *
         * The `(2*i)`-th column of `v_alpha` contains the weights of 
         * - spanning forests rooted at `(A,i)`
         * - spanning forests rooted at `\{(A,i),(A,N)\}` with path `(B,N) -> (A,i)`
         * - spanning forests rooted at `\{(A,i),(B,N)\}` with path `(A,N) -> (A,i)`
         * The `(2*i+1)`-th column of `v_alpha` contains the weights of
         * - spanning forests rooted at `(B,i)`
         * - spanning forests rooted at `\{(B,i),(A,N)\}` with path `(B,N) -> (B,i)`
         * - spanning forests rooted at `\{(B,i),(B,N)\}` with path `(A,N) -> (B,i)`
         *
         * The `i`-th column of `v_beta1` contains the weights of 
         * - spanning forests rooted at `(A,i+1)`
         * - spanning forests rooted at `(B,i+1)`
         * - spanning forests rooted at `\{(A,i+1),(B,i+1)\}`
         * on the `(i+1)`-th grid subgraph
         *
         * The `(2*i)`-th column of `v_beta2` contains the weights of
         * - spanning forests rooted at `\{(A,i),(A,N)\}`
         * - spanning forests rooted at `\{(A,i),(B,N)\}`
         * - spanning forests rooted at `\{(A,i),(A,N),(B,N)\}`
         * The `(2*i+1)`-th column of `v_beta2` contains the weights of
         * - spanning forests rooted at `\{(B,i),(A,N)\}`
         * - spanning forests rooted at `\{(B,i),(B,N)\}`
         * - spanning forests rooted at `\{(B,i),(A,N),(B,N)\}`
         *
         * The `(2*i)`-th column of `v_beta3` contains the weights of
         * - spanning forests rooted at `\{(A,i),(A,N)\}` with path `(A,0) -> (A,i)`
         * - spanning forests rooted at `\{(A,i),(B,N)\}` with path `(A,0) -> (A,i)`
         * - spanning forests rooted at `\{(A,i),(A,N),(B,N)\}` with path `(A,0) -> (A,i)`
         * The `(2*i+1)`-th column of `v_beta3` contains the weights of
         * - spanning forests rooted at `\{(B,i),(A,N)\}` with path `(A,0) -> (B,i)`
         * - spanning forests rooted at `\{(B,i),(B,N)\}` with path `(A,0) -> (B,i)`
         * - spanning forests rooted at `\{(B,i),(A,N),(B,N)\}` with path `(A,0) -> (B,i)`
         *
         * The `i`-th column of `v_delta1` contains the weights of
         * - spanning forests rooted at `(A,i+1)`
         * - spanning forests rooted at `\{(A,i+1),(B,i+1)\}` with path `(A,0) -> (A,i+1)`
         * on the `(i+1)`-th grid subgraph
         *
         * The `i`-th column of `v_delta2` contains the weights of
         * - spanning forests rooted at `(B,i+1)`
         * - spanning forests rooted at `\{(A,i+1),(B,i+1)\}` with path `(A,0) -> (B,i+1)`
         * on the `(i+1)`-th grid subgraph
         *
         * The `(2*i)`-th column of `v_theta1` contains the weights of 
         * - spanning forests rooted at `\{(A,0),(A,N)\}` with path `(A,i) -> (A,N)`
         * - spanning forests rooted at `\{(A,0),(B,N)\}` with path `(A,i) -> (B,N)`
         * - spanning forests rooted at `\{(A,0),(A,N),(B,N)\}` with path `(A,i) -> (B,N)`
         * - spanning forests rooted at `\{(A,0),(A,N),(B,N)\}` with path `(A,i) -> (A,N)`
         *
         * The `(2*i+1)`-th column of `v_theta1` contains the weights of 
         * - spanning forests rooted at `\{(A,0),(A,N)\}` with path `(B,i) -> (A,N)`
         * - spanning forests rooted at `\{(A,0),(B,N)\}` with path `(B,i) -> (B,N)`
         * - spanning forests rooted at `\{(A,0),(A,N),(B,N)\}` with path `(B,i) -> (B,N)`
         * - spanning forests rooted at `\{(A,0),(A,N),(B,N)\}` with path `(B,i) -> (A,N)`
         *
         * The `i`-th column of `v_theta2` contains the weights of
         * - spanning forests rooted at `\{(A,0),(B,i+1)\}` with path `(A,i+1) -> (B,i+1)`
         * - spanning forests rooted at `\{(A,0),(A,i+1)\}` with path `(B,i+1) -> (A,i+1)`
         * on the `(i+1)`-th grid subgraph 
         *
         * This method assumes that `this->N` is greater than 1.  
         *
         * @returns               The above six matrices. 
         */
        std::tuple<Matrix<InternalType, 3, Dynamic>,    // v_alpha
                   Matrix<InternalType, 3, Dynamic>,    // v_beta1
                   Matrix<InternalType, 3, Dynamic>,    // v_beta2
                   Matrix<InternalType, 3, Dynamic>,    // v_beta3
                   Matrix<InternalType, 2, Dynamic>,    // v_delta1
                   Matrix<InternalType, 2, Dynamic>,    // v_delta2
                   Matrix<InternalType, 4, Dynamic>,    // v_theta1 and v_theta2
                   Matrix<InternalType, 2, Dynamic> > getExitRelatedSpanningForestWeights()
        {
            Node* node_A0 = this->getNode("A0"); 
            Node* node_B0 = this->getNode("B0"); 

            // First define the Laplacian matrix of the subgraph on A0, B0, A1, B1
            Matrix<InternalType, Dynamic, Dynamic> laplacian = Matrix<InternalType, Dynamic, Dynamic>::Zero(4, 4);
            Node* node_A1 = this->getNode("A1");
            Node* node_B1 = this->getNode("B1");
            laplacian(0, 1) = -this->edges[node_A0][node_B0];    // A0 -> B0
            laplacian(0, 2) = -this->edges[node_A0][node_A1];    // A0 -> A1
            laplacian(1, 0) = -this->edges[node_B0][node_A0];    // B0 -> A0
            laplacian(1, 3) = -this->edges[node_B0][node_B1];    // B0 -> B1
            laplacian(2, 0) = -this->edges[node_A1][node_A0];    // A1 -> A0
            laplacian(2, 3) = -this->edges[node_A1][node_B1];    // A1 -> B1
            laplacian(3, 1) = -this->edges[node_B1][node_B0];    // B1 -> B0
            laplacian(3, 2) = -this->edges[node_B1][node_A1];    // B1 -> A1
            for (unsigned i = 0; i < 4; ++i)
                laplacian(i, i) = -laplacian.row(i).sum();

            // Remove the edges outgoing from A1 and apply the Chebotarev-Agaev
            // recurrence to obtain the required spanning forest weights
            Matrix<InternalType, Dynamic, Dynamic> curr = Matrix<InternalType, Dynamic, Dynamic>::Identity(4, 4);
            laplacian(2, 0) = 0; 
            laplacian(2, 2) = 0; 
            laplacian(2, 3) = 0; 
            curr = chebotarevAgaevRecurrence<InternalType>(0, laplacian, curr);
            curr = chebotarevAgaevRecurrence<InternalType>(1, laplacian, curr);
            // All spanning forests rooted at A0, A1
            InternalType weight_A0_A1 = curr(0, 0);
            // All spanning forests rooted at A0, A1 with path from B0 to A0
            InternalType weight_A0_A1_with_path_B0_to_A0 = curr(1, 0); 
            // All spanning forests rooted at A0, A1 with path from B1 to A0
            InternalType weight_A0_A1_with_path_B1_to_A0 = curr(3, 0);
            // All spanning forests rooted at B0, A1 with path from A0 to B0
            InternalType weight_B0_A1_with_path_A0_to_B0 = curr(0, 1); 
            // All spanning forests rooted at B0, A1 with path from B1 to B0
            InternalType weight_B0_A1_with_path_B1_to_B0 = curr(3, 1);
            // All spanning forests rooted at B0, A1
            InternalType weight_B0_A1 = curr(1, 1); 
            // All spanning forests rooted at A1, B1
            InternalType weight_A1_B1 = curr(3, 3);
            // All spanning forests rooted at A1, B1 with path from A0 to B1
            InternalType weight_A1_B1_with_path_A0_to_B1 = curr(0, 3);

            // Now remove the edges outgoing from B1 (without adding the edges
            // outgoing from A1 back into the graph!) and apply the Chebotarev-
            // Agaev recurrence again
            laplacian(3, 1) = 0;
            laplacian(3, 2) = 0;
            laplacian(3, 3) = 0;
            curr = Matrix<InternalType, Dynamic, Dynamic>::Identity(4, 4);
            curr = chebotarevAgaevRecurrence<InternalType>(0, laplacian, curr);
            // All spanning forests rooted at A0, A1, B1
            InternalType weight_A0_A1_B1 = curr(0, 0);
            // All spanning forests rooted at B0, A1, B1
            InternalType weight_B0_A1_B1 = curr(1, 1); 
            // All spanning forests rooted at A0, A1, B1 with path from B0 to A0
            InternalType weight_A0_A1_B1_with_path_B0_to_A0 = curr(1, 0);  
            // All spanning forests rooted at B0, A1, B1 with path from A0 to B0
            InternalType weight_B0_A1_B1_with_path_A0_to_B0 = curr(0, 1); 

            // Now add the edges outgoing from A1 back into the graph, and 
            // apply the Chebotarev-Agaev recurrence again 
            laplacian(2, 0) = -this->edges[node_A1][node_A0];    // A1 -> A0
            laplacian(2, 3) = -this->edges[node_A1][node_B1];    // A1 -> B1
            laplacian(2, 2) = -laplacian(2, 0) - laplacian(2, 3); 
            curr = Matrix<InternalType, Dynamic, Dynamic>::Identity(4, 4); 
            curr = chebotarevAgaevRecurrence<InternalType>(0, laplacian, curr);
            curr = chebotarevAgaevRecurrence<InternalType>(1, laplacian, curr);
            // All spanning forests rooted at A0, B1
            InternalType weight_A0_B1 = curr(0, 0);
            // All spanning forests rooted at B0, B1
            InternalType weight_B0_B1 = curr(1, 1); 
            // All spanning forests rooted at A0, B1 with path from B0 to A0
            InternalType weight_A0_B1_with_path_B0_to_A0 = curr(1, 0); 
            // All spanning forests rooted at A0, B1 with path from A1 to A0
            InternalType weight_A0_B1_with_path_A1_to_A0 = curr(2, 0);
            // All spanning forests rooted at B0, B1 with path from A0 to B0
            InternalType weight_B0_B1_with_path_A0_to_B0 = curr(0, 1);  
            // All spanning forests rooted at B0, B1 with path from A1 to B0
            InternalType weight_B0_B1_with_path_A1_to_B0 = curr(2, 1);
            // All spanning forests rooted at A1, B1 with path from A0 to A1
            InternalType weight_A1_B1_with_path_A0_to_A1 = curr(0, 2);

            // Now add the edges outgoing from B1 back into the graph, and
            // apply the Chebotarev-Agaev recurrence again
            laplacian(3, 1) = -this->edges[node_B1][node_B0];    // B1 -> B0
            laplacian(3, 2) = -this->edges[node_B1][node_A1];    // B1 -> A1
            laplacian(3, 3) = -laplacian(3, 1) - laplacian(3, 2);
            curr = Matrix<InternalType, Dynamic, Dynamic>::Identity(4, 4);
            curr = chebotarevAgaevRecurrence<InternalType>(0, laplacian, curr);
            curr = chebotarevAgaevRecurrence<InternalType>(1, laplacian, curr);
            curr = chebotarevAgaevRecurrence<InternalType>(2, laplacian, curr);
            InternalType weight_A0 = curr(0, 0);    // All spanning trees rooted at A0
            InternalType weight_B0 = curr(1, 1);    // All spanning trees rooted at B0
            InternalType weight_A1 = curr(2, 2);    // All spanning trees rooted at A1
            InternalType weight_B1 = curr(3, 3);    // All spanning trees rooted at B1

            // Now remove the edges outgoing from A0 from the graph, and
            // apply the Chebotarev-Agaev recurrence again
            laplacian(0, 0) = 0;
            laplacian(0, 1) = 0; 
            laplacian(0, 2) = 0; 
            curr = Matrix<InternalType, Dynamic, Dynamic>::Identity(4, 4);
            curr = chebotarevAgaevRecurrence<InternalType>(0, laplacian, curr);
            curr = chebotarevAgaevRecurrence<InternalType>(1, laplacian, curr);
            // All spanning forests rooted at A0, A1 with path from B0 to A1
            InternalType weight_A0_A1_with_path_B0_to_A1 = curr(1, 2); 
            // All spanning forests rooted at A0, B1 with path from B0 to B1
            InternalType weight_A0_B1_with_path_B0_to_B1 = curr(1, 3); 
            // All spanning forests rooted at A0, B1 with path from A1 to B1
            InternalType weight_A0_B1_with_path_A1_to_B1 = curr(2, 3);
            // All spanning forests rooted at A0, A1 with path from B1 to A1
            InternalType weight_A0_A1_with_path_B1_to_A1 = curr(3, 2); 

            // Now remove the edges outgoing from A1 from the graph (again), and
            // apply the Chebotarev-Agaev recurrence again
            laplacian(2, 0) = 0; 
            laplacian(2, 2) = 0; 
            laplacian(2, 3) = 0; 
            curr = Matrix<InternalType, Dynamic, Dynamic>::Identity(4, 4);
            curr = chebotarevAgaevRecurrence<InternalType>(0, laplacian, curr);
            // All spanning forests rooted at A0, A1, B1 with path from B0 to B1
            InternalType weight_A0_A1_B1_with_path_B0_to_B1 = curr(1, 3);

            // Now add the edges outgoing from A1 back into the graph, then
            // remove the edges outgoing from B1 from the graph (again), and 
            // apply the Chebotarev-Agaev recurrence again (one last time!) 
            laplacian(2, 0) = -this->edges[node_A1][node_A0];    // A1 -> A0
            laplacian(2, 3) = -this->edges[node_A1][node_B1];    // A1 -> B1
            laplacian(2, 2) = -laplacian(2, 0) - laplacian(2, 3);
            laplacian(3, 1) = 0; 
            laplacian(3, 2) = 0; 
            laplacian(3, 3) = 0;
            curr = Matrix<InternalType, Dynamic, Dynamic>::Identity(4, 4);
            curr = chebotarevAgaevRecurrence<InternalType>(0, laplacian, curr);
            // All spanning forests rooted at A0, A1, B1 with path from B0 to A1
            InternalType weight_A0_A1_B1_with_path_B0_to_A1 = curr(1, 2);

            // Assemble vectors to pass to the five operators defined above ...
            //
            // The (2*i)-th column of v_alpha contains the weights of 
            // - spanning forests rooted at (A,i)
            // - spanning forests rooted at \{(A,i),(A,N)\} with path (B,N) -> (A,i)
            // - spanning forests rooted at \{(A,i),(B,N)\} with path (A,N) -> (A,i)
            //
            // The (2*i+1)-th column of v_alpha contains the weights of  
            // - spanning forests rooted at (B,i)
            // - spanning forests rooted at \{(B,i),(A,N)\} with path (B,N) -> (B,i)
            // - spanning forests rooted at \{(B,i),(B,N)\} with path (A,N) -> (B,i)
            Matrix<InternalType, 3, Dynamic> v_alpha(3, 2 * this->N + 2);
            
            // The i-th column of v_beta1 contains the weights of 
            // - spanning forests rooted at (A,i+1)
            // - spanning forests rooted at (B,i+1)
            // - spanning forests rooted at \{(A,i+1),(B,i+1)\}
            // on the (i+1)-th grid subgraph
            Matrix<InternalType, 3, Dynamic> v_beta1(3, this->N);

            // The (2*i)-th column of v_beta2 contains the weights of 
            // - spanning forests rooted at \{(A,i),(A,N)\}
            // - spanning forests rooted at \{(A,i),(B,N)\}
            // - spanning forests rooted at \{(A,i),(A,N),(B,N)\}
            //
            // The (2*i+1)-th column of v_beta2 contains the weights of 
            // - spanning forests rooted at \{(B,i),(A,N)\}
            // - spanning forests rooted at \{(B,i),(B,N)\}
            // - spanning forests rooted at \{(B,i),(A,N),(B,N)\}
            Matrix<InternalType, 3, Dynamic> v_beta2(3, 2 * this->N + 2);

            // The (2*i)-th column of v_beta3 contains the weights of 
            // - spanning forests rooted at \{(A,i),(A,N)\} with path (A,0) -> (A,i)
            // - spanning forests rooted at \{(A,i),(B,N)\} with path (A,0) -> (A,i)
            // - spanning forests rooted at \{(A,i),(A,N),(B,N)\} with path (A,0) -> (A,i)
            //
            // The (2*i+1)-th column of v_beta3 contains the weights of 
            // - spanning forests rooted at \{(B,i),(A,N)\} with path (A,0) -> (B,i)
            // - spanning forests rooted at \{(B,i),(B,N)\} with path (A,0) -> (B,i)
            // - spanning forests rooted at \{(B,i),(A,N),(B,N)\} with path (A,0) -> (B,i)
            Matrix<InternalType, 3, Dynamic> v_beta3(3, 2 * this->N + 2);

            // The i-th column of v_beta4 contains the weights of 
            // - spanning forests rooted at (A,0)
            // - spanning forests rooted at \{(A,0),(A,i)\}
            // - spanning forests rooted at \{(A,0),(B,i)\}
            // - spanning forests rooted at \{(A,0),(A,i),(B,i)\}
            // on the (i+1)-th grid subgraph 
            Matrix<InternalType, 4, Dynamic> v_beta4(4, this->N);  
            
            // The i-th column of v_delta1 contains the weights of 
            // - spanning forests rooted at (A,i+1)
            // - spanning forests rooted at \{(A,i+1),(B,i+1)\} with path (A,0) -> (A,i+1)
            // on the (i+1)-th grid subgraph
            //
            // The i-th column of v_delta2 contains the weights of 
            // - spanning forests rooted at (B,i+1)
            // - spanning forests rooted at \{(A,i+1),(B,i+1)\} with path (A,0) -> (B,i+1)
            // on the (i+1)-th grid subgraph
            Matrix<InternalType, 2, Dynamic> v_delta1(2, this->N);
            Matrix<InternalType, 2, Dynamic> v_delta2(2, this->N);

            // The (2*i)-th column of v_theta1 contains the weights of 
            // - spanning forests rooted at \{(A,0),(A,N)\} with path (A,i) -> (A,N)
            // - spanning forests rooted at \{(A,0),(B,N)\} with path (A,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (A,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (A,i) -> (A,N)
            // 
            // The (2*i+1)-th column of v_theta1 contains the weights of 
            // - spanning forests rooted at \{(A,0),(A,N)\} with path (B,i) -> (A,N)
            // - spanning forests rooted at \{(A,0),(B,N)\} with path (B,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (B,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (B,i) -> (A,N)
            Matrix<InternalType, 4, Dynamic> v_theta1(4, 2 * this->N + 2);

            // The i-th column of v_theta2 contains the weights of 
            // - spanning forests rooted at \{(A,0),(B,i+1)\} with path (A,i+1) -> (B,i+1)
            // - spanning forests rooted at \{(A,0),(A,i+1)\} with path (B,i+1) -> (A,i+1)
            // on the (i+1)-th grid subgraph 
            Matrix<InternalType, 2, Dynamic> v_theta2(2, this->N); 

            v_alpha(0, 0) = weight_A0;
            v_alpha(1, 0) = weight_A0_A1_with_path_B1_to_A0;
            v_alpha(2, 0) = weight_A0_B1_with_path_A1_to_A0; 
            v_alpha(0, 1) = weight_B0;
            v_alpha(1, 1) = weight_B0_A1_with_path_B1_to_B0;
            v_alpha(2, 1) = weight_B0_B1_with_path_A1_to_B0;
            v_beta1(0, 0) = weight_A1;
            v_beta1(1, 0) = weight_B1;
            v_beta1(2, 0) = weight_A1_B1;
            v_beta2(0, 0) = weight_A0_A1; 
            v_beta2(1, 0) = weight_A0_B1; 
            v_beta2(2, 0) = weight_A0_A1_B1;
            v_beta2(0, 1) = weight_B0_A1; 
            v_beta2(1, 1) = weight_B0_B1; 
            v_beta2(2, 1) = weight_B0_A1_B1; 
            v_beta3(0, 0) = weight_A0_A1; 
            v_beta3(1, 0) = weight_A0_B1; 
            v_beta3(2, 0) = weight_A0_A1_B1;
            v_beta3(0, 1) = weight_B0_A1_with_path_A0_to_B0;
            v_beta3(1, 1) = weight_B0_B1_with_path_A0_to_B0; 
            v_beta3(2, 1) = weight_B0_A1_B1_with_path_A0_to_B0;
            v_beta4(0, 0) = weight_A0; 
            v_beta4(1, 0) = weight_A0_A1; 
            v_beta4(2, 0) = weight_A0_B1; 
            v_beta4(3, 0) = weight_A0_A1_B1; 
            v_delta1(0, 0) = weight_A1; 
            v_delta1(1, 0) = weight_A1_B1_with_path_A0_to_A1;
            v_delta2(0, 0) = weight_B1; 
            v_delta2(1, 0) = weight_A1_B1_with_path_A0_to_B1;
            v_theta1(0, 0) = 0; 
            v_theta1(1, 0) = 0; 
            v_theta1(2, 0) = 0; 
            v_theta1(3, 0) = 0;
            v_theta1(0, 1) = weight_A0_A1_with_path_B0_to_A1;
            v_theta1(1, 1) = weight_A0_B1_with_path_B0_to_B1; 
            v_theta1(2, 1) = weight_A0_A1_B1_with_path_B0_to_B1;
            v_theta1(3, 1) = weight_A0_A1_B1_with_path_B0_to_A1;
            v_theta2(0, 0) = weight_A0_B1_with_path_A1_to_B1; 
            v_theta2(1, 0) = weight_A0_A1_with_path_B1_to_A1;

            // Apply operator Alpha (Eqn.19 in the SI) onto columns 0 and 1
            // of v_alpha to yield the weights of: 
            // - spanning forests rooted at (A,0) 
            // - spanning forests rooted at \{(A,0),(A,N)\} with path (B,N) -> (A,0)
            // - spanning forests rooted at \{(A,0),(B,N)\} with path (A,N) -> (A,0)
            // - spanning forests rooted at (B,0)
            // - spanning forests rooted at \{(B,0),(A,N)\} with path (B,N) -> (B,0)
            // - spanning forests rooted at \{(B,0),(B,N)\} with path (A,N) -> (A,0)
            // on the full grid graph ...
            //
            // ... while also applying operator Beta1 (Eqn.24 in the SI) onto
            // columns 0 and 1 of v_beta2 to yield the weights of: 
            // - spanning forests rooted at \{(A,0),(A,N)\}
            // - spanning forests rooted at \{(A,0),(B,N)\} 
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} 
            // - spanning forests rooted at \{(B,0),(A,N)\}
            // - spanning forests rooted at \{(B,0),(B,N)\} 
            // - spanning forests rooted at \{(B,0),(A,N),(B,N)\} 
            // on the full grid graph
            //
            // ... while also applying operator Beta1 (Eqn.27 in the SI) to
            // obtain the weights of: 
            // - spanning forests rooted at \{(A,0),(A,N)\} (with path (A,0) -> (A,0))
            // - spanning forests rooted at \{(A,0),(B,N)\} (with path (A,0) -> (A,0))
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} (with path (A,0) -> (A,0))
            // - spanning forests rooted at \{(B,0),(A,N)\} with path (A,0) -> (B,0)
            // - spanning forests rooted at \{(B,0),(B,N)\} with path (A,0) -> (B,0)
            // - spanning forests rooted at \{(B,0),(A,N),(B,N)\} with path (A,0) -> (B,0)
            // on the (i+j)-th grid subgraph
            //
            // ... while also applying operator Theta1 (Eqn.32 in the SI) to 
            // obtain the weights of: 
            // - spanning forests rooted at \{(A,0),(A,N)\} with path (B,0) -> (A,N)
            // - spanning forests rooted at \{(A,0),(B,N)\} with path (B,0) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (B,0) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (B,0) -> (A,N)
            // on the full grid graph 
            //
            // The j-th rung includes the labels between (A,j), (B,j), (A,j+1), (B,j+1)
            InternalType weight_A0_sub = v_alpha(0, 0); 
            InternalType weight_B0_sub = v_alpha(0, 1);
            Matrix<InternalType, 4, 1> y; 
            for (unsigned j = 1; j < this->N; ++j)
            {
                v_alpha.col(0) = alpha<InternalType>(this->rung_labels[j], v_alpha.col(0));      // Eqn.19
                v_alpha.col(1) = alpha<InternalType>(this->rung_labels[j], v_alpha.col(1));      // Eqn.19
                y << weight_A0_sub, v_beta2(0, 0), v_beta2(1, 0), v_beta2(2, 0); 
                v_beta2.col(0) = beta1<InternalType>(this->rung_labels[j], y);                   // Eqn.24
                y << weight_B0_sub, v_beta2(0, 1), v_beta2(1, 1), v_beta2(2, 1); 
                v_beta2.col(1) = beta1<InternalType>(this->rung_labels[j], y);                   // Eqn.24
                y << weight_A0_sub, v_beta3(0, 0), v_beta3(1, 0), v_beta3(2, 0); 
                v_beta3.col(0) = beta1<InternalType>(this->rung_labels[j], y);                   // Eqn.27
                y << weight_B0_sub, v_beta3(0, 1), v_beta3(1, 1), v_beta3(2, 1);
                v_beta3.col(1) = beta1<InternalType>(this->rung_labels[j], y);                   // Eqn.27
                v_theta1.col(1) = theta1<InternalType>(this->rung_labels[j], v_theta1.col(1));   // Eqn.32
                    
                // The new values of v_alpha(0, 0) and v_alpha(0, 1) are the 
                // weights of spanning forests rooted at (A,0) or at (B,0) in
                // the (j+1)-th grid subgraph
                weight_A0_sub = v_alpha(0, 0); 
                weight_B0_sub = v_alpha(0, 1);

                // The new values of v_beta2.col(0) are the weights of: 
                // - spanning forests rooted at \{(A,0),(A,j+1)\}
                // - spanning forests rooted at \{(A,0),(B,j+1)\}
                // - spanning forests rooted at \{(A,0),(A,j+1),(B,j+1)\}
                // of the (j+1)-th grid graph  
                v_beta4(0, j) = weight_A0_sub;
                v_beta4(1, j) = v_beta2(0, 0); 
                v_beta4(2, j) = v_beta2(1, 0); 
                v_beta4(3, j) = v_beta2(2, 0);
            }

            // For i = 1, ..., this->N - 1, do the following:
            Matrix<InternalType, 3, 1> v_gamma1, v_gamma2, v_epsilon1, v_epsilon2, v_epsilon3, v_epsilon4;
            Matrix<InternalType, 4, 1> v_omega1, v_omega2;  
            for (unsigned i = 1; i < this->N; ++i)
            {
                // Apply operators Gamma1 (Eqn.22 in the SI) and Gamma2 (Eqn.23
                // in the SI) onto the current values in v_beta
                //
                // Gamma1 yields the weights of:
                // - spanning forests rooted at (A,i)
                // - spanning forests rooted at \{(A,i),(A,i+1)\} with path (B,i+1) -> (A,i)
                // - spanning forests rooted at \{(A,i),(B,i+1)\} with path (A,i+1) -> (A,i) 
                // on the (i+1)-th grid subgraph
                //
                // Gamma2 yields the weights of: 
                // - spanning forests rooted at (B,i)
                // - spanning forests rooted at \{(B,i),(A,i+1)\} with path (B,i+1) -> (B,i)
                // - spanning forests rooted at \{(B,i),(B,i+1)\} with path (A,i+1) -> (B,i)
                // on the (i+1)-th grid subgraph
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                Matrix<InternalType, 2, 1> v_beta_A, v_beta_B; 
                v_beta_A << v_beta1(0, i-1), v_beta1(2, i-1); 
                v_beta_B << v_beta1(1, i-1), v_beta1(2, i-1); 
                v_gamma1 = gamma1<InternalType>(this->rung_labels[i], v_beta_A); 
                v_gamma2 = gamma2<InternalType>(this->rung_labels[i], v_beta_B);

                // Apply operator Epsilon1 (Eqn.25 in the SI) to obtain the 
                // weights of
                // - spanning forests rooted at \{(A,i),(A,i+1)\}
                // - spanning forests rooted at \{(A,i),(B,i+1)\}
                // - spanning forests rooted at \{(A,i),(A,i+1),(B,i+1)\}
                // of the (i+1)-th grid subgraph
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                v_epsilon1 = epsilon1<InternalType>(this->rung_labels[i], v_beta_A);

                // Apply operator Epsilon2 (Eqn.26 in the SI) to obtain the 
                // weights of 
                // - spanning forests rooted at \{(B,i),(A,i+1)\}
                // - spanning forests rooted at \{(B,i),(B,i+1)\}
                // - spanning forests rooted at \{(B,i),(A,i+1),(B,i+1)\}
                // of the (i+1)-th grid subgraph
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                v_epsilon2 = epsilon2<InternalType>(this->rung_labels[i], v_beta_B);  

                // Apply operator Epsilon1 (Eqn.28 in the SI) to obtain the 
                // weights of 
                // - spanning forests rooted at \{(A,i),(A,i+1)\} with path (A,0) -> (A,i)
                // - spanning forests rooted at \{(A,i),(B,i+1)\} with path (A,0) -> (A,i)
                // - spanning forests rooted at \{(A,i),(A,i+1),(B,i+1)\} with path (A,0) -> (A,i)
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                //
                // The (i-1)-th column of v_delta1 contains the weights of 
                // - spanning forests rooted at (A,i)
                // - spanning forests rooted at \{(A,i),(B,i)\} with path (A,0) -> (A,i)
                // of the i-th grid subgraph
                Matrix<InternalType, 2, 1> x;
                x << v_delta1(0, i-1), v_delta1(1, i-1); 
                v_epsilon3 = epsilon1<InternalType>(this->rung_labels[i], x);
                
                // Apply operator Epsilon2 (Eqn.29 in the SI) to obtain the 
                // weights of 
                // - spanning forests rooted at \{(B,i),(A,i+1)\} with path (A,0) -> (B,i)
                // - spanning forests rooted at \{(B,i),(B,i+1)\} with path (A,0) -> (B,i)
                // - spanning forests rooted at \{(B,i),(A,i+1),(B,i+1)\} with path (A,0) -> (B,i) 
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                //
                // The (i-1)-th column of v_delta2 contains the weights of 
                // - spanning forests rooted at (B,i)
                // - spanning forests rooted at \{(A,i),(B,i)\} with path (A,0) -> (B,i)
                // on the i-th grid subgraph
                x << v_delta2(0, i-1), v_delta2(1, i-1); 
                v_epsilon4 = epsilon2<InternalType>(this->rung_labels[i], x);

                // Apply operator Theta2 (Eqn.33 in the SI) to obtain the
                // weights of 
                // - spanning forests rooted at \{(A,0),(A,i+1)\} with path (A,i) -> (A,i+1)
                // - spanning forests rooted at \{(A,0),(B,i+1)\} with path (A,i) -> (B,i+1)
                // - spanning forests rooted at \{(A,0),(A,i+1),(B,i+1)\} with path (A,i) -> (B,i+1)
                // - spanning forests rooted at \{(A,0),(A,i+1),(B,i+1)\} with path (A,i) -> (A,i+1)
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                //
                // The (i-1)-th column of v_beta4 contains the weights of 
                // - spanning forests rooted at (A,0)
                // - spanning forests rooted at \{(A,0),(A,i)\}
                // - spanning forests rooted at \{(A,0),(B,i)\}
                // - spanning forests rooted at \{(A,0),(A,i),(B,i)\}
                // on the i-th grid subgraph
                //
                // The (i-1)-th column of v_theta2 contains the weights of 
                // - spanning forests rooted at \{(A,0),(B,i)\} with path (A,i) -> (B,i)
                // - spanning forests rooted at \{(A,0),(A,i)\} with path (B,i) -> (A,i)
                // on the i-th grid subgraph
                Matrix<InternalType, 3, 1> z;
                z << v_beta4(1, i-1), v_theta2(0, i-1), v_beta4(3, i-1); 
                v_omega1 = theta2<InternalType>(this->rung_labels[i], z);

                // Apply operator Theta3 (Eqn.34 in the SI) to obtain the
                // weights of 
                // - spanning forests rooted at \{(A,0),(A,i+1)\} with path (B,i) -> (A,i+1)
                // - spanning forests rooted at \{(A,0),(B,i+1)\} with path (B,i) -> (B,i+1)
                // - spanning forests rooted at \{(A,0),(A,i+1),(B,i+1)\} with path (B,i) -> (B,i+1)
                // - spanning forests rooted at \{(A,0),(A,i+1),(B,i+1)\} with path (B,i) -> (A,i+1)
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                //
                // The (i-1)-th column of v_beta4 contains the weights of 
                // - spanning forests rooted at (A,0)
                // - spanning forests rooted at \{(A,0),(A,i)\}
                // - spanning forests rooted at \{(A,0),(B,i)\}
                // - spanning forests rooted at \{(A,0),(A,i),(B,i)\}
                // on the i-th grid subgraph
                //
                // The (i-1)-th column of v_theta2 contains the weights of 
                // - spanning forests rooted at \{(A,0),(B,i)\} with path (A,i) -> (B,i)
                // - spanning forests rooted at \{(A,0),(A,i)\} with path (B,i) -> (A,i)
                // on the i-th grid subgraph
                z << v_theta2(1, i-1), v_beta4(2, i-1), v_beta4(3, i-1); 
                v_omega2 = theta3<InternalType>(this->rung_labels[i], z);

                // Then, for j = 2, ..., this->N - i, apply operator Alpha
                // (Eqn.21 in the SI) to obtain the weights of: 
                // - spanning forests rooted at (A,i)
                // - spanning forests rooted at \{(A,i),(A,i+j)\} with path (B,i+j) -> (A,i)
                // - spanning forests rooted at \{(A,i),(B,i+j)\} with path (A,i+j) -> (A,i)
                // - spanning forests rooted at (B,i)
                // - spanning forests rooted at \{(B,i),(A,i+j)\} with path (B,i+j) -> (B,i)
                // - spanning forests rooted at \{(B,i),(B,i+j)\} with path (A,i+j) -> (B,i)
                // on the (i+j)-th grid subgraph ...
                //
                // ... while also applying operator Beta1 (Eqn.24 in the SI) to 
                // obtain the weights of: 
                // - spanning forests rooted at \{(A,i),(A,i+j)\}
                // - spanning forests rooted at \{(A,i),(B,i+j)\}
                // - spanning forests rooted at \{(A,i),(A,i+j),(B,i+j)\}
                // - spanning forests rooted at \{(B,i),(A,i+j)\}
                // - spanning forests rooted at \{(B,i),(B,i+j)\}
                // - spanning forests rooted at \{(B,i),(A,i+j),(B,i+j)\}
                // on the (i+j)-th grid subgraph ...
                //
                // ... while also applying operator Beta1 (Eqn.27 in the SI) to
                // obtain the weights of: 
                // - spanning forests rooted at \{(A,i),(A,i+j)\} with path (A,0) -> (A,i)
                // - spanning forests rooted at \{(A,i),(B,i+j)\} with path (A,0) -> (A,i) 
                // - spanning forests rooted at \{(A,i),(A,i+j),(B,i+j)\} with path (A,0) -> (A,i)
                // - spanning forests rooted at \{(B,i),(A,i+j)\} with path (A,0) -> (B,i)
                // - spanning forests rooted at \{(B,i),(B,i+j)\} with path (A,0) -> (B,i)
                // - spanning forests rooted at \{(B,i),(A,i+j),(B,i+j)\} with path (A,0) -> (B,i)
                // on the (i+j)-th grid subgraph
                //
                // ... while also applying operator Theta1 (Eqn.32 in the SI) to 
                // obtain the weights of: 
                // - spanning forests rooted at \{(A,0),(A,i+j)\} with path (A,i) -> (A,i+j)
                // - spanning forests rooted at \{(A,0),(B,i+j)\} with path (A,i) -> (B,i+j)
                // - spanning forests rooted at \{(A,0),(A,i+j),(B,i+j)\} with path (A,i) -> (B,i+j)
                // - spanning forests rooted at \{(A,0),(A,i+j),(B,i+j)\} with path (A,i) -> (A,i+j)
                // - spanning forests rooted at \{(A,0),(A,i+j)\} with path (B,i) -> (A,i+j)
                // - spanning forests rooted at \{(A,0),(B,i+j)\} with path (B,i) -> (B,i+j)
                // - spanning forests rooted at \{(A,0),(A,i+j),(B,i+j)\} with path (B,i) -> (B,i+j)
                // - spanning forests rooted at \{(A,0),(A,i+j),(B,i+j)\} with path (B,i) -> (A,i+j)
                // on the (i+j)-th grid subgraph 
                //
                // The (i+j-1)-th rung includes the labels between (A,i+j-1), (B,i+j-1), (A,i+j), (B,i+j)
                for (unsigned j = 2; j <= this->N - i; ++j)
                {
                    // v_gamma1(0) is, at this point, the weight of spanning
                    // forests rooted at (A,i) for the (i+j-1)-th grid subgraph
                    InternalType weight_Ai = v_gamma1(0); 

                    // v_gamma2(0) is, at this point, the weight of spanning 
                    // forests rooted at (B,i) for the (i+j-1)-th grid subgraph
                    InternalType weight_Bi = v_gamma2(0);

                    v_gamma1 = alpha<InternalType>(this->rung_labels[i+j-1], v_gamma1);    // Eqn.21 in the SI
                    v_gamma2 = alpha<InternalType>(this->rung_labels[i+j-1], v_gamma2);    // Eqn.21 in the SI
                    y << weight_Ai, v_epsilon1(0), v_epsilon1(1), v_epsilon1(2);
                    v_epsilon1 = beta1<InternalType>(this->rung_labels[i+j-1], y);         // Eqn.24 in the SI
                    y << weight_Bi, v_epsilon2(0), v_epsilon2(1), v_epsilon2(2);  
                    v_epsilon2 = beta1<InternalType>(this->rung_labels[i+j-1], y);         // Eqn.24 in the SI
                    y << weight_Ai, v_epsilon3(0), v_epsilon3(1), v_epsilon3(2); 
                    v_epsilon3 = beta1<InternalType>(this->rung_labels[i+j-1], y);         // Eqn.27 in the SI
                    y << weight_Bi, v_epsilon4(0), v_epsilon4(1), v_epsilon4(2);
                    v_epsilon4 = beta1<InternalType>(this->rung_labels[i+j-1], y);         // Eqn.27 in the SI
                    v_omega1 = theta1<InternalType>(this->rung_labels[i+j-1], v_omega1);   // Eqn.32 in the SI
                    v_omega2 = theta1<InternalType>(this->rung_labels[i+j-1], v_omega2);   // Eqn.32 in the SI
                }
                int index_Ai = 2 * i; 
                int index_Bi = 2 * i + 1; 
                v_alpha.col(index_Ai) = v_gamma1; 
                v_alpha.col(index_Bi) = v_gamma2;
                v_beta2.col(index_Ai) = v_epsilon1;
                v_beta2.col(index_Bi) = v_epsilon2;
                v_beta3.col(index_Ai) = v_epsilon3; 
                v_beta3.col(index_Bi) = v_epsilon4;
                v_theta1.col(index_Ai) = v_omega1; 
                v_theta1.col(index_Bi) = v_omega2;

                // Apply operator Beta (Eqn.20 in the SI) to obtain the
                // weights of
                // - spanning forests rooted at (A,i+1)
                // - spanning forests rooted at (B,i+1)
                // - spanning forests rooted at \{(A,i+1),(B,i+1)\}
                // on the (i+1)-th grid subgraph
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                //
                // The (i-1)-th column of v_beta1 contains the weights of 
                // - spanning forests rooted at (A,i)
                // - spanning forests rooted at (B,i)
                // - spanning forests rooted at \{(A,i),(B,i)\}
                // on the i-th grid subgraph 
                v_beta1.col(i) = beta<InternalType>(this->rung_labels[i], v_beta1.col(i-1));

                // Apply operator Delta1 (Eqn.30 in the SI) to obtain the
                // weights of 
                // - spanning forests rooted at (A,i+1)
                // - spanning forests rooted at \{(A,i+1),(B,i+1)\} with path (A,0) -> (A,i+1)
                // on the (i+1)-th grid subgraph
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                //
                // The (i-1)-th column of v_delta1 contains the corresponding 
                // spanning forests of the i-th grid subgraph 
                InternalType result_delta1 = delta1<InternalType>(this->rung_labels[i], v_delta1.col(i-1));
                v_delta1(0, i) = v_beta1(0, i); 
                v_delta1(1, i) = result_delta1;

                // Apply operator Delta2 (Eqn.31 in the SI) to obtain the 
                // weights of 
                // - spanning forests rooted at (B,i+1)
                // - spanning forests rooted at \{(A,i+1),(B,i+1)\} with path (A,0) -> (B,i+1)
                // on the (i+1)-th grid subgraph
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                //
                // The (i-1)-th column of v_delta2 contains the corresponding 
                // spanning forests of the i-th grid subgraph 
                InternalType result_delta2 = delta2<InternalType>(this->rung_labels[i], v_delta2.col(i-1));
                v_delta2(0, i) = v_beta1(1, i); 
                v_delta2(1, i) = result_delta2;

                // Apply operator Theta4 (Eqn.35 in the SI) to obtain the 
                // weights of 
                // - spanning forests rooted at \{(A,0),(B,i+1)\} with path (A,i+1) -> (B,i+1)
                // - spanning forests rooted at \{(A,0),(A,i+1)\} with path (B,i+1) -> (A,i+1)
                // on the (i+1)-th grid subgraph 
                //
                // The i-th rung includes the labels between (A,i), (B,i), (A,i+1), (B,i+1)
                //
                // The (i-1)-th column of v_beta4 contains the weights of: 
                // - spanning forests rooted at \{(A,0)\}
                // - spanning forests rooted at \{(A,0),(A,i)\} 
                // - spanning forests rooted at \{(A,0),(B,i)\} 
                // - spanning forests rooted at \{(A,0),(A,i),(B,i)\} 
                // on the i-th grid subgraph
                //
                // The (i-1)-th column of v_theta2 contains the weights of:
                // - spanning forests rooted at \{(A,0),(B,i)\} with path (A,i) -> (B,i) 
                // - spanning forests rooted at \{(A,0),(A,i)\} with path (B,i) -> (A,i)
                // on the i-th grid subgraph 
                Matrix<InternalType, 6, 1> w;
                w << v_beta4(0, i-1), v_beta4(1, i-1), v_beta4(2, i-1), v_beta4(3, i-1), 
                     v_theta2(0, i-1), v_theta2(1, i-1); 
                v_theta2.col(i) = theta4<InternalType>(this->rung_labels[i], w); 
            }
           
            return std::make_tuple(v_alpha, v_beta1, v_beta2, v_beta3, v_delta1, v_delta2, v_theta1, v_theta2); 
        }

    public:
        /**
         * Constructor for a grid graph of length 0, i.e., the two nodes 
         * `A0` and `B0` and the edges between them, with edge labels 
         * set to unity. 
         */
        GridGraph() : LabeledDigraph<InternalType, IOType>()
        {
            // Add nodes ...
            this->N = 0;
            Node* node_A = this->__addNode("A0");
            Node* node_B = this->__addNode("B0");

            // ... and edges 
            this->edges[node_A][node_B] = 1;
            this->edges[node_B][node_A] = 1;
            this->start.first = 1;
            this->start.second = 1;
        }

        /**
         * Constructor for a grid graph of given length, with edge labels 
         * set to unity.
         *
         * @param N Length of desired grid graph; `this->N` is set to `N`, and
         *          `this->numnodes` to `2 * N + 2`. 
         */
        GridGraph(unsigned N) : LabeledDigraph<InternalType, IOType>()
        {
            // Add the zeroth nodes ...
            this->N = N;
            Node* node_A = this->__addNode("A0");
            Node* node_B = this->__addNode("B0");
            Node* next_A = nullptr;
            Node* next_B = nullptr;

            // ... and the zeroth edges 
            this->edges[node_A][node_B] = 1;
            this->edges[node_B][node_A] = 1;
            this->start.first = 1;
            this->start.second = 1;

            for (unsigned i = 0; i < N; ++i)
            {
                // Add the (i+1)-th nodes ...
                std::stringstream ssa, ssb; 
                ssa << "A" << i + 1;
                ssb << "B" << i + 1;
                next_A = this->__addNode(ssa.str());
                next_B = this->__addNode(ssb.str());

                // ... and the six edges between/among the i-th and (i+1)-th nodes
                this->edges[node_A][next_A] = 1;
                this->edges[next_A][node_A] = 1;
                this->edges[node_B][next_B] = 1; 
                this->edges[next_B][node_B] = 1;
                this->edges[next_A][next_B] = 1;
                this->edges[next_B][next_A] = 1;
                std::array<InternalType, 6> labels = {1, 1, 1, 1, 1, 1};
                this->rung_labels.push_back(labels);
                node_A = next_A; 
                node_B = next_B; 
            }
        }

        /**
         * Trivial destructor.
         */
        ~GridGraph()
        {
        }

        /**
         * Set the labels on the edges `A0 -> B0` and `B0 -> A0` to the 
         * given values. 
         *
         * @param A0_to_B0 Label on `A0 -> B0`.
         * @param B0_to_A0 Label on `B0 -> A0`.
         */
        void setZerothLabels(IOType A0_to_B0, IOType B0_to_A0)
        {
            Node* node_A = this->nodes["A0"]; 
            Node* node_B = this->nodes["B0"];
            InternalType _A0_to_B0 = static_cast<InternalType>(A0_to_B0);
            InternalType _B0_to_A0 = static_cast<InternalType>(B0_to_A0); 
            this->edges[node_A][node_B] = _A0_to_B0; 
            this->edges[node_B][node_A] = _B0_to_A0;
            this->start.first = _A0_to_B0;
            this->start.second = _B0_to_A0;
        }

        /**
         * Add a new pair of nodes to the end of the graph (along with the 
         * six new edges), increasing its length by one. 
         *
         * @param labels Labels on the six new edges: `A{N} -> A{N+1}`, 
         *               `A{N+1} -> A{N}`, `B{N} -> B{N+1}`, `B{N+1} -> B{N}`,
         *               `A{N+1} -> B{N+1}`, and `B{N+1} -> A{N+1}`.
         */
        void addRungToEnd(const std::array<IOType, 6>& labels)
        {
            // Add the two new nodes ...
            std::stringstream sai, sbi, saj, sbj;
            sai << "A" << this->N;
            sbi << "B" << this->N;
            saj << "A" << this->N + 1;
            sbj << "B" << this->N + 1;
            Node* node_A = this->nodes[sai.str()];
            Node* node_B = this->nodes[sbi.str()];
            Node* next_A = this->__addNode(saj.str());
            Node* next_B = this->__addNode(sbj.str());

            // ... and the six new edges
            std::array<InternalType, 6> _labels; 
            for (unsigned i = 0; i < 6; ++i)
                _labels[i] = static_cast<InternalType>(labels[i]);
            this->edges[node_A][next_A] = _labels[0]; 
            this->edges[next_A][node_A] = _labels[1];
            this->edges[node_B][next_B] = _labels[2];
            this->edges[next_B][node_B] = _labels[3]; 
            this->edges[next_A][next_B] = _labels[4];
            this->edges[next_B][next_A] = _labels[5];
            this->rung_labels.push_back(_labels);
            this->N++;
        }

        /** 
         * Remove the last pair of nodes from the graph (along with the six
         * associated edges), decreasing its length by one. 
         *
         * @throws std::runtime_error if `this->N` is zero. 
         */
        void removeRungFromEnd()
        {
            // Throw an exception if N == 0, in which case removal is disallowed
            if (this->N == 0)
                throw std::runtime_error("Grid graph cannot be empty");

            // Delete A{N} and B{N} from this->order
            this->order.pop_back();
            this->order.pop_back();

            // Delete all edges with A{N} as source
            std::stringstream ss; 
            ss << "A" << this->N; 
            Node* node_A = this->nodes[ss.str()];
            this->edges[node_A].clear(); 

            // Delete all edges with B{N} as source
            ss.str(std::string()); 
            ss << "B" << this->N; 
            Node* node_B = this->nodes[ss.str()];
            this->edges[node_B].clear();

            // Remove the edges A{N-1} -> A{N} and B{N-1} -> B{N}
            ss.str(std::string()); 
            ss << "A" << this->N - 1; 
            Node* prev_A = this->nodes[ss.str()];
            this->edges[prev_A].erase(node_A);
            ss.str(std::string()); 
            ss << "B" << this->N - 1; 
            Node* prev_B = this->nodes[ss.str()];
            this->edges[prev_B].erase(node_B);  

            // Delete the heap-allocated Nodes themselves
            delete node_A; 
            delete node_B; 
            ss.str(std::string()); 
            ss << "A" << this->N; 
            this->nodes.erase(ss.str()); 
            ss.str(std::string()); 
            ss << "B" << this->N; 
            this->nodes.erase(ss.str()); 
            this->numnodes -= 2;

            // Remove the last sextet of edge labels
            this->rung_labels.pop_back(); 

            // Decrement N
            this->N--; 
        }

        /**
         * Ban node addition via `addNode()`: nodes can be added or removed 
         * only at the upper end of the graph. 
         *
         * @param id ID of node to be added (to match signature with parent method). 
         * @throws std::runtime_error if invoked at all. 
         */
        void addNode(std::string id)
        {
            throw std::runtime_error(
                "LabeledDigraph<...>::addNode() cannot be called; "
                "use GridGraph<...>::addRungToEnd() to successively add "
                "adjacent nodes and the edges between them"
            );
        }

        /**
         * Ban node removal via `removeNode()`: nodes can be added or removed 
         * only at the upper end of the graph. 
         *
         * @param id ID of node to be removed (to match signature with parent method). 
         * @throws std::runtime_error if invoked at all. 
         */
        void removeNode(std::string id)
        {
            throw std::runtime_error(
                "LabeledDigraph<...>::removeNode() cannot be called; "
                "use GridGraph<...>::removeRungFromEnd() to successively "
                "remove adjacent nodes and the edges between them"
            ); 
        }

        /**
         * Ban edge addition via `addEdge()`: edges can be added or removed 
         * only at the upper end of the graph. 
         *
         * @param source_id ID of source node of new edge (to match signature 
         *                  with parent method). 
         * @param target_id ID of target node of new edge (to match signature 
         *                  with parent method). 
         * @param label     Label on new edge (to match signature with parent method).
         * @throws std::runtime_error if invoked at all. 
         */
        void addEdge(std::string source_id, std::string target_id, IOType label = 1)
        {
            throw std::runtime_error(
                "LabeledDigraph<...>::addEdge() cannot be called; "
                "use GridGraph<...>::addRungToEnd() to successively add "
                "adjacent nodes and the edges between them"
            ); 
        }

        /**
         * Ban edge removal via `removeEdge()`: edges can be added or removed 
         * only at the upper end of the graph. 
         *
         * @param source_id ID of source node of edge to be removed (to match 
         *                  signature with parent method). 
         * @param target_id ID of target node of edge to be removed (to match
         *                  signature with parent method).
         * @throws std::runtime_error if invoked at all.
         */
        void removeEdge(std::string source_id, std::string target_id)
        {
            throw std::runtime_error(
                "LabeledDigraph<...>::removeEdge() cannot be called; "
                "use GridGraph<...>::removeRungFromEnd() to successively "
                "remove adjacent nodes and the edges between them"
            ); 
        }

        /**
         * Ban clearing via `clear()`: the graph must be non-empty. 
         *
         * @throws std::runtime_error if invoked at all. 
         */
        void clear()
        {
            throw std::runtime_error(
                "LabeledDigraph<...>::clear() cannot be called; use "
                "GridGraph<...>::reset() to remove all nodes but A0 and B0"
            ); 
        }

        /**
         * Remove all nodes and edges but `A0` and `B0` and the edges in
         * between. 
         */
        void reset()
        {
            // Clear the graph first, then add A0 and B0 back in 
            this->LabeledDigraph<InternalType, IOType>::clear();
            this->N = 0;
            Node* node_A = this->__addNode("A0");
            Node* node_B = this->__addNode("B0");
            this->edges[node_A][node_B] = 1;
            this->edges[node_B][node_A] = 1; 
            this->start.first = 1; 
            this->start.second = 1; 
            this->rung_labels.clear();
        }

        /**
         * Set the labels on the i-th sextet of edges (`A{i} -> A{i+1}`, 
         * `A{i+1} -> A{i}`, `B{i} -> B{i+1}`, `B{i+1} -> B{i}`,
         * `A{i+1} -> B{i+1}`, and `B{i+1} -> A{i+1}`) to the given values.
         *
         * This method throws an exception if `i` is not a valid index in 
         * the graph (i.e., if `i >= this->N`).
         *
         * @param i      Index of edge labels to update (see below). 
         * @param labels Sextet of new edge label values: `A{i} -> A{i+1}`, 
         *               `A{i+1} -> A{i}`, `B{i} -> B{i+1}`, `B{i+1} -> B{i}`,
         *               `A{i+1} -> B{i+1}`, and `B{i+1} -> A{i+1}`.
         * @throws std::invalid_argument if `i >= this->N`.
         */
        void setRungLabels(int i, const std::array<IOType, 6>& labels)
        {
            // Check that i < this->N
            if (i >= this->N)
                throw std::invalid_argument("Invalid rung index specified"); 

            // Obtain the i-th and (i+1)-th nodes
            std::stringstream sai, sbi, saj, sbj;
            sai << "A" << i;
            sbi << "B" << i;
            saj << "A" << i + 1;
            sbj << "B" << i + 1;
            Node* node_A = this->nodes[sai.str()]; 
            Node* node_B = this->nodes[sbi.str()];
            Node* next_A = this->nodes[saj.str()];
            Node* next_B = this->nodes[sbj.str()];

            // Cast the given label values to InternalType
            InternalType Ai_to_Aj = static_cast<InternalType>(labels[0]);
            InternalType Aj_to_Ai = static_cast<InternalType>(labels[1]);
            InternalType Bi_to_Bj = static_cast<InternalType>(labels[2]); 
            InternalType Bj_to_Bi = static_cast<InternalType>(labels[3]); 
            InternalType Aj_to_Bj = static_cast<InternalType>(labels[4]); 
            InternalType Bj_to_Aj = static_cast<InternalType>(labels[5]);

            // Update the stored edge label values 
            this->edges[node_A][next_A] = Ai_to_Aj;
            this->edges[next_A][node_A] = Aj_to_Ai;
            this->edges[node_B][next_B] = Bi_to_Bj; 
            this->edges[next_B][node_B] = Bj_to_Bi; 
            this->edges[next_A][next_B] = Aj_to_Bj;
            this->edges[next_B][next_A] = Bj_to_Aj;
            this->rung_labels[i][0] = Ai_to_Aj;
            this->rung_labels[i][1] = Aj_to_Ai; 
            this->rung_labels[i][2] = Bi_to_Bj; 
            this->rung_labels[i][3] = Bj_to_Bi;
            this->rung_labels[i][4] = Aj_to_Bj; 
            this->rung_labels[i][5] = Bj_to_Aj;
        }
        
        /**
         * Compute and return three quantities: 
         *
         * - the *splitting probability* of exiting the graph through
         *   `"B{this->N}"` (reaching an auxiliary upper exit node), and not
         *   through `"A0"` (reaching an auxiliary lower exit node);
         * - the reciprocal of the *unconditional mean first-passage time* to
         *   exiting the graph through `"A0"`, given that the exit rate from
         *   `"B{this->N}"` is zero.
         * - the reciprocal of the *conditional mean first-passage time* to 
         *   exiting the graph through `"B{this->N}"`, given that the exit
         *   through `"B{this->N}"` indeed occurs.  
         *
         * @param lower_exit_rate Rate of lower exit from `A0`.
         * @param upper_exit_rate Rate of upper exit from `B{this->N}`.
         * @returns               The above three quantities. 
         */
        std::tuple<IOType, IOType, IOType> getExitStats(IOType lower_exit_rate, IOType upper_exit_rate)
        {
            InternalType _lower_exit_rate = static_cast<InternalType>(lower_exit_rate); 
            InternalType _upper_exit_rate = static_cast<InternalType>(upper_exit_rate);
            Node* node_A0 = this->getNode("A0"); 
            Node* node_B0 = this->getNode("B0"); 

            // If this->N == 0, then ...
            if (this->N == 0)
            {
                // Compute the splitting probability of exiting the graph 
                // through (B,0)
                //
                // Get the following weights:
                // - spanning forests rooted at lower & upper exits with path
                //   (A,0) -> upper exit
                // - spanning forests rooted at lower & upper exits 
                InternalType weight_lower_upper_with_path_A0_to_upper = this->edges[node_A0][node_B0] * _upper_exit_rate;
                InternalType weight_lower_upper = weight_lower_upper_with_path_A0_to_upper; 
                weight_lower_upper += _lower_exit_rate * (this->edges[node_B0][node_A0] + _upper_exit_rate);
                InternalType upper_exit_prob = weight_lower_upper_with_path_A0_to_upper / weight_lower_upper; 

                // Compute the unconditional mean first-passage time to
                // exiting the graph through (A,0) (disregarding the given 
                // upper exit rate)
                // 
                // Get the following weights:
                // - spanning forests rooted at (A,0) & lower exit
                // - spanning forests rooted at (B,0) & lower exit with path 
                //   (A,0) -> (B,0)
                // - spanning forests rooted at lower exit
                InternalType weight_A0_lower = this->edges[node_B0][node_A0];
                InternalType weight_B0_lower_with_path_A0_to_B0 = this->edges[node_A0][node_B0];
                InternalType weight_lower = this->edges[node_B0][node_A0] * _lower_exit_rate;
                InternalType uncond_inv_mean_FPT = weight_lower / (
                    weight_A0_lower + weight_B0_lower_with_path_A0_to_B0
                );

                // Now compute the conditional mean first-passage time to 
                // exiting the graph through (B,N), given that this exit 
                // indeed occurs 
                //
                // Get the following weights:
                // - spanning forests rooted at \{(A,0), lower exit, upper exit\}
                // - spanning forests rooted at \{(B,0), lower exit, upper exit\}
                //   with path (A,0) -> (B,0)
                // - spanning forests rooted at \{lower exit, upper exit\} with 
                //   path (A,0) -> upper exit
                // - spanning forests rooted at \{lower exit, upper exit\} with
                //   path (B,0) -> upper exit
                InternalType weight_A0_lower_upper = this->edges[node_B0][node_A0] + _upper_exit_rate;
                InternalType weight_B0_lower_upper_with_path_A0_to_B0 = this->edges[node_A0][node_B0];
                InternalType weight_lower_upper_with_path_B0_to_upper = (
                    this->edges[node_A0][node_B0] * _upper_exit_rate +
                    _lower_exit_rate * _upper_exit_rate
                );
                InternalType cond_inv_mean_FPT = (weight_lower_upper_with_path_A0_to_upper * weight_lower_upper) / (
                    (weight_A0_lower_upper * weight_lower_upper_with_path_A0_to_upper) +
                    (weight_B0_lower_upper_with_path_A0_to_B0 * weight_lower_upper_with_path_B0_to_upper)
                );

                return std::make_tuple(
                    static_cast<IOType>(upper_exit_prob), 
                    static_cast<IOType>(uncond_inv_mean_FPT), 
                    static_cast<IOType>(cond_inv_mean_FPT)
                );  
            }

            // Compute the exit-related spanning forest weights of the graph 
            auto forest_weights = this->getExitRelatedSpanningForestWeights();

            // The (2*i)-th column of v_alpha contains the weights of 
            // - spanning forests rooted at (A,i)
            // - spanning forests rooted at \{(A,i),(A,N)\} with path (B,N) -> (A,i)
            // - spanning forests rooted at \{(A,i),(B,N)\} with path (A,N) -> (A,i)
            //
            // The (2*i+1)-th column of v_alpha contains the weights of  
            // - spanning forests rooted at (B,i)
            // - spanning forests rooted at \{(B,i),(A,N)\} with path (B,N) -> (B,i)
            // - spanning forests rooted at \{(B,i),(B,N)\} with path (A,N) -> (B,i)
            Matrix<InternalType, 3, Dynamic> v_alpha = std::get<0>(forest_weights);    // size (3, 2 * this->N + 2) 
            
            // The i-th column of v_beta1 contains the weights of 
            // - spanning forests rooted at (A,i+1)
            // - spanning forests rooted at (B,i+1)
            // - spanning forests rooted at \{(A,i+1),(B,i+1)\}
            // on the (i+1)-th grid subgraph
            Matrix<InternalType, 3, Dynamic> v_beta1 = std::get<1>(forest_weights);    // size (3, this->N)

            // The (2*i)-th column of v_beta2 contains the weights of 
            // - spanning forests rooted at \{(A,i),(A,N)\}
            // - spanning forests rooted at \{(A,i),(B,N)\}
            // - spanning forests rooted at \{(A,i),(A,N),(B,N)\}
            //
            // The (2*i+1)-th column of v_beta2 contains the weights of 
            // - spanning forests rooted at \{(B,i),(A,N)\}
            // - spanning forests rooted at \{(B,i),(B,N)\}
            // - spanning forests rooted at \{(B,i),(A,N),(B,N)\}
            Matrix<InternalType, 3, Dynamic> v_beta2 = std::get<2>(forest_weights);    // size (3, 2 * this->N + 2)
            
            // The (2*i)-th column of v_beta3 contains the weights of 
            // - spanning forests rooted at \{(A,i),(A,N)\} with path (A,0) -> (A,i)
            // - spanning forests rooted at \{(A,i),(B,N)\} with path (A,0) -> (A,i)
            // - spanning forests rooted at \{(A,i),(A,N),(B,N)\} with path (A,0) -> (A,i)
            //
            // The (2*i+1)-th column of v_beta3 contains the weights of 
            // - spanning forests rooted at \{(B,i),(A,N)\} with path (A,0) -> (B,i)
            // - spanning forests rooted at \{(B,i),(B,N)\} with path (A,0) -> (B,i)
            // - spanning forests rooted at \{(B,i),(A,N),(B,N)\} with path (A,0) -> (B,i)
            Matrix<InternalType, 3, Dynamic> v_beta3 = std::get<3>(forest_weights);    // size (3, 2 * this->N + 2)
            
            // The i-th column of v_delta1 contains the weights of 
            // - spanning forests rooted at (A,i+1)
            // - spanning forests rooted at \{(A,i+1),(B,i+1)\} with path (A,0) -> (A,i+1)
            // on the (i+1)-th grid subgraph
            //
            // The i-th column of v_delta2 contains the weights of 
            // - spanning forests rooted at (B,i+1)
            // - spanning forests rooted at \{(A,i+1),(B,i+1)\} with path (A,0) -> (B,i+1)
            // on the (i+1)-th grid subgraph
            Matrix<InternalType, 2, Dynamic> v_delta1 = std::get<4>(forest_weights);   // size (2, this->N)
            Matrix<InternalType, 2, Dynamic> v_delta2 = std::get<5>(forest_weights);   // size (2, this->N)

            // The (2*i)-th column of v_theta1 contains the weights of 
            // - spanning forests rooted at \{(A,0),(A,N)\} with path (A,i) -> (A,N)
            // - spanning forests rooted at \{(A,0),(B,N)\} with path (A,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (A,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (A,i) -> (A,N)
            // 
            // The (2*i+1)-th column of v_theta1 contains the weights of 
            // - spanning forests rooted at \{(A,0),(A,N)\} with path (B,i) -> (A,N)
            // - spanning forests rooted at \{(A,0),(B,N)\} with path (B,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (B,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (B,i) -> (A,N)
            Matrix<InternalType, 4, Dynamic> v_theta1 = std::get<6>(forest_weights);   // size (4, 2 * this->N + 2)

            // The i-th column of v_theta2 contains the weights of 
            // - spanning forests rooted at \{(A,0),(B,i+1)\} with path (A,i+1) -> (B,i+1)
            // - spanning forests rooted at \{(A,0),(A,i+1)\} with path (B,i+1) -> (A,i+1)
            // on the (i+1)-th grid subgraph 
            Matrix<InternalType, 2, Dynamic> v_theta2 = std::get<7>(forest_weights);   // size (2, this->N) 

            // Compute the splitting probability of exiting the graph through
            // B{this->N}
            InternalType weight_A0 = v_alpha(0, 0); 
            InternalType weight_AN = v_beta1(0, this->N - 1); 
            InternalType weight_BN = v_beta1(1, this->N - 1);
            InternalType weight_A0_BN = v_beta2(1, 0);
            InternalType tmp0 = weight_A0 * _lower_exit_rate;
            InternalType tmp1 = _lower_exit_rate * _upper_exit_rate;
            InternalType tmp2 = weight_BN * _upper_exit_rate; 
            InternalType tmp3 = weight_A0_BN * tmp1;
            InternalType weight_lower_upper = tmp0 + tmp2 + tmp3;  
            InternalType upper_exit_prob = tmp2 / weight_lower_upper; 

            // Compute the unconditional mean first-passage time to exiting
            // the graph through A0 (disregarding the given upper exit rate) 
            InternalType numer = 0;
            for (unsigned i = 0; i < this->N; ++i)
            {
                int index_Ai = 2 * i; 
                int index_Bi = 2 * i + 1;
                InternalType weight_Ai = v_alpha(0, index_Ai); 
                InternalType weight_Bi = v_alpha(0, index_Bi);
                numer += weight_Ai; 
                numer += weight_Bi; 
            }
            numer += weight_AN; 
            numer += weight_BN;
            InternalType uncond_inv_mean_FPT = _lower_exit_rate * weight_A0 / numer; 

            // Compute the conditional mean first-passage time to exiting 
            // the graph through (B,N), given that exit through (B,N) indeed
            // occurs 
            numer = 0; 
            //InternalType weight_A0_1 = weight_A0 + _upper_exit_rate * weight_A0_BN;
            //InternalType weight_A0_2 = tmp2;  
            //InternalType weight_B0_1 = v_alpha(0, 1) + _upper_exit_rate * v_beta3(1, 1); 
            //InternalType weight_B0_2 = tmp2 + tmp1 * v_theta1(1, 1);
            //numer += (weight_A0_1 * weight_A0_2); 
            //numer += (weight_B0_1 * weight_B0_2);  
            for (unsigned i = 0; i < this->N; ++i)
            {
                int index_Ai = 2 * i;
                int index_Bi = 2 * i + 1;
                InternalType weight_Ai_1 = v_alpha(0, index_Ai) + _upper_exit_rate * v_beta3(1, index_Ai);
                InternalType weight_Ai_2 = tmp2 + tmp1 * v_theta1(1, index_Ai);
                InternalType weight_Bi_1 = v_alpha(0, index_Bi) + _upper_exit_rate * v_beta3(1, index_Bi); 
                InternalType weight_Bi_2 = tmp2 + tmp1 * v_theta1(1, index_Bi);
                numer += (weight_Ai_1 * weight_Ai_2); 
                numer += (weight_Bi_1 * weight_Bi_2);  
            }
            InternalType weight_AN_1 = v_beta1(0, this->N - 1) + _upper_exit_rate * v_delta1(1, this->N - 1);
            InternalType weight_AN_2 = tmp2 + tmp1 * v_theta2(0, this->N - 1);  
            InternalType weight_BN_1 = weight_BN; 
            InternalType weight_BN_2 = tmp2 + tmp1 * weight_A0_BN;
            numer += (weight_AN_1 * weight_AN_2); 
            numer += (weight_BN_1 * weight_BN_2);  
            InternalType cond_inv_mean_FPT = tmp2 * weight_lower_upper / numer;  

            return std::make_tuple(
                static_cast<IOType>(upper_exit_prob),
                static_cast<IOType>(uncond_inv_mean_FPT), 
                static_cast<IOType>(cond_inv_mean_FPT)
            );
        }
};

#endif 
