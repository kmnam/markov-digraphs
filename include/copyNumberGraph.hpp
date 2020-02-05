#ifndef COPY_NUMBER_GRAPHS_HPP
#define COPY_NUMBER_GRAPHS_HPP

#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/container_hash/hash.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include "digraph.hpp"
#include "linalg.hpp"

/*
 * An implementation of a copy-number graph class (Nam, 2019) that inherits
 * from the MarkovDigraph class. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     2/3/2020
 */
using namespace Eigen;

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

template <typename T>
class CopyNumberGraph : public MarkovDigraph<T>
{
    /*
     * An implementation of a 1-D copy-number graph. 
     */
    protected:
        // All production edges in the graph, stored as a hash table
        std::unordered_map<Node<T>*, std::pair<Node<T>*, T> > prod;

    public:
        CopyNumberGraph() : MarkovDigraph<T>()
        {
            /*
             * Empty constructor.
             */
        }

        void addProductionEdge(std::string source_id, std::string target_id, T label)
        {
            /*
             * Add an mRNA production edge between two nodes, with the given 
             * nonzero label. If one already exists emanating from the given
             * source vertex, this method overwrites the existing edge. 
             */
            Node<T>* source = this->getNode(source_id);
            Node<T>* target = this->getNode(target_id);
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist");
            if (target == nullptr)
                throw std::runtime_error("Specified target node does not exist");
            if (label <= 0.0)
                throw std::invalid_argument("Specified production edge label is not positive");
            this->prod[source] = std::make_pair(target, label);
        }

        void clear()
        {
            /* 
             * Clear the graph's contents.
             *
             * Overrides MarkovDigraph<T>::clear().
             */
            // De-allocate all Node instances from heap memory
            for (auto&& node : this->nodes) delete node;

            // Clear all attributes
            this->nodes.clear();
            this->edges.clear();
            this->labels.clear();
            this->prod.clear();
        }

        template <typename U>
        CopyNumberGraph<U>* copy() const
        {
            /*
             * Return pointer to a new CopyNumberGraph object, possibly
             * with a different scalar type, with the same graph structure
             * and edge label values.
             *
             * Overrides MarkovDigraph<T>::copy().
             */
            CopyNumberGraph<U>* graph = new CopyNumberGraph<U>();

            // Copy over nodes with the same IDs
            for (auto&& node : this->nodes)
                graph->addNode(node->id);
            
            // Copy over edges and edge label values
            for (auto&& edge_label : this->labels)
            {
                Edge<T> edge = edge_label.first;
                U label(edge_label.second);
                graph->addEdge(edge.first->id, edge.second->id, label);
            }

            // Copy over mRNA production edges
            for (auto&& prod_edge : this->prod)
            {
                Node<T>* source = prod_edge.first;
                Node<T>* target = prod_edge.second.first;
                U label(prod_edge.second.second);
                graph->addProductionEdge(source->id, target->id, label);
            }
            
            return graph;
        }

        template <typename U>
        void copy(CopyNumberGraph<U>* graph) const
        {
            /*
             * Given pointer to an existing CopyNumberGraph object, possibly
             * with a different scalar type, copy over the graph details.
             *
             * Overrides MarkovDigraph<T>::copy(). 
             */
            // Clear the input graph's contents
            graph->clear();

            // Copy over nodes with the same IDs
            for (auto&& node : this->nodes)
                graph->addNode(node->id);

            // Copy over edges and edge label values
            for (auto&& edge_label : this->labels)
            {
                Edge<T> edge = edge_label.first;
                U label(edge_label.second);
                graph->addEdge(edge.first->id, edge.second->id, label);
            }

            // Copy over mRNA production edges
            for (auto&& prod_edge : this->prod)
            {
                Node<T>* source = prod_edge.first;
                Node<T>* target = prod_edge.second.first;
                U label(prod_edge.second.second);
                graph->addProductionEdge(source->id, target->id, label);
            }
        }

        Matrix<T, Dynamic, Dynamic> getProductionMatrix(bool diag = false)
        {
            /*
             * Return matrix of mRNA production rates. 
             *
             * If diag is true, then return a diagonal matrix: the (ii)-th
             * entry is the label of the production edge emanating from 
             * node i. If diag is false, then return a matrix in which the
             * (ij)-th entry is the label of the production edge from node 
             * j to node i, if one exists, and zero otherwise. 
             */
            unsigned dim = this->nodes.size();
            if (diag)    // If diag is true, run through the nodes in the graph,
            {            // looking for production edges emanating from them
                Matrix<T, Dynamic, 1> production_rates = Matrix<T, Dynamic, 1>::Zero(dim, 1);
                for (unsigned j = 0; j < dim; j++)
                {
                    auto it = this->prod.find(this->nodes[j]);
                    if (it != this->prod.end())
                        production_rates(j) = (it->second).second;
                }
                return production_rates.asDiagonal();
            }
            else    // If diag is false, run through the production edges, identify
            {       // their source/target nodes, and update corresponding entry
                Matrix<T, Dynamic, Dynamic> production_rates = Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
                for (unsigned j = 0; j < dim; j++)
                {
                    Node<T>* source = this->nodes[j];
                    auto it = this->prod.find(source);
                    if (it != this->prod.end())
                    {
                        Node<T>* target = (it->second).first;
                        T rate = (it->second).second;
                        unsigned k =
                            std::find(this->nodes.begin(), this->nodes.end(), target) - this->nodes.begin();
                        production_rates(k,j) = rate;
                    }
                }
                return production_rates;
            }
        }

        Matrix<T, Dynamic, 1> getCopyNumberMoments(unsigned nmoments, T tol)
        {
            /*
             * Compute the steady-state moments of the copy-number distribution.
             *
             * All mRNA production rates are assumed to be normalized by the 
             * mRNA degradation rate.
             */
            using std::pow;

            // Get the Laplacian and production rate matrices
            Matrix<T, Dynamic, Dynamic> laplacian = this->getLaplacian();
            Matrix<T, Dynamic, Dynamic> production_matrix = this->getProductionMatrix();
            Matrix<T, Dynamic, Dynamic> production_diag = this->getProductionMatrix(true);

            // Obtain the nullspace matrix of the Laplacian matrix, modified
            // by the production rate matrices
            // TODO Replace with Chebotarev-Agaev's recurrence for graphs with 
            // fewer than 10 vertices
            Matrix<T, Dynamic, Dynamic> A = laplacian + production_matrix - production_diag;
            Matrix<T, Dynamic, Dynamic> nullmat = nullspaceSVD<T>(A, tol);

            // Each column of the nullspace matrix corresponds to a basis
            // vector of the nullspace; each row corresponds to a single 
            // vertex in the graph; since each vertex lies in at most a
            // single terminal SCC of the graph, each vertex has only
            // one nullspace basis vector contributing to (defining) its 
            // steady-state probability
            Array<T, Dynamic, Dynamic> steady_state = nullmat.array().rowwise().sum();
            steady_state /= steady_state.sum();
           
            // Solve for the first <nmoments> binomial moments via 
            // the recurrence relation
            unsigned dim = this->nodes.size();
            Matrix<T, Dynamic, Dynamic> binomial_moments(nmoments+1, dim);
            binomial_moments.row(0) = steady_state.matrix().transpose();
            Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(dim, dim);
            // TODO Replace with an algorithm based on the Chebotarev-Agaev
            // recurrence for graphs with fewer than 10 vertices 
            for (unsigned j = 1; j < nmoments + 1; j++)
            {
                binomial_moments.row(j) = (production_diag - production_matrix - laplacian + j * identity)
                    .colPivHouseholderQr().solve(production_matrix * binomial_moments.row(j-1).transpose()).transpose();
            }
            
            // Compute non-central moments from the binomial moments, via the 
            // following lower-triangular system: 
            //
            // [ binomials.row(1).sum() ]
            // [ binomials.row(2).sum() ]
            // [           ...          ]
            // [ binomials.row(m).sum() ]
            //
            //       [ s(1,1)/1!                                 ] [ 1st moment ]
            //     = [ s(2,1)/2!  s(2,2)/2!                      ] [ 2nd moment ]
            //       [    ...        ...        ...              ] [     ...    ]
            //       [ s(m,1)/m!  s(m,2)/m!     ...    s(m,m)/m! ] [ mth moment ]
            //
            // where m == nmoments and s(i,j) are Stirling numbers of the first kind
            Matrix<T, Dynamic, Dynamic> moments(nmoments+1, 1);
            moments(0) = 1.0;
            Matrix<T, Dynamic, Dynamic> S =
                signedStirlingNumbersOfFirstKindByFactorial<T>(nmoments);
            Matrix<T, Dynamic, Dynamic> binom_sums =
                binomial_moments.block(1, 0, nmoments, dim).rowwise().sum();
            S.template triangularView<Eigen::Lower>().solveInPlace(binom_sums);
            moments.block(1, 0, nmoments, 1) = binom_sums;

            // Compute standardized moments from the non-central moments
            Matrix<T, Dynamic, 1> standard_moments(nmoments);
            standard_moments(0) = moments(1);                         // Mean
            standard_moments(1) = moments(2) - pow(moments(1), 2.0);  // Variance
            for (unsigned i = 2; i < nmoments; i++)
            {
                // Compute higher standardized moments
                standard_moments(i) = 0.0;
                for (unsigned j = 0; j <= i+1; j++)
                {
                    standard_moments(i) += (
                        boost::math::binomial_coefficient<T>(i+1, j)
                        * pow(-moments(1), j) * moments(i+1-j)
                    );
                }
                standard_moments(i) /= pow(moments(2), 0.5 * (i+1));
            }

            return standard_moments;
        }
};

#endif
