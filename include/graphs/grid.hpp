/**
 * \line include/graphs/grid.hpp
 *
 * An implementation of the grid graph. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/29/2021
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
 * Apply operator alpha onto the given vector `v`, with the given sextet
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
    A(0, 0) = rA * (rB + d) + rB * c; 
    A(0, 1) = tmp0 * c; 
    A(0, 2) = tmp1 * d; 
    A(1, 0) = rB;
    A(1, 1) = tmp0;
    A(2, 0) = rA; 
    A(2, 2) = tmp1; 

    return A * v; 
}

/**
 * Apply operator beta onto the given vector `v`, with the given sextet 
 * of edge labels.
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 3). 
 * @returns           Output vector (length 8). 
 */
template <typename T>
Matrix<T, 9, 1> beta(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 3, 1> >& v)
{
    Matrix<T, 9, 3> A = Matrix<T, 9, 3>::Zero(); 
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = d + rB; 
    T tmp1 = rA * tmp0; 
    T tmp2 = rB * c;
    T tmp3 = fA * fB;
    T tmp4 = fB * rA;  
    A(0, 0) = tmp1 + tmp2;
    A(0, 2) = tmp4 * d; 
    A(1, 1) = tmp1 + tmp2; 
    A(1, 2) = fA * tmp2;
    A(2, 0) = fA * tmp0;
    A(2, 1) = fB * d; 
    A(2, 2) = tmp3 * d;
    A(3, 0) = fA * c; 
    A(3, 1) = fB * (c + rA);
    A(3, 2) = tmp3 * c;
    A(4, 0) = rB; 
    A(5, 0) = rA; 
    A(5, 2) = tmp4; 
    A(6, 1) = rB; 
    A(6, 2) = fA * rB;
    A(7, 1) = rA; 
    A(8, 0) = fA; 
    A(8, 1) = fB; 
    A(8, 2) = tmp3;

    return A * v;
}

/**
 * Apply operator gamma onto the given vector `v`, with the given sextet 
 * of edge labels.
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 4). 
 * @returns           Output vector (length 3). 
 */
template <typename T>
Matrix<T, 3, 1> gamma(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 4, 1> >& v)
{
    Matrix<T, 3, 4> A = Matrix<T, 3, 4>::Zero(); 
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = d + rB; 
    T tmp1 = fB * d;
    T tmp2 = c + rA;
    T tmp3 = fA * c;  
    A(0, 0) = tmp0; 
    A(0, 1) = fA * tmp0; 
    A(0, 2) = tmp1;
    A(0, 3) = fA * tmp1; 
    A(1, 0) = tmp2; 
    A(1, 1) = tmp3; 
    A(1, 2) = fB * tmp2;
    A(1, 3) = fB * tmp3; 
    A(2, 0) = 1; 
    A(2, 1) = fA; 
    A(2, 2) = fB; 
    A(2, 3) = fA * fB; 

    return A * v;
}

/**
 * Apply operator delta onto the given vector `v`, with the given sextet
 * of edge labels.
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 4). 
 * @returns           Output vector (length 8). 
 */
template <typename T>
Matrix<T, 8, 1> delta(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 4, 1> >& v)
{
    Matrix<T, 8, 4> A = Matrix<T, 8, 4>::Zero();  
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = d + rB; 
    T tmp1 = c + rA; 
    T tmp2 = fA * fB;
    A(0, 0) = tmp0; 
    A(0, 3) = fB * d; 
    A(1, 1) = tmp0; 
    A(1, 2) = fA * tmp0; 
    A(2, 0) = tmp1; 
    A(2, 3) = fB * tmp1; 
    A(3, 1) = tmp1; 
    A(3, 2) = fA * c; 
    A(4, 0) = fA;
    A(4, 2) = tmp2; 
    A(5, 1) = fB; 
    A(5, 3) = tmp2; 
    A(6, 0) = 1;
    A(6, 3) = fB; 
    A(7, 1) = 1; 
    A(7, 2) = fA;

    return A * v;
}

/**
 * Apply operator epsilon onto the given vector `v`, with the given sextet
 * of edge labels.
 *
 * @param rung_labels Sextet of edge labels taken from a rung in the grid graph.
 * @param v           Input vector (length 3). 
 * @returns           Output vector (length 8). 
 */
template <typename T>
Matrix<T, 8, 1> epsilon(const std::array<T, 6>& rung_labels, const Ref<const Matrix<T, 3, 1> >& v)
{
    Matrix<T, 8, 3> A = Matrix<T, 8, 3>::Zero();  
    T fA = rung_labels[0];
    T rA = rung_labels[1];
    T fB = rung_labels[2];
    T rB = rung_labels[3];
    T c = rung_labels[4];
    T d = rung_labels[5];
    T tmp0 = d + rB;
    T tmp1 = c + rA;
    A(0, 0) = tmp0; 
    A(0, 2) = fB * d; 
    A(1, 0) = tmp1; 
    A(1, 2) = fB * tmp1; 
    A(2, 0) = tmp0; 
    A(2, 1) = fA * tmp0;
    A(3, 0) = tmp1;
    A(3, 1) = fA * c;
    A(4, 0) = rB;
    A(4, 1) = fA * rB;
    A(5, 0) = rA; 
    A(5, 2) = fB * rA;
    A(6, 0) = 1;
    A(6, 2) = fB; 
    A(7, 0) = 1;
    A(7, 1) = fA; 

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
    private:
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
         * Compute and return two quantities: 
         *
         * - the *splitting probability* of exiting the graph through
         *   `"B{this->N}"` (reaching an auxiliary upper exit node), and not
         *   through `"A0"` (reaching an auxiliary lower exit node);
         * - the reciprocal of the *unconditional mean first-passage time* to
         *   exiting the graph through `"A0"`, given that the exit rate from
         *   `"B{this->N}"` is zero. 
         *
         * @param lower_exit_rate Rate of lower exit from `A0`.
         * @param upper_exit_rate Rate of upper exit from `B{this->N}`.
         * @returns               The above two quantities. 
         */
        std::pair<IOType, IOType> getExitStats(IOType lower_exit_rate, IOType upper_exit_rate)
        {
            InternalType _lower_exit_rate = static_cast<InternalType>(lower_exit_rate); 
            InternalType _upper_exit_rate = static_cast<InternalType>(upper_exit_rate);
            Node* node_A0 = this->getNode("A0"); 
            Node* node_B0 = this->getNode("B0"); 

            // If this->N == 0, then ...
            if (this->N == 0)
            {
                // Compute the splitting probability of exiting the graph 
                // through B0
                //
                // Start with the weight of all spanning forests rooted at 
                // lower & upper exits with a path from A0 to upper exit
                InternalType weight_exits_with_path_A0_to_upper = this->edges[node_A0][node_B0] * _upper_exit_rate;

                // Then get the weight of all spanning forests rooted at 
                // the two exits
                InternalType weight_exits = weight_exits_with_path_A0_to_upper; 
                weight_exits += _lower_exit_rate * (this->edges[node_B0][node_A0] + _upper_exit_rate);

                // Compute the unconditional mean first-passage time to
                // exiting the graph through A0 (disregarding the given 
                // upper exit rate)
                // 
                // Start with the weight of all spanning forests rooted at
                // A0 & lower exit (with a path from A0 to A0)
                InternalType weight_A0_lower = this->edges[node_B0][node_A0];

                // Then get the weight of all spanning forests rooted at
                // B0 & lower exit with a path from A0 to B0
                InternalType weight_B0_lower_with_path_A0_to_B0 = this->edges[node_A0][node_B0];

                // Then get the weight of all spanning trees rooted at lower exit
                InternalType weight_lower = this->edges[node_B0][node_A0] * _lower_exit_rate; 

                return std::make_pair(
                    static_cast<IOType>(
                        weight_exits_with_path_A0_to_upper / weight_exits
                    ), 
                    static_cast<IOType>(
                        weight_lower /
                        (weight_A0_lower + weight_B0_lower_with_path_A0_to_B0)
                    )
                );  
            } 

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

            // Finally add the edges outgoing from B1 back into the graph, 
            // and apply the Chebotarev-Agaev recurrence again
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

            // Assemble vectors to pass to the five operators defined above
            Matrix<InternalType, 3, Dynamic> v_alpha(3, 2 * this->N);
            Matrix<InternalType, 3, 1> v_beta, v_epsilon; 
            Matrix<InternalType, 4, Dynamic> v_gamma(4, 2 * this->N);
            Matrix<InternalType, 4, Dynamic> w_gamma(4, 2 * this->N); 
            Matrix<InternalType, 4, 1> v_delta; 
            v_alpha(0, 0) = weight_A0;
            v_alpha(1, 0) = weight_A0_A1_with_path_B1_to_A0;
            v_alpha(2, 0) = weight_A0_B1_with_path_A1_to_A0; 
            v_alpha(0, 1) = weight_B0;
            v_alpha(1, 1) = weight_B0_A1_with_path_B1_to_B0;
            v_alpha(2, 1) = weight_B0_B1_with_path_A1_to_B0;
            v_beta << weight_A1,
                      weight_B1,
                      weight_A1_B1;
            v_gamma(0, 0) = weight_A0;
            v_gamma(1, 0) = weight_A0_A1;
            v_gamma(2, 0) = weight_A0_B1;
            v_gamma(3, 0) = weight_A0_A1;
            v_gamma(0, 1) = weight_B0;
            v_gamma(1, 1) = weight_B0_A1_with_path_A0_to_B0;
            v_gamma(2, 1) = weight_B0_B1_with_path_A0_to_B0;
            v_gamma(3, 1) = weight_B0_A1_B1_with_path_A0_to_B0;
            w_gamma(0, 0) = weight_A0;
            w_gamma(1, 0) = weight_A0_A1;
            w_gamma(2, 0) = weight_A0_B1;
            w_gamma(3, 0) = weight_A0_A1_B1;
            w_gamma(0, 1) = weight_A0;
            w_gamma(1, 1) = weight_A0_A1_with_path_B0_to_A0;
            w_gamma(2, 1) = weight_A0_B1_with_path_B0_to_A0;
            w_gamma(3, 1) = weight_A0_A1_B1_with_path_B0_to_A0;
            v_delta << weight_A1,
                       weight_B1,
                       weight_A1_B1_with_path_A0_to_B1,
                       weight_A1_B1_with_path_A0_to_A1;
            v_epsilon << weight_A0,
                         weight_A0_A1_with_path_B1_to_A0,
                         weight_A0_B1_with_path_A1_to_A0;

            for (unsigned i = 1; i < this->N; ++i)
            {
                int next = 2 * i; 
                int after = 2 * i + 1;

                // Apply operator alpha onto the first 2 * i columns of v_alpha,
                // and overwrite these columns with the results
                for (unsigned j = 0; j < next; ++j)
                    v_alpha.col(j) = alpha<InternalType>(this->rung_labels[i], v_alpha.col(j));

                // Apply operator beta onto v_beta, populate the next 2 columns 
                // of v_alpha, and update v_beta
                Matrix<InternalType, 9, 1> w_beta = beta<InternalType>(this->rung_labels[i], v_beta); 
                v_alpha(0, next) = w_beta(0); 
                v_alpha(1, next) = w_beta(4); 
                v_alpha(2, next) = w_beta(5); 
                v_alpha(0, after) = w_beta(1); 
                v_alpha(1, after) = w_beta(6); 
                v_alpha(2, after) = w_beta(7);
                v_beta(0) = w_beta(2); 
                v_beta(1) = w_beta(3); 
                v_beta(2) = w_beta(8);

                // Apply operator gamma onto the first 2 * i columns of v_gamma,
                // and overwrite these columns with the results
                for (unsigned j = 0; j < next; ++j)
                {
                    Matrix<InternalType, 3, 1> u_gamma = gamma<InternalType>(this->rung_labels[i], v_gamma.col(j));
                    v_gamma(0, j) = v_alpha(0, j); 
                    v_gamma(1, j) = u_gamma(0); 
                    v_gamma(2, j) = u_gamma(1); 
                    v_gamma(3, j) = u_gamma(2); 
                }

                // Apply operator delta onto v_delta, populate the next 2 columns
                // of v_gamma, and update v_delta
                Matrix<InternalType, 8, 1> w_delta = delta<InternalType>(this->rung_labels[i], v_delta);
                v_gamma(0, next) = v_alpha(0, next);
                v_gamma(1, next) = w_delta(0); 
                v_gamma(2, next) = w_delta(2); 
                v_gamma(3, next) = w_delta(6); 
                v_gamma(0, after) = v_alpha(0, after); 
                v_gamma(1, after) = w_delta(1); 
                v_gamma(2, after) = w_delta(3); 
                v_gamma(3, after) = w_delta(7); 
                v_delta(0) = v_beta(0); 
                v_delta(1) = v_beta(1); 
                v_delta(2) = w_delta(4); 
                v_delta(3) = w_delta(5);

                // Apply operator gamma onto the first 2 * i columns of w_gamma,
                // and overwrite these columns with the results
                for (unsigned j = 0; j < next; ++j)
                {
                    Matrix<InternalType, 3, 1> u_gamma = gamma<InternalType>(this->rung_labels[i], w_gamma.col(j));
                    w_gamma(0, j) = v_alpha(0, j); 
                    w_gamma(1, j) = u_gamma(0); 
                    w_gamma(2, j) = u_gamma(1); 
                    w_gamma(3, j) = u_gamma(2); 
                }

                // Apply operator epsilon onto v_epsilon, populate the next 
                // 2 columns of w_gamma, and update v_epsilon
                Matrix<InternalType, 8, 1> w_epsilon = epsilon<InternalType>(this->rung_labels[i], v_epsilon);
                w_gamma(0, next) = v_alpha(0, next); 
                w_gamma(1, next) = w_epsilon(0); 
                w_gamma(2, next) = w_epsilon(1); 
                w_gamma(3, next) = w_epsilon(6); 
                w_gamma(0, after) = v_alpha(0, after); 
                w_gamma(1, after) = w_epsilon(2); 
                w_gamma(2, after) = w_epsilon(3); 
                w_gamma(3, after) = w_epsilon(7); 
                v_epsilon(0) = v_alpha(0, 0);
                v_epsilon(1) = w_epsilon(4); 
                v_epsilon(2) = w_epsilon(5);
            }

            // Compute the splitting probability of exiting the graph through
            // B{this->N}
            weight_A0 = v_alpha(0, 0); 
            InternalType weight_AN = v_beta(0); 
            InternalType weight_BN = v_beta(1);
            InternalType weight_A0_BN = v_gamma(2, 0);
            InternalType tmp0 = weight_A0 * _lower_exit_rate;
            InternalType tmp1 = _lower_exit_rate * _upper_exit_rate;
            InternalType tmp2 = weight_BN * _upper_exit_rate; 
            InternalType tmp3 = weight_A0_BN * tmp1; 
            InternalType _upper_exit_prob = tmp2 / (tmp0 + tmp2 + tmp3);

            // Compute the unconditional mean first-passage time to exiting
            // the graph through A0 (disregarding the given upper exit rate) 
            InternalType numer = 0;
            InternalType denom = 0;
            for (unsigned i = 0; i < this->N; ++i)
            {
                int index_Ai = 2 * i; 
                int index_Bi = 2 * i + 1;
                InternalType weight_Ai = v_alpha(0, index_Ai); 
                InternalType weight_Bi = v_alpha(0, index_Bi);
                InternalType weight_Ai_BN_with_path_A0_to_Ai = v_gamma(2, index_Ai); 
                InternalType weight_Bi_BN_with_path_A0_to_Bi = v_gamma(2, index_Bi);
                InternalType weight_A0_BN_with_path_Ai_to_A0 = w_gamma(2, index_Ai);
                InternalType weight_A0_BN_with_path_Bi_to_A0 = w_gamma(2, index_Bi);
                numer += (
                    (weight_Ai + weight_Ai_BN_with_path_A0_to_Ai * _upper_exit_rate) *
                    (tmp0 + weight_A0_BN_with_path_Ai_to_A0 * tmp1)
                );
                numer += (
                    (weight_Bi + weight_Bi_BN_with_path_A0_to_Bi * _upper_exit_rate) *
                    (tmp0 + weight_A0_BN_with_path_Bi_to_A0 * tmp1)
                );
            }
            InternalType weight_AN_BN_with_path_A0_to_AN = v_delta(3); 
            InternalType weight_A0_BN_with_path_AN_to_A0 = v_epsilon(2);
            numer += (
                (weight_AN + weight_AN_BN_with_path_A0_to_AN * _upper_exit_rate) * 
                (tmp0 + weight_A0_BN_with_path_AN_to_A0 * tmp1)
            );
            numer += (weight_BN * tmp0);
            denom = (tmp0 + tmp3) * (tmp0 + tmp2 + tmp3);

            return std::make_pair(
                static_cast<IOType>(_upper_exit_prob),
                static_cast<IOType>(denom / numer)
            ); 
        }
};

#endif 
