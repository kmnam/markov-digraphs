/** \file include/digraph.hpp
 *
 *  Main header file for MarkovDigraphs.
 *
 *  **Author:**
 *      Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 *  **Last updated:** 
 *      6/1/2023
 */

#ifndef LABELED_DIGRAPHS_HPP
#define LABELED_DIGRAPHS_HPP

#include <cmath>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/random.hpp>
#include <boost/random/exponential_distribution.hpp>
#include "math.hpp"

using namespace Eigen;

// ---------------------------------------------------------------------- //
//                            NODES AND EDGES                             //
// ---------------------------------------------------------------------- //

/**
 * A minimal struct that represents a vertex or node. 
 */
struct Node
{
    std::string id;       /**< String identifier. */

    /**
     * Trivial constructor. 
     */
    Node(std::string id)
    {
        this->id = id;
    }

    /**
     * Trivial destructor. 
     */
    ~Node()
    {
    }

    /**
     * Return the ID of the node.
     */
    std::string getId() const 
    {
        return this->id; 
    }

    /**
     * Set the ID of the node to the given string.
     */
    void setId(const std::string id)
    {
        this->id = id; 
    }

    /**
     * Equality operator. 
     */
    bool operator==(const Node& other) const
    {
        return (!this->id.compare(other.id));
    }

    /**
     * Inequality operator. 
     */
    bool operator!=(const Node& other) const
    {
        return (this->id.compare(other.id));
    }
};

/**
 * Edges are simply pairs of Node pointers. 
 */
using Edge = std::pair<Node*, Node*>;

// ---------------------------------------------------------------------- //
//                        THE LABELEDDIGRAPH CLASS                        //
// ---------------------------------------------------------------------- //

/**
 * An implementation of a labeled digraph.
 *
 * This class has two template parameters, `InternalType` and `IOType`:
 * - `InternalType` is the scalar type used to store all edge labels and
 *   perform mathematical calculations involving the row and column Laplacian
 *   matrices.
 * - `IOType` is the scalar type used for input values to some public methods
 *   (`addEdge()`, `setEdgeLabel()`, `getEdgeLabel()`, `getLaplacian()`, 
 *   `getSpanningForestMatrix()`, `getSpanningForestMatrixSparse()`.).
 *
 * These types can be finite-precision exact (e.g., integer types), finite-
 * precision floating-point, or arbitrary-precision.
 *
 * The public mathematical methods for computing steady-state probabilities 
 * or first-passage time moments are required to return a floating-point
 * scalar, and so these methods involve another template parameter that
 * specifies this type (see below).
 */
template <typename InternalType, typename IOType>
class LabeledDigraph
{
    protected:
        // ----------------------------------------------------- //
        //                       ATTRIBUTES                      //
        // ----------------------------------------------------- //
        /** Number of nodes in the graph. */ 
        unsigned numnodes = 0;

        /** Canonical ordering of nodes. */ 
        std::vector<Node*> order;

        /** Dictionary that maps `Node` IDs to `Node` pointers. */  
        std::unordered_map<std::string, Node*> nodes;

        /** Dictionary that maps outgoing edges from each `Node`, along with edge labels. */ 
        std::unordered_map<Node*, std::unordered_map<Node*, InternalType> > edges;
        
        // ----------------------------------------------------- //
        //                   PROTECTED METHODS                   //
        // ----------------------------------------------------- //

        /**
         * Return a pointer to the node with the given ID; return `nullptr` if 
         * a node with the given ID does not exist. 
         *
         * @param id ID of desired node. 
         * @returns  A pointer to node with the given ID (`nullptr` if such a node
         *           does not exist). 
         */
        Node* getNode(std::string id) const
        {
            auto it = this->nodes.find(id);
            if (it == this->nodes.end()) return nullptr;
            else                         return it->second; 
        }

        /**
         * Return the canonical ordering of nodes in the graph (`this->order`). 
         *
         * @returns A copy of `this->order`. 
         */
        std::vector<Node*> getAllNodes() const 
        {
            return this->order; 
        }

        /**
         * Return the edge between the specified nodes, along with the
         * edge label.
         *
         * This method returns a `nullptr` pair and zero for the edge label
         * if the edge does not exist but the nodes do, but throws an 
         * exception if either node does not exist.
         *
         * @param source_id ID of source node. 
         * @param target_id ID of target node. 
         * @returns         A pair containing the edge (as a `<Node*, Node*>`
         *                  pair) and the edge label.
         * @throws std::runtime_error if either node does not exist. 
         */
        std::pair<Edge, InternalType> getEdge(std::string source_id, std::string target_id)
        {
            Node* source = this->getNode(source_id);
            Node* target = this->getNode(target_id);

            // Check that both nodes exist 
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist; use LabeledDigraph<...>::addNode() to add node");
            if (target == nullptr)
                throw std::runtime_error("Specified target node does not exist; use LabeledDigraph<...>::addNode() to add node");

            // Check that the edge exists 
            auto it = this->edges[source].find(target);
            if (it != this->edges[source].end())
                return std::make_pair(std::make_pair(source, it->first), it->second);
            else    // Return pair of nullptrs and zero label if no edge exists
                return std::make_pair(std::make_pair(nullptr, nullptr), 0);
        }

        /**
         * Return a vector of outgoing edges from the given source node, given
         * the index of the source node in the canonical ordering (`this->order`). 
         *
         * @param source_idx Index of source node in the canonical ordering. 
         * @returns          `std::vector` of pairs containing the target node 
         *                   index and label of each edge
         * @throws std::runtime_error if the given node does not exist. 
         */
        std::vector<std::pair<int, InternalType> > getAllEdgesFromNode(int source_idx) 
        {
            // Check that the given node exists
            if (source_idx < 0 || source_idx > this->order.size() - 1)
                throw std::runtime_error("Specified source node does not exist");
            
            std::vector<std::pair<int, InternalType> > edges_from_node;
            Node* source = this->order[source_idx];

            // Run through all nodes in this->order ...
            for (int i = 0; i < this->numnodes; ++i)
            {
                // Is there an edge to this node?
                Node* target = this->order[i]; 
                if (this->edges[source].find(target) != this->edges[source].end())
                {
                    // If so, instantiate the edge and get the label 
                    InternalType label = this->edges[source][target];
                    edges_from_node.push_back(std::make_pair(i, label)); 
                }
            }

            return edges_from_node; 
        }

        /**
         * Return a vector of outgoing edges from the given source node, given
         * a pointer to the source node. 
         *
         * @param source Pointer to source node. 
         * @returns      `std::vector` of pairs containing each edge (as a 
         *               `<Node*, Node*>` pair) and edge label
         * @throws std::runtime_error if the given node does not exist. 
         */
        std::vector<std::pair<Edge, InternalType> > getAllEdgesFromNode(Node* source) 
        {
            std::vector<std::pair<Edge, InternalType> > edges_from_node;

            // Check that the given node exists
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist; use LabeledDigraph<...>::addNode() to add node");

            // Run through all nodes in this->order ...
            for (auto&& node : this->order)
            {
                // Is there an edge to this node?
                if (this->edges[source].find(node) != this->edges[source].end())
                {
                    // If so, instantiate the edge and get the label 
                    Edge edge = std::make_pair(source, node);
                    InternalType label = this->edges[source][node];
                    edges_from_node.push_back(std::make_pair(edge, label)); 
                }
            }

            return edges_from_node; 
        }

        /**
         * Return true if the specified edge exists, given pointers to the 
         * two nodes, and false otherwise.
         *
         * This method also returns false if either node does not exist.  
         *
         * @param source Pointer to source node. 
         * @param target Pointer to target node. 
         * @returns      true if the edge exists, false otherwise.  
         */
        bool hasEdge(Node* source, Node* target) const 
        {
            return (this->edges.count(source) && (this->edges.find(source))->second.count(target)); 
        }

        /**
         * Compute the k-th spanning forest matrix, using the recurrence
         * of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs.\ 17-18), with 
         * a *dense* Laplacian matrix. 
         *
         * A private version of the corresponding public method, in which 
         * a pre-computed Laplacian matrix is provided as an argument.
         * 
         * @param k         Index of desired spanning forest matrix.  
         * @param laplacian Input Laplacian matrix.
         * @returns         k-th spanning forest matrix. 
         */
        Matrix<InternalType, Dynamic, Dynamic> getSpanningForestMatrix(const int k,
                                                                       const Ref<const Matrix<InternalType, Dynamic, Dynamic> >& laplacian)
        {
            // Begin with the identity matrix 
            Matrix<InternalType, Dynamic, Dynamic> curr
                = Matrix<InternalType, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes);

            // Apply the recurrence ...
            for (unsigned i = 0; i < k; ++i)
                curr = chebotarevAgaevRecurrence<InternalType>(i, laplacian, curr);

            return curr; 
        }

        /**
         * Compute the k-th spanning forest matrix, using the recurrence
         * of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs.\ 17-18), with 
         * a *compressed sparse row-major* Laplacian matrix. 
         *
         * A private version of the corresponding public method, in which 
         * a pre-computed Laplacian matrix is provided as an argument.
         * 
         * @param k         Index of desired spanning forest matrix.  
         * @param laplacian Input Laplacian matrix (compressed sparse row-major).
         * @returns         k-th spanning forest matrix. 
         */
        Matrix<InternalType, Dynamic, Dynamic> getSpanningForestMatrix(const int k,
                                                                       const SparseMatrix<InternalType, RowMajor>& laplacian)
        {
            // Begin with the identity matrix
            Matrix<InternalType, Dynamic, Dynamic> curr
                = Matrix<InternalType, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes); 

            // Apply the recurrence ...
            for (unsigned i = 0; i < k; ++i)
                curr = chebotarevAgaevRecurrence<InternalType>(i, laplacian, curr); 

            return curr; 
        }

        /**
         * Return the *column Laplacian* matrix, with the nodes ordered
         * according to the graph's canonical ordering of nodes, as a *dense*
         * matrix.
         *
         * @returns      Laplacian matrix of the graph (as a dense matrix). 
         */
        Matrix<InternalType, Dynamic, Dynamic> getColumnLaplacianDense()
        {
            // Initialize a zero matrix with #rows = #cols = #nodes
            Matrix<InternalType, Dynamic, Dynamic> laplacian
                = Matrix<InternalType, Dynamic, Dynamic>::Zero(this->numnodes, this->numnodes);

            // Populate the off-diagonal entries of the matrix first: 
            // (i,j)-th entry is the label of the edge j -> i
            unsigned i = 0;
            for (auto&& v : this->order)
            {
                unsigned j = 0;
                for (auto&& w : this->order)
                {
                    if (i != j)
                    {
                        // Get the edge label for j -> i
                        if (this->edges[w].find(v) != this->edges[w].end())
                        {
                            InternalType label = this->edges[w][v];
                            laplacian(i,j) = label;
                            if (laplacian(i,j) < 0)
                                throw std::runtime_error("Negative edge label found");
                        }
                    }
                    j++;
                }
                i++;
            }

            // Populate diagonal entries as negative sums of the off-diagonal
            // entries in each column
            for (unsigned i = 0; i < this->numnodes; ++i)
                laplacian(i, i) = -(laplacian.col(i).sum()); 

            return laplacian;
        }

        /**
         * Return the *row Laplacian* matrix, according to the graph's canonical
         * ordering of nodes, as a *compressed row-major sparse* matrix. 
         *
         * @returns      Laplacian matrix of the graph (as a sparse row-major
         *               matrix). 
         */
        SparseMatrix<InternalType, RowMajor> getRowLaplacianSparse()
        {
            // Initialize a zero matrix with #rows = #cols = #nodes
            SparseMatrix<InternalType, RowMajor> laplacian(this->numnodes, this->numnodes); 

            // Populate the entries of the matrix: the off-diagonal (i,j)-th  
            // entry is the *negative* of the label of the edge i -> j, and 
            // the diagonal entries are set so that each *row* sum is zero 
            std::vector<Triplet<InternalType> > laplacian_triplets; 
            unsigned i = 0;
            for (auto&& v : this->order)
            {
                std::vector<InternalType> row_entries;    // All nonzero off-diagonal entries in i-th row
                unsigned j = 0;
                for (auto&& w : this->order)
                {
                    if (i != j)
                    {
                        // Get the edge label for i -> j 
                        if (this->edges[v].find(w) != this->edges[v].end())
                        {
                            InternalType label = this->edges[v][w];
                            laplacian_triplets.push_back(Triplet<InternalType>(i, j, -label));
                            row_entries.push_back(label);  
                        }
                    }
                    j++;
                }
                // Compute the negative of the i-th row sum
                InternalType row_sum = 0;
                for (const InternalType entry : row_entries)
                    row_sum += entry; 
                laplacian_triplets.push_back(Triplet<InternalType>(i, i, row_sum)); 
                i++;
            }
            laplacian.setFromTriplets(laplacian_triplets.begin(), laplacian_triplets.end());

            return laplacian; 
        }

        /**
         * Return the *row "sub-Laplacian"* matrix obtained by removing the 
         * edges outgoing from the given "target" node, according to the graph's
         * canonical ordering of nodes, as a *dense* matrix.
         *
         * @param target Pointer to node whose outgoing edges are to be removed. 
         * @returns      Laplacian matrix of the graph (as a dense matrix).  
         */
        Matrix<InternalType, Dynamic, Dynamic> getRowSublaplacianDense(Node* target)
        {
            // Initialize a zero matrix with #rows = #cols = #nodes
            Matrix<InternalType, Dynamic, Dynamic> laplacian
                = Matrix<InternalType, Dynamic, Dynamic>::Zero(this->numnodes, this->numnodes);

            // Populate the off-diagonal entries of the matrix first: 
            // (i,j)-th entry is the *negative* of the label of the edge i -> j
            unsigned i = 0;
            for (auto&& v : this->order)
            {
                unsigned j = 0;
                for (auto&& w : this->order)
                {
                    if (i != j)
                    {
                        // Get the edge label for i -> j, omitting all edges
                        // for which i is the given target node 
                        if (v != target && this->edges[v].find(w) != this->edges[v].end())
                        {
                            InternalType label = this->edges[v][w];
                            laplacian(i, j) = -label;
                        }
                    }
                    j++;
                }
                i++;
            }

            // Populate diagonal entries as negative sums of the off-diagonal
            // entries in each row
            for (unsigned i = 0; i < this->numnodes; ++i)
                laplacian(i, i) = -(laplacian.row(i).sum()); 

            return laplacian; 
        }

        /**
         * Return the *row "sub-Laplacian"* matrix obtained by removing the 
         * edges outgoing from the given "target" node, according to the graph's
         * canonical ordering of nodes, as a *compressed row-major sparse* matrix.
         *
         * @param target Pointer to node whose outgoing edges are to be removed. 
         * @returns      Laplacian matrix of the graph (as a compressed row-major
         *               sparse matrix). 
         */
        SparseMatrix<InternalType, RowMajor> getRowSublaplacianSparse(Node* target)
        {
            // Initialize a zero matrix with #rows = #cols = #nodes
            SparseMatrix<InternalType, RowMajor> laplacian(this->numnodes, this->numnodes); 

            // Populate the entries of the matrix: the off-diagonal (i,j)-th  
            // entry is the *negative* of the label of the edge i -> j, and 
            // the diagonal entries are set so that each *row* sum is zero 
            std::vector<Triplet<InternalType> > laplacian_triplets; 
            unsigned i = 0;
            for (auto&& v : this->order)
            {
                std::vector<InternalType> row_entries;    // All nonzero off-diagonal entries in i-th row
                unsigned j = 0;
                for (auto&& w : this->order)
                {
                    if (i != j)
                    {
                        // Get the edge label for i -> j, omitting all edges 
                        // for which i is the given target node 
                        if (v != target && this->edges[v].find(w) != this->edges[v].end())
                        {
                            InternalType label = this->edges[v][w]; 
                            laplacian_triplets.push_back(Triplet<InternalType>(i, j, -label));
                            row_entries.push_back(label);  
                        }
                    }
                    j++;
                }
                // Compute the negative of the i-th row sum
                InternalType row_sum = 0;
                for (const InternalType entry : row_entries)
                    row_sum += entry;
                laplacian_triplets.push_back(Triplet<InternalType>(i, i, row_sum)); 
                i++;
            }
            laplacian.setFromTriplets(laplacian_triplets.begin(), laplacian_triplets.end());

            return laplacian; 
        }

    public:
        /**
         * Empty constructor. 
         */
        LabeledDigraph()
        {
        }

        /**
         * Destructor; de-allocates each node from heap memory.
         */
        ~LabeledDigraph()
        {
            for (auto&& edge_set : this->edges) delete edge_set.first;
        }

        /**
         * Return the number of nodes in the graph.
         *
         * @returns Number of nodes in the graph.  
         */
        unsigned getNumNodes() const 
        {
            return this->numnodes; 
        }

        // ----------------------------------------------------- //
        //         NODE/EDGE/LABEL-ADDING/SETTING METHODS        //
        // ----------------------------------------------------- //
        
        /**
         * Add a node to the graph with the given ID.
         *
         * @param id ID for new node. 
         * @throws std::runtime_error if node already exists with the given ID. 
         */
        void addNode(std::string id)
        {
            // Check that a node with the given id doesn't exist already
            if (this->nodes.find(id) != this->nodes.end())
                throw std::runtime_error("Node already exists"); 

            Node* node = new Node(id);
            this->nodes[id] = node;
            this->edges.emplace(node, std::unordered_map<Node*, InternalType>());
            this->numnodes++;
            this->order.push_back(node);
        }

        /**
         * Remove a node from the graph with the given ID.
         *
         * @param id ID of node to be removed. 
         * @throws std::runtime_error if node with given ID does not exist.  
         */
        void removeNode(std::string id)
        {
            // Check that a node with the given id exists
            Node* node = this->getNode(id);
            if (node == nullptr)
                throw std::runtime_error("Specified node does not exist; use LabeledDigraph<...>::addNode() to add node");

            // Find and delete node in this->order
            for (auto it = this->order.begin(); it != this->order.end(); ++it)
            {
                if (*it == node)
                {
                    this->order.erase(it);
                    break;
                }
            }

            // Run through and delete all edges with given node as source
            this->edges[node].clear();

            // Run through and delete all edges with given node as target 
            for (auto&& v : this->edges)
            {
                // v.first of type Node*, v.second of type std::unordered_map<Node*, InternalType>
                for (auto it = v.second.begin(); it != v.second.end(); ++it)
                {
                    if (it->first == node)
                    {
                        (v.second).erase(it);
                        break;
                    }
                } 
            }

            // Erase given node from this->edges 
            for (auto it = this->edges.begin(); it != this->edges.end(); ++it)
            {
                if (it->first == node)
                {
                    this->edges.erase(it);
                    break;
                }
            }
           
            // Delete the heap-allocated Node itself 
            delete node;
            this->nodes.erase(id);
            this->numnodes--;
        }

        /**
         * Return true if node with given ID exists in the graph.
         *
         * @param id ID of desired node. 
         * @returns  true if node exists with the given ID, false otherwise.  
         */
        bool hasNode(std::string id) const 
        {
            return this->nodes.count(id);
        }

        /**
         * Return a vector of the IDs of the nodes, ordered according to the 
         * canonical ordering (`this->order`). 
         *
         * @returns A vector of all node IDs, ordered according to `this->order`.
         */
        std::vector<std::string> getAllNodeIds() const 
        {
            std::vector<std::string> node_ids; 
            for (auto&& node : this->order)
                node_ids.push_back(node->id);

            return node_ids; 
        }

        /**
         * Add an edge between two nodes.
         *
         * If either ID does not correspond to a node in the graph, this function
         * instantiates these nodes. 
         *
         * @param source_id ID of source node of new edge. 
         * @param target_id ID of target node of new edge. 
         * @param label     Label on new edge.
         * @throws std::runtime_error if the edge already exists. 
         */
        void addEdge(std::string source_id, std::string target_id, IOType label = 1)
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
            {
                this->addNode(source_id);
                source = this->nodes[source_id]; 
            }
            if (target == nullptr)
            {
                this->addNode(target_id); 
                target = this->nodes[target_id]; 
            }

            // Then define the edge
            this->edges[source][target] = static_cast<InternalType>(label);
        }

        /**
         * Remove the edge between the two given nodes.
         *
         * This method does nothing if the edge does not exist but the nodes 
         * do, but throws an exception if either node does not exist. 
         *
         * @param source_id ID of source node of edge to be removed. 
         * @param target_id ID of target node of edge to be removed.
         * @throws std::runtime_error if either node does not exist.  
         */
        void removeEdge(std::string source_id, std::string target_id)
        {
            Node* source = this->getNode(source_id);
            Node* target = this->getNode(target_id);

            // Check that both nodes exist 
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist; use LabeledDigraph<...>::addNode() to add node");
            if (target == nullptr)
                throw std::runtime_error("Specified target node does not exist; use LabeledDigraph<...>::addNode() to add node");

            // Does the given edge exist? 
            if (this->edges[source].find(target) != this->edges[source].end())
            {
                // If so, then get rid of it 
                this->edges[source].erase(this->edges[source].find(target));
            }
            // If not, then do nothing!
        }

        /**
         * Return true if the specified edge exists, given the IDs of the 
         * two nodes, and false otherwise.
         *
         * This method also returns false if either node does not exist.  
         *
         * @param source_id ID of source node. 
         * @param target_id ID of target node. 
         * @returns         true if the edge exists, false otherwise.  
         */
        bool hasEdge(std::string source_id, std::string target_id) const
        {
            // Check that the two nodes exist 
            if (this->nodes.count(source_id) && this->nodes.count(target_id))
            {
                // Look up the edge to see if it exists 
                Node* source = this->nodes.find(source_id)->second;
                Node* target = this->nodes.find(target_id)->second;
                return (this->edges.count(source) && (this->edges.find(source))->second.count(target));
            }
            return false;
        }

        /**
         * Return a vector of nodes to which there are edges outgoing from the 
         * given source node, along with the corresponding edge label, given 
         * the ID of the source node. 
         *
         * @param source_id ID of source node. 
         * @returns         `std::vector` of `<std::string, IOType>` pairs
         *                  containing each target node and corresponding 
         *                  edge label
         * @throws std::runtime_error if a node with the given ID does not exist. 
         */
        std::vector<std::pair<std::string, IOType> > getAllEdgesFromNode(std::string source_id) 
        {
            std::vector<std::pair<std::string, IOType> > edges_from_node;
            Node* source = this->getNode(source_id);

            // Check that the given node exists
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist; use LabeledDigraph<...>::addNode() to add node");

            // Run through all nodes in this->order ...
            for (auto&& node : this->order)
            {
                // Is there an edge to this node?
                if (this->edges[source].find(node) != this->edges[source].end())
                {
                    // If so, instantiate the edge and get the label 
                    std::string target_id = node->id; 
                    IOType label = static_cast<IOType>(this->edges[source][node]);
                    edges_from_node.push_back(std::make_pair(target_id, label)); 
                }
            }

            return edges_from_node; 
        }

        /**
         * Get the label on the specified edge.
         *
         * This method throws an exception if either node does not exist, and 
         * also if the specified edge does not exist. 
         *
         * @param source_id ID of source node. 
         * @param target_id ID of target node. 
         * @returns         Edge label. 
         * @throws std::runtime_error if either node or the edge does not exist.  
         */
        IOType getEdgeLabel(std::string source_id, std::string target_id)
        {
            Node* source = this->getNode(source_id); 
            Node* target = this->getNode(target_id);

            // Check that both nodes exist 
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist");
            if (target == nullptr)
                throw std::runtime_error("Specified target node does not exist");

            // If the edge exists, return the edge label; otherwise, throw 
            // std::runtime_error
            auto it = this->edges[source].find(target);
            if (it != this->edges[source].end())
                return static_cast<IOType>(this->edges[source][target]);
            else 
                throw std::runtime_error("Specified edge does not exist");
        }

        /**
         * Set the label on the specified edge to the given value. 
         *
         * This method throws an exception if either node does not exist, and 
         * also if the specified edge does not exist. 
         *
         * @param source_id ID of source node. 
         * @param target_id ID of target node. 
         * @param value     New edge label. 
         * @throws std::runtime_error if either node or the edge does not exist.
         */
        void setEdgeLabel(std::string source_id, std::string target_id, IOType value)
        {
            Node* source = this->getNode(source_id);
            Node* target = this->getNode(target_id);

            // Check that both nodes exist 
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist");
            if (target == nullptr)
                throw std::runtime_error("Specified target node does not exist");

            // If the edge exists, change the edge label; otherwise, throw 
            // std::runtime_error
            auto it = this->edges[source].find(target);
            if (it != this->edges[source].end())
                this->edges[source][target] = static_cast<InternalType>(value);
            else 
                throw std::runtime_error("Specified edge does not exist");
        }

        // ----------------------------------------------------- //
        //                     OTHER METHODS                     //
        // ----------------------------------------------------- //
        
        /**
         * Return a pointer to a new (dynamically allocated) subgraph induced
         * by the given subset of nodes.
         *
         * An exception is thrown if a node in the given subset does not exist. 
         *
         * @param nodes An unordered set of pointers to nodes in the graph. 
         * @returns     A pointer to the induced subgraph (instantiated as a 
         *              new `LabeledDigraph` object). 
         * @throws std::runtime_error if the given subset contains a node that
         *                            does not exist in the graph.  
         */
        LabeledDigraph<InternalType, IOType>* subgraph(const std::unordered_set<Node*>& nodes)
        {
            LabeledDigraph<InternalType, IOType>* subgraph = new LabeledDigraph<InternalType, IOType>();

            // For each node in the subset ...
            for (auto&& v : nodes)
            {
                // Check that the node exists in the larger graph
                auto it = this->edges.find(v);
                if (it == this->edges.end())
                    throw std::runtime_error("Specified subset contains non-existent node");

                // Try adding the node to the subgraph (if already added as 
                // target of an edge, will throw std::runtime_error)
                try
                {
                    subgraph->addNode(v->id); 
                }
                catch (const std::runtime_error& e) { }

                // Find all edges between pairs of nodes in the subset
                for (auto&& edge : this->edges[v])
                {
                    Node* w = edge.first;                // Target node 
                    InternalType label = edge.second;    // Edge label  

                    // If the target node is also in the subset ...
                    if (nodes.find(w) != nodes.end())
                    {
                        // ... and the target node has *not* been added to the subgraph, 
                        // add it now 
                        if (!subgraph->hasNode(w->id))
                            subgraph->addNode(w->id); 

                        // Then add each edge directly to subgraph->edges to avoid
                        // possible loss of precision from InternalType to IOType
                        // and back 
                        subgraph->edges[v][w] = label; 
                    }
                }
            }

            return subgraph; 
        }

        /**
         * Clears the graph's contents.  
         */        
        void clear()
        {
            // De-allocate all Nodes from heap memory
            for (auto&& edge_set : this->edges) delete edge_set.first;

            // Clear all attributes
            this->numnodes = 0; 
            this->nodes.clear();
            this->edges.clear();
            this->order.clear(); 
        }

        /**
         * Return pointer to a new (dynamically allocated) copy of the graph, 
         * possibly with different scalar types.
         *
         * @returns A pointer to a new copy of the graph.
         */
        template <typename NewInternalType = InternalType, typename NewIOType = IOType>
        LabeledDigraph<NewInternalType, NewIOType>* copy() const
        {
            LabeledDigraph<NewInternalType, NewIOType>* graph = new LabeledDigraph<NewInternalType, NewIOType>();

            // Copy over nodes with the same ids
            for (auto&& node : this->order)
                graph->addNode(node->id);
            
            // Copy over edges and edge labels
            for (auto&& edge_set : this->edges)
            {
                for (auto&& dest : edge_set.second)
                {
                    InternalType label = dest.second;

                    // Add the edge with the edge label cast into the copy's IO scalar type 
                    graph->addEdge(edge_set.first->id, dest.first->id, static_cast<NewIOType>(label));
                }
            }

            return graph;
        }

        /**
         * Copy over the contents of this graph to another (dynamically
         * allocated) graph, possibly with different scalar types.
         *
         * @param graph A pointer to another graph instance. 
         */
        template <typename NewInternalType = InternalType, typename NewIOType = IOType>
        void copy(LabeledDigraph<NewInternalType, NewIOType>* graph) const
        {
            // Clear the input graph's contents
            graph->clear();

            // Copy over nodes with the same IDs
            for (auto&& node : this->order)
                graph->addNode(node->id);

            // Copy over edges and edge labels
            for (auto&& edge_set : this->edges)
            {
                for (auto&& dest : edge_set.second)
                {
                    InternalType label = dest.second;
                    graph->addEdge(edge_set.first->id, dest.first->id, static_cast<NewIOType>(label));
                }
            }
        }

        /**
         * Return the Laplacian matrix, with the nodes ordered according to
         * the graph's canonical ordering of nodes.
         *
         * @returns      Laplacian matrix of the graph (as a dense matrix). 
         */
        Matrix<IOType, Dynamic, Dynamic> getLaplacian()
        {
            // Initialize a zero matrix with #rows = #cols = #nodes
            Matrix<InternalType, Dynamic, Dynamic> laplacian
                = Matrix<InternalType, Dynamic, Dynamic>::Zero(this->numnodes, this->numnodes);

            // Populate the off-diagonal entries of the matrix first: 
            // (i,j)-th entry is the label of the edge j -> i
            unsigned i = 0;
            for (auto&& v : this->order)
            {
                unsigned j = 0;
                for (auto&& w : this->order)
                {
                    if (i != j)
                    {
                        // Get the edge label for j -> i
                        if (this->edges[w].find(v) != this->edges[w].end())
                        {
                            InternalType label = this->edges[w][v];
                            laplacian(i,j) = label;
                            if (laplacian(i,j) < 0)
                                throw std::runtime_error("Negative edge label found");
                        }
                    }
                    j++;
                }
                i++;
            }

            // Populate diagonal entries as negative sums of the off-diagonal
            // entries in each column
            for (unsigned i = 0; i < this->numnodes; ++i)
                laplacian(i, i) = -(laplacian.col(i).sum()); 

            return laplacian.template cast<IOType>();
        }

        /**
         * Compute the k-th spanning forest matrix, using the recurrence
         * of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs.\ 17-18), with
         * a *dense* Laplacian matrix.
         *
         * @param k      Index of the desired spanning forest matrix. 
         * @returns      k-th spanning forest matrix.
         */
        Matrix<IOType, Dynamic, Dynamic> getSpanningForestMatrix(const int k)
        {
            // Begin with the identity matrix 
            Matrix<InternalType, Dynamic, Dynamic> curr
                = Matrix<InternalType, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes);

            // Initialize a zero matrix with #rows = #cols = #nodes
            Matrix<InternalType, Dynamic, Dynamic> laplacian
                = Matrix<InternalType, Dynamic, Dynamic>::Zero(this->numnodes, this->numnodes);

            // Populate the off-diagonal entries of the matrix first: 
            // (i,j)-th entry is the *negative* of the label of the edge i -> j
            unsigned i = 0;
            for (auto&& v : this->order)
            {
                unsigned j = 0;
                for (auto&& w : this->order)
                {
                    if (i != j)
                    {
                        // Get the edge label for i -> j
                        if (this->edges[v].find(w) != this->edges[v].end())
                        {
                            InternalType label = this->edges[v][w];
                            laplacian(i, j) = -label;
                        }
                    }
                    j++;
                }
                i++;
            }

            // Populate diagonal entries as negative sums of the off-diagonal
            // entries in each row
            for (unsigned i = 0; i < this->numnodes; ++i)
                laplacian(i, i) = -(laplacian.row(i).sum()); 

            // Apply the recurrence ...
            for (unsigned i = 0; i < k; ++i)
                curr = chebotarevAgaevRecurrence<InternalType>(i, laplacian, curr);

            return curr.template cast<IOType>();
        }

        /**
         * Compute the k-th spanning forest matrix, using the recurrence
         * of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs.\ 17-18), with
         * a *compressed row-major sparse* Laplacian matrix.
         *
         * @param k      Index of the desired spanning forest matrix. 
         * @returns      k-th spanning forest matrix.
         * @throws std::invalid_argument if summation method is not recognized.
         */
        Matrix<IOType, Dynamic, Dynamic> getSpanningForestMatrixSparse(const int k)
        {
            // Begin with the identity matrix
            Matrix<InternalType, Dynamic, Dynamic> curr
                = Matrix<InternalType, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes); 

            // Initialize a zero matrix with #rows = #cols = #nodes
            SparseMatrix<InternalType, RowMajor> laplacian(this->numnodes, this->numnodes); 

            // Populate the entries of the matrix: the off-diagonal (i,j)-th  
            // entry is the *negative* of the label of the edge i -> j, and 
            // the diagonal entries are set so that each *row* sum is zero 
            std::vector<Triplet<InternalType> > laplacian_triplets; 
            unsigned i = 0;
            for (auto&& v : this->order)
            {
                std::vector<InternalType> row_entries;    // All nonzero off-diagonal entries in i-th row
                unsigned j = 0;
                for (auto&& w : this->order)
                {
                    if (i != j)
                    {
                        // Get the edge label for i -> j
                        if (this->edges[v].find(w) != this->edges[v].end())
                        {
                            InternalType label = this->edges[v][w];
                            laplacian_triplets.push_back(Triplet<InternalType>(i, j, -label));
                            row_entries.push_back(label);  
                        }
                    }
                    j++;
                }

                // Compute the negative of the i-th row sum
                InternalType row_sum = 0;
                for (const InternalType entry : row_entries)
                    row_sum += entry; 
                laplacian_triplets.push_back(Triplet<InternalType>(i, i, row_sum)); 
                i++;
            }
            laplacian.setFromTriplets(laplacian_triplets.begin(), laplacian_triplets.end());

            // Apply the recurrence ...
            for (unsigned i = 0; i < k; ++i)
                curr = chebotarevAgaevRecurrence<InternalType>(i, laplacian, curr); 

            return curr.template cast<IOType>(); 
        }

        /**
         * Compute a vector in the kernel of the Laplacian matrix of the 
         * graph, normalized by its 1-norm, by singular value decomposition 
         * of the Laplacian matrix.
         *
         * This vector coincides with the vector of steady-state probabilities 
         * of the nodes in the Markov process associated with the graph.
         *
         * This method *assumes* that this graph is strongly connected, in which 
         * case the Laplacian matrix has a one-dimensional kernel and so the 
         * returned vector serves as a basis for this kernel.
         *
         * Since this method necessarily returns a vector with floating-point
         * quantities, which may not be accommodated by `IOType`, this method 
         * allows the specification of a new output type, `FloatIOType`.
         * Moreover, since Eigen's SVD module requires a floating-point 
         * scalar type, this method also allows the specification of a new 
         * internal type, `FloatInternalType`.
         *
         * @returns Vector in the kernel of the graph's Laplacian matrix, 
         *          normalized by its 1-norm.
         */
        template <typename FloatInternalType = InternalType, typename FloatIOType = IOType>
        Matrix<FloatIOType, Dynamic, 1> getSteadyStateFromSVD()
        {
            Matrix<FloatInternalType, Dynamic, Dynamic> laplacian = this->getColumnLaplacianDense().template cast<FloatInternalType>();
            
            // Obtain the steady-state vector of the Laplacian matrix
            Matrix<FloatInternalType, Dynamic, 1> steady_state;
            try
            {
                steady_state = getOneDimNullspaceFromSVD<FloatInternalType>(laplacian);
            }
            catch (const std::runtime_error& e)
            {
                throw;
            }

            // Normalize by the sum of the entries and return 
            FloatInternalType norm = steady_state.sum();
            return (steady_state / norm).template cast<FloatIOType>();
        }

        /**
         * Compute a vector in the kernel of the Laplacian matrix of the 
         * graph, normalized by its 1-norm, by the recurrence of Chebotarev
         * and Agaev (Lin Alg Appl, 2002, Eq.\ 19).
         *
         * This vector coincides with the vector of steady-state probabilities 
         * of the nodes in the Markov process associated with the graph.
         *
         * This method *assumes* that this graph is strongly connected, in which 
         * case the Laplacian matrix has a one-dimensional kernel and so the 
         * returned vector serves as a basis for this kernel.
         *
         * Since this method necessarily returns a vector with floating-point
         * quantities, which may not be accommodated by `IOType`, this method 
         * allows the specification of a new output type, `FloatIOType`. 
         *
         * @param sparse If true, use a sparse Laplacian matrix in the calculations. 
         * @returns      Vector in the kernel of the graph's Laplacian matrix, 
         *               normalized by its 1-norm.
         */
        template <typename FloatIOType = IOType>
        Matrix<FloatIOType, Dynamic, 1> getSteadyStateFromRecurrence(const bool sparse)
        {
            // Obtain the spanning tree weight matrix from the row Laplacian matrix
            Matrix<InternalType, Dynamic, Dynamic> tree_matrix;
            if (sparse)
            {
                SparseMatrix<InternalType, RowMajor> laplacian = this->getRowLaplacianSparse(); 
                tree_matrix = this->getSpanningForestMatrix(this->numnodes - 1, laplacian);  
            } 
            else
            {
                Matrix<InternalType, Dynamic, Dynamic> laplacian = -this->getColumnLaplacianDense().transpose(); 
                tree_matrix = this->getSpanningForestMatrix(this->numnodes - 1, laplacian);  
            } 

            // Return any row of the matrix, normalized by the sum of its entries
            return tree_matrix.row(0).template cast<FloatIOType>() / static_cast<FloatIOType>(tree_matrix.row(0).sum());
        }

        /**
         * Compute the vector of *unconditional* mean first-passage times in 
         * the Markov process associated with the graph from each node to the 
         * target node, using the given linear solver method.
         *
         * This method assumes that the associated Markov process certainly 
         * eventually reaches the target node from each node in the graph,
         * meaning that there are no alternative terminal nodes (or rather
         * SCCs) to which the process can travel and get "stuck".
         *
         * Since this method necessarily returns a vector with floating-point
         * quantities, which may not be accommodated by `IOType`, this method 
         * allows the specification of a new output type, `FloatIOType`.
         * Moreover, since Eigen's QR/LU solvers requires a floating-point 
         * scalar type, this method also allows the specification of a new 
         * internal type, `FloatInternalType`.
         *
         * @param target_id ID of target node. 
         * @param method    Linear solver method for computing the mean first-passage
         *                  time vector. 
         * @returns Vector of mean first-passage times to the target node from
         *          every node in the graph.
         * @throws std::invalid_argument if solver method is not recognized. 
         * @throws std::runtime_error    if target node does not exist.
         */
        template <typename FloatInternalType = InternalType, typename FloatIOType = IOType>
        Matrix<FloatIOType, Dynamic, 1> getMeanFirstPassageTimesFromSolver(std::string target_id,
                                                                           const SolverMethod method = QRDecomposition) 
        {
            // Get the index of the target node
            Node* target = this->getNode(target_id);
            if (target == nullptr)
            {
                std::cout << "here?\n"; 
                throw std::runtime_error("Specified source node does not exist; use LabeledDigraph<...>::addNode() to add node");
            }
            int t = 0;  
            for (auto it = this->order.begin(); it != this->order.end(); ++it)
            {
                if (*it == target)
                {
                    t = std::distance(this->order.begin(), it);
                    break;
                }
            } 

            // Get the Laplacian matrix of the graph and drop the row and column
            // corresponding to the target node 
            Matrix<FloatInternalType, Dynamic, Dynamic> laplacian = this->getColumnLaplacianDense().template cast<FloatInternalType>();
            Matrix<FloatInternalType, Dynamic, Dynamic> sublaplacian(this->numnodes - 1, this->numnodes - 1);
            int z = this->numnodes - 1 - t;
            if (t == 0)
            {
                sublaplacian = laplacian.block(1, 1, z, z);
            }
            else if (z == 0)
            {
                sublaplacian = laplacian.block(0, 0, t, t); 
            }
            else 
            { 
                sublaplacian.block(0, 0, t, t) = laplacian.block(0, 0, t, t);
                sublaplacian.block(0, t, t, z) = laplacian.block(0, t + 1, t, z); 
                sublaplacian.block(t, 0, z, t) = laplacian.block(t + 1, 0, z, t); 
                sublaplacian.block(t, t, z, z) = laplacian.block(t + 1, t + 1, z, z);
            } 

            // Get the left-hand matrix in the first-passage time linear system
            Matrix<FloatInternalType, Dynamic, Dynamic> A = sublaplacian.transpose() * sublaplacian.transpose();

            // Get the right-hand vector in the first-passage time linear system 
            Matrix<FloatInternalType, Dynamic, 1> b = Matrix<FloatInternalType, Dynamic, 1>::Zero(this->numnodes - 1); 
            for (unsigned i = 0; i < t; ++i)
            {
                if (this->hasEdge(this->order[i], target))
                    b(i) = this->edges[this->order[i]][target];
            }
            for (unsigned i = t + 1; i < this->numnodes; ++i)
            {
                if (this->hasEdge(this->order[i], target))
                    b(i - 1) = this->edges[this->order[i]][target]; 
            }

            // Solve the linear system with the specified method 
            Matrix<FloatInternalType, Dynamic, 1> solution;
            if (method == QRDecomposition)
            {
                solution = solveByQRD<FloatInternalType>(A, b);
            }
            else if (method == LUDecomposition)
            {
                solution = solveByLUD<FloatInternalType>(A, b);
            }
            else 
            {
                std::stringstream ss; 
                ss << "Invalid linear system solution method specified: " << method; 
                throw std::invalid_argument(ss.str());
            } 

            // Return an augmented vector with the zero mean FPT from the 
            // target node to itself
            Matrix<FloatInternalType, Dynamic, 1> fpt_vec = Matrix<FloatInternalType, Dynamic, 1>::Zero(this->numnodes);
            for (unsigned i = 0; i < t; ++i)
                fpt_vec(i) = solution(i); 
            for (unsigned i = t + 1; i < this->numnodes; ++i)
                fpt_vec(i) = solution(i - 1);

            return fpt_vec.template cast<FloatIOType>(); 
        }

        /**
         * Compute the vector of *unconditional* mean first-passage times in 
         * the Markov process associated with the graph from each node to the 
         * target node, using the recurrence of Chebotarev and Agaev (Lin Alg
         * Appl, 2002, Eqs.\ 17-18).
         *
         * This method assumes that the associated Markov process certainly 
         * eventually reaches the target node from each node in the graph,
         * meaning that there are no alternative terminal nodes (or rather
         * SCCs) to which the process can travel and get "stuck".   
         *
         * Since this method necessarily returns a vector with floating-point
         * quantities, which may not be accommodated by `IOType`, this method 
         * allows the specification of a new output type, `FloatIOType`. 
         *
         * @param target_id ID of target node.
         * @param sparse    If true, use a sparse Laplacian matrix in the calculations. 
         * @returns Vector of mean first-passage times to the target node from
         *          every node in the graph.
         * @throws std::runtime_error    if target node does not exist. 
         */
        template <typename FloatIOType = IOType>
        Matrix<FloatIOType, Dynamic, 1> getMeanFirstPassageTimesFromRecurrence(std::string target_id,
                                                                               const bool sparse)
        {
            Node* target = this->getNode(target_id);
            if (target == nullptr)
            {
                throw std::runtime_error("Specified source node does not exist; use LabeledDigraph<...>::addNode() to add node");
            }

            // Compute the required spanning forest matrices ...
            Matrix<InternalType, Dynamic, Dynamic> forest_one_root, forest_two_roots; 
            if (sparse)
            {
                // Instantiate a *sparse row-major* sub-Laplacian matrix
                SparseMatrix<InternalType, RowMajor> sublaplacian = this->getRowSublaplacianSparse(target);  

                // Then run the Chebotarev-Agaev recurrence to get the two-root
                // spanning forest matrix
                forest_two_roots = this->getSpanningForestMatrix(this->numnodes - 2, sublaplacian);

                // Then run the Chebotarev-Agaev recurrence one more time to get 
                // the one-root spanning forest (tree) matrix  
                forest_one_root = chebotarevAgaevRecurrence<InternalType>(this->numnodes - 2, sublaplacian, forest_two_roots); 
            }
            else 
            {
                // Instantiate a *dense* sub-Laplacian matrix
                Matrix<InternalType, Dynamic, Dynamic> sublaplacian = this->getRowSublaplacianDense(target); 

                // Then run the Chebotarev-Agaev recurrence to get the two-root
                // forest matrix 
                forest_two_roots = this->getSpanningForestMatrix(this->numnodes - 2, sublaplacian);

                // Then run the Chebotarev-Agaev recurrence one more time to get 
                // the one-root forest (tree) matrix  
                forest_one_root = chebotarevAgaevRecurrence<InternalType>(this->numnodes - 2, sublaplacian, forest_two_roots); 
            }

            // Get the index of the target node
            int t = 0;  
            for (auto it = this->order.begin(); it != this->order.end(); ++it)
            {
                if (*it == target)
                {
                    t = std::distance(this->order.begin(), it);
                    break;
                }
            } 

            // Now compute the desired mean first-passage times ...
            int z = this->numnodes - 1 - t;
            Matrix<InternalType, Dynamic, 1> mean_times = Matrix<InternalType, Dynamic, 1>::Zero(this->numnodes); 
            if (t == 0)
            {
                mean_times.tail(z) = forest_two_roots.block(1, 1, z, z).rowwise().sum(); 
            }
            else if (z == 0)
            {
                mean_times.head(t) = forest_two_roots.block(0, 0, t, t).rowwise().sum();
            }
            else
            {
                mean_times.head(t) = forest_two_roots.block(0, 0, t, t).rowwise().sum(); 
                mean_times.head(t) += forest_two_roots.block(0, t + 1, t, z).rowwise().sum(); 
                mean_times.tail(z) = forest_two_roots.block(t + 1, 0, z, t).rowwise().sum(); 
                mean_times.tail(z) += forest_two_roots.block(t + 1, t + 1, z, z).rowwise().sum();
            } 

            return (mean_times / forest_one_root(t, t)).template cast<FloatIOType>(); 
        }

        /**
         * Compute the vector of second moments of the *unconditional* 
         * first-passage times in the Markov process associated with the
         * graph from each node to the target node, using the given linear
         * solver method.
         *
         * This method assumes that the associated Markov process certainly 
         * eventually reaches the target node from each node in the graph,
         * meaning that there are no alternative terminal nodes (or rather
         * SCCs) to which the process can travel and get "stuck".
         *
         * Since this method necessarily returns a vector with floating-point
         * quantities, which may not be accommodated by `IOType`, this method 
         * allows the specification of a new output type, `FloatIOType`.
         * Moreover, since Eigen's QR/LU solvers requires a floating-point 
         * scalar type, this method also allows the specification of a new 
         * internal type, `FloatInternalType`.
         *
         * @param target_id ID of target node. 
         * @param method    Linear solver method for computing the vector of 
         *                  first-passage time second moments.
         * @returns Vector of first-passage time second moments to the target
         *          node from every node in the graph.
         * @throws std::invalid_argument if solver method is not recognized.
         * @throws std::runtime_error    if target node does not exist. 
         */
        template <typename FloatInternalType = InternalType, typename FloatIOType = IOType>
        Matrix<FloatIOType, Dynamic, 1> getSecondMomentsOfFirstPassageTimesFromSolver(std::string target_id, 
                                                                                      const SolverMethod method = QRDecomposition)
        {
            // Get the index of the target node
            Node* target = this->getNode(target_id);
            if (target == nullptr)
            {
                throw std::runtime_error("Specified source node does not exist; use LabeledDigraph<...>::addNode() to add node");
            }
            int t = 0;  
            for (auto it = this->order.begin(); it != this->order.end(); ++it)
            {
                if (*it == target)
                {
                    t = std::distance(this->order.begin(), it);
                    break;
                }
            } 

            // Get the row Laplacian matrix of the graph and drop the row and
            // column corresponding to the target node 
            Matrix<FloatInternalType, Dynamic, Dynamic> laplacian =
                -this->getColumnLaplacianDense().transpose().template cast<FloatInternalType>();
            Matrix<FloatInternalType, Dynamic, Dynamic> sublaplacian(this->numnodes - 1, this->numnodes - 1);
            int z = this->numnodes - 1 - t;
            if (t == 0)
            {
                sublaplacian = laplacian.block(1, 1, z, z);
            }
            else if (z == 0)
            {
                sublaplacian = laplacian.block(0, 0, t, t); 
            }
            else 
            { 
                sublaplacian.block(0, 0, t, t) = laplacian.block(0, 0, t, t);
                sublaplacian.block(0, t, t, z) = laplacian.block(0, t + 1, t, z); 
                sublaplacian.block(t, 0, z, t) = laplacian.block(t + 1, 0, z, t); 
                sublaplacian.block(t, t, z, z) = laplacian.block(t + 1, t + 1, z, z);
            } 

            // Get the left-hand matrix in the first-passage time linear system
            Matrix<FloatInternalType, Dynamic, Dynamic> A = sublaplacian * sublaplacian * sublaplacian; 

            // Get the right-hand vector in the first-passage time linear system 
            Matrix<FloatInternalType, Dynamic, 1> b = Matrix<FloatInternalType, Dynamic, 1>::Zero(this->numnodes - 1); 
            for (unsigned i = 0; i < t; ++i)
            {
                if (this->hasEdge(this->order[i], target))
                    b(i) = this->edges[this->order[i]][target];
            }
            for (unsigned i = t + 1; i < this->numnodes; ++i)
            {
                if (this->hasEdge(this->order[i], target))
                    b(i - 1) = this->edges[this->order[i]][target]; 
            }

            // Solve the linear system with the specified method 
            Matrix<FloatInternalType, Dynamic, 1> solution; 
            if (method == QRDecomposition)
            { 
                solution = solveByQRD<FloatInternalType>(A, b);
            }
            else if (method == LUDecomposition)
            { 
                solution = solveByLUD<FloatInternalType>(A, b);
            }
            else
            {
                std::stringstream ss; 
                ss << "Invalid linear system solution method specified: " << method; 
                throw std::invalid_argument(ss.str());
            } 

            // Return an augmented vector with the zero mean FPT from the 
            // target node to itself
            Matrix<FloatInternalType, Dynamic, 1> fpt_vec = Matrix<FloatInternalType, Dynamic, 1>::Zero(this->numnodes);
            for (unsigned i = 0; i < t; ++i)
                fpt_vec(i) = solution(i); 
            for (unsigned i = t + 1; i < this->numnodes; ++i)
                fpt_vec(i) = solution(i - 1);

            return (fpt_vec * 2).template cast<FloatIOType>(); 
        }

        /**
         * Compute the vector of second moments of the *unconditional* 
         * first-passage times in the Markov process associated with the
         * graph from each node to the target node, using the recurrence 
         * of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs.\ 17-18).
         *
         * This method assumes that the associated Markov process certainly 
         * eventually reaches the target node from each node in the graph,
         * meaning that there are no alternative terminal nodes (or rather
         * SCCs) to which the process can travel and get "stuck".
         *
         * Since this method necessarily returns a vector with floating-point
         * quantities, which may not be accommodated by `IOType`, this method 
         * allows the specification of a third output type, `FloatIOType`. 
         *
         * @param target_id ID of target node.
         * @param sparse    If true, use a sparse Laplacian matrix in the calculations. 
         * @returns Vector of first-passage time second moments to the target
         *          node from every node in the graph.
         * @throws std::runtime_error    if target node does not exist. 
         */
        template <typename FloatIOType = IOType>
        Matrix<FloatIOType, Dynamic, 1> getSecondMomentsOfFirstPassageTimesFromRecurrence(std::string target_id,
                                                                                          const bool sparse)
        {
            Node* target = this->getNode(target_id);
            if (target == nullptr)
            {
                throw std::runtime_error("Specified source node does not exist; use LabeledDigraph<...>::addNode() to add node");
            }

            // Compute the required spanning forest matrices ...
            Matrix<InternalType, Dynamic, Dynamic> forest_one_root, forest_two_roots;
            if (sparse)
            {
                // Instantiate a *sparse row-major* sub-Laplacian matrix
                SparseMatrix<InternalType, RowMajor> sublaplacian = this->getRowSublaplacianSparse(target);  
                
                // Then run the Chebotarev-Agaev recurrence to get the two-root
                // forest matrix 
                forest_two_roots = this->getSpanningForestMatrix(this->numnodes - 2, sublaplacian);

                // Then run the Chebotarev-Agaev recurrence one more time to get 
                // the one-root forest (tree) matrix  
                forest_one_root = chebotarevAgaevRecurrence<InternalType>(this->numnodes - 2, sublaplacian, forest_two_roots);
            }
            else 
            {
                // Instantiate a *dense* sub-Laplacian matrix
                Matrix<InternalType, Dynamic, Dynamic> sublaplacian = this->getRowSublaplacianDense(target); 

                // Then run the Chebotarev-Agaev recurrence to get the two-root
                // forest matrix 
                forest_two_roots = this->getSpanningForestMatrix(this->numnodes - 2, sublaplacian);

                // Then run the Chebotarev-Agaev recurrence one more time to get 
                // the one-root forest (tree) matrix  
                forest_one_root = chebotarevAgaevRecurrence<InternalType>(this->numnodes - 2, sublaplacian, forest_two_roots);
            }

            // Get the index of the target node
            int t = 0;  
            for (auto it = this->order.begin(); it != this->order.end(); ++it)
            {
                if (*it == target)
                {
                    t = std::distance(this->order.begin(), it);
                    break;
                }
            } 
            int z = this->numnodes - 1 - t; 

            // Finally compute the desired second moments ...
            Matrix<InternalType, Dynamic, 1> second_moments = Matrix<InternalType, Dynamic, 1>::Zero(this->numnodes);

            // 1) Get the sub-matrix of the two-root forest matrix obtained by deleting 
            // the row and column corresponding to the target node 
            Matrix<InternalType, Dynamic, Dynamic> forest_two_roots_sub =
                Matrix<InternalType, Dynamic, Dynamic>::Zero(this->numnodes - 1, this->numnodes - 1);
            if (t == 0)
            {
                forest_two_roots_sub = forest_two_roots.block(1, 1, z, z); 
            }
            else if (z == 0)
            {
                forest_two_roots_sub = forest_two_roots.block(0, 0, t, t); 
            }
            else 
            {
                forest_two_roots_sub.block(0, 0, t, t) = forest_two_roots.block(0, 0, t, t); 
                forest_two_roots_sub.block(0, t, t, z) = forest_two_roots.block(0, t + 1, t, z); 
                forest_two_roots_sub.block(t, 0, z, t) = forest_two_roots.block(t + 1, 0, z, t); 
                forest_two_roots_sub.block(t, t, z, z) = forest_two_roots.block(t + 1, t + 1, z, z); 
            }

            // 2) Get the square of this sub-matrix 
            Matrix<InternalType, Dynamic, Dynamic> forest_two_roots_sub_squared = forest_two_roots_sub * forest_two_roots_sub;

            // 3) The second moments of each FPT to the target node is given by the
            // corresponding row sum in this squared sub-matrix
            if (t == 0)
            {
                second_moments.tail(z) = forest_two_roots_sub_squared.block(1, 1, z, z).rowwise().sum(); 
            }
            else if (z == 0)
            {
                second_moments.head(t) = forest_two_roots_sub_squared.block(0, 0, t, t).rowwise().sum(); 
            }
            else 
            {
                second_moments.head(t) = forest_two_roots_sub_squared.block(0, 0, t, t).rowwise().sum(); 
                second_moments.head(t) += forest_two_roots_sub_squared.block(0, t + 1, t, z).rowwise().sum(); 
                second_moments.tail(z) = forest_two_roots_sub_squared.block(t + 1, 0, z, t).rowwise().sum(); 
                second_moments.tail(z) += forest_two_roots_sub_squared.block(t + 1, t + 1, z, z).rowwise().sum();
            } 

            return (2 * second_moments).template cast<FloatIOType>() / (
                static_cast<FloatIOType>(forest_one_root(t, t) * forest_one_root(t, t))
            );
        }

        /**
         * Simulate the Markov process associated with the graph up to a 
         * given maximum time.
         *
         * This method returns a vector of lifetimes of scalar type double,
         * together with the ID of the last node that was reached.
         *
         * @param init_node_id ID of initial node.
         * @param max_time     Time limit.
         * @param seed         Seed for random number generator. 
         * @returns Vector of visited nodes and their lifetimes, together 
         *          with the ID of the last node that was reached.
         * @throws std::invalid_argument If a node with the given ID does
         *                               not exist or if the maximum time 
         *                               is not positive.
         */
        std::pair<std::vector<std::pair<std::string, double> >, std::string> simulate(const std::string init_node_id,
                                                                                      const IOType max_time,
                                                                                      const int seed)
        {
            boost::random::mt19937 rng(seed);

            // Check that a Node with the given ID exists 
            Node* node = this->getNode(init_node_id);
            if (node == nullptr)
                throw std::invalid_argument("Node with specified ID does not exist");

            // Check that the given time limit is positive 
            if (max_time <= 0)
                throw std::invalid_argument("Specified time limit is not positive");

            // Simulate the process until the time limit is exceeded or a node 
            // with no outgoing edges (an absorbing node) is reached
            std::vector<std::pair<std::string, double> > simulation;
            double time = 0;
            std::string curr_id = init_node_id;
            Node* curr_node = this->getNode(curr_id);
            Node* final_node = this->getNode(curr_id); 
            while (time < max_time)
            {
                // First choose the next node
                std::vector<Node*> dests; 
                std::vector<double> weights;
                for (auto it = this->edges[curr_node].begin(); it != this->edges[curr_node].end(); ++it)
                {
                    dests.push_back(it->first);
                    weights.push_back(static_cast<double>(it->second));
                }
                if (dests.size() == 0)    // If there is no node to go to from here
                    break;
                boost::random::discrete_distribution<> dist_next(weights);
                int next_idx = dist_next(rng);

                // If the next node was successfully selected, add current node
                // and its lifetime
                InternalType rate_sum = 0;
                for (auto it = this->edges[curr_node].begin(); it != this->edges[curr_node].end(); ++it)
                    rate_sum += it->second;
                boost::random::exponential_distribution<> dist_time(static_cast<double>(rate_sum));
                double lifetime = dist_time(rng);
                simulation.emplace_back(std::make_pair(curr_id, lifetime));
                time += lifetime;
                
                // Update current node
                curr_node = dests[next_idx];
                curr_id = curr_node->getId();
                final_node = curr_node;
            }

            return std::make_pair(simulation, final_node->getId()); 
        }
};

#endif
