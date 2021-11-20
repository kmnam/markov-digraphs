#ifndef LABELED_DIGRAPHS_HPP
#define LABELED_DIGRAPHS_HPP

#include <cmath>
#include <vector>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "kahan.hpp"

/*
 * An implementation of a labeled directed graph.  
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/19/2021
 */
using namespace Eigen;

// ----------------------------------------------------- //
//       LINEAR ALGEBRA AND OTHER HELPER FUNCTIONS       //
// ----------------------------------------------------- //
template <typename T>
bool isclose(T a, T b, T tol)
{
    /*
     * Return true if abs(a - b) < tol.
     */
    T c = a - b;
    return ((c >= 0 && c < tol) || (c < 0 && -c < tol));
}

template <typename T>
Matrix<T, Dynamic, 1> nullspaceSVD(const Ref<const Matrix<T, Dynamic, Dynamic> >& A)
{
    /*
     * Compute the nullspace of A by performing a singular value decomposition.
     *
     * This function returns the column of V in the SVD of A = USV corresponding
     * to the least singular value (recall that singular values are always
     * non-negative). It therefore effectively assumes that the A has a 
     * nullspace of dimension one.
     */
    // Perform a singular value decomposition of A, only computing V in full
    Eigen::BDCSVD<Matrix<T, Dynamic, Dynamic> > svd(A, Eigen::ComputeFullV);

    // Return the column of V corresponding to the least singular value of A
    // (always given last in the decomposition) 
    Matrix<T, Dynamic, 1> singular = svd.singularValues(); 
    Matrix<T, Dynamic, Dynamic> V = svd.matrixV();
    return V.col(singular.rows() - 1); 
}

template <typename T>
Matrix<T, Dynamic, Dynamic> nullspaceSVD(const Ref<const Matrix<T, Dynamic, Dynamic> >& A, const T tol)
{
    /*
     * Compute the nullspace of A by performing a singular value decomposition.
     *
     * This function returns the column(s) of V in the SVD of A = USV
     * corresponding to singular values with absolute value < tol.
     */
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

template <typename T>
Matrix<T, Dynamic, 1> solveByQRD(const Ref<const Matrix<T, Dynamic, Dynamic> >& A, 
                                 const Ref<const Matrix<T, Dynamic, 1> >& b)
{
    /*
     * Solve the non-homogeneous linear system Ax = b by obtaining a QR 
     * decomposition of A. 
     *
     * A is assumed to be square.  
     */
    // Obtain a QR decomposition of A 
    Eigen::ColPivHouseholderQR<Matrix<T, Dynamic, Dynamic> > qrd(A);

    // Get and return the solution to Ax = b
    return qrd.solve(b); 
} 

template <typename T>
Matrix<T, Dynamic, Dynamic> chebotarevAgaevRecurrence(const Ref<const Matrix<T, Dynamic, Dynamic> >& laplacian,
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& curr,
                                                      int k, 
                                                      bool use_kahan_sum)
{
    /*
     * Apply one iteration of the recurrence of Chebotarev & Agaev (Lin Alg
     * Appl, 2002, Eqs. 17-18) for the in-forest matrices of the graph.
     *
     * This function takes and outputs *dense* matrices.
     *
     * This function also allows for usage of Kahan's compensated summation 
     * algorithm for matrix multiplication and trace evaluation.  
     */
    T K(k + 1);
    Matrix<T, Dynamic, Dynamic> product;
    T sigma; 
    if (use_kahan_sum)
    {
        product = kahanMultiply(laplacian, curr);
        sigma = kahanTrace(product) / K;
    } 
    else
    {
        product = laplacian * curr;
        sigma = product.trace() / K; 
    }
    Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(laplacian.rows(), laplacian.cols());

    return (sigma * identity) - product;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> chebotarevAgaevRecurrence(const SparseMatrix<T, RowMajor>& laplacian, 
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& curr,
                                                      int k,
                                                      bool use_kahan_sum)
{
    /*
     * Apply one iteration of the recurrence of Chebotarev & Agaev (Lin Alg
     * Appl, 2002, Eqs. 17-18) for the in-forest matrices of the graph.
     *
     * This function takes a *compressed sparse row-major* Laplacian matrix
     * as input, but takes and computes *dense* forest matrices.
     *
     * This function also allows for usage of Kahan's compensated summation 
     * algorithm for matrix multiplication and trace evaluation.  
     */
    T K(k + 1);
    Matrix<T, Dynamic, Dynamic> product;
    T sigma; 
    if (use_kahan_sum)
    {
        product = kahanMultiply(laplacian, curr);
        sigma = kahanTrace(product) / K;
    } 
    else
    {
        product = laplacian * curr;
        sigma = product.trace() / K; 
    }
    Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(laplacian.rows(), laplacian.cols());

    return (sigma * identity) - product;
}

// ----------------------------------------------------- //
//                    NODES AND EDGES                    //
// ----------------------------------------------------- //
struct Node
{
    /*
     * A minimal struct that represents a vertex or node. 
     */
    std::string id;       // String identifier

    Node(std::string id)
    {
        /*
         * Trivial constructor.
         */
        this->id = id;
    }

    ~Node()
    {
        /*
         * Trivial destructor. 
         */
    }

    bool operator==(const Node& other) const
    {
        /*
         * Trivial equality operator.
         */
        return (!this->id.compare(other.id));
    }

    bool operator!=(const Node& other) const
    {
        /*
         * Trivial inequality operator.
         */ 
        return (this->id.compare(other.id));
    }
};

// An Edge is simply a pair of Node pointers
using Edge = std::pair<Node*, Node*>;

template <typename T>
class LabeledDigraph
{
    /*
     * An implementation of a labeled digraph.  
     */
    protected:
        // ----------------------------------------------------- //
        //                       ATTRIBUTES                      //
        // ----------------------------------------------------- //
        // Number of nodes 
        unsigned numnodes = 0;

        // Store a canonical ordering of nodes 
        std::vector<Node*> order; 

        // Dictionary mapping string (ids) to Node pointers 
        std::unordered_map<std::string, Node*> nodes;

        // Maintain edges in a nested dictionary of adjacency "dictionaries"  
        std::unordered_map<Node*, std::unordered_map<Node*, T> > edges;

        // ----------------------------------------------------- //
        //                     PRIVATE METHODS                   //
        // ----------------------------------------------------- //
        Matrix<T, Dynamic, Dynamic> getSpanningForestMatrix(int k,
                                                            const Ref<const Matrix<T, Dynamic, Dynamic> >& laplacian,
                                                            bool use_kahan_sum)
        {
            /*
             * Compute the k-th spanning forest matrix, using the recurrence
             * of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs. 17-18).
             *
             * A private version of the corresponding public method, in which 
             * a pre-computed Laplacian matrix is provided as an argument. 
             *
             * This method uses a *dense* Laplacian matrix.
             */
            // Begin with the identity matrix 
            Matrix<T, Dynamic, Dynamic> curr = Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes);

            // Apply the recurrence ...
            for (unsigned i = 0; i < k; ++i)
                curr = chebotarevAgaevRecurrence<T>(laplacian, curr, i, use_kahan_sum);

            return curr; 
        }

        Matrix<T, Dynamic, Dynamic> getSpanningForestMatrixSparse(int k,
                                                                  const SparseMatrix<T, RowMajor>& laplacian,
                                                                  bool use_kahan_sum)
        {
            /*
             * Compute the k-th spanning forest matrix, using the recurrence
             * of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs. 17-18).
             *
             * A private version of the corresponding public method, in which 
             * a pre-computed Laplacian matrix is provided as an argument. 
             *
             * This method uses a *compressed sparse row-major* Laplacian matrix.  
             */
            // Begin with the identity matrix
            Matrix<T, Dynamic, Dynamic> curr = Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes); 

            // Apply the recurrence ...
            for (unsigned i = 0; i < k; ++i)
                curr = chebotarevAgaevRecurrence<T>(laplacian, curr, i, use_kahan_sum); 

            return curr; 
        }

        std::vector<Node*> DFS(Node* init)
        {
            /*
             * Perform a DFS starting from the given node.
             */
            std::vector<Node*> order;
            std::stack<Node*> stack;
            std::unordered_map<Node*, bool> visited;

            // Initialize every node as having been unvisited
            for (auto&& edge_set : this->edges)
                visited[edge_set.first] = false;

            // Starting from init, run until the current node has no 
            // unvisited neighbors
            Node* curr;
            stack.push(init);
            visited[init] = true;
            while (!stack.empty())
            {
                // Pop topmost node from the stack 
                curr = stack.top();
                stack.pop();
                order.push_back(curr);

                // Push every unvisited neighbor onto the stack
                for (auto&& dest : this->edges[curr])
                {
                    if (!visited[dest.first])
                    {
                        stack.push(dest.first);
                        visited[dest.first] = true;
                    }
                }
            }

            return order; 
        }

        void tarjanIter(Node* node, int curr, std::unordered_map<Node*, int>& index,
                        std::unordered_map<Node*, int>& lowlink, std::stack<Node*>& stack,
                        std::unordered_map<Node*, bool>& onstack,
                        std::vector<std::unordered_set<Node*> >& components)
        {
            /*
             * Recursive function to be called as part of Tarjan's algorithm. 
             */
            index[node] = curr;
            lowlink[node] = curr;
            curr++;
            stack.push(node);
            onstack[node] = true; 

            // Run through all the edges leaving the given node
            for (auto&& dest : this->edges[node])
            {
                if (index[dest.first] == -1)
                {
                    tarjanIter(dest.first, curr, index, lowlink, stack, onstack, components);
                    if (lowlink[dest.first] < lowlink[node])
                        lowlink[node] = lowlink[dest.first];
                }
                else if (onstack[dest.first])
                {
                    if (index[dest.first] < lowlink[node])
                        lowlink[node] = index[dest.first];
                }
            }

            // If the given node is a root (index[node] == lowlink[node]), 
            // pop successively from the stack until node is reached
            if (index[node] == lowlink[node])
            {
                std::unordered_set<Node*> component; 
                Node* next;
                while (next != node && !stack.empty())
                {
                    next = stack.top();
                    onstack[next] = false;
                    component.insert(next);
                    stack.pop();
                }
                
                // Append new component onto vector of components
                components.push_back(component);
            }
        }

        bool isTerminal(Node* node, std::vector<std::unordered_set<Node*> >& components,
                        unsigned component_index)
        {
            /*
             * Return true if the node lies in a terminal strongly connected
             * component.
             *
             * Search, via DFS, for another node outside the given node's SCC.
             */
            // Perform a DFS of the graph starting from the given node
            std::vector<Node*> traversal = this->DFS(node);

            // Is there a node in the traversal that does not fall into the
            // given node's SCC?
            for (auto&& v : traversal)
            {
                if (components[component_index].find(v) == components[component_index].end())
                    return false;
            }
            return true;
        }

        std::vector<std::unordered_set<Node*> > enumStronglyConnectedComponents()
        {
            /*
             * Run Tarjan's algorithm to enumerate the strongly connected
             * components (SCCs) of the graph. 
             */
            std::vector<std::unordered_set<Node*> > components; 
            std::stack<Node*> stack;
            std::unordered_map<Node*, int> index; 
            std::unordered_map<Node*, int> lowlink;
            std::unordered_map<Node*, bool> onstack;

            // Initialize index, lowlink, onstack dictionaries 
            for (auto&& edge_set : this->edges)
            {
                Node* node = edge_set.first;
                index[node] = -1;
                lowlink[node] = -1;
                onstack[node] = false;
            }

            // Traverse the nodes in the graph with DFS
            for (auto&& edge_set : this->edges)
            {
                Node* node = edge_set.first;

                // If not yet visited, call the recursive function on the node 
                if (index[node] == -1)
                    tarjanIter(node, 0, index, lowlink, stack, onstack, components);
            }

            return components;
        }

    public:
        LabeledDigraph()
        {
            /*
             * Empty constructor.
             */
        }

        ~LabeledDigraph()
        {
            /*
             * Destructor; de-allocates each node from heap memory.
             */
            for (auto&& edge_set : this->edges) delete edge_set.first;
        }

        unsigned getNumNodes() const 
        {
            /*
             * Return the number of nodes in the graph. 
             */
            return this->numnodes; 
        }

        // ----------------------------------------------------- //
        //              NODE-ADDING/GETTING METHODS              //
        // ----------------------------------------------------- //
        Node* addNode(std::string id)
        {
            /*
             * Add a node to the graph with the given id, and return pointer
             * to the new node.
             *
             * Throw std::runtime_error if node with given id already exists.
             */
            // Check that a node with the given id doesn't exist already
            if (this->nodes.find(id) != this->nodes.end())
                throw std::runtime_error("Node exists with specified id");

            Node* node = new Node(id);
            this->order.push_back(node); 
            this->nodes[id] = node;
            this->edges.emplace(node, std::unordered_map<Node*, T>());
            this->numnodes++;
            return node;
        }

        void removeNode(std::string id)
        {
            /*
             * Remove a node from the graph with the given id.
             *
             * Throw std::runtime_error if node with given id does not exist. 
             */
            // Check that a node with the given id exists
            if (this->nodes.find(id) == this->nodes.end())
                throw std::runtime_error("Node does not exist with specified id");

            // Find and delete node in this->order
            Node* node = this->getNode(id);
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
                // v.first of type Node*, v.second of type std::unordered_map<Node*, T>
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

        Node* getNode(std::string id) const
        {
            /*
             * Return pointer to node with given id.
             */
            auto it = this->nodes.find(id);
            if (it == this->nodes.end()) return nullptr;
            else return it->second; 
        }

        bool hasNode(std::string id) const 
        {
            /*
             * Return true if node with given id exists in the graph. 
             */
            return this->nodes.count(id);
        }

        std::vector<Node*> getAllNodes() const 
        {
            /*
             * Return this->order. 
             */
            return this->order; 
        }

        // ----------------------------------------------------- //
        //              EDGE-ADDING/GETTING METHODS              //
        // ----------------------------------------------------- //
        void addEdge(std::string source_id, std::string target_id, T label = 1)
        {
            /*
             * Add an edge between two nodes.
             *
             * If either id does not correspond to a node in the graph,
             * instantiate them.
             */
            // Look for the two nodes ...
            Node* source = this->getNode(source_id);
            Node* target = this->getNode(target_id);

            // ... and if they don't exist, define them 
            if (source == nullptr) source = this->addNode(source_id);
            if (target == nullptr) target = this->addNode(target_id);

            // Then define the edge
            this->edges[source][target] = label;
        }

        void removeEdge(std::string source_id, std::string target_id)
        {
            /*
             * Remove the edge between the two given nodes.
             *
             * Throw std::runtime_error if node with either given id does
             * not exist. 
             */
            // Check that nodes with the given ids exist
            if (this->nodes.find(source_id) == this->nodes.end())
                throw std::runtime_error("Node does not exist with specified id");
            if (this->nodes.find(target_id) == this->nodes.end())
                throw std::runtime_error("Node does not exist with specified id");

            // Does the given edge exist? 
            Node* source = this->getNode(source_id);
            Node* target = this->getNode(target_id);
            if (this->edges[source].find(target) != this->edges[source].end())
            {
                // If so, then get rid of it 
                this->edges[source].erase(this->edges[source].find(target));
            }
            // If not, then do nothing!
        }
        
        std::pair<Edge, T> getEdge(std::string source_id, std::string target_id)
        {
            /*
             * Return the edge between the specified nodes, along with the
             * edge label.  
             */
            Node* source = this->getNode(source_id);
            Node* target = this->getNode(target_id);

            // Check that both nodes exist 
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist");
            if (target == nullptr)
                throw std::runtime_error("Specified target node does not exist");

            // Check that the edge exists 
            auto it = this->edges[source].find(target);
            if (it != this->edges[source].end())
                return std::make_pair(std::make_pair(source, it->first), it->second);
            else    // Return pair of nullptrs and zero label if no edge exists
                return std::make_pair(std::make_pair(nullptr, nullptr), 0);
        }

        std::vector<std::pair<Edge, T> > getAllEdgesFromNode(std::string source_id) 
        {
            /*
             * Return the std::vector of edges leaving the given node, given 
             * the *id* of the source node.  
             */
            std::vector<std::pair<Edge, T> > edges_from_node;
            Node* source = this->getNode(source_id);

            // Check that the given node exists
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist");

            // Run through all nodes in this->order ...
            for (auto&& node : this->order)
            {
                // Is there an edge to this node?
                if (this->edges[source].find(node) != this->edges[source].end())
                {
                    // If so, instantiate the edge and get the label 
                    Edge edge = std::make_pair(source, node);
                    T label = this->edges[source][node];
                    edges_from_node.push_back(std::make_pair(edge, label)); 
                }
            }

            return edges_from_node; 
        }

        std::vector<std::pair<Edge, T> > getAllEdgesFromNode(Node* source) 
        {
            /*
             * Return the std::vector of edges leaving the given node, 
             * given the *pointer* to the source Node object itself.  
             */
            std::vector<std::pair<Edge, T> > edges_from_node;

            // Check that the given node exists
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist");

            // Run through all nodes in this->order ...
            for (auto&& node : this->order)
            {
                // Is there an edge to this node?
                if (this->edges[source].find(node) != this->edges[source].end())
                {
                    // If so, instantiate the edge and get the label 
                    Edge edge = std::make_pair(source, node);
                    T label = this->edges[source][node];
                    edges_from_node.push_back(std::make_pair(edge, label)); 
                }
            }

            return edges_from_node; 
        }

        std::vector<std::pair<int, T> > getAllEdgesFromNode(int source_idx) 
        {
            /*
             * Return the std::vector of edges leaving the given node. 
             *
             * This method takes the *index* of the source node in this->order 
             * and returns the *indices* of the target nodes with an edge from
             * the source node in this->order.  
             */
            std::vector<std::pair<int, T> > edges_from_node;
            Node* source = this->order[source_idx];

            // Check that the given node exists
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist");

            // Run through all nodes in this->order ...
            for (int i = 0; i < this->numnodes; ++i)
            {
                // Is there an edge to this node?
                Node* target = this->order[i]; 
                if (this->edges[source].find(target) != this->edges[source].end())
                {
                    // If so, instantiate the edge and get the label 
                    T label = this->edges[source][target];
                    edges_from_node.push_back(std::make_pair(i, label)); 
                }
            }

            return edges_from_node; 
        }

        bool hasEdge(std::string source_id, std::string target_id) const
        {
            /*
             * Given the *ids* of two nodes, return true if the given edge
             * exists in the graph.  
             */
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

        bool hasEdge(Node* source, Node* target) const 
        {
            /*
             * Given the *pointers* to two nodes, return true if the given
             * edge exists in the graph. 
             */
            return (this->edges.count(source) && (this->edges.find(source))->second.count(target)); 
        }


        void setEdgeLabel(std::string source_id, std::string target_id, T value)
        {
            /*
             * Update the numerical value of the given edge label. 
             * 
             * Throw std::runtime_error if either node or the edge does not exist.
             */
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
                this->edges[source][target] = value;
            else 
                throw std::runtime_error("Specified edge does not exist");
        }

        // ----------------------------------------------------- //
        //                     OTHER METHODS                     //
        // ----------------------------------------------------- //
        LabeledDigraph<T>* subgraph(std::unordered_set<Node*> nodes)
        {
            /*
             * Return the subgraph induced by the given subset of nodes.
             */
            LabeledDigraph<T>* subgraph = new LabeledDigraph<T>();

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
                    if (nodes.find(edge.first) != nodes.end())
                    {
                        subgraph->addEdge(v->id, edge.first->id, edge.second);
                    }
                }
            }

            return subgraph; 
        }
        
        void clear()
        {
            /* 
             * Clear the graph's contents.
             */
            // De-allocate all Nodes from heap memory
            for (auto&& edge_set : this->edges) delete edge_set.first;

            // Clear all attributes
            this->nodes.clear();
            this->edges.clear();
        }

        template <typename U = T>
        LabeledDigraph<U>* copy() const
        {
            /*
             * Return pointer to a new LabeledDigraph object, possibly
             * with a different scalar type, with the same graph structure
             * and edge label values.
             */
            LabeledDigraph<U>* graph = new LabeledDigraph<U>();

            // Copy over nodes with the same ids
            for (auto&& node : this->order)
                graph->addNode(node->id);
            
            // Copy over edges and edge labels
            for (auto&& edge_set : this->edges)
            {
                for (auto&& dest : edge_set.second)
                {
                    U label(dest.second);
                    graph->addEdge(edge_set.first->id, dest.first->id, label);
                }
            }

            return graph;
        }

        template <typename U = T>
        void copy(LabeledDigraph<U>* graph) const
        {
            /*
             * Given pointer to an existing LabeledDigraph object, possibly
             * with a different scalar type, copy over the graph details.  
             */
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
                    U label(dest.second);
                    graph->addEdge(edge_set.first->id, dest.first->id, label);
                }
            }
        }

        Matrix<T, Dynamic, Dynamic> getLaplacian()
        {
            /*
             * Return a numerical Laplacian matrix with the given scalar type,
             * according to the given node ordering.
             *
             * The ordering is assumed to include every node in the graph once.
             */
            // Initialize a zero matrix with #rows = #cols = #nodes
            Matrix<T, Dynamic, Dynamic> laplacian = Matrix<T, Dynamic, Dynamic>::Zero(this->numnodes, this->numnodes);

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
                            T label(this->edges[w][v]);
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
                laplacian(i,i) = -(laplacian.col(i).sum());

            return laplacian;
        }

        Matrix<T, Dynamic, Dynamic> getSpanningForestMatrix(int k, bool use_kahan_sum)
        {
            /*
             * Compute the k-th spanning forest matrix, using the recurrence
             * of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs. 17-18).
             *
             * This method uses a *dense* Laplacian matrix.  
             */
            // Begin with the identity matrix 
            Matrix<T, Dynamic, Dynamic> curr = Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes);

            // Initialize a zero matrix with #rows = #cols = #nodes
            Matrix<T, Dynamic, Dynamic> laplacian = Matrix<T, Dynamic, Dynamic>::Zero(this->numnodes, this->numnodes);

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
                            T label(this->edges[v][w]);
                            laplacian(i, j) = -label;
                        }
                    }
                    j++;
                }
                i++;
            }

            // Populate diagonal entries as negative sums of the off-diagonal
            // entries in each row
            if (use_kahan_sum)
            {
                for (unsigned i = 0; i < this->numnodes; ++i)
                    laplacian(i, i) = -kahanRowSum(laplacian, i);
            }
            else
            { 
                for (unsigned i = 0; i < this->numnodes; ++i)
                    laplacian(i, i) = -(laplacian.row(i).sum());
            }

            // Apply the recurrence ...
            for (unsigned i = 0; i < k; ++i)
                curr = chebotarevAgaevRecurrence<T>(laplacian, curr, i, use_kahan_sum);

            return curr; 
        }

        Matrix<T, Dynamic, Dynamic> getSpanningForestMatrixSparse(int k, bool use_kahan_sum)
        {
            /*
             * Compute the k-th spanning forest matrix, using the recurrence
             * of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs. 17-18).
             *
             * This method uses a *sparse* Laplacian matrix.  
             */
            // Begin with the identity matrix
            Matrix<T, Dynamic, Dynamic> curr = Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes); 

            // Initialize a zero matrix with #rows = #cols = #nodes
            SparseMatrix<T, RowMajor> laplacian(this->numnodes, this->numnodes); 

            // Populate the entries of the matrix: the off-diagonal (i,j)-th  
            // entry is the *negative* of the label of the edge i -> j, and 
            // the diagonal entries are set so that each *row* sum is zero 
            std::vector<Triplet<T> > laplacian_triplets; 
            unsigned i = 0;
            for (auto&& v : this->order)
            {
                std::vector<T> row_entries;    // All nonzero off-diagonal entries in i-th row
                unsigned j = 0;
                for (auto&& w : this->order)
                {
                    if (i != j)
                    {
                        // Get the edge label for i -> j
                        if (this->edges[v].find(w) != this->edges[v].end())
                        {
                            T label(this->edges[v][w]);
                            laplacian_triplets.push_back(Triplet<T>(i, j, -label));
                            row_entries.push_back(label);  
                        }
                    }
                    j++;
                }

                // Compute the negative of the i-th row sum
                T row_sum = 0; 
                if (use_kahan_sum)
                {
                    row_sum = kahanVectorSum(row_entries);
                }
                else 
                {
                    for (const T entry : row_entries)
                        row_sum += entry; 
                } 
                laplacian_triplets.push_back(Triplet<T>(i, i, row_sum)); 
                i++;
            }
            laplacian.setFromTriplets(laplacian_triplets.begin(), laplacian_triplets.end());

            // Apply the recurrence ...
            for (unsigned i = 0; i < k; ++i)
                curr = chebotarevAgaevRecurrence<T>(laplacian, curr, i, use_kahan_sum); 

            return curr; 
        }

        Matrix<T, Dynamic, 1> getSteadyStateFromSVD()
        {
            /*
             * Return a *normalized* steady-state vector for the Laplacian 
             * dynamics on the graph, with the nodes ordered according to the
             * canonical ordering, by solving for the nullspace of the Laplacian
             * matrix.
             *
             * This method assumes that the graph is strongly connected, in
             * which case there is a unique (up to normalization) steady-state
             * vector. If the graph is not strongly connected, then this 
             * method returns only one basis vector for the nullspace of 
             * the Laplacian matrix.   
             */
            Matrix<T, Dynamic, Dynamic> laplacian = this->getLaplacian();
            
            // Obtain the nullspace matrix of the Laplacian matrix
            Matrix<T, Dynamic, Dynamic> nullmat;
            try
            {
                nullmat = nullspaceSVD<T>(laplacian);
            }
            catch (const std::runtime_error& e)
            {
                throw;
            }

            // Each column is a nullspace basis vector (for each SCC) and
            // each row corresponds to a node in the graph
            Matrix<T, Dynamic, 1> steady_state = nullmat.array().rowwise().sum().matrix();
           
            // Normalize by the sum of its entries and return 
            T norm = steady_state.sum();
            return steady_state / norm;
        }

        Matrix<T, Dynamic, 1> getSteadyStateFromRecurrence(bool sparse, bool use_kahan_sum)
        {
            /*
             * Return a *normalized* steady-state vector for the Laplacian 
             * dynamics on the graph, with the nodes ordered according to the
             * canonical ordering, by solving the recurrence relation of 
             * Chebotarev & Agaev for the k-th forest matrix (Chebotarev & Agaev,
             * Lin Alg Appl, 2002, Eqs. 17-18).
             *
             * Setting sparse = true enforces the use of a sparse Laplacian
             * matrix in the calculations. 
             */
            // Obtain the spanning tree weight matrix from the row Laplacian matrix
            Matrix<T, Dynamic, Dynamic> forest_matrix; 
            if (sparse)
                forest_matrix = this->getSpanningForestMatrixSparse(this->numnodes - 1, use_kahan_sum); 
            else 
                forest_matrix = this->getSpanningForestMatrix(this->numnodes - 1, use_kahan_sum);

            // Return any row of the matrix (after normalizing by the sum of 
            // its entries)
            T norm;
            if (use_kahan_sum)
                norm = kahanRowSum(forest_matrix, 0); 
            else 
                norm = forest_matrix.row(0).sum(); 
            return forest_matrix.row(0) / norm; 
        }

        Matrix<T, Dynamic, 1> getMeanFirstPassageTimesFromQRD(Node* target)
        {
            /*
             * Return the mean first-passage time to the given target node from 
             * *every node* (including itself), assuming that the target node
             * will always eventually be reached, no matter which starting node
             * is chosen.  
             *
             * This quantity is only well-defined if there are no alternative 
             * terminal SCCs to which a path from the source node can travel.
             * In other words, for any node in the graph with a path from the
             * source node, there must also exist a path from that node to the
             * target node.
             */
            // Get the index of the target node 
            int t; 
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
            Matrix<T, Dynamic, Dynamic> laplacian = this->getLaplacian();
            Matrix<T, Dynamic, Dynamic> sublaplacian(this->numnodes - 1, this->numnodes - 1);
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
            Matrix<T, Dynamic, Dynamic> A = sublaplacian.transpose() * sublaplacian.transpose();

            // Get the right-hand vector in the first-passage time linear system 
            Matrix<T, Dynamic, 1> b = Matrix<T, Dynamic, 1>::Zero(this->numnodes - 1); 
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

            // Solve the linear system and return the coordinate corresponding 
            // to the given source node 
            Matrix<T, Dynamic, 1> solution = solveByQRD<T>(A, b);

            // Return an augmented vector with the zero mean FPT from the 
            // target node to itself
            Matrix<T, Dynamic, 1> fpt_vec = Matrix<T, Dynamic, 1>::Zero(this->numnodes);
            for (unsigned i = 0; i < t; ++i)
                fpt_vec(i) = solution(i); 
            for (unsigned i = t + 1; i < this->numnodes; ++i)
                fpt_vec(i) = solution(i - 1);

            return fpt_vec;  
        }

        Matrix<T, Dynamic, 1> getMeanFirstPassageTimesFromRecurrence(Node* target,
                                                                     bool sparse,
                                                                     bool use_kahan_sum)
        {
            /*
             * Return the mean first-passage time to the given target node from 
             * *every node* (including itself), assuming that the target node
             * will always eventually be reached, no matter which starting node
             * is chosen.  
             *
             * This quantity is only well-defined if there are no alternative 
             * terminal SCCs to which a path from the source node can travel.
             * In other words, for any node in the graph with a path from the
             * source node, there must also exist a path from that node to the
             * target node.  
             */
            // Compute the required spanning forest matrices ...
            Matrix<T, Dynamic, Dynamic> forest_one_root, forest_two_roots;
            if (sparse)
            {
                // Instantiate a *sparse row-major* Laplacian matrix ...
                //
                // Initialize a zero matrix with #rows = #cols = #nodes
                SparseMatrix<T, RowMajor> laplacian(this->numnodes, this->numnodes); 

                // Populate the entries of the matrix: the off-diagonal (i,j)-th  
                // entry is the *negative* of the label of the edge i -> j, and 
                // the diagonal entries are set so that each *row* sum is zero 
                std::vector<Triplet<T> > laplacian_triplets; 
                unsigned i = 0;
                for (auto&& v : this->order)
                {
                    std::vector<T> row_entries;    // All nonzero off-diagonal entries in i-th row
                    unsigned j = 0;
                    for (auto&& w : this->order)
                    {
                        if (i != j)
                        {
                            // Get the edge label for i -> j, omitting all edges 
                            // for which i is the target node 
                            if (v != target && this->edges[v].find(w) != this->edges[v].end())
                            {
                                T label(this->edges[v][w]);
                                laplacian_triplets.push_back(Triplet<T>(i, j, -label));
                                row_entries.push_back(label);  
                            }
                        }
                        j++;
                    }
                    // Compute the negative of the i-th row sum
                    T row_sum = 0; 
                    if (use_kahan_sum)
                    {
                        row_sum = kahanVectorSum(row_entries);
                    }
                    else 
                    {
                        for (const T entry : row_entries)
                            row_sum += entry; 
                    } 
                    laplacian_triplets.push_back(Triplet<T>(i, i, row_sum)); 
                    i++;
                }
                laplacian.setFromTriplets(laplacian_triplets.begin(), laplacian_triplets.end());

                // Then run the Chebotarev-Agaev recurrence to get the two-root
                // forest matrix 
                forest_two_roots = this->getSpanningForestMatrixSparse(
                    this->numnodes - 2, laplacian, use_kahan_sum
                );

                // Then run the Chebotarev-Agaev recurrence one more time to get 
                // the one-root forest (tree) matrix  
                forest_one_root = chebotarevAgaevRecurrence<T>(
                    laplacian, forest_two_roots, this->numnodes - 2, use_kahan_sum
                );
            }
            else 
            {
                // Instantiate a *dense* Laplacian matrix ...
                //
                // Initialize a zero matrix with #rows = #cols = #nodes
                Matrix<T, Dynamic, Dynamic> laplacian = Matrix<T, Dynamic, Dynamic>::Zero(this->numnodes, this->numnodes);

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
                            // for which i is the target node 
                            if (v != target && this->edges[v].find(w) != this->edges[v].end())
                            {
                                T label(this->edges[v][w]);
                                laplacian(i, j) = -label;
                            }
                        }
                        j++;
                    }
                    i++;
                }

                // Populate diagonal entries as negative sums of the off-diagonal
                // entries in each row
                if (use_kahan_sum)
                {
                    for (unsigned i = 0; i < this->numnodes; ++i)
                        laplacian(i, i) = -kahanRowSum(laplacian, i);
                }
                else
                { 
                    for (unsigned i = 0; i < this->numnodes; ++i)
                        laplacian(i, i) = -(laplacian.row(i).sum());
                }

                // Then run the Chebotarev-Agaev recurrence to get the two-root
                // forest matrix 
                forest_two_roots = this->getSpanningForestMatrix(
                    this->numnodes - 2, laplacian, use_kahan_sum
                );

                // Then run the Chebotarev-Agaev recurrence one more time to get 
                // the one-root forest (tree) matrix  
                forest_one_root = chebotarevAgaevRecurrence<T>(
                    laplacian, forest_two_roots, this->numnodes - 2, use_kahan_sum
                );
            }

            // Now compute the desired mean first-passage times ...
            int t;
            Matrix<T, Dynamic, 1> mean_times = Matrix<T, Dynamic, 1>::Zero(this->numnodes);  
            for (auto it = this->order.begin(); it != this->order.end(); ++it)
            {
                if (*it == target)
                {
                    t = std::distance(this->order.begin(), it); 
                    break;
                }
            }
            int z = this->numnodes - 1 - t;
            if (use_kahan_sum)
            {
                if (t == 0)
                {
                    mean_times.tail(z) = kahanRowSum(forest_two_roots.block(1, 1, z, z)); 
                }
                else if (z == 0)
                {
                    mean_times.head(t) = kahanRowSum(forest_two_roots.block(0, 0, t, t)); 
                }
                else
                { 
                    mean_times.head(t) = kahanRowSum(forest_two_roots.block(0, 0, t, t)); 
                    mean_times.head(t) += kahanRowSum(forest_two_roots.block(0, t + 1, t, z)); 
                    mean_times.tail(z) = kahanRowSum(forest_two_roots.block(t + 1, 0, z, t)); 
                    mean_times.tail(z) += kahanRowSum(forest_two_roots.block(t + 1, t + 1, z, z));
                }
            }
            else
            {
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
            } 
            mean_times /= forest_one_root(t, t); 

            return mean_times;  
        }

        Matrix<T, Dynamic, 1> getVarianceOfFirstPassageTimesFromRecurrence(Node* target,
                                                                           bool sparse,
                                                                           bool use_kahan_sum) 
        {
            /*
             * Return the mean first-passage time to the given target node from 
             * *every node* (including itself), assuming that the target node
             * will always eventually be reached, no matter which starting node
             * is chosen.  
             *
             * This quantity is only well-defined if there are no alternative 
             * terminal SCCs to which a path from the source node can travel.
             * In other words, for any node in the graph with a path from the
             * source node, there must also exist a path from that node to the
             * target node.  
             */
            // Compute the required spanning forest matrices ...
            Matrix<T, Dynamic, Dynamic> forest_one_root, forest_two_roots;
            if (sparse)
            {
                // Instantiate a *sparse row-major* Laplacian matrix ...
                //
                // Initialize a zero matrix with #rows = #cols = #nodes
                SparseMatrix<T, RowMajor> laplacian(this->numnodes, this->numnodes); 

                // Populate the entries of the matrix: the off-diagonal (i,j)-th  
                // entry is the *negative* of the label of the edge i -> j, and 
                // the diagonal entries are set so that each *row* sum is zero 
                std::vector<Triplet<T> > laplacian_triplets; 
                unsigned i = 0;
                for (auto&& v : this->order)
                {
                    std::vector<T> row_entries;    // All nonzero off-diagonal entries in i-th row
                    unsigned j = 0;
                    for (auto&& w : this->order)
                    {
                        if (i != j)
                        {
                            // Get the edge label for i -> j, omitting all edges 
                            // for which i is the target node 
                            if (v != target && this->edges[v].find(w) != this->edges[v].end())
                            {
                                T label(this->edges[v][w]);
                                laplacian_triplets.push_back(Triplet<T>(i, j, -label));
                                row_entries.push_back(label);  
                            }
                        }
                        j++;
                    }
                    // Compute the negative of the i-th row sum
                    T row_sum = 0; 
                    if (use_kahan_sum)
                    {
                        row_sum = kahanVectorSum(row_entries);
                    }
                    else 
                    {
                        for (const T entry : row_entries)
                            row_sum += entry; 
                    } 
                    laplacian_triplets.push_back(Triplet<T>(i, i, row_sum)); 
                    i++;
                }
                laplacian.setFromTriplets(laplacian_triplets.begin(), laplacian_triplets.end());

                // Then run the Chebotarev-Agaev recurrence to get the two-root
                // forest matrix 
                forest_two_roots = this->getSpanningForestMatrixSparse(
                    this->numnodes - 2, laplacian, use_kahan_sum
                );

                // Then run the Chebotarev-Agaev recurrence one more time to get 
                // the one-root forest (tree) matrix  
                forest_one_root = chebotarevAgaevRecurrence<T>(
                    laplacian, forest_two_roots, this->numnodes - 2, use_kahan_sum
                );
            }
            else 
            {
                // Instantiate a *dense* Laplacian matrix ...
                //
                // Initialize a zero matrix with #rows = #cols = #nodes
                Matrix<T, Dynamic, Dynamic> laplacian = Matrix<T, Dynamic, Dynamic>::Zero(this->numnodes, this->numnodes);

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
                            // for which i is the target node 
                            if (v != target && this->edges[v].find(w) != this->edges[v].end())
                            {
                                T label(this->edges[v][w]);
                                laplacian(i, j) = -label;
                            }
                        }
                        j++;
                    }
                    i++;
                }

                // Populate diagonal entries as negative sums of the off-diagonal
                // entries in each row
                if (use_kahan_sum)
                {
                    for (unsigned i = 0; i < this->numnodes; ++i)
                        laplacian(i, i) = -kahanRowSum(laplacian, i);
                }
                else
                { 
                    for (unsigned i = 0; i < this->numnodes; ++i)
                        laplacian(i, i) = -(laplacian.row(i).sum());
                }

                // Then run the Chebotarev-Agaev recurrence to get the two-root
                // forest matrix 
                forest_two_roots = this->getSpanningForestMatrix(
                    this->numnodes - 2, laplacian, use_kahan_sum
                );

                // Then run the Chebotarev-Agaev recurrence one more time to get 
                // the one-root forest (tree) matrix  
                forest_one_root = chebotarevAgaevRecurrence<T>(
                    laplacian, forest_two_roots, this->numnodes - 2, use_kahan_sum
                );
            }

            // Now compute the desired first moments (means) and second moments 
            // of the first-passage times ...
            int t;
            Matrix<T, Dynamic, 1> mean_times = Matrix<T, Dynamic, 1>::Zero(this->numnodes);
            Matrix<T, Dynamic, 1> second_moments = Matrix<T, Dynamic, 1>::Zero(this->numnodes);
            Matrix<T, Dynamic, Dynamic> forest_two_roots_squared; 
            if (use_kahan_sum)
                forest_two_roots_squared = kahanMultiply(forest_two_roots, forest_two_roots); 
            else 
                forest_two_roots_squared = forest_two_roots * forest_two_roots;
            for (auto it = this->order.begin(); it != this->order.end(); ++it)
            {
                if (*it == target)
                {
                    t = std::distance(this->order.begin(), it); 
                    break;
                }
            }
            int z = this->numnodes - 1 - t; 
            if (use_kahan_sum)
            {
                if (t == 0)
                {
                    mean_times.tail(z) = kahanRowSum(forest_two_roots.block(1, 1, z, z)); 
                    second_moments.tail(z) = kahanRowSum(forest_two_roots_squared.block(1, 1, z, z)); 
                }
                else if (z == 0)
                {
                    mean_times.head(t) = kahanRowSum(forest_two_roots.block(0, 0, t, t)); 
                    second_moments.head(t) = kahanRowSum(forest_two_roots_squared.block(0, 0, t, t)); 
                }
                else 
                { 
                    mean_times.head(t) = kahanRowSum(forest_two_roots.block(0, 0, t, t)); 
                    mean_times.head(t) += kahanRowSum(forest_two_roots.block(0, t + 1, t, z)); 
                    mean_times.tail(z) = kahanRowSum(forest_two_roots.block(t + 1, 0, z, t)); 
                    mean_times.tail(z) += kahanRowSum(forest_two_roots.block(t + 1, t + 1, z, z));
                    second_moments.head(t) = kahanRowSum(forest_two_roots_squared.block(0, 0, t, t)); 
                    second_moments.head(t) += kahanRowSum(forest_two_roots_squared.block(0, t + 1, t, z)); 
                    second_moments.tail(z) = kahanRowSum(forest_two_roots_squared.block(t + 1, 0, z, t)); 
                    second_moments.tail(z) += kahanRowSum(forest_two_roots_squared.block(t + 1, t + 1, z, z));
                } 
            }
            else
            {
                if (t == 0)
                {
                    mean_times.tail(z) = forest_two_roots.block(1, 1, z, z).rowwise().sum();  
                    second_moments.tail(z) = forest_two_roots_squared.block(1, 1, z, z).rowwise().sum(); 
                }
                else if (z == 0)
                {
                    mean_times.head(t) = forest_two_roots.block(0, 0, t, t).rowwise().sum(); 
                    second_moments.head(t) = forest_two_roots_squared.block(0, 0, t, t).rowwise().sum(); 
                }
                else 
                {
                    mean_times.head(t) = forest_two_roots.block(0, 0, t, t).rowwise().sum(); 
                    mean_times.head(t) += forest_two_roots.block(0, t + 1, t, z).rowwise().sum(); 
                    mean_times.tail(z) = forest_two_roots.block(t + 1, 0, z, t).rowwise().sum(); 
                    mean_times.tail(z) += forest_two_roots.block(t + 1, t + 1, z, z).rowwise().sum();
                    second_moments.head(t) = forest_two_roots_squared.block(0, 0, t, t).rowwise().sum(); 
                    second_moments.head(t) += forest_two_roots_squared.block(0, t + 1, t, z).rowwise().sum(); 
                    second_moments.tail(z) = forest_two_roots_squared.block(t + 1, 0, z, t).rowwise().sum(); 
                    second_moments.tail(z) += forest_two_roots_squared.block(t + 1, t + 1, z, z).rowwise().sum();
                } 
            } 
            mean_times /= forest_one_root(t, t);
            second_moments /= forest_one_root(t, t); 

            return second_moments - (mean_times.array() * mean_times.array()).matrix();  
        } 
};

#endif
