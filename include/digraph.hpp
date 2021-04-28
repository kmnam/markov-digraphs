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

/*
 * An implementation of a labeled directed graph.  
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     4/28/2021
 */
using namespace Eigen;

// ----------------------------------------------------- //
//                    LINEAR ALGEBRA                     //
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
Matrix<T, Dynamic, Dynamic> nullspaceSVD(const Ref<const Matrix<T, Dynamic, Dynamic> >& A, const T sv_tol)
{
    /*
     * Compute the nullspace of A by performing a singular value decomposition.
     *
     * This function does not check that the given matrix is indeed a
     * valid row Laplacian matrix (zero row sums, positive diagonal,
     * negative off-diagonal). 
     */
    // Perform a singular value decomposition of A, only computing V in full
    Eigen::BDCSVD<Matrix<T, Dynamic, Dynamic> > svd(A, Eigen::ComputeFullV);

    // Initialize nullspace basis matrix
    Matrix<T, Dynamic, Dynamic> nullmat;
    unsigned ncols = 0;
    unsigned nrows = A.cols();

    // Run through the singular values of A (in ascending, i.e., reverse order) ...
    Matrix<T, Dynamic, 1> S = svd.singularValues();
    Matrix<T, Dynamic, Dynamic> V = svd.matrixV();
    unsigned nsingvals = S.rows();
    unsigned j = nsingvals - 1;
    while (isclose<T>(S(j), 0.0, sv_tol) && j >= 0)
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
Matrix<T, Dynamic, Dynamic> chebotarevAgaevRecurrence(const Ref<const Matrix<T, Dynamic, Dynamic> >& laplacian,
                                                      const Ref<const Matrix<T, Dynamic, Dynamic> >& curr,
                                                      int k)
{
    /*
     * Apply one iteration of the recurrence of Chebotarev & Agaev (Lin Alg
     * Appl, 2002, Eqs. 17-18) for the in-forest matrices of the graph. 
     */
    T K(k + 1);
    Matrix<T, Dynamic, Dynamic> product = (-laplacian) * curr;
    T sigma = -product.trace() / K;
    Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(laplacian.rows(), laplacian.cols());
    return product + (sigma * identity);
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

        bool hasEdge(std::string source_id, std::string target_id) const
        {
            /*
             * Return true if the given edge exists in the graph. 
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

        Matrix<T, Dynamic, Dynamic> getSpanningForestMatrix(int k)
        {
            /*
             * Compute the k-th spanning forest matrix, using the recurrence
             * of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs. 17-18). 
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
                            laplacian(i,j) = -label;
                            if (laplacian(i,j) > 0)
                                throw std::runtime_error("Negative edge label found");
                        }
                    }
                    j++;
                }
                i++;
            }

            // Populate diagonal entries as negative sums of the off-diagonal
            // entries in each row
            for (unsigned i = 0; i < this->numnodes; ++i)
                laplacian(i,i) = -(laplacian.row(i).sum());

            // Apply the recurrence ...
            for (unsigned i = 0; i < k; ++i)
                curr = chebotarevAgaevRecurrence<T>(laplacian, curr, i);

            return curr; 
        }

        Matrix<T, Dynamic, 1> getSteadyStateFromSVD(T sv_tol)
        {
            /*
             * Return a vector of steady-state probabilities for the nodes,
             * according to the canonical ordering of nodes, by solving for
             * the nullspace of the Laplacian matrix.
             */
            Matrix<T, Dynamic, Dynamic> laplacian = this->getLaplacian();
            
            // Obtain the nullspace matrix of the Laplacian matrix
            Matrix<T, Dynamic, Dynamic> nullmat;
            try
            {
                nullmat = nullspaceSVD<T>(laplacian, sv_tol);
            }
            catch (const std::runtime_error& e)
            {
                throw;
            }

            // Each column is a nullspace basis vector (for each SCC) and
            // each row corresponds to a node in the graph
            Matrix<T, Dynamic, 1> steady_state = nullmat.array().rowwise().sum().matrix();
            
            return steady_state;
        }

        Matrix<T, Dynamic, 1> getSteadyStateFromRecurrence()
        {
            /*
             * Return a vector of steady-state probabilities for the nodes,
             * according to the given ordering of nodes, by the recurrence
             * relation of Chebotarev & Agaev for the k-th forest matrix
             * (Chebotarev & Agaev, Lin Alg Appl, 2002, Eqs. 17-18).
             */
            // Obtain the spanning tree weight matrix from the row Laplacian matrix
            Matrix<T, Dynamic, Dynamic> forest_matrix = this->getSpanningForestMatrix(this->numnodes - 1);

            // Return any row of the matrix 
            return forest_matrix.row(0); 
        }
};

#endif
