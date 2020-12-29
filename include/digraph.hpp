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
 *     12/29/2020
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
Matrix<T, Dynamic, 1> spanningTreeWeights(const Ref<const Matrix<T, Dynamic, Dynamic> >& laplacian)
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
        T K(k);   // Need to cast as type T (to accommodate boost::multiprecision types)
        T sigma = (laplacian * weights).trace() / K;
        weights = -laplacian * weights + sigma * identity;
    }

    // Return the row of the weight matrix whose product with the (negative
    // transpose of) the Laplacian matrix has the smallest norm
    Matrix<T, Dynamic, 1> norm = (weights * (-laplacian)).rowwise().norm();
    unsigned min_i = 0;
    for (unsigned i = 1; i < norm.size(); ++i)
    {
        if (norm(i) < norm(min_i)) min_i = i;
    }
    return weights.row(min_i);
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

// ------------------------------------------------------- //
//                     HELPER FUNCTIONS                    //
// ------------------------------------------------------- //
bool isBackEdge(const std::vector<Edge>& tree, const Edge& edge)
{
    /*
     * Determines whether the given edge is a "back-edge" with respect to a
     * given tree (i.e., if its target vertex has a path to its source vertex
     * in the tree).
     */
    Node* source = edge.first;
    Node* target = edge.second;

    // Try to draw a path of edges in the tree from target to source
    Node* curr = target;
    bool at_root = false;
    while (!at_root)
    {
        // Find the unique edge in the tree leaving curr 
        auto found = std::find_if(
            tree.begin(), tree.end(),
            [curr](const Edge& e){ return (curr == e.first); }
        );

        // If this edge doesn't exist, then we have reached the root
        if (found == tree.end()) at_root = true;
        else
        {
            // If this edge does exist, check that its destination is 
            // the desired source vertex 
            curr = found->second;
            if (curr == source) return true;
        }
    }
    return false;
}

void enumSpanningInTreesIter(const std::vector<Edge>& parent_tree,
                             const std::vector<Edge>& tree0,
                             const std::vector<Edge>& edges,
                             std::vector<std::vector<Edge> >& trees)
{
    /*
     * Recursive function to be called as part of Uno's algorithm.
     */
    // Define function that returns, given a (in-)tree and a vertex, the edge 
    // with that vertex as its source
    std::function<Edge(const std::vector<Edge>&, const Node*)> get_leaving_edge
        = [](const std::vector<Edge>& tree, const Node* node)
    {
        return *std::find_if(
            tree.begin(), tree.end(),
            [node](const Edge& e){ return (node == e.first); }
        );
    };

    // Find the minimal edge that is in tree0 and not in the parent tree
    // (An edge in the graph is *valid* if less than this minimal edge)
    Edge min_edge_not_in_parent_tree;
    auto ite = std::find_if(
        edges.begin(), edges.end(), [tree0, parent_tree](const Edge& e)
        {
            return (
                std::find(tree0.begin(), tree0.end(), e) != tree0.end() &&
                std::find(parent_tree.begin(), parent_tree.end(), e) == parent_tree.end()
            );
        }
    );
    if (ite == edges.end())
        min_edge_not_in_parent_tree = std::make_pair(nullptr, nullptr);
    else
        min_edge_not_in_parent_tree = *ite;
    
    // Find each valid non-back-edge with respect to the parent tree
    for (auto it = edges.begin(); it != ite; ++it)    // Iterate over only valid edges 
    {
        Edge curr_valid_edge = *it;

        // Is this valid edge (1) not in the parent tree, and (2) a non-back-edge
        // with respect to the parent tree?
        if (std::find(parent_tree.begin(), parent_tree.end(), curr_valid_edge) == parent_tree.end()
            && !isBackEdge(parent_tree, curr_valid_edge))
        {
            // Add the non-back-edge to the parent tree and remove the edge
            // with the same source vertex as the non-back-edge, to obtain
            // the *child tree*
            Edge to_be_removed = get_leaving_edge(parent_tree, curr_valid_edge.first);
            auto it = std::find(parent_tree.begin(), parent_tree.end(), to_be_removed);
            unsigned i = it - parent_tree.begin();
            std::vector<Edge> child_tree(parent_tree);
            child_tree[i] = curr_valid_edge;

            // Add the child tree and call the function recursively on
            // the child tree
            trees.push_back(child_tree);
            enumSpanningInTreesIter(child_tree, tree0, edges, trees);
        }
    }
}

using namespace Eigen;

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
        // Dictionary mapping string (ids) to Node pointers 
        std::unordered_map<std::string, Node*> nodes;

        // Maintain edges in a nested dictionary of adjacency "dictionaries"  
        std::unordered_map<Node*, std::unordered_map<Node*, T> > edges;

        // ----------------------------------------------------- //
        //                     PRIVATE METHODS                   //
        // ----------------------------------------------------- //
        template <typename U = T>
        Matrix<U, Dynamic, Dynamic> getLaplacian(std::vector<Node*> nodes)
        {
            /*
             * Return a numerical Laplacian matrix with the given scalar type,
             * according to the given node ordering.
             *
             * The ordering is assumed to include every node in the graph once.
             */
            // Initialize a zero matrix with #rows = #cols = #nodes
            unsigned dim = nodes.size();
            Matrix<U, Dynamic, Dynamic> laplacian = Matrix<U, Dynamic, Dynamic>::Zero(dim, dim);

            // Populate the off-diagonal entries of the matrix first: 
            // (i,j)-th entry is the label of the edge j -> i
            unsigned i = 0;
            for (auto&& v : nodes)
            {
                unsigned j = 0;
                for (auto&& w : nodes)
                {
                    if (i != j)
                    {
                        // Get the edge label for j -> i
                        if (this->edges[w].find(v) != this->edges[w].end())
                        {
                            U label(this->edges[w][v]);
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
            for (unsigned i = 0; i < dim; ++i)
                laplacian(i,i) = -(laplacian.col(i).sum());

            return laplacian;
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

        std::pair<std::vector<Edge>, std::vector<Node*> > getSpanningInTreeFromDFS(Node* root)
        {
            /*
             * Obtain a vector of edges in a single spanning in-tree, with  
             * the edges ordered via a topological sort on the second
             * index (i.e., for any two edges (i, j) and (k, l) in the
             * vector such that (j, l) is an edge, (i, j) precedes (k, l)).
             */
            std::vector<Edge> tree;
            std::stack<Node*> stack;
            std::unordered_set<Node*> visited;
            std::vector<Node*> traversal;

            // Initiate depth-first search from the root
            visited.insert(root);
            stack.push(root);
            traversal.push_back(root);

            // Run until stack is empty
            while (!stack.empty())
            {
                // Pop topmost node from the stack
                Node* curr = stack.top();
                stack.pop();

                // Find all nodes that have an edge *leading to* curr
                std::vector<Node*> neighbors;
                for (auto&& edge_set : this->edges)
                {
                    if (edge_set.second.find(curr) != edge_set.second.end())
                        neighbors.push_back(edge_set.first);
                }

                // For each such source node, if not visited already: 
                // mark as visited, add onto stack, and add the edge to curr 
                // onto the tree 
                for (auto&& neighbor : neighbors)
                {
                    if (visited.find(neighbor) == visited.end())
                    {
                        tree.push_back(std::make_pair(neighbor, curr));
                        stack.push(neighbor);
                        visited.insert(neighbor);
                        traversal.push_back(neighbor);
                    }
                }
            }

            return std::make_pair(tree, traversal);
        }

        std::pair<std::vector<Edge>, std::vector<Node*> > getSpanningOutTreeFromDFS(Node* root)
        {
            /*
             * Obtain a vector of edges in a single spanning out-tree, with  
             * the edges ordered via a topological sort on the second
             * index (i.e., for any two edges (i, j) and (k, l) in the
             * vector such that (j, l) is an edge, (i, j) precedes (k, l)).
             */
            std::vector<Edge> tree;
            std::stack<Node*> stack;
            std::unordered_set<Node*> visited;
            std::vector<Node*> traversal;

            // Initiate depth-first search from the root
            visited.insert(root);
            stack.push(root);
            traversal.push_back(root);

            // Run until stack is empty
            while (!stack.empty())
            {
                // Pop topmost vertex from the stack
                Node* curr = stack.top();
                stack.pop();

                // Simply run through the unvisited neighbors of curr,
                // marking each as visited, pushing onto stack, and adding
                // the edge from curr to the tree
                //
                // (Note that this tree is an *out-tree*, and so each vertex 
                // has one edge coming into it, not one edge leaving it)
                for (auto&& dest : this->edges[curr])
                {
                    Node* neighbor = dest.first;
                    if (visited.find(neighbor) == visited.end())
                    {
                        tree.push_back(std::make_pair(curr, neighbor));
                        stack.push(neighbor);
                        visited.insert(neighbor);
                        traversal.push_back(neighbor);
                    }
                }
            }

            return std::make_pair(tree, traversal);
        }

        std::vector<std::vector<Edge> > enumSpanningInTrees(Node* root)
        {
            /*
             * Perform Uno's algorithm for enumerating the spanning trees 
             * rooted at the given node. 
             */
            // Initialize a vector of spanning trees
            std::vector<std::vector<Edge> > trees; 

            // Get a DFS spanning tree starting at the root vertex
            std::pair<std::vector<Edge>, std::vector<Node*> > tree_dfs = this->getSpanningInTreeFromDFS(root);
            trees.push_back(tree_dfs.first);

            // Get all the edges in the graph and sort them lexicographically
            // with respect to the new (DFS) node order 
            std::vector<std::pair<Edge, T> > edges = this->getEdges();
            if (tree_dfs.second.size() == 1)    // If the graph has a single vertex, return the empty tree
            {
                return trees;
            }
            std::sort(
                edges.begin(), edges.end(),
                [tree_dfs](const std::pair<Edge, T>& left, const std::pair<Edge, T>& right)
                {
                    Node* lv = left.first.first;
                    Node* rv = right.first.first;

                    // Is the source vertex in left less than the source vertex
                    // in right, according to the DFS order?
                    bool lt = (
                        std::find(tree_dfs.second.begin(), tree_dfs.second.end(), lv)
                        < std::find(tree_dfs.second.begin(), tree_dfs.second.end(), rv)
                    );
                    if (lt) return true;
                    // Otherwise, if the source vertices are the same, what about
                    // the target vertices?
                    else if (lv == rv)
                    {
                        Node* lw = left.first.second;
                        Node* rw = left.first.second;
                        lt = (
                            std::find(tree_dfs.second.begin(), tree_dfs.second.end(), lw)
                            < std::find(tree_dfs.second.begin(), tree_dfs.second.end(), rw)
                        );
                        return lt;
                    }
                    return false;   // Otherwise, return false
                }
            );
            std::vector<Edge> edges_without_labels;
            for (auto&& edge : edges)
                edges_without_labels.push_back(edge.first);

            // Call recursive function
            enumSpanningInTreesIter(tree_dfs.first, tree_dfs.first, edges_without_labels, trees);

            return trees;
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

        std::pair<std::vector<std::unordered_set<Node*> >,
                  std::vector<std::unordered_map<Node*, std::vector<std::vector<Edge> > > > >
            enumTerminalSpanningInTrees()
        {
            /*
             * Perform Uno's algorithm to enumerate all the spanning (in-)trees
             * of each terminal strongly connected component in the graph. 
             */
            // Run Tarjan's algorithm to get the SCCs of the graph
            std::vector<std::unordered_set<Node*> > components = this->enumStronglyConnectedComponents();

            // Maintain a vector of dictionaries of vectors of spanning trees:
            // - all_trees[i] is a dictionary containing all spanning trees 
            //   of the terminal SCC given by components[i]
            // - all_trees[i] is empty if components[i] is not terminal
            // - all_trees[i][node] is a vector of spanning trees of the 
            //   spanning trees of components[i] rooted at node
            std::vector<std::unordered_map<Node*, std::vector<std::vector<Edge> > > > all_trees;

            // Run Uno's algorithm on each node in each terminal SCC
            for (unsigned i = 0; i < components.size(); ++i)
            {
                std::unordered_set<Node*> component = components[i];
                all_trees.emplace_back(std::unordered_map<Node*, std::vector<std::vector<Edge> > >());

                // Is the current SCC terminal?
                if (this->isTerminal(*component.begin(), components, i))
                {
                    // If so, run through the nodes in the SCC and get spanning
                    // trees rooted at each node
                    LabeledDigraph<T>* subgraph = this->subgraph(component);
                    std::vector<Node*> subnodes = subgraph->getNodes();
                    std::vector<std::pair<Edge, T> > subedges = subgraph->getEdges();
                    for (auto&& root : component)
                    {
                        // IMPORTANT: Find the pointer to the root node in
                        // the *induced subgraph*
                        Node* subroot = subgraph->getNode(root->id);
                        std::vector<std::vector<Edge> > subtrees = subgraph->enumSpanningInTrees(subroot);
                        
                        // IMPORTANT: Then find the pointer to each node in 
                        // each tree in the *original graph*
                        std::vector<std::vector<Edge> > trees;
                        for (auto&& subtree : subtrees)
                        {
                            std::vector<Edge> tree;
                            for (auto&& subedge : subtree)
                            {
                                Edge edge = std::make_pair(this->getNode((subedge.first)->id), this->getNode((subedge.second)->id)); 
                                tree.push_back(edge);
                            }
                            trees.push_back(tree);
                        }
                    }
                    
                    delete subgraph;
                }
            } 

            return std::make_pair(components, all_trees);
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
            this->nodes[id] = node;
            this->edges.emplace(node, std::unordered_map<Node*, T>());
            return node;
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

        std::vector<Node*> getNodes() const
        {
            /*
             * Return a vector of the node pointers in arbitrary order.  
             */
            std::vector<Node*> nodes;
            for (auto&& edge_set : this->edges) nodes.push_back(edge_set.first);
            return nodes; 
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

        std::vector<std::pair<Edge, T> > getEdges() const
        {
            /*
             * Return a vector of all the edges in the graph, along with the 
             * edge labels. 
             */
            std::vector<std::pair<Edge, T> > edges;
            for (auto&& edge_set : this->edges)
            {
                for (auto&& dest : edge_set.second)
                {
                    Edge edge = std::make_pair(edge_set.first, dest.first);
                    edges.push_back(std::make_pair(edge, dest.second));
                }
            }
            return edges;
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

        template <typename U>
        LabeledDigraph<U>* copy() const
        {
            /*
             * Return pointer to a new LabeledDigraph object, possibly
             * with a different scalar type, with the same graph structure
             * and edge label values.
             */
            LabeledDigraph<U>* graph = new LabeledDigraph<U>();

            // Copy over nodes with the same ids
            for (auto&& node : this->nodes)
                graph->addNode(node.first);
            
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

        template <typename U>
        void copy(LabeledDigraph<U>* graph) const
        {
            /*
             * Given pointer to an existing LabeledDigraph object, possibly
             * with a different scalar type, copy over the graph details.  
             */
            // Clear the input graph's contents
            graph->clear();

            // Copy over nodes with the same IDs
            for (auto&& node : this->nodes)
                graph->addNode(node.first);

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

        template <typename U = T>
        Matrix<U, Dynamic, 1> getSteadyStateFromSVD(std::vector<Node*> nodes, U sv_tol)
        {
            /*
             * Return a vector of steady-state probabilities for the nodes,
             * according to the given ordering of nodes, by solving for the
             * nullspace of the Laplacian matrix.
             */
            Matrix<U, Dynamic, Dynamic> laplacian = this->getLaplacian<U>(nodes);
            
            // Obtain the nullspace matrix of the Laplacian matrix
            Matrix<U, Dynamic, Dynamic> nullmat;
            try
            {
                nullmat = nullspaceSVD<U>(laplacian, sv_tol);
            }
            catch (const std::runtime_error& e)
            {
                throw;
            }

            // Each column is a nullspace basis vector (for each SCC) and
            // each row corresponds to a node in the graph
            Matrix<U, Dynamic, 1> steady_state = nullmat.array().rowwise().sum().matrix();
            
            return steady_state;
        }

        template <typename U = T>
        Matrix<U, Dynamic, 1> getSteadyStateFromRecurrence(std::vector<Node*> nodes)
        {
            /*
             * Return a vector of steady-state probabilities for the nodes,
             * according to the given ordering of nodes, by the recurrence
             * relation of Chebotarev & Agaev for the k-th forest matrix
             * (Chebotarev & Agaev, Lin Alg Appl, 2002, Eqs. 17-18).
             */
            // Get the row Laplacian matrix
            Matrix<U, Dynamic, Dynamic> laplacian = (-this->getLaplacian<U>(nodes)).transpose();

            // Obtain the spanning tree weight vector from the row Laplacian matrix
            Matrix<U, Dynamic, 1> steady_state;
            try
            {
                steady_state = spanningTreeWeights<U>(laplacian);
            }
            catch (const std::runtime_error& e)
            {
                throw;
            }
            
            return steady_state; 
        }

        template <typename U = T>
        Matrix<U, Dynamic, 1> getSteadyStateFromTrees(std::vector<Node*> nodes)
        {
            /*
             * Return a vector of steady-state probabilities for the nodes, 
             * according to the given ordering of nodes, by enumerating the 
             * spanning trees of the terminal strongly connected components
             * of the graph. 
             */
            // Enumerate the spanning trees of the terminal SCCs of the graph
            auto data = this->enumTerminalSpanningInTrees();
            std::vector<std::unordered_set<Node*> > components = data.first;
            std::vector<std::unordered_map<Node*, std::vector<std::vector<Edge> > > > trees = data.second;
            
            // For each node in the ordering, find the SCC it lies within, and 
            // get the sum of the weights of the spanning trees given by 
            // the corresponding dictionary
            Matrix<U, Dynamic, 1> steady_state = Matrix<U, Dynamic, 1>::Zero(nodes.size());
            for (unsigned i = 0; i < nodes.size(); ++i)
            {
                // What SCC does the node lie in?
                auto it = std::find_if(
                    components.begin(), components.end(), [nodes, i](const std::unordered_set<Node*> component)
                    {
                        return (component.find(nodes[i]) != component.end());
                    }
                );

                // If the SCC is terminal, find all the spanning trees of the
                // SCC rooted at the node
                unsigned component_index = it - components.begin();
                if (trees[component_index].size() > 0)
                {
                    U total_weight = 0;
                    for (auto&& tree : trees[component_index][nodes[i]])
                    {
                        U tree_weight = 1;
                        for (auto&& edge : tree)
                            tree_weight *= this->edges[edge.first][edge.second];
                        total_weight += tree_weight;
                    }
                    steady_state[i] = total_weight;
                }
            }

            return steady_state;
        }
};

#endif
