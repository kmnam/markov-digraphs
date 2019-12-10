#ifndef MARKOV_DIGRAPHS_HPP
#define MARKOV_DIGRAPHS_HPP

#include <cmath>
#include <vector>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <boost/container_hash/hash.hpp>
#include "linalg.hpp"

/*
 * An implementation of a digraph associated with a continuous-time 
 * finite-state Markov process.  
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/9/2019
 */

// ----------------------------------------------------- //
//                    NODES AND EDGES                    //
// ----------------------------------------------------- //
template <typename T>
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
};

namespace std {

template <typename T>
struct hash<Node<T> >
{
    std::size_t operator()(const Node<T>& node) const noexcept
    {
        return std::hash<std::string>{}(node.id);
    }
};

}   // namespace std

// An Edge is simply a pair of Node pointers
template <typename T>
using Edge = std::pair<Node<T>*, Node<T>*>;

// Pairs of strings are used for outputting edges
typedef std::pair<std::string, std::string> StrPair;

// ------------------------------------------------------- //
//                     HELPER FUNCTIONS                    //
// ------------------------------------------------------- //
template <typename T>
bool isBackEdge(const std::vector<Edge<T> >& tree, const Edge<T>& edge)
{
    /*
     * Determines whether the given edge is a "back edge" with respect to a
     * given tree (i.e., if its target vertex has a path to its source vertex
     * in the tree).
     */
    Node<T>* source = edge.first;
    Node<T>* target = edge.second;

    // Try to draw a path of edges in the tree from target to source
    Node<T>* curr = target;
    bool at_root = false;
    while (!at_root)
    {
        // Get the unique edge in the tree leaving curr 
        auto found = std::find_if(
            tree.begin(), tree.end(),
            [curr](const Edge<T>& e){ return (curr == e.first); }
        );
        if (found == tree.end()) at_root = true;
        else
        {
            curr = found->second;
            if (curr == source) return true;
        }
    }
    return false;
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
    powerset.emplace_back(std::vector<T>());

    // Run through all k-combinations for increasing k
    std::vector<std::vector<T> > k_combinations;
    for (unsigned k = 1; k <= data.size(); k++)
    {
        k_combinations = combinations<T>(data, k);
        powerset.insert(
            powerset.end(),
            std::make_move_iterator(k_combinations.begin()),
            std::make_move_iterator(k_combinations.end())
        );
    }

    return powerset;
}

template <typename T>
void enumSpanningTreesIter(const std::vector<Edge<T> >& parent_tree,
                           const std::vector<Edge<T> >& tree0,
                           const std::vector<Edge<T> >& edges,
                           const std::vector<Edge<T> >& back_edges,
                           const std::vector<Edge<T> >& nonback_edges,
                           const std::vector<Node<T>*>& order, 
                           std::vector<std::vector<StrPair> >& trees)
{
    /*
     * Recursive function to be called as part of Uno's algorithm. 
     */
    // Define function for obtaining the edge in a given tree with source
    // given by the input vertex
    std::function<Edge<T>(const std::vector<Edge<T> >&, const Node<T>*)> edge_with_source
        = [](const std::vector<Edge<T> >& tree, const Node<T>* node)
    {
        return *std::find_if(
            tree.begin(), tree.end(),
            [node](const Edge<T>& e){ return (node == e.first); }
        );
    };

    // Find the least edge (with respect to source index) that is in 
    // tree0 and not in the parent tree
    Edge<T> min_edge_not_in_parent_tree;
    auto ite = std::find_if(
        edges.begin(), edges.end(), [tree0, parent_tree](const Edge<T>& e)
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
    
    // Find each valid non-back edge with respect to the parent tree 
    for (auto&& e : nonback_edges)
    {
        if (min_edge_not_in_parent_tree.first == nullptr ||
            std::find(order.begin(), order.end(), e.first) < std::find(order.begin(), order.end(), min_edge_not_in_parent_tree.first))
        {
            // Add the nonback edge and find the edge with the source vertex 
            // of the nonback edge
            Edge<T> to_be_removed = edge_with_source(parent_tree, e.first);
            auto it = std::find(parent_tree.begin(), parent_tree.end(), to_be_removed);
            unsigned i = it - parent_tree.begin();
            std::vector<Edge<T> > child_tree(parent_tree);
            child_tree[i] = e;

            // Classify all edges *not* in the new child tree as either back
            // or non-back edges with respect to the child tree
            std::vector<Edge<T> > new_back_edges;
            std::vector<Edge<T> > new_nonback_edges;
            for (auto&& e : edges)
            {
                if (std::find(child_tree.begin(), child_tree.end(), e) == child_tree.end())
                {
                    if (isBackEdge<T>(child_tree, e)) new_back_edges.push_back(e);
                    else                              new_nonback_edges.push_back(e);
                }
                // Note that these edge sets should already be sorted with 
                // respect to source index 
            }

            // Add the child tree and call the function recursively on the child tree
            std::vector<StrPair> child_tree_out;
            for (auto&& e : child_tree) 
            {
                std::string v(e.first->id);
                std::string w(e.second->id);
                child_tree_out.push_back(std::make_pair(v, w));
            }
            trees.push_back(child_tree_out);
            enumSpanningTreesIter(
                child_tree, tree0, edges, new_back_edges, new_nonback_edges, order, trees
            );
        }
    }
}

using namespace Eigen;

template <typename T>
class MarkovDigraph
{
    /*
     * An implementation of a labeled digraph associated with a Markov process. 
     */
    protected:
        // ----------------------------------------------------- //
        //                       ATTRIBUTES                      //
        // ----------------------------------------------------- //
        // All nodes in the graph, stored as an ordered vector
        std::vector<Node<T>*> nodes;

        // All edges in the graph, stored as "adjacency sets"
        std::unordered_map<Node<T>*, std::unordered_set<Node<T>*> > edges;

        // All edge labels in the graph, stored as a hash table    
        std::unordered_map<Edge<T>, T, boost::hash<Edge<T> > > labels;

    public:
        MarkovDigraph()
        {
            /*
             * Empty constructor.
             */
        }

        ~MarkovDigraph()
        {
            /*
             * Destructor; de-allocates each microstate from heap memory.
             */
            for (auto&& node : this->nodes) delete node;
        }

        // ----------------------------------------------------- //
        //              NODE-ADDING/GETTING METHODS              //
        // ----------------------------------------------------- //
        Node<T>* addNode(std::string id)
        {
            /*
             * Add a node to the graph with the given ID, and return pointer
             * to the new node. Throw std::runtime_error if node with given ID
             * already exists.
             */
            for (auto&& node : this->nodes)
            {
                if (!id.compare(node->id))
                    throw std::runtime_error("Node exists with specified ID");
            }
            Node<T>* node = new Node<T>(id);
            this->nodes.push_back(node);
            this->edges.emplace(node, std::unordered_set<Node<T>*>());
            return node;
        }

        Node<T>* getNode(std::string id) const
        {
            /*
             * Return pointer to node with given ID.
             */
            for (auto&& node : this->nodes)
            {
                if (!id.compare(node->id))
                    return node;
            }
            return nullptr;    // Return nullptr if no matching node exists
        }

        std::vector<Node<T>*> getNodes() const
        {
            /*
             * Return the vector of nodes. 
             */
            return this->nodes; 
        }

        // ----------------------------------------------------- //
        //              EDGE-ADDING/GETTING METHODS              //
        // ----------------------------------------------------- //
        void addEdge(std::string source_id, std::string target_id, T label = 1.0)
        {
            /*
             * Add an edge between two nodes. If either ID does not 
             * correspond to a node in the graph, instantiate them.
             * Return the instantiated edge. 
             */
            Node<T>* source = this->getNode(source_id);
            Node<T>* target = this->getNode(target_id);
            if (source == nullptr)
                source = this->addNode(source_id);
            if (target == nullptr)
                target = this->addNode(target_id);
            Edge<T> edge = std::make_pair(source, target);
            this->edges[source].insert(target);
            this->labels[edge] = label;
        }

        Edge<T> getEdge(std::string source_id, std::string target_id) const
        {
            /*
             * Return edge between the specified nodes.
             */
            Node<T>* source = this->getNode(source_id);
            Node<T>* target = this->getNode(target_id);
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist");
            if (target == nullptr)
                throw std::runtime_error("Specified target node does not exist");
            auto it = this->edges.find(source)->second.find(target);
            if (it != this->edges.find(source)->second.end())
                return std::make_pair(source, *it);
            else    // Return pair of nullptrs if no edge exists
                return std::make_pair(nullptr, nullptr);
        }

        std::vector<Edge<T> > getEdges() const
        {
            /*
             * Return a vector of all the edges in the graph.
             */
            std::vector<Edge<T> > edges;
            for (auto&& v : this->nodes)
            {
                for (auto&& w : this->edges.find(v)->second)
                {
                    Edge<T> e = std::make_pair(v, w);
                    edges.push_back(e);
                }
            }
            return edges;
        }

        T getEdgeLabel(std::string source_id, std::string target_id)
        {
            /*
             * Get the current numerical value of the given edge label.
             * Return zero if edge does not exist. 
             */
            Node<T>* source = this->getNode(source_id);
            Node<T>* target = this->getNode(target_id);
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist");
            if (target == nullptr)
                throw std::runtime_error("Specified target node does not exist");
            auto it = this->edges[source].find(target);
            if (it != this->edges[source].end())
            {
                Edge<T> edge = std::make_pair(source, *it);
                return this->labels[edge];
            }
            else return 0.0;
        }

        std::unordered_map<Edge<T>, T, boost::hash<Edge<T> > > getEdgeLabels()
        {
            /*
             * Return a copy of the edge label data. 
             */
            std::unordered_map<Edge<T>, T, boost::hash<Edge<T> > > labels;
            labels = this->labels;
            return labels;
        }

        void setEdgeLabel(std::string source_id, std::string target_id, T value)
        {
            /*
             * Update the numerical value of the given edge label. 
             * Throw std::runtime_error if either node or the edge 
             * does not exist.
             */
            Node<T>* source = this->getNode(source_id);
            Node<T>* target = this->getNode(target_id);
            if (source == nullptr)
                throw std::runtime_error("Specified source node does not exist");
            if (target == nullptr)
                throw std::runtime_error("Specified target node does not exist");
            auto it = this->edges[source].find(target);
            if (it != this->edges[source].end())
            {
                Edge<T> edge = std::make_pair(source, *it);
                this->labels[edge] = value;
            }
            else
                throw std::runtime_error("Specified edge does not exist");
        }

        // ----------------------------------------------------- //
        //                     OTHER METHODS                     //
        // ----------------------------------------------------- //
        MarkovDigraph<T>* subgraph(std::vector<Node<T>*> nodes) const
        {
            /*
             * Return the subgraph induced by the given vector of nodes.
             */
            MarkovDigraph<T>* graph = new MarkovDigraph<T>();

            // Add each edge which lies between two of the given nodes
            // to the subgraph
            for (auto&& v : nodes)
            {
                try    // Try adding the node (will throw runtime_error if it already was added)
                {
                    graph->addNode(v->id);
                }
                catch (const std::runtime_error& e) { }
                for (auto&& w : this->edges.find(v)->second)
                {
                    if (std::find(nodes.begin(), nodes.end(), w) != nodes.end())
                    {
                        Edge<T> edge = std::make_pair(v, w);
                        graph->addEdge(v->id, w->id, this->labels.find(edge)->second);
                    }
                }
            }

            return graph; 
        }
        
        void clear()
        {
            /* 
             * Clear the graph's contents.
             */
            // De-allocate all Node instances from heap memory
            for (auto&& node : this->nodes) delete node;

            // Clear all attributes
            this->nodes.clear();
            this->edges.clear();
            this->labels.clear();
        }

        template <typename U>
        MarkovDigraph<U>* copy() const
        {
            /*
             * Return pointer to a new MarkovDigraph object, possibly
             * with a different scalar type, with the same graph structure
             * and edge label values.
             */
            MarkovDigraph<U>* graph = new MarkovDigraph<U>();

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

            return graph;
        }

        template <typename U>
        void copy(MarkovDigraph<U>* graph) const
        {
            /*
             * Given pointer to an existing MarkovDigraph object, possibly
             * with a different scalar type, copy over the graph details.  
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
        }

        Matrix<T, Dynamic, Dynamic> getLaplacian()
        {
            /*
             * Return a numerical Laplacian matrix, according to the 
             * currently stored values for the edge labels.
             */
            // Initialize a zero matrix with #rows = #cols = #nodes
            unsigned dim = this->nodes.size();
            Matrix<T, Dynamic, Dynamic> laplacian = Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);

            // Populate the off-diagonal entries of the matrix first: 
            // (i,j)-th entry is the label of the edge j -> i
            for (unsigned i = 0; i < dim; ++i)
            {
                for (unsigned j = 0; j < dim; ++j)
                {
                    if (i != j)
                    {
                        Node<T>* source = this->nodes[j];
                        Node<T>* target = this->nodes[i];

                        // Look for value specified for edge j -> i
                        auto it = this->labels.find(std::make_pair(source, target));
                        if (it != this->labels.end())
                        {
                            laplacian(i,j) = it->second;
                            if (laplacian(i,j) <= 0)
                                throw std::runtime_error("Non-positive value specified for edge");
                        }
                    }
                }
            }

            // Populate diagonal entries as negative sums of the
            // off-diagonal entries of each column
            for (unsigned i = 0; i < dim; ++i)
                laplacian(i,i) = -(laplacian.col(i).sum());

            return laplacian;
        }

        void setRatesFromLaplacian(Matrix<T, Dynamic, Dynamic> laplacian)
        {
            /*
             * Given a Laplacian matrix of the appropriate size, set the 
             * edge labels in the graph accordingly. 
             */
            unsigned dim = this->nodes.size();
            for (unsigned j = 0; j < dim; ++j)
            {
                for (unsigned k = 0; k < dim; ++k)
                {
                    this->setEdgeLabel(this->nodes[j]->id, this->nodes[k]->id, laplacian(k,j)); 
                }
            }
        }

        std::vector<Edge<T> > getSpanningTreeFromDFS(std::string id)
        {
            /*
             * Obtain a vector of edges in a single spanning in-tree, with  
             * the edges ordered via a topological sort on the second
             * index (i.e., for any two edges (i, j) and (k, l) in the
             * vector such that (j, l) is an edge, (i, j) precedes (k, l)).
             */
            std::vector<Edge<T> > tree;
            std::stack<Node<T>*> stack;
            std::unordered_set<Node<T>*> visited;

            // Initiate depth-first search from the root
            Node<T>* root = this->getNode(id);
            visited.insert(root);
            stack.push(root);

            // Run until stack is empty
            while (!stack.empty())
            {
                // Pop topmost vertex from the stack ...
                Node<T>* curr = stack.top();
                stack.pop();

                // ... and push its unvisited neighbors onto the stack
                std::vector<Node<T>*> neighbors;
                for (auto&& v : this->edges)
                {
                    if (std::find(v.second.begin(), v.second.end(), curr) != v.second.end())
                        neighbors.push_back(v.first);
                }
                for (auto&& neighbor : neighbors)
                {
                    if (visited.find(neighbor) == visited.end())
                    {
                        tree.push_back(std::make_pair(neighbor, curr));
                        stack.push(neighbor);
                        visited.insert(neighbor);
                    }
                }
            }

            return tree;
        }

        std::vector<Edge<T> > getOutwardTreeFromDFS(Node<T>* root)
        {
            /*
             * Obtain a vector of edges in a single spanning out-tree, with  
             * the edges ordered via a topological sort on the second
             * index (i.e., for any two edges (i, j) and (k, l) in the
             * vector such that (j, l) is an edge, (i, j) precedes (k, l)).
             */
            std::vector<Edge<T> > tree;
            std::stack<Node<T>*> stack;
            std::unordered_set<Node<T>*> visited;

            // Initiate depth-first search from the root
            visited.insert(root);
            stack.push(root);

            // Run until stack is empty
            while (!stack.empty())
            {
                // Pop topmost vertex from the stack ...
                Node<T>* curr = stack.top();
                stack.pop();

                // ... and push its unvisited neighbors onto the stack
                for (auto&& neighbor : this->edges[curr])
                {
                    if (visited.find(neighbor) == visited.end())
                    {
                        tree.push_back(std::make_pair(curr, neighbor));
                        stack.push(neighbor);
                        visited.insert(neighbor);
                    }
                }
            }

            return tree;
        }

        std::vector<std::vector<StrPair> > enumSpanningTrees(std::string id)
        {
            /*
             * Perform Uno's algorithm for enumerating the spanning trees 
             * rooted at the given node. 
             */
            // Initialize a vector of spanning trees
            std::vector<std::vector<StrPair> > trees; 

            // Get a DFS spanning tree at the root vertex
            Node<T>* root = this->getNode(id);
            std::vector<Edge<T> > tree0 = this->getSpanningTreeFromDFS(id);
            std::vector<StrPair> tree0_out;
            for (auto&& e : tree0)
            {
                std::string v(e.first->id);
                std::string w(e.second->id);
                tree0_out.push_back(std::make_pair(v, w));
            }
            trees.push_back(tree0_out);

            // Order the nodes with respect to the DFS traversal 
            std::vector<Node<T>*> order;
            order.push_back(root);
            for (auto&& e : tree0) order.push_back(e.second);

            // Get all the edges in the graph and sort them by w.r.t to 
            // DFS traversal 
            std::vector<Edge<T> > edges = this->getEdges();
            if (order.size() == 1)    // If the graph has a single vertex, return the empty tree
            {
                return trees;
            }
            std::sort(
                edges.begin(), edges.end(), [order](const Edge<T>& left, const Edge<T>& right)
                {
                    return std::find(order.begin(), order.end(), left.first) < std::find(order.begin(), order.end(), right.first);
                }
            );

            // Classify all edges *not* in tree0 as either back or non-back 
            // edges with respect to tree0 
            std::vector<Edge<T> > back_edges;
            std::vector<Edge<T> > nonback_edges;
            for (auto&& e : edges)
            {
                if (std::find(tree0.begin(), tree0.end(), e) == tree0.end())
                {
                    if (isBackEdge<T>(tree0, e)) back_edges.push_back(e);
                    else                         nonback_edges.push_back(e);
                }
                // Note that these edge sets should already be sorted with 
                // respect to source index 
            }

            // Call recursive function
            enumSpanningTreesIter(
                tree0, tree0, edges, back_edges, nonback_edges, order, trees
            );

            return trees;
        }

        std::vector<std::vector<StrPair> > enumAllSpanningTrees()
        {
            /*
             * Perform Uno's algorithm to enumerate all the spanning trees on 
             * the given graph. 
             */
            // Initialize a vector of spanning trees
            std::vector<std::vector<StrPair> > trees; 

            // Run Uno's algorithm for each node in the tree 
            std::vector<Node<T>*> nodes = this->getNodes();
            for (auto&& node : nodes)
            {
                std::vector<std::vector<StrPair> > trees_rooted_at_node = this->enumSpanningTrees(node->id);
                for (auto&& tree : trees_rooted_at_node) trees.push_back(tree);
            }

            return trees;
        }

        std::vector<std::vector<StrPair> > enumDoubleSpanningForests(std::string id1, std::string id2)
        {
            /*
             * Enumerate the two-component spanning forests of the given graph at 
             * the two given root nodes, by partitioning the nodes into all possible
             * pairs of subsets and performing Uno's algorithm on each induced 
             * subgraph. 
             */
            // Initialize a vector of spanning forests
            std::vector<std::vector<StrPair> > forests;
            Node<T>* root1 = this->getNode(id1);
            Node<T>* root2 = this->getNode(id2);

            // Get the nodes in the graph
            std::vector<Node<T>*> nodes = this->getNodes();
            std::vector<Node<T>*> nonroot;
            for (auto&& v : nodes)
            {
                if (v != root1 && v != root2) nonroot.push_back(v);
            }

            // Get all subsets of nodes containing root1 and not root2
            std::vector<std::vector<Node<T>*> > subsets = powerset<Node<T>*>(nonroot);
            std::vector<std::vector<Node<T>*> > subsets1;
            for (auto&& subset : subsets)
            {
                std::vector<Node<T>*> subset1(subset);
                subset1.push_back(root1);
                subsets1.push_back(subset1);
            }

            // Run Uno's algorithm on each pair of induced subgraphs
            for (auto&& subset1 : subsets1)
            {
                std::vector<Node<T>*> subset2;
                for (auto&& v : nonroot)
                {
                    if (std::find(subset1.begin(), subset1.end(), v) == subset1.end())
                        subset2.push_back(v);
                }
                subset2.push_back(root2);
                MarkovDigraph<T>* subgraph1 = this->subgraph(subset1);
                MarkovDigraph<T>* subgraph2 = this->subgraph(subset2);
                
                // Compute a spanning tree of the two subgraphs and ensure 
                // that they are connected
                std::vector<Edge<T> > tree1 = subgraph1->getSpanningTreeFromDFS(id1);
                std::vector<Edge<T> > tree2 = subgraph2->getSpanningTreeFromDFS(id2);
                if (tree1.size() == subset1.size() - 1 && tree2.size() == subset2.size() - 1)
                {
                    std::vector<std::vector<StrPair> > trees1 = subgraph1->enumSpanningTrees(id1);
                    std::vector<std::vector<StrPair> > trees2 = subgraph2->enumSpanningTrees(id2);

                    // Concatenate each pair of trees to get the desired forests 
                    for (auto&& t1 : trees1)
                    {
                        for (auto&& t2 : trees2)
                        {
                            std::vector<StrPair> forest;
                            for (auto&& e : t1) forest.push_back(e);
                            for (auto&& e : t2) forest.push_back(e);
                            if (forest.size() == nodes.size() - 2)
                            {
                                forests.push_back(forest);
                            }
                        }
                    }
                }

                // Make sure that dynamically allocated memory is freed!
                delete subgraph1;
                delete subgraph2;
            }

            return forests;
        }

        Array<T, Dynamic, 1> getSteadyStateFromSVD(T sv_tol, bool normalize = true)
        {
            /*
             * Return a vector of steady-state probabilities for the 
             * nodes, in the order given by this->nodes, by solving
             * for the nullspace of the Laplacian matrix.
             */
            Matrix<T, Dynamic, Dynamic> laplacian = this->getLaplacian();
            
            // Obtain the nullspace matrix of the Laplacian matrix
            Matrix<T, Dynamic, Dynamic> nullspace;
            try
            {
                nullspace = nullspaceSVD<T>(laplacian, sv_tol);
            }
            catch (const std::runtime_error& e)
            {
                throw;
            }

            // Each column of the nullspace matrix corresponds to a basis
            // vector of the nullspace; each row corresponds to a single 
            // microstate in the graph; since each microstate lies in a
            // single terminal SCC of the graph, each microstate has only
            // one nullspace basis vector contributing to (defining) its 
            // steady-state probability
            Array<T, Dynamic, 1> steady_state = nullspace.array().rowwise().sum();
            if (normalize)
                return (steady_state / steady_state.sum());
            else
                return steady_state;
        }

        Array<T, Dynamic, 1> getSteadyStateFromRecurrence(bool normalize, T ztol)
        {
            /*
             * Return a vector of steady-state probabilities for the
             * nodes, in the order given by this->nodes, by the recurrence
             * relation of Chebotarev & Agaev for the k-th forest matrix
             * (Chebotarev & Agaev, Lin Alg Appl, 2002, Eqs. 17-18).
             */
            // Get the row Laplacian matrix
            Matrix<T, Dynamic, Dynamic> laplacian = (-this->getLaplacian()).transpose();

            // Obtain the spanning tree weight vector from the row Laplacian matrix
            Matrix<T, Dynamic, 1> steady_state;
            try
            {
                steady_state = spanningTreeWeightVector<T>(laplacian, ztol);
            }
            catch (const std::runtime_error& e)
            {
                throw;
            }
            if (normalize)
            {
                T norm = steady_state.sum();
                return (steady_state / norm).array();
            }
            return steady_state.array(); 
        }

        Array<T, Dynamic, 1> getSteadyStateFromPaths(bool normalize = true)
        {
            /*
             * Return a vector of steady-state probabilities for the
             * nodes, in the order given by this->nodes, by following
             * paths from arbitrarily chosen nodes in each terminal SCC.
             * 
             * This calculation returns correct steady-state probabilities
             * if, and only if, the given values for the edge labels 
             * satisfy detailed balance. This method does not check 
             * whether detailed balance is satisfied.
             */
            using std::log;

            unsigned dim = this->nodes.size();
            Array<T, Dynamic, 1> steady_state(dim);

            // Maintain a checklist of indicators for whether each 
            // node was visited
            std::vector<bool> visited;
            for (auto&& node : this->nodes) visited.push_back(false);

            // While at least one node has not been visited ...
            auto curr_unvisited = visited.begin();
            while (curr_unvisited != visited.end())
            {
                // Assign a canonical steady-state value of unity (zero
                // in log-scale) to the first unvisited node and mark it
                // as visited
                unsigned curr_idx = curr_unvisited - visited.begin();
                Node<T>* curr_node = this->nodes[curr_idx];
                steady_state(curr_idx) = 0.0;
                visited[curr_idx] = true;

                // Obtain a spanning tree of all accessible nodes from
                // the first unvisited node (note that all such nodes 
                // should also be unvisited)
                std::vector<Edge<T> > tree = this->getOutwardTreeFromDFS(curr_node);

                // Propagate ratios of forward/reverse edge label values
                // down the tree in log-scale
                unsigned source_idx, target_idx;
                T forward, reverse, log_ratio;
                for (auto&& edge : tree)
                {
                    Node<T>* source = edge.first;
                    Node<T>* target = edge.second;
                    source_idx = std::find(this->nodes.begin(), this->nodes.end(), source) - this->nodes.begin();
                    target_idx = std::find(this->nodes.begin(), this->nodes.end(), target) - this->nodes.begin();
                    
                    // Check that the reverse edge exists in the graph
                    if (this->edges[target].find(source) == this->edges[target].end())
                        throw std::runtime_error("Graph is not reversible, violated detailed balance");

                    // Compute the log-ratio of forward/reverse edge label values
                    // (given that they were both specified)
                    forward = log(this->labels[edge]);
                    reverse = log(this->labels[std::make_pair(target, source)]);
                    log_ratio = forward - reverse;

                    // Assign the steady-state value for the target node
                    steady_state(target_idx) = steady_state(source_idx) + log_ratio;

                    // Mark the target node as visited
                    visited[target_idx] = true;
                }

                // Check if there are any unvisited nodes remaining
                curr_unvisited = std::find(visited.begin(), visited.end(), false);
            }

            if (normalize)    // Normalize by log-sum-exp if desired
            {
                T logsumexp = log((steady_state - steady_state.maxCoeff()).exp().sum());
                return (steady_state - logsumexp).exp();
            }
            else              // Otherwise, simply exponentiate and return
                return steady_state.exp();
        }

        void randomizeFree(T param_lower, T param_upper, std::mt19937& rng)
        {
            /*
             * Randomly sample new values from a logarithmic distribution
             * between 10 ^ param_lower and 10 ^ param_upper for the model
             * parameters without any constraints.
             */
            using std::pow;

            // Uniform distribution over [0, 1)
            std::uniform_real_distribution<T> dist;

            // Iterate over all edges ...
            T rand, value;
            for (auto&& node : this->nodes)
            {
                for (auto&& dest : this->edges[node])
                {
                    // ... and update their edge label values, up to the 
                    // precision of the scalar type
                    rand = dist(rng);
                    value = pow(10.0, param_lower + (param_upper - param_lower) * rand);
                    Edge<T> edge = std::make_pair(node, dest);
                    this->labels[edge] = value;
                }
            }
        }
};

#endif
