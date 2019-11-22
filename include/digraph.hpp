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
 *     11/21/2019
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

        // ----------------------------------------------------- //
        //                     PRIVATE METHODS                   //
        // ----------------------------------------------------- //
        std::vector<Edge<T> > getSpanningTreeFromRoot(Node<T>* root)
        {
            /*
             * Obtain a vector of edges in a single spanning tree, with  
             * the edges ordered via a topological sort on the second
             * index (i.e., for any two edges (i, j) and (k, l) in the
             * vector such that (j, l) is an edge, (i, j) precedes (k, l)).
             */
            std::vector<Edge<T> > tree;
            std::stack<Edge<T> > stack;
            std::unordered_set<Node<T>*> visited;

            // Initiate depth-first search from the root
            visited.insert(root);
            for (auto&& neighbor : this->edges[root])
                stack.push(std::make_pair(root, neighbor));

            // Run until stack is empty
            while (!stack.empty())
            {
                // Pop topmost microstate from the stack ...
                Edge<T> curr_edge = stack.top();
                stack.pop();
                tree.push_back(curr_edge);
                Node<T>* curr_node = curr_edge.second;
                visited.insert(curr_node);

                // ... and push its unvisited neighbors onto the stack
                for (auto&& neighbor : this->edges[curr_node])
                {
                    if (visited.find(neighbor) == visited.end())
                        stack.push(std::make_pair(curr_node, neighbor));
                }
            }

            return tree;
        }

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
                std::vector<Edge<T> > tree = this->getSpanningTreeFromRoot(curr_node);

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
