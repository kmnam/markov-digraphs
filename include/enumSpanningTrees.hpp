#ifndef ENUM_SPANNING_TREES_HPP
#define ENUM_SPANNING_TREES_HPP
#include <vector>
#include <algorithm>
#include <iterator>
#include "digraph.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/9/2019
 */

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

    // Get number of vertices in the graph (tree)
    unsigned nvertices = tree.size() + 1;

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
void enumSpanningTreesIter(const std::vector<Edge<T> >& parent_tree,
                           const std::vector<Edge<T> >& tree0,
                           const std::vector<Edge<T> >& edges,
                           const std::vector<Edge<T> >& back_edges,
                           const std::vector<Edge<T> >& nonback_edges,
                           const std::vector<Node<T>*>& order, 
                           std::vector<std::vector<Edge<T> > >& trees)
{
    /*
     * Recursive function to be called as part of Uno's algorithm. 
     */
    // Define function for obtaining the least edge in the complement 
    // of a given tree w.r.t tree0 (i.e., tree0 - tree)
    std::function<Edge<T>(const std::vector<Edge<T> >&)> least_edge_not_in_tree
        = [edges, tree0](const std::vector<Edge<T> >& tree)
    {
        return *std::find_if(
            edges.begin(), edges.end(),
            [tree, tree0](const Edge<T>& e)
            {
                return (
                    std::find(tree0.begin(), tree0.end(), e) != tree0.end() &&
                    std::find(tree.begin(), tree.end(), e) == tree.end()
                );
            }
        );
    };

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
                    if (isBackEdge(child_tree, e)) new_back_edges.push_back(e);
                    else                           new_nonback_edges.push_back(e);
                }
                // Note that these edge sets should already be sorted with 
                // respect to source index 
            }

            // Add the child tree and call the function recursively on the child tree
            trees.push_back(child_tree);
            enumSpanningTreesIter(
                child_tree, tree0, edges, new_back_edges, new_nonback_edges, order, trees
            );
        }
    }
}

template <typename T>
std::vector<std::vector<Edge<T> > > enumSpanningTrees(MarkovDigraph<T>* graph, Node<T>* root)
{
    /*
     * Perform Uno's algorithm for enumerating the spanning trees on 
     * the given graph rooted at the given node. 
     */
    // Initialize a vector of spanning trees
    std::vector<std::vector<Edge<T> > > trees; 

    // Get a DFS spanning tree emanating from each root vertex
    std::vector<Edge<T> > tree_dfs = graph->getSpanningTreeFromDFS(root);

    // Order the nodes with respect to the DFS traversal 
    std::vector<Node<T>*> order;
    order.push_back(root);
    for (auto&& e : tree_dfs) order.push_back(e.second);

    // Reverse the edges in the DFS spanning tree to get a spanning 
    // tree of the graph 
    std::vector<Edge<T> > tree0;
    for (auto&& e : tree_dfs)
        tree0.push_back(std::make_pair(e.second, e.first));
    trees.push_back(tree0);

    // Get all the edges in the graph and sort them by w.r.t to 
    // DFS traversal 
    std::vector<Edge<T> > edges = graph->getEdges();
    if (edges.size() == 0)    // If the graph has no edges, return the empty tree
    {
        std::vector<Edge<T> > empty;
        trees.push_back(empty);
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
            if (isBackEdge(tree0, e)) back_edges.push_back(e);
            else                      nonback_edges.push_back(e);
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

template <typename T>
std::vector<std::vector<Edge<T> > > enumAllSpanningTrees(MarkovDigraph<T>* graph)
{
    /*
     * Perform Uno's algorithm to enumerate all the spanning trees on 
     * the given graph. 
     */
    // Initialize a vector of spanning trees
    std::vector<std::vector<Edge<T> > > trees; 

    // Run Uno's algorithm for each node in the tree 
    std::vector<Node<T>*> nodes = graph->getNodes();
    for (auto&& node : nodes)
    {
        std::vector<std::vector<Edge<T> > > trees_rooted_at_node = enumSpanningTrees(graph, node);
        for (auto&& tree : trees_rooted_at_node) trees.push_back(tree);
    }

    return trees;
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
        k_combinations = combinations(data, k);
        powerset.insert(
            powerset.end(),
            std::make_move_iterator(k_combinations.begin()),
            std::make_move_iterator(k_combinations.end())
        );
    }

    return powerset;
}

template <typename T>
std::vector<std::vector<Edge<T> > > enumDoubleSpanningForests(MarkovDigraph<T>* graph,
                                                              Node<T>* root1, Node<T>* root2)
{
    /*
     * Enumerate the two-component spanning forests of the given graph at 
     * the two given root nodes, by partitioning the nodes into all possible
     * pairs of subsets and performing Uno's algorithm on each induced 
     * subgraph. 
     */
    // Initialize a vector of spanning forests
    std::vector<std::vector<Edge<T> > > forests;

    // Get the nodes in the graph
    std::vector<Node<T>*> nodes = graph->getNodes();
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
        for (auto&& v : subset1) std::cout << v->id << " "; std::cout << "| ";
        for (auto&& v : subset2) std::cout << v->id << " "; std::cout << std::endl;
        MarkovDigraph<T>* subgraph1 = graph->subgraph(subset1);
        MarkovDigraph<T>* subgraph2 = graph->subgraph(subset2);
        std::vector<std::vector<Edge<T> > > trees1 = enumSpanningTrees(subgraph1, root1);
        std::vector<std::vector<Edge<T> > > trees2 = enumSpanningTrees(subgraph2, root2);

        // Concatenate each pair of trees to get the desired forests 
        for (auto&& t1 : trees1)
        {
            for (auto&& t2 : trees2)
            {
                std::vector<Edge<T> > forest;
                for (auto&& e : t1) forest.push_back(e);
                for (auto&& e : t2) forest.push_back(e);
                forests.push_back(forest);
                for (auto&& e : forest) std::cout << e.first->id << "," << e.second->id << " ";
                std::cout << std::endl;
            }
        }

        // Make sure that dynamically allocated memory is freed!
        delete subgraph1;
        delete subgraph2;
    }
}

#endif
