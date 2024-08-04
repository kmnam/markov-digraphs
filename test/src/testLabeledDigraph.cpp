/**
 * Test module for the `LabeledDigraph` class.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/3/2024
 */

#include <string>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/digraph.hpp"

using namespace Eigen;

using std::abs; 
using boost::multiprecision::abs;
using boost::multiprecision::number; 
using boost::multiprecision::mpfr_float_backend;

typedef number<mpfr_float_backend<100> > PreciseType;

/**
 * Generate a random string of alphanumeric characters with the given length. 
 */
std::string getRandomString(const int l, boost::random::mt19937& rng)
{
    boost::random::uniform_int_distribution<> dist(0, 35);
    std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::stringstream ss; 
    for (int j = 0; j < l; ++j)
        ss << alphabet[dist(rng)];
    return ss.str();
}

/**
 * Get the submatrix of the given matrix obtained by removing the indicated
 * rows and columns. 
 */
template <typename T>
Matrix<T, Dynamic, Dynamic> getSubmatrix(const Ref<const Matrix<T, Dynamic, Dynamic> >& A, 
                                         const std::unordered_set<int>& rows, 
                                         const std::unordered_set<int>& cols)
{
    std::vector<int> rows_include, cols_include; 
    for (int i = 0; i < A.rows(); ++i)
    {
        if (rows.find(i) == rows.end())
            rows_include.push_back(i); 
    }
    for (int j = 0; j < A.cols(); ++j)
    {
        if (cols.find(j) == cols.end()) 
            cols_include.push_back(j);
    }

    return A(rows_include, cols_include); 
}

/**
 * Get the set of all size-p subsets of the given set, represented here 
 * as a vector of distinct ints. 
 */
void choose(const std::vector<int>& set, const int p, std::vector<int>& curr_subset, 
            const int idx, std::vector<std::vector<int> >& all_subsets)
{
    // If the current subset is size p, then add it to the collection 
    if (curr_subset.size() == p)
    {
        all_subsets.push_back(curr_subset); 
        return;
    } 

    // If we have processed the entire set, then return
    if (idx >= set.size())
        return; 

    // Include the current element ... 
    curr_subset.push_back(set[idx]); 
    choose(set, p, curr_subset, idx + 1, all_subsets);

    // ... or exclude the current element
    curr_subset.pop_back();
    choose(set, p, curr_subset, idx + 1, all_subsets); 
}

/**
 * Get the set of all subsets of the range [0, ..., k - 1] of size p. 
 */
std::vector<std::vector<int> > choose(const int k, const int p)
{
    std::vector<int> set;  
    for (int i = 0; i < k; ++i)
        set.push_back(i); 
    std::vector<int> curr_subset;
    std::vector<std::vector<int> > all_subsets;
    choose(set, p, curr_subset, 0, all_subsets);

    return all_subsets; 
}

// ----------------------------------------------------------------------- // 
//                               TEST MODULES                              // 
// ----------------------------------------------------------------------- // 
/** 
 * Test the following methods:
 *
 * - `LabeledDigraph<...>::addNode()` 
 * - `LabeledDigraph<...>::removeNode()`
 * - `LabeledDigraph<...>::hasNode()`
 * - `LabeledDigraph<...>::getNumNodes()`
 * - `LabeledDigraph<...>::getAllNodeIds()`
 */
template <typename T, typename U>
void testNodeMethods(LabeledDigraph<T, U>* graph, int num_nodes,
                     std::vector<std::string>& node_ids)
{
    // Check that the number of nodes is correct 
    REQUIRE(graph->getNumNodes() == num_nodes);
    REQUIRE(num_nodes == node_ids.size()); 

    // Check that the list of node IDs is correct, and specified in the 
    // correct order
    REQUIRE_THAT(graph->getAllNodeIds(), Catch::Matchers::Equals(node_ids));

    // Check that the graph has every specified node 
    for (auto it = node_ids.begin(); it != node_ids.end(); ++it)
        REQUIRE(graph->hasNode(*it));

    // Try adding a node that already exists in the graph 
    std::string first_id = node_ids[0];
    REQUIRE_THROWS(graph->addNode(first_id));

    // Try adding three nodes into the graph ... 
    boost::random::mt19937 rng(1234567890);
    for (int i = 0; i < 3; ++i)
    {
        // Generate a node ID that does not exist within the graph
        std::string node_id;
        bool found_new_id = false;
        while (!found_new_id)
        {
            node_id = getRandomString(10, rng);
            if (std::find(node_ids.begin(), node_ids.end(), node_id) != node_ids.end())
            {
                // The node ID already exists within the graph, in which case 
                // hasNode() should return true 
                REQUIRE(graph->hasNode(node_id)); 
            }
            else 
            {
                // Otherwise, the node ID does not exist, in which case 
                // hasNode() should return false 
                REQUIRE_FALSE(graph->hasNode(node_id));
                found_new_id = true;
            }
        }

        // Add the node to the graph ... 
        graph->addNode(node_id); 
        node_ids.push_back(node_id);
        num_nodes++; 

        // ... and test that getNumNodes(), hasNode(), and getAllNodeIds() 
        // return correctly updated values 
        REQUIRE(graph->getNumNodes() == num_nodes);
        REQUIRE(graph->hasNode(node_id));
        REQUIRE_THAT(graph->getAllNodeIds(), Catch::Matchers::Equals(node_ids)); 
    }

    // Among the new nodes, remove the second, then the first, then test that 
    // the node IDs are stored in the order in which they were added
    for (int i = 0; i < 2; ++i)
    {
        std::string penultimate_id = node_ids[node_ids.size() - 2];
        graph->removeNode(penultimate_id);
        node_ids.erase(node_ids.end() - 2);
        num_nodes--;
        REQUIRE(graph->getNumNodes() == num_nodes); 
        REQUIRE_FALSE(graph->hasNode(penultimate_id));
        REQUIRE_THAT(graph->getAllNodeIds(), Catch::Matchers::Equals(node_ids));
    }

    // Try removing a non-existent node 
    REQUIRE_THROWS(graph->removeNode("this node does not exist"));
}

/** 
 * Test the following methods:
 *
 * - `LabeledDigraph<...>::addEdge()` 
 * - `LabeledDigraph<...>::removeEdge()`
 * - `LabeledDigraph<...>::hasEdge()`
 * - `LabeledDigraph<...>::getEdgeLabel()`
 * - `LabeledDigraph<...>::setEdgeLabel()`
 * - `LabeledDigraph<...>::getAllEdgesFromNode()`
 */
template <typename T, typename U>
void testEdgeMethods(LabeledDigraph<T, U>* graph, int num_nodes,
                     std::vector<std::string>& node_ids,
                     std::vector<std::pair<std::string, std::string> >& edges,
                     std::vector<U>& labels, 
                     std::pair<std::string, std::string> example_nonedge,
                     const double eps)
{
    // Check that hasEdge() returns true for all edges in the graph 
    for (auto it = edges.begin(); it != edges.end(); ++it)
        REQUIRE(graph->hasEdge(it->first, it->second)); 

    // Check that hasEdge() returns false for the given example non-edge 
    REQUIRE_FALSE(graph->hasEdge(example_nonedge.first, example_nonedge.second)); 
    
    // Check that hasEdge() returns false for any edge in which either node 
    // does not exist in the graph
    std::string first_id = node_ids[0]; 
    REQUIRE_FALSE(graph->hasEdge(first_id, "this node does not exist")); 
    REQUIRE_FALSE(graph->hasEdge("this node does not exist", first_id));

    // Check that hasEdge() returns false for all possible self-loops
    for (int i = 0; i < num_nodes; ++i)
        REQUIRE_FALSE(graph->hasEdge(node_ids[i], node_ids[i]));

    // Check that getEdgeLabel() returns the correct values for all edges
    // in the graph
    int num_edges = edges.size(); 
    for (int i = 0; i < num_edges; ++i)
    {
        std::string source = edges[i].first; 
        std::string target = edges[i].second; 
        U label = labels[i]; 
        REQUIRE_THAT(
            static_cast<double>(graph->getEdgeLabel(source, target)),
            Catch::Matchers::WithinAbs(static_cast<double>(label), eps)
        ); 
    }

    // Try accessing the label on a non-existent edge
    REQUIRE_THROWS(graph->getEdgeLabel(example_nonedge.first, example_nonedge.second));

    // Try accessing the label on an edge with a non-existent source or target
    // node 
    REQUIRE_THROWS(graph->getEdgeLabel(node_ids[0], "this node does not exist")); 
    REQUIRE_THROWS(graph->getEdgeLabel("this node does not exist", node_ids[0]));
    REQUIRE_THROWS(graph->getEdgeLabel("this node does not exist", "this node does not exist either"));

    // For each node in the graph ...
    for (int i = 0; i < num_nodes; ++i)
    {
        // Check that all outgoing edges from the node are correctly identified 
        std::vector<std::pair<std::string, U> > out_edges = graph->getAllEdgesFromNode(node_ids[i]);
        std::vector<std::string> out_edges1, out_edges2;
        std::vector<U> out_labels1, out_labels2;
        for (auto it = out_edges.begin(); it != out_edges.end(); ++it)
        {
            out_edges1.push_back(it->first); 
            out_labels1.push_back(it->second); 
        }
        for (int j = 0; j < num_edges; ++j)
        {
            if (edges[j].first == node_ids[i])
            {
                out_edges2.push_back(edges[j].second); 
                out_labels2.push_back(labels[j]); 
            }
        }
        REQUIRE_THAT(out_edges1, Catch::Matchers::UnorderedEquals(out_edges2));

        // To check the labels, iterate through the first vector of edges
        // (obtained via getEdgesFromNode()), identify the corresponding 
        // entry in the second vector of edges, and compare the corresponding
        // edge labels
        for (auto it1 = out_edges1.begin(); it1 != out_edges1.end(); ++it1)
        {
            for (auto it2 = out_edges2.begin(); it2 != out_edges2.end(); ++it2)
            {
                if (*it1 == *it2)
                {
                    int j = std::distance(out_edges1.begin(), it1); 
                    int k = std::distance(out_edges2.begin(), it2); 
                    REQUIRE_THAT(
                        static_cast<double>(out_labels1[j]),
                        Catch::Matchers::WithinAbs(static_cast<double>(out_labels2[k]), eps)
                    );
                }
            }
        }
    }

    // Check that setEdgeLabel() changes the desired edge label 
    graph->setEdgeLabel(edges[1].first, edges[1].second, 512); 
    REQUIRE_THAT(
        static_cast<double>(graph->getEdgeLabel(edges[1].first, edges[1].second)),
        Catch::Matchers::WithinAbs(512, eps)
    );

    // Check that all other edge labels remain as they were
    for (int i = 0; i < num_edges; ++i)
    {
        if (i != 1)
        {
            std::string source = edges[i].first;
            std::string target = edges[i].second; 
            U label = labels[i];
            REQUIRE_THAT(
                static_cast<double>(graph->getEdgeLabel(source, target)),
                Catch::Matchers::WithinAbs(static_cast<double>(label), eps)
            ); 
        }
    }

    // Try setting the label for the given example non-edge 
    REQUIRE_THROWS(graph->setEdgeLabel(example_nonedge.first, example_nonedge.second, 16));

    // Try setting the label for an edge with a non-existent source or target
    // node 
    REQUIRE_THROWS(graph->setEdgeLabel(node_ids[0], "this node does not exist", 96)); 
    REQUIRE_THROWS(graph->setEdgeLabel("this node does not exist", node_ids[0], 3.14159)); 
    REQUIRE_THROWS(graph->setEdgeLabel("this node does not exist", "this node does not exist either", 7));

    // Try removing the given example non-edge -- this should do nothing! 
    REQUIRE_NOTHROW(graph->removeEdge(example_nonedge.first, example_nonedge.second)); 

    // Try removing an edge with a non-existent source or target node 
    REQUIRE_THROWS(graph->removeEdge(node_ids[0], "this node does not exist")); 
    REQUIRE_THROWS(graph->removeEdge("this node does not exist", node_ids[0]));
    REQUIRE_THROWS(graph->removeEdge("this node does not exist", "this node does not exist either"));

    // Remove an edge in the graph and check that hasEdge() returns false 
    graph->removeEdge(edges[0].first, edges[0].second); 
    REQUIRE_FALSE(graph->hasEdge(edges[0].first, edges[0].second));

    // Remove a node in the graph, and check that hasEdge() returns false 
    // for all edges involving the removed node, but returns true for all 
    // other edges 
    std::string second_id = node_ids[1]; 
    graph->removeNode(second_id);
    for (auto it = edges.begin() + 1; it != edges.end(); ++it)   // Recall that the first edge was removed
    {
        if (it->first == second_id || it->second == second_id)
            REQUIRE_FALSE(graph->hasEdge(it->first, it->second)); 
        else 
            REQUIRE(graph->hasEdge(it->first, it->second)); 
    }
}

/**
 * Test `LabeledDigraph<...>::clear()`. 
 */
template <typename T, typename U>
void testClear(LabeledDigraph<T, U>* graph, std::vector<std::string>& node_ids)
{
    // Clear the graph and check that getNumNodes() and getAllNodeIds() 
    // return correct values 
    graph->clear();
    REQUIRE(graph->getNumNodes() == 0);
    REQUIRE_THAT(graph->getAllNodeIds(), Catch::Matchers::Equals(std::vector<std::string>{}));

    // Check that hasNode() returns false for every node that existed within
    // the graph
    for (auto it = node_ids.begin(); it != node_ids.end(); ++it)
        REQUIRE_FALSE(graph->hasNode(*it)); 

    // Check that removeNode() throws an exception if we try to remove any 
    // of the nodes again
    for (auto it = node_ids.begin(); it != node_ids.end(); ++it)
        REQUIRE_THROWS(graph->removeNode(*it)); 

    // Check that hasEdge() returns false for all possible combinations of 
    // the removed nodes 
    for (auto it1 = node_ids.begin(); it1 != node_ids.end(); ++it1)
    {
        for (auto it2 = node_ids.begin(); it2 != node_ids.end(); ++it2)
        {
            REQUIRE_FALSE(graph->hasEdge(*it1, *it2)); 
        }
    }
}

/**
 * Test `LabeledDigraph<...>::getLaplacian()`.
 */
template <typename T, typename U>
void testGetLaplacian(LabeledDigraph<T, U>* graph, const int num_nodes, 
                      const Ref<const Matrix<U, Dynamic, Dynamic> >& laplacian,
                      const double eps) 
{
    // Compute and check the Laplacian matrix
    Matrix<U, Dynamic, Dynamic> laplacian2 = graph->getLaplacian();
    REQUIRE(laplacian2.rows() == num_nodes); 
    REQUIRE(laplacian2.cols() == num_nodes);
    double err = static_cast<double>((-laplacian.transpose() - laplacian2).array().abs().maxCoeff());
    REQUIRE_THAT(err, Catch::Matchers::WithinAbs(0.0, eps));
}

/**
 * Test `LabeledDigraph<...>::getSpanningForestMatrix()` and verify some 
 * basic properties. 
 */
template <typename T, typename U>
void testGetSpanningForestMatrixProperties(LabeledDigraph<T, U>* graph,
                                           const int num_nodes,
                                           const Ref<const Matrix<U, Dynamic, Dynamic> >& laplacian,
                                           const int num_terminal_sccs,
                                           const double eps) 
{
    // Compute and check the 0-th spanning forest matrix
    Matrix<U, Dynamic, Dynamic> Q0 = graph->getSpanningForestMatrix(0);
    Matrix<U, Dynamic, Dynamic> identity = Matrix<U, Dynamic, Dynamic>::Identity(num_nodes, num_nodes);
    double err = static_cast<double>((identity - Q0).array().abs().maxCoeff()); 
    REQUIRE_THAT(err, Catch::Matchers::WithinAbs(0.0, eps)); 

    // Compute and check the 1st spanning forest matrix 
    Matrix<U, Dynamic, Dynamic> Q1 = graph->getSpanningForestMatrix(1);
    U sum_labels = 0; 
    for (int i = 0; i < num_nodes; ++i)
        sum_labels += laplacian(i, i);
    err = static_cast<double>((-laplacian + sum_labels * identity - Q1).array().abs().maxCoeff());
    REQUIRE_THAT(err, Catch::Matchers::WithinAbs(0.0, eps));

    // Compute and check that all spanning forest matrices up to k = N - T, 
    // where N is the number of nodes and T is the number of terminal SCCs,
    // have constant row sums 
    for (int k = 0; k <= num_nodes - num_terminal_sccs; ++k)
    {
        Matrix<double, Dynamic, 1> rowsums =
            (graph->getSpanningForestMatrix(k) * Matrix<U, Dynamic, 1>::Ones(num_nodes)).template cast<double>();
        for (int i = 1; i < num_nodes; ++i)
            REQUIRE_THAT(rowsums(0), Catch::Matchers::WithinAbs(rowsums(i), eps));
    }

    // Compute and check all spanning forest matrices for k > N - T 
    for (int i = 0; i < 10; ++i)
    {
        int k = num_nodes - num_terminal_sccs + 1 + i; 
        Matrix<double, Dynamic, Dynamic> Qk = graph->getSpanningForestMatrix(k).template cast<double>();
        REQUIRE_THAT(Qk.array().abs().maxCoeff(), Catch::Matchers::WithinAbs(0.0, eps));
    }
}

/**
 * Test `LabeledDigraph<...>::getSpanningForestMatrix()` for strongly 
 * connected graphs. 
 * 
 * This test performs alternative calculations of the (N - 1)-th spanning
 * forest matrix entries using the matrix-tree theorem.
 */
template <typename T, typename U>
void testGetSpanningTreeMatrixStronglyConnected(LabeledDigraph<T, U>* graph,
                                                const int num_nodes,
                                                const Ref<const Matrix<U, Dynamic, Dynamic> >& laplacian,
                                                const double eps)
{
    // Compute and check the (N - 1)-th spanning forest matrix
    //
    // The (i, j)-th entry of this matrix is the weight of all spanning trees
    // rooted at j
    //
    // In other words, the (i, j)-th entry should match the absolute value
    // of the minor of the Laplacian matrix with row j and column i removed
    Matrix<double, Dynamic, Dynamic> Q1 = graph->getSpanningForestMatrix(num_nodes - 1).template cast<double>();
    for (int j = 0; j < num_nodes; ++j)
    {
        // Check that each column is constant 
        for (int i = 1; i < num_nodes; ++i)
            REQUIRE_THAT(Q1(0, j), Catch::Matchers::WithinAbs(Q1(i, j), eps));

        // Check that this constant value is equal to the corresponding 
        // Laplacian minor
        int i = 0; 
        Matrix<U, Dynamic, Dynamic> sublaplacian = getSubmatrix<U>(
            laplacian, std::unordered_set<int>{j}, std::unordered_set<int>{i}
        );
        double minor = static_cast<double>(abs(sublaplacian.determinant())); 
        REQUIRE_THAT(Q1(i, j), Catch::Matchers::WithinAbs(minor, eps)); 
    }
}

/**
 * Test `LabeledDigraph<...>::getSpanningForestMatrix()` for graphs with 
 * a specified number of terminal nodes.
 *
 * This test performs alternative calculations of the (N - T)-th and
 * (N - T - 1)-th spanning forest matrix entries, where T is the number of
 * terminal nodes, using the all-minors matrix-tree theorem.
 *
 * This test assumes that the terminal nodes are indexed as the final T 
 * nodes in the graph. 
 */
template <typename T, typename U>
void testGetSpanningForestMatrixTerminalNodes(LabeledDigraph<T, U>* graph,
                                              const int num_nodes,
                                              std::vector<std::string>& node_ids, 
                                              const Ref<const Matrix<U, Dynamic, Dynamic> >& laplacian,
                                              std::vector<std::string>& terminal_node_ids,
                                              const double eps)
{
    const int num_terminal_nodes = terminal_node_ids.size(); 
    const int k = num_nodes - num_terminal_nodes;

    // Get the indices of the non-terminal and terminal nodes 
    std::vector<int> nonterminal_nodes, terminal_nodes; 
    for (int i = 0; i < num_nodes; ++i)
    {
        if (std::find(terminal_node_ids.begin(), terminal_node_ids.end(), node_ids[i]) == terminal_node_ids.end())
            nonterminal_nodes.push_back(i);
        else
            terminal_nodes.push_back(i); 
    }

    // Compute and check the (N - T - p)-th spanning forest matrix, for
    // p = 0, 1, 2, ... 
    //
    // The (i, j)-th entry of this matrix is given as follows:
    // - If i and j are non-terminal, then this entry is the weight of all
    //   spanning forests rooted at the terminal nodes and p non-terminal
    //   nodes (including j), in which there is a path from i to j
    // - If i is non-terminal and j is terminal, then this entry is the weight
    //   of all spanning forests rooted at the terminal nodes plus p non-
    //   terminal nodes, in which there is a path from i to j 
    // - If i is terminal and i != j, then this entry is zero
    // - If i is terminal and i == j, then this entry is the weight of all 
    //   spanning forests rooted at the terminal nodes plus p non-terminal 
    //   nodes
    for (int p = 0; p <= k; ++p)
    {
        Matrix<double, Dynamic, Dynamic> Q = graph->getSpanningForestMatrix(k - p).template cast<double>();

        // Get the set of all subsets of size p among the non-terminal nodes
        std::vector<std::vector<int> > subsets = choose(k, p);

        // We first check the entries for which i and j are non-terminal ...
        for (const int& i : nonterminal_nodes)
        {
            for (const int& j : nonterminal_nodes)
            {
                if (i != j)
                {
                    // Here, we collect the Laplacian minors obtained by removing
                    // the rows corresponding to the terminal nodes plus p
                    // non-terminal nodes (among which j must be one and i cannot 
                    // be one), and the columns corresponding to the row indices
                    // minus j plus i
                    U total = 0; 
                    for (auto&& subset : subsets)
                    {
                        // Get the indices of the chosen non-terminal nodes
                        std::vector<int> nonterminal_subset; 
                        for (const int& m : subset)
                            nonterminal_subset.push_back(nonterminal_nodes[m]);
                        // If i is not in the chosen set and j is, then the
                        // corresponding Laplacian minor contributes to the
                        // total 
                        if (std::find(nonterminal_subset.begin(), nonterminal_subset.end(), j) != nonterminal_subset.end() &&
                            std::find(nonterminal_subset.begin(), nonterminal_subset.end(), i) == nonterminal_subset.end()
                        )
                        {
                            std::unordered_set<int> rows, cols; 
                            for (const int& m : terminal_nodes) 
                            {
                                rows.insert(m); 
                                cols.insert(m); 
                            }
                            for (const int& m : nonterminal_subset)
                            {
                                rows.insert(m);
                                cols.insert(m);
                            }
                            cols.erase(j);
                            cols.insert(i);
                            Matrix<U, Dynamic, Dynamic> sublaplacian = getSubmatrix<U>(laplacian, rows, cols);
                            total += abs(sublaplacian.determinant()); 
                        }
                    }
                    REQUIRE_THAT(Q(i, j), Catch::Matchers::WithinAbs(static_cast<double>(total), eps));
                }
                else    // i == j
                {
                    // Here, we collect the Laplacian minors obtained by removing
                    // the rows and columns corresponding to the terminal nodes
                    // plus p non-terminal nodes (among which i == j must be one)
                    U total = 0; 
                    for (auto&& subset : subsets)
                    {
                        // Get the indices of the chosen non-terminal nodes
                        std::vector<int> nonterminal_subset; 
                        for (const int& m : subset)
                            nonterminal_subset.push_back(nonterminal_nodes[m]);
                        // If j is in the chosen subset, then the corresponding
                        // Laplacian minor contributes to the total 
                        if (std::find(nonterminal_subset.begin(), nonterminal_subset.end(), j) != nonterminal_subset.end())
                        {
                            std::unordered_set<int> rows; 
                            for (const int& m : terminal_nodes)
                                rows.insert(m); 
                            for (const int& m : nonterminal_subset)
                                rows.insert(m);
                            Matrix<U, Dynamic, Dynamic> sublaplacian = getSubmatrix<U>(laplacian, rows, rows);
                            total += abs(sublaplacian.determinant()); 
                        }
                    }
                    REQUIRE_THAT(Q(i, i), Catch::Matchers::WithinAbs(static_cast<double>(total), eps));
                }
            }
        }

        // We then check the entries for which i is non-terminal and j is terminal ... 
        for (const int& i : nonterminal_nodes)
        {
            for (const int& j : terminal_nodes)
            {
                // Here, we collect the Laplacian minors obtained by removing
                // the rows corresponding to the terminal nodes plus p
                // non-terminal nodes (among which i cannot be one), and the 
                // columns corresponding to the row indices minus j (which is
                // terminal) plus i
                U total = 0;
                for (auto&& subset : subsets)
                {
                    // Get the indices of the chosen non-terminal nodes
                    std::vector<int> nonterminal_subset; 
                    for (const int& m : subset)
                        nonterminal_subset.push_back(nonterminal_nodes[m]);
                    // If i is in not the chosen subset, then the corresponding
                    // Laplacian minor contributes to the total (note that j 
                    // is terminal here)
                    if (std::find(nonterminal_subset.begin(), nonterminal_subset.end(), i) == nonterminal_subset.end())
                    {
                        std::unordered_set<int> rows, cols; 
                        for (const int& m : terminal_nodes)
                        {
                            rows.insert(m); 
                            cols.insert(m); 
                        }
                        for (const int& m : nonterminal_subset)
                        {
                            rows.insert(m); 
                            cols.insert(m); 
                        }
                        cols.erase(j); 
                        cols.insert(i); 
                        Matrix<U, Dynamic, Dynamic> sublaplacian = getSubmatrix<U>(laplacian, rows, cols);
                        total += abs(sublaplacian.determinant()); 
                    }
                }
                REQUIRE_THAT(Q(i, j), Catch::Matchers::WithinAbs(static_cast<double>(total), eps));
            }
        }

        // We then check the entries for which i is terminal and i != j ... 
        for (const int& i : terminal_nodes)
        {
            for (int j = 0; j < num_nodes; ++j)
            {
                if (i != j)
                    REQUIRE_THAT(Q(i, j), Catch::Matchers::WithinAbs(0.0, eps));
            }
        }

        // We then check the entries for which i is terminal and i == j ... 
        for (const int& i : terminal_nodes)
        {
            // Here, we collect the Laplacian minors obtained by removing
            // the rows and columns corresponding to the terminal nodes plus
            // p non-terminal nodes
            U total = 0;
            for (auto&& subset : subsets)
            {
                // Get the indices of the chosen non-terminal nodes
                std::vector<int> nonterminal_subset; 
                for (const int& m : subset)
                    nonterminal_subset.push_back(nonterminal_nodes[m]);
                // Compute the corresponding Laplacian minor 
                std::unordered_set<int> rows;
                for (const int& m : terminal_nodes)
                    rows.insert(m); 
                for (const int& m : nonterminal_subset)
                    rows.insert(m); 
                Matrix<U, Dynamic, Dynamic> sublaplacian = getSubmatrix<U>(laplacian, rows, rows);
                total += abs(sublaplacian.determinant()); 
            }
            REQUIRE_THAT(Q(i, i), Catch::Matchers::WithinAbs(static_cast<double>(total), eps));
        }
    }
}

/**
 * Test `LabeledDigraph<...>::getSteadyStateFromSVD()`. 
 */
template <typename T, typename U>
void testGetSteadyStateFromSVD(LabeledDigraph<T, U>* graph,
                               const Ref<const Matrix<U, Dynamic, Dynamic> >& laplacian,
                               const double eps)
{
    // Compute the steady-state vector
    Matrix<U, Dynamic, 1> ss = graph->template getSteadyStateFromSVD<T, U>();

    // Compute the error incurred by the steady-state computation, as the 
    // magnitude of the steady-state vector left-multiplied by the negative
    // transpose of the given Laplacian matrix (this product should be zero)
    //
    // (Note that taking the negative should not be required)
    double err = static_cast<double>((laplacian.transpose() * ss).array().abs().maxCoeff());
    REQUIRE_THAT(err, Catch::Matchers::WithinAbs(0.0, eps)); 
}

/**
 * Test `LabeledDigraph<...>::getSteadyStateFromRecurrence()`. 
 */
template <typename T, typename U>
void testGetSteadyStateFromRecurrence(LabeledDigraph<T, U>* graph,
                                      const Ref<const Matrix<U, Dynamic, Dynamic> >& laplacian,
                                      const double eps)
{
    // Compute the steady-state vector
    Matrix<U, Dynamic, 1> ss = graph->template getSteadyStateFromRecurrence<U>(false);

    // Compute the error incurred by the steady-state computation, as the 
    // magnitude of the steady-state vector left-multiplied by the negative
    // transpose of the given Laplacian matrix (this product should be zero)
    //
    // (Note that taking the negative should not be required)
    double err = static_cast<double>((laplacian.transpose() * ss).array().abs().maxCoeff());
    REQUIRE_THAT(err, Catch::Matchers::WithinAbs(0.0, eps)); 
}

/**
 * Test `LabeledDigraph<...>::getFPTMomentsFromSolver()`.
 *
 * Since this method assumes that the input graph has a single terminal node, 
 * this test function assumes the same. 
 */
template <typename T, typename U>
void testGetFPTMomentsFromSolver(LabeledDigraph<T, U>* graph, const int num_nodes,
                                 std::vector<std::string>& node_ids, 
                                 std::string target_id,
                                 const Ref<const Matrix<U, Dynamic, Dynamic> >& laplacian, 
                                 const double eps)
{
    // Compute the 0-th moments with QR decomposition and check that they are
    // all one 
    Matrix<U, Dynamic, 1> mu0 = graph->template getFPTMomentsFromSolver<T, U>(target_id, 0);
    for (int i = 0; i < num_nodes; ++i)
        REQUIRE_THAT(static_cast<double>(mu0(i)), Catch::Matchers::WithinAbs(1.0, eps));

    // Compute the 0-th moments with LU decomposition and check that they are
    // all one 
    mu0 = graph->template getFPTMomentsFromSolver<T, U>(target_id, 0, SolverMethod::LUDecomposition);
    for (int i = 0; i < num_nodes; ++i)
        REQUIRE_THAT(static_cast<double>(mu0(i)), Catch::Matchers::WithinAbs(1.0, eps));

    // Compute the r-th moments with QR decomposition, for r = 1, ..., 10, and
    // check that they satisfy the corresponding linear equation
    const int t = std::distance(
        node_ids.begin(), std::find(node_ids.begin(), node_ids.end(), target_id)
    );
    Matrix<U, Dynamic, Dynamic> sublaplacian = getSubmatrix<U>(
        laplacian, std::unordered_set<int>{t}, std::unordered_set<int>{t}
    );
    for (int r = 1; r < 11; ++r)
    {
        Matrix<U, Dynamic, 1> mu = graph->template getFPTMomentsFromSolver<T, U>(target_id, r);
        Matrix<U, Dynamic, 1> mu_sub(num_nodes - 1);
        mu_sub(Eigen::seq(0, t - 1)) = mu(Eigen::seq(0, t - 1)); 
        mu_sub(Eigen::seq(t, num_nodes - 2)) = mu(Eigen::seq(t + 1, num_nodes - 1)); 
        for (int i = 0; i < r; ++i)
            mu_sub = (sublaplacian * mu_sub).eval();
        for (int i = 0; i < num_nodes - 1; ++i)
            REQUIRE_THAT(
                static_cast<double>(mu_sub(i)),
                Catch::Matchers::WithinAbs(boost::math::factorial<double>(r), eps)
            );
    }

    // Compute the r-th moments with LU decomposition, for r = 1, ..., 10, and
    // check that they satisfy the corresponding linear equation
    for (int r = 1; r < 11; ++r)
    {
        Matrix<U, Dynamic, 1> mu = graph->template getFPTMomentsFromSolver<T, U>(
            target_id, r, SolverMethod::LUDecomposition
        );
        Matrix<U, Dynamic, 1> mu_sub(num_nodes - 1);
        mu_sub(Eigen::seq(0, t - 1)) = mu(Eigen::seq(0, t - 1)); 
        mu_sub(Eigen::seq(t, num_nodes - 2)) = mu(Eigen::seq(t + 1, num_nodes - 1)); 
        for (int i = 0; i < r; ++i)
            mu_sub = (sublaplacian * mu_sub).eval();
        for (int i = 0; i < num_nodes - 1; ++i)
            REQUIRE_THAT(
                static_cast<double>(mu_sub(i)),
                Catch::Matchers::WithinAbs(boost::math::factorial<double>(r), eps)
            );
    }
}

/**
 * Test `LabeledDigraph<...>::getFPTMomentsFromRecurrence()`.
 *
 * Since this method assumes that the input graph has a single terminal node, 
 * this test function assumes the same. 
 */
template <typename T, typename U>
void testGetFPTMomentsFromRecurrence(LabeledDigraph<T, U>* graph,
                                     const int num_nodes,
                                     std::vector<std::string>& node_ids, 
                                     std::string target_id,
                                     const Ref<const Matrix<U, Dynamic, Dynamic> >& laplacian, 
                                     const double eps)
{
    // Compute the 0-th moments and check that they are all one 
    Matrix<U, Dynamic, 1> mu0 = graph->template getFPTMomentsFromRecurrence<U>(target_id, 0);
    for (int i = 0; i < num_nodes; ++i)
        REQUIRE_THAT(static_cast<double>(mu0(i)), Catch::Matchers::WithinAbs(1.0, eps));

    // Compute the r-th moments, for r = 1, ..., 10, and check that they
    // satisfy the corresponding linear equation
    const int t = std::distance(
        node_ids.begin(), std::find(node_ids.begin(), node_ids.end(), target_id)
    );
    Matrix<U, Dynamic, Dynamic> sublaplacian = getSubmatrix<U>(
        laplacian, std::unordered_set<int>{t}, std::unordered_set<int>{t}
    );
    for (int r = 1; r < 11; ++r)
    {
        Matrix<U, Dynamic, 1> mu = graph->template getFPTMomentsFromRecurrence<U>(target_id, r);
        Matrix<U, Dynamic, 1> mu_sub(num_nodes - 1);
        mu_sub(Eigen::seq(0, t - 1)) = mu(Eigen::seq(0, t - 1)); 
        mu_sub(Eigen::seq(t, num_nodes - 2)) = mu(Eigen::seq(t + 1, num_nodes - 1)); 
        for (int i = 0; i < r; ++i)
            mu_sub = (sublaplacian * mu_sub).eval();
        for (int i = 0; i < num_nodes - 1; ++i)
            REQUIRE_THAT(
                static_cast<double>(mu_sub(i)),
                Catch::Matchers::WithinAbs(boost::math::factorial<double>(r), eps)
            );
    }
}

// ----------------------------------------------------------------------- // 
//                           SOME EXAMPLE GRAPHS                           //
// ----------------------------------------------------------------------- //
template <typename T, typename U>
using GraphData = std::tuple<LabeledDigraph<T, U>*, 
                             int,
                             std::vector<std::string>, 
                             std::vector<std::pair<std::string, std::string> >, 
                             std::vector<U>, 
                             std::pair<std::string, std::string>,
                             Matrix<U, Dynamic, Dynamic>,
                             int,
                             std::vector<std::string> >;

/**
 * Return an instance of the five-node reversible cycle graph, with integer 
 * values for the edge labels (in the given scalar types), together with all 
 * required information regarding the graph (nodes, edges, Laplacian matrix, 
 * etc.). 
 */
template <typename T, typename U>
GraphData<T, U> cycleGraph()
{
    LabeledDigraph<T, U>* graph = new LabeledDigraph<T, U>(); 
    graph->addNode("first"); 
    graph->addNode("second");
    graph->addNode("third"); 
    graph->addNode("fourth"); 
    graph->addNode("fifth"); 
    graph->addEdge("first", "second", 1);
    graph->addEdge("first", "fifth", 2);
    graph->addEdge("second", "first", 3);
    graph->addEdge("second", "third", 4); 
    graph->addEdge("third", "second", 5); 
    graph->addEdge("third", "fourth", 6); 
    graph->addEdge("fourth", "third", 7); 
    graph->addEdge("fourth", "fifth", 8); 
    graph->addEdge("fifth", "first", 9);
    graph->addEdge("fifth", "fourth", 10);
    Matrix<U, Dynamic, Dynamic> laplacian(5, 5); 
    laplacian << 1 + 2,    -1,     0,     0,     -2,
                  -3,   3 + 4,    -4,     0,      0,
                   0,      -5, 5 + 6,    -6,      0,
                   0,       0,    -7, 7 + 8,     -8,
                  -9,       0,     0,   -10, 9 + 10;
   
    return std::make_tuple(
        graph, 5,
        std::vector<std::string>{"first", "second", "third", "fourth", "fifth"},
        std::vector<std::pair<std::string, std::string> >{
            std::make_pair("first", "second"),
            std::make_pair("first", "fifth"),
            std::make_pair("second", "first"), 
            std::make_pair("second", "third"), 
            std::make_pair("third", "second"),
            std::make_pair("third", "fourth"),
            std::make_pair("fourth", "third"),
            std::make_pair("fourth", "fifth"), 
            std::make_pair("fifth", "first"),
            std::make_pair("fifth", "fourth")
        },
        std::vector<U>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        std::make_pair("second", "fourth"),
        laplacian, 1,
        std::vector<std::string>{}
    ); 
}

/**
 * Return an instance of the five-node graph in the FPT paper, with integer 
 * values for the edge labels (in the given scalar types), together with all 
 * required information regarding the graph (nodes, edges, Laplacian matrix, 
 * etc.). 
 */
template <typename T, typename U>
GraphData<T, U> fiveNodeGraph()
{
    LabeledDigraph<T, U>* graph = new LabeledDigraph<T, U>(); 
    graph->addNode("first"); 
    graph->addNode("second");
    graph->addNode("third"); 
    graph->addNode("fourth"); 
    graph->addNode("fifth"); 
    graph->addEdge("first", "second", 1); 
    graph->addEdge("second", "first", 2);
    graph->addEdge("second", "third", 3); 
    graph->addEdge("second", "fifth", 4); 
    graph->addEdge("third", "second", 5); 
    graph->addEdge("third", "fourth", 6); 
    graph->addEdge("third", "fifth", 7); 
    graph->addEdge("fourth", "second", 8); 
    graph->addEdge("fourth", "third", 9);
    graph->addEdge("fourth", "fifth", 10); 
    graph->addEdge("fifth", "first", 11);
    Matrix<U, Dynamic, Dynamic> laplacian(5, 5); 
    laplacian <<   1,        -1,         0,          0,   0,
                  -2, 2 + 3 + 4,        -3,          0,  -4,
                   0,        -5, 5 + 6 + 7,         -6,  -7,
                   0,        -8,        -9, 8 + 9 + 10, -10,
                 -11,         0,         0,          0,  11;
   
    return std::make_tuple(
        graph, 5,
        std::vector<std::string>{"first", "second", "third", "fourth", "fifth"},
        std::vector<std::pair<std::string, std::string> >{
            std::make_pair("first", "second"), 
            std::make_pair("second", "first"), 
            std::make_pair("second", "third"), 
            std::make_pair("second", "fifth"),
            std::make_pair("third", "second"),
            std::make_pair("third", "fourth"),
            std::make_pair("third", "fifth"),
            std::make_pair("fourth", "second"),
            std::make_pair("fourth", "third"),
            std::make_pair("fourth", "fifth"), 
            std::make_pair("fifth", "first")
        },
        std::vector<U>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        std::make_pair("first", "fifth"),
        laplacian, 1,
        std::vector<std::string>{}
    ); 
}

/**
 * Return an instance of the six-node graph with one terminal node in the FPT
 * paper, with integer values for the edge labels (in the given scalar types),
 * together with all required information regarding the graph (nodes, edges,
 * Laplacian matrix, etc.). 
 */
template <typename T, typename U>
GraphData<T, U> sixNodeGraph()
{
    LabeledDigraph<T, U>* graph = new LabeledDigraph<T, U>(); 
    graph->addNode("first"); 
    graph->addNode("second");
    graph->addNode("third"); 
    graph->addNode("fourth"); 
    graph->addNode("fifth");
    graph->addNode("sixth");
    graph->addEdge("first", "second", 1);
    graph->addEdge("first", "sixth", 2); 
    graph->addEdge("second", "first", 3); 
    graph->addEdge("second", "third", 4); 
    graph->addEdge("third", "fourth", 5);
    graph->addEdge("third", "sixth", 6);
    graph->addEdge("fourth", "second", 7); 
    graph->addEdge("fourth", "fifth", 8); 
    graph->addEdge("fourth", "sixth", 9); 
    graph->addEdge("fifth", "first", 10); 
    graph->addEdge("fifth", "second", 11);
    graph->addEdge("fifth", "sixth", 12);
    Matrix<U, Dynamic, Dynamic> laplacian(6, 6);
    laplacian << 1 + 2,    -1,     0,         0,            0,  -2,
                    -3, 3 + 4,    -4,         0,            0,   0,
                     0,     0, 5 + 6,        -5,            0,  -6,
                     0,    -7,     0, 7 + 8 + 9,           -8,  -9,
                   -10,   -11,     0,         0, 10 + 11 + 12, -12,
                     0,     0,     0,         0,            0,   0;
   
    return std::make_tuple(
        graph, 6,
        std::vector<std::string>{
            "first", "second", "third", "fourth", "fifth", "sixth"
        },
        std::vector<std::pair<std::string, std::string> >{
            std::make_pair("first", "second"),
            std::make_pair("first", "sixth"), 
            std::make_pair("second", "first"), 
            std::make_pair("second", "third"), 
            std::make_pair("third", "fourth"),
            std::make_pair("third", "sixth"),
            std::make_pair("fourth", "second"), 
            std::make_pair("fourth", "fifth"), 
            std::make_pair("fourth", "sixth"), 
            std::make_pair("fifth", "first"), 
            std::make_pair("fifth", "second"),
            std::make_pair("fifth", "sixth")
        },
        std::vector<U>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
        std::make_pair("fifth", "fourth"),
        laplacian, 1,
        std::vector<std::string>{"sixth"}
    ); 
}

/**
 * Return an instance of a one-exit pipeline graph on 11 nodes (10 non-terminal
 * and one terminal), with integer values for the edge labels (in the given
 * scalar types), together with all required information regarding the graph 
 * (nodes, edges, Laplacian matrix, etc.). 
 */
template <typename T, typename U>
GraphData<T, U> oneExitPipelineGraph()
{
    LabeledDigraph<T, U>* graph = new LabeledDigraph<T, U>(); 
    graph->addNode("first"); 
    graph->addNode("second");
    graph->addNode("third"); 
    graph->addNode("fourth"); 
    graph->addNode("fifth");
    graph->addNode("sixth");
    graph->addNode("seventh");
    graph->addNode("eighth"); 
    graph->addNode("ninth"); 
    graph->addNode("tenth"); 
    graph->addNode("eleventh"); 
    graph->addEdge("first", "second", 1);
    graph->addEdge("second", "first", 2); 
    graph->addEdge("second", "third", 3);
    graph->addEdge("third", "second", 4); 
    graph->addEdge("third", "fourth", 5); 
    graph->addEdge("fourth", "third", 6); 
    graph->addEdge("fourth", "fifth", 7);
    graph->addEdge("fifth", "fourth", 8);
    graph->addEdge("fifth", "sixth", 9);
    graph->addEdge("sixth", "fifth", 10);
    graph->addEdge("sixth", "seventh", 11);
    graph->addEdge("seventh", "sixth", 12);
    graph->addEdge("seventh", "eighth", 13);
    graph->addEdge("eighth", "seventh", 14);
    graph->addEdge("eighth", "ninth", 15);
    graph->addEdge("ninth", "eighth", 16);
    graph->addEdge("ninth", "tenth", 17);
    graph->addEdge("tenth", "ninth", 18);
    graph->addEdge("tenth", "eleventh", 19);
    Matrix<U, Dynamic, Dynamic> laplacian = Matrix<U, Dynamic, Dynamic>::Zero(11, 11);
    U label = 1;
    for (int i = 0; i < 10; ++i)
    {
        U row_sum = 0; 
        if (i > 0)
        {
            laplacian(i, i - 1) = -label;
            row_sum += label; 
            label += 1;
        }
        laplacian(i, i + 1) = -label; 
        row_sum += label; 
        label += 1;
        laplacian(i, i) = row_sum; 
    }

    return std::make_tuple(
        graph, 11,
        std::vector<std::string>{
            "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
            "eighth", "ninth", "tenth", "eleventh"
        },
        std::vector<std::pair<std::string, std::string> >{
            std::make_pair("first", "second"),
            std::make_pair("second", "first"), 
            std::make_pair("second", "third"),
            std::make_pair("third", "second"), 
            std::make_pair("third", "fourth"), 
            std::make_pair("fourth", "third"), 
            std::make_pair("fourth", "fifth"),
            std::make_pair("fifth", "fourth"),
            std::make_pair("fifth", "sixth"),
            std::make_pair("sixth", "fifth"),
            std::make_pair("sixth", "seventh"),
            std::make_pair("seventh", "sixth"),
            std::make_pair("seventh", "eighth"),
            std::make_pair("eighth", "seventh"),
            std::make_pair("eighth", "ninth"),
            std::make_pair("ninth", "eighth"),
            std::make_pair("ninth", "tenth"),
            std::make_pair("tenth", "ninth"),
            std::make_pair("tenth", "eleventh")
        },
        std::vector<U>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
        std::make_pair("fifth", "ninth"),
        laplacian, 1,
        std::vector<std::string>{"eleventh"}
    ); 
}

/**
 * Return an instance of a two-exit pipeline graph on 12 nodes (10 non-terminal
 * and two terminal), with integer values for the edge labels (in the given
 * scalar types), together with all required information regarding the graph 
 * (nodes, edges, Laplacian matrix, etc.). 
 */
template <typename T, typename U>
GraphData<T, U> twoExitPipelineGraph()
{
    LabeledDigraph<T, U>* graph = new LabeledDigraph<T, U>();
    graph->addNode("zeroth"); 
    graph->addNode("first"); 
    graph->addNode("second");
    graph->addNode("third"); 
    graph->addNode("fourth"); 
    graph->addNode("fifth");
    graph->addNode("sixth");
    graph->addNode("seventh");
    graph->addNode("eighth"); 
    graph->addNode("ninth"); 
    graph->addNode("tenth"); 
    graph->addNode("eleventh");
    graph->addEdge("first", "zeroth", 1); 
    graph->addEdge("first", "second", 2);
    graph->addEdge("second", "first", 3); 
    graph->addEdge("second", "third", 4);
    graph->addEdge("third", "second", 5); 
    graph->addEdge("third", "fourth", 6); 
    graph->addEdge("fourth", "third", 7); 
    graph->addEdge("fourth", "fifth", 8);
    graph->addEdge("fifth", "fourth", 9);
    graph->addEdge("fifth", "sixth", 10);
    graph->addEdge("sixth", "fifth", 11);
    graph->addEdge("sixth", "seventh", 12);
    graph->addEdge("seventh", "sixth", 13);
    graph->addEdge("seventh", "eighth", 14);
    graph->addEdge("eighth", "seventh", 15);
    graph->addEdge("eighth", "ninth", 16);
    graph->addEdge("ninth", "eighth", 17);
    graph->addEdge("ninth", "tenth", 18);
    graph->addEdge("tenth", "ninth", 19);
    graph->addEdge("tenth", "eleventh", 20);
    Matrix<U, Dynamic, Dynamic> laplacian = Matrix<U, Dynamic, Dynamic>::Zero(12, 12);
    U label = 1;
    for (int i = 1; i < 11; ++i)
    {
        U row_sum = 0; 
        laplacian(i, i - 1) = -label;
        row_sum += label; 
        label += 1;
        laplacian(i, i + 1) = -label; 
        row_sum += label; 
        label += 1;
        laplacian(i, i) = row_sum; 
    }

    return std::make_tuple(
        graph, 12,
        std::vector<std::string>{
            "zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
            "seventh", "eighth", "ninth", "tenth", "eleventh"
        },
        std::vector<std::pair<std::string, std::string> >{
            std::make_pair("first", "zeroth"),
            std::make_pair("first", "second"),
            std::make_pair("second", "first"), 
            std::make_pair("second", "third"),
            std::make_pair("third", "second"), 
            std::make_pair("third", "fourth"), 
            std::make_pair("fourth", "third"), 
            std::make_pair("fourth", "fifth"),
            std::make_pair("fifth", "fourth"),
            std::make_pair("fifth", "sixth"),
            std::make_pair("sixth", "fifth"),
            std::make_pair("sixth", "seventh"),
            std::make_pair("seventh", "sixth"),
            std::make_pair("seventh", "eighth"),
            std::make_pair("eighth", "seventh"),
            std::make_pair("eighth", "ninth"),
            std::make_pair("ninth", "eighth"),
            std::make_pair("ninth", "tenth"),
            std::make_pair("tenth", "ninth"),
            std::make_pair("tenth", "eleventh")
        },
        std::vector<U>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
        std::make_pair("tenth", "zeroth"),
        laplacian, 2,
        std::vector<std::string>{"zeroth", "eleventh"}
    ); 
}

/**
 * Return an instance of the butterfly graph in the FPT paper, with integer 
 * values for the edge labels (in the given scalar types), together with all 
 * required information regarding the graph (nodes, edges, Laplacian matrix, 
 * etc.). 
 */
template <typename T, typename U>
GraphData<T, U> butterflyGraph()
{
    LabeledDigraph<T, U>* graph = new LabeledDigraph<T, U>(); 
    graph->addNode("first"); 
    graph->addNode("second");
    graph->addNode("third"); 
    graph->addNode("fourth"); 
    graph->addNode("fifth");
    graph->addNode("sixth");
    graph->addNode("seventh"); 
    graph->addEdge("first", "second", 1);
    graph->addEdge("first", "third", 2); 
    graph->addEdge("first", "fourth", 3); 
    graph->addEdge("first", "fifth", 4); 
    graph->addEdge("second", "first", 5);
    graph->addEdge("second", "third", 6);
    graph->addEdge("second", "sixth", 7); 
    graph->addEdge("third", "first", 8); 
    graph->addEdge("third", "second", 9); 
    graph->addEdge("fourth", "first", 10); 
    graph->addEdge("fourth", "fifth", 11);
    graph->addEdge("fourth", "seventh", 12); 
    graph->addEdge("fifth", "first", 13); 
    graph->addEdge("fifth", "fourth", 14);
    Matrix<U, Dynamic, Dynamic> laplacian(7, 7);
    laplacian << 1 + 2 + 3 + 4,        -1,    -2,           -3,      -4,  0,   0,
                            -5, 5 + 6 + 7,    -6,            0,       0, -7,   0,
                            -8,        -9, 8 + 9,            0,       0,  0,   0,
                           -10,         0,     0, 10 + 11 + 12,     -11,  0, -12,
                           -13,         0,     0,          -14, 13 + 14,  0,   0,
                             0,         0,     0,            0,       0,  0,   0,
                             0,         0,     0,            0,       0,  0,   0;
   
    return std::make_tuple(
        graph, 7,
        std::vector<std::string>{
            "first", "second", "third", "fourth", "fifth", "sixth", "seventh"
        },
        std::vector<std::pair<std::string, std::string> >{
            std::make_pair("first", "second"), 
            std::make_pair("first", "third"), 
            std::make_pair("first", "fourth"), 
            std::make_pair("first", "fifth"), 
            std::make_pair("second", "first"),
            std::make_pair("second", "third"),
            std::make_pair("second", "sixth"), 
            std::make_pair("third", "first"), 
            std::make_pair("third", "second"), 
            std::make_pair("fourth", "first"), 
            std::make_pair("fourth", "fifth"),
            std::make_pair("fourth", "seventh"), 
            std::make_pair("fifth", "first"), 
            std::make_pair("fifth", "fourth")
        },
        std::vector<U>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
        std::make_pair("first", "seventh"),
        laplacian, 2,
        std::vector<std::string>{"sixth", "seventh"}
    ); 
}

// ----------------------------------------------------------------------- // 
//                         TEST CASES DECLARATIONS                         // 
// ----------------------------------------------------------------------- // 
TEST_CASE("Tests for node methods")
{
    typedef PreciseType T;

    // Run the tests for all graphs defined above ...
    std::vector<std::function<GraphData<T, T>()> > graph_funcs {
        cycleGraph<T, T>,
        fiveNodeGraph<T, T>,
        sixNodeGraph<T, T>,
        oneExitPipelineGraph<T, T>,
        twoExitPipelineGraph<T, T>, 
        butterflyGraph<T, T>
    }; 
    for (auto&& func : graph_funcs)
    {
        auto result = func();
        LabeledDigraph<T, T>* graph = std::get<0>(result); 
        testNodeMethods<T, T>(graph, std::get<1>(result), std::get<2>(result));
        delete graph;
    }
}

TEST_CASE("Tests for edge methods")
{
    typedef PreciseType T; 

    // Run the tests for all graphs defined above ...
    std::vector<std::function<GraphData<T, T>()> > graph_funcs {
        cycleGraph<T, T>,
        fiveNodeGraph<T, T>,
        sixNodeGraph<T, T>,
        oneExitPipelineGraph<T, T>,
        twoExitPipelineGraph<T, T>, 
        butterflyGraph<T, T>
    }; 
    for (auto&& func : graph_funcs)
    {
        auto result = func();
        LabeledDigraph<T, T>* graph = std::get<0>(result); 
        testEdgeMethods<T, T>(
            graph, std::get<1>(result), std::get<2>(result), std::get<3>(result),
            std::get<4>(result), std::get<5>(result), 1e-50
        ); 
        delete graph;
    }
}

TEST_CASE("Tests for clear()")
{
    typedef PreciseType T; 

    // Run the tests for all graphs defined above ... 
    std::vector<std::function<GraphData<T, T>()> > graph_funcs {
        cycleGraph<T, T>,
        fiveNodeGraph<T, T>,
        sixNodeGraph<T, T>,
        oneExitPipelineGraph<T, T>,
        twoExitPipelineGraph<T, T>,
        butterflyGraph<T, T>
    }; 
    for (auto&& func : graph_funcs)
    {
        auto result = func();
        LabeledDigraph<T, T>* graph = std::get<0>(result); 
        testClear<T, T>(graph, std::get<2>(result));
        delete graph;
    }
}

TEST_CASE("Tests for getLaplacian()")
{
    typedef PreciseType T; 

    // Run the tests for all graphs defined above ... 
    std::vector<std::function<GraphData<T, T>()> > graph_funcs {
        cycleGraph<T, T>,
        fiveNodeGraph<T, T>,
        sixNodeGraph<T, T>,
        oneExitPipelineGraph<T, T>,
        twoExitPipelineGraph<T, T>,
        butterflyGraph<T, T>
    }; 
    for (auto&& func : graph_funcs)
    {
        auto result = func();
        LabeledDigraph<T, T>* graph = std::get<0>(result); 
        testGetLaplacian<T, T>(graph, std::get<1>(result), std::get<6>(result), 1e-50);
        delete graph;
    }
}

TEST_CASE("Tests for getSpanningForestMatrix()")
{
    typedef PreciseType T; 

    // Run the following tests:
    //
    // - testGetSpanningForestMatrixProperties() for all graphs defined above 
    // - testGetSpanningTreeMatrixStronglyConnected() for strongly connected
    //   graphs
    // - testGetSpanningForestMatrixTerminalNodes() for all graphs with terminal
    //   nodes
    //
    // For all strongly connected graphs ... 
    std::vector<std::function<GraphData<T, T>()> > strongly_connected_graph_funcs {
        cycleGraph<T, T>,
        fiveNodeGraph<T, T>
    };
    for (auto&& func : strongly_connected_graph_funcs)
    {
        auto result = func();
        LabeledDigraph<T, T>* graph = std::get<0>(result); 
        testGetSpanningForestMatrixProperties<T, T>(
            graph, std::get<1>(result), std::get<6>(result), std::get<7>(result),
            1e-50
        );
        testGetSpanningTreeMatrixStronglyConnected<T, T>(
            graph, std::get<1>(result), std::get<6>(result), 1e-50
        );
        delete graph;
    }
    
    // For all graphs with terminal nodes ... 
    std::vector<std::function<GraphData<T, T>()> > terminal_node_graph_funcs {
        sixNodeGraph<T, T>,
        oneExitPipelineGraph<T, T>,
        twoExitPipelineGraph<T, T>,
        butterflyGraph<T, T>
    };
    for (auto&& func : terminal_node_graph_funcs)
    {
        auto result = func();
        LabeledDigraph<T, T>* graph = std::get<0>(result); 
        testGetSpanningForestMatrixProperties<T, T>(
            graph, std::get<1>(result), std::get<6>(result), std::get<7>(result),
            1e-50
        ); 
        testGetSpanningForestMatrixTerminalNodes<T, T>(
            graph, std::get<1>(result), std::get<2>(result), std::get<6>(result),
            std::get<8>(result), 1e-50
        );
        delete graph;
    }
}

TEST_CASE("Tests for getSteadyStateFromSVD()")
{
    typedef PreciseType T; 

    // Run the tests for all strongly connected graphs ...
    std::vector<std::function<GraphData<T, T>()> > strongly_connected_graph_funcs {
        cycleGraph<T, T>,
        fiveNodeGraph<T, T>
    };
    for (auto&& func : strongly_connected_graph_funcs)
    {
        auto result = func();
        LabeledDigraph<T, T>* graph = std::get<0>(result); 
        testGetSteadyStateFromSVD<T, T>(graph, std::get<6>(result), 1e-50);
        delete graph;
    }
}

TEST_CASE("Tests for getSteadyStateFromRecurrence()")
{
    typedef PreciseType T; 

    // Run the tests for all strongly connected graphs ...
    std::vector<std::function<GraphData<T, T>()> > strongly_connected_graph_funcs {
        cycleGraph<T, T>,
        fiveNodeGraph<T, T>
    };
    for (auto&& func : strongly_connected_graph_funcs)
    {
        auto result = func();
        LabeledDigraph<T, T>* graph = std::get<0>(result); 
        testGetSteadyStateFromRecurrence<T, T>(graph, std::get<6>(result), 1e-50);
        delete graph;
    }
}

TEST_CASE("Tests for getFPTMomentsFromSolver()")
{
    typedef PreciseType T; 

    // Run the tests for all graphs with a single terminal node ...
    std::vector<std::function<GraphData<T, T>()> > terminal_node_graph_funcs {
        sixNodeGraph<T, T>,
        oneExitPipelineGraph<T, T>
    };
    std::vector<std::string> terminal_node_ids {
        "sixth",
        "eleventh"
    }; 
    for (int i = 0; i < terminal_node_graph_funcs.size(); ++i)
    {
        auto result = terminal_node_graph_funcs[i](); 
        LabeledDigraph<T, T>* graph = std::get<0>(result); 
        testGetFPTMomentsFromSolver<T, T>(
            graph, std::get<1>(result), std::get<2>(result), terminal_node_ids[i],
            std::get<6>(result), 1e-50
        );
        delete graph;
    }
}

TEST_CASE("Tests for getFPTMomentsFromRecurrence()")
{
    typedef PreciseType T; 

    // Run the tests for all graphs with a single terminal node ...
    std::vector<std::function<GraphData<T, T>()> > terminal_node_graph_funcs {
        sixNodeGraph<T, T>,
        oneExitPipelineGraph<T, T>
    };
    std::vector<std::string> terminal_node_ids {
        "sixth",
        "eleventh"
    };
    for (int i = 0; i < terminal_node_graph_funcs.size(); ++i)
    {
        auto result = terminal_node_graph_funcs[i](); 
        LabeledDigraph<T, T>* graph = std::get<0>(result); 
        testGetFPTMomentsFromRecurrence<T, T>(
            graph, std::get<1>(result), std::get<2>(result), terminal_node_ids[i],
            std::get<6>(result), 1e-50
        );
        delete graph;
    }
}

