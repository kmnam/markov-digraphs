#include <iostream>
#include <assert.h>
#include <string>
#include <sstream>
#include <array>
#include <iomanip>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include "../../include/digraph.hpp"

/**
 * Test module for the `LabeledDigraph` class.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/12/2021
 */

using namespace Eigen;
using boost::multiprecision::number; 
using boost::multiprecision::mpfr_float_backend;
typedef number<mpfr_float_backend<100> > PreciseType; 

/** 
 * Test the following methods:
 * - `LabeledDigraph<...>::addNode()` 
 * - `LabeledDigraph<...>::removeNode()`
 * - `LabeledDigraph<...>::hasNode()`
 * - `LabeledDigraph<...>::getNumNodes()`
 * - `LabeledDigraph<...>::getAllNodeIds()`
 */
void TEST_MODULE_NODE_METHODS()
{
    // Instantiate a graph, add three nodes, and check that this->numnodes
    // increments correctly
    LabeledDigraph<double, double>* graph = new LabeledDigraph<double, double>(); 
    graph->addNode("first"); 
    assert(graph->getNumNodes() == 1); 
    graph->addNode("second"); 
    assert(graph->getNumNodes() == 2); 
    graph->addNode("third"); 
    assert(graph->getNumNodes() == 3); 

    // Check that the list of node IDs obeys the order in which the nodes 
    // were added
    std::vector<std::string> node_ids = graph->getAllNodeIds();
    assert(node_ids.size() == 3);  
    assert(node_ids[0] == "first");
    assert(node_ids[1] == "second");
    assert(node_ids[2] == "third");

    // Remove the second node, and check that this->numnodes decrements
    // correctly
    graph->removeNode("second"); 
    assert(graph->getNumNodes() == 2);
    
    // Check that the list of node IDs obeys the order in which the nodes 
    // were added
    node_ids = graph->getAllNodeIds(); 
    assert(node_ids.size() == 2); 
    assert(node_ids[0] == "first"); 
    assert(node_ids[1] == "third");

    // Try to remove a non-existent node; an exception should be raised
    try
    {
        graph->removeNode("this node doesn't exist");
        assert(false);   // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e)
    {
        // Do nothing if an exception is raised
    }

    // Check that hasNode() returns true/false for nodes in the graph and 
    // nodes not in the graph, respectively
    assert(graph->hasNode("first")); 
    assert(graph->hasNode("third")); 
    assert(!graph->hasNode("second")); 
    assert(!graph->hasNode("here's another node that doesn't exist"));

    delete graph; 
}

/** 
 * Test the following methods:
 * - `LabeledDigraph<...>::addEdge()` 
 * - `LabeledDigraph<...>::removeEdge()`
 * - `LabeledDigraph<...>::hasEdge()`
 * - `LabeledDigraph<...>::getEdgeLabel()`
 * - `LabeledDigraph<...>::setEdgeLabel()`
 */
void TEST_MODULE_EDGE_METHODS()
{
    // Instantiate a graph, add three nodes and three edges, and check that
    // hasEdge() returns true/false for edges in the graph and edges not in 
    // the graph, respectively 
    LabeledDigraph<double, double>* graph = new LabeledDigraph<double, double>(); 
    graph->addNode("first"); 
    graph->addNode("second"); 
    graph->addNode("third");
    graph->addEdge("first", "third", 1); 
    graph->addEdge("third", "first", 2); 
    graph->addEdge("second", "third", 3);
    assert(graph->hasEdge("first", "third")); 
    assert(graph->hasEdge("second", "third")); 
    assert(graph->hasEdge("third", "first"));
    assert(!graph->hasEdge("first", "second")); 
    assert(!graph->hasEdge("second", "first")); 
    assert(!graph->hasEdge("third", "second"));

    // Check that hasEdge() returns false when either of the queried nodes 
    // does not exist
    assert(!graph->hasEdge("first", "here's one node that doesn't exist")); 
    assert(!graph->hasEdge("here's another node that isn't there", "second")); 
    assert(!graph->hasEdge("one more non-existent node", "another non-existent node"));

    // Check that hasEdge() returns false for all possible self-loops
    assert(!graph->hasEdge("first", "first")); 
    assert(!graph->hasEdge("second", "second")); 
    assert(!graph->hasEdge("third", "third")); 

    // Try to remove an edge with non-existent source/target nodes
    try
    {
        graph->removeEdge("first", "yet another non-existent node"); 
        assert(false);    // Break here if an exception is not raised
    } 
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised
    try
    {
        graph->removeEdge("one more non-existent node, as before", "second"); 
        assert(false);    // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised
    try
    {
        graph->removeEdge("to dance beneath the diamond sky", "with one hand waving free"); 
        assert(false);    // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised

    // Remove an edge in the graph and check that hasEdge() returns false 
    graph->removeEdge("first", "third"); 
    assert(!graph->hasEdge("first", "third"));

    // Add two more nodes and three more edges, for the sake of variety; then  
    // remove a node in the graph; then check that hasEdge() returns false 
    // for all edges involving the removed node
    graph->addNode("fourth"); 
    graph->addNode("fifth"); 
    graph->addEdge("second", "fourth", 50); 
    graph->addEdge("fifth", "first", 100); 
    graph->addEdge("third", "fifth", 1000); 
    graph->removeNode("first"); 
    assert(graph->getNumNodes() == 4); 
    assert(!graph->hasEdge("third", "first"));   // This edge was just removed
    assert(!graph->hasEdge("fifth", "first"));   // This edge was just removed
    assert(!graph->hasEdge("first", "third"));   // This edge was removed in the previous block

    // Check that hasEdge() returns true for all edges remaining 
    assert(graph->hasEdge("second", "third")); 
    assert(graph->hasEdge("second", "fourth")); 
    assert(graph->hasEdge("third", "fifth"));

    // Check that hasEdge() returns false for edges that weren't there to begin with
    assert(!graph->hasEdge("second", "fifth"));   // This edge never existed

    // Check that getEdgeLabel() returns the correct values for edges that exist 
    assert(graph->getEdgeLabel("second", "third") == 3); 
    assert(graph->getEdgeLabel("second", "fourth") == 50); 
    assert(graph->getEdgeLabel("third", "fifth") == 1000);

    // Check that getEdgeLabel() raises an exception if the queried edge does not exist
    try
    {
        graph->getEdgeLabel("second", "fifth"); 
        assert(false);    // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised

    // Check that getEdgeLabel() raises an exception if either of the queried 
    // nodes doesn't exist 
    try
    {
        graph->getEdgeLabel("first", "fifth");    // "first" does not exist anymore 
        assert(false);    // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised
    try
    {
        graph->getEdgeLabel("second", "here's another node that never existed"); 
        assert(false);    // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised
    try
    {
        graph->getEdgeLabel("get jailed, jump bail", "join the army if you fail"); 
        assert(false);    // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised

    // Check that setEdgeLabel() changes the desired edge label 
    graph->setEdgeLabel("second", "fourth", 75); 
    assert(graph->getEdgeLabel("second", "fourth") == 75); 

    // Check that all other edge labels remain as they were 
    assert(graph->getEdgeLabel("second", "third") == 3);
    assert(graph->getEdgeLabel("third", "fifth") == 1000);

    // Check that setEdgeLabel() raises an exception if the queried edge does not exist
    try
    {
        graph->setEdgeLabel("third", "fourth", 2.5); 
        assert(false);    // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised

    // Check that setEdgeLabel() raises an exception if either of the queried 
    // nodes doesn't exist 
    try
    {
        graph->setEdgeLabel("first", "second", 16);    // "first" does not exist anymore 
        assert(false);    // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised
    try
    {
        graph->setEdgeLabel("second", "here's yet another node that never came into being", 7e+10); 
        assert(false);    // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised
    try
    {
        graph->setEdgeLabel("every one of them words rang true", "and glowed like burning coal", 0.0001); 
        assert(false);    // Break here if an exception is not raised
    }
    catch (const std::runtime_error& e) {}   // Do nothing if an exception is raised

    delete graph; 
}

/**
 * Test `LabeledDigraph<...>::clear()`. 
 */
void TEST_MODULE_CLEAR()
{
    // Add a succession of nodes and edges, then clear, then check that the 
    // graph is empty 
    LabeledDigraph<double, double>* graph = new LabeledDigraph<double, double>(); 
    graph->addNode("first"); 
    graph->addNode("second"); 
    graph->addNode("third");
    graph->addNode("fourth"); 
    graph->addNode("fifth"); 
    graph->addEdge("first", "third", 1); 
    graph->addEdge("third", "first", 2); 
    graph->addEdge("second", "third", 3);
    graph->addEdge("second", "fourth", 50); 
    graph->addEdge("fifth", "first", 100); 
    graph->addEdge("third", "fifth", 1000);
    graph->clear();
    assert(graph->getNumNodes() == 0);
    assert(graph->getAllNodeIds().size() == 0);

    // Check that hasNode() returns false for every previously inserted node
    assert(!graph->hasNode("first")); 
    assert(!graph->hasNode("second")); 
    assert(!graph->hasNode("third")); 
    assert(!graph->hasNode("fourth")); 
    assert(!graph->hasNode("fifth"));

    // Check that removeNode() throws an exception if we try to remove any 
    // (already-removed) node again
    try { graph->removeNode("first"); assert(false); }
    catch (const std::runtime_error& e) {} 
    try { graph->removeNode("second"); assert(false); }
    catch (const std::runtime_error& e) {} 
    try { graph->removeNode("third"); assert(false); }
    catch (const std::runtime_error& e) {}
    try { graph->removeNode("fourth"); assert(false); } 
    catch (const std::runtime_error& e) {} 
    try { graph->removeNode("fifth"); assert(false); }
    catch (const std::runtime_error& e) {}

    // Check that hasEdge() returns false for all possible combinations of 
    // the (already-removed) nodes 
    std::vector<std::string> node_ids = {"first", "second", "third", "fourth", "fifth"}; 
    for (const std::string& s : node_ids)
    {
        for (const std::string& t : node_ids)
        {
            assert(!graph->hasEdge(s, t)); 
        }
    }

    delete graph;  
}

/**
 * Test `LabeledDigraph<...>::getLaplacian()`.
 *
 * With a graph with integer scalars, check that `getLaplacian()` returns an 
 * *exactly* correct Laplacian matrix for a 5-vertex graph.
 */
void TEST_MODULE_GET_LAPLACIAN()
{
    LabeledDigraph<int, int>* graph = new LabeledDigraph<int, int>(); 
    graph->addNode("first"); 
    graph->addNode("second"); 
    graph->addNode("third");
    graph->addNode("fourth"); 
    graph->addNode("fifth"); 
    graph->addEdge("first", "third", 1); 
    graph->addEdge("third", "first", 2); 
    graph->addEdge("second", "third", 3);
    graph->addEdge("second", "fourth", 50); 
    graph->addEdge("fifth", "first", 100); 
    graph->addEdge("third", "fifth", 1000);
    MatrixXi laplacian = graph->getLaplacian();
    assert(laplacian.rows() == 5); 
    assert(laplacian.cols() == 5); 
    for (unsigned i = 0; i < 5; ++i)
    {
        for (unsigned j = 0; j < 5; ++j)
        {
            if (i == 0 && j == 2)         // label(third -> first) == 2
                assert(laplacian(i, j) == 2);
            else if (i == 0 && j == 4)    // label(fifth -> first) == 100
                assert(laplacian(i, j) == 100); 
            else if (i == 2 && j == 0)    // label(first -> third) == 1
                assert(laplacian(i, j) == 1); 
            else if (i == 2 && j == 1)    // label(second -> third) == 3
                assert(laplacian(i, j) == 3);
            else if (i == 3 && j == 1)    // label(second -> fourth) == 50
                assert(laplacian(i, j) == 50); 
            else if (i == 4 && j == 2)    // label(third -> fifth) == 1000
                assert(laplacian(i, j) == 1000); 
            else if (i == 0 && j == 0)    // Sum of all outgoing edges from first == 1
                assert(laplacian(i, j) == -1); 
            else if (i == 1 && j == 1)    // Sum of all outgoing edges from second == 53
                assert(laplacian(i, j) == -53); 
            else if (i == 2 && j == 2)    // Sum of all outgoing edges from third == 1002
                assert(laplacian(i, j) == -1002);
            else if (i == 4 && j == 4)    // Sum of all outgoing edges from fifth == 100
                assert(laplacian(i, j) == -100); 
            else                          // All other entries should be zero
                assert(laplacian(i, j) == 0);  
        }
    }

    delete graph;  
}

/**
 * Test `LabeledDigraph<...>::getSpanningForestMatrix()`. 
 *
 * With a graph with (long) integer scalars, check that `getSpanningForestMatrix()`
 * returns *exactly* correct spanning forest matrices for a 5-vertex graph.
 */
void TEST_MODULE_GET_SPANNING_FOREST_MATRIX()
{
    LabeledDigraph<long, long>* graph = new LabeledDigraph<long, long>(); 
    graph->addNode("first"); 
    graph->addNode("second"); 
    graph->addNode("third");
    graph->addNode("fourth"); 
    graph->addNode("fifth");
    graph->addEdge("first", "second", 1);
    graph->addEdge("first", "fifth", 13); 
    graph->addEdge("second", "first", 20); 
    graph->addEdge("second", "third", 67); 
    graph->addEdge("third", "second", 42); 
    graph->addEdge("third", "fourth", 1007); 
    graph->addEdge("fourth", "third", 17); 
    graph->addEdge("fourth", "fifth", 512); 
    graph->addEdge("fifth", "first", 179); 
    graph->addEdge("fifth", "fourth", 285);

    // Check that the zeroth spanning forest matrix equals the identity
    Matrix<long, Dynamic, Dynamic> forest_zero = graph->getSpanningForestMatrix(0);
    assert(forest_zero.rows() == 5);
    assert(forest_zero.cols() == 5);
    for (unsigned i = 0; i < 5; ++i)
    {
        for (unsigned j = 0; j < 5; ++j)
        {
            if (i == j) assert(forest_zero(i, j) == 1); 
            else        assert(forest_zero(i, j) == 0); 
        }
    }

    // Check that the first spanning forest matrix has the following entries:
    // 1) i != j and abs(i - j) == 1 or abs(i - j) == 4: A(i, j) == label(i -> j)
    // 2) i != j and abs(i - j) == 2 or abs(i - j) == 3: A(i, j) == 0
    // 3) i == j: A(i, j) == sum of all edge labels except for all edges leaving j
    Matrix<long, Dynamic, Dynamic> forest_one = graph->getSpanningForestMatrix(1);
    std::vector<std::string> node_ids = {"first", "second", "third", "fourth", "fifth"}; 
    long sum_of_all_edge_labels = 1 + 13 + 20 + 67 + 42 + 1007 + 17 + 512 + 179 + 285;
    for (unsigned i = 0; i < 5; ++i)
    {
        for (unsigned j = 0; j < 5; ++j)
        {
            if (i == j)
            {
                if (j == 0)
                    assert(
                        forest_one(i, j) == sum_of_all_edge_labels
                        - graph->getEdgeLabel(node_ids[0], node_ids[4])
                        - graph->getEdgeLabel(node_ids[0], node_ids[1])
                    );
                else if (j == 4)
                    assert(
                        forest_one(i, j) == sum_of_all_edge_labels
                        - graph->getEdgeLabel(node_ids[4], node_ids[0])
                        - graph->getEdgeLabel(node_ids[4], node_ids[3])
                    );
                else
                    assert(
                        forest_one(i, j) == sum_of_all_edge_labels
                        - graph->getEdgeLabel(node_ids[j], node_ids[j-1])
                        - graph->getEdgeLabel(node_ids[j], node_ids[j+1])
                    ); 
            }
            else if (i != j && (std::abs((int)(i - j)) == 1 || std::abs((int)(i - j)) == 4))
            {
                assert(forest_one(i, j) == graph->getEdgeLabel(node_ids[i], node_ids[j])); 
            }
            else
            {
                assert(forest_one(i, j) == 0); 
            }
        }
    }

    // Check that the second spanning forest matrix has the correct entries ... 
    Matrix<long, Dynamic, Dynamic> forest_two = graph->getSpanningForestMatrix(2);
    
    // To take advantage of the symmetry of the cycle graph, we introduce a 
    // "node mapping" that allows us to calculate the spanning forest weights
    // without having to repeatedly rewrite equivalent products of edge labels
    std::function<long(const std::unordered_map<std::string, std::string>&)> all_two_edge_forests
        = [graph](const std::unordered_map<std::string, std::string>& node_map)
        {
            std::string _first = node_map.find("first")->second;
            std::string _second = node_map.find("second")->second; 
            std::string _third = node_map.find("third")->second; 
            std::string _fourth = node_map.find("fourth")->second; 
            std::string _fifth = node_map.find("fifth")->second;  
            return (
                graph->getEdgeLabel(_second, _first) * (
                    graph->getEdgeLabel(_third, _second) +
                    graph->getEdgeLabel(_third, _fourth) +
                    graph->getEdgeLabel(_fourth, _third) +
                    graph->getEdgeLabel(_fourth, _fifth) +
                    graph->getEdgeLabel(_fifth, _fourth) +
                    graph->getEdgeLabel(_fifth, _first)
                ) + graph->getEdgeLabel(_second, _third) * (
                    graph->getEdgeLabel(_third, _fourth) +
                    graph->getEdgeLabel(_fourth, _third) +
                    graph->getEdgeLabel(_fourth, _fifth) + 
                    graph->getEdgeLabel(_fifth, _fourth) +
                    graph->getEdgeLabel(_fifth, _first)
                ) + graph->getEdgeLabel(_third, _second) * (
                    graph->getEdgeLabel(_fourth, _third) +
                    graph->getEdgeLabel(_fourth, _fifth) +
                    graph->getEdgeLabel(_fifth, _fourth) +
                    graph->getEdgeLabel(_fifth, _first)
                ) + graph->getEdgeLabel(_third, _fourth) * (
                    graph->getEdgeLabel(_fourth, _fifth) +
                    graph->getEdgeLabel(_fifth, _fourth) +
                    graph->getEdgeLabel(_fifth, _first)
                ) + graph->getEdgeLabel(_fourth, _third) * (
                    graph->getEdgeLabel(_fifth, _fourth) +
                    graph->getEdgeLabel(_fifth, _first)
                ) + graph->getEdgeLabel(_fourth, _fifth) * graph->getEdgeLabel(_fifth, _first)
            ); 
        };
    std::function<long(const std::unordered_map<std::string, std::string>&)> two_edge_forests_with_path_from_1_to_2
        = [graph](const std::unordered_map<std::string, std::string>& node_map)
        {
            std::string _first = node_map.find("first")->second;
            std::string _second = node_map.find("second")->second; 
            std::string _third = node_map.find("third")->second; 
            std::string _fourth = node_map.find("fourth")->second; 
            std::string _fifth = node_map.find("fifth")->second;  
            return graph->getEdgeLabel(_first, _second) * (
                graph->getEdgeLabel(_third, _second) +
                graph->getEdgeLabel(_third, _fourth) +
                graph->getEdgeLabel(_fourth, _third) +
                graph->getEdgeLabel(_fourth, _fifth) +
                graph->getEdgeLabel(_fifth, _fourth) +
                graph->getEdgeLabel(_fifth, _first)
            ); 
        };
    std::function<long(const std::unordered_map<std::string, std::string>&)> two_edge_forests_with_path_from_1_to_5
        = [graph](const std::unordered_map<std::string, std::string>& node_map)
        {
            std::string _first = node_map.find("first")->second;
            std::string _second = node_map.find("second")->second; 
            std::string _third = node_map.find("third")->second; 
            std::string _fourth = node_map.find("fourth")->second; 
            std::string _fifth = node_map.find("fifth")->second;  
            return graph->getEdgeLabel(_first, _fifth) * (
                graph->getEdgeLabel(_second, _first) +
                graph->getEdgeLabel(_second, _third) +
                graph->getEdgeLabel(_third, _second) +
                graph->getEdgeLabel(_third, _fourth) +
                graph->getEdgeLabel(_fourth, _third) +
                graph->getEdgeLabel(_fourth, _fifth)
            ); 
        }; 

    // (0, 0): 2-edge forests with 1 as a root
    const std::unordered_map<std::string, std::string> map_to_first = {
        {"first", "first"}, {"second", "second"}, {"third", "third"}, {"fourth", "fourth"}, {"fifth", "fifth"}
    };  
    assert(forest_two(0, 0) == all_two_edge_forests(map_to_first));  
    // (0, 1): 2-edge forests with 2 as a root and path from 1 to 2
    assert(forest_two(0, 1) == two_edge_forests_with_path_from_1_to_2(map_to_first));
    // (0, 2): 2-edge forests with 3 as a root and path from 1 to 3
    assert(forest_two(0, 2) == graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third"));
    // (0, 3): 2-edge forests with 4 as a root and path from 1 to 4
    assert(forest_two(0, 3) == graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("fifth", "fourth"));
    // (0, 4): 2-edge forests with 5 as a root and path from 1 to 5
    assert(forest_two(0, 4) == two_edge_forests_with_path_from_1_to_5(map_to_first)); 
    // (1, 0): 2-edge forests with 1 as a root and path from 2 to 1
    const std::unordered_map<std::string, std::string> map_to_second = {
        {"first", "second"}, {"second", "third"}, {"third", "fourth"}, {"fourth", "fifth"}, {"fifth", "first"}
    }; 
    assert(forest_two(1, 0) == two_edge_forests_with_path_from_1_to_5(map_to_second));
    // (1, 1): 2-edge forests with 2 as a root
    assert(forest_two(1, 1) == all_two_edge_forests(map_to_second));
    // (1, 2): 2-edge forests with 3 as a root and path from 2 to 3
    assert(forest_two(1, 2) == two_edge_forests_with_path_from_1_to_2(map_to_second));
    // (1, 3): 2-edge forests with 4 as a root and path from 2 to 4
    assert(forest_two(1, 3) == graph->getEdgeLabel("second", "third") * graph->getEdgeLabel("third", "fourth"));
    // (1, 4): 2-edge forests with 5 as a root and path from 2 to 5
    assert(forest_two(1, 4) == graph->getEdgeLabel("second", "first") * graph->getEdgeLabel("first", "fifth"));
    // (2, 0): 2-edge forests with 1 as a root and path from 3 to 1
    const std::unordered_map<std::string, std::string> map_to_third = {
        {"first", "third"}, {"second", "fourth"}, {"third", "fifth"}, {"fourth", "first"}, {"fifth", "second"}
    }; 
    assert(forest_two(2, 0) == graph->getEdgeLabel("third", "second") * graph->getEdgeLabel("second", "first")); 
    // (2, 1): 2-edge forests with 2 as a root and path from 3 to 2
    assert(forest_two(2, 1) == two_edge_forests_with_path_from_1_to_5(map_to_third)); 
    // (2, 2): 2-edge forests with 3 as a root
    assert(forest_two(2, 2) == all_two_edge_forests(map_to_third)); 
    // (2, 3): 2-edge forests with 4 as a root and path from 3 to 4
    assert(forest_two(2, 3) == two_edge_forests_with_path_from_1_to_2(map_to_third)); 
    // (2, 4): 2-edge forests with 5 as a root and path from 3 to 5
    assert(forest_two(2, 4) == graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fourth", "fifth")); 
    // (3, 0): 2-edge forests with 1 as a root and path from 4 to 1
    const std::unordered_map<std::string, std::string> map_to_fourth = {
        {"first", "fourth"}, {"second", "fifth"}, {"third", "first"}, {"fourth", "second"}, {"fifth", "third"}
    }; 
    assert(forest_two(3, 0) == graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first"));
    // (3, 1): 2-edge forests with 2 as a root and path from 4 to 2
    assert(forest_two(3, 1) == graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("third", "second")); 
    // (3, 2): 2-edge forests with 3 as a root and path from 4 to 3
    assert(forest_two(3, 2) == two_edge_forests_with_path_from_1_to_5(map_to_fourth)); 
    // (3, 3): 2-edge forests with 4 as a root
    assert(forest_two(3, 3) == all_two_edge_forests(map_to_fourth)); 
    // (3, 4): 2-edge forests with 5 as a root and path from 4 to 5
    assert(forest_two(3, 4) == two_edge_forests_with_path_from_1_to_2(map_to_fourth));
    // (4, 0): 2-edge forests with 1 as a root and path from 5 to 1
    const std::unordered_map<std::string, std::string> map_to_fifth = {
        {"first", "fifth"}, {"second", "first"}, {"third", "second"}, {"fourth", "third"}, {"fifth", "fourth"}
    }; 
    assert(forest_two(4, 0) == two_edge_forests_with_path_from_1_to_2(map_to_fifth)); 
    // (4, 1): 2-edge forests with 2 as a root and path from 5 to 2
    assert(forest_two(4, 1) == graph->getEdgeLabel("fifth", "first") * graph->getEdgeLabel("first", "second")); 
    // (4, 2): 2-edge forests with 3 as a root and path from 5 to 3
    assert(forest_two(4, 2) == graph->getEdgeLabel("fifth", "fourth") * graph->getEdgeLabel("fourth", "third")); 
    // (4, 3): 2-edge forests with 4 as a root and path from 5 to 4
    assert(forest_two(4, 3) == two_edge_forests_with_path_from_1_to_5(map_to_fifth)); 
    // (4, 4): 2-edge forests with 5 as a root
    assert(forest_two(4, 4) == all_two_edge_forests(map_to_fifth));

    // Skip the third spanning forest matrix, and examine the fourth ...
    Matrix<long, Dynamic, Dynamic> forest_four = graph->getSpanningForestMatrix(4);
    // (0, 0): trees with 1 as a root 
    assert(
        forest_four(0, 0) == (
            graph->getEdgeLabel("second", "first") * graph->getEdgeLabel("third", "second") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("second", "first") * graph->getEdgeLabel("third", "second") * 
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "first") + 
            graph->getEdgeLabel("second", "first") * graph->getEdgeLabel("third", "second") * 
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first") + 
            graph->getEdgeLabel("second", "first") * graph->getEdgeLabel("third", "fourth") * 
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first") + 
            graph->getEdgeLabel("second", "third") * graph->getEdgeLabel("third", "fourth") * 
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first")
        )
    );
    // (0, 1): trees with 2 as a root
    assert(
        forest_four(0, 1) == (
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("third", "fourth") *
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("third", "second") *
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("third", "second") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("third", "second") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("third", "second") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth")
        )
    );
    // (0, 2): trees with 3 as a root
    assert(
        forest_four(0, 2) == (
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth")
        )
    );
    // (0, 3): trees with 4 as a root
    assert(
        forest_four(0, 3) == (
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("third", "second") * graph->getEdgeLabel("fifth", "fourth") 
        )
    );
    // (0, 4): trees with 5 as a root
    assert(
        forest_four(0, 4) == (
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fourth", "fifth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fourth", "fifth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fourth", "fifth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("third", "second") * graph->getEdgeLabel("fourth", "fifth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("third", "second") * graph->getEdgeLabel("fourth", "third")
        )
    );
    // Check that each column has the same entry in each row
    for (unsigned i = 0; i < 5; ++i)
    {
        for (unsigned j = 1; j < 5; ++j)
        {
            assert(forest_four(0, i) == forest_four(j, i));
        }
    } 

    delete graph; 
}

/**
 * Test `LabeledDigraph<...>::getSpanningForestMatrixSparse()`. 
 *
 * With a graph with (long) integer scalars, check that `getSpanningForestMatrixSparse()`
 * returns *exactly* correct spanning forest matrices for a 5-vertex graph.
 *
 * This module performs precisely the same tests as `TEST_MODULE_GET_SPANNING_FOREST_MATRIX()`.
 */
void TEST_MODULE_GET_SPANNING_FOREST_MATRIX_SPARSE()
{
    LabeledDigraph<long, long>* graph = new LabeledDigraph<long, long>(); 
    graph->addNode("first"); 
    graph->addNode("second"); 
    graph->addNode("third");
    graph->addNode("fourth"); 
    graph->addNode("fifth");
    graph->addEdge("first", "second", 1);
    graph->addEdge("first", "fifth", 13); 
    graph->addEdge("second", "first", 20); 
    graph->addEdge("second", "third", 67); 
    graph->addEdge("third", "second", 42); 
    graph->addEdge("third", "fourth", 1007); 
    graph->addEdge("fourth", "third", 17); 
    graph->addEdge("fourth", "fifth", 512); 
    graph->addEdge("fifth", "first", 179); 
    graph->addEdge("fifth", "fourth", 285);

    // Check that the zeroth spanning forest matrix equals the identity
    Matrix<long, Dynamic, Dynamic> forest_zero = graph->getSpanningForestMatrixSparse(0);
    for (unsigned i = 0; i < 5; ++i)
    {
        for (unsigned j = 0; j < 5; ++j)
        {
            if (i == j) assert(forest_zero(i, j) == 1); 
            else        assert(forest_zero(i, j) == 0); 
        }
    }

    // Check that the first spanning forest matrix has the following entries:
    // 1) i != j and abs(i - j) == 1 or abs(i - j) == 4: A(i, j) == label(i -> j)
    // 2) i != j and abs(i - j) == 2 or abs(i - j) == 3: A(i, j) == 0
    // 3) i == j: A(i, j) == sum of all edge labels except for all edges leaving j
    Matrix<long, Dynamic, Dynamic> forest_one = graph->getSpanningForestMatrixSparse(1);
    std::vector<std::string> node_ids = {"first", "second", "third", "fourth", "fifth"}; 
    long sum_of_all_edge_labels = 1 + 13 + 20 + 67 + 42 + 1007 + 17 + 512 + 179 + 285;
    for (unsigned i = 0; i < 5; ++i)
    {
        for (unsigned j = 0; j < 5; ++j)
        {
            if (i == j)
            {
                if (j == 0)
                    assert(
                        forest_one(i, j) == sum_of_all_edge_labels
                        - graph->getEdgeLabel(node_ids[0], node_ids[4])
                        - graph->getEdgeLabel(node_ids[0], node_ids[1])
                    );
                else if (j == 4)
                    assert(
                        forest_one(i, j) == sum_of_all_edge_labels
                        - graph->getEdgeLabel(node_ids[4], node_ids[0])
                        - graph->getEdgeLabel(node_ids[4], node_ids[3])
                    );
                else
                    assert(
                        forest_one(i, j) == sum_of_all_edge_labels
                        - graph->getEdgeLabel(node_ids[j], node_ids[j-1])
                        - graph->getEdgeLabel(node_ids[j], node_ids[j+1])
                    ); 
            }
            else if (i != j && (std::abs((int)(i - j)) == 1 || std::abs((int)(i - j)) == 4))
            {
                assert(forest_one(i, j) == graph->getEdgeLabel(node_ids[i], node_ids[j])); 
            }
            else
            {
                assert(forest_one(i, j) == 0); 
            }
        }
    }

    // Check that the second spanning forest matrix has the correct entries ... 
    Matrix<long, Dynamic, Dynamic> forest_two = graph->getSpanningForestMatrixSparse(2);
    
    // To take advantage of the symmetry of the cycle graph, we introduce a 
    // "node mapping" that allows us to calculate the spanning forest weights
    // without having to repeatedly rewrite equivalent products of edge labels
    std::function<long(const std::unordered_map<std::string, std::string>&)> all_two_edge_forests
        = [graph](const std::unordered_map<std::string, std::string>& node_map)
        {
            std::string _first = node_map.find("first")->second;
            std::string _second = node_map.find("second")->second; 
            std::string _third = node_map.find("third")->second; 
            std::string _fourth = node_map.find("fourth")->second; 
            std::string _fifth = node_map.find("fifth")->second;  
            return (
                graph->getEdgeLabel(_second, _first) * (
                    graph->getEdgeLabel(_third, _second) +
                    graph->getEdgeLabel(_third, _fourth) +
                    graph->getEdgeLabel(_fourth, _third) +
                    graph->getEdgeLabel(_fourth, _fifth) +
                    graph->getEdgeLabel(_fifth, _fourth) +
                    graph->getEdgeLabel(_fifth, _first)
                ) + graph->getEdgeLabel(_second, _third) * (
                    graph->getEdgeLabel(_third, _fourth) +
                    graph->getEdgeLabel(_fourth, _third) +
                    graph->getEdgeLabel(_fourth, _fifth) + 
                    graph->getEdgeLabel(_fifth, _fourth) +
                    graph->getEdgeLabel(_fifth, _first)
                ) + graph->getEdgeLabel(_third, _second) * (
                    graph->getEdgeLabel(_fourth, _third) +
                    graph->getEdgeLabel(_fourth, _fifth) +
                    graph->getEdgeLabel(_fifth, _fourth) +
                    graph->getEdgeLabel(_fifth, _first)
                ) + graph->getEdgeLabel(_third, _fourth) * (
                    graph->getEdgeLabel(_fourth, _fifth) +
                    graph->getEdgeLabel(_fifth, _fourth) +
                    graph->getEdgeLabel(_fifth, _first)
                ) + graph->getEdgeLabel(_fourth, _third) * (
                    graph->getEdgeLabel(_fifth, _fourth) +
                    graph->getEdgeLabel(_fifth, _first)
                ) + graph->getEdgeLabel(_fourth, _fifth) * graph->getEdgeLabel(_fifth, _first)
            ); 
        };
    std::function<long(const std::unordered_map<std::string, std::string>&)> two_edge_forests_with_path_from_1_to_2
        = [graph](const std::unordered_map<std::string, std::string>& node_map)
        {
            std::string _first = node_map.find("first")->second;
            std::string _second = node_map.find("second")->second; 
            std::string _third = node_map.find("third")->second; 
            std::string _fourth = node_map.find("fourth")->second; 
            std::string _fifth = node_map.find("fifth")->second;  
            return graph->getEdgeLabel(_first, _second) * (
                graph->getEdgeLabel(_third, _second) +
                graph->getEdgeLabel(_third, _fourth) +
                graph->getEdgeLabel(_fourth, _third) +
                graph->getEdgeLabel(_fourth, _fifth) +
                graph->getEdgeLabel(_fifth, _fourth) +
                graph->getEdgeLabel(_fifth, _first)
            ); 
        };
    std::function<long(const std::unordered_map<std::string, std::string>&)> two_edge_forests_with_path_from_1_to_5
        = [graph](const std::unordered_map<std::string, std::string>& node_map)
        {
            std::string _first = node_map.find("first")->second;
            std::string _second = node_map.find("second")->second; 
            std::string _third = node_map.find("third")->second; 
            std::string _fourth = node_map.find("fourth")->second; 
            std::string _fifth = node_map.find("fifth")->second;  
            return graph->getEdgeLabel(_first, _fifth) * (
                graph->getEdgeLabel(_second, _first) +
                graph->getEdgeLabel(_second, _third) +
                graph->getEdgeLabel(_third, _second) +
                graph->getEdgeLabel(_third, _fourth) +
                graph->getEdgeLabel(_fourth, _third) +
                graph->getEdgeLabel(_fourth, _fifth)
            ); 
        }; 

    // (0, 0): 2-edge forests with 1 as a root
    const std::unordered_map<std::string, std::string> map_to_first = {
        {"first", "first"}, {"second", "second"}, {"third", "third"}, {"fourth", "fourth"}, {"fifth", "fifth"}
    };  
    assert(forest_two(0, 0) == all_two_edge_forests(map_to_first));  
    // (0, 1): 2-edge forests with 2 as a root and path from 1 to 2
    assert(forest_two(0, 1) == two_edge_forests_with_path_from_1_to_2(map_to_first));
    // (0, 2): 2-edge forests with 3 as a root and path from 1 to 3
    assert(forest_two(0, 2) == graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third"));
    // (0, 3): 2-edge forests with 4 as a root and path from 1 to 4
    assert(forest_two(0, 3) == graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("fifth", "fourth"));
    // (0, 4): 2-edge forests with 5 as a root and path from 1 to 5
    assert(forest_two(0, 4) == two_edge_forests_with_path_from_1_to_5(map_to_first)); 
    // (1, 0): 2-edge forests with 1 as a root and path from 2 to 1
    const std::unordered_map<std::string, std::string> map_to_second = {
        {"first", "second"}, {"second", "third"}, {"third", "fourth"}, {"fourth", "fifth"}, {"fifth", "first"}
    }; 
    assert(forest_two(1, 0) == two_edge_forests_with_path_from_1_to_5(map_to_second));
    // (1, 1): 2-edge forests with 2 as a root
    assert(forest_two(1, 1) == all_two_edge_forests(map_to_second));
    // (1, 2): 2-edge forests with 3 as a root and path from 2 to 3
    assert(forest_two(1, 2) == two_edge_forests_with_path_from_1_to_2(map_to_second));
    // (1, 3): 2-edge forests with 4 as a root and path from 2 to 4
    assert(forest_two(1, 3) == graph->getEdgeLabel("second", "third") * graph->getEdgeLabel("third", "fourth"));
    // (1, 4): 2-edge forests with 5 as a root and path from 2 to 5
    assert(forest_two(1, 4) == graph->getEdgeLabel("second", "first") * graph->getEdgeLabel("first", "fifth"));
    // (2, 0): 2-edge forests with 1 as a root and path from 3 to 1
    const std::unordered_map<std::string, std::string> map_to_third = {
        {"first", "third"}, {"second", "fourth"}, {"third", "fifth"}, {"fourth", "first"}, {"fifth", "second"}
    }; 
    assert(forest_two(2, 0) == graph->getEdgeLabel("third", "second") * graph->getEdgeLabel("second", "first")); 
    // (2, 1): 2-edge forests with 2 as a root and path from 3 to 2
    assert(forest_two(2, 1) == two_edge_forests_with_path_from_1_to_5(map_to_third)); 
    // (2, 2): 2-edge forests with 3 as a root
    assert(forest_two(2, 2) == all_two_edge_forests(map_to_third)); 
    // (2, 3): 2-edge forests with 4 as a root and path from 3 to 4
    assert(forest_two(2, 3) == two_edge_forests_with_path_from_1_to_2(map_to_third)); 
    // (2, 4): 2-edge forests with 5 as a root and path from 3 to 5
    assert(forest_two(2, 4) == graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fourth", "fifth")); 
    // (3, 0): 2-edge forests with 1 as a root and path from 4 to 1
    const std::unordered_map<std::string, std::string> map_to_fourth = {
        {"first", "fourth"}, {"second", "fifth"}, {"third", "first"}, {"fourth", "second"}, {"fifth", "third"}
    }; 
    assert(forest_two(3, 0) == graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first"));
    // (3, 1): 2-edge forests with 2 as a root and path from 4 to 2
    assert(forest_two(3, 1) == graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("third", "second")); 
    // (3, 2): 2-edge forests with 3 as a root and path from 4 to 3
    assert(forest_two(3, 2) == two_edge_forests_with_path_from_1_to_5(map_to_fourth)); 
    // (3, 3): 2-edge forests with 4 as a root
    assert(forest_two(3, 3) == all_two_edge_forests(map_to_fourth)); 
    // (3, 4): 2-edge forests with 5 as a root and path from 4 to 5
    assert(forest_two(3, 4) == two_edge_forests_with_path_from_1_to_2(map_to_fourth));
    // (4, 0): 2-edge forests with 1 as a root and path from 5 to 1
    const std::unordered_map<std::string, std::string> map_to_fifth = {
        {"first", "fifth"}, {"second", "first"}, {"third", "second"}, {"fourth", "third"}, {"fifth", "fourth"}
    }; 
    assert(forest_two(4, 0) == two_edge_forests_with_path_from_1_to_2(map_to_fifth)); 
    // (4, 1): 2-edge forests with 2 as a root and path from 5 to 2
    assert(forest_two(4, 1) == graph->getEdgeLabel("fifth", "first") * graph->getEdgeLabel("first", "second")); 
    // (4, 2): 2-edge forests with 3 as a root and path from 5 to 3
    assert(forest_two(4, 2) == graph->getEdgeLabel("fifth", "fourth") * graph->getEdgeLabel("fourth", "third")); 
    // (4, 3): 2-edge forests with 4 as a root and path from 5 to 4
    assert(forest_two(4, 3) == two_edge_forests_with_path_from_1_to_5(map_to_fifth)); 
    // (4, 4): 2-edge forests with 5 as a root
    assert(forest_two(4, 4) == all_two_edge_forests(map_to_fifth));

    // Skip the third spanning forest matrix, and examine the fourth ...
    Matrix<long, Dynamic, Dynamic> forest_four = graph->getSpanningForestMatrixSparse(4);
    // (0, 0): trees with 1 as a root 
    assert(
        forest_four(0, 0) == (
            graph->getEdgeLabel("second", "first") * graph->getEdgeLabel("third", "second") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("second", "first") * graph->getEdgeLabel("third", "second") * 
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "first") + 
            graph->getEdgeLabel("second", "first") * graph->getEdgeLabel("third", "second") * 
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first") + 
            graph->getEdgeLabel("second", "first") * graph->getEdgeLabel("third", "fourth") * 
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first") + 
            graph->getEdgeLabel("second", "third") * graph->getEdgeLabel("third", "fourth") * 
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first")
        )
    );
    // (0, 1): trees with 2 as a root
    assert(
        forest_four(0, 1) == (
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("third", "fourth") *
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("third", "second") *
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("third", "second") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("third", "second") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("third", "second") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth")
        )
    );
    // (0, 2): trees with 3 as a root
    assert(
        forest_four(0, 2) == (
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("fourth", "fifth") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("fourth", "third") * graph->getEdgeLabel("fifth", "fourth")
        )
    );
    // (0, 3): trees with 4 as a root
    assert(
        forest_four(0, 3) == (
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fifth", "first") +
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fifth", "fourth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("third", "second") * graph->getEdgeLabel("fifth", "fourth") 
        )
    );
    // (0, 4): trees with 5 as a root
    assert(
        forest_four(0, 4) == (
            graph->getEdgeLabel("first", "second") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fourth", "fifth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "third") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fourth", "fifth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("third", "fourth") * graph->getEdgeLabel("fourth", "fifth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("third", "second") * graph->getEdgeLabel("fourth", "fifth") +
            graph->getEdgeLabel("first", "fifth") * graph->getEdgeLabel("second", "first") *
            graph->getEdgeLabel("third", "second") * graph->getEdgeLabel("fourth", "third")
        )
    );
    // Check that each column has the same entry in each row
    for (unsigned i = 0; i < 5; ++i)
    {
        for (unsigned j = 1; j < 5; ++j)
        {
            assert(forest_four(0, i) == forest_four(j, i));
        }
    } 

    delete graph; 
}

/**
 * Test `LabeledDigraph<...>::getSteadyStateFromSVD()`. 
 *
 * With a graph with (long) integer scalars, check that `getSteadyStateFromSVD()`
 * returns an *approximately* correct steady-state vector with highly precise 
 * multiple-precision scalars (100 digits in the mantissa).
 */
void TEST_MODULE_GET_STEADY_STATE_FROM_SVD()
{
    LabeledDigraph<long, long>* graph = new LabeledDigraph<long, long>(); 
    graph->addNode("first"); 
    graph->addNode("second"); 
    graph->addNode("third");
    graph->addNode("fourth"); 
    graph->addNode("fifth");
    graph->addEdge("first", "second", 1);
    graph->addEdge("first", "fifth", 13); 
    graph->addEdge("second", "first", 20); 
    graph->addEdge("second", "third", 67); 
    graph->addEdge("third", "second", 42); 
    graph->addEdge("third", "fourth", 1007); 
    graph->addEdge("fourth", "third", 17); 
    graph->addEdge("fourth", "fifth", 512); 
    graph->addEdge("fifth", "first", 179); 
    graph->addEdge("fifth", "fourth", 285);

    // Compute the steady-state vector
    Matrix<PreciseType, Dynamic, Dynamic> laplacian = graph->getLaplacian().template cast<PreciseType>();  
    Matrix<PreciseType, Dynamic, 1> ss = graph->template getSteadyStateFromSVD<PreciseType, PreciseType>();

    // Compute the error incurred by the steady-state computation, as the 
    // magnitude of the steady-state vector left-multiplied by the Laplacian 
    // matrix (this product should be *zero*)
    PreciseType err = (laplacian * ss).array().abs().matrix().sum();

    // Check that this error is tiny
    //
    // NOTE: The threshold chosen here is minuscule, but seeing as it was
    // deliberately chosen after some experimentation, this test should not
    // be taken as exact
    assert(err < 1e-90);

    delete graph;  
}

/**
 * Test `LabeledDigraph<...>::getSteadyStateFromRecurrence()`. 
 *
 * With a graph with (long) integer scalars, check that `getSteadyStateFromRecurrence()`
 * returns an *approximately* correct steady-state vector with highly precise 
 * multiple-precision scalars (100 digits in the mantissa).
 */
void TEST_MODULE_GET_STEADY_STATE_FROM_RECURRENCE()
{
    LabeledDigraph<long, long>* graph = new LabeledDigraph<long, long>(); 
    graph->addNode("first"); 
    graph->addNode("second"); 
    graph->addNode("third");
    graph->addNode("fourth"); 
    graph->addNode("fifth");
    graph->addEdge("first", "second", 1);
    graph->addEdge("first", "fifth", 13); 
    graph->addEdge("second", "first", 20); 
    graph->addEdge("second", "third", 67); 
    graph->addEdge("third", "second", 42); 
    graph->addEdge("third", "fourth", 1007); 
    graph->addEdge("fourth", "third", 17); 
    graph->addEdge("fourth", "fifth", 512); 
    graph->addEdge("fifth", "first", 179); 
    graph->addEdge("fifth", "fourth", 285);

    // Compute the steady-state vector
    Matrix<PreciseType, Dynamic, Dynamic> laplacian = graph->getLaplacian().template cast<PreciseType>();  
    Matrix<PreciseType, Dynamic, 1> ss = graph->template getSteadyStateFromRecurrence<PreciseType>(false);

    // Compute the error incurred by the steady-state computation, as the 
    // magnitude of the steady-state vector left-multiplied by the Laplacian 
    // matrix (this product should be *zero*)
    PreciseType err = (laplacian * ss).array().abs().matrix().sum();

    // Check that this error is tiny
    //
    // NOTE: The threshold chosen here is minuscule, but seeing as it was
    // deliberately chosen after some experimentation, this test should not
    // be taken as exact
    assert(err < 1e-90);

    delete graph;  
}

int main()
{
    TEST_MODULE_NODE_METHODS();
    std::cout << "TEST_MODULE_NODE_METHODS: all tests passed" << std::endl; 
    TEST_MODULE_EDGE_METHODS(); 
    std::cout << "TEST_MODULE_EDGE_METHODS: all tests passed" << std::endl;
    TEST_MODULE_CLEAR(); 
    std::cout << "TEST_MODULE_CLEAR: all tests passed" << std::endl;
    TEST_MODULE_GET_LAPLACIAN(); 
    std::cout << "TEST_MODULE_GET_LAPLACIAN: all tests passed" << std::endl;
    TEST_MODULE_GET_SPANNING_FOREST_MATRIX(); 
    std::cout << "TEST_MODULE_GET_SPANNING_FOREST_MATRIX: all tests passed" << std::endl;
    TEST_MODULE_GET_SPANNING_FOREST_MATRIX_SPARSE();
    std::cout << "TEST_MODULE_GET_SPANNING_FOREST_MATRIX_SPARSE: all tests passed" << std::endl;
    TEST_MODULE_GET_STEADY_STATE_FROM_SVD(); 
    std::cout << "TEST_MODULE_GET_STEADY_STATE_FROM_SVD: all tests passed" << std::endl;
    TEST_MODULE_GET_STEADY_STATE_FROM_RECURRENCE(); 
    std::cout << "TEST_MODULE_GET_STEADY_STATE_FROM_RECURRENCE: all tests passed" << std::endl;  

    return 0; 
}

