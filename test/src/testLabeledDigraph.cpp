#include <iostream>
#include <assert.h>
#include <string>
#include <sstream>
#include <cmath>
#include <array>
#include <Eigen/Dense>
#include "../../include/digraph.hpp"

/*
 * Test module for the LabeledDigraph class.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/11/2021
 */

using namespace Eigen;

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
    assert(!graph->hasEdge("fourth", "fourth")); 
    assert(!graph->hasEdge("fifth", "fifth")); 

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
        graph->setEdgeLabel("every one of them words rang true", "and burned like burning coal", 0.0001); 
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
    for (const std::string s : node_ids)
    {
        for (const std::string t : node_ids)
        {
            assert(!graph->hasEdge(s, t)); 
        }
    }

    delete graph;  
}

/**
 * Test `LabeledDigraph<...>::getLaplacian()`.
 */
void TEST_MODULE_GET_LAPLACIAN()
{
    // With an integer-scalar graph, check that getLaplacian() returns the 
    // exactly correct Laplacian matrix
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

    return 0; 
}

