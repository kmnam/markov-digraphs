#define BOOST_TEST_MODULE testEnumSpanningTrees
#define BOOST_TEST_DYN_LINK
#include <iostream>
#include <string>
#include <random>
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include "../../include/digraph.hpp"
#include "../../include/enumSpanningTrees.hpp"

/*
 * Test module for the implementation of Uno's algorithm.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/9/2019
 */

using namespace Eigen;

// Use double scalars
typedef MarkovDigraph<double> Graph;

// Initialize random number generator and uniform distribution
std::mt19937 rng(1234567890);
std::uniform_real_distribution<> dist(0, 1);

Graph* triangle(std::mt19937& rng)
{
    /*
     * Return a triangle graph with edge labels uniformly distributed
     * between 0 and 1.
     */
    Graph* graph = new Graph();
    graph->addNode("1");
    graph->addNode("2");
    graph->addNode("3");
    graph->addEdge("1", "2", dist(rng));
    graph->addEdge("2", "1", dist(rng));
    graph->addEdge("2", "3", dist(rng));
    graph->addEdge("3", "2", dist(rng));
    graph->addEdge("3", "1", dist(rng));
    graph->addEdge("1", "3", dist(rng));
    return graph;
}

Graph* square(std::mt19937& rng)
{
    /*
     * Return a square graph with edge labels uniformly distributed
     * between 0 and 1.
     */
    Graph* graph = new Graph();
    graph->addNode("1");
    graph->addNode("2");
    graph->addNode("3");
    graph->addNode("4");
    graph->addEdge("1", "2", dist(rng));
    graph->addEdge("2", "1", dist(rng));
    graph->addEdge("2", "3", dist(rng));
    graph->addEdge("3", "2", dist(rng));
    graph->addEdge("3", "4", dist(rng));
    graph->addEdge("4", "3", dist(rng));
    graph->addEdge("4", "1", dist(rng));
    graph->addEdge("1", "4", dist(rng));
    return graph;
}

BOOST_AUTO_TEST_CASE(testTriangleGraphSpanningTrees)
{
    /*
     * Test that the square graph has the correct set of vertices and edges.
     */
    Graph* graph = triangle(rng);
    std::vector<std::vector<Edge<double> > > trees = enumSpanningTrees<double>(graph);
    for (auto&& tree : trees)
    {
        for (auto&& e : tree)
        {
            std::cout << "(" << e.first->id << " " << e.second->id << ") ";
        }
        std::cout << std::endl;
    }
    delete graph;
}

BOOST_AUTO_TEST_CASE(testSquareGraphSpanningTrees)
{
    /*
     * Test that the square graph has the correct set of vertices and edges.
     */
    Graph* graph = square(rng);
    std::vector<std::vector<Edge<double> > > trees = enumSpanningTrees<double>(graph);
    for (auto&& tree : trees)
    {
        for (auto&& e : tree)
        {
            std::cout << "(" << e.first->id << " " << e.second->id << ") ";
        }
        std::cout << std::endl;
    }
    delete graph;
}
