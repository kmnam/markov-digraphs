#define BOOST_TEST_MODULE testCopyNumberGraph
#define BOOST_TEST_DYN_LINK
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/test/included/unit_test.hpp>
#include "../../include/copyNumberGraph.hpp"

/*
 * Test module for the CopyNumberGraph class.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     2/3/2020
 */

using namespace Eigen;

// Use double scalars
typedef CopyNumberGraph<double> Graph;

// Initialize random number generator and uniform distribution
boost::random::mt19937 rng(1234567890);
boost::random::uniform_real_distribution<> dist(0, 1);

Graph* square(boost::random::mt19937& rng)
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

BOOST_AUTO_TEST_CASE(testSquareGraph)
{
    /*
     * Test that the square graph has the correct set of vertices and edges.
     */
    Graph* graph = square(rng);

    // Check that all four vertices exist
    BOOST_TEST(graph->getNode("1") != nullptr);
    BOOST_TEST(graph->getNode("2") != nullptr);
    BOOST_TEST(graph->getNode("3") != nullptr);
    BOOST_TEST(graph->getNode("4") != nullptr);

    // Try accessing vertices that don't exist
    BOOST_TEST(graph->getNode("5") == nullptr);

    // Check that all edge labels are greater than zero
    BOOST_TEST(graph->getEdgeLabel("1", "2") > 0);
    BOOST_TEST(graph->getEdgeLabel("2", "1") > 0);
    BOOST_TEST(graph->getEdgeLabel("2", "3") > 0);
    BOOST_TEST(graph->getEdgeLabel("3", "2") > 0);
    BOOST_TEST(graph->getEdgeLabel("3", "4") > 0);
    BOOST_TEST(graph->getEdgeLabel("4", "3") > 0);
    BOOST_TEST(graph->getEdgeLabel("4", "1") > 0);
    BOOST_TEST(graph->getEdgeLabel("1", "4") > 0);
    
    // Try accessing edges that don't exist
    Edge<double> edge = graph->getEdge("1", "3");
    BOOST_TEST(edge.first == nullptr);
    BOOST_TEST(edge.second == nullptr);
    BOOST_TEST(graph->getEdgeLabel("1", "3") == 0);

    // Compute the Laplacian of the graph
    MatrixXd laplacian = graph->getLaplacian();
    double e12 = graph->getEdgeLabel("1", "2");
    double e21 = graph->getEdgeLabel("2", "1");
    double e23 = graph->getEdgeLabel("2", "3");
    double e32 = graph->getEdgeLabel("3", "2");
    double e34 = graph->getEdgeLabel("3", "4");
    double e43 = graph->getEdgeLabel("4", "3");
    double e41 = graph->getEdgeLabel("4", "1");
    double e14 = graph->getEdgeLabel("1", "4");
    BOOST_TEST(laplacian(0,0) == -e12 - e14);
    BOOST_TEST(laplacian(1,1) == -e21 - e23);
    BOOST_TEST(laplacian(2,2) == -e32 - e34);
    BOOST_TEST(laplacian(3,3) == -e43 - e41);

    delete graph;
}

BOOST_AUTO_TEST_CASE(testSquareGraphRecurrence)
{
    /*
     * Test that the Chebotarev-Agaev recurrence gives the steady-state
     * vector for the Laplacian matrix of the square graph.
     */
    Graph* graph = square(rng);

    // Compute the Laplacian of the graph
    MatrixXd laplacian = graph->getLaplacian();

    // Compute the unnormalized steady-state vector
    VectorXd steady_state = graph->getSteadyStateFromRecurrence(false, 1e-10);
    BOOST_TEST(((laplacian * steady_state.matrix()).array() < 1e-10).all());

    // Compute the normalized steady-state vector
    steady_state = graph->getSteadyStateFromRecurrence(true, 1e-10);
    BOOST_TEST(steady_state.sum() == 1.0);
    BOOST_TEST(((laplacian * steady_state.matrix()).array() < 1e-10).all());

    delete graph;
}

