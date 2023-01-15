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
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     1/15/2023
 */

using namespace Eigen;
using boost::multiprecision::number; 
using boost::multiprecision::mpfr_float_backend;
typedef number<mpfr_float_backend<100> > PreciseType; 

/**
 * Run a simulation on the triangle graph. 
 */
void TEST_SIMULATE_TRIANGLE_GRAPH()
{
    // Instantiate a graph, add three nodes, and check that this->numnodes
    // increments correctly
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<double> dist;
    LabeledDigraph<double, double>* graph = new LabeledDigraph<double, double>(); 
    graph->addNode("1"); 
    graph->addNode("2");
    graph->addNode("3");
    graph->addEdge("1", "2", dist(rng));
    graph->addEdge("2", "1", dist(rng));
    graph->addEdge("2", "3", dist(rng));
    graph->addEdge("3", "1", dist(rng));
    graph->simulate("1", 20, 1234567890);

    delete graph; 
}

int main()
{
    TEST_SIMULATE_TRIANGLE_GRAPH();
    //std::cout << "TEST_MODULE_NODE_METHODS: all tests passed" << std::endl; 

    return 0; 
}

