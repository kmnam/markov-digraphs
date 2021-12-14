#include <iostream>
#include <assert.h>
#include <string>
#include <sstream>
#include <array>
#include <iomanip>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <graphviz/gvc.h>
#include "../../include/digraph.hpp"
#include "../../include/viz.hpp"

/**
 * Test module for the functions in `viz.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/13/2021
 */

using namespace Eigen;
using boost::multiprecision::number; 
using boost::multiprecision::mpfr_float_backend;
typedef number<mpfr_float_backend<100> > PreciseType; 

/**
 * Use `vizLabeledDigraph()` to draw a 5-vertex cycle graph.  
 */
void TEST_MODULE_VIZ_LABELED_DIGRAPH()
{
    LabeledDigraph<long, long> graph; 
    graph.addNode("first"); 
    graph.addNode("second"); 
    graph.addNode("third");
    graph.addNode("fourth"); 
    graph.addNode("fifth");
    graph.addEdge("first", "second", 1);
    graph.addEdge("first", "fifth", 13); 
    graph.addEdge("second", "first", 20); 
    graph.addEdge("second", "third", 67); 
    graph.addEdge("third", "second", 42); 
    graph.addEdge("third", "fourth", 1007); 
    graph.addEdge("fourth", "third", 17); 
    graph.addEdge("fourth", "fifth", 512); 
    graph.addEdge("fifth", "first", 179); 
    graph.addEdge("fifth", "fourth", 285);

    // Draw the graph with the dot, neato, and circo layout algorithms 
    GVC_t* context = gvContext();
    vizLabeledDigraph(graph, "dot", "png", "test-dot.png", context); 
    vizLabeledDigraph(graph, "neato", "png", "test-neato.png", context);
    vizLabeledDigraph(graph, "circo", "png", "test-circo.png", context);  
    gvFreeContext(context); 
}

int main()
{
    TEST_MODULE_VIZ_LABELED_DIGRAPH(); 
    std::cout << "TEST_MODULE_VIZ_LABELED_DIGRAPH: all tests passed" << std::endl; 

    return 0; 
}

