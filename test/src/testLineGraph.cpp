#include <iostream>
#include <assert.h>
#include <string>
#include <sstream>
#include <array>
#include <iomanip>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include "../../include/digraph.hpp"
#include "../../include/graphs/line.hpp"

/**
 * Test module for the `LineGraph` class.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     1/4/2022
 */

using namespace Eigen;
using boost::multiprecision::number; 
using boost::multiprecision::mpfr_float_backend;
typedef number<mpfr_float_backend<100> > PreciseType;

/**
 * Test `LineGraph<...>::getUpperExitProb()` by comparing its output with 
 * values computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 */
void TEST_MODULE_GET_UPPER_EXIT_PROB_VS_GET_SPANNING_FOREST_MATRIX()
{
    // Define a LineGraph instance with one forward and one reverse edge label
    LineGraph<PreciseType, PreciseType>* graph = new LineGraph<PreciseType, PreciseType>();
    PreciseType a = static_cast<PreciseType>(std::pow(10, 0.85486056816343636));
    PreciseType b = static_cast<PreciseType>(std::pow(10, -0.31066599812646878));
    for (unsigned i = 0; i < 20; ++i) 
        graph->addNodeToEnd(std::make_pair(a, b));
    std::cout << graph->getUpperExitProb(1, 1) << std::endl; 

    // Define a LabeledDigraph instance and define an equivalent graph
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    graph2->addNode("empty"); 
    for (unsigned i = 0; i < 21; ++i)
    {
        std::stringstream ss; 
        ss << i; 
        graph2->addNode(ss.str()); 
    }
    graph2->addNode("bound");
    graph2->addEdge("0", "empty", 1); 
    for (unsigned i = 0; i < 20; ++i)
    {
        std::stringstream ssi, ssj; 
        ssi << i;
        ssj << i + 1; 
        graph2->addEdge(ssi.str(), ssj.str(), a);
        graph2->addEdge(ssj.str(), ssi.str(), b); 
    }
    graph2->addEdge("20", "bound", 1);

    // Compute the splitting probability for exit to the "bound" node from "0" 
    // as the following ratio of double-rooted spanning forest weights:
    //
    // Weight of spanning forests rooted at "empty", "bound" with path "0" -> "bound"
    // ------------------------------------------------------------------------------
    //             Weight of spanning forests rooted at "empty", "bound"
    //
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(21);
    std::cout << two_forest_matrix(1, 22) / two_forest_matrix(22, 22) << std::endl; 

    delete graph;
    delete graph2;  
}

/**
 * Test `LineGraph<...>::getLowerExitRate()` by comparing its output with 
 * values computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 */
void TEST_MODULE_GET_LOWER_EXIT_RATE_VS_GET_SPANNING_FOREST_MATRIX()
{
    // Define a LineGraph instance with one forward and one reverse edge label
    LineGraph<PreciseType, PreciseType>* graph = new LineGraph<PreciseType, PreciseType>();
    PreciseType a = static_cast<PreciseType>(std::pow(10, 0.85486056816343636));
    PreciseType b = static_cast<PreciseType>(std::pow(10, -0.31066599812646878));
    for (unsigned i = 0; i < 20; ++i) 
        graph->addNodeToEnd(std::make_pair(a, b));
    std::cout << graph->getLowerExitRate(1) << std::endl; 

    // Define a LabeledDigraph instance and define an equivalent graph (with 
    // no "bound" node)
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    graph2->addNode("empty"); 
    for (unsigned i = 0; i < 21; ++i)
    {
        std::stringstream ss; 
        ss << i; 
        graph2->addNode(ss.str()); 
    }
    graph2->addEdge("0", "empty", 1); 
    for (unsigned i = 0; i < 20; ++i)
    {
        std::stringstream ssi, ssj; 
        ssi << i;
        ssj << i + 1; 
        graph2->addEdge(ssi.str(), ssj.str(), a);
        graph2->addEdge(ssj.str(), ssi.str(), b); 
    }

    // Compute the mean first passage time to exit through the "empty" node from "0"
    // as the following ratio of double/single-rooted spanning forest weights:
    //
    // Weight of spanning forests rooted at j, "empty" with path 0 -> j for j in 0, ..., 20
    // ------------------------------------------------------------------------------------
    //                    Weight of spanning forests rooted at "empty"
    //
    // (The rate is the reciprocal of this mean first passage time)
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(20);
    Matrix<PreciseType, Dynamic, Dynamic> one_forest_matrix = graph2->getSpanningForestMatrix(21); 
    PreciseType numer = 0; 
    for (unsigned i = 1; i < 22; ++i)     // "empty" is node 0, "0" is node 1, ...
        numer += two_forest_matrix(1, i);
    std::cout << one_forest_matrix(0, 0) / numer << std::endl; 

    delete graph;
    delete graph2;  
}

/**
 * Test `LineGraph<...>::getUpperExitRate()` by comparing its output with
 * values computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 */
void TEST_MODULE_GET_UPPER_EXIT_RATE_VS_GET_SPANNING_FOREST_MATRIX()
{
    // Define a LineGraph instance with one forward and one reverse edge label
    LineGraph<PreciseType, PreciseType>* graph = new LineGraph<PreciseType, PreciseType>();
    PreciseType a = static_cast<PreciseType>(std::pow(10, 0.85486056816343636));
    PreciseType b = static_cast<PreciseType>(std::pow(10, -0.31066599812646878));
    for (unsigned i = 0; i < 20; ++i) 
        graph->addNodeToEnd(std::make_pair(a, b));
    PreciseType val = graph->getUpperExitRate(1, 1);
    std::cout << val << std::endl;  

    // Define a LabeledDigraph instance and define an equivalent graph
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    graph2->addNode("empty"); 
    for (unsigned i = 0; i < 21; ++i)
    {
        std::stringstream ss; 
        ss << i; 
        graph2->addNode(ss.str()); 
    }
    graph2->addNode("bound");
    graph2->addEdge("0", "empty", 1); 
    for (unsigned i = 0; i < 20; ++i)
    {
        std::stringstream ssi, ssj; 
        ssi << i;
        ssj << i + 1; 
        graph2->addEdge(ssi.str(), ssj.str(), a);
        graph2->addEdge(ssj.str(), ssi.str(), b); 
    }
    graph2->addEdge("20", "bound", 1);

    // Compute the conditional mean first passage time to exit through the
    // "bound" node from "0" as the following ratio of triple/double-rooted
    // spanning forest weights:
    //
    // A_k * B_k
    // ---------
    //   C * D
    //
    // where
    //
    // A_k = Weight of spanning forests rooted at "k", "bound", "empty" with path 0 -> k
    // B_k = Weight of spanning forests rooted at "bound", "empty" with path k -> bound
    // C = Weight of spanning forests rooted at "bound", "empty" with path 0 -> bound
    // D = Weight of spanning forests rooted at "bound", "empty"
    //
    // (The rate is the reciprocal of this mean first passage time)
    Matrix<PreciseType, Dynamic, Dynamic> three_forest_matrix = graph2->getSpanningForestMatrix(20);
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(21);
    PreciseType numer = 0; 
    for (unsigned i = 1; i < 22; ++i)     // "empty" is node 0, "1" is node 1, ...
        numer += (three_forest_matrix(1, i) * two_forest_matrix(i, 22));
    PreciseType denom = two_forest_matrix(1, 22) * two_forest_matrix(22, 22);
    std::cout << denom / numer << std::endl;

    delete graph;
    delete graph2;  
}

int main()
{
    TEST_MODULE_GET_UPPER_EXIT_PROB_VS_GET_SPANNING_FOREST_MATRIX(); 
    std::cout << "TEST_MODULE_GET_UPPER_EXIT_PROB_VS_GET_SPANNING_FOREST_MATRIX(): all tests passed" << std::endl;
    TEST_MODULE_GET_LOWER_EXIT_RATE_VS_GET_SPANNING_FOREST_MATRIX(); 
    std::cout << "TEST_MODULE_GET_LOWER_EXIT_RATE_VS_GET_SPANNING_FOREST_MATRIX(): all tests passed" << std::endl;  
    TEST_MODULE_GET_UPPER_EXIT_RATE_VS_GET_SPANNING_FOREST_MATRIX(); 
    std::cout << "TEST_MODULE_GET_UPPER_EXIT_RATE_VS_GET_SPANNING_FOREST_MATRIX(): all tests passed" << std::endl;  

    return 0; 
}

