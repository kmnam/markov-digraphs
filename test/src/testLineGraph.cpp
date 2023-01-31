#include <iostream>
#include <assert.h>
#include <string>
#include <sstream>
#include <array>
#include <iomanip>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../../include/digraph.hpp"
#include "../../include/graphs/line.hpp"

/**
 * Test module for the `LineGraph` class.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     1/31/2023
 */

using namespace Eigen;
using boost::multiprecision::number; 
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::abs; 
typedef number<mpfr_float_backend<1000> > PreciseType;
const PreciseType TOLERANCE("1e-900");

/**
 * Test `LineGraph<...>::getUpperEndToEndTime()` by comparing its output 
 * with values computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 */
TEST_CASE("Test that getUpperEndToEndTime() and getSpanningForestMatrix() match")
{
    int length = 20; 

    // Define a LineGraph instance with one forward and one reverse edge label 
    LineGraph<PreciseType, PreciseType>* graph = new LineGraph<PreciseType, PreciseType>(); 
    PreciseType a = 2;
    PreciseType b = 3;
    for (int i = 0; i < length; ++i)
        graph->addNodeToEnd(std::make_pair(a, b));

    // Define an equivalent LabeledDigraph instance with no outgoing edges
    // from N
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    for (int i = 0; i <= length; ++i)
    {
        std::stringstream ss; 
        ss << i; 
        graph2->addNode(ss.str()); 
    }
    for (int i = 0; i < length - 1; ++i)
    {
        std::stringstream ssi, ssj; 
        ssi << i;
        ssj << i + 1; 
        graph2->addEdge(ssi.str(), ssj.str(), a);
        graph2->addEdge(ssj.str(), ssi.str(), b); 
    }
    std::stringstream ssi, ssj;
    ssi << length - 1;
    ssj << length;  
    graph2->addEdge(ssi.str(), ssj.str(), a);

    // Compute the mean first-passage time to N from 0 as the following 
    // ratio of spanning forest weights:
    //
    // Weight of spanning forests rooted at i, N with path 0 -> i for i = 0, ..., N - 1
    // --------------------------------------------------------------------------------
    //                     Weight of spanning forests rooted at N
    //
    // Note that the line graph without exit nodes has (length + 1) nodes
    Matrix<PreciseType, Dynamic, Dynamic> one_forest_matrix = graph2->getSpanningForestMatrix(length); 
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(length - 1);
    PreciseType numer = two_forest_matrix.row(0).head(length).sum();    // Node "0" is 0, "1" is 1, ...
    PreciseType denom = one_forest_matrix(length, length); 
    PreciseType error = abs(graph->getUpperEndToEndTime() - numer / denom);
    REQUIRE(error < TOLERANCE);

    delete graph;
    delete graph2;
}

/**
 * Test `LineGraph<...>::getLowerEndToEndTime()` by comparing its output 
 * with values computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 */
TEST_CASE("Test that getLowerEndToEndTime() and getSpanningForestMatrix() match")
{
    int length = 20; 

    // Define a LineGraph instance with one forward and one reverse edge label 
    LineGraph<PreciseType, PreciseType>* graph = new LineGraph<PreciseType, PreciseType>(); 
    PreciseType a = 2;
    PreciseType b = 3;
    for (int i = 0; i < length; ++i)
        graph->addNodeToEnd(std::make_pair(a, b));

    // Define an equivalent LabeledDigraph instance with no outgoing edges
    // from 0
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    for (int i = 0; i <= length; ++i)
    {
        std::stringstream ss; 
        ss << i; 
        graph2->addNode(ss.str()); 
    }
    graph2->addEdge("1", "0", b);
    for (int i = 1; i < length; ++i)
    {
        std::stringstream ssi, ssj; 
        ssi << i;
        ssj << i + 1; 
        graph2->addEdge(ssi.str(), ssj.str(), a);
        graph2->addEdge(ssj.str(), ssi.str(), b); 
    }

    // Compute the mean first-passage time to 0 from N as the following 
    // ratio of spanning forest weights:
    //
    // Weight of spanning forests rooted at 0, i with path N -> i for i = 1, ..., N
    // ----------------------------------------------------------------------------
    //                    Weight of spanning forests rooted at 0
    //
    // Note that the line graph without exit nodes has (length + 1) nodes
    Matrix<PreciseType, Dynamic, Dynamic> one_forest_matrix = graph2->getSpanningForestMatrix(length); 
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(length - 1);
    PreciseType numer = two_forest_matrix.row(length).tail(length).sum();    // Node "0" is 0, "1" is 1, ...
    PreciseType denom = one_forest_matrix(0, 0);
    PreciseType error = abs(graph->getLowerEndToEndTime() - numer / denom);
    REQUIRE(error < TOLERANCE);

    delete graph;
    delete graph2;
}

/**
 * Test `LineGraph<...>::getUpperExitProb()` by comparing its output with 
 * values computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 */
TEST_CASE("Test that getUpperExitProb() and getSpanningForestMatrix() match")
{
    int length = 20;

    // Define a LineGraph instance with one forward and one reverse edge label
    LineGraph<PreciseType, PreciseType>* graph = new LineGraph<PreciseType, PreciseType>();
    PreciseType a = 2; 
    PreciseType b = 3; 
    for (int i = 0; i < length; ++i) 
        graph->addNodeToEnd(std::make_pair(a, b));

    // Define an equivalent LabeledDigraph instance with exit nodes
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    graph2->addNode("empty"); 
    for (int i = 0; i <= length; ++i)
    {
        std::stringstream ss; 
        ss << i; 
        graph2->addNode(ss.str()); 
    }
    graph2->addNode("bound");
    graph2->addEdge("0", "empty", 1); 
    for (int i = 0; i < length; ++i)
    {
        std::stringstream ssi, ssj; 
        ssi << i;
        ssj << i + 1; 
        graph2->addEdge(ssi.str(), ssj.str(), a);
        graph2->addEdge(ssj.str(), ssi.str(), b); 
    }
    std::stringstream ss_final; 
    ss_final << length; 
    graph2->addEdge(ss_final.str(), "bound", 1);

    // Compute the splitting probability for exit to the "bound" node from "0" 
    // as the following ratio of double-rooted spanning forest weights:
    //
    // Weight of spanning forests rooted at "empty", "bound" with path "0" -> "bound"
    // ------------------------------------------------------------------------------
    //             Weight of spanning forests rooted at "empty", "bound"
    //
    // Note that the full line graph has (length + 3) nodes
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(length + 1);
    PreciseType numer = two_forest_matrix(1, length + 2);            // Path from 0 to bound
    PreciseType denom = two_forest_matrix(length + 2, length + 2);   // All double-rooted forests
    PreciseType error = abs(graph->getUpperExitProb(1, 1) - numer / denom);
    REQUIRE(error < TOLERANCE); 

    delete graph;
    delete graph2;  
}

/**
 * Test `LineGraph<...>::getLowerExitRate(PreciseType)` by comparing its output
 * with values computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 */
TEST_CASE("Test that getLowerExitRate(PreciseType) and getSpanningForestMatrix() match")
{
    int length = 20;

    // Define a LineGraph instance with one forward and one reverse edge label
    LineGraph<PreciseType, PreciseType>* graph = new LineGraph<PreciseType, PreciseType>();
    PreciseType a = 2; 
    PreciseType b = 3; 
    for (int i = 0; i < length; ++i) 
        graph->addNodeToEnd(std::make_pair(a, b));

    // Define an equivalent LabeledDigraph instance with lower exit node
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    graph2->addNode("empty"); 
    for (int i = 0; i <= length; ++i)
    {
        std::stringstream ss; 
        ss << i; 
        graph2->addNode(ss.str()); 
    }
    graph2->addEdge("0", "empty", 1); 
    for (int i = 0; i < length; ++i)
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
    // Weight of spanning forests rooted at j, "empty" with path 0 -> j for j in 0, ..., N
    // -----------------------------------------------------------------------------------
    //                   Weight of spanning forests rooted at "empty"
    //
    // (The rate is the reciprocal of this mean first passage time)
    //
    // Note that the full line graph has (length + 2) nodes, since it has 
    // no "bound" node 
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(length);
    Matrix<PreciseType, Dynamic, Dynamic> one_forest_matrix = graph2->getSpanningForestMatrix(length + 1); 
    PreciseType numer = two_forest_matrix.row(1)(Eigen::seqN(1, length + 1)).sum();
    PreciseType denom = one_forest_matrix(0, 0); 
    PreciseType error = abs(graph->getLowerExitRate(1) - denom / numer);
    REQUIRE(error < TOLERANCE); 

    delete graph;
    delete graph2;  
}

/**
 * Test `LineGraph<...>::getEntryToUpperExitTime()` by comparing its output 
 * with values computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 */
TEST_CASE("Test that getEntryToUpperExitTime() and getSpanningForestMatrix() match")
{
    int length = 20; 

    // Define a LineGraph instance with one forward and one reverse edge label 
    LineGraph<PreciseType, PreciseType>* graph = new LineGraph<PreciseType, PreciseType>(); 
    PreciseType a = 2;
    PreciseType b = 3;
    for (int i = 0; i < length; ++i)
        graph->addNodeToEnd(std::make_pair(a, b));

    // Define an equivalent LabeledDigraph instance with exit nodes
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    graph2->addNode("empty");
    for (int i = 0; i <= length; ++i)
    {
        std::stringstream ss; 
        ss << i; 
        graph2->addNode(ss.str()); 
    }
    graph2->addNode("bound");  
    graph2->addEdge("0", "empty", 1);
    graph2->addEdge("empty", "0", 1);
    for (int i = 0; i < length; ++i)
    {
        std::stringstream ssi, ssj; 
        ssi << i;
        ssj << i + 1; 
        graph2->addEdge(ssi.str(), ssj.str(), a);
        graph2->addEdge(ssj.str(), ssi.str(), b); 
    }
    std::stringstream ss_final; 
    ss_final << length; 
    graph2->addEdge(ss_final.str(), "bound", 1);

    // Compute the mean first-passage time to "empty" from "bound" as the following 
    // ratio of spanning forest weights:
    //
    // Weight of spanning forests rooted at i, "bound" with path "empty" -> i for i = "empty", 0, ..., N
    // -------------------------------------------------------------------------------------------------
    //                          Weight of spanning forests rooted at "bound"
    //
    // Note that the full line graph has (length + 3) nodes 
    Matrix<PreciseType, Dynamic, Dynamic> one_forest_matrix = graph2->getSpanningForestMatrix(length + 2); 
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(length + 1);
    PreciseType numer = two_forest_matrix.row(0).head(length + 2).sum();    // Node "0" is 1, "1" is 2, ...
    PreciseType denom = one_forest_matrix(length + 2, length + 2);
    PreciseType error = abs(graph->getEntryToUpperExitTime(1, 1, 1) - numer / denom);
    REQUIRE(error < TOLERANCE);

    delete graph;
    delete graph2;
}

/**
 * Test `LineGraph<...>::getLowerExitRate(PreciseType, PreciseType)` by comparing
 * its output with values computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 */
TEST_CASE("Test that getLowerExitRate(PreciseType, PreciseType) and getSpanningForestMatrix() match")
{
    int length = 20;

    // Define a LineGraph instance with one forward and one reverse edge label
    LineGraph<PreciseType, PreciseType>* graph = new LineGraph<PreciseType, PreciseType>();
    PreciseType a = 2; 
    PreciseType b = 3; 
    for (int i = 0; i < length; ++i) 
        graph->addNodeToEnd(std::make_pair(a, b));

    // Define an equivalent LabeledDigraph instance with exit nodes 
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    graph2->addNode("empty");
    for (int i = 0; i <= length; ++i)
    {
        std::stringstream ss; 
        ss << i; 
        graph2->addNode(ss.str()); 
    }
    graph2->addNode("bound");  
    graph2->addEdge("0", "empty", 1);
    for (int i = 0; i < length; ++i)
    {
        std::stringstream ssi, ssj; 
        ssi << i;
        ssj << i + 1; 
        graph2->addEdge(ssi.str(), ssj.str(), a);
        graph2->addEdge(ssj.str(), ssi.str(), b); 
    }
    std::stringstream ss_final; 
    ss_final << length; 
    graph2->addEdge(ss_final.str(), "bound", 1);

    // Compute the conditional mean first passage time to exit through the
    // "empty" node from "0" as the following ratio of triple/double-rooted
    // spanning forest weights:
    //
    // A_k * B_k
    // ---------
    //   C * D
    //
    // where
    //
    // A_k = Weight of spanning forests rooted at "k", "bound", "empty" with path 0 -> k
    // B_k = Weight of spanning forests rooted at "bound", "empty" with path k -> empty
    // C = Weight of spanning forests rooted at "bound", "empty" with path 0 -> empty
    // D = Weight of spanning forests rooted at "bound", "empty"
    //
    // (The rate is the reciprocal of this mean first passage time)
    //
    // Note that the full line graph has (length + 3) nodes 
    Matrix<PreciseType, Dynamic, Dynamic> three_forest_matrix = graph2->getSpanningForestMatrix(length);
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(length + 1);
    PreciseType numer = 0; 
    for (int i = 1; i < length + 2; ++i)     // "empty" is node 0, "0" is node 1, ...
        numer += (three_forest_matrix(1, i) * two_forest_matrix(i, 0));
    PreciseType denom = two_forest_matrix(1, 0) * two_forest_matrix(length + 2, length + 2); 
    PreciseType error = abs(graph->getLowerExitRate(1, 1) - denom / numer); 
    REQUIRE(error < TOLERANCE); 

    delete graph;
    delete graph2;  
}

/**
 * Test `LineGraph<...>::getUpperExitRate()` by comparing its output with
 * values computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 */
TEST_CASE("Test that getUpperExitRate() and getSpanningForestMatrix() match")
{
    int length = 20;

    // Define a LineGraph instance with one forward and one reverse edge label
    LineGraph<PreciseType, PreciseType>* graph = new LineGraph<PreciseType, PreciseType>();
    PreciseType a = 2;
    PreciseType b = 3;
    for (int i = 0; i < length; ++i) 
        graph->addNodeToEnd(std::make_pair(a, b));

    // Define an equivalent LabeledDigraph instance with exit nodes 
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    graph2->addNode("empty"); 
    for (int i = 0; i <= length; ++i)
    {
        std::stringstream ss; 
        ss << i; 
        graph2->addNode(ss.str()); 
    }
    graph2->addNode("bound");
    graph2->addEdge("0", "empty", 1); 
    for (int i = 0; i < length; ++i)
    {
        std::stringstream ssi, ssj; 
        ssi << i;
        ssj << i + 1; 
        graph2->addEdge(ssi.str(), ssj.str(), a);
        graph2->addEdge(ssj.str(), ssi.str(), b); 
    }
    std::stringstream ss_final; 
    ss_final << length; 
    graph2->addEdge(ss_final.str(), "bound", 1);

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
    //
    // Note that the full line graph has (length + 3) nodes 
    Matrix<PreciseType, Dynamic, Dynamic> three_forest_matrix = graph2->getSpanningForestMatrix(length);
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(length + 1);
    PreciseType numer = 0; 
    for (int i = 1; i < length + 2; ++i)     // "empty" is node 0, "1" is node 1, ...
        numer += (three_forest_matrix(1, i) * two_forest_matrix(i, length + 2));
    PreciseType denom = two_forest_matrix(1, length + 2) * two_forest_matrix(length + 2, length + 2);
    PreciseType error = abs(graph->getUpperExitRate(1, 1) - denom / numer);
    REQUIRE(error < TOLERANCE);  

    delete graph;
    delete graph2;  
}


