#include <iostream>
#include <assert.h>
#include <string>
#include <sstream>
#include <array>
#include <iomanip>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include "../../include/digraph.hpp"
#include "../../include/graphs/grid.hpp"

/**
 * Test module for the `GridGraph` class.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     1/18/2022
 */

using namespace Eigen;
using boost::multiprecision::number; 
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::abs; 
typedef number<mpfr_float_backend<1000> > PreciseType; 
const PreciseType TOLERANCE("1e-900"); 

/**
 * An test class that inherits from the `GridGraph` class and can test 
 * protected methods. 
 */
template <typename InternalType, typename IOType>
class TestGridGraph : public GridGraph<InternalType, IOType>
{
    public:
        /**
         * Test the protected method `GridGraph<...>::getExitRelatedSpanningForestWeights()`.
         */
        void testGetExitRelatedSpanningForestWeights()
        {
            InternalType error; 

            auto forest_weights = this->getExitRelatedSpanningForestWeights();
 
            // The (2*i)-th column of v_alpha contains the weights of 
            // - spanning forests rooted at (A,i)
            // - spanning forests rooted at \{(A,i),(A,N)\} with path (B,N) -> (A,i)
            // - spanning forests rooted at \{(A,i),(B,N)\} with path (A,N) -> (A,i)
            //
            // The (2*i+1)-th column of v_alpha contains the weights of  
            // - spanning forests rooted at (B,i)
            // - spanning forests rooted at \{(B,i),(A,N)\} with path (B,N) -> (B,i)
            // - spanning forests rooted at \{(B,i),(B,N)\} with path (A,N) -> (B,i)
            Matrix<InternalType, 3, Dynamic> v_alpha = std::get<0>(forest_weights);    // size (3, 2 * this->N + 2) 
            
            // The i-th column of v_beta1 contains the weights of 
            // - spanning forests rooted at (A,i+1)
            // - spanning forests rooted at (B,i+1)
            // - spanning forests rooted at \{(A,i+1),(B,i+1)\}
            // on the (i+1)-th grid subgraph
            Matrix<InternalType, 3, Dynamic> v_beta1 = std::get<1>(forest_weights);    // size (3, this->N)

            // The (2*i)-th column of v_beta2 contains the weights of 
            // - spanning forests rooted at \{(A,i),(A,N)\}
            // - spanning forests rooted at \{(A,i),(B,N)\}
            // - spanning forests rooted at \{(A,i),(A,N),(B,N)\}
            //
            // The (2*i+1)-th column of v_beta2 contains the weights of 
            // - spanning forests rooted at \{(B,i),(A,N)\}
            // - spanning forests rooted at \{(B,i),(B,N)\}
            // - spanning forests rooted at \{(B,i),(A,N),(B,N)\}
            Matrix<InternalType, 3, Dynamic> v_beta2 = std::get<2>(forest_weights);    // size (3, 2 * this->N + 2)
            
            // The (2*i)-th column of v_beta3 contains the weights of 
            // - spanning forests rooted at \{(A,i),(A,N)\} with path (A,0) -> (A,i)
            // - spanning forests rooted at \{(A,i),(B,N)\} with path (A,0) -> (A,i)
            // - spanning forests rooted at \{(A,i),(A,N),(B,N)\} with path (A,0) -> (A,i)
            //
            // The (2*i+1)-th column of v_beta3 contains the weights of 
            // - spanning forests rooted at \{(B,i),(A,N)\} with path (A,0) -> (B,i)
            // - spanning forests rooted at \{(B,i),(B,N)\} with path (A,0) -> (B,i)
            // - spanning forests rooted at \{(B,i),(A,N),(B,N)\} with path (A,0) -> (B,i)
            Matrix<InternalType, 3, Dynamic> v_beta3 = std::get<3>(forest_weights);    // size (3, 2 * this->N + 2)
            
            // The i-th column of v_delta1 contains the weights of 
            // - spanning forests rooted at (A,i+1)
            // - spanning forests rooted at \{(A,i+1),(B,i+1)\} with path (A,0) -> (B,i+1)
            // on the (i+1)-th grid subgraph
            //
            // The i-th column of v_delta2 contains the weights of 
            // - spanning forests rooted at (B,i+1)
            // - spanning forests rooted at \{(A,i+1),(B,i+1)\} with path (A,0) -> (A,i+1)
            // on the (i+1)-th grid subgraph
            Matrix<InternalType, 2, Dynamic> v_delta1 = std::get<4>(forest_weights);   // size (2, this->N)
            Matrix<InternalType, 2, Dynamic> v_delta2 = std::get<5>(forest_weights);   // size (2, this->N)

            // The (2*i)-th column of v_theta1 contains the weights of 
            // - spanning forests rooted at \{(A,0),(A,N)\} with path (A,i) -> (A,N)
            // - spanning forests rooted at \{(A,0),(B,N)\} with path (A,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (A,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (A,i) -> (A,N)
            // 
            // The (2*i+1)-th column of v_theta1 contains the weights of 
            // - spanning forests rooted at \{(A,0),(A,N)\} with path (B,i) -> (A,N)
            // - spanning forests rooted at \{(A,0),(B,N)\} with path (B,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (B,i) -> (B,N)
            // - spanning forests rooted at \{(A,0),(A,N),(B,N)\} with path (B,i) -> (A,N)
            Matrix<InternalType, 4, Dynamic> v_theta1 = std::get<6>(forest_weights);   // size (4, 2 * this->N + 2)

            // The i-th column of v_theta2 contains the weights of 
            // - spanning forests rooted at \{(A,0),(B,i+1)\} with path (A,i+1) -> (B,i+1)
            // - spanning forests rooted at \{(A,0),(A,i+1)\} with path (B,i+1) -> (A,i+1)
            // on the (i+1)-th grid subgraph 
            Matrix<InternalType, 2, Dynamic> v_theta2 = std::get<7>(forest_weights);   // size (2, this->N)

            // Compare values for:
            // - the weight of all spanning forests rooted at (A,i+1) in the
            //   (i+1)-th grid subgraph
            // in v_beta1 and v_delta1
            for (unsigned i = 0; i < this->N; ++i)
            {
                error = abs(v_beta1(0, i) - v_delta1(0, i));
                assert(error < TOLERANCE);
            }

            // Compare values for:
            // - the weight of all spanning forests rooted at (B,i+1) in the
            //   (i+1)-th grid subgraph
            // in v_beta1 and v_delta2
            for (unsigned i = 0; i < this->N; ++i)
            {
                error = abs(v_beta1(1, i) - v_delta2(0, i)); 
                assert(error < TOLERANCE);
            }

            // Compare values for:
            // - the weight of all spanning forests rooted at \{(A,i+1),(B,i+1)\}
            //   in the (i+1)-th grid subgraph 
            // in v_beta1 and (v_delta1 + v_delta2)
            //
            // Here, v_delta1(1, i) + v_delta2(1, i) should sum to this weight
            for (unsigned i = 0; i < this->N; ++i)
            {
                error = abs(v_beta1(2, i) - v_delta1(1, i) - v_delta2(1, i)); 
                assert(error < TOLERANCE); 
            }

            // Compute the spanning tree weights using the Chebotarev-Agaev 
            // recurrence 
            Matrix<InternalType, Dynamic, Dynamic> tree_matrix = this->getSpanningForestMatrix(2 * this->N + 1);

            // Compare values for: 
            // - the weight of all spanning forests rooted at (A,i) in the 
            //   full grid graph 
            // in v_alpha and tree_matrix
            //
            // Note that v_alpha contains only the spanning forests rooted at 
            // (A,0), (B,0), ..., (A,N-1), (B,N-1)
            for (unsigned i = 0; i < this->N; ++i)
            {
                int index_Ai = 2 * i; 
                int index_Bi = 2 * i + 1;
                error = abs(v_alpha(0, index_Ai) - tree_matrix(0, index_Ai)); 
                assert(error < TOLERANCE); 
                error = abs(v_alpha(0, index_Bi) - tree_matrix(0, index_Bi)); 
                assert(error < TOLERANCE); 
            } 

            // Compare values for: 
            // - the weight of all spanning forests rooted at (A,N) in the 
            //   full grid graph 
            // - the weight of all spanning forests rooted at (B,N) in the 
            //   full grid graph 
            // in v_beta1 and tree_matrix
            error = abs(v_beta1(0, this->N - 1) - tree_matrix(0, 2 * this->N)); 
            assert(error < TOLERANCE); 
            error = abs(v_beta1(1, this->N - 1) - tree_matrix(0, 2 * this->N + 1));
            assert(error < TOLERANCE); 

            // Take the row Laplacian matrix of the grid graph ...
            Matrix<InternalType, Dynamic, Dynamic> laplacian = -this->getColumnLaplacianDense().transpose();

            // ... remove the edges outgoing from (A,N) ...
            for (unsigned i = 0; i < laplacian.cols(); ++i)
                laplacian(2 * this->N, i) = 0;

            // ... and obtain the corresponding two-rooted spanning forest matrix
            // with the Chebotarev-Agaev recurrence
            Matrix<InternalType, Dynamic, Dynamic> forest_matrix_AN = this->getSpanningForestMatrix(2 * this->N, laplacian);

            // Compare values for: 
            // - the weight of all spanning forests rooted at \{(A,N),(B,N)\} 
            //   in the full grid graph 
            // in v_beta1 and forest_matrix_AN
            error = abs(v_beta1(2, this->N - 1) - forest_matrix_AN(2 * this->N + 1, 2 * this->N + 1));
            assert(error < TOLERANCE); 

            // Compare values for: 
            // - the weight of all spanning forests rooted at \{(A,i),(A,N)\}
            //   for i = 0, ..., N-1 in the full grid graph
            // - the weight of all spanning forests rooted at \{(B,i),(A,N)\}
            //   for i = 0, ..., N-1 in the full grid graph
            // - the weight of all spanning forests rooted at \{(A,i),(A,N)\}
            //   with path (A,0) -> (A,i), for i = 0, ..., N-1 in the full
            //   grid graph
            // - the weight of all spanning forests rooted at \{(B,i),(A,N)\}
            //   with path (A,0) -> (B,i), for i = 0, ..., N-1 in the full
            //   grid graph
            for (unsigned i = 0; i < this->N; ++i)
            {
                int index_Ai = 2 * i; 
                int index_Bi = 2 * i + 1;
                error = abs(v_beta2(0, index_Ai) - forest_matrix_AN(index_Ai, index_Ai));
                assert(error < TOLERANCE);  
                error = abs(v_beta2(0, index_Bi) - forest_matrix_AN(index_Bi, index_Bi));
                assert(error < TOLERANCE); 
                error = abs(v_beta3(0, index_Ai) - forest_matrix_AN(0, index_Ai));
                assert(error < TOLERANCE);  
                error = abs(v_beta3(0, index_Bi) - forest_matrix_AN(0, index_Bi));
                assert(error < TOLERANCE);  
            }

            // Compare values for:
            // - the weight of all spanning forests rooted at \{(A,N),(B,N)\}
            //   with path (A,0) -> (B,N) in the full grid graph
            // in v_delta2 and forest_matrix_AN 
            error = abs(v_delta2(1, this->N - 1) - forest_matrix_AN(0, 2 * this->N + 1));
            assert(error < TOLERANCE); 

            // Now again take the row Laplacian matrix of the (full) grid graph ...
            laplacian = -this->getColumnLaplacianDense().transpose();

            // ... remove the edges outgoing from (B,N) ...
            for (unsigned i = 0; i < laplacian.cols(); ++i)
                laplacian(2 * this->N + 1, i) = 0;

            // ... and obtain the corresponding two-rooted spanning forest matrix
            // with the Chebotarev-Agaev recurrence
            Matrix<InternalType, Dynamic, Dynamic> forest_matrix_BN = this->getSpanningForestMatrix(2 * this->N, laplacian);

            // Compare values for: 
            // - the weight of all spanning forests rooted at \{(A,N),(B,N)\} 
            //   in the full grid graph 
            // in v_beta1 and forest_matrix_BN
            error = abs(v_beta1(2, this->N - 1) - forest_matrix_BN(2 * this->N, 2 * this->N));
            assert(error < TOLERANCE); 

            // Compare values for: 
            // - the weight of all spanning forests rooted at \{(A,i),(B,N)\}
            //   for i = 0, ..., N-1 in the full grid graph
            // - the weight of all spanning forests rooted at \{(B,i),(B,N)\}
            //   for i = 0, ..., N-1 in the full grid graph
            // - the weight of all spanning forests rooted at \{(A,i),(B,N)\}
            //   with path (A,0) -> (A,i), for i = 0, ..., N-1 in the full
            //   grid graph
            // - the weight of all spanning forests rooted at \{(B,i),(B,N)\}
            //   with path (A,0) -> (B,i), for i = 0, ..., N-1 in the full
            //   grid graph
            for (unsigned i = 0; i < this->N; ++i)
            {
                int index_Ai = 2 * i; 
                int index_Bi = 2 * i + 1;
                error = abs(v_beta2(1, index_Ai) - forest_matrix_BN(index_Ai, index_Ai));
                assert(error < TOLERANCE);  
                error = abs(v_beta2(1, index_Bi) - forest_matrix_BN(index_Bi, index_Bi));
                assert(error < TOLERANCE);  
                error = abs(v_beta3(1, index_Ai) - forest_matrix_BN(0, index_Ai));
                assert(error < TOLERANCE);  
                error = abs(v_beta3(1, index_Bi) - forest_matrix_BN(0, index_Bi)); 
                assert(error < TOLERANCE);  
            }

            // Compare values for:
            // - the weight of all spanning forests rooted at \{(A,N),(B,N)\}
            //   with path (A,0) -> (A,N) in the full grid graph
            // in v_delta1 and forest_matrix_BN 
            error = abs(v_delta1(1, this->N - 1) - forest_matrix_BN(0, 2 * this->N));
            assert(error < TOLERANCE);  

            // Now also remove the edges outgoing from (A,N) (again) ...
            for (unsigned i = 0; i < laplacian.cols(); ++i)
                laplacian(2 * this->N, i) = 0;

            // ... and obtain the corresponding three-rooted spanning forest matrix
            // with the Chebotarev-Agaev recurrence
            Matrix<InternalType, Dynamic, Dynamic> forest_matrix_AN_BN = this->getSpanningForestMatrix(2 * this->N - 1, laplacian);

            // Compare values for: 
            // - the weight of all spanning forests rooted at \{(A,i),(A,N),(B,N)\}
            //   for i = 0, ..., N-1 in the full grid graph
            // - the weight of all spanning forests rooted at \{(B,i),(A,N),(B,N)\}
            //   for i = 0, ..., N-1 in the full grid graph
            // - the weight of all spanning forests rooted at \{(A,i),(A,N),(B,N)\}
            //   with path (A,0) -> (A,i), for i = 0, ..., N-1 in the full
            //   grid graph
            // - the weight of all spanning forests rooted at \{(B,i),(A,N),(B,N)\}
            //   with path (A,0) -> (B,i), for i = 0, ..., N-1 in the full
            //   grid graph
            for (unsigned i = 0; i < this->N; ++i)
            {
                int index_Ai = 2 * i; 
                int index_Bi = 2 * i + 1;
                error = abs(v_beta2(2, index_Ai) - forest_matrix_AN_BN(index_Ai, index_Ai));
                assert(error < TOLERANCE);  
                error = abs(v_beta2(2, index_Bi) - forest_matrix_AN_BN(index_Bi, index_Bi));
                assert(error < TOLERANCE);  
                error = abs(v_beta3(2, index_Ai) - forest_matrix_AN_BN(0, index_Ai));
                assert(error < TOLERANCE);  
                error = abs(v_beta3(2, index_Bi) - forest_matrix_AN_BN(0, index_Bi)); 
                assert(error < TOLERANCE);  
            }

            // Now again take the row Laplacian matrix of the (full) grid graph ...
            laplacian = -this->getColumnLaplacianDense().transpose();

            // ... remove the edges outgoing from (A,0) ...
            for (unsigned i = 0; i < laplacian.cols(); ++i)
                laplacian(0, i) = 0;

            // ... and obtain the corresponding two-rooted spanning forest matrix
            // with the Chebotarev-Agaev recurrence
            Matrix<InternalType, Dynamic, Dynamic> forest_matrix_A0 = this->getSpanningForestMatrix(2 * this->N, laplacian);

            // Compare values for: 
            // - the weight of all spanning forests rooted at \{(A,0),(A,N)\}
            //   with path (A,i) -> (A,N)
            // - the weight of all spanning forests rooted at \{(A,0),(B,N)\}
            //   with path (A,i) -> (B,N)
            // - the weight of all spanning forests rooted at \{(A,0),(A,N)\}
            //   with path (B,i) -> (A,N)
            // - the weight of all spanning forests rooted at \{(A,0),(B,N)\}
            //   with path (B,i) -> (B,N)
            error = abs(v_theta1(0, 1) - forest_matrix_A0(1, 2 * this->N));
            assert(error < TOLERANCE);  
            for (unsigned i = 1; i < this->N; ++i)
            {
                int index_Ai = 2 * i; 
                int index_Bi = 2 * i + 1;
                error = abs(v_theta1(0, index_Ai) - forest_matrix_A0(index_Ai, 2 * this->N));
                assert(error < TOLERANCE);  
                error = abs(v_theta1(0, index_Bi) - forest_matrix_A0(index_Bi, 2 * this->N));
                assert(error < TOLERANCE);  
                error = abs(v_theta1(1, index_Ai) - forest_matrix_A0(index_Ai, 2 * this->N + 1));
                assert(error < TOLERANCE);  
                error = abs(v_theta1(1, index_Bi) - forest_matrix_A0(index_Bi, 2 * this->N + 1));
                assert(error < TOLERANCE);  
            }

            // Compare values for: 
            // - the weight of all spanning forests rooted at \{(A,0),(B,N)\} 
            //   with path (A,N) -> (B,N)
            error = abs(v_theta2(0, this->N - 1) - forest_matrix_A0(2 * this->N, 2 * this->N + 1));
            assert(error < TOLERANCE);  

            // Compare values for: 
            // - the weight of all spanning forests rooted at \{(A,0),(A,N)\}
            //   with path (B,N) -> (A,N)
            error = abs(v_theta2(1, this->N - 1) - forest_matrix_A0(2 * this->N + 1, 2 * this->N));
            assert(error < TOLERANCE);  

            // Now also remove the edges outgoing from (A,N) (again) ...
            for (unsigned i = 0; i < laplacian.cols(); ++i)
                laplacian(2 * this->N, i) = 0;

            // ... and obtain the corresponding three-rooted spanning forest matrix
            // with the Chebotarev-Agaev recurrence
            Matrix<InternalType, Dynamic, Dynamic> forest_matrix_A0_AN = this->getSpanningForestMatrix(2 * this->N - 1, laplacian);

            // Compare values for: 
            // - the weight of all spanning forests rooted at \{(A,0),(A,N),(B,N)\}
            //   with path (A,i) -> (B,N)
            // - the weight of all spanning forests rooted at \{(A,0),(A,N),(B,N)\}
            //   with path (B,i) -> (B,N)
            for (unsigned i = 0; i < this->N; ++i)
            {
                int index_Ai = 2 * i; 
                int index_Bi = 2 * i + 1;
                error = abs(v_theta1(2, index_Ai) - forest_matrix_A0_AN(index_Ai, 2 * this->N + 1));
                assert(error < TOLERANCE);  
                error = abs(v_theta1(2, index_Bi) - forest_matrix_A0_AN(index_Bi, 2 * this->N + 1));
                assert(error < TOLERANCE);  
            }

            // Now again take the row Laplacian matrix of the (full) grid graph ...
            laplacian = -this->getColumnLaplacianDense().transpose();

            // ... remove the edges outgoing from (A,0) ...
            for (unsigned i = 0; i < laplacian.cols(); ++i)
                laplacian(0, i) = 0;

            // ... and also the edges outgoing from (B,N) ...
            for (unsigned i = 0; i < laplacian.cols(); ++i)
                laplacian(2 * this->N + 1, i) = 0; 

            // ... and obtain the corresponding three-rooted spanning forest matrix
            // with the Chebotarev-Agaev recurrence
            Matrix<InternalType, Dynamic, Dynamic> forest_matrix_A0_BN = this->getSpanningForestMatrix(2 * this->N - 1, laplacian);

            // Compare values for: 
            // - the weight of all spanning forests rooted at \{(A,0),(A,N),(B,N)\}
            //   with path (A,i) -> (A,N)
            // - the weight of all spanning forests rooted at \{(A,0),(A,N),(B,N)\}
            //   with path (B,i) -> (A,N)
            for (unsigned i = 0; i < this->N; ++i)
            {
                int index_Ai = 2 * i; 
                int index_Bi = 2 * i + 1;
                error = abs(v_theta1(3, index_Ai) - forest_matrix_A0_BN(index_Ai, 2 * this->N));
                assert(error < TOLERANCE);  
                error = abs(v_theta1(3, index_Bi) - forest_matrix_A0_BN(index_Bi, 2 * this->N));
                assert(error < TOLERANCE);  
            }
        }
};

/**
 * Test `GridGraph<...>::getExitRelatedSpanningForestWeights()` via 
 * `TestGridGraph<...>::testGetExitRelatedSpanningForestWeights()`.
 *
 * @param length Length of the grid graph to be constructed. 
 */
void TEST_MODULE_GET_EXIT_RELATED_SPANNING_FOREST_WEIGHTS(int length)
{
    // Define a TestGridGraph instance with one set of six edge label values 
    TestGridGraph<PreciseType, PreciseType>* graph = new TestGridGraph<PreciseType, PreciseType>();
    PreciseType a = 43; 
    PreciseType b = 71;
    PreciseType c = 11; 
    PreciseType d = 37; 
    PreciseType e = 29; 
    PreciseType f = 89;  
    std::array<PreciseType, 6> labels = {a, b, c, d, e, f}; 
    graph->setZerothLabels(e, f);  
    for (unsigned i = 0; i < length; ++i) 
        graph->addRungToEnd(labels);

    // Run the test method 
    graph->testGetExitRelatedSpanningForestWeights(); 
    
    delete graph;
}

/**
 * Test `GridGraph<...>::getExitStats()` by comparing its output with values
 * computed via `LabeledDigraph<...>::getSpanningForestMatrix()`.
 *
 * @param length Length of the grid graph to be constructed.  
 */
void TEST_MODULE_GET_EXIT_STATS_VS_GET_SPANNING_FOREST_MATRIX(int length)
{
    PreciseType error; 

    // Define a GridGraph instance with one set of four edge label values 
    GridGraph<PreciseType, PreciseType>* graph = new GridGraph<PreciseType, PreciseType>();
    PreciseType a = 43; 
    PreciseType b = 71;
    PreciseType c = 11; 
    PreciseType d = 37; 
    PreciseType e = 29; 
    PreciseType f = 89;
    std::array<PreciseType, 6> labels = {a, b, c, d, e, f}; 
    graph->setZerothLabels(e, f);  
    for (unsigned i = 0; i < length; ++i) 
        graph->addRungToEnd(labels);

    // Define a LabeledDigraph instance and define an equivalent graph
    LabeledDigraph<PreciseType, PreciseType>* graph2 = new LabeledDigraph<PreciseType, PreciseType>();
    graph2->addNode("empty"); 
    for (unsigned i = 0; i < length + 1; ++i)   // A0, B0, ..., A20, B20
    {
        std::stringstream ss; 
        ss << "A" << i; 
        graph2->addNode(ss.str());
        ss.str(std::string()); 
        ss << "B" << i; 
        graph2->addNode(ss.str());  
    }
    graph2->addNode("bound");
    graph2->addEdge("A0", "empty", 1);
    graph2->addEdge("A0", "B0", e); 
    graph2->addEdge("B0", "A0", f);
    for (unsigned i = 0; i < length; ++i)
    {
        std::stringstream ssAi, ssAj, ssBi, ssBj; 
        ssAi << "A" << i;
        ssAj << "A" << i + 1;
        ssBi << "B" << i;  
        ssBj << "B" << i + 1; 
        graph2->addEdge(ssAi.str(), ssAj.str(), a);
        graph2->addEdge(ssAj.str(), ssAi.str(), b); 
        graph2->addEdge(ssBi.str(), ssBj.str(), c); 
        graph2->addEdge(ssBj.str(), ssBi.str(), d);
        graph2->addEdge(ssAj.str(), ssBj.str(), e); 
        graph2->addEdge(ssBj.str(), ssAj.str(), f); 
    }
    std::stringstream ss_final; 
    ss_final << "B" << length; 
    graph2->addEdge(ss_final.str(), "bound", 1);

    // Compute the exit statistics of the grid graph
    std::tuple<PreciseType, PreciseType, PreciseType> exit_stats = graph->getExitStats(1, 1); 

    // Compute the splitting probability for exit to the "bound" node from "A0" 
    // as the following ratio of double-rooted spanning forest weights:
    //
    // Weight of spanning forests rooted at "empty", "bound" with path "A0" -> "bound"
    // ------------------------------------------------------------------------------
    //             Weight of spanning forests rooted at "empty", "bound"
    //
    // Note that there are (2 * length + 4) nodes in the full grid graph
    Matrix<PreciseType, Dynamic, Dynamic> two_forest_matrix = graph2->getSpanningForestMatrix(2 * length + 2);
    error = abs(std::get<0>(exit_stats) - two_forest_matrix(1, 2 * length + 3) / two_forest_matrix(0, 0));
    assert(error < TOLERANCE); 

    // Compute the conditional mean first passage time to exit through the "bound"
    // node from "A0" as the following ratio of triple/double-rooted spanning
    // forest weights:
    //
    // \sum{A_k * B_k}
    // ---------------
    //      C * D
    //
    // where 
    //
    // A_k = Weight of spanning forests rooted at k, "empty," "bound" with path A0 -> k
    // B_k = Weight of spanning forests rooted at "empty," "bound" with path k -> "bound"
    // C = Weight of spanning forests rooted at "empty," "bound" with path A0 -> "bound"
    // D = Weight of spanning forests rooted at "empty," "bound"
    //
    // and the sum runs over k = (A,0), (B,0), ..., (A,N), (B,N)
    //
    // (The rate is the reciprocal of this mean first passage time)
    //
    // Note that there are (2 * length + 4) nodes in the full grid graph
    Matrix<PreciseType, Dynamic, Dynamic> three_forest_matrix = graph2->getSpanningForestMatrix(2 * length + 1); 
    PreciseType numer = 0; 
    for (unsigned i = 1; i <= 2 * length + 2; ++i)     // "empty" is node 0, "A0" is node 1, ...
        numer += (three_forest_matrix(1, i) * two_forest_matrix(i, 2 * length + 3));
    PreciseType denom = two_forest_matrix(1, 2 * length + 3) * two_forest_matrix(0, 0); 
    error = abs(std::get<2>(exit_stats) - denom / numer);
    assert(error < TOLERANCE);

    // Compute the mean first passage time to exit through the "empty" node from "A0"
    // as the following ratio of double/single-rooted spanning forest weights:
    //
    // \sum{(Weight of spanning forests rooted at k, "empty" with path A0 -> k)}
    // -------------------------------------------------------------------------
    //              Weight of spanning forests rooted at "empty"
    //
    // and the sum runs over k = (A,0), (B,0), ..., (A,N), (B,N)
    //
    // (The rate is the reciprocal of this mean first passage time)
    //
    // Note that there are (2 * length + 3) nodes after removing "bound" as a node
    graph2->removeEdge(ss_final.str(), "bound");
    graph2->removeNode("bound");  
    two_forest_matrix = graph2->getSpanningForestMatrix(2 * length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> one_forest_matrix = graph2->getSpanningForestMatrix(2 * length + 2); 
    numer = 0; 
    for (unsigned i = 1; i < 2 * length + 3; ++i)     // "empty" is node 0, "A0" is node 1, ...
        numer += two_forest_matrix(1, i);
    error = abs(std::get<1>(exit_stats) - one_forest_matrix(0, 0) / numer);
    assert(error < TOLERANCE); 

    delete graph;
    delete graph2;  
}

int main()
{
    int length = 20;

    TEST_MODULE_GET_EXIT_RELATED_SPANNING_FOREST_WEIGHTS(length); 
    std::cout << "TEST_MODULE_GET_EXIT_RELATED_SPANNING_FOREST_WEIGHTS(): all tests passed" << std::endl; 
    TEST_MODULE_GET_EXIT_STATS_VS_GET_SPANNING_FOREST_MATRIX(length); 
    std::cout << "TEST_MODULE_GET_EXIT_STATS_VS_GET_SPANNING_FOREST_MATRIX(): all tests passed" << std::endl; 

    return 0; 
}

