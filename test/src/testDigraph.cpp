#define BOOST_TEST_MODULE testDigraph
#define BOOST_TEST_DYN_LINK
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <array>
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include "../../include/digraph.hpp"

/*
 * Test module for the LabeledDigraph class.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/29/2020
 */

using namespace Eigen;

template <typename T, unsigned N>
class Polygon : public LabeledDigraph<T>
{
    /*
     * A polygonal graph with reversible edges. 
     */
    public:
        Polygon()
        {
            /*
             * Instantiate a polygonal graph with edge labels of unity. 
             */
            // Start with the zeroth node
            this->addNode("0");

            for (unsigned i = 1; i < N; ++i)
            {
                std::stringstream ss;
                ss << i;

                // Add the i-th node
                this->addNode(ss.str());

                // Add edges between i and i - 1
                std::stringstream ss_prev;
                ss_prev << i - 1;
                this->addEdge(ss_prev.str(), ss.str());
                this->addEdge(ss.str(), ss_prev.str());
            }

            // Loop back to the zeroth node
            std::stringstream ss; 
            ss << N - 1;
            this->addEdge(ss.str(), "0");
            this->addEdge("0", ss.str());
        }

        Polygon(std::array<std::pair<T, T>, N> labels)
        {
            /*
             * Instantiate a polygonal graph with the given edge label values.
             *
             * The correct number of parameter (edge label) values is assumed 
             * to have been given.  
             */
            // Start with the zeroth node
            this->addNode("0");

            for (unsigned i = 1; i < N; ++i)
            {
                std::stringstream ss;
                ss << i;

                // Add the i-th node
                this->addNode(ss.str());

                // Add edges between i and i - 1
                std::stringstream ss_prev;
                ss_prev << i - 1;
                T forward = labels[i - 1].first;
                T reverse = labels[i - 1].second;
                this->addEdge(ss_prev.str(), ss.str(), forward);
                this->addEdge(ss.str(), ss_prev.str(), reverse);
            }

            // Loop back to the zeroth node
            std::stringstream ss; 
            ss << N - 1;
            this->addEdge(ss.str(), "0", labels[N - 1].first);
            this->addEdge("0", ss.str(), labels[N - 1].second);
        }

        template <typename U = T>
        Matrix<U, Dynamic, Dynamic> laplacian(std::vector<Node*> nodes)
        {
            /*
             * Public version of the this->getLaplacian() method. 
             */
            return this->template getLaplacian<U>(nodes);
        }
};

bool containsNode(std::vector<Node*> nodes, std::string id)
{
    /*
     * Check if the given vector of nodes contains a node with the given id.  
     */
    for (auto&& node : nodes)
    {
        if (node->id == id)
            return true;
    }
    return false;
}

template <typename T>
bool containsEdge(std::vector<std::pair<Edge, T> > edges, std::string source_id,
                  std::string target_id, T label)
{
    /*
     * Check if the given vector of edges contains an edge with the given
     * source and target ids. 
     */
    for (auto&& edge: edges)
    {
        if ((edge.first.first)->id == source_id && (edge.first.second)->id == target_id && edge.second == label)
            return true;
    }
    return false;
}

BOOST_AUTO_TEST_CASE(testTriangle)
{
    /*
     * Test all public methods on the triangle graph. 
     */
    // Set all edge labels equal to one
    Polygon<int, 3> triangle;

    // Check that the graph contains the three nodes 0, 1, 2
    std::vector<Node*> nodes = triangle.getNodes();
    BOOST_TEST(nodes.size() == 3);
    BOOST_TEST(containsNode(nodes, "0"));
    BOOST_TEST(containsNode(nodes, "1"));
    BOOST_TEST(containsNode(nodes, "2"));

    // Check that the graph contains edges (0,1), (1,0), (1,2), (2,1), (2,0), (0,2)
    // with edge labels all set to 1
    std::vector<std::pair<Edge, int> > edges = triangle.getEdges();
    BOOST_TEST(edges.size() == 6);
    BOOST_TEST(containsEdge<int>(edges, "0", "1", 1));
    BOOST_TEST(containsEdge<int>(edges, "1", "0", 1));
    BOOST_TEST(containsEdge<int>(edges, "1", "2", 1));
    BOOST_TEST(containsEdge<int>(edges, "2", "1", 1));
    BOOST_TEST(containsEdge<int>(edges, "2", "0", 1));
    BOOST_TEST(containsEdge<int>(edges, "0", "2", 1));

    // Try getting the nodes individually 
    BOOST_TEST(triangle.getNode("0") != nullptr);
    BOOST_TEST(triangle.getNode("1") != nullptr);
    BOOST_TEST(triangle.getNode("2") != nullptr);

    // Try getting a non-existent node
    BOOST_TEST(triangle.getNode("3") == nullptr);

    // Try getting the edges individually
    for (unsigned i = 0; i < 3; ++i)
    {
        std::stringstream ss0, ss1;
        ss0 << i;
        if (i == 2)
            ss1 << 0;
        else
            ss1 << i + 1;

        // Try getting the forward edge
        std::pair<Edge, int> forward = triangle.getEdge(ss0.str(), ss1.str());
        BOOST_TEST((forward.first.first)->id == ss0.str());
        BOOST_TEST((forward.first.second)->id == ss1.str());
        BOOST_TEST(forward.second == 1);

        // Try getting the reverse edge
        std::pair<Edge, int> reverse = triangle.getEdge(ss1.str(), ss0.str());
        BOOST_TEST((reverse.first.first)->id == ss1.str());
        BOOST_TEST((reverse.first.second)->id == ss0.str());
        BOOST_TEST(reverse.second == 1);
    }

    // Try setting certain edge labels to different values 
    triangle.setEdgeLabel("0", "1", 2);
    BOOST_TEST(triangle.getEdge("0", "1").second == 2);
    triangle.setEdgeLabel("1", "0", 3);
    BOOST_TEST(triangle.getEdge("1", "0").second == 3);

    // Try defining the induced subgraph on {0, 1}
    std::unordered_set<Node*> subset;
    subset.insert(triangle.getNode("0"));
    subset.insert(triangle.getNode("1"));
    LabeledDigraph<int>* subgraph = triangle.subgraph(subset);
    std::vector<Node*> subnodes = subgraph->getNodes();    // Check that only nodes are 0 and 1
    BOOST_TEST(subnodes.size() == 2);
    BOOST_TEST(containsNode(subnodes, "0"));
    BOOST_TEST(containsNode(subnodes, "1"));
    std::vector<std::pair<Edge, int> > subedges = subgraph->getEdges();   // Check that 0 <-> 1 are the only edges
    BOOST_TEST(subedges.size() == 2);
    BOOST_TEST(containsEdge<int>(subedges, "0", "1", 2));
    BOOST_TEST(containsEdge<int>(subedges, "1", "0", 3));

    // Compute the steady-state probabilities with SVD
    std::vector<Node*> order;
    order.push_back(triangle.getNode("0"));
    order.push_back(triangle.getNode("1"));
    order.push_back(triangle.getNode("2"));
    MatrixXd laplacian = triangle.laplacian<double>(order);
    VectorXd ss_svd = triangle.getSteadyStateFromSVD<double>(order, 1e-10);
    for (unsigned i = 0; i < 3; ++i)
        BOOST_TEST(std::abs((laplacian * ss_svd)(i)) < 1e-10);

    // Compute the steady-state probabilities with the Chebotarev-Agaev recurrence
    VectorXd ss_rec = triangle.getSteadyStateFromRecurrence<double>(order);
    for (unsigned i = 0; i < 3; ++i)
        BOOST_TEST(std::abs((laplacian * ss_rec)(i)) < 1e-10);

    // Compute the steady-state probabilities by enumerating spanning trees
    VectorXd ss_tree = triangle.getSteadyStateFromTrees<double>(order);
    for (unsigned i = 0; i < 3; ++i)
        BOOST_TEST(std::abs((laplacian * ss_tree)(i)) < 1e-10); 
}

BOOST_AUTO_TEST_CASE(testSquare)
{
    /*
     * Test all public methods on the square graph. 
     */
    // Set all edge labels equal to one
    Polygon<int, 4> square;

    // Check that the graph contains the four nodes 0, 1, 2, 3
    std::vector<Node*> nodes = square.getNodes();
    BOOST_TEST(nodes.size() == 4);
    BOOST_TEST(containsNode(nodes, "0"));
    BOOST_TEST(containsNode(nodes, "1"));
    BOOST_TEST(containsNode(nodes, "2"));
    BOOST_TEST(containsNode(nodes, "3"));

    // Check that the graph contains edges (0,1), (1,0), (1,2), (2,1), (2,3),
    // (3,2), (3,0), (0,3) with edge labels all set to 1
    std::vector<std::pair<Edge, int> > edges = square.getEdges();
    BOOST_TEST(edges.size() == 8);
    BOOST_TEST(containsEdge<int>(edges, "0", "1", 1));
    BOOST_TEST(containsEdge<int>(edges, "1", "0", 1));
    BOOST_TEST(containsEdge<int>(edges, "1", "2", 1));
    BOOST_TEST(containsEdge<int>(edges, "2", "1", 1));
    BOOST_TEST(containsEdge<int>(edges, "2", "3", 1));
    BOOST_TEST(containsEdge<int>(edges, "3", "2", 1));
    BOOST_TEST(containsEdge<int>(edges, "3", "0", 1));
    BOOST_TEST(containsEdge<int>(edges, "0", "3", 1));

    // Try getting the nodes individually 
    BOOST_TEST(square.getNode("0") != nullptr);
    BOOST_TEST(square.getNode("1") != nullptr);
    BOOST_TEST(square.getNode("2") != nullptr);
    BOOST_TEST(square.getNode("3") != nullptr);

    // Try getting a non-existent node
    BOOST_TEST(square.getNode("4") == nullptr);

    // Try getting the edges individually
    for (unsigned i = 0; i < 4; ++i)
    {
        std::stringstream ss0, ss1;
        ss0 << i;
        if (i == 3)
            ss1 << 0;
        else
            ss1 << i + 1;

        // Try getting the forward edge
        std::pair<Edge, int> forward = square.getEdge(ss0.str(), ss1.str());
        BOOST_TEST((forward.first.first)->id == ss0.str());
        BOOST_TEST((forward.first.second)->id == ss1.str());
        BOOST_TEST(forward.second == 1);

        // Try getting the reverse edge
        std::pair<Edge, int> reverse = square.getEdge(ss1.str(), ss0.str());
        BOOST_TEST((reverse.first.first)->id == ss1.str());
        BOOST_TEST((reverse.first.second)->id == ss0.str());
        BOOST_TEST(reverse.second == 1);
    }

    // Try setting certain edge labels to different values 
    square.setEdgeLabel("2", "3", 2);
    BOOST_TEST(square.getEdge("2", "3").second == 2);
    square.setEdgeLabel("1", "0", 3);
    BOOST_TEST(square.getEdge("1", "0").second == 3);

    // Try defining the induced subgraph on {0, 1}
    std::unordered_set<Node*> subset;
    subset.insert(square.getNode("0"));
    subset.insert(square.getNode("1"));
    LabeledDigraph<int>* subgraph = square.subgraph(subset);
    std::vector<Node*> subnodes = subgraph->getNodes();    // Check that only nodes are 0 and 1
    BOOST_TEST(subnodes.size() == 2);
    BOOST_TEST(containsNode(subnodes, "0"));
    BOOST_TEST(containsNode(subnodes, "1"));
    std::vector<std::pair<Edge, int> > subedges = subgraph->getEdges();   // Check that 0 <-> 1 are the only edges
    BOOST_TEST(subedges.size() == 2);
    BOOST_TEST(containsEdge<int>(subedges, "0", "1", 1));
    BOOST_TEST(containsEdge<int>(subedges, "1", "0", 3));

    // Compute the steady-state probabilities with SVD
    std::vector<Node*> order;
    order.push_back(square.getNode("0"));
    order.push_back(square.getNode("1"));
    order.push_back(square.getNode("2"));
    order.push_back(square.getNode("3"));
    MatrixXd laplacian = square.laplacian<double>(order);
    VectorXd ss_svd = square.getSteadyStateFromSVD<double>(order, 1e-10);
    for (unsigned i = 0; i < 4; ++i)
        BOOST_TEST(std::abs((laplacian * ss_svd)(i)) < 1e-10);

    // Compute the steady-state probabilities with the Chebotarev-Agaev recurrence
    VectorXd ss_rec = square.getSteadyStateFromRecurrence<double>(order);
    for (unsigned i = 0; i < 4; ++i)
        BOOST_TEST(std::abs((laplacian * ss_rec)(i)) < 1e-10);

    // Compute the steady-state probabilities by enumerating spanning trees
    VectorXd ss_tree = square.getSteadyStateFromTrees<double>(order);
    for (unsigned i = 0; i < 4; ++i)
        BOOST_TEST(std::abs((laplacian * ss_tree)(i)) < 1e-10); 
}
