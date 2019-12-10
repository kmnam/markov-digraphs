#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "../include/digraph.hpp"
#include "../include/forests.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/9/2019
 */

int main(int argc, char** argv)
{
    // Parse the graph given in the input file
    std::ifstream infile(argv[1]);
    MarkovDigraph<double>* graph = new MarkovDigraph<double>();
    if (infile.is_open())
    {
        std::string line;
        while (std::getline(infile, line))
        {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (std::getline(iss, token, ',')) tokens.push_back(token);
            if (tokens.size() != 2)
            {
                std::stringstream ss;
                ss << "Edge incorrectly specified: " << line; 
                throw std::runtime_error(ss.str()); 
            }
            graph->addEdge(tokens[0], tokens[1], 1.0);
        }
    }
    infile.close();

    // Compute the spanning forests rooted at the given pair of nodes
    std::string root1 = argv[2];
    std::string root2 = argv[3];
    std::vector<Node<double>*> nodes = graph->getNodes();
    auto found_root1 = std::find_if(
        nodes.begin(), nodes.end(), [root1](const Node<double>* v){ return v->id == root1; }
    );
    auto found_root2 = std::find_if(
        nodes.begin(), nodes.end(), [root2](const Node<double>* v){ return v->id == root2; }
    );
    if (found_root1 == nodes.end())
    {
        std::stringstream ss;
        ss << "Input node is not a node in the graph: " << root1;
        throw std::runtime_error(ss.str());
    }
    if (found_root2 == nodes.end())
    {
        std::stringstream ss;
        ss << "Input node is not a node in the graph: " << root2;
        throw std::runtime_error(ss.str());
    }
    std::vector<std::vector<Edge<double> > > forests
        = enumDoubleSpanningForests<double>(graph, *found_root1, *found_root2);
    for (auto&& forest : forests)
    {
        for (auto&& e : forest)
        {
            std::cout << e.first->id << "," << e.second->id << "\t";
        }
        std::cout << std::endl;
    }

    delete graph;
    return 0;
}
