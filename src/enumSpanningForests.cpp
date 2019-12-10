#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "../include/digraph.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/9/2019
 */

int main(int argc, char** argv)
{
    // Check that the command-line arguments have been specified
    if (argc != 4)
    {
        std::cout << "Invalid call signature\n\nHelp:\n\n"
                  << "    ./enumSpanningForests.cpp [INPUT FILE] [ROOT 1] [ROOT 2]\n\n";
        return -1;
    }

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
        infile.close();
    }
    else
    {
        std::stringstream ss; 
        ss << "Input file cannot be opened: " << argv[1];
        throw std::runtime_error(ss.str());
    }

    // Compute the spanning forests rooted at the given pair of nodes
    std::string root1 = argv[2];
    std::string root2 = argv[3];
    std::vector<std::vector<std::pair<std::string, std::string> > > forests
        = graph->enumDoubleSpanningForests(root1, root2);

    // Output forests to stdout
    for (auto&& forest : forests)
    {
        for (auto&& e : forest)
        {
            std::cout << e.first << "," << e.second << "\t";
        }
        std::cout << std::endl;
    }

    delete graph;
    return 0;
}
