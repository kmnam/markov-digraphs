/** \file include/viz.hpp 
 *
 *  Functions for producing visualizations of `LabeledDigraph` instances 
 *  with the graphviz library.
 *
 *  **Author:**
 *      Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 *  **Last updated:**
 *      12/14/2021 
 */

#ifndef LABELED_DIGRAPH_GRAPHVIZ_HPP 
#define LABELED_DIGRAPH_GRAPHVIZ_HPP

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>
#include "digraph.hpp"

/**
 * Produce a new file with a visualization of the given `LabeledDigraph` 
 * instance with the graphviz library. 
 *
 * @param graph    `LabeledDigraph` instance. 
 * @param layout   Graphviz layout algorithm (`"dot"`, `"neato"`, `"fdp"`, 
 *                 `"sfdp"`, `"twopi"`, `"circo"`).
 * @param format   Format of output file (`"png"`, `"pdf"`, etc.)
 * @param filename Output file name.
 * @param context  Pre-defined graphviz context.  
 */
template <typename InternalType, typename IOType>
void vizLabeledDigraph(LabeledDigraph<InternalType, IOType>& graph,
                       std::string layout,
                       std::string format,
                       std::string filename,  
                       GVC_t* context)
{
    // Instantiate an anonymous graph 
    Agraph_t* g = agopen(NULL, Agdirected, 0);

    // Add each node first  
    for (std::string node_id : graph.getAllNodeIds())
    {
        // Parse each node ID into a non-const char array
        char* _node_id = new char[node_id.length() + 1]; 
        strcpy(_node_id, node_id.c_str()); 
        agnode(g, _node_id, true);
        delete [] _node_id;
    }
    
    // Then add the edges 
    for (std::string node_id : graph.getAllNodeIds())
    {
        // Parse each node ID (again) into a non-const char array
        char* _node_id = new char[node_id.length() + 1]; 
        strcpy(_node_id, node_id.c_str()); 
        Agnode_t* source = agnode(g, _node_id, false); 

        // Get all outgoing edges from each node 
        std::vector<std::pair<std::string, IOType> > edges_from_node = graph.getAllEdgesFromNode(node_id);
        for (auto&& entry : edges_from_node)
        {
            // Parse each target node ID into a non-const char array
            char* _target_id = new char[entry.first.length() + 1]; 
            strcpy(_target_id, entry.first.c_str()); 
            Agnode_t* target = agnode(g, _target_id, false); 
            Agedge_t* edge = agedge(g, source, target, NULL, true);
            delete [] _target_id;
        }
        delete [] _node_id;
    }

    // Draw the graph
    gvLayout(context, g, layout.c_str());
    gvRenderFilename(context, g, format.c_str(), filename.c_str());
    gvFreeLayout(context, g);  
    agclose(g);  
}

#endif 
