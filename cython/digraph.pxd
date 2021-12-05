"""
.pxd declarations file for include/digraph.hpp. 

Author:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    12/5/2021
"""

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set

cdef extern from '../include/digraph.hpp':
    cdef cppclass Node:
        Node(string id)
        bint operator==(Node& other)
        bint operator!=(Node& other)

    cdef cppclass LabeledDigraph[T]:
        LabeledDigraph()
        unsigned getNumNodes()
        Node* addNode(string id)
        void removeNode(string id)
        bint hasNode(string id)
        vector[string] getAllNodeIds()
        void addEdge(string source_id, string target_id, T label)
        void removeEdge(string source_id, string target_id)
        bint hasEdge(string source_id, string target_id)
        T getEdgeLabel(string source_id, string target_id)
        void setEdgeLabel(string source_id, string target_id, T value)
        #LabeledDigraph[T]* subgraph(unordered_set[Node*] nodes)
        void clear()
        #LabeledDigraph[U]* copy[U]()
        #void copy[U](LabeledDigraph[U]* graph)

        #TODO Expose methods with Eigen types

