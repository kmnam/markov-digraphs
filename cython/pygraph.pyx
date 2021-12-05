# distutils: language = c++

"""
.pyx declarations for include/digraph.hpp.

Author:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    12/5/2021
"""
from libcpp.string cimport string
from digraph cimport LabeledDigraph

##################################################################
cdef class Graph:
    cdef LabeledDigraph[double] graph

    def get_num_nodes(self):
        return self.graph.getNumNodes()

    def add_node(self, id: str):
        id_bytes = <string> id.encode('utf-8')
        self.graph.addNode(id_bytes)    # Don't return the pointer to Node, as addNode() does  

    def remove_node(self, id: str):
        id_bytes = <string> id.encode('utf-8')
        self.graph.removeNode(id_bytes)

    def has_node(self, id: str) -> bool:
        id_bytes = <string> id.encode('utf-8')
        return self.graph.hasNode(id_bytes)

    def get_all_nodes(self):
        """
        Return the node IDs as unicode strings.
        """
        node_ids_bytes = self.graph.getAllNodeIds()
        node_ids = []
        for s in node_ids_bytes:
            node_ids.append(s.decode('utf-8'))
        return node_ids

    def add_edge(self, source_id: str, target_id: str, label: double = 1):
        source_id_bytes = <string> source_id.encode('utf-8')
        target_id_bytes = <string> target_id.encode('utf-8')
        self.graph.addEdge(source_id_bytes, target_id_bytes, label)

    def remove_edge(self, source_id: str, target_id: str):
        source_id_bytes = <string> source_id.encode('utf-8')
        target_id_bytes = <string> target_id.encode('utf-8')
        self.graph.removeEdge(source_id_bytes, target_id_bytes)

    def has_edge(self, source_id: str, target_id: str):
        source_id_bytes = <string> source_id.encode('utf-8')
        target_id_bytes = <string> target_id.encode('utf-8')
        return self.graph.hasEdge(source_id_bytes, target_id_bytes)

    def get_edge_label(self, source_id: str, target_id: str):
        source_id_bytes = <string> source_id.encode('utf-8')
        target_id_bytes = <string> target_id.encode('utf-8')
        return self.graph.getEdgeLabel(source_id_bytes, target_id_bytes)

    def set_edge_label(self, source_id: str, target_id: str, value: double):
        source_id_bytes = <string> source_id.encode('utf-8')
        target_id_bytes = <string> target_id.encode('utf-8')
        self.graph.setEdgeLabel(source_id_bytes, target_id_bytes, value)

    def clear(self):
        self.graph.clear()


