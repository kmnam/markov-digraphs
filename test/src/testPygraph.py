#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath('../..'))
import unittest
import numpy as np
import pygraph

############################################################################
class TestPreciseDigraph(unittest.TestCase):
    """
    Test module for the `PreciseDigraph` class. 

    This test module simply replicates the C++ tests for `LabeledDigraph`. 
    """
    def test_node_methods(self):
        """
        Test the following methods:
        - `PreciseDigraph.add_node()`
        - `PreciseDigraph.remove_node()`
        - `PreciseDigraph.has_node()`
        - `PreciseDigraph.get_num_nodes()`
        - `PreciseDigraph.get_all_node_ids()`
        """
        # Instantiate a graph, add three nodes, and check that self.numnodes 
        # increments correctly 
        graph = pygraph.PreciseDigraph()
        graph.add_node('first')
        self.assertEqual(graph.get_num_nodes(), 1)
        graph.add_node('second')
        self.assertEqual(graph.get_num_nodes(), 2)
        graph.add_node('third')
        self.assertEqual(graph.get_num_nodes(), 3)
        
        # Check that the list of node IDs obeys the order in which the nodes
        # were added
        node_ids = graph.get_all_node_ids()
        self.assertEqual(len(node_ids), 3)
        self.assertEqual(node_ids[0], 'first')
        self.assertEqual(node_ids[1], 'second')
        self.assertEqual(node_ids[2], 'third')

        # Remove the second node, and check that self.numnodes decrements
        # correctly
        graph.remove_node('second')
        self.assertEqual(graph.get_num_nodes(), 2)

        # Check that the list of node IDs obeys the order in which the nodes 
        # were added
        node_ids = graph.get_all_node_ids()
        self.assertEqual(len(node_ids), 2)
        self.assertEqual(node_ids[0], 'first')
        self.assertEqual(node_ids[1], 'third')

        # Try to remove a non-existent node; a RuntimeError should be raised
        try:
            graph.remove_node('this node does not exist')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised

        # Check that has_node() returns True/False for nodes in the graph and 
        # not in the graph, respectively 
        self.assertTrue(graph.has_node('first'))
        self.assertTrue(graph.has_node('third'))
        self.assertFalse(graph.has_node('second'))
        self.assertFalse(graph.has_node('here is another node that does not exist'))

    ########################################################################
    def test_edge_methods(self):
        """
        Test the following methods:
        - `PreciseDigraph.add_edge()`
        - `PreciseDigraph.remove_edge()`
        - `PreciseDigraph.has_edge()`
        - `PreciseDigraph.get_edge_label()`
        - `PreciseDigraph.set_edge_label()`
        """
        # Instantiate a graph, add three nodes and three edges, and check that
        # has_edge() returns True/False for edges in the graph and edges not 
        # in the graph, respectively
        graph = pygraph.PreciseDigraph()
        graph.add_node('first')
        graph.add_node('second')
        graph.add_node('third')
        graph.add_edge('first', 'third', 1)
        graph.add_edge('third', 'first', 2)
        graph.add_edge('second', 'third', 3)
        self.assertTrue(graph.has_edge('first', 'third'))
        self.assertTrue(graph.has_edge('second', 'third'))
        self.assertTrue(graph.has_edge('third', 'first'))
        self.assertFalse(graph.has_edge('first', 'second'))
        self.assertFalse(graph.has_edge('second', 'first'))
        self.assertFalse(graph.has_edge('third', 'second'))

        # Check that has_edge() returns False when either of the queried nodes
        # does not exist
        self.assertFalse(graph.has_edge('first', 'here is one node that does not exist'))
        self.assertFalse(graph.has_edge('here is another node that is not there', 'second'))
        self.assertFalse(graph.has_edge('one more non-existent node', 'another non-existent node'))

        # Check that has_edge() returns False for all possible self-loops
        self.assertFalse(graph.has_edge('first', 'first'))
        self.assertFalse(graph.has_edge('second', 'second'))
        self.assertFalse(graph.has_edge('third', 'third'))

        # Try to remove an edge with non-existent source/target nodes
        try:
            graph.remove_edge('first', 'yet another non-existent node')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised
        try:
            graph.remove_edge('one more non-existent node, as before', 'second')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised
        try:
            graph.remove_edge('to dance beneath the diamond sky', 'with one hand waving free')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised

        # Remove an edge in the graph and check that has_edge() returns False
        graph.remove_edge('first', 'third')
        self.assertFalse(graph.has_edge('first', 'third'))

        # Add two more nodes and three more edges, for the sake of variety; 
        # then remove a node in the graph; then check that has_edge() returns 
        # False for all edges involving the removed node
        graph.add_node('fourth')
        graph.add_node('fifth')
        graph.add_edge('second', 'fourth', 50)
        graph.add_edge('fifth', 'first', 100)
        graph.add_edge('third', 'fifth', 1000)
        graph.remove_node('first')
        self.assertEqual(graph.get_num_nodes(), 4)
        self.assertFalse(graph.has_edge('third', 'first'))  # This edge was just removed
        self.assertFalse(graph.has_edge('fifth', 'first'))  # This edge was just removed
        self.assertFalse(graph.has_edge('first', 'third'))  # This edge was removed in the previous block

        # Check that has_edge() returns True for all edges remaining 
        self.assertTrue(graph.has_edge('second', 'third'))
        self.assertTrue(graph.has_edge('second', 'fourth'))
        self.assertTrue(graph.has_edge('third', 'fifth'))

        # Check that has_edge() returns False for edges that weren't there to
        # begin with 
        self.assertFalse(graph.has_edge('second', 'fifth'))  # This edge never existed

        # Check that get_edge_label() returns the correct values for edges that exist
        self.assertEqual(graph.get_edge_label('second', 'third'), 3)
        self.assertEqual(graph.get_edge_label('second', 'fourth'), 50)
        self.assertEqual(graph.get_edge_label('third', 'fifth'), 1000)

        # Check that get_edge_label() raises an exception if the queried edge 
        # does not exist
        try:
            graph.get_edge_label('second', 'fifth')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised 
        else:
            self.assertTrue(False)    # Break here if an exception is not raised

        # Check that get_edge_label() raises an exception if either of the
        # queried nodes doesn't exist
        try:
            graph.get_edge_label('first', 'fifth')    # 'first' does not exist anymore
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised
        try:
            graph.get_edge_label('second', 'here is another node that never existed')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised
        try:
            graph.get_edge_label('get jailed, jump bail', 'join the army if you fail')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised

        # Check that set_edge_label() changes the desired edge label
        graph.set_edge_label('second', 'fourth', 75)
        self.assertEqual(graph.get_edge_label('second', 'fourth'), 75)

        # Check that all other edge labels remain as they were
        self.assertEqual(graph.get_edge_label('second', 'third'), 3)
        self.assertEqual(graph.get_edge_label('third', 'fifth'), 1000)

        # Check that set_edge_label() raises an exception if the queried edge
        # does not exist 
        try:
            graph.set_edge_label('third', 'fourth', 2.5)
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised

        # Check that set_edge_label() raises an exception if either of the
        # queried nodes does not exist
        try:
            graph.set_edge_label('first', 'second', 16)    # 'first' does not exist anymore 
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised
        try:
            graph.set_edge_label('second', 'here is yet another node that never came into being', 7e+10)
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised
        try:
            graph.set_edge_label('every one of them words rang true', 'and glowed like burning coal', 0.0001)
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised

    ########################################################################
    def test_clear(self):
        """
        Test `PreciseDigraph.clear()`. 
        """
        # Add a succession of nodes and edges, then clear, then check that 
        # the graph is empty 
        graph = pygraph.PreciseDigraph()
        graph.add_node('first')
        graph.add_node('second')
        graph.add_node('third')
        graph.add_node('fourth')
        graph.add_node('fifth')
        graph.add_edge('first', 'third', 1)
        graph.add_edge('third', 'first', 2)
        graph.add_edge('second', 'third', 3)
        graph.add_edge('second', 'fourth', 50)
        graph.add_edge('fifth', 'first', 100)
        graph.add_edge('third', 'fifth', 1000)
        graph.clear()
        self.assertEqual(graph.get_num_nodes(), 0)
        self.assertEqual(len(graph.get_all_node_ids()), 0)

        # Check that has_node() returns False for every previously inserted node
        self.assertFalse(graph.has_node('first'))
        self.assertFalse(graph.has_node('second'))
        self.assertFalse(graph.has_node('third'))
        self.assertFalse(graph.has_node('fourth'))
        self.assertFalse(graph.has_node('fifth'))

        # Check that remove_node() throws an exception if we try to remove any
        # (already-removed) node again
        try:
            graph.remove_node('first')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised
        try:
            graph.remove_node('second')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised
        try:
            graph.remove_node('third')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised
        try:
            graph.remove_node('fourth')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised
        try:
            graph.remove_node('fifth')
        except RuntimeError:
            pass                      # Do nothing if an exception is raised
        else:
            self.assertTrue(False)    # Break here if an exception is not raised

        # Check that has_edge() returns False for all possible combinations 
        # of the (already-removed) nodes
        node_ids = ['first', 'second', 'third', 'fourth', 'fifth']
        for s in node_ids:
            for t in node_ids:
                self.assertFalse(graph.has_edge(s, t))

    ########################################################################
    def test_get_laplacian(self):
        """
        Test `PreciseDigraph.get_laplacian()`.

        Check that `get_laplacian()` returns an *exactly* correct Laplacian
        matrix for a 5-vertex graph with reasonably sized integer-valued
        edge labels (1 <= x <= 1000). 
        """
        graph = pygraph.PreciseDigraph()
        graph.add_node('first')
        graph.add_node('second')
        graph.add_node('third')
        graph.add_node('fourth')
        graph.add_node('fifth')
        graph.add_edge('first', 'third', 1)
        graph.add_edge('third', 'first', 2)
        graph.add_edge('second', 'third', 3)
        graph.add_edge('second', 'fourth', 50)
        graph.add_edge('fifth', 'first', 100)
        graph.add_edge('third', 'fifth', 1000)
        laplacian = graph.get_laplacian()
        self.assertEqual(laplacian.shape, (5, 5))
        for i in range(5):
            for j in range(5):
                if i == 0 and j == 2:      # label(third -> first) == 2
                    self.assertEqual(laplacian[i, j], 2)
                elif i == 0 and j == 4:    # label(fifth -> first) == 100
                    self.assertEqual(laplacian[i, j], 100)
                elif i == 2 and j == 0:    # label(first -> third) == 1
                    self.assertEqual(laplacian[i, j], 1)
                elif i == 2 and j == 1:    # label(second -> third) == 3
                    self.assertEqual(laplacian[i, j], 3)
                elif i == 3 and j == 1:    # label(second -> fourth) == 50
                    self.assertEqual(laplacian[i, j], 50)
                elif i == 4 and j == 2:    # label(third -> fifth) == 1000
                    self.assertEqual(laplacian[i, j], 1000)
                elif i == 0 and j == 0:    # Sum of all outgoing edges from first == 1
                    self.assertEqual(laplacian[i, j], -1)
                elif i == 1 and j == 1:    # Sum of all outgoing edges from second == 53
                    self.assertEqual(laplacian[i, j], -53)
                elif i == 2 and j == 2:    # Sum of all outgoing edges from third == 1002
                    self.assertEqual(laplacian[i, j], -1002)
                elif i == 4 and j == 4:    # Sum of all outgoing edges from fifth == 100
                    self.assertEqual(laplacian[i, j], -100)
                else:                      # All other entries should be zero
                    self.assertEqual(laplacian[i, j], 0)

    ########################################################################
    def test_get_spanning_forest_matrix(self):
        """
        Test `PreciseDigraph.get_spanning_forest_matrix()`. 

        Check that `get_spanning_forest_matrix()` returns *exactly* correct
        spanning forest matrices for a 5-vertex graph.  
        """
        graph = pygraph.PreciseDigraph()
        graph.add_node('first')
        graph.add_node('second')
        graph.add_node('third')
        graph.add_node('fourth')
        graph.add_node('fifth')
        graph.add_edge('first', 'second', 1)
        graph.add_edge('first', 'fifth', 13)
        graph.add_edge('second', 'first', 20)
        graph.add_edge('second', 'third', 67)
        graph.add_edge('third', 'second', 42)
        graph.add_edge('third', 'fourth', 1007)
        graph.add_edge('fourth', 'third', 17)
        graph.add_edge('fourth', 'fifth', 512)
        graph.add_edge('fifth', 'first', 179)
        graph.add_edge('fifth', 'fourth', 285)

        # Check that the zeroth spanning forest matrix equals the identity
        forest_zero = graph.get_spanning_forest_matrix(0)
        self.assertEqual(forest_zero.shape, (5, 5))
        for i in range(5):
            for j in range(5):
                if i == j:
                    self.assertEqual(forest_zero[i, j], 1)
                else:
                    self.assertEqual(forest_zero[i, j], 0)

        # Check that the first spanning forest matrix has the following entries:
        # 1) i != j and abs(i - j) == 1 or abs(i - j) == 4: A(i, j) == label(i -> j)
        # 2) i != j and abs(i - j) == 2 or abs(i - j) == 3: A(i, j) == 0
        # 3) i == j: A(i, j) == sum of all edge labels except for all edges leaving j
        forest_one = graph.get_spanning_forest_matrix(1)
        node_ids = ['first', 'second', 'third', 'fourth', 'fifth']
        sum_of_all_edge_labels = 1 + 13 + 20 + 67 + 42 + 1007 + 17 + 512 + 179 + 285
        for i in range(5):
            for j in range(5):
                if i == j:
                    self.assertEqual(
                        forest_one[i, j],
                        sum_of_all_edge_labels\
                            - graph.get_edge_label(node_ids[i], node_ids[(i-1) % 5])\
                            - graph.get_edge_label(node_ids[i], node_ids[(i+1) % 5])
                    )
                elif abs(i - j) == 1 or abs(i - j) == 4:
                    self.assertEqual(
                        forest_one[i, j], graph.get_edge_label(node_ids[i], node_ids[j])
                    )
                else:
                    self.assertEqual(forest_one[i, j], 0)

        # Check that the second spanning forest matrix has the correct entries ...
        forest_two = graph.get_spanning_forest_matrix(2) 

        # To take advantage of the symmetry of the cycle graph, we introduce a 
        # "node mapping" that allows us to calculate the spanning forest weights
        # without having to repeatedly rewrite equivalent products of edge labels
        def all_two_edge_forests(node_map):
            _first = node_map['first']
            _second = node_map['second']
            _third = node_map['third']
            _fourth = node_map['fourth']
            _fifth = node_map['fifth']
            return (
                graph.get_edge_label(_second, _first) * (
                    graph.get_edge_label(_third, _second) +
                    graph.get_edge_label(_third, _fourth) +
                    graph.get_edge_label(_fourth, _third) +
                    graph.get_edge_label(_fourth, _fifth) + 
                    graph.get_edge_label(_fifth, _fourth) +
                    graph.get_edge_label(_fifth, _first)
                ) + graph.get_edge_label(_second, _third) * (
                    graph.get_edge_label(_third, _fourth) +
                    graph.get_edge_label(_fourth, _third) +
                    graph.get_edge_label(_fourth, _fifth) +
                    graph.get_edge_label(_fifth, _fourth) +
                    graph.get_edge_label(_fifth, _first)
                ) + graph.get_edge_label(_third, _second) * (
                    graph.get_edge_label(_fourth, _third) +
                    graph.get_edge_label(_fourth, _fifth) +
                    graph.get_edge_label(_fifth, _fourth) +
                    graph.get_edge_label(_fifth, _first)
                ) + graph.get_edge_label(_third, _fourth) * (
                    graph.get_edge_label(_fourth, _fifth) +
                    graph.get_edge_label(_fifth, _fourth) +
                    graph.get_edge_label(_fifth, _first)
                ) + graph.get_edge_label(_fourth, _third) * (
                    graph.get_edge_label(_fifth, _fourth) +
                    graph.get_edge_label(_fifth, _first)
                ) + graph.get_edge_label(_fourth, _fifth) * graph.get_edge_label(_fifth, _first)
            )

        def two_edge_forests_with_path_from_1_to_2(node_map):
            _first = node_map['first']
            _second = node_map['second']
            _third = node_map['third']
            _fourth = node_map['fourth']
            _fifth = node_map['fifth']
            return (
                graph.get_edge_label(_first, _second) * (
                    graph.get_edge_label(_third, _second) +
                    graph.get_edge_label(_third, _fourth) + 
                    graph.get_edge_label(_fourth, _third) + 
                    graph.get_edge_label(_fourth, _fifth) + 
                    graph.get_edge_label(_fifth, _fourth) + 
                    graph.get_edge_label(_fifth, _first)
                )
            )

        def two_edge_forests_with_path_from_1_to_5(node_map):
            _first = node_map['first']
            _second = node_map['second']
            _third = node_map['third']
            _fourth = node_map['fourth']
            _fifth = node_map['fifth']
            return (
                graph.get_edge_label(_first, _fifth) * (
                    graph.get_edge_label(_second, _first) +
                    graph.get_edge_label(_second, _third) + 
                    graph.get_edge_label(_third, _second) + 
                    graph.get_edge_label(_third, _fourth) + 
                    graph.get_edge_label(_fourth, _third) + 
                    graph.get_edge_label(_fourth, _fifth)
                )
            )

        # (0, 0): 2-edge forests with 1 as a root
        map_to_first = {s: s for s in node_ids}
        self.assertEqual(forest_two[0, 0], all_two_edge_forests(map_to_first))
        # (0, 1): 2-edge forests with 2 as a root and path from 1 to 2
        self.assertEqual(forest_two[0, 1], two_edge_forests_with_path_from_1_to_2(map_to_first))
        # (0, 2): 2-edge forests with 3 as a root and path from 1 to 3
        self.assertEqual(
            forest_two[0, 2],
            graph.get_edge_label('first', 'second') * graph.get_edge_label('second', 'third')
        )
        # (0, 3): 2-edge forests with 4 as a root and path from 1 to 4
        self.assertEqual(
            forest_two[0, 3],
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('fifth', 'fourth')
        )
        # (0, 4): 2-edge forests with 5 as a root and path from 1 to 5
        self.assertEqual(forest_two[0, 4], two_edge_forests_with_path_from_1_to_5(map_to_first))
        # (1, 0): 2-edge forests with 1 as a root and path from 2 to 1
        map_to_second = {node_ids[i]: node_ids[(i+1) % 5] for i in range(5)}
        self.assertEqual(forest_two[1, 0], two_edge_forests_with_path_from_1_to_5(map_to_second))
        # (1, 1): 2-edge forests with 2 as a root
        self.assertEqual(forest_two[1, 1], all_two_edge_forests(map_to_second))
        # (1, 2): 2-edge forests with 3 as a root and path from 2 to 3
        self.assertEqual(forest_two[1, 2], two_edge_forests_with_path_from_1_to_2(map_to_second))
        # (1, 3): 2-edge forests with 4 as a root and path from 2 to 4
        self.assertEqual(
            forest_two[1, 3],
            graph.get_edge_label('second', 'third') * graph.get_edge_label('third', 'fourth')
        )
        # (1, 4): 2-edge forests with 5 as a root and path from 2 to 5
        self.assertEqual(
            forest_two[1, 4],
            graph.get_edge_label('second', 'first') * graph.get_edge_label('first', 'fifth')
        )
        # (2, 0): 2-edge forests with 1 as a root and path from 3 to 1
        map_to_third = {node_ids[i]: node_ids[(i+2) % 5] for i in range(5)}
        self.assertEqual(
            forest_two[2, 0],
            graph.get_edge_label('third', 'second') * graph.get_edge_label('second', 'first')
        )
        # (2, 1): 2-edge forests with 2 as a root and path from 3 to 2
        self.assertEqual(forest_two[2, 1], two_edge_forests_with_path_from_1_to_5(map_to_third))
        # (2, 2): 2-edge forests with 3 as a root
        self.assertEqual(forest_two[2, 2], all_two_edge_forests(map_to_third))
        # (2, 3): 2-edge forests with 4 as a root and path from 3 to 4
        self.assertEqual(forest_two[2, 3], two_edge_forests_with_path_from_1_to_2(map_to_third))
        # (2, 4): 2-edge forests with 5 as a root and path from 3 to 5
        self.assertEqual(
            forest_two[2, 4],
            graph.get_edge_label('third', 'fourth') * graph.get_edge_label('fourth', 'fifth')
        )
        # (3, 0): 2-edge forests with 1 as a root and path from 4 to 1
        map_to_fourth = {node_ids[i]: node_ids[(i+3) % 5] for i in range(5)}
        self.assertEqual(
            forest_two[3, 0],
            graph.get_edge_label('fourth', 'fifth') * graph.get_edge_label('fifth', 'first')
        )
        # (3, 1): 2-edge forests with 2 as a root and path from 4 to 2
        self.assertEqual(
            forest_two[3, 1],
            graph.get_edge_label('fourth', 'third') * graph.get_edge_label('third', 'second')
        )
        # (3, 2): 2-edge forests with 3 as a root and path from 4 to 3
        self.assertEqual(forest_two[3, 2], two_edge_forests_with_path_from_1_to_5(map_to_fourth))
        # (3, 3): 2-edge forests with 4 as a root
        self.assertEqual(forest_two[3, 3], all_two_edge_forests(map_to_fourth))
        # (3, 4): 2-edge forests wtih 5 as a root and path from 4 to 5
        self.assertEqual(forest_two[3, 4], two_edge_forests_with_path_from_1_to_2(map_to_fourth))
        # (4, 0): 2-edge forests with 1 as a root and path from 5 to 1
        map_to_fifth = {node_ids[i]: node_ids[(i+4) % 5] for i in range(5)}
        self.assertEqual(forest_two[4, 0], two_edge_forests_with_path_from_1_to_2(map_to_fifth))
        # (4, 1): 2-edge forests with 2 as a root and path from 5 to 2
        self.assertEqual(
            forest_two[4, 1],
            graph.get_edge_label('fifth', 'first') * graph.get_edge_label('first', 'second')
        )
        # (4, 2): 2-edge forests with 3 as a root and path from 5 to 3
        self.assertEqual(
            forest_two[4, 2], 
            graph.get_edge_label('fifth', 'fourth') * graph.get_edge_label('fourth', 'third')
        )
        # (4, 3): 2-edge forests with 4 as a root and path from 5 to 4
        self.assertEqual(forest_two[4, 3], two_edge_forests_with_path_from_1_to_5(map_to_fifth))
        # (4, 4): 2-edge forests with 5 as a root
        self.assertEqual(forest_two[4, 4], all_two_edge_forests(map_to_fifth))

        # Set some of the edge labels to be smaller, so that the spanning tree
        # weights can be exactly represented with double-precision floats
        graph.set_edge_label('third', 'fourth', 60)
        graph.set_edge_label('fourth', 'fifth', 22)
        graph.set_edge_label('fifth', 'first', 8)
        graph.set_edge_label('fifth', 'fourth', 55)

        # Now all spanning tree weights should be less than 67 ** 4 == 20,151,121
        #
        # Skip the third spanning forest matrix, and examine the fourth ...
        forest_four = graph.get_spanning_forest_matrix(4)
        # (0, 0): trees with 1 as a root
        self.assertEqual(
            forest_four[0, 0],
            graph.get_edge_label('second', 'first') * graph.get_edge_label('third', 'second') *
            graph.get_edge_label('fourth', 'third') * graph.get_edge_label('fifth', 'fourth') +
            graph.get_edge_label('second', 'first') * graph.get_edge_label('third', 'second') *
            graph.get_edge_label('fourth', 'third') * graph.get_edge_label('fifth', 'first') +
            graph.get_edge_label('second', 'first') * graph.get_edge_label('third', 'second') *
            graph.get_edge_label('fourth', 'fifth') * graph.get_edge_label('fifth', 'first') +
            graph.get_edge_label('second', 'first') * graph.get_edge_label('third', 'fourth') *
            graph.get_edge_label('fourth', 'fifth') * graph.get_edge_label('fifth', 'first') +
            graph.get_edge_label('second', 'third') * graph.get_edge_label('third', 'fourth') *
            graph.get_edge_label('fourth', 'fifth') * graph.get_edge_label('fifth', 'first')
        )
        # (0, 1): trees with 2 as a root
        self.assertEqual(
            forest_four[0, 1],
            graph.get_edge_label('first', 'second') * graph.get_edge_label('third', 'fourth') *
            graph.get_edge_label('fourth', 'fifth') * graph.get_edge_label('fifth', 'first') +
            graph.get_edge_label('first', 'second') * graph.get_edge_label('third', 'second') *
            graph.get_edge_label('fourth', 'fifth') * graph.get_edge_label('fifth', 'first') +
            graph.get_edge_label('first', 'second') * graph.get_edge_label('third', 'second') *
            graph.get_edge_label('fourth', 'third') * graph.get_edge_label('fifth', 'first') +
            graph.get_edge_label('first', 'second') * graph.get_edge_label('third', 'second') *
            graph.get_edge_label('fourth', 'third') * graph.get_edge_label('fifth', 'fourth') +
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('third', 'second') *
            graph.get_edge_label('fourth', 'third') * graph.get_edge_label('fifth', 'fourth')
        )
        # (0, 2): trees with 3 as a root
        self.assertEqual(
            forest_four[0, 2],
            graph.get_edge_label('first', 'second') * graph.get_edge_label('second', 'third') *
            graph.get_edge_label('fourth', 'fifth') * graph.get_edge_label('fifth', 'first') +
            graph.get_edge_label('first', 'second') * graph.get_edge_label('second', 'third') *
            graph.get_edge_label('fourth', 'third') * graph.get_edge_label('fifth', 'first') +
            graph.get_edge_label('first', 'second') * graph.get_edge_label('second', 'third') *
            graph.get_edge_label('fourth', 'third') * graph.get_edge_label('fifth', 'fourth') +
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('second', 'third') *
            graph.get_edge_label('fourth', 'third') * graph.get_edge_label('fifth', 'fourth') +
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('second', 'first') *
            graph.get_edge_label('fourth', 'third') * graph.get_edge_label('fifth', 'fourth')
        )
        # (0, 3): trees with 4 as a root
        self.assertEqual(
            forest_four[0, 3],
            graph.get_edge_label('first', 'second') * graph.get_edge_label('second', 'third') *
            graph.get_edge_label('third', 'fourth') * graph.get_edge_label('fifth', 'first') +
            graph.get_edge_label('first', 'second') * graph.get_edge_label('second', 'third') *
            graph.get_edge_label('third', 'fourth') * graph.get_edge_label('fifth', 'fourth') +
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('second', 'third') *
            graph.get_edge_label('third', 'fourth') * graph.get_edge_label('fifth', 'fourth') +
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('second', 'first') *
            graph.get_edge_label('third', 'fourth') * graph.get_edge_label('fifth', 'fourth') +
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('second', 'first') *
            graph.get_edge_label('third', 'second') * graph.get_edge_label('fifth', 'fourth')
        )
        # (0, 4): trees with 5 as a root
        self.assertEqual(
            forest_four[0, 4],
            graph.get_edge_label('first', 'second') * graph.get_edge_label('second', 'third') *
            graph.get_edge_label('third', 'fourth') * graph.get_edge_label('fourth', 'fifth') +
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('second', 'third') *
            graph.get_edge_label('third', 'fourth') * graph.get_edge_label('fourth', 'fifth') +
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('second', 'first') *
            graph.get_edge_label('third', 'fourth') * graph.get_edge_label('fourth', 'fifth') +
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('second', 'first') *
            graph.get_edge_label('third', 'second') * graph.get_edge_label('fourth', 'fifth') +
            graph.get_edge_label('first', 'fifth') * graph.get_edge_label('second', 'first') *
            graph.get_edge_label('third', 'second') * graph.get_edge_label('fourth', 'third')
        )
        # Check that each column has the same entry in each row 
        for i in range(5):
            for j in range(5):
                self.assertEqual(forest_four[0, i], forest_four[j, i])

    ########################################################################
    def test_get_steady_state_from_svd(self):
        """
        Test `PreciseDigraph.get_steady_state_from_svd()`. 

        Check that `get_steady_state_from_svd()` returns an *approximately*
        correct steady-state vector with highly precise multiple-precision 
        internal scalars (100 digits in the mantissa). 
        """
        graph = pygraph.PreciseDigraph()
        graph.add_node('first')
        graph.add_node('second')
        graph.add_node('third')
        graph.add_node('fourth')
        graph.add_node('fifth')
        graph.add_edge('first', 'second', 1)
        graph.add_edge('first', 'fifth', 13)
        graph.add_edge('second', 'first', 20)
        graph.add_edge('second', 'third', 67)
        graph.add_edge('third', 'second', 42)
        graph.add_edge('third', 'fourth', 1007)
        graph.add_edge('fourth', 'third', 17)
        graph.add_edge('fourth', 'fifth', 512)
        graph.add_edge('fifth', 'first', 179)
        graph.add_edge('fifth', 'fourth', 285)

        # Compute the steady-state vector 
        laplacian = graph.get_laplacian()
        ss = graph.get_steady_state_from_svd()

        # Compute the error incurred by the steady-state computation, as 
        # the magnitude of the steady-state vector left-multiplied by the 
        # Laplacian matrix (this product should be *zero*)
        err = np.abs(np.dot(laplacian, ss)).sum()

        # Check that this error is tiny
        # 
        # NOTE: The threshold chosen here in minuscule, but seeing as it was
        # deliberately chosen after some experimentation, this test should
        # not be taken as exact
        self.assertTrue(err < 1e-14)

    ########################################################################
    def test_get_steady_state_from_recurrence(self):
        """
        Test `PreciseDigraph.get_steady_state_from_recurrence()`. 

        Check that `get_steady_state_from_recurrence()` returns an
        *approximately* correct steady-state vector with highly precise
        multiple-precision internal scalars (100 digits in the mantissa). 
        """
        graph = pygraph.PreciseDigraph()
        graph.add_node('first')
        graph.add_node('second')
        graph.add_node('third')
        graph.add_node('fourth')
        graph.add_node('fifth')
        graph.add_edge('first', 'second', 1)
        graph.add_edge('first', 'fifth', 13)
        graph.add_edge('second', 'first', 20)
        graph.add_edge('second', 'third', 67)
        graph.add_edge('third', 'second', 42)
        graph.add_edge('third', 'fourth', 1007)
        graph.add_edge('fourth', 'third', 17)
        graph.add_edge('fourth', 'fifth', 512)
        graph.add_edge('fifth', 'first', 179)
        graph.add_edge('fifth', 'fourth', 285)

        # Compute the steady-state vector 
        laplacian = graph.get_laplacian()
        ss = graph.get_steady_state_from_recurrence(False)

        # Compute the error incurred by the steady-state computation, as 
        # the magnitude of the steady-state vector left-multiplied by the 
        # Laplacian matrix (this product should be *zero*)
        err = np.abs(np.dot(laplacian, ss)).sum()

        # Check that this error is tiny
        # 
        # NOTE: The threshold chosen here in minuscule, but seeing as it was
        # deliberately chosen after some experimentation, this test should
        # not be taken as exact
        self.assertTrue(err < 1e-14)

    ########################################################################
    def test_viz_digraph(self):
        """
        Use `viz_digraph()` to draw a 5-vertex cycle graph. 
        """
        graph = pygraph.PreciseDigraph()
        graph.add_node('first')
        graph.add_node('second')
        graph.add_node('third')
        graph.add_node('fourth')
        graph.add_node('fifth')
        graph.add_edge('first', 'second', 1)
        graph.add_edge('first', 'fifth', 13)
        graph.add_edge('second', 'first', 20)
        graph.add_edge('second', 'third', 67)
        graph.add_edge('third', 'second', 42)
        graph.add_edge('third', 'fourth', 1007)
        graph.add_edge('fourth', 'third', 17)
        graph.add_edge('fourth', 'fifth', 512)
        graph.add_edge('fifth', 'first', 179)
        graph.add_edge('fifth', 'fourth', 285)

        # Draw the graph with the dot, neato, and circo layout algorithms
        pygraph.viz_digraph(graph, 'dot', 'png', 'test-dot.png')
        pygraph.viz_digraph(graph, 'neato', 'png', 'test-neato.png')
        pygraph.viz_digraph(graph, 'circo', 'png', 'test-circo.png')

############################################################################
if __name__ == '__main__':
    unittest.main()
