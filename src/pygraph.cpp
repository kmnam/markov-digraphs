#include <boost/multiprecision/mpfr.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "../include/digraph.hpp"
#include "../include/graphs/line.hpp"
#include "../include/graphs/grid.hpp"
#include "../include/viz.hpp"

/**
 * Python bindings for the `LabeledDigraph<...>` class and other functions in 
 * `MarkovDigraphs` through pybind11. 
 *
 * **Author:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     8/3/2024
 */

namespace py = pybind11;

// Use variable-precision floating-point arithmetic 
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<0> > PreciseType;

/** ----------------------------------------------------------------- // 
 *           FUNCTIONS FOR DECLARING GENERIC PYGRAPH CLASSES          //
 *  ----------------------------------------------------------------- */
/**
 * A templated function for exposing versions of the `LabeledDigraph`
 * class with different numeric types to Python.
 *
 * @param m pybind11 module. 
 * @param class_name Name to be given to the (Python) class.
 */
template <typename T>
void declareLabeledDigraphClass(py::module& m, const std::string& class_name)
{
    py::class_<LabeledDigraph<T, double> >(m, class_name.c_str())
        .def(py::init<>(),
            R"delim(Empty constructor.)delim"
        )
        .def("get_num_nodes",
            &LabeledDigraph<T, double>::getNumNodes,
            R"delim(
    Return the number of nodes in the graph.
    
    :return: Number of nodes in the graph.
    :rtype: int
)delim"
        )
        .def("add_node",
            &LabeledDigraph<T, double>::addNode,
            R"delim(
    Add a node to the graph with the given ID.

    :param id: ID for new node.
    :type id: str
    :raise RuntimeError: If node already exists with the given ID.
)delim",
            py::arg("id")
        )
        .def("remove_node",
            &LabeledDigraph<T, double>::removeNode,
            R"delim(
    Remove a node from the graph with the given ID.

    :param id: ID of node to be removed.
    :type id: str
    :raise RuntimeError: If node with given ID does not exist.
)delim",
            py::arg("id")
        )
        .def("has_node",
            &LabeledDigraph<T, double>::hasNode,
            R"delim(
    Return True if node with given ID exists in the graph.

    :param id: ID of desired node.
    :type id: str
    :return: True if node exists with the given ID, False otherwise.
    :rtype: bool
)delim",
            py::arg("id")
        )
        .def("get_all_node_ids",
            &LabeledDigraph<T, double>::getAllNodeIds,
            R"delim(
    Return the list of IDs of all nodes in the graph, ordered
    according to the graph's canonical ordering of nodes.

    :return: List of all node IDs.
    :rtype: list
)delim" 
        )
        .def("add_edge",
            &LabeledDigraph<T, double>::addEdge,
            R"delim(
    Add an edge between two nodes.

    If either ID does not correspond to a node in the graph, this 
    function instantiates these nodes. 

    :param source_id: ID of source node of new edge.
    :type source_id: str 
    :param target_id: ID of target node of new edge.
    :type target_id: str
    :param label: Label on new edge.
    :type label: float
    :raise RuntimeError: If the edge already exists.
)delim",
            py::arg("source_id"),
            py::arg("target_id"),
            py::arg("label") = 1
        )
        .def("remove_edge",
            &LabeledDigraph<T, double>::removeEdge,
            R"delim(
    Remove the edge between the two given nodes.

    This method does nothing if the node does not exist but the
    nodes do, but throws an exception if either node does not 
    exist. 

    :param source_id: ID of source node of edge to be removed.
    :type source_id: str
    :param target_id: ID of target node of edge to be removed.
    :type target_id: str
    :raise RuntimeError: If either node does not exist.
)delim",
            py::arg("source_id"),
            py::arg("target_id")
        )
        .def("has_edge",
            static_cast<bool (LabeledDigraph<T, double>::*)(std::string, std::string) const>(
                &LabeledDigraph<T, double>::hasEdge
            ),
            R"delim(
    Return True if the specified edge exists, given the IDs of
    the two nodes, and False otherwise.

    This method also returns False if either node does not exist.

    :param source_id: ID of source node.
    :type source_id: str
    :param target_id: ID of target node.
    :type target_id: str
    :return: True if the edge exists, False otherwise.
    :rtype: bool
)delim",
            py::arg("source_id"),
            py::arg("target_id")
        )
        .def("get_all_edges_from_node", 
            static_cast<std::vector<std::pair<std::string, double> > (LabeledDigraph<T, double>::*)(std::string)>(
                &LabeledDigraph<T, double>::getAllEdgesFromNode
            ), 
            R"delim(
    Return a list of nodes to which there are edges outgoing from
    the given source node, along with the corresponding edge labels, 
    given the ID of the source node. 

    :param source_id: ID of source node.
    :type source_id: str
    :return: List of pairs containing each target node ID and the 
        corresponding edge label. 
    :rtype: list of tuples of length two
    :raise RuntimeError: If the source node does not exist. 
)delim",
            py::arg("source_id")
        )
        .def("get_edge_label",
            &LabeledDigraph<T, double>::getEdgeLabel,
            R"delim(
    Get the label on the specified edge. 

    This method throws an exception if either node does not exist,
    and also if the specified edge does not exist.

    :param source_id: ID of source node.
    :type source_id: str
    :param target_id: ID of target node.
    :type target_id: str
    :return: Edge label.
    :rtype: float
    :raise RuntimeError: If either node or the edge does not exist.
)delim",
            py::arg("source_id"),
            py::arg("target_id")
        )
        .def("set_edge_label",
            &LabeledDigraph<T, double>::setEdgeLabel,
            R"delim(
    Set the label on the specified edge to the given value.

    This method throws an exception if either node does not exist, 
    and also if the specified edge does not exist.

    :param source_id: ID of source node.
    :type source_id: str
    :param target_id: ID of target node.
    :type target_id: str
    :param value: New edge label.
    :type value: float
    :raise RuntimeError: If either node or the edge does not exist.
)delim",
            py::arg("source_id"),
            py::arg("target_id"),
            py::arg("value")
        )
        .def("clear",
            &LabeledDigraph<T, double>::clear,
            R"delim(
    Clear the graph's contents.
)delim"
        )
        .def("get_laplacian",
            &LabeledDigraph<T, double>::getLaplacian,
            R"delim(
    Return the Laplacian matrix, with the nodes ordered according 
    to the graph's canonical ordering of nodes.

    :return: Laplacian matrix of the graph (as a dense matrix).
    :rtype: numpy.ndarray
)delim"
        )
        .def("get_spanning_forest_matrix",
            static_cast<Eigen::MatrixXd (LabeledDigraph<T, double>::*)(const int)>(
                &LabeledDigraph<T, double>::getSpanningForestMatrix
            ),
            R"delim(
    Compute the k-th spanning forest matrix, using the recurrence 
    of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs.\ 17-18), with
    a *dense* Laplacian matrix.

    :param k: Index of the desired spanning forest matrix.
    :type k: int
    :return: k-th spanning forest matrix.
    :rtype: numpy.ndarray
)delim",
            py::arg("k")
        )
        .def("get_steady_state_from_svd",
            &LabeledDigraph<T, double>::template getSteadyStateFromSVD<T, double>,
            R"delim(
    Compute a vector in the kernel of the Laplacian matrix of the
    graph, normalized by its 1-norm, by singular value decomposition.

    This vector coincides with the vector of steady-state probabilities
    of the nodes in the Markov process associated with the graph. 

    This method *assumes* that this graph is strongly connected, in
    which case the Laplacian matrix has a one-dimensional kernel and
    so the returned vector serves as a basis for this kernel. 

    :return: Vector in the kernel of the graph's Laplacian matrix,
        normalized by its 1-norm.
    :rtype: numpy.ndarray 
)delim"
        )
        .def("get_steady_state_from_recurrence",
            &LabeledDigraph<T, double>::template getSteadyStateFromRecurrence<double>,
            R"delim(
    Compute a vector in the kernel of the Laplacian matrix of the
    graph, normalized by its 1-norm, by the recurrence of Chebotarev
    and Agaev (Lin Alg Appl, 2002, Eqs.\ 17-18).

    This vector coincides with the vector of steady-state probabilities
    of the nodes in the Markov process associated with the graph. 

    This method *assumes* that this graph is strongly connected, in
    which case the Laplacian matrix has a one-dimensional kernel and
    so the returned vector serves as a basis for this kernel. 

    :param sparse: If True, use a sparse Laplacian matrix in the
        calculations.
    :type sparse: bool
    :return: Vector in the kernel of the graph's Laplacian matrix,
        normalized by its 1-norm.
    :rtype: numpy.ndarray
)delim",
            py::arg("sparse")
        )
        .def("get_fpt_moments_from_solver",
            &LabeledDigraph<T, double>::template getFPTMomentsFromSolver<T, double>,
            R"delim(
    Compute the vector of `r`-th moments for the *unconditional* first-passage
    times (FPTs) in the Markov process associated with the graph from each
    node to the target node, using the given linear solver method. 

    This method assumes that the associated Markov process eventually reaches
    the target node from every other node in the graph with certainty, meaning
    that there are, e.g., no alternative terminal nodes to which the process
    can travel and get "stuck".

    :param target_id: ID of target node.
    :type target_id: str
    :param r: Index of the desired moment.
    :type r: int
    :param method: Linear solver method for computing the mean first-
        passage time vector.
    :type method: SolverMethod
    :return: Vector of `r`-th moments for the FPT to the target node from
        every node in the graph.
    :rtype: numpy.ndarray 
    :raise ValueError: If solver method is not recognized.
    :raise RuntimeError: If target node does not exist.
)delim",
            py::arg("target_id"),
            py::arg("r"),
            py::arg("method") = SolverMethod::QRDecomposition
        )
        .def("get_mean_fpts_from_solver",
            &LabeledDigraph<T, double>::template getMeanFPTsFromSolver<T, double>,
            R"delim(
    Compute the vector of *unconditional* mean FPTs in the Markov process
    associated with the graph from each node to the target node, using the
    given linear solver method. 

    This method assumes that the associated Markov process eventually reaches
    the target node from every other node in the graph with certainty, meaning
    that there are, e.g., no alternative terminal nodes to which the process
    can travel and get "stuck".

    :param target_id: ID of target node.
    :type target_id: str
    :param method: Linear solver method for computing the mean first-
        passage time vector.
    :type method: SolverMethod
    :return: Vector of mean FPTs to the target node from every node in the 
        graph.
    :rtype: numpy.ndarray 
    :raise ValueError: If solver method is not recognized.
    :raise RuntimeError: If target node does not exist.
)delim",
            py::arg("target_id"),
            py::arg("method") = SolverMethod::QRDecomposition
        )
        .def("get_fpt_variances_from_solver",
            &LabeledDigraph<T, double>::template getFPTVariancesFromSolver<T, double>,
            R"delim(
    Compute the vector of variances of the *unconditional* FPTs in the
    Markov process associated with the graph from each node to the target
    node, using the given linear solver method. 

    This method assumes that the associated Markov process eventually reaches
    the target node from every other node in the graph with certainty, meaning
    that there are, e.g., no alternative terminal nodes to which the process
    can travel and get "stuck".

    :param target_id: ID of target node.
    :type target_id: str
    :param method: Linear solver method for computing the mean first-
        passage time vector.
    :type method: SolverMethod
    :return: Vector of mean FPTs to the target node from every node in the 
        graph.
    :rtype: numpy.ndarray 
    :raise ValueError: If solver method is not recognized.
    :raise RuntimeError: If target node does not exist.
)delim",
            py::arg("target_id"),
            py::arg("method") = SolverMethod::QRDecomposition
        )
        .def("get_fpt_moments_from_recurrence",
            &LabeledDigraph<T, double>::template getFPTMomentsFromRecurrence<double>,
            R"delim(
    Compute the vector of `r`-th moments for the *unconditional* first-passage
    times (FPTs) in the Markov process associated with the graph from each
    node to the target node, using the recurrence of Chebotarev and Agaev
    (Lin Alg Appl, 2002, Eqs.\ 17-18).

    This method assumes that the associated Markov process eventually reaches
    the target node from every other node in the graph with certainty, meaning
    that there are, e.g., no alternative terminal nodes to which the process
    can travel and get "stuck".

    :param target_id: ID of target node.
    :type target_id: str
    :param r: Index of the desired moment.
    :type r: int
    :return: Vector of `r`-th moments for the FPT to the target node from
        every node in the graph.
    :rtype: numpy.ndarray 
    :raise ValueError: If solver method is not recognized.
    :raise RuntimeError: If target node does not exist.
)delim",
            py::arg("target_id"),
            py::arg("r")
        )
        .def("get_mean_fpts_from_recurrence",
            &LabeledDigraph<T, double>::template getMeanFPTsFromRecurrence<double>,
            R"delim(
    Compute the vector of *unconditional* mean FPTs in the Markov process 
    associated with the graph from each node to the target node, using the
    recurrence of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs.\ 17-18).

    This method assumes that the associated Markov process eventually reaches
    the target node from every other node in the graph with certainty, meaning
    that there are, e.g., no alternative terminal nodes to which the process
    can travel and get "stuck".

    :param target_id: ID of target node.
    :type target_id: str
    :return: Vector of mean FPTs to the target node from every node in the
        graph.
    :rtype: numpy.ndarray 
    :raise RuntimeError: If target node does not exist.
)delim",
            py::arg("target_id")
        )
        .def("get_fpt_variances_from_recurrence",
            &LabeledDigraph<T, double>::template getFPTVariancesFromRecurrence<double>,
            R"delim(
    Compute the vector of variances of the *unconditional* FPTs in the
    Markov process associated with the graph from each node to the target
    node, using the recurrence of Chebotarev and Agaev (Lin Alg Appl, 2002,
    Eqs.\ 17-18).

    This method assumes that the associated Markov process eventually reaches
    the target node from every other node in the graph with certainty, meaning
    that there are, e.g., no alternative terminal nodes to which the process
    can travel and get "stuck".

    :param target_id: ID of target node.
    :type target_id: str
    :return: Vector of variances of the FPTs to the target node from every 
        node in the graph.
    :rtype: numpy.ndarray 
    :raise RuntimeError: If target node does not exist.
)delim",
            py::arg("target_id")
        )
        .def("simulate",
            &LabeledDigraph<T, double>::simulate,
            R"delim(
    Simulate the Markov process associated with the graph up to a given 
    maximum time.

    :param init_node_id: ID of initial node.
    :type init_node_id: str
    :param max_time: Time limit.
    :type max_time: float
    :param seed: Seed for random number generator.
    :type seed: int
    :return: List of visited nodes and their lifetimes, together with the 
        final node reached throughout the simulation.
    :rtype: list
    :raise ValueError: If initial node does not exist or time limit is not 
        positive.
)delim",
            py::arg("init_node_id"),
            py::arg("max_time"),
            py::arg("seed")
        );
}

/**
 * A templated function for exposing versions of the `LineGraph` class 
 * with different numeric types to Python. 
 *
 * @param m pybind11 module. 
 * @param class_name Name to be given to the (Python) class.
 */
template <typename T>
void declareLineGraphClass(py::module& m, const std::string& class_name)
{
    py::class_<LineGraph<T, double>, LabeledDigraph<T, double> >(m, class_name.c_str())
        .def(py::init<>(),
            R"delim(
    Constructor for a line graph of length 0, i.e., a single vertex
    named "0". 
)delim"
        )
        .def(py::init<int>(),
            R"delim(
    Constructor for a line graph of given length, with edge labels 
    set to unity.

    :param N: Length of desired line graph; `self.N` is set to `N`, 
        and `self.numnodes` to `N + 1`.
    :type N: int 
)delim",
            py::arg("N")
        )
        .def("add_node",
            &LineGraph<T, double>::addNode,
            R"delim(
    Ban node addition via `addNode()`: nodes can be added or removed 
    only at the upper end of the graph. 

    :param id: ID of node to be added (to match signature with parent
        method). 
    :type id: str
    :raise RuntimeError: If invoked at all.
)delim",
            py::arg("id")
        )
        .def("remove_node",
            &LineGraph<T, double>::removeNode,
            R"delim(
    Ban node removal via `removeNode()`: nodes can be added or removed 
    only at the upper end of the graph.

    :param id: ID of node to be removed (to match signature with parent
        method).
    :type id: str
    :raise RuntimeError: If invoked at all.
)delim",
            py::arg("id")
        )
        .def("add_edge",
            &LineGraph<T, double>::addEdge,
            R"delim(
    Ban edge addition via `addEdge()`: edges can be added or removed 
    only at the upper end of the graph.

    :param source_id: ID of source node of new edge (to match signature
        with parent method).
    :type source_id: str
    :param target_id: ID of target node of new edge (to match signature
        with parent method).
    :type target_id: str
    :param label: Label on new edge (to match signature with parent
        method).
    :type label: float
    :raise RuntimeError: If invoked at all. 
)delim",
            py::arg("source_id"),
            py::arg("target_id"),
            py::arg("label") = 1
        )
        .def("remove_edge",
            &LineGraph<T, double>::removeEdge,
            R"delim(
    Ban edge removal via `removeEdge()`: edges can be added or removed 
    only at the upper end of the graph.

    :param source_id: ID of source node of edge to be removed (to match
        signature with parent method).
    :type source_id: str
    :param target_id: ID of target node of edge to be removed (to match
        signature with parent method).
    :type target_id: str
    :raise RuntimeError: If invoked at all.
)delim",
            py::arg("source_id"),
            py::arg("target_id")
        )
        .def("add_node_to_end",
            &LineGraph<T, double>::addNodeToEnd,
            R"delim(
    Add a new node to the end of the graph (along with the two edges), 
    increasing its length by one.

    :param labels: A pair of edge labels, forward edge then reverse edge.
    :type labels: tuple of two floats
)delim",
            py::arg("labels")
        )
        .def("remove_node_from_end",
            &LineGraph<T, double>::removeNodeFromEnd,
            R"delim(
    Remove the last node (N) from the graph (along with the two edges),
    decreasing its length by one.

    :raise RuntimeError: If `self.N` is zero.
)delim"
        )
        .def("set_edge_labels",
            &LineGraph<T, double>::setEdgeLabels,
            R"delim(
    Set the edge labels between the i-th and (i+1)-th nodes (`i -> i+1`
    then `i+1 -> i`) to the given values.

    :param i: Index of edge labels to update.
    :type i: int
    :param labels: A pair of edge labels, `i -> i+1` then `i+1 -> i`.
    :type labels: tuple of two floats
)delim",
            py::arg("i"),
            py::arg("labels")
        )
        .def("clear",
            &LineGraph<T, double>::clear,
            R"delim(
    Ban clearing via `clear()`: the graph must be non-empty.
    
    :raise RuntimeError: If invoked at all.
)delim"
        )
        .def("reset",
            &LineGraph<T, double>::reset,
            R"delim(Remove all nodes and edges but 0.)delim"
        )
        .def("get_upper_exit_prob",
            &LineGraph<T, double>::getUpperExitProb,
            R"delim(
    Compute the probability of exiting the line graph through the upper
    node, `self.N` (to an auxiliary "upper exit" node), rather than 
    through the lower node, `0` (to an auxiliary "lower exit" node),
    starting from `0`.

    :param lower_exit_rate: Rate of exit through the lower node (`0`).
    :type lower_exit_rate: float
    :param upper_exit_rate: Rate of exit through the upper node (`self.N`).
    :type upper_exit_rate: float
    :return: Probability of exit from `0` through `self.N`.
    :rtype: float
)delim",
            py::arg("lower_exit_rate"),
            py::arg("upper_exit_rate")
        )
        .def("get_lower_end_to_end_time",
            &LineGraph<T, double>::getLowerEndToEndTime,
            R"delim(
    Compute the mean first-passage time from `self.N` to `0`.

    :return: Mean first-passage time from `self.N` to `0`.
    :rtype: float
)delim"
        )
        .def("get_upper_end_to_end_time",
            &LineGraph<T, double>::getUpperEndToEndTime,
            R"delim(
    Compute the mean first-passage time from `0` to `self.N`.

    :return: Mean first-passage time from `0` to `self.N`.
    :rtype: float
)delim"
        )
        .def("get_lower_exit_rate",
            py::overload_cast<double>(&LineGraph<T, double>::getLowerExitRate),
            R"delim(
    Compute the reciprocal of the *unconditional* mean first-passage
    time to exit from the line graph through the lower node, `0`
    (to an auxiliary "lower exit" node), starting from `0`, given that
    exit through the upper node, `self.N`, is impossible. 

    :param lower_exit_rate: Rate of exit through the lower node (`0`).
    :type lower_exit_rate: float
    :return: Reciprocal of mean first-passage time from `0` to exit 
        through `0`. 
    :rtype: float
)delim",
            py::arg("lower_exit_rate")
        )
        .def("get_lower_exit_rate",
            py::overload_cast<double, double>(&LineGraph<T, double>::getLowerExitRate),
            R"delim(
    Compute the reciprocal of the *conditional* mean first-passage
    time to exit from the line graph through the lower node, `0`
    (to an auxiliary "lower exit" node), starting from `0`, given that
    exit through the lower node indeed occurs.

    :param lower_exit_rate: Rate of exit through the lower node (`0`).
    :type lower_exit_rate: float
    :param upper_exit_rate: Rate of exit through the upper node (`self.N`).
    :type upper_exit_rate: float
    :return: Reciprocal of conditional mean first-passage time from `0`
        to exit through `0`. 
    :rtype: float
)delim",
            py::arg("lower_exit_rate"),
            py::arg("upper_exit_rate")
        )
        .def("get_lower_exit_rate_from_zero",
            &LineGraph<T, double>::getLowerExitRateFromZero,
            R"delim(
    Compute the reciprocal of the *unconditional* mean first-passage 
    time to exit from the line graph through the lower node, `0` (to
    an auxiliary "lower exit" node), starting from `0`, given that
    exit through the upper node, `self.N`, is impossible.

    :param lower_exit_rate: Rate of exit through the lower node (`0`).
    :type lower_exit_rate: float
    :return: Reciprocal of unconditional mean first-passage time from `0`
        to exit through `0`. 
    :rtype: float
)delim",
            py::arg("lower_exit_rate")
        )
        .def("get_entry_to_upper_exit_time",
            &LineGraph<T, double>::getEntryToUpperExitTime,
            R"delim(
    Compute the *unconditional* mean first-passage time to exit from the 
    line graph through the upper node, `self.N` (to an auxiliary "upper
    exit" node), starting from an auxiliary "lower entry" node into `0`.

    :param lower_entry_rate: Rate of entry through the lower node (`0`).
    :type lower_entry_rate: float
    :param lower_exit_rate: Rate of exit through the lower node (`0`).
    :type lower_exit_rate: float
    :param upper_exit_rate: Rate of exit through the upper node (`self.N`).
    :type upper_exit_rate: float
    :return: Unconditional mean first-passage time from the entry node to
        exit through `self.N`.
    :rtype: float
)delim",
            py::arg("lower_entry_rate"),
            py::arg("lower_exit_rate"),
            py::arg("upper_exit_rate")
        )
        .def("get_upper_exit_rate",
            &LineGraph<T, double>::getUpperExitRate,
            R"delim(
    Compute the reciprocal of the *conditional* mean first-passage 
    time to exit from the line graph through the upper node, `self.N`
    (to an auxiliary "upper exit" node), starting from `0`, given that
    exit through the upper node indeed occurs.

    :param lower_exit_rate: Rate of exit through the lower node (`0`).
    :type lower_exit_rate: float
    :param upper_exit_rate: Rate of exit through the upper node (`self.N`).
    :type upper_exit_rate: float
    :return: Reciprocal of conditional mean first-passage time from `0`
        to exit through `self.N`.
    :rtype: float
)delim",
            py::arg("lower_exit_rate"),
            py::arg("upper_exit_rate")
        );
}

/**
 * A templated function for exposing versions of the `GridGraph` class 
 * with different numeric types to Python. 
 *
 * @param m pybind11 module. 
 * @param class_name Name to be given to the (Python) class.
 */
template <typename T>
void declareGridGraphClass(py::module& m, const std::string& class_name)
{
    py::class_<GridGraph<T, double>, LabeledDigraph<T, double> >(m, class_name.c_str())
        .def(py::init<>(),
            R"delim(
    Constructor for a grid graph of length 0, i.e., the two nodes 
    `A0` and `B0` and the edges between them, with edge labels 
    set to unity.
)delim"
        )
        .def(py::init<int>(),
            R"delim(
    Constructor for a grid graph of given length, with edge labels 
    set to unity.

    :param N: Length of desired grid graph; `self.N` is set to `N`, 
        and `self.numnodes` to `2 * N + 2`.
    :type N: int 
)delim",
            py::arg("N")
        )
        .def("set_zeroth_labels",
            &GridGraph<T, double>::setZerothLabels,
            R"delim(
    Set the labels on the edges `A0 -> B0` and `B0 -> A0` to the 
    given values.

    :param A0_to_B0: Label on `A0 -> B0`.
    :type A0_to_B0: float
    :param B0_to_A0: Label on `B0 -> A0`.
    :type B0_to_A0: float
)delim",
            py::arg("A0_to_B0"),
            py::arg("B0_to_A0")
        )
        .def("add_rung_to_end",
            &GridGraph<T, double>::addRungToEnd,
            R"delim(
    Add a new pair of nodes to the end of the graph (along with the 
    six new edges), increasing its length by one.

    :param labels: Labels on the six new edges: `A{N} -> A{N+1}`, 
        `A{N+1} -> A{N}`, `B{N} -> B{N+1}`, `B{N+1} -> B{N}`, 
        `A{N+1} -> B{N+1}`, and `B{N+1} -> A{N+1}`.
    :type labels: list
)delim",
            py::arg("labels")
        )
        .def("remove_rung_from_end",
            &GridGraph<T, double>::removeRungFromEnd,
            R"delim(
    Remove the last pair of nodes from the graph (along with the six 
    associated edges), decreasing its length by one.

    :raise RuntimeError: If `self.N` is zero.
)delim"
        )
        .def("add_node",
            &GridGraph<T, double>::addNode,
            R"delim(
    Ban node addition via `add_node()`: nodes can be added or removed 
    only at the upper end of the graph.

    :param id: ID of node to be added (to match signature with parent
        method).
    :type id: str
    :raise RuntimeError: If invoked at all.
)delim",
            py::arg("id")
        )
        .def("remove_node",
            &GridGraph<T, double>::removeNode,
            R"delim(
    Ban node removal via `remove_node()`: nodes can be added or removed
    only at the upper end of the graph.

    :param id: ID of node to be removed (to match signature with parent
        method).
    :type id: str
    :raise RuntimeError: If invoked at all.
)delim",
            py::arg("id")
        )
        .def("add_edge",
            &GridGraph<T, double>::addEdge,
            R"delim(
    Ban edge addition via `add_edge()`: edges can be added or removed
    only at the upper end of the graph.

    :param source_id: ID of source node of new edge (to match signature
        with parent method).
    :type source_id: str
    :param target_id: ID of target node of new edge (to match signature
        with parent method).
    :type target_id: str
    :param label: Label on new edge (to match signature with parent
        method).
    :type label: float
    :raise RuntimeError: If invoked at all.
)delim",
            py::arg("source_id"),
            py::arg("target_id"),
            py::arg("label") = 1
        )
        .def("remove_edge",
            &GridGraph<T, double>::removeEdge,
            R"delim(
    Ban edge removal via `removeEdge()`: edges can be added or removed 
    only at the upper end of the graph.

    :param source_id: ID of source node of edge to be removed (to match
        signature with parent method).
    :type source_id: str
    :param target_id: ID of target node of edge to be removed (to match
        signature with parent method).
    :type target_id: str
    :raise RuntimeError: If invoked at all.
)delim",
            py::arg("source_id"),
            py::arg("target_id")
        )
        .def("clear",
            &GridGraph<T, double>::clear,
            R"delim(
    Ban clearing via `clear()`: the graph must be non-empty.

    :raise RuntimeError: If invoked at all.
)delim"
        )
        .def("reset",
            &GridGraph<T, double>::reset,
            R"delim(
    Remove all nodes and edges but `A0` and `B0` and the edges in
    between.
)delim"
        )
        .def("set_rung_labels",
            &GridGraph<T, double>::setRungLabels,
            R"delim(
    Set the labels on the i-th sextet of edges (`A{i} -> A{i+1}`, 
    `A{i+1} -> A{i}`, `B{i} -> B{i+1}`, `B{i+1} -> B{i}`,
    `A{i+1} -> B{i+1}`, and `B{i+1} -> A{i+1}`) to the given values.

    This method throws an exception if `i` is not a valid index in 
    the graph (i.e., if `i >= self.N`).

    :param i: Index of edge labels to update (see below).
    :type i: int 
    :param labels: Sextet of new edge label values: `A{i} -> A{i+1}`, 
        `A{i+1} -> A{i}`, `B{i} -> B{i+1}`, `B{i+1} -> B{i}`,
        `A{i+1} -> B{i+1}`, and `B{i+1} -> A{i+1}`.
    :type labels: list of six floats
    :raise ValueError: If `i >= self.N`.
)delim",
            py::arg("i"),
            py::arg("labels")
        )
        .def("get_exit_stats",
            &GridGraph<T, double>::getExitStats, 
            R"delim(
    Compute and return two quantities: the *splitting probability* of exiting
    the graph through `B{self.N}` (reaching an auxiliary upper exit node),
    and not through `A0` (reaching an auxiliary lower exit node); and the
    reciprocal of the *unconditional mean first-passage time* to exiting the
    graph through `A0`, given that the exit rate from `B{self.N}` is zero.

    :param lower_exit_rate: Rate of lower exit from `A0`.
    :type lower_exit_rate: float
    :param upper_exit_rate: Rate of upper exit from `B{self.N}`.
    :type upper_exit_rate: float
    :return: The above two quantities.
    :rtype: tuple of three floats
)delim",
            py::arg("lower_exit_rate"),
            py::arg("upper_exit_rate")
        );
}

/** ----------------------------------------------------------------- // 
 *                     PYGRAPH MODULE DECLARATIONS                    //
 *  ----------------------------------------------------------------- */
PYBIND11_MODULE(pygraph, m)
{
    m.doc() = R"delim(
    .. currentmodule:: pygraph

    .. autosummary::
       :toctree: _generate

       PreciseDigraph
       PreciseLineGraph
       PreciseGridGraph
       viz_digraph
)delim"; 

    py::enum_<SolverMethod>(m, "SolverMethod")
        .value("QRDecomposition", SolverMethod::QRDecomposition)
        .value("LUDecomposition", SolverMethod::LUDecomposition);  

    py::class_<Node>(m, "Node")
        .def(py::init<std::string>())
        .def("get_id", &Node::getId)
        .def("set_id", &Node::setId);

    // Expose the desired LabeledDigraph classes ... 
    declareLabeledDigraphClass<PreciseType>(m, "PreciseDigraph");
    declareLineGraphClass<PreciseType>(m, "PreciseLineGraph");
    declareGridGraphClass<PreciseType>(m, "PreciseGridGraph");

    // Expose a version of vizLabeledDigraph<PreciseType, double>() that 
    // can be called from Python
    m.def("viz_digraph",
        [](LabeledDigraph<PreciseType, double>& graph,
           std::string layout,
           std::string format,
           std::string filename)
        {
            // Establish a graphviz context, visualize the graph, then 
            // free the context 
            GVC_t* context = gvContext();
            vizLabeledDigraph<PreciseType, double>(graph, layout, format, filename, context); 
            gvFreeContext(context); 
        },
        R"delim(
    Produce a new file with a visualization of the given `PreciseDigraph`
    instance with the graphviz library.

    :param graph: Input graph. 
    :type graph: PreciseDigraph
    :param layout: Graphviz layout algorithm: `"dot"`, `"neato"`, `"fdp"`,
        `"sfdp"`, `"twopi"`, or `"circo"`. 
    :type layout: str
    :param format: Output file format: `"png"`, `"pdf"`, etc.
    :type layout: str
    :param filename: Output file name. 
    :type layout: str
)delim",
        py::arg("graph"),
        py::arg("layout"),  
        py::arg("format"), 
        py::arg("filename")
    );
}
