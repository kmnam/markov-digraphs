#include <boost/multiprecision/mpfr.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "../include/digraph.hpp"
#include "../include/graphs/line.hpp"

/**
 * Python bindings for the `LabeledDigraph<...>` class through pybind11. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/8/2021
 */

namespace py = pybind11;

// MANTISSA_PRECISION is a parameter to be defined at compile-time 
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<MANTISSA_PRECISION> > PreciseType;

/**
 * Python bindings for `LabeledDigraph<PreciseType, double>` and all of its 
 * subclasses.
 */
PYBIND11_MODULE(pygraph, m)
{
    m.doc() = R"delim(
    PyGraph: Labeled directed graphs for modeling Markov processes
    --------------------------------------------------------------

    .. currentmodule:: pygraph
)delim"; 

    py::enum_<SummationMethod>(m, "SummationMethod")
        .value("NaiveSummation", SummationMethod::NaiveSummation)
        .value("KahanSummation", SummationMethod::KahanSummation)
        .value("KBNSummation",   SummationMethod::KBNSummation);

    py::enum_<SolverMethod>(m, "SolverMethod")
        .value("QRDecomposition", SolverMethod::QRDecomposition)
        .value("LUDecomposition", SolverMethod::LUDecomposition);  

    py::class_<Node>(m, "Node")
        .def(py::init<std::string>())
        .def("get_id", &Node::getId)
        .def("set_id", &Node::setId);

    /**
     * Expose `LabeledDigraph<PreciseType, double>` as `pygraph.PreciseDigraph`. 
     */
    py::class_<LabeledDigraph<PreciseType, double> >(m, "PreciseDigraph")
        .def(py::init<>(),
            R"delim(Empty constructor.)delim"
        )
        .def("get_num_nodes",
            &LabeledDigraph<PreciseType, double>::getNumNodes,
            R"delim(
    Return the number of nodes in the graph.
    
    Returns
    -------
    int 
        Number of nodes in the graph.
)delim")
        .def("add_node",
            &LabeledDigraph<PreciseType, double>::addNode,
            R"delim(
    Add a node to the graph with the given ID.

    Parameters
    ----------
    id : str
        ID for new node. 

    Exceptions
    ----------
    RuntimeError
        If node already exists with the given ID.
)delim")
        .def("remove_node",
            &LabeledDigraph<PreciseType, double>::removeNode,
            R"delim(
    Remove a node from the graph with the given ID.

    Parameters
    ----------
    id : str
        ID of node to be removed.

    Exceptions
    ----------
    RuntimeError
        If node with given ID does not exist.
)delim")
        .def("has_node",
            &LabeledDigraph<PreciseType, double>::hasNode,
            R"delim(
    Return True if node with given ID exists in the graph.

    Parameters
    ----------
    id : str
        ID of desired node.

    Returns
    -------
    bool
        True if node exists with the given ID, False otherwise.
)delim")
        .def("get_all_node_ids",
            &LabeledDigraph<PreciseType, double>::getAllNodeIds,
            R"delim(
    Return the list of IDs of all nodes in the graph, ordered
    according to the graph's canonical ordering of nodes.

    Returns
    -------
    list
        List of all node IDs.
)delim")
        .def("add_edge",
            &LabeledDigraph<PreciseType, double>::addEdge,
            R"delim(
    Add an edge between two nodes.

    If either ID does not correspond to a node in the graph, this 
    function instantiates these nodes. 

    Parameters
    ----------
    source_id : str
        ID of source node of new edge. 
    target_id : str
        ID of target node of new edge.
    label : float
        Label on new edge.

    Exceptions
    ----------
    RuntimeError
        If the edge already exists.
)delim",
            py::arg("source_id"),
            py::arg("target_id"),
            py::arg("label") = 1
        )
        .def("remove_edge",
            &LabeledDigraph<PreciseType, double>::removeEdge,
            R"delim(
    Remove the edge between the two given nodes.

    This method does nothing if the node does not exist but the
    nodes do, but throws an exception if either node does not 
    exist. 

    Parameters
    ----------
    source_id : str
        ID of source node of edge to be removed.
    target_id : str
        ID of target node of edge to be removed.

    Exceptions
    ----------
    RuntimeError
        If either node does not exist.
)delim"
        )
        .def("has_edge",
            static_cast<bool (LabeledDigraph<PreciseType, double>::*)(std::string, std::string) const>(
                &LabeledDigraph<PreciseType, double>::hasEdge
            ),
            R"delim(
    Return True if the specified edge exists, given the IDs of
    the two nodes, and False otherwise.

    This method also returns False if either node does not exist.

    Parameters
    ----------
    source_id : str
        ID of source node.
    target_id : str
        ID of target node.

    Returns
    -------
    bool
        True if the edge exists, False otherwise.
)delim"
        )
        .def("get_edge_label",
            &LabeledDigraph<PreciseType, double>::getEdgeLabel,
            R"delim(
    Get the label on the specified edge. 

    This method throws an exception if either node does not exist,
    and also if the specified edge does not exist.

    Parameters
    ----------
    source_id : str
        ID of source node.
    target_id : str
        ID of target node.

    Returns
    -------
    float
        Edge label.

    Exceptions
    ----------
    RuntimeError
        If either node or the edge does not exist.
)delim"
        )
        .def("set_edge_label",
            &LabeledDigraph<PreciseType, double>::setEdgeLabel,
            R"delim(
    Set the label on the specified edge to the given value.

    This method throws an exception if either node does not exist, 
    and also if the specified edge does not exist.

    Parameters
    ----------
    source_id : str
        ID of source node.
    target_id : str
        ID of target node.
    value : float
        New edge label.

    Exceptions
    ----------
    RuntimeError
        If either node or the edge does not exist.
)delim"
        )
        .def("clear",
            &LabeledDigraph<PreciseType, double>::clear,
            R"delim(
    Clear the graph's contents.
)delim"
        )
        .def("get_laplacian",
            &LabeledDigraph<PreciseType, double>::getLaplacian,
            R"delim(
    Return the Laplacian matrix, with the nodes ordered according 
    to the graph's canonical ordering of nodes.

    Parameters
    ----------
    method : `SummationMethod`
        Summation method.

    Returns
    -------
    `numpy.array`
        Laplacian matrix of the graph (as a dense matrix). 
)delim",
            py::arg("method") = SummationMethod::NaiveSummation
        )
        .def("get_spanning_forest_matrix",
            static_cast<Eigen::MatrixXd (LabeledDigraph<PreciseType, double>::*)(const int, const SummationMethod)>(
                &LabeledDigraph<PreciseType, double>::getSpanningForestMatrix
            ),
            R"delim(
    Compute the k-th spanning forest matrix, using the recurrence 
    of Chebotarev and Agaev (Lin Alg Appl, 2002, Eqs. 17-18), with
    a *dense* Laplacian matrix.

    Parameters
    ----------
    k : int
        Index of the desired spanning forest matrix.
    method : `SummationMethod`
        Summation method.

    Returns
    -------
    `numpy.array`
        k-th spanning forest matrix.

    Exceptions
    ----------
    ValueError
        If summation method is not recognized.
)delim",
            py::arg("k"),
            py::arg("method") = SummationMethod::NaiveSummation
        )
        .def("get_steady_state_from_svd",
            &LabeledDigraph<PreciseType, double>::getSteadyStateFromSVD,
            R"delim(
    Compute a vector in the kernel of the Laplacian matrix of the
    graph, normalized by its 1-norm, by singular value decomposition.

    This vector coincides with the vector of steady-state probabilities
    of the nodes in the Markov process associated with the graph. 

    This method *assumes* that this graph is strongly connected, in
    which case the Laplacian matrix has a one-dimensional kernel and
    so the returned vector serves as a basis for this kernel. 

    Returns
    -------
    `numpy.array`
        Vector in the kernel of the graph's Laplacian matrix, normalized
        by its 1-norm. 
)delim"
        )
        .def("get_steady_state_from_recurrence",
            &LabeledDigraph<PreciseType, double>::getSteadyStateFromRecurrence,
            R"delim(
    Compute a vector in the kernel of the Laplacian matrix of the
    graph, normalized by its 1-norm, by the recurrence of Chebotarev
    and Agaev (Lin Alg Appl, 2002, Eqs. 17-18).

    This vector coincides with the vector of steady-state probabilities
    of the nodes in the Markov process associated with the graph. 

    This method *assumes* that this graph is strongly connected, in
    which case the Laplacian matrix has a one-dimensional kernel and
    so the returned vector serves as a basis for this kernel. 

    Parameters
    ----------
    sparse : bool
        If True, use a sparse Laplacian matrix in the calculations.
    method : `SummationMethod`
        Summation method.

    Returns
    -------
    `numpy.array`
        Vector in the kernel of the graph's Laplacian matrix, normalized
        by its 1-norm.

    Exceptions
    ----------
    ValueError
        If summation method is not recognized. 
)delim",
            py::arg("sparse"),
            py::arg("method") = SummationMethod::NaiveSummation
        )
        .def("get_mean_first_passage_times_from_solver",
            &LabeledDigraph<PreciseType, double>::getMeanFirstPassageTimesFromSolver,
            R"delim(
    Compute the vector of *unconditional* mean first-passage times in 
    the Markov process associated with the graph from each node to the
    target node, using the given linear solver method. 

    This method assumes that the associated Markov process certainly 
    eventually reaches the target node from each node in the graph, 
    meaning that there are no alternative terminal nodes (or rather 
    SCCs) to which the process can travel and get "stuck."

    Parameters
    ----------
    target_id : str
        ID of target node.
    method : `SolverMethod`
        Linear solver method for computing the mean first-passage 
        time vector.

    Returns
    -------
    `numpy.array`
        Vector of mean first-passage times to the target node from
        every node in the graph. 

    Exceptions
    ----------
    ValueError
        If solver method is not recognized.
    RuntimeError
        If target node does not exist.
)delim",
            py::arg("target_id"),
            py::arg("method") = SolverMethod::QRDecomposition
        )
        .def("get_mean_first_passage_times_from_recurrence",
            &LabeledDigraph<PreciseType, double>::getMeanFirstPassageTimesFromRecurrence,
            R"delim(
    Compute the vector of *unconditional* mean first-passage times in 
    the Markov process associated with the graph from each node to the
    target node, using the recurrence of Chebotarev and Agaev (Lin Alg
    Appl, 2002, Eqs. 17-18).

    This method assumes that the associated Markov process certainly 
    eventually reaches the target node from each node in the graph, 
    meaning that there are no alternative terminal nodes (or rather 
    SCCs) to which the process can travel and get "stuck."

    Parameters
    ----------
    target_id : str
        ID of target node.
    sparse : bool
        If True, use a sparse Laplacian matrix in the calculations.
    method : `SummationMethod`
        Summation method. 

    Returns
    -------
    `numpy.array`
        Vector of mean first-passage times to the target node from
        every node in the graph. 

    Exceptions
    ----------
    ValueError
        If summation method is not recognized.
    RuntimeError
        If target node does not exist.
)delim",
            py::arg("target_id"),
            py::arg("sparse"), 
            py::arg("method") = SummationMethod::NaiveSummation
        )
        .def("get_second_moments_of_first_passage_times_from_solver",
            &LabeledDigraph<PreciseType, double>::getSecondMomentsOfFirstPassageTimesFromSolver,
            R"delim(
    Compute the vector of second moments of the *unconditional* first-
    passage times in the Markov process associated with the graph from
    each node to the target node, using the given linear solver method.

    This method assumes that the associated Markov process certainly
    eventually reaches the target node from each node in the graph,
    meaning that there are no alternative terminal nodes (or rather
    SCCs) to which the process can travel and get "stuck".

    Parameters
    ----------
    target_id : str
        ID of target node.
    method : `SolverMethod`
        Linear solver method for computing the second-moment time
        vector.

    Returns
    -------
    `numpy.array`
        Vector of first-passage time second moments to the target
        node from every node in the graph. 

    Exceptions
    ----------
    ValueError
        If solver method is not recognized.
    RuntimeError
        If target node does not exist.
)delim", 
            py::arg("target_id"),
            py::arg("method") = SolverMethod::QRDecomposition
        )
        .def("get_second_moments_of_first_passage_times_from_recurrence",
            &LabeledDigraph<PreciseType, double>::getSecondMomentsOfFirstPassageTimesFromRecurrence,
            R"delim(
    Compute the vector of second moments of the *unconditional* first-
    passage times in the Markov process associated with the graph from
    each node to the target node, using the recurrence of Chebotarev
    and Agaev (Lin Alg Appl, 2002, Eqs. 17-18).

    This method assumes that the associated Markov process certainly
    eventually reaches the target node from each node in the graph,
    meaning that there are no alternative terminal nodes (or rather
    SCCs) to which the process can travel and get "stuck".

    Parameters
    ----------
    target_id : str
        ID of target node.
    sparse : bool
        If True, use a sparse Laplacian matrix in the calculations.
    method : `SummationMethod`
        Summation method. 

    Returns
    -------
    `numpy.array`
        Vector of first-passage time second moments to the target
        node from every node in the graph. 

    Exceptions
    ----------
    ValueError
        If summation method is not recognized.
    RuntimeError
        If target node does not exist.
)delim", 
            py::arg("target_id"),
            py::arg("sparse"), 
            py::arg("method") = SummationMethod::NaiveSummation 
        );

    /**
     * Expose `LineGraph<PreciseType, double>` as `pygraph.PreciseLineGraph`. 
     */
    py::class_<LineGraph<PreciseType, double>, LabeledDigraph<PreciseType, double> >(m, "PreciseLineGraph")
        .def(py::init<>())
        .def(py::init<int>())
        .def("add_node",             &LineGraph<PreciseType, double>::addNode)
        .def("remove_node",          &LineGraph<PreciseType, double>::removeNode)
        .def("add_edge",             &LineGraph<PreciseType, double>::addEdge,
            py::arg("source_id"),
            py::arg("target_id"),
            py::arg("label") = 1
        )
        .def("remove_edge",          &LineGraph<PreciseType, double>::removeEdge)
        .def("add_node_to_end",      &LineGraph<PreciseType, double>::addNodeToEnd)
        .def("remove_node_from_end", &LineGraph<PreciseType, double>::removeNodeFromEnd)
        .def("get_edge_label",       &LabeledDigraph<PreciseType, double>::getEdgeLabel)
        .def("set_edge_label",       &LabeledDigraph<PreciseType, double>::setEdgeLabel)
        .def("set_edge_labels",      &LineGraph<PreciseType, double>::setEdgeLabels)
        .def("clear",                &LineGraph<PreciseType, double>::clear)
        .def("reset",                &LineGraph<PreciseType, double>::reset)
        .def("get_upper_exit_prob",  &LineGraph<PreciseType, double>::getUpperExitProb)
        .def("get_lower_exit_rate",  &LineGraph<PreciseType, double>::getLowerExitRate)
        .def("get_upper_exit_rate",
            static_cast<double (LineGraph<PreciseType, double>::*)(double)>(
                &LineGraph<PreciseType, double>::getUpperExitRate
            )
        )
        .def("get_upper_exit_rate",
            static_cast<double (LineGraph<PreciseType, double>::*)(double, double)>(
                &LineGraph<PreciseType, double>::getUpperExitRate
            )
        );
}
