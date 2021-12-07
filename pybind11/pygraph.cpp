#include <boost/multiprecision/mpfr.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "../include/digraph.hpp"

/**
 * Python bindings for the `LabeledDigraph<T>` class through pybind11. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/7/2021
 */

using namespace Eigen; 
namespace py = pybind11;

// MANTISSA_PRECISION is a parameter to be defined at compile-time 
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<MANTISSA_PRECISION> > PreciseType;

/**
 * Python bindings for `LabeledDigraph<PreciseType, double>`.
 */
PYBIND11_MODULE(pygraph, m)
{
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
        .def(py::init<>())
        .def("get_num_nodes",    &LabeledDigraph<PreciseType, double>::getNumNodes)
        .def("add_node",         &LabeledDigraph<PreciseType, double>::addNode)
        .def("remove_node",      &LabeledDigraph<PreciseType, double>::removeNode)
        .def("has_node",         &LabeledDigraph<PreciseType, double>::hasNode)
        .def("get_all_node_ids", &LabeledDigraph<PreciseType, double>::getAllNodeIds)
        .def("add_edge",
            &LabeledDigraph<PreciseType, double>::addEdge,
            py::arg("source_id"),
            py::arg("target_id"),
            py::arg("label") = 1
        )
        .def("remove_edge",      &LabeledDigraph<PreciseType, double>::removeEdge)
        .def("has_edge",
            static_cast<bool (LabeledDigraph<PreciseType, double>::*)(std::string, std::string) const>(
                &LabeledDigraph<PreciseType, double>::hasEdge
            )
        )
        .def("get_edge_label",   &LabeledDigraph<PreciseType, double>::getEdgeLabel)
        .def("set_edge_label",   &LabeledDigraph<PreciseType, double>::setEdgeLabel)
        .def("clear",            &LabeledDigraph<PreciseType, double>::clear)
        .def("get_laplacian",
            &LabeledDigraph<PreciseType, double>::getLaplacian,
            py::arg("method") = SummationMethod::NaiveSummation
        )
        .def("get_spanning_forest_matrix",
            &LabeledDigraph<PreciseType, double>::getSpanningForestMatrix,
            py::arg("k"),
            py::arg("method") = SummationMethod::NaiveSummation
        )
        .def("get_steady_state_from_svd",
            &LabeledDigraph<PreciseType, double>::getSteadyStateFromSVD
        )
        .def("get_steady_state_from_recurrence",
            &LabeledDigraph<PreciseType, double>::getSteadyStateFromRecurrence,
            py::arg("sparse"),
            py::arg("method") = SummationMethod::NaiveSummation
        )
        .def("get_mean_first_passage_times_from_solver",
            &LabeledDigraph<PreciseType, double>::getMeanFirstPassageTimesFromSolver,
            py::arg("target_id"),
            py::arg("method") = SolverMethod::QRDecomposition
        )
        .def("get_mean_first_passage_times_from_recurrence",
            &LabeledDigraph<PreciseType, double>::getMeanFirstPassageTimesFromRecurrence,
            py::arg("target_id"),
            py::arg("sparse"), 
            py::arg("method") = SummationMethod::NaiveSummation
        )
        .def("get_second_moments_of_first_passage_times_from_solver",
            &LabeledDigraph<PreciseType, double>::getSecondMomentsOfFirstPassageTimesFromSolver,
            py::arg("target_id"),
            py::arg("method") = SolverMethod::QRDecomposition
        )
        .def("get_second_moments_of_first_passage_times_from_recurrence",
            &LabeledDigraph<PreciseType, double>::getSecondMomentsOfFirstPassageTimesFromRecurrence,
            py::arg("target_id"),
            py::arg("sparse"), 
            py::arg("method") = SummationMethod::NaiveSummation 
        );
}
