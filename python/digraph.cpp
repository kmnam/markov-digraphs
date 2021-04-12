#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include "../include/digraph.hpp"

/*
 * Python bindings for digraph.hpp. 
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     4/12/2021
 */

namespace py = pybind11;

template <typename T>
void declareModule(py::module& m, const std::string& typestr)
{
    using Class = LabeledDigraph<T>;
    std::string class_name = std::string("LabeledDigraph") + typestr;

    // Define Python bindings for Node
    py::class_<Node>(m, "Node")
        .def(py::init<std::string>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        ;

    // Define Python bindings for LabeledDigraph<T>
    py::class_<Class>(m, class_name.c_str())
        .def(py::init<>())
        .def("addNode", &Class::addNode, py::return_value_policy::reference)
        .def("removeNode", &Class::removeNode)
        .def("getNode", &Class::getNode)
        .def("hasNode", &Class::hasNode)
        .def("addEdge", &Class::addEdge, py::return_value_policy::reference)
        .def("getEdge", &Class::getEdge)
        .def("hasEdge", &Class::hasEdge)
        .def("setEdgeLabel", &Class::setEdgeLabel)
        //.def("subgraph", &Class::subgraph)
        .def("clear", &Class::clear)
        //.def("copy", static_cast<Class* (Class::*)()>(&Class::template copy<T>))
        //.def("copy", static_cast<void (Class::*)(Class*)>(&Class::template copy<T>))
        .def(
            "getLaplacian", &Class::getLaplacian,
            py::return_value_policy::reference_internal
        )
        .def(
            "getSpanningForestMatrix", &Class::getSpanningForestMatrix,
            py::return_value_policy::reference_internal
        )
        .def(
            "getSteadyStateFromSVD", &Class::getSteadyStateFromSVD,
            py::return_value_policy::reference_internal
        )
        .def(
            "getSteadyStateFromRecurrence", &Class::getSteadyStateFromRecurrence,
            py::return_value_policy::reference_internal
        )
        ;
}

// Then declare each template specialization
PYBIND11_MODULE(digraph, m)
{
    declareModule<double>(m, "Double");
}

