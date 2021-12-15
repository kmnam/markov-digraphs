\mainpage MarkovDigraphs: directed graphs for studying biochemical systems 

**MarkovDigraphs** is a header-only C++ package that enables the study
of **labeled directed graphs** as representations of **continuous-time
discrete-state Markov processes**, as formulated in the
[**linear framework**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036321)
for modeling a wide range of biochemical processes.

Background: the linear framework
--------------------------------

In short, **MarkovDigraphs** implements a class, `LabeledDigraph`, with
which one can use a directed graph with labeled edges to encode a Markov 
process that, in turn, encodes the dynamics of a biochemical system.
The **nodes** of the graph represent the **states** of the system, 
the **edges** represent the **transitions** that the system can undergo 
from one state to another, and the **edge labels** represent the **rates** 
at which these transitions occur. By examining the structure of the graph,
we can then begin to understand the properties of this system's dynamics.

From here, **MarkovDigraphs** allows us to answer the following questions
about any directed graph and its associated Markov process: 

1. What is the **steady-state probability** that the process occupies a 
   particular node in the graph?
2. What is the **mean amount of time** that the process takes to reach 
   one node from another for the first time, i.e., the **mean first-passage
   time** to one node from another?
3. What is the **variance** of the first-passage time to one node from 
   another?

Many problems in single-molecule biochemistry can be, and have been,
phrased in this language, from
[gene regulation](https://www.sciencedirect.com/science/article/pii/S0092867416307413)
to
[enzyme kinetics](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.012420)
to
[protein conformational landscapes](https://doi.org/10.1016/j.sbi.2014.04.002)
and [so on](https://vcp.med.harvard.edu/papers.html). This approach has
also proved fruitful for understanding the importance of
**energy expenditure** in biology, and, more generally, gleaning insights
into the hairy world of non-equilibrium physics.  

PyGraph: accessing MarkovDigraphs through Python 3
--------------------------------------------------

Also available is [**PyGraph**](https://kmnam.github.io/pygraph-docs/), a
set of Python 3 bindings for **MarkovDigraphs** implemented with
[**pybind11**](https://pybind11.readthedocs.io/) with a nearly identical API.
**MarkovDigraphs** and **PyGraph** are under rapid parallel development, 
and updates to the former will nearly always be accompanied by immediate 
updates to the latter. 

Requirements for MarkovDigraphs
-------------------------------

**MarkovDigraphs** requires 
[**Eigen**](https://eigen.tuxfamily.org/index.php?title=Main_Page)
for linear algebra. Using the graph visualization function
``vizLabeledDigraph()`` additionally requires
[**Graphviz**](https://graphviz.org/).

See [here](install.html) for details on how to start using **MarkovDigraphs**. 

Requirements for PyGraph
------------------------

To install **PyGraph** as a Python 3 extension module that can be imported into
your code, you will additionally need **Boost.Multiprecision**, **Graphviz**
(see above), and [**pybind11**](https://pybind11.readthedocs.io/).
See [here](install.html) for details. 

