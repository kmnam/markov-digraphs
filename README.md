# MarkovDigraphs and PyGraph

## Directed graphs as representations of Markov processes in the linear framework

**Author: Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School**

[**MarkovDigraphs**](https://kmnam.github.io/markov-digraphs/)
is a header-only C++ package that enables the study of **labeled directed
graphs** as representations of **continuous-time discrete-state Markov
processes**, as formulated in the
[linear framework](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036321)
for modeling biochemical processes such as
[enhancer-mediated gene regulation](https://www.cell.com/cell/fulltext/S0092-8674(16)30741-3),
[post-translational modification](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007573),
[substrate discrimination by enzymes](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.012420),
among [others](https://vcp.med.harvard.edu/papers.html).

The classes that comprise MarkovDigraphs are further available for use in
Python, as [**PyGraph**](https://kmnam.github.io/pygraph-docs/), through 
bindings implemented using [pybind11](https://pybind11.readthedocs.io/en/stable/).

MarkovDigraphs requires [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
(version >= 3.4) and [Boost](https://www.boost.org/), in particular
[Boost.Multiprecision](https://www.boost.org/doc/libs/1_78_0/libs/multiprecision/doc/html/index.html),
which wraps the [MPFR library](https://www.mpfr.org/) for multiple-precision
arithmetic. PyGraph additionally requires pybind11.

Detailed instructions on installation and usage are rapidly forthcoming.  
