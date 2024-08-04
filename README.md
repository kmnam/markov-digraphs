# MarkovDigraphs and PyGraph

## Directed labeled graphs as representations of Markov processes in the linear framework

**Author: Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School**

[**MarkovDigraphs**](https://kmnam.github.io/markov-digraphs/)
is a header-only C++ package that enables the study of **labeled directed
graphs** as representations of **continuous-time discrete-state Markov
processes**, as formulated in the
**linear framework** ([see](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036321) [these](https://link.springer.com/article/10.1007/s11538-013-9884-8) [papers](https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-014-0102-4) [for](https://royalsocietypublishing.org/doi/10.1098/rsfs.2022.0013) [details](https://www.frontiersin.org/articles/10.3389/fcell.2023.1233808/abstract))
for modeling a wide range of biochemical processes. See the
[**MarkovDigraphs** documentation](https://kmnam.github.io/markov-digraphs/)
for further background and a survey of its capabilities.  

The classes that comprise **MarkovDigraphs** are further available for use in
Python 3, as [**PyGraph**](https://kmnam.github.io/pygraph-docs/_build/html/), through 
bindings implemented using [pybind11](https://pybind11.readthedocs.io/en/stable/).

## Installation

MarkovDigraphs requires the following:
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (version >= 3.4),
- [Boost](https://www.boost.org/), and
- [Graphviz](https://www.graphviz.org/).

PyGraph additionally requires:
- pybind11; and
- [GMP](https://gmplib.org)/[MPFR](https://mpfr.org) for multiple-precision
  arithmetic.

MarkovDigraphs is a header-only library, and therefore requires no separate
installation. You need only download the code, as follows:
```
$ git clone https://github.com/kmnam/markov-digraphs.git
```
and include the necessary header files to begin using the classes implemented
therein. 

To install PyGraph, you can do the following:
1. The script ``setup.py`` is written to compile PyGraph under ``build/``.
   If you would like to overwrite a previously installed version PyGraph, you
   can simply remove ``build/`` before proceeding as described below. 
2. Edit the include paths in ``setup.py`` according to your installations of 
   GMP, MPFR, Eigen, Boost, and Graphviz.
3. Run the script, as 
   ```
   $ python setup.py build_ext
   ```
4. This will yield a directory, ``build/``, with a sub-directory containing
   the compiled library, which (on a Mac) should be named as something like,
   ``lib.macosx-14.0-arm64-cpython-312/``. To expose this library to Python, 
   you can run:
   ```
   >> import sys
   >> import os
   >> sys.path.append(os.path.abspath('build/lib.macosx-14.0-arm64-cpython-312/'))
   >> import pygraph    # This should not raise an error
   ```

## Documentation

The documentation for MarkovDigraphs is available [here](https://kmnam.github.io/markov-digraphs/).

The documentation for PyGraph is available [here](https://kmnam.github.io/pygraph-docs/_build/html/).
