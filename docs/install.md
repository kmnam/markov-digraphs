\page install Installing MarkovDigraphs and PyGraph

As **MarkovDigraphs** is a header-only package, no installation is necessary!
You only need to include the necessary header file(s) to use the classes therein,
e.g.,

```
#include <digraph.hpp>

int main()
{
    // Instantiate a LabeledDigraph object with double scalars
    LabeledDigraph<double, double>* graph = new LabeledDigraph<double, double>(); 

    // Do stuff with the graph ... 

    // Then make sure to delete the graph from the heap when you're done with it
    delete graph;
} 
```

Dependencies
------------

[**Eigen**](https://eigen.tuxfamily.org/index.php?title=Main_Page)
is the only truly unavoidable requirement for **MarkovDigraphs**. As such,
when compiling the above code (which, let's say, has been placed in a file
called `src/foo.cpp`), it is imperative that

1. the compiler uses the C++11 standard and 
2. the compiler knows where to find your copy of Eigen.

For instance, if you're using g++, then you could use something like:  

```
$ g++ --std=c++11 -O2 -Wall -I/absolute/path/to/eigen/include/eigen3 -o foo src/foo.cpp
```

Additional software that may be used with **MarkovDigraphs** include: 

- [**Boost.Multiprecision.**](https://www.boost.org/doc/libs/1_78_0/libs/multiprecision/doc/html/index.html)
  Although the fundamental class template in **MarkovDigraphs**, `LabeledDigraph`,
  allows the use of any (Eigen-compatible) scalar type, accurately computing the
  quantities enumerated above for even relatively simple graphs requires the use
  of
  [multiple-precision arithmetic](https://en.wikipedia.org/wiki/Arbitrary-precision_arithmetic).
  Many packages can be used for this purpose, but we recommend
  Boost.Multiprecision, which wraps the [**MPFR** library](https://www.mpfr.org/).

  When using Boost.Multiprecision scalars, it is additionally necessary to
  provide the location of Boost's header files to your compiler, **and also**
  use a linker flag to direct the compiler to use MPFR, as follows:
  ```
  $ g++ --std=c++11 -O2 -Wall -I/absolute/path/to/eigen/include/eigen3 \
      -I/absolute/path/to/boost/include -lmpfr -o foo src/foo.cpp 
  ```
  (Here, only Boost's header files are required because Boost.Multiprecision 
  is header-only.)
 
- [**Graphviz.**](https://graphviz.org/)
  The header file `viz.hpp` contains the function `vizLabeledDigraph()`, which 
  can be used to visualize a `LabeledDigraph` instance with Graphviz. Thus, 
  to use this function in the above example, we need to edit `src/foo.cpp` to
  include `viz.hpp`, as in: 
  ```
  #include <digraph.hpp>
  #include <viz.hpp>

  int main() 
  {
      // Instantiate a LabeledDigraph object with double scalars
      LabeledDigraph<double, double>* graph = new LabeledDigraph<double, double>(); 

      // Do stuff with the graph ...

      // Initialize a Graphviz context, draw the graph, then free the context 
      GVC_t* context = gvContext(); 
      vizLabeledDigraph(graph, "dot", "png", "foo.png", context); 
      gvFreeContext(context); 

      // Then make sure to delete the graph from the heap when you're done with it
      delete graph;
  } 
  ```
  To compile this code, you will need to provide the location of your copy of
  Graphviz to your compiler (both its header files and its library files),
  **as well as** provide the corresponding linker flags, e.g., 
  ```
  $ g++ --std=c++11 -O2 -Wall -I/absolute/path/to/eigen/include/eigen3 \
      -I/absolute/path/to/graphviz/include -L/absolute/path/to/graphviz/lib \
      -lgvc -lcgraph -o foo src/foo.cpp 
  ```

Installing PyGraph
------------------

Installing **PyGraph**, on the other hand, requires **Eigen**,
**Boost.Multiprecision**, **Graphviz**, and
[**pybind11**](https://pybind11.readthedocs.org/). To minimize the probability 
that you encounter problems when compiling **PyGraph**, you should first make
sure that you install pybind11 with the Python package manager that concords 
with your Python installation of choice. In other words, if you are using the
built-in installation of Python that comes with most Unix machines, then you
should use pip to install pybind11:
```
$ pip3 install pybind11
```
If you use [Anaconda](https://anaconda.org/), then you should use conda: 
```
$ conda install pybind11
```
If you prefer to keep your version of Python up-to-date with
[homebrew](https://brew.sh/), then you should use brew:
```
$ brew install pybind11
``` 

Now, to install **PyGraph**, you need to edit `setup.py` in the root directory
of the **MarkovDigraphs** package to indicate the locations of Eigen, Boost,
and Graphviz's header files, e.g., 
```
# In setup.py
...
BOOST_INCLUDE = '/usr/local/Cellar/boost/1.76.0/include'
EIGEN_INCLUDE = '/usr/local/Cellar/eigen/3.4.0_1/include/eigen3'
GRAPHVIZ_INCLUDE = '/usr/local/Cellar/graphviz/2.40.1/include'
...
```
then run the command (from the root directory)
```
$ python3 setup.py build_ext --inplace
```

This should produce a compiled object file with the name `pygraph.[SUFFIX]`,
where `SUFFIX` is determined by your version of Python (e.g., `cpython-39-darwin.so`).
If you're curious, you can find out what this suffix is (without compiling
**PyGraph**) by running
```
$ python3-config --extension-suffix
```
Now, all you have to do to import **PyGraph** into your Python code is to 
add the location of this object file (i.e., the **MarkovDigraphs** root 
directory) to Python's path: 
```
import os
import sys 
sys.path.append(os.path.abspath('/relative/path/to/markovdigraphs/root/'))

# Now you are free to import pygraph
import pygraph
graph = pygraph.PreciseDigraph()
...
```
More details on how to use **PyGraph** and its API can be found on the 
[**PyGraph** docs page](https://kmnam.github.io/pygraph-docs/). 
