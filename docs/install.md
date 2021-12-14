\page install Installing MarkovDigraphs and PyGraph

As **MarkovDigraphs** is a header-only package, no installation is necessary!
You only need to include the necessary header file(s) to use the classes therein,
e.g.,

```
#include <digraph.hpp>

// Instantiate a LabeledDigraph object with double scalars
LabeledDigraph<double, double>* graph = new LabeledDigraph<double, double>(); 

// Do stuff with the graph ... 

// Then make sure to delete it when you're done with it
delete graph; 
```

Dependencies
------------

Now, when compiling the above code, it is imperative that

1. the compiler uses the C++11 standard and 
2. the compiler knows where to find your copy of Eigen.

For instance, if you're using g++, then you could use something like:  

```
g++ --std=c++11 -O2 -Wall -I/absolute/path/to/eigen/include/eigen3 -o foo src/foo.cpp
```

When using Boost.Multiprecision scalar types (which is essential even for 
relatively simple graphs), it is additionally necessary to provide the 
location of your copy of Boost to your compiler, **and also** use a linker
flag to direct the compiler to use the MPFR library: 

```
g++ --std=c++11 -O2 -Wall -I/absolute/path/to/eigen/include/eigen3 \
    -I/absolute/path/to/boost/include -lmpfr -o foo src/foo.cpp 
```
