\page pageviz Visualizing graphs with Graphviz 

**MarkovDigraphs** also provides a function, `vizLabeledDigraph()`, for
producing visualizations of `LabeledDigraph` instances using the
[**Graphviz**](https://graphviz.org/) library.

In addition to (a pointer to) the `LabeledDigraph` instance and a Graphviz 
context, `vizLabeledDigraph()` requires three arguments:
 
- One of the [eight Graphviz layout algorithms](https://graphviz.org/docs/layouts/)
  to use to distribute the nodes and edges, 
- The format of the output file (`"png"`, `"pdf"`, `"jpeg"`, `"gif"`, etc.), and
- The output file name. 

See [here](https://graphviz.org/docs/outputs/) for a list of all possible 
output file formats.

Here are three visualizations of the cycle graph on five vertices (named
**first**, **second**, **third**, **fourth**, and **fifth**), 
produced using the [dot](https://graphviz.org/docs/layouts/dot/),
[neato](https://graphviz.org/docs/layouts/neato/), and
[circo](https://graphviz.org/docs/layouts/circo/) layout algorithms.

dot                  | neato                    | circo
:-------------------:|:------------------------:|:------------------------:
![dot](test-dot.png) | ![neato](test-neato.png) | ![circo](test-circo.png) 
