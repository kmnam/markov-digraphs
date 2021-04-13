"""
Test code for Python bindings for MarkovDigraph<double>. 

Authors:
    Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
Last updated:
    4/12/2021
"""
from digraph import LabeledDigraphDouble

# Instantiate a graph of the form 0 <--> 1 <--> 2 <--> 3 <--> 4 <--> 5
graph = LabeledDigraphDouble()
for i in range(5):
    source = str(i)
    target = str(i + 1)
    graph.addEdge(source, target, 1.0)
    graph.addEdge(target, source, 1.0)

# Compute the Laplacian and spanning forest matrices 
print(graph.getLaplacian())
print(graph.getSpanningForestMatrix(4))
print(graph.getSpanningForestMatrix(5))
