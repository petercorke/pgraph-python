Adding and removing vertices and edges
========================================

Building up and tearing down a graph's structure can be achieved by methods on the graph, vertex, and edge classes.
These methods work for directed and undirected graphs.


Graph
-----

The graph class provides methods for adding and removing vertices and edges.

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Graph.add_vertex
   ~pgraph.PGraph.Graph.add_edge
   ~pgraph.PGraph.Graph.remove_edge
   ~pgraph.PGraph.Graph.remove_vertex

Vertex
------

The vertex class provides methods for connecting a vertex to another vertex and for removing
a vertex, and its edges, from the graph.

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Vertex.connect
   ~pgraph.PGraph.Vertex.remove

Edge
------

The edge class provides methods for attaching an edge to a pair of vertices and for removing
an edge from the graph.  They are described on the :doc:`edges_vertices` page.

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Edge.connect
   ~pgraph.PGraph.Edge.remove

Details
-------

.. automethod:: pgraph.PGraph.Graph.add_vertex
.. automethod:: pgraph.PGraph.Graph.add_edge
.. automethod:: pgraph.PGraph.Graph.remove_edge
.. automethod:: pgraph.PGraph.Graph.remove_vertex
.. automethod:: pgraph.PGraph.Graph.remove

.. automethod:: pgraph.PGraph.Vertex.connect
.. automethod:: pgraph.PGraph.Vertex.remove
