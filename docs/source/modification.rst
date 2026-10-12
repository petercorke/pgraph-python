Adding and removing vertices and edges
========================================

Building up and tearing down a graph's structure. ``connect()`` is the same
for both kinds of graph: the graph decides how the edge is recorded at its
vertices, so an undirected edge is found from either end and a directed one
leaves its start vertex and enters its end vertex. See :class:`Edge` for
constructing/removing an edge directly rather than via a vertex or graph.

Graph
-----

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Graph.add_vertex
   ~pgraph.PGraph.Graph.add_edge
   ~pgraph.PGraph.Graph.remove_edge
   ~pgraph.PGraph.Graph.remove_vertex

Vertex
------

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Vertex.connect
   ~pgraph.PGraph.Vertex.edgeto
   ~pgraph.PGraph.Vertex.remove

Details
-------

.. automethod:: pgraph.PGraph.Graph.add_vertex
.. automethod:: pgraph.PGraph.Graph.add_edge
.. automethod:: pgraph.PGraph.Graph.remove_edge
.. automethod:: pgraph.PGraph.Graph.remove_vertex
.. automethod:: pgraph.PGraph.Graph.remove

.. automethod:: pgraph.PGraph.Vertex.connect
.. automethod:: pgraph.PGraph.Vertex.edgeto
.. automethod:: pgraph.PGraph.Vertex.remove
