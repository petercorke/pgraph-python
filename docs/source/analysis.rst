Graph analysis
===============

Querying structure -- counts, adjacent vertices, connectivity, distance -- for directed and
undirected graphs.
See :doc:`planning` for
finding paths between vertices, and :doc:`representations` for matrix/dot forms of the
whole graph.

Graph
-----

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Graph.n
   ~pgraph.PGraph.Graph.ne
   ~pgraph.PGraph.Graph.nc
   ~pgraph.PGraph.Graph.edges
   ~pgraph.PGraph.Graph.average_degree
   ~pgraph.PGraph.Graph.iscyclic
   ~pgraph.PGraph.Graph.connectivity
   ~pgraph.PGraph.Graph.component
   ~pgraph.PGraph.Graph.samecomponent
   ~pgraph.PGraph.Graph.closest

Vertex
------

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Vertex.edges
   ~pgraph.PGraph.Vertex.adjacent
   ~pgraph.PGraph.Vertex.isadjacent
   ~pgraph.PGraph.Vertex.incidences
   ~pgraph.PGraph.Vertex.degree
   ~pgraph.PGraph.Vertex.heuristic_distance
   ~pgraph.PGraph.Vertex.distance
   ~pgraph.PGraph.Vertex.x
   ~pgraph.PGraph.Vertex.y
   ~pgraph.PGraph.Vertex.z

Details
-------

.. autoproperty:: pgraph.PGraph.Graph.n
.. autoproperty:: pgraph.PGraph.Graph.ne
.. autoproperty:: pgraph.PGraph.Graph.nc
.. automethod:: pgraph.PGraph.Graph.edges
.. automethod:: pgraph.PGraph.Graph.average_degree
.. automethod:: pgraph.PGraph.Graph.iscyclic
.. automethod:: pgraph.PGraph.Graph.connectivity
.. automethod:: pgraph.PGraph.Graph.component
.. automethod:: pgraph.PGraph.Graph.samecomponent
.. automethod:: pgraph.PGraph.Graph.closest

.. automethod:: pgraph.PGraph.Vertex.edges
.. automethod:: pgraph.PGraph.Vertex.adjacent
.. automethod:: pgraph.PGraph.Vertex.isadjacent
.. automethod:: pgraph.PGraph.Vertex.incidences
.. autoproperty:: pgraph.PGraph.Vertex.degree
.. automethod:: pgraph.PGraph.Vertex.heuristic_distance
.. automethod:: pgraph.PGraph.Vertex.distance
.. autoproperty:: pgraph.PGraph.Vertex.x
.. autoproperty:: pgraph.PGraph.Vertex.y
.. autoproperty:: pgraph.PGraph.Vertex.z
