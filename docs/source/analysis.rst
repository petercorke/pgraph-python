Graph analysis
===============

Querying structure -- counts, neighbours, connectivity, distance -- for directed and
undirected graphs.
See :doc:`planning` for
finding paths between vertices, and :doc:`representations` for matrix/dot forms of the
whole graph.

Graph
-----

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph._BaseGraph.n
   ~pgraph.PGraph._BaseGraph.ne
   ~pgraph.PGraph._BaseGraph.nc
   ~pgraph.PGraph._BaseGraph.edges
   ~pgraph.PGraph._BaseGraph.average_degree
   ~pgraph.PGraph._BaseGraph.iscyclic
   ~pgraph.PGraph._BaseGraph.connectivity
   ~pgraph.PGraph._BaseGraph.component
   ~pgraph.PGraph._BaseGraph.samecomponent
   ~pgraph.PGraph._BaseGraph.closest

Vertex
------

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.BaseVertex.edges
   ~pgraph.PGraph.BaseVertex.neighbours
   ~pgraph.PGraph.BaseVertex.neighbors
   ~pgraph.PGraph.BaseVertex.isneighbour
   ~pgraph.PGraph.BaseVertex.incidences
   ~pgraph.PGraph.BaseVertex.degree
   ~pgraph.PGraph.BaseVertex.heuristic_distance
   ~pgraph.PGraph.BaseVertex.distance
   ~pgraph.PGraph.BaseVertex.x
   ~pgraph.PGraph.BaseVertex.y
   ~pgraph.PGraph.BaseVertex.z

Details
-------

.. autoproperty:: pgraph.PGraph._BaseGraph.n
.. autoproperty:: pgraph.PGraph._BaseGraph.ne
.. autoproperty:: pgraph.PGraph._BaseGraph.nc
.. automethod:: pgraph.PGraph._BaseGraph.edges
.. automethod:: pgraph.PGraph._BaseGraph.average_degree
.. automethod:: pgraph.PGraph._BaseGraph.iscyclic
.. automethod:: pgraph.PGraph._BaseGraph.connectivity
.. automethod:: pgraph.PGraph._BaseGraph.component
.. automethod:: pgraph.PGraph._BaseGraph.samecomponent
.. automethod:: pgraph.PGraph._BaseGraph.closest

.. automethod:: pgraph.PGraph.BaseVertex.edges
.. automethod:: pgraph.PGraph.BaseVertex.neighbours
.. automethod:: pgraph.PGraph.BaseVertex.neighbors
.. automethod:: pgraph.PGraph.BaseVertex.adjacent
.. automethod:: pgraph.PGraph.BaseVertex.isneighbour
.. automethod:: pgraph.PGraph.BaseVertex.incidences
.. autoproperty:: pgraph.PGraph.BaseVertex.degree
.. automethod:: pgraph.PGraph.BaseVertex.heuristic_distance
.. automethod:: pgraph.PGraph.BaseVertex.distance
.. autoproperty:: pgraph.PGraph.BaseVertex.x
.. autoproperty:: pgraph.PGraph.BaseVertex.y
.. autoproperty:: pgraph.PGraph.BaseVertex.z
