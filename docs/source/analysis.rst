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

   ~pgraph.PGraph.BaseGraph.n
   ~pgraph.PGraph.BaseGraph.ne
   ~pgraph.PGraph.BaseGraph.nc
   ~pgraph.PGraph.BaseGraph.edges
   ~pgraph.PGraph.BaseGraph.average_degree
   ~pgraph.PGraph.BaseGraph.iscyclic
   ~pgraph.PGraph.BaseGraph.connectivity
   ~pgraph.PGraph.BaseGraph.component
   ~pgraph.PGraph.BaseGraph.samecomponent
   ~pgraph.PGraph.BaseGraph.closest

Vertex
------

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.BaseVertex.edges
   ~pgraph.PGraph.BaseVertex.adjacent
   ~pgraph.PGraph.BaseVertex.isadjacent
   ~pgraph.PGraph.BaseVertex.incidences
   ~pgraph.PGraph.BaseVertex.degree
   ~pgraph.PGraph.BaseVertex.heuristic_distance
   ~pgraph.PGraph.BaseVertex.distance
   ~pgraph.PGraph.BaseVertex.x
   ~pgraph.PGraph.BaseVertex.y
   ~pgraph.PGraph.BaseVertex.z

Details
-------

.. autoproperty:: pgraph.PGraph.BaseGraph.n
.. autoproperty:: pgraph.PGraph.BaseGraph.ne
.. autoproperty:: pgraph.PGraph.BaseGraph.nc
.. automethod:: pgraph.PGraph.BaseGraph.edges
.. automethod:: pgraph.PGraph.BaseGraph.average_degree
.. automethod:: pgraph.PGraph.BaseGraph.iscyclic
.. automethod:: pgraph.PGraph.BaseGraph.connectivity
.. automethod:: pgraph.PGraph.BaseGraph.component
.. automethod:: pgraph.PGraph.BaseGraph.samecomponent
.. automethod:: pgraph.PGraph.BaseGraph.closest

.. automethod:: pgraph.PGraph.BaseVertex.edges
.. automethod:: pgraph.PGraph.BaseVertex.adjacent
.. automethod:: pgraph.PGraph.BaseVertex.isadjacent
.. automethod:: pgraph.PGraph.BaseVertex.incidences
.. autoproperty:: pgraph.PGraph.BaseVertex.degree
.. automethod:: pgraph.PGraph.BaseVertex.heuristic_distance
.. automethod:: pgraph.PGraph.BaseVertex.distance
.. autoproperty:: pgraph.PGraph.BaseVertex.x
.. autoproperty:: pgraph.PGraph.BaseVertex.y
.. autoproperty:: pgraph.PGraph.BaseVertex.z
