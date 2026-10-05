Graph analysis
===============

Querying structure -- counts, neighbours, connectivity, distance -- without
converting the graph into another representation. See :doc:`planning` for
finding paths between vertices, and :doc:`representations` for matrix/dot
forms of the whole graph.

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
