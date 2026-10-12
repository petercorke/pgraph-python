Path planning
=============

These methods find a path from a start vertex to a goal vertex.  :meth:`~pgraph.PGraph.Graph.path_BFS`
finds the path with the fewest edges, while :meth:`~pgraph.PGraph.Graph.path_UCS` and
:meth:`~pgraph.PGraph.Graph.path_Astar` find the path with the lowest total cost.

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Graph.path_BFS
   ~pgraph.PGraph.Graph.path_UCS
   ~pgraph.PGraph.Graph.path_Astar

Details
-------

.. automethod:: pgraph.PGraph.Graph.path_BFS
.. automethod:: pgraph.PGraph.Graph.path_UCS
.. automethod:: pgraph.PGraph.Graph.path_Astar
