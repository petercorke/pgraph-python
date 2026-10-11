Construction and basic usage
=============================

There are a number of ways to build a :class:`DGraph`/:class:`UGraph` (a plain constructor, or from
a dictionary/adjacency matrix).

See :doc:`policy` for the rules around constructing vertices/edges
standalone, independent of a graph.

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.DGraph
   ~pgraph.PGraph.UGraph
   ~pgraph.PGraph.BaseGraph.Dict
   ~pgraph.PGraph.BaseGraph.Adjacency


.. autoclass:: pgraph.PGraph.DGraph
   :show-inheritance:

.. autoclass:: pgraph.PGraph.UGraph
   :show-inheritance:

.. autoclass:: pgraph.PGraph.BaseGraph.Adjacency
   :show-inheritance:

.. autoclass:: pgraph.PGraph.BaseGraph.Dict
   :show-inheritance:


Inherited methods for all graph types
--------------------------------------

.. autoclass:: pgraph.PGraph.BaseGraph
   :members: copy, vertex_copy, metric, heuristic, show, vertex_class, edge_class
   :special-members: __init__, __str__, __repr__, __getitem__, __iter__, __contains__


