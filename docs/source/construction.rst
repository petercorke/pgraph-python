Construction and basic usage
=============================

How to build a :class:`DGraph`/:class:`UGraph` (a plain constructor, or from
a dictionary/adjacency matrix), copy one, configure its distance metric, and
the basic object protocol (``str``/``repr``, indexing, iteration,
membership).

See :doc:`policy` for the rules around constructing vertices/edges
standalone, independent of a graph.

.. autoclass:: pgraph.PGraph._BaseGraph
   :members: Dict, Adjacency, copy, vertex_copy, metric, heuristic, show
   :special-members: __init__, __str__, __repr__, __getitem__, __iter__, __contains__

.. autoclass:: pgraph.PGraph.BaseVertex
   :members: copy
   :special-members: __init__, __str__, __repr__
