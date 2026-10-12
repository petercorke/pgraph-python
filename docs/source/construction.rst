Construction and basic usage
=============================

There are a number of ways to build a :class:`~pgraph.PGraph.UGraph` (undirected) or :class:`~pgraph.PGraph.DGraph` (directed): a plain
constructor, or from a dictionary or an adjacency matrix.  :class:`~pgraph.PGraph.Graph` is their common base class,
used in type hints, and ``Graph(directed=flag)`` chooses the kind from a value.

See :doc:`policy` for the rules around constructing vertices/edges
standalone, independent of a graph.

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Graph
   ~pgraph.PGraph.DGraph
   ~pgraph.PGraph.UGraph
   ~pgraph.PGraph.Graph.Dict
   ~pgraph.PGraph.Graph.Adjacency


.. autoclass:: pgraph.PGraph.DGraph
   :show-inheritance:

.. autoclass:: pgraph.PGraph.UGraph
   :show-inheritance:

.. autoclass:: pgraph.PGraph.Graph.Adjacency
   :show-inheritance:

.. autoclass:: pgraph.PGraph.Graph.Dict
   :show-inheritance:


Inherited methods for all graph types
--------------------------------------

.. autoclass:: pgraph.PGraph.Graph
   :members: copy, vertex_copy, metric, heuristic, show, vertex_class, edge_class
   :special-members: __init__, __str__, __repr__, __getitem__, __iter__, __contains__

Copying a vertex
----------------

.. automethod:: pgraph.PGraph.Vertex.copy


