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

Distance metrics
----------------

The cost of an edge, the nearest vertex to a point and the heuristic of the A*
planner all come from a distance metric, a function of the difference between
two coordinates.  It is chosen by name, ``"L2"`` (the default, Euclidean),
``"L1"`` (Manhattan) or ``"SE2"``, or you can supply your own function; see
:attr:`~pgraph.PGraph.Graph.metric` and :attr:`~pgraph.PGraph.Graph.heuristic`.

The ``"SE2"`` metric is for poses :math:`(x, y, \theta)` and adds a position
in metres to an angle in radians, which are not the same kind of quantity.  It
counts a radian as a metre.  If a turn should count for more or less, use
:class:`~pgraph.PGraph.SE2Metric` with a weight, which is the length that a
radian is worth.  Below, one vertex is 2 m ahead and facing the same way, and
another is at our position but turned through 1.5 radians.  With a weight of 1 the
turn costs 1.5, so the turned vertex is the nearer; with a weight of 2 it costs 3,
and the vertex ahead is nearer

.. runblock:: pycon

    >>> from pgraph import UGraph, SE2Metric
    >>> for weight in (1.0, 2.0):
    ...     g = UGraph(dim=3, metric=SE2Metric(weight=weight))
    ...     v = g.add_vertex([2, 0, 0], name="ahead")
    ...     v = g.add_vertex([0, 0, 1.5], name="turned")
    ...     vertex, distance = g.closest([0, 0, 0])
    ...     print(weight, vertex.name, distance)

.. autoclass:: pgraph.PGraph.SE2Metric


