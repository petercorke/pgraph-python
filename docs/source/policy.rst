Object lifecycle and graph-membership policy
=============================================

``Vertex`` and ``Edge`` can both be constructed
directly, independent of any graph. This is deliberate -- it is not an
oversight and not a "factory-only" design -- but it means the rules for what
those objects can and can't do *before* they belong to a graph need to be
stated explicitly rather than left implicit.

Standalone construction
------------------------

A vertex or edge built directly, e.g. ``Vertex(coord=[1, 2], name="A")`` or
``Edge(v1, v2, cost=5.0)``, is a fully valid object in its own right. The
main supported use of this is lightweight, graph-independent work: building a
couple of vertices purely to demonstrate or test :class:`Edge` cost
computation, without needing a graph at all.

An object in this state has ``._graph is None``. It cannot be connected to
anything (see below) until it is added to a graph via
:meth:`UGraph.add_vertex` / :meth:`DGraph.add_vertex`.

Adding a vertex to a graph
---------------------------

A vertex is the same for undirected and directed graphs: it has no
direction of its own, and the graph it belongs to decides how an edge is
recorded at its ends.  :meth:`Graph.add_vertex` accepts coordinate data, from
which a vertex is built, or an existing ``Vertex`` that does not belong to a
graph.  A graph given a ``vertex_class`` only accepts instances of that class;
passing any other vertex raises ``TypeError`` rather than being silently
accepted or converted.  A vertex that already belongs to a graph raises
``ValueError``; remove it from that graph first.

This is the mechanism that makes graph membership meaningful: once added, a
vertex's ``._graph`` is fixed to that specific graph instance.

Connecting two vertices
-------------------------

``v1.connect(v2)`` (and the equivalent ``graph.add_edge(v1, v2)``) requires
**both vertices to already belong to the same graph**:

- If either vertex has not been added to any graph (``._graph is None``),
  ``connect()`` raises ``ValueError``.
- If the two vertices belong to different graph instances, ``connect()``
  raises ``ValueError``.

There is no such thing as connecting an "undirected" vertex to a "directed"
one, since a vertex is neither: the kind of edge is the graph's, so the
same-graph check above is the only one needed.

Constructing an ``Edge`` directly
-----------------------------------

``Edge(v1, v2)`` (without going through ``connect()``) does not raise if the
vertices aren't graph-connected -- this constructor is also the supported
standalone-use path described above, so it stays permissive. Instead:

- If an explicit ``cost`` is given, it is used as-is.
- Otherwise, the edge cost is auto-computed from vertex coordinates via the
  graph's metric *only if* both vertices are added to the same graph.
  Otherwise ``cost`` is left as ``None`` rather than raising.

This means a bare ``Edge(v1, v2)`` with orphaned or cross-graph vertices is a
valid object with ``cost=None`` -- it is only the *connecting* operations
(``connect()``, ``add_edge()``) that enforce the graph-membership invariant
strictly.

Summary
--------

.. list-table::
   :header-rows: 1

   * - Operation
     - Requires graph membership?
     - Failure mode
   * - ``Vertex()``
     - No
     - --
   * - ``Edge(v1, v2)``
     - No (cost falls back to ``None`` if ungraphed/cross-graph)
     - --
   * - ``graph.add_vertex(v)``
     - ``v`` must be of the graph's ``vertex_class`` and not already in a
       graph, or coordinate data
     - ``TypeError`` on a vertex of another class, ``ValueError`` if it is
       already in a graph
   * - ``v1.connect(v2)`` / ``graph.add_edge(v1, v2)``
     - Both vertices must already belong to the same graph
     - ``ValueError`` if either is ungraphed or they belong to different
       graphs
