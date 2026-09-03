Object lifecycle and graph-membership policy
=============================================

``BaseVertex``, ``UVertex``, ``DVertex`` and ``Edge`` can all be constructed
directly, independent of any graph. This is deliberate -- it is not an
oversight and not a "factory-only" design -- but it means the rules for what
those objects can and can't do *before* they belong to a graph need to be
stated explicitly rather than left implicit.

Standalone construction
------------------------

A vertex or edge built directly, e.g. ``UVertex(coord=[1, 2], name="A")`` or
``Edge(v1, v2, cost=5.0)``, is a fully valid object in its own right. The
main supported use of this is lightweight, graph-independent work: building a
couple of vertices purely to demonstrate or test :class:`Edge` cost
computation, without needing a graph at all.

An object in this state has ``._graph is None``. It cannot be connected to
anything (see below) until it is added to a graph via
:meth:`UGraph.add_vertex` / :meth:`DGraph.add_vertex`.

Adding a vertex to a graph
---------------------------

:meth:`UGraph.add_vertex` and :meth:`DGraph.add_vertex` each only accept
their own matching vertex subclass -- a ``UVertex`` (or coordinate data, from
which one is built) for :class:`UGraph`, a ``DVertex`` for :class:`DGraph`.
Passing the wrong kind of vertex -- a plain ``BaseVertex``, or the other
graph's subclass -- raises ``TypeError`` rather than being silently accepted
or converted.

This is the mechanism that makes graph membership meaningful: once added, a
vertex's ``._graph`` is fixed to that specific graph instance, and its
concrete type (``UVertex`` or ``DVertex``) is guaranteed correct for that
graph's edge semantics (undirected vs. directed edge-list bookkeeping).

Connecting two vertices
-------------------------

``v1.connect(v2)`` (and the equivalent ``graph.add_edge(v1, v2)``) requires
**both vertices to already belong to the same graph**:

- If either vertex has not been added to any graph (``._graph is None``),
  ``connect()`` raises ``ValueError``.
- If the two vertices belong to different graph instances, ``connect()``
  raises ``ValueError``.

There is deliberately no separate check for "same vertex subclass" (e.g.
rejecting a ``UVertex``-to-``DVertex`` connection directly). It doesn't need
one: since a ``UVertex`` can only ever belong to a ``UGraph`` and a
``DVertex`` only to a ``DGraph`` (enforced by ``add_vertex()`` above), the
same-graph check above already makes a ``UVertex``-to-``DVertex`` connection
impossible as a structural consequence, not as a separately-maintained rule.

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
   * - ``UVertex()`` / ``DVertex()`` / ``BaseVertex()``
     - No
     - --
   * - ``Edge(v1, v2)``
     - No (cost falls back to ``None`` if ungraphed/cross-graph)
     - --
   * - ``graph.add_vertex(v)``
     - ``v`` must be the graph's own vertex subclass, or coordinate data
     - ``TypeError`` on wrong vertex subclass
   * - ``v1.connect(v2)`` / ``graph.add_edge(v1, v2)``
     - Both vertices must already belong to the same graph
     - ``ValueError`` if either is ungraphed or they belong to different
       graphs
