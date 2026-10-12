Graph analysis
===============

Querying structure -- counts, adjacent vertices, incident edges, degree,
connectivity, distance -- for directed and undirected graphs.
See :doc:`planning` for finding paths between vertices, and
:doc:`representations` for the matrix and dot forms of the whole graph.

A vertex is queried in two complementary ways: by the vertices it is
*adjacent* to, and by the *edges* incident on it.  For the adjacent vertices
:meth:`~pgraph.PGraph.Vertex.adjacent` is the canonical name; for a directed
graph :meth:`~pgraph.PGraph.Vertex.successors` is the same list under its
directed name, and :meth:`~pgraph.PGraph.Vertex.predecessors` is the vertices
with an edge *to* the vertex.  The edges are
:meth:`~pgraph.PGraph.Vertex.out_edges` and
:meth:`~pgraph.PGraph.Vertex.in_edges`, and
:meth:`~pgraph.PGraph.Vertex.incidences` pairs each adjacent vertex with the
edge that leads to it.  :meth:`~pgraph.PGraph.Vertex.edgeto` is the edge to
one adjacent vertex, the cheapest if there are parallel edges, and
:meth:`~pgraph.PGraph.Vertex.edges_to` lists all of them.

Graph
-----

The size and kind of the graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Graph.n
   ~pgraph.PGraph.Graph.ne
   ~pgraph.PGraph.Graph.nc
   ~pgraph.PGraph.Graph.directed
   ~pgraph.PGraph.Graph.edges

Structure and connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Graph.iscyclic
   ~pgraph.PGraph.Graph.component
   ~pgraph.PGraph.Graph.samecomponent
   ~pgraph.PGraph.Graph.connectivity
   ~pgraph.PGraph.Graph.average_degree

Geometry and consistency
^^^^^^^^^^^^^^^^^^^^^^^^^

For a graph whose vertices have coordinates,
:meth:`~pgraph.PGraph.Graph.coordinates` is the matrix of them, one column per
vertex.  :meth:`~pgraph.PGraph.Graph.closest` finds the nearest vertex to a
point, :meth:`~pgraph.PGraph.Graph.near` the vertices within a distance of a
point and :meth:`~pgraph.PGraph.Graph.within` the vertices inside a box.  Passing
the result of the last to :meth:`~pgraph.PGraph.Graph.subgraph` gives the part of
the graph in a region, without the edges that cross its border, and a boolean
mask computed from the coordinates selects in the same way.

These are a single pass over the vertices.  For a large graph that is queried
often, :meth:`~pgraph.PGraph.Graph.build_kdtree` builds an index that
:meth:`~pgraph.PGraph.Graph.closest` and :meth:`~pgraph.PGraph.Graph.near` then
use, which needs scipy and is for the built-in metrics.

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Graph.coordinates
   ~pgraph.PGraph.Graph.closest
   ~pgraph.PGraph.Graph.near
   ~pgraph.PGraph.Graph.within
   ~pgraph.PGraph.Graph.build_kdtree
   ~pgraph.PGraph.Graph.subgraph
   ~pgraph.PGraph.Graph.validate

The matrices of a graph (:meth:`~pgraph.PGraph.Graph.adjacency`,
:meth:`~pgraph.PGraph.Graph.degree`, :meth:`~pgraph.PGraph.Graph.incidence`,
:meth:`~pgraph.PGraph.Graph.laplacian`, :meth:`~pgraph.PGraph.Graph.distance`)
are in :doc:`representations`.

Vertex
------

Identity and position
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Vertex.name
   ~pgraph.PGraph.Vertex.label
   ~pgraph.PGraph.Vertex.x
   ~pgraph.PGraph.Vertex.y
   ~pgraph.PGraph.Vertex.z

Adjacent vertices
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Vertex.adjacent
   ~pgraph.PGraph.Vertex.isadjacent
   ~pgraph.PGraph.Vertex.successors
   ~pgraph.PGraph.Vertex.predecessors

.. note:: ``neighbours()`` and ``neighbors()`` are the same method as
   :meth:`~pgraph.PGraph.Vertex.adjacent`, and ``isneighbour()`` and
   ``isneighbor()`` the same as :meth:`~pgraph.PGraph.Vertex.isadjacent`.
   They are spelling variants, not deprecated.

Incident edges
^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Vertex.edges
   ~pgraph.PGraph.Vertex.out_edges
   ~pgraph.PGraph.Vertex.in_edges
   ~pgraph.PGraph.Vertex.incidences
   ~pgraph.PGraph.Vertex.edgeto
   ~pgraph.PGraph.Vertex.edges_to

Degree
^^^^^^^

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Vertex.degree
   ~pgraph.PGraph.Vertex.in_degree
   ~pgraph.PGraph.Vertex.out_degree

Distance
^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Vertex.distance
   ~pgraph.PGraph.Vertex.heuristic_distance
   ~pgraph.PGraph.Vertex.closest

Edge
----

An :class:`~pgraph.PGraph.Edge` is described in full on :doc:`edges_vertices`.  The
attributes and methods used to query one are

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Edge.v1
   ~pgraph.PGraph.Edge.v2
   ~pgraph.PGraph.Edge.endpoints
   ~pgraph.PGraph.Edge.next

and every edge also has ``cost`` and ``data`` attributes.

Details
-------

.. autoproperty:: pgraph.PGraph.Graph.n
.. autoproperty:: pgraph.PGraph.Graph.ne
.. autoproperty:: pgraph.PGraph.Graph.nc
.. autoproperty:: pgraph.PGraph.Graph.directed
.. automethod:: pgraph.PGraph.Graph.edges
.. automethod:: pgraph.PGraph.Graph.iscyclic
.. automethod:: pgraph.PGraph.Graph.component
.. automethod:: pgraph.PGraph.Graph.samecomponent
.. automethod:: pgraph.PGraph.Graph.connectivity
.. automethod:: pgraph.PGraph.Graph.average_degree
.. automethod:: pgraph.PGraph.Graph.coordinates
.. automethod:: pgraph.PGraph.Graph.closest
.. automethod:: pgraph.PGraph.Graph.near
.. automethod:: pgraph.PGraph.Graph.within
.. automethod:: pgraph.PGraph.Graph.build_kdtree
.. automethod:: pgraph.PGraph.Graph.subgraph
.. automethod:: pgraph.PGraph.Graph.validate

.. autoproperty:: pgraph.PGraph.Vertex.name
.. autoproperty:: pgraph.PGraph.Vertex.label
.. autoproperty:: pgraph.PGraph.Vertex.x
.. autoproperty:: pgraph.PGraph.Vertex.y
.. autoproperty:: pgraph.PGraph.Vertex.z
.. automethod:: pgraph.PGraph.Vertex.adjacent
.. automethod:: pgraph.PGraph.Vertex.isadjacent
.. automethod:: pgraph.PGraph.Vertex.successors
.. automethod:: pgraph.PGraph.Vertex.predecessors
.. automethod:: pgraph.PGraph.Vertex.edges
.. automethod:: pgraph.PGraph.Vertex.out_edges
.. automethod:: pgraph.PGraph.Vertex.in_edges
.. automethod:: pgraph.PGraph.Vertex.incidences
.. automethod:: pgraph.PGraph.Vertex.edgeto
.. automethod:: pgraph.PGraph.Vertex.edges_to
.. autoproperty:: pgraph.PGraph.Vertex.degree
.. autoproperty:: pgraph.PGraph.Vertex.in_degree
.. autoproperty:: pgraph.PGraph.Vertex.out_degree
.. automethod:: pgraph.PGraph.Vertex.distance
.. automethod:: pgraph.PGraph.Vertex.heuristic_distance
.. automethod:: pgraph.PGraph.Vertex.closest
