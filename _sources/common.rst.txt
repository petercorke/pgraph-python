Common methods
===============

The package supports both directed (prefix ``D``) and undirected graphs (prefix ``U``), and the classes for these
share a common interface. 

The graph classes represent a single graph, either directed or undirected, and they share :class:`~pgraph.PGraph.Graph`:

.. inheritance-diagram:: pgraph.PGraph.UGraph pgraph.PGraph.DGraph
   :parts: 1
   :private-bases:
   :top-classes: pgraph.PGraph.Graph

The vertices of both kinds of graph are :class:`~pgraph.PGraph.Vertex` objects.  A vertex has
no direction of its own: the graph it belongs to decides how an edge is recorded
in the edge lists of its vertices.
Edges are represented by the :class:`~pgraph.PGraph.Edge` class, which is used by both directed and undirected graphs.

The graph classes share almost their entire interface, defined once on their
common base class :class:`~pgraph.PGraph.Graph`. This section documents that shared interface
by task rather than by class; see :doc:`construction` for the :class:`~pgraph.PGraph.DGraph`
and :class:`~pgraph.PGraph.UGraph` classes and :doc:`edges_vertices` for vertices and edges.



The following sections describe the common methods.

.. toctree::
   :maxdepth: 2

   construction
   modification
   analysis
   representations
   visualization
   export
   planning
