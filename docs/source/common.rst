Common methods
===============

The package supports both directed (prefix ``D``) and undirected graphs (prefix ``U``), and the classes for these
share a common interface. 

The graph classes represent a single graph, either directed or undirected, and they share :class:`Graph`:

.. inheritance-diagram:: pgraph.PGraph.UGraph pgraph.PGraph.DGraph
   :parts: 1
   :private-bases:
   :top-classes: pgraph.PGraph.Graph

The vertices of both kinds of graph are :class:`Vertex` objects: a vertex has
no direction of its own, the graph it belongs to decides how an edge is
recorded at its ends.

The graph classes share almost their entire interface, defined once on their
common base class :class:`Graph`. This section documents that shared interface
by task rather than by class; see :doc:`construction` for the :class:`DGraph`
and :class:`UGraph` classes and :doc:`edges_vertices` for vertices and edges.

Edges are represented by the :class:`Edge` class, which is used by both directed and undirected graphs.

.. inheritance-diagram:: pgraph.PGraph.Edge
   :parts: 1



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
