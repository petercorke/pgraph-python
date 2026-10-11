Common methods
===============

The package supports both directed (prefix ``D``) and undirected graphs (prefix ``U``), and the classes for these
share a common interface. 

The graph classes represent a single graph, either directed or undirected, and they share :class:`_BaseGraph`:

.. inheritance-diagram:: pgraph.PGraph.UGraph pgraph.PGraph.DGraph
   :parts: 1
   :private-bases:
   :top-classes: pgraph.PGraph._BaseGraph

The vertices in a directed graph (``DGraph``) inherit from :class:`DVertex` and the
vertices in an undirected graph (``UGraph``) inherit from :class:`UVertex`. 

.. inheritance-diagram:: pgraph.PGraph.UVertex pgraph.PGraph.DVertex
   :parts: 1

The vertex classes share
almost their entire interface, defined once on their common base classes
(:class:`_BaseGraph`, :class:`BaseVertex`). This section documents that
shared interface by task rather than by class; see :doc:`construction` for
the :class:`DGraph` and :class:`UGraph` classes and :doc:`edges_vertices` for
what's specific to directed vs. undirected vertices.

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
