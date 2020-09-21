.. pgraph documentation master file, created by
   sphinx-quickstart on Mon Sep 21 18:39:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Graphs for Python
=================

This package provides a set of classes for manipulating simple directed and undirected graphs in Python.

Undirected graphs
-----------------

.. automodule:: PGraph
   :members: UGraph
   :exclude-members: PGraph, DGraph, Vertex, DVertex, UVertex, Edge
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. automodule:: PGraph
   :members: UVertex
   :exclude-members: PGraph, DGraph, Vertex, DVertex, UGraph, Edge
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. automodule:: PGraph
   :members: Edge
   :exclude-members: PGraph, DGraph, Vertex, DVertex, UGraph, UVertex
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Directed graphs
---------------

.. automodule:: PGraph
   :members: DGraph
   :exclude-members: PGraph, UGraph, Vertex, DVertex, UVertex, Edge
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. automodule:: PGraph
   :members: DVertex
   :exclude-members: PGraph, DGraph, Vertex, UVertex, UGraph, Edge
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. automodule:: PGraph
   :members: Edge
   :exclude-members: PGraph, DGraph, Vertex, DVertex, UGraph, UVertex
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
