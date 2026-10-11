Adding and removing vertices and edges
========================================

Building up and tearing down a graph's structure. ``connect()`` is shown
here in its base-class form; :doc:`DVertex <dvertex>` and
:doc:`UVertex <uvertex>` override it with different edge-list bookkeeping
-- see those pages for the concrete behaviour. See :class:`Edge` for
constructing/removing an edge directly rather than via a vertex or graph.

Graph
-----

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph._BaseGraph.add_vertex
   ~pgraph.PGraph._BaseGraph.add_edge
   ~pgraph.PGraph._BaseGraph.remove_edge
   ~pgraph.PGraph._BaseGraph.remove_vertex

Vertex
------

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.BaseVertex.connect
   ~pgraph.PGraph.BaseVertex.edgeto
   ~pgraph.PGraph.BaseVertex.remove

Details
-------

.. automethod:: pgraph.PGraph._BaseGraph.add_vertex
.. automethod:: pgraph.PGraph._BaseGraph.add_edge
.. automethod:: pgraph.PGraph._BaseGraph.remove_edge
.. automethod:: pgraph.PGraph._BaseGraph.remove_vertex
.. automethod:: pgraph.PGraph._BaseGraph.remove

.. automethod:: pgraph.PGraph.BaseVertex.connect
.. automethod:: pgraph.PGraph.BaseVertex.edgeto
.. automethod:: pgraph.PGraph.BaseVertex.remove
