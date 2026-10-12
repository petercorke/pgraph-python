Visualization
==============

Rendering the graph itself, with matplotlib or via GraphViz.  See
:doc:`export` for exporting the graph as data or an image file (dot, Mermaid,
GraphML, ELK, PNG, ...) and :doc:`representations` for its matrices.

:meth:`~pgraph.PGraph.Graph.plot` draws each edge once.  A directed graph has an
arrowhead at the end of each edge, edges that join the same pair of vertices
are drawn as arcs that bow apart so that parallel and antiparallel edges can all
be seen, and a loop is drawn as a circle at its vertex.  The same is true of a
3D plot, which follows a rotation of the view.

The appearance is set by dicts of Matplotlib options:

=================  ====================================================  =============================
option             applies to                                            default
=================  ====================================================  =============================
``vertex_opt``     the vertex marker (``plot``)                          ``marker="o", markersize=12``
``edge_opt``       the edge line (``FancyArrowPatch``): ``color``,       ``linewidth=3``
                   ``linewidth``, ``linestyle``, ``alpha``
``arrow_opt``      the arrowhead and the path: ``arrowstyle``,           ``arrowstyle="-|>",``
                   ``mutation_scale``, ``connectionstyle``               ``mutation_scale=15`` for a
                                                                         directed graph, ``"-"``
                                                                         for an undirected one
``text_opt``       the vertex labels (``text``)                          Matplotlib's
=================  ====================================================  =============================

``labels=False`` removes the labels, and ``color_components=False`` draws the
graph in one color rather than a color for each component.

.. note:: The options were named ``vopt``, ``eopt``, ``text``, ``colorcomponents``
   and ``force2d`` in 0.6.5.  Those names still work but give a
   ``DeprecationWarning``.

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Graph.plot
   ~pgraph.PGraph.Graph.showgraph
   ~pgraph.PGraph.Graph.highlight_path
   ~pgraph.PGraph.Graph.highlight_edge
   ~pgraph.PGraph.Graph.highlight_vertex

Details
-------

.. automethod:: pgraph.PGraph.Graph.plot
.. automethod:: pgraph.PGraph.Graph.showgraph
.. automethod:: pgraph.PGraph.Graph.highlight_path
.. automethod:: pgraph.PGraph.Graph.highlight_edge
.. automethod:: pgraph.PGraph.Graph.highlight_vertex
