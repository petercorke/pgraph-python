Exporting a graph
=================

Graphs can be exported to a number of different formats, including dot (used by
`GraphViz <https://graphviz.org>`_), Mermaid (a JavaScript-based graph
visualization tool that can be used in Markdown documents), GraphML (an
XML-based format for representing graphs), ELK (a graph
layout engine), as well as various image formats (e.g., PNG, SVG, PDF).

See :doc:`visualization` for rendering the graph using Matplotlib itself rather than exporting
it.

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.Graph.export
   ~pgraph.PGraph.Graph.dotfile

Details
-------

.. automethod:: pgraph.PGraph.Graph.export
.. automethod:: pgraph.PGraph.Graph.dotfile

The exporters
-------------

:meth:`~pgraph.PGraph.Graph.export` hands the work to the functions of
:mod:`pgraph.exporters`.  The options of each format are the keyword arguments
of its function, which are passed through ``export(..., **options)``.  The
module also provides the means to add a format of your own.

.. automodule:: pgraph.exporters
   :members: formats, format_from_filename, register_exporter, get_exporter, dot, mermaid, graphml, elk, render, ENGINES, RENDER_FORMATS
   :member-order: bysource
