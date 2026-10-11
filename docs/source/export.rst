Exporting a graph
=================

Graphs can be exported to a number of different formats, including dot (used by
`GraphViz <https://graphviz.org>`_) mermaid (a JavaScript-based graph
visualization tool that can be used in MarkDown documents), GraphML (an
XML-based format for representing graphs), ELK (a graph
layout engine), as well as various image formats (e.g., PNG, SVG, PDF).

See :doc:`visualization` for rendering the graph using Matplotlib itself rather than exporting
it.

.. autosummary::
   :nosignatures:

   ~pgraph.PGraph.BaseGraph.export
   ~pgraph.PGraph.BaseGraph.dotfile

Details
-------

.. automethod:: pgraph.PGraph.BaseGraph.export
.. automethod:: pgraph.PGraph.BaseGraph.dotfile
