Edges & Vertices
=================

Underlying data structures
----------------------------

.. image:: ../figs/datastructures.svg

The graph class maintains a list of vertices and a dict of edges. For the undirected case,
each vertex maintains
a list of edges that connect to it, and each edge maintains a reference to the two
vertices it connects. In the case of a loop, the edge appears twice in the vertex's edge list.
For the directed case, each vertex maintains two edge lists: a list of outgoing edges and a list of incoming edges.

The graph class provides methods for adding and removing vertices and edges (see :doc:`modification`)
and for querying its structure (see :doc:`analysis`).


Working with vertices and edges directly
-----------------------------------------

In the earlier examples we did not explicitly create vertices or edges, but we
can do so if we want to. The graph classes provide methods to create vertices and edges,
and to add them to the graph.


.. runblock:: pycon
   :session:

   >>> from pgraph import UGraph, Vertex, Edge
   >>> import numpy as np
   >>> g = UGraph()

Next, we will explicitly create two vertices

.. runblock:: pycon
   :session:

   >>> v1 = Vertex(np.random.rand(2))
   >>> v2 = Vertex(np.random.rand(2))


and then add them to the graph

.. runblock:: pycon
   :session:

   >>> g.add_vertex(v1)
   >>> g.add_vertex(v2)

Our graph now has two vertices, but no edges

.. runblock:: pycon
   :session:

   >>> print(g)

We can explicitly create an edge between the two vertices

.. runblock:: pycon
   :session:

   >>> e = Edge(v1, v2)

and then add it to the graph

.. runblock:: pycon
   :session:

   >>> g.add_edge(e)
   >>> print(g)

Creating the edge and then adding it to the graph, as above, is equivalent to

.. code-block:: python

   >>> e = Edge()
   >>> e.connect(v1, v2)

Alternatively, we can create a vertex and *connect* it to an existing vertex, which will implicitly create an edge between the two vertices and add it to the graph.

.. runblock:: pycon
   :session:

   >>> v3 = Vertex(np.random.rand(2))
   >>> g.add_vertex(v3)
   >>> v2.connect(v3)
   >>> print(g)

The ``connect`` method of the vertex class creates an edge between the two
vertices and adds it to the graph. The two vertices must have been added to the same graph for the edge to be created.
The ``connect`` method is the same for directed and undirected graphs: the graph decides how the edge is
recorded at its vertices.


.. note:: Wherever a vertex or edge is required, a subclass of the appropriate base class can be used. For example, we can create a subclass of ``Vertex`` called ``MyVertex`` and a subclass of ``Edge`` called ``MyEdge`` and then create a new graph using these classes, by passing them as the ``vertex_class`` and ``edge_class`` arguments of the graph constructor.


The Vertex class
-----------------

.. autoclass:: pgraph.PGraph.Vertex
   :show-inheritance:
   :special-members: __init__

The Edge class
-----------------

.. autoclass:: pgraph.PGraph.Edge
   :show-inheritance:
   :members:
   :special-members: __init__

