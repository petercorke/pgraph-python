Edges & Vertices
=================

Underlying data structures
----------------------------

.. image:: ../figs/datastructures.svg

The graph class maintains a list of vertices and a list of edges. Each vertex maintains
a list of edges that connect to it, and each edge maintains a reference to the two
vertices it connects. The graph class provides methods for adding and removing vertices
and edges, as well as for querying the graph's structure.

Working with vertices and edges directly
-----------------------------------------

In the earlier examples we did not explicitly create vertices or edges, but we
can do so if we want to. The graph classes provide methods to create vertices and edges,
and to add them to the graph.


.. runblock:: pycon
   :session:

   >>> from pgraph import UGraph, UVertex, Edge
   >>> import numpy as np
   >>> g = UGraph()

Next, we will explicitly create two vertices

.. runblock:: pycon
   :session:

   >>> v1 = UVertex(np.random.rand(2))
   >>> v2 = UVertex(np.random.rand(2))


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

The ``Edge`` constuctor line above is equivalent to 

.. code-block:: python

   >>> e = Edge()
   >>> e.connect(v1, v2)

Alternatively, we can create a vertex and *connect* it to an existing vertex, which will implitically create an edge between the two vertices and add it to the graph.

.. runblock:: pycon
   :session:

   >>> v3 = UVertex(np.random.rand(2))
   >>> g.add_vertex(v3)
   >>> v2.connect(v3)
   >>> print(g)

The ``connect`` method of the vertex class creates an edge between the two
vertices and adds it to the graph. The two vertices must have been added  the same graph for the edge to be created.
The ``connect`` method is available on both the
directed and undirected vertex classes.


.. note:: Where ever a vertex or edge is required, a subclass of the appropriate base class can be used. For example, we can create a subclass of ``UVertex`` called ``MyVertex`` and a subclass of ``Edge`` called ``MyEdge`` and then create a new graph using these classes.

.. toctree::
   :maxdepth: 2

   dvertex
   uvertex
   edge
