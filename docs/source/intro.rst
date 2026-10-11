Introduction
=================

This package provides classes for directed and undirected graphs: vertices
connected by edges, with embedded coordinates, edge costs and path planning.
Self-loops and parallel edges are permitted. It is a graph-theory package, not
a plotting library.

What is a graph?
-----------------

A graph is an abstract representation of a set of objects connected by links and
depicted visually as shown below. Graphs can be
used to represent transport, communications or social networks, and this branch of
mathematics is graph theory.

Mathematically, a graph is denoted by
:math:`G(V, E)` where :math:`V` are the vertices or nodes, and :math:`E` are the links that connect pairs
of vertices and are called edges or arcs. Edges can be directed (shown as arrows) or
undirected (shown as line segments) as in this case. Edges can have an associated
weight or cost associated with moving from one vertex to another. A sequence of
edges from one vertex to another is a path, and a sequence that starts and ends at
the same vertex is a cycle. An edge from a vertex to itself is a loop. 

Creating a graph using the ``pgraph`` package
----------------------------------------------

To create a new graph using the ``pgraph`` package, we first import the package and create
a new instance of the ``UGraph`` class, which represents an undirected graph. By default, the vertices of the graph exist in 2-dimensional space. 

.. runblock:: pycon
   :session:

   >>> import pgraph
   >>> import numpy as np
   >>> g = pgraph.UGraph()

Next, we add some vertices to the graph, each with a random 2D coordinate which is
specified as a 2-element array. The coordinates can be used to compute the cost of
edges, and to plot the graph.

.. runblock:: pycon
   :session:

   >>> np.random.seed(0) # ensure repeatable results
   >>> for i in range(5):
   ...     g.add_vertex(np.random.rand(2))


The method ``add_vertex`` returns a reference to the newly created vertex object, but
we can obtain the reference by indexing the graph object. For example vertex 1 is

.. runblock:: pycon
   :session:

   >>> g[1]

and is an instance of ``UVertex``, a vertex in an undirected graph; it has the coordinate
:math:`(0.6028, 0.5449)` and has been assigned a default name ``"#1"`` but we could specify
an explicit name to ``add_vertex``.
We can also reference a vertex by its name

.. runblock:: pycon
   :session:

   >>> g["#1"]

A summary of the graph can be displayed

.. runblock:: pycon
   :session:

   >>> print(g)

and we see that it has 5 vertices, no edges and 5 components, that is, each vertex is isolated. 

Next, we add edges to connect specific pairs of vertices

 .. runblock:: pycon
   :session:

   >>> g.add_edge(g[0], g[1])
   >>> g.add_edge(g[0], g[2])
   >>> g.add_edge(g[0], g[3])
   >>> g.add_edge(g[1], g[2])
   >>> g.add_edge(g[1], g[3])
   >>> g.add_edge(g[3], g[4])


The ``add_edge`` method returns a reference to the edge object just created. By
default, the edge cost is the Euclidean distance between the vertices, but this can
be overridden by a third argument to ``add_edge``.

A summary of the graph can be displayed

 .. runblock:: pycon
   :session:

   >>> print(g)

and we see that it has 5 vertices, six edges and one component, that is, all the vertices are connected into one net-
work. 

The graph can be plotted by

.. code-block:: python

   >>> g.plot()

.. plot::

   >>> import pgraph
   >>> import numpy as np
   >>> g = pgraph.UGraph()
   >>> np.random.seed(0) # ensure repeatable results
   >>> for i in range(5):
   >>>   g.add_vertex(np.random.rand(2))
   >>>
   >>> g.add_edge(g[0], g[1])
   >>> g.add_edge(g[0], g[2])
   >>> g.add_edge(g[0], g[3])
   >>> g.add_edge(g[1], g[2])
   >>> g.add_edge(g[1], g[3])
   >>> g.add_edge(g[3], g[4])
   >>> g.plot()
   
The vertices are shown as blue circles, and the edges are
shown as lines joining vertices. Many options exist to change default plotting be-
havior. Note that only graphs embedded in 2- and 3-dimensional space can be
plotted.

The neighbours of vertex 1 are

 .. runblock:: pycon
   :session:

   >>> g[1].adjacent()


which is a list of neighboring vertex objects, those connected to it by an edge. Each
edge is an object and the edges connecting to vertex 1 are

 .. runblock:: pycon
   :session:

   >>> g[1].edges()

and each edge has references to the vertices it connects

 .. runblock:: pycon
   :session:

   >>> g[1].edges()[0].endpoints


The cost or length of an edge is

 .. runblock:: pycon
   :session:

   >>> g[1].edges()[0].cost

The vertex closest to the coordinate :math:`(0.5, 0.5)` is

 .. runblock:: pycon
   :session:

   >>> g.closest((0.5, 0.5))

The minimum-distance path between any two vertices in the graph can be computed using well-known algorithms such as A (Hart et al. 1968)

 .. runblock:: pycon
   :session:

   >>> path, length, _ = g.path_Astar(g[2], g[4])
   >>> path


which is a list of vertices representing a total path length of

 .. runblock:: pycon
   :session:

   >>> length

Graphs can also be represented in other ways, such as adjacency, incidence, degree and Laplacian matrices, and
methods exist to convert the graph to these.

Arbitrary data can be attached to any vertex or edge by assigning values to attributes

 .. runblock:: pycon
   :session:

   >>> g[1].data = "this is vertex 1"
   >>> g[1].edges()[0].data = "this is the edge from vertex 0 to vertex 1"

This exploits the flexible nature of Python, but care must be taken to avoid name clashes with existing attributes.

A more robust and Pythonic approach is to create a subclass of the ``UVertex`` or ``DVertex`` classes class and
declare them at graph creation time. For example, we can create a subclass of ``UVertex`` called ``MyVertex`` and a subclass of ``Edge`` called ``MyEdge`` and then create a new graph using these classes.

.. runblock:: pycon
   :session:

   >>> import pgraph
   >>> import numpy as np
   >>> class MyVertex(pgraph.UVertex):
   ...     pass
   >>> class MyEdge(pgraph.Edge):
   ...     pass
   >>> g = pgraph.UGraph(vertex_class=MyVertex, edge_class=MyEdge)
   >>> np.random.seed(0) # ensure repeatable results
   >>> for i in range(5):
   ...     v = g.add_vertex(np.random.rand(2))
   >>> g.add_edge(g[0], g[1])
   >>> g.add_edge(g[0], g[2])
   >>> g.add_edge(g[0], g[3])
   >>> g.add_edge(g[1], g[2])
   >>> g.add_edge(g[1], g[3])
   >>> g.add_edge(g[3], g[4])
   >>> print(g)

we can see that the type of the vertices and edges is now the user-defined classes

.. runblock:: pycon
   :session:

   >>> type(g[0])
   >>> type(g[0].edges()[0])

History
--------

``pgaph-python`` is based on ``PGraph.m`` which is part of the
earlier `Spatial Math Toolbox for MATLAB <https://github.com/petercorke/spatialmath-matlab>`_.
