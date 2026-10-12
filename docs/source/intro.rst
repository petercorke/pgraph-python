Introduction
=================

This pure-Python package provides classes for directed and undirected graphs: vertices
connected by edges, with edge costs and path planning.  Self-loops and
parallel edges are permitted.

Vertices can optionally be *embedded*: given a coordinate in a space of any
dimension.  Edge costs are then computed from the vertex positions using a
distance metric (Euclidean, Manhattan, SE(2) with a weight on the angle, or your own), the vertex nearest
a point can be found, ``A*`` uses the metric as its heuristic, and the graph can
be plotted, or exported with its layout preserved.

What is a graph?
-----------------

A graph is an abstract representation of a set of objects connected by links and
depicted visually as shown below. Graphs can be
used to represent transport, communications or social networks, and this branch of
mathematics is graph theory.

.. plot::
   :include-source: no
   :alt: two small graphs on the same five vertices, one undirected and one directed, each with parallel edges and a loop

   import matplotlib.pyplot as plt
   from matplotlib.patches import Circle, FancyArrowPatch

   pos = {"A": (0.0, 1.0), "B": (2.0, 1.7), "C": (3.3, 0.3), "D": (1.2, -0.5), "E": (4.9, 1.3)}
   # (from, to, curvature): two edges between the same vertices are drawn as
   # two arcs, curving to either side
   undirected = [("A", "B", 0), ("A", "D", 0), ("B", "D", 0.28), ("B", "D", -0.28),
                 ("B", "C", 0), ("D", "C", 0), ("C", "E", 0)]
   directed = [("A", "B", 0.28), ("A", "B", -0.28), ("A", "D", 0), ("B", "D", 0),
               ("B", "C", 0.28), ("C", "B", 0.28), ("D", "C", 0), ("C", "E", 0)]

   fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
   for ax, edges, is_directed, title in zip(
       axs, (undirected, directed), (False, True), ("Undirected graph", "Directed graph")
   ):
       style = "-|>" if is_directed else "-"
       for a, b, rad in edges:
           ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle=style, mutation_scale=18,
                                        connectionstyle=f"arc3,rad={rad}", shrinkA=17, shrinkB=17,
                                        color="0.15", lw=1.8))
       x, y = pos["E"]  # a loop: an edge from a vertex to itself
       ax.add_patch(FancyArrowPatch((x - 0.2, y + 0.17), (x + 0.2, y + 0.17), arrowstyle=style,
                                    mutation_scale=18, connectionstyle="arc3,rad=-1.8",
                                    color="0.15", lw=1.8))
       for name, (x, y) in pos.items():
           ax.add_patch(Circle((x, y), 0.2, fc="#1f77b4", ec="none", zorder=3))
           ax.text(x, y, name, ha="center", va="center", color="white", fontsize=11, zorder=4)
       ax.set_xlim(-0.6, 5.7)
       ax.set_ylim(-1.0, 2.6)
       ax.set_aspect("equal")
       ax.axis("off")
       ax.set_title(title, fontsize=11)
   fig.tight_layout()

Mathematically, a graph is denoted by
:math:`G(V, E)` where :math:`V` are the vertices or nodes, and :math:`E` are the links that connect pairs
of vertices and are called edges or arcs. Edges can be directed (shown as arrows, as in the graph
on the right) or undirected (shown as line segments, as in the graph on the left). Edges can have an associated
weight or cost associated with moving from one vertex to another. A sequence of
edges from one vertex to another is a path, and a sequence that starts and ends at
the same vertex is a cycle. An edge from a vertex to itself is a loop, and edges that join the same pair
of vertices are parallel edges: B and D on the left are joined by two edges, as are A and B on the right.
On the right, B and C are also joined by edges in opposite directions.

Creating a graph using the ``pgraph`` package
----------------------------------------------

To create a new graph using the ``pgraph`` package, we first import the package and create
a new instance of the ``UGraph`` class, which represents an undirected graph with vertices that exist in 2-dimensional space. 

.. runblock:: pycon
   :session:

   >>> from pgraph import UGraph
   >>> import numpy as np
   >>> g = UGraph(dim=2)

Next, we add some vertices to the graph, each with a random 2D coordinate which is
specified as a 2-element array. The coordinates can be used to compute the cost of
edges, and to plot the graph.

.. runblock:: pycon
   :session:

   >>> np.random.seed(0) # ensure repeatable results
   >>> for i in range(5):
   ...     g.add_vertex(np.random.rand(2))


The method ``add_vertex`` returns a reference to the newly created vertex object, and
a vertex can also be obtained by indexing the graph object. For example vertex 1 is

.. runblock:: pycon
   :session:

   >>> g[1]

and is an instance of ``Vertex``; it has the coordinate
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

and we see that it has 5 vertices, six edges and one component, that is, all the vertices are connected into one
network.

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
shown as lines joining vertices. Many options exist to change default plotting
behavior. Note that only graphs embedded in 2- and 3-dimensional space can be
plotted.

The vertices adjacent to vertex 1 are

 .. runblock:: pycon
   :session:

   >>> g[1].adjacent()


which is a list of adjacent vertex objects, those connected to it by an edge. Each
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

The minimum-distance path between any two vertices in the graph can be computed using well-known algorithms such as ``A*`` (Hart et al. 1968)

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

A more robust and Pythonic approach is to create a subclass of the ``Vertex`` class and
pass them to the graph constructor. For example, we can create a subclass of ``Vertex`` called ``MyVertex`` and a subclass of ``Edge`` called ``MyEdge`` and then create a new graph using these classes.

.. runblock:: pycon
   :session:

   >>> import pgraph
   >>> import numpy as np
   >>> class MyVertex(pgraph.Vertex):
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


Creating directed and undirected graphs
----------------------------------------

The example above illustrates an undirected graph, but a directed graph can be created by using
the ``DGraph`` class instead of ``UGraph``. The methods are the same, but the edges are
directed and the path-finding algorithms take this into account.

.. runblock:: pycon

   >>> from pgraph import DGraph
   >>> g = DGraph(dim=2)
   >>> type(g)

Alternatively, we can make the decision at runtime by using the ``directed`` argument to the ``Graph`` constructor

.. runblock:: pycon

   >>> from pgraph import Graph
   >>> g = Graph(directed=True, dim=2)
   >>> type(g)
   >>> g = Graph(directed=False, dim=2)
   >>> type(g)

History
--------

``pgaph-python`` is based on ``PGraph.m`` which is part of the
earlier `Spatial Math Toolbox for MATLAB <https://github.com/petercorke/spatialmath-matlab>`_.
