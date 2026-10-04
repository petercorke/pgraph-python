# PGraph: graphs for Python

<div align="center">
  <strong>Mathematical graphs for Python</strong>
  <br><br>

[![PyPI version](https://img.shields.io/pypi/v/pgraph-python?style=for-the-badge&color=blue)](https://pypi.org/project/pgraph-python/)
  [![Documentation](https://img.shields.io/badge/Docs-View_Online-blue?style=for-the-badge)](https://petercorke.github.io/pgraph-python)

  <p>
    <a href="https://github.com/petercorke/pgraph-python">GitHub</a> •
    <a href="https://github.com/petercorke/pgraph-python/wiki">Wiki</a> •
    <a href="https://github.com/petercorke/pgraph-python/blob/main/CHANGELOG.md">Changelog</a>
  </p>
</div>

---

### Status & Project Health
[![Build Status](https://github.com/petercorke/pgraph-python/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/petercorke/pgraph-python/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/pgraph-python/month)](https://pepy.tech/projects/pgraph-python)
![Python Version](https://img.shields.io/pypi/pyversions/pgraph-python.svg)
[![Coverage](https://codecov.io/gh/petercorke/pgraph-python/branch/main/graph/badge.svg)](https://codecov.io/gh/petercorke/pgraph-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Ecosystem & Dependencies
[![A Python Robotics Package](https://raw.githubusercontent.com/petercorke/robotics-toolbox-python/main/.github/svg/py_collection.min.svg)](https://github.com/petercorke/robotics-toolbox-python)
[![QUT Centre for Robotics Open Source](https://github.com/qcr/qcr.github.io/raw/master/misc/badge.svg)](https://qcr.github.io)

[![powered by NumPy](https://img.shields.io/badge/powered_by-NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![powered by SciPy](https://img.shields.io/badge/powered_by-SciPy-0054a6?logo=scipy&logoColor=white)](https://scipy.org)
[![powered by Matplotlib](https://img.shields.io/badge/powered_by-Matplotlib-11557c?logo=matplotlib&logoColor=white)](https://matplotlib.org)
[![Powered by Spatial Maths](https://raw.githubusercontent.com/petercorke/spatialmath-python/master/.github/svg/sm_powered.min.svg)](https://github.com/petercorke/spatialmath-python)

This Python package allows the manipulation of directed and non-directed graphs.  Also supports embedded graphs.  It is suitable for graphs with thousands of nodes.

![road network](https://github.com/petercorke/pgraph-python/raw/main/examples/roads.png)

```
from pgraph import *
import json

# load places and routes
with open('places.json', 'r') as f:
    places = json.loads(f.read())
with open('routes.json', 'r') as f:
    routes = json.loads(f.read())

# build the graph
g = UGraph()

for name, info in places.items():
    g.add_vertex(name=name, coord=info["utm"])

for route in routes:
    g.add_edge(route[0], route[1], cost=route[2])

# plan a path from Hughenden to Brisbane
p = g.path_Astar('Hughenden', 'Brisbane')
g.plot(block=False) # plot it
g.highlight_path(p)  # overlay the path
```

### Properties and methods of the graph
Graphs belong to the class `UGraph` or `DGraph` for undirected or directed graphs respectively.  The graph is essentially a container for the vertices.

- `g.add_vertex()` add a vertex
- `g.n` the number of vertices
- `g` is an iterator over vertices, can be used as `for vertex in g:`
- `g[i]` reference a vertex by its index or name

    ***
- `g.add_edge()` connect two vertices
- `g.edges()` all edges in the graph
- `g.plot()` plots the vertices and edges
- `g.nc` the number of graph components, 1 if fully connected
- `g.component(v)` the component that vertex `v` belongs to

    ***
- `g.path_BFS()` breadth-first search
- `g.path_Astar()` A* search

    ***
- `g.adjacency()` adjacency matrix
- `g.Laplacian()` Laplacian matrix
- `g.incidence()` incidence matrix

### Properties and methods of a vertex
Vertices belong to the class `UVertex` (for undirected graphs) or `DVertex` (for directed graphs), which are each subclasses of `Vertex`.

- `v.coord` the coordinate vector for embedded graph (optional)
- `v.name` the name of the vertex (optional)
- `v.neighbours()` is a list of the neighbouring vertices
- `v1.samecomponent(v2)` predicate for vertices belonging to the same component

Vertices can be named and referenced by name.

### Properties and methods of an edge
Edges are instances of the class `Edge`.
Edges are not referenced by the graph object, each edge references a pair of vertices, and the vertices reference the edges.  For a directed graph only the start vertex of an edge references the edge object, whereas for an undirected graph both vertices reference the edge object.

- `e.cost` cost of edge for planning methods
- `e.next(v)` vertex on edge `e` that is not `v`
- `e.v1`, `e.v2` the two vertices that define the edge `e`

## Modifying a graph

- `g.remove(v)` remove vertex `v`
- `e.remove()` remove edge `e`

## Subclasing pgraph classes

Consider a user class `Foo` that we would like to connect using a graph _overlay_, ie.
instances of `Foo` becomes vertices in a graph.

- Have it subclass either `DVertex` or `UVertex` depending on graph type
- Then place instances of `Foo` into the graph using `add_vertex` and create edges as required

```
class Foo(UVertex):
  # foo stuff goes here
  
f1 = Foo(...)
f2 = Foo(...)

g = UGraph() # create a new undirected graph
g.add_vertex(f1)
g.add_vertex(f2)

f1.connect(f2, cost=3)
for f in f1.neighbours():
    # say hi to the neighbours
```

## Under the hood

The key objects and their interactions are shown below.

![data structures](https://github.com/petercorke/pgraph-python/raw/main/docs/source/datastructures.png)

## MATLAB version

This is a re-engineered version of [PGraph.m](https://github.com/petercorke/spatialmath-matlab/blob/main/PGraph.m) which ships as part of the [Spatial Math Toolbox for MATLAB](https://github.com/petercorke/spatialmath-matlab).  This class is used to support bundle adjustment, pose-graph SLAM and various planners such as PRM, RRT and Lattice.

The Python version was designed from the start to work with directed and undirected graphs, whereas directed graphs were a late addition to the MATLAB version.  Semantics are similar but not identical.  In particular the use of subclassing rather than references to
_user data_ is encouraged.





