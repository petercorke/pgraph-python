# PGraph: graphs for Python

<div align="center">
  <img src="https://raw.githubusercontent.com/petercorke/pgraph-python/main/docs/figs/pgraph_logo.png" width="450" alt="pgraph logo">
  <br>
  <strong>Graphs for Python</strong>
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
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/728d19eca33946fa8cfe5ae536c52bc0)](https://app.codacy.com/gh/petercorke/pgraph-python/dashboard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Ecosystem & Dependencies
[![A Python Robotics Package](https://raw.githubusercontent.com/petercorke/robotics-toolbox-python/main/.github/svg/py_collection.min.svg)](https://github.com/petercorke/robotics-toolbox-python)
[![QUT Centre for Robotics Open Source](https://github.com/qcr/qcr.github.io/raw/master/misc/badge.svg)](https://qcr.github.io)

[![powered by NumPy](https://img.shields.io/badge/powered_by-NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![powered by Matplotlib](https://img.shields.io/badge/powered_by-Matplotlib-11557c?logo=matplotlib&logoColor=white)](https://matplotlib.org)
[![Powered by Spatial Maths](https://raw.githubusercontent.com/petercorke/spatialmath-python/master/.github/svg/sm_powered.min.svg)](https://github.com/petercorke/spatialmath-python)

This pure-Python package provides classes for directed and undirected graphs: vertices
connected by edges, with edge costs and path planning.  Self-loops and
parallel edges are permitted.

Vertices can optionally be *embedded*: given a coordinate in a space of any
dimension.  Edge costs are then computed from the vertex positions using a
distance metric (Euclidean, Manhattan, SE(2), or your own), the vertex nearest
a point can be found, `A*` uses the metric as its heuristic, and the graph can
be plotted, or exported with its layout preserved.

## Installation

```
pip install pgraph-python
```

This needs Python 3.9 or later.  The dependencies, NumPy, Matplotlib and Spatial Maths, are installed with it.

## Examples

### A small example: routing through towns in Queensland

![Map of 20 towns in Queensland as blue circles joined by roads, with the shortest route from Hughenden to Brisbane, 1,659 km via Barcaldine, Blackall, Charleville and Roma, highlighted in red](https://github.com/petercorke/pgraph-python/raw/main/examples/roads.png)

```python
from pgraph import UGraph
import json

# load the places and routes
with open('queensland.json', 'r') as f:
    data = json.load(f)

# build the graph
g = UGraph()
for name, info in data['places'].items():
    g.add_vertex(name=name, coord=info['utm'])
for route in data['routes']:
    g.add_edge(route['start'], route['end'], cost=route['distance'])

# plan a path from Hughenden to Brisbane
p, length, _ = g.path_Astar('Hughenden', 'Brisbane')
g.plot(block=False)  # plot it
g.highlight_path(p)  # overlay the path
```

The data file and the complete example are in the notebook [`examples/roads.ipynb`](https://github.com/petercorke/pgraph-python/blob/main/examples/roads.ipynb).

### A bigger example: routing through central London

![Map of central London as a network of grey streets, with two routes between Buckingham Palace and Tower Bridge: a solid blue route of 5,515 m via Big Ben, and a dashed orange route of 5,948 m back through the north of the area, with five landmarks labelled](https://github.com/petercorke/pgraph-python/raw/main/docs/figs/london_route.png)

A directed graph of the road network of central London, taken from [OpenStreetMap](https://www.openstreetmap.org/copyright), and the shortest drive between Buckingham Palace and Tower Bridge, there (solid) and back (dashed).  The two routes differ in length, 5,515 m and 5,948 m, because of the one-way streets.  The notebook [`examples/london.ipynb`](https://github.com/petercorke/pgraph-python/blob/main/examples/london.ipynb) builds the graph from the map data, keeps the part of it that can be driven around in, and plans with `A*`.

Map data © OpenStreetMap contributors, available under the [Open Database License](https://www.openstreetmap.org/copyright).

## Using pgraph

Full details are given in the [online documentation](https://petercorke.github.io/pgraph-python).

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
- `g.component(c)` the vertices in component number `c`, numbered from 0
- `g.samecomponent(v1, v2)` predicate for vertices belonging to the same component

    ***
- `g.path_BFS()` breadth-first search, the path with the fewest edges
- `g.path_UCS()` uniform-cost search, the cheapest path
- `g.path_Astar()` A* search, the cheapest path using a heuristic

    ***
- `g.closest(point)` the vertex nearest to a point
- `g.near(point, radius)` the vertices within a distance of a point
- `g.within(box)` the vertices inside a box
- `g.subgraph(vertices)` the part of the graph that those vertices induce, as a new graph
- `g.coordinates()` the vertex coordinates as a matrix, a column per vertex
- `g.build_kdtree()` an optional spatial index (needs scipy) that makes `closest` and `near` faster on a large graph
- `g.metric` the distance metric: `"L2"` (the default), `"L1"`, `"SE2"`, `SE2Metric(weight)` that weights the angle of a pose, or a function of your own

    ***
- `g.adjacency()` adjacency matrix
- `g.laplacian()` Laplacian matrix
- `g.incidence()` incidence matrix

    ***
- `g.copy()` a deep copy; a graph can also be pickled
- `g.export()` write the graph as GraphViz DOT, Mermaid, GraphML, ELK, or an image (which needs GraphViz)
- `g.validate()` check that the graph's records agree with each other

### Properties and methods of a vertex
Vertices are instances of the class `Vertex`, for both undirected and directed graphs.

- `v.coord` the coordinate vector for embedded graph (optional)
- `v.name` the name of the vertex (optional)
- `v.label` the number of the component that the vertex belongs to
- `v.adjacent()` is a list of the adjacent vertices (`neighbours()` and `neighbors()` are the same)

Vertices can be named and referenced by name.

### Properties and methods of an edge
Edges are instances of the class `Edge`.
Each edge references a pair of vertices, the graph lists all of its edges in `g.edges()`, and the vertices reference the edges.

- `e.cost` cost of edge for planning methods
- `e.next(v)` vertex on edge `e` that is not `v`
- `e.v1`, `e.v2` the two vertices that define the edge `e`

### Modifying a graph

- `g.remove_vertex(v)` or `v.remove()` remove vertex `v`
- `g.remove_edge(e)` or `e.remove()` remove edge `e`

### Subclassing `pgraph` classes

Consider a user class `Foo` that we would like to connect using a graph _overlay_, i.e.
instances of `Foo` become vertices in a graph.

- Have it subclass `Vertex`
- Then place instances of `Foo` into the graph using `add_vertex` and create edges as required

```python
from pgraph import UGraph, Vertex

class Foo(Vertex):
    # foo stuff goes here
    def say_hi(self):
        print(f"hi from {self.name}")

f1 = Foo(name="f1")
f2 = Foo(name="f2")

g = UGraph() # create a new undirected graph
g.add_vertex(f1)
g.add_vertex(f2)

f1.connect(f2, cost=3)
for f in f1.adjacent():
    f.say_hi()  # say hi to the neighbours
```

Or tell the graph which classes to make, with `UGraph(vertex_class=Foo, edge_class=Bar)`, and `g.add_vertex()` and `g.add_edge()` create instances of them, so they never need to be constructed by hand.

## Performance

Timings of the key operations on random geometric graphs, each vertex joined to its 4 nearest neighbours, of 100, 1,000 and 10,000 vertices, measured with [`examples/benchmark.py`](https://github.com/petercorke/pgraph-python/blob/main/examples/benchmark.py) on an Apple M4 with Python 3.12 and NumPy 2.5.  The path times are the median of 10 random start and goal pairs.  The last column is the exponent *b* in time ∝ N<sup>*b*</sup> between the smallest and the largest graph for the quantity shown, so 0 is constant, 1 is linear and 2 is quadratic.  The adjacency, Laplacian and incidence matrices are dense, so they are only formed up to 1,000 vertices.

### Undirected graphs

| Operation                            | per         |  N=100 | N=1,000 | N=10,000 | growth exponent |
| :----------------------------------- | :---------- | -----: | ------: | -------: | --------------: |
| add a vertex                         | µs / vertex |   1.52 |    1.65 |     1.59 |             0.0 |
| add an edge (cost from the metric)   | µs / edge   |   1.55 |    1.58 |     1.85 |             0.0 |
| copy the graph                       | ms          |  0.823 |    8.17 |      151 |             1.1 |
| pickle and unpickle the graph        | ms          |  0.456 |    4.76 |      113 |             1.2 |
| number of components, after a change | ms          | 0.0292 |   0.319 |     3.86 |             1.1 |
| number of components, unchanged      | µs          |  0.083 |   0.083 |    0.083 |             0.0 |
| remove a vertex                      | µs          |   6.19 |    11.6 |     56.5 |             0.5 |
| closest vertex to a point            | µs / query  |   69.9 |     708 |    7,056 |             1.0 |
| adjacency matrix                     | ms          |  0.107 |    1.46 |        – |             1.1 |
| Laplacian matrix                     | ms          |  0.121 |    2.76 |        – |             1.4 |
| incidence matrix                     | ms          | 0.0565 |    1.44 |        – |             1.4 |
| path, breadth-first                  | ms / query  |  0.038 |   0.317 |      5.4 |             1.1 |
| path, uniform cost                   | ms / query  | 0.0822 |   0.806 |     11.7 |             1.1 |
| path, A*                             | ms / query  | 0.0498 |   0.295 |     2.48 |             0.8 |

### Directed graphs

| Operation                            | per         |  N=100 | N=1,000 | N=10,000 | growth exponent |
| :----------------------------------- | :---------- | -----: | ------: | -------: | --------------: |
| add a vertex                         | µs / vertex |   1.53 |    1.58 |     1.59 |             0.0 |
| add an edge (cost from the metric)   | µs / edge   |   1.56 |    1.64 |     1.84 |             0.0 |
| copy the graph                       | ms          |   1.12 |    12.1 |      227 |             1.2 |
| pickle and unpickle the graph        | ms          |  0.573 |    6.58 |      151 |             1.2 |
| number of components, after a change | ms          | 0.0405 |   0.485 |     5.87 |             1.1 |
| number of components, unchanged      | µs          |  0.083 |   0.083 |    0.083 |             0.0 |
| remove a vertex                      | µs          |   8.04 |    15.7 |       70 |             0.5 |
| closest vertex to a point            | µs / query  |   70.2 |     711 |    6,921 |             1.0 |
| adjacency matrix                     | ms          | 0.0815 |    1.21 |        – |             1.2 |
| Laplacian matrix                     | ms          | 0.0928 |    2.64 |        – |             1.5 |
| incidence matrix                     | ms          | 0.0871 |    2.41 |        – |             1.4 |
| path, breadth-first                  | ms / query  | 0.0225 |   0.386 |     3.87 |             1.1 |
| path, uniform cost                   | ms / query  | 0.0536 |   0.936 |     8.06 |             1.1 |
| path, A*                             | ms / query  |  0.046 |   0.721 |     3.68 |             1.0 |

## MATLAB version

This is a re-engineered version of [PGraph.m](https://github.com/petercorke/spatialmath-matlab/blob/main/PGraph.m) which ships as part of the [Spatial Math Toolbox for MATLAB](https://github.com/petercorke/spatialmath-matlab).  This class is used to support bundle adjustment, pose-graph SLAM and various planners such as PRM, RRT and Lattice.

The Python version was designed from the start to work with directed and undirected graphs, whereas directed graphs were a late addition to the MATLAB version.  Semantics are similar but not identical.  In particular the use of subclassing rather than references to
_user data_ is encouraged.
