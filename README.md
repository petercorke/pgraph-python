# PGraph: simple graphs for Python

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![pypi downloads](https://img.shields.io/pypi/dw/pgraph-python)
[![PyPI version](https://badge.fury.io/py/pgraph-python.svg)](https://badge.fury.io/py/spatialmath-python)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/petercorke/pgraph-python/graphs/commit-activity)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/petercorke/pgraph-python/blob/master/LICENSE)

- GitHub repository: [https://github.com/petercorke/pgraph-python](https://github.com/petercorke/pgraph-python)
- Documentation: [https://petercorke.github.io/pgraph-python](https://petercorke.github.io/pgraph-python)
- Dependencies: `numpy`


This Python package allows the manipulation of directed and non-directed graphs.  Also supports embedded graphs.

![road network](https://github.com/petercorke/pgraph-python/raw/master/examples/roads.png)

```
from pgraph import *

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
Graphs belong to the class `UGraph` or `DGraph` for undirected or directed graphs respectively

- `g.add_vertex()` add a vertex
- `g.add_edge()` connect two vertices
- `g.n` the number of vertices
- supports iteration: `for vertex in graph:`
- `g.edges()` all edges in the graph
- `g[i]` reference a vertex by its index or name
- `g.nc` the number of graph components, 1 if fully connected
- `g.component(v)` the component that vertex `v` belongs to
- `g.plot()` plots the vertices and edges
- `g.path_BFS()` breadth-first search
- `g.path_Astar()` A* search
- `g.adjacency()` adjacency matrix
- `g.Laplacian()` Laplacian matrix
- `g.incidence()` incidence matrix

### Properties and methods of a vertex
Vertices belong to the class `UVertex` or `DVertex` for undirected or directed graphs respectively

- `v.coord` the coordinate vector for embedded graph
- `v.name` the name of the vertex
- access the neighbours of a vertex by `v.neighbours()`.

We can name the vertices and reference them by name

### Properties and methods of an edge
- `e.cost` cost of edge for planning methods
- `e.next(v)` vertex on edge `e` that is not `v`
- `e.v1`, `e.v2` the two vertices that define the edge `e`

## Modifying a graph

- `g.remove(v)` remove vertex `v`
- `e.remove()` remove edge `e`

## Inheritance

Consider a user class `Foo` that we would like to represent vertices in a graph.

- Have it subclass either `DVertex` or `UVertex`
- Then place instances of `Foo` into the graph using `add_vertex`




