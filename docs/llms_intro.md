pgraph is a pure-Python package of graph classes: directed and undirected graphs of vertices connected by edges, with edge costs and path planning (breadth-first, uniform-cost and A*). Vertices can be *embedded*, given a coordinate in a space of any dimension; edge costs then come from a distance metric (L2, L1, SE2 or your own), and the graph can be queried spatially, plotted and exported. Self-loops and parallel edges are allowed. It requires Python 3.9 or later and depends on NumPy, Matplotlib and spatialmath-python. Install with `pip install pgraph-python`; import it as `pgraph`.

## Quick start

```python
from pgraph import UGraph

g = UGraph(dim=2)                       # DGraph for directed; Graph(directed=flag) chooses
a = g.add_vertex([0, 0], name="a")      # returns the Vertex
b = g.add_vertex([3, 4], name="b")
c = g.add_vertex([6, 0], name="c")
g.add_edge(a, b)                        # cost defaults to the metric distance, 5.0
g.add_edge("b", "c", cost=7)            # vertices or their names; or give the cost

path, length, _ = g.path_Astar("a", "c")        # None if there is no path
print([v.name for v in path], length)           # ['a', 'b', 'c'] 12.0
g.plot()
g.highlight_path(path)
```

## Key points

- **Classes.** `UGraph` (undirected) and `DGraph` (directed) share the base class `Graph`, which is concrete: `Graph(directed=True)` is a `DGraph`. There is one `Vertex` class for both and one `Edge` class. `UVertex` and `DVertex` are deprecated aliases of `Vertex`. Constructor arguments are keyword-only: `UGraph(dim=2, metric="L2", heuristic=None, vertex_class=None, edge_class=None, verbose=False)`; `UGraph(2)` is a `TypeError`.
- **Vertices.** `g.add_vertex(coord=None, name=None)` returns the vertex. Default names are `"#0"`, `"#1"`, ..., names are unique, and `g["a"]` or `g[0]` finds a vertex. `for v in g` iterates in the order added; `g.n`, `g.ne` and `g.nc` are the numbers of vertices, edges and components. `g.remove_vertex(v)` and `g.remove_edge(e)` remove them, with their edges.
- **Edges.** `g.add_edge(v1, v2, cost=None)`; with no `cost` it is the metric distance between the coordinates, and `None` if the vertices have none (planning then raises `ValueError`). `g.edges()` and `v.edges()` return new lists in the order added. `e.v1` and `e.v2` are read-only. `v.adjacent()` is the neighbours, `v.successors()` and `v.predecessors()` are the directed ones, `v.edgeto(w)` the cheapest edge to `w` and `v.edges_to(w)` all of them. A self-loop of an undirected vertex is in `v.edges()` twice and counts 2 in its degree.
- **Directed graphs.** `degree` is in plus out (`in_degree`, `out_degree`); `nc` and `component` are weakly connected, ignoring direction.
- **Planning.** `path_BFS(S, G)` returns `(path, length)` with the fewest edges; `path_UCS(S, G)` and `path_Astar(S, G)` return `(path, length, parents)` of least cost. `S` and `G` are vertices or names, and all three return `None` when there is no path, so check before unpacking.
- **Metrics.** `"L2"` (default), `"L1"`, `"SE2"` for `(x, y, theta)` poses, `SE2Metric(weight)` to weight the angle, or a function of one difference vector. A* uses `heuristic` (default: the metric), which must be in the same units as the edge costs and must not exceed the true cost to go, or the path is not the shortest.
- **Spatial queries** on an embedded graph: `closest(point)` returns `(vertex, distance)`, `near(point, radius)`, `within(box)` with `box` a 2 x n array of lower and upper corner, `coordinates()` the `(dim, n)` matrix of coordinates, and `subgraph(vertices)`, which takes vertices, names or a boolean mask and returns a new graph. `build_kdtree()` is an optional index (it needs scipy) that `closest` and `near` then use.
- **Components and matrices.** `g.component(c)` is the vertices in component number `c`, `v.label` is the component of a vertex, `g.samecomponent(v1, v2)`. `adjacency()`, `incidence()`, `laplacian()`, `degree()` and `connectivity()` return dense NumPy arrays.
- **Output.** `g.plot()` (arrowheads for a directed graph, arcs for parallel edges, circles for loops), `highlight_path(path)`, and `g.export(file)` for GraphViz dot, Mermaid, GraphML, ELK JSON or an image (images need the GraphViz executables). Both plot methods default to `block=None`, which does not call `plt.show()` and lets the environment display the figure (useful for notebook overlays); `block=False` shows without waiting, and `block=True` waits in desktop GUI applications.
- **Subclassing.** `class Foo(Vertex)` and `UGraph(vertex_class=Foo, edge_class=Bar)` make the graph build them; a subclass `__init__` must pass `coord` and `name` (vertices) or `v1, v2, cost, data` (edges) to its parent.
- **Copying.** `g.copy()` and `pickle` work for a whole graph; a vertex or edge cannot be copied or pickled on its own, and a graph whose metric is a `lambda` cannot be pickled.

## Common mistakes

- `UGraph(6)` raises `TypeError`: write `UGraph(dim=6)`.
- Unpacking the result of a planner without checking for `None`.
- Coordinates in one unit and costs in another (metres against kilometres): the A* heuristic is then too large and the path found is not the shortest.
- `g.component(v)` takes a component *number*; use `v.label` for the component of a vertex.
- Assigning `e.v1 = ...` fails, they are read-only; `add_vertex` raises `ValueError` for a name already used or a vertex that is already in a graph.
- After `build_kdtree()`, change a coordinate by assigning (`v.coord = x`, `v.coord += x`), not in place (`v.coord[0] = x`), which the index cannot see.
- `g.edges()` is a copy: changing the list does not change the graph.
