Unreleased

Behaviour change

- `Graph.plot()` and `Graph.highlight_path()` now default to `block=None`
  instead of `False`. They no longer call `plt.show()` unless `block` is
  specified, allowing notebooks and other environments to manage display.
  Scripts and plain IPython sessions without `%matplotlib` or `plt.ion()`
  that relied on `g.plot()` opening a window now need an explicit
  `plt.show()` after drawing, or `block=False` for the previous behaviour.
  `block=True` still waits for the window to close.

Performance

- `closest()` and `near()` no longer make a Python-level pass over the vertices
  when there is no k-d tree.  The graph keeps the coordinates as a matrix, built
  on first use and appended to by `add_vertex`, and applies the built-in metrics
  to all rows at once.  A query on a 1,000-vertex graph takes 12 microseconds
  rather than 730, and stays that fast while vertices are being added, which is
  what the Robotics Toolbox's `LatticePlanner` and `RRTPlanner` do (a lattice of
  1,292 vertices is planned in 0.06 s rather than 1.07 s).  Results are the
  same.  The matrix is a copy of the coordinates, so changing one in place,
  `vertex.coord[0] = x`, is not seen by `closest()`, as was already so with
  `build_kdtree()`; assigning `vertex.coord` is.  A metric of your own is still
  applied to each vertex in turn.

1.0.0

First 1.0 release: six years of production use since the first commit in 2020
(including as a real dependency of both the Robotics Toolbox for Python and the
Machine Vision Toolbox for Python), and this cycle's type hints, documentation,
testing and bug-fixing pass, felt like the right point to commit to API
stability.  This is the first release with a considered API, so it contains
several deliberate breaking changes, listed first with how to migrate.

Breaking changes (from 0.6.5)

- The `UGraph`/`DGraph` constructor arguments are keyword-only.  0.6.5 accepted
  and silently ignored a first positional argument, so `UGraph(6)` looked as
  though it set a dimension but did nothing.  Now it is a `TypeError`: write
  `UGraph(dim=6)`.
- `degree` of a `DGraph` is the graph-theory degree, in plus out, so a
  self-loop counts 2 and the degrees sum to twice the number of edges.  It used
  to count only the edges leaving the vertex.  This changes `vertex.degree`,
  `graph.degree()` and `graph.connectivity()`, and `graph.average_degree()` is
  now 2E/N for both kinds of graph (it was E/N for a `DGraph`).  The previous
  values are `vertex.out_degree`, `[v.out_degree for v in g]` and `g.ne / g.n`.
  `laplacian()` is unchanged (out-degree minus adjacency).  Undirected graphs
  are unchanged.
- `graph.edges()` is a `list` in the order the edges were added.  It was the
  graph's own `set`, with no defined order, although documented as a list.  Both
  `graph.edges()` and `vertex.edges()` return a new list, so changing the result
  cannot corrupt the graph.  Use `set(g.edges())` for set operations.
- `Edge.v1` and `Edge.v2` are read-only, and so is `Vertex.label` (the graph
  assigns it, and reading it now finds the components, where 0.6.5 gave `None`
  until `nc` had been asked for).  Assigning an endpoint of an edge that was in a graph left the
  graph inconsistent and the edge impossible to remove.  Give the vertices to
  `Edge(v1, v2)` or connect the edge with `v1.connect(v2, edge=e)`.
- Assigning `Vertex.name` renames it in the graph (so `g[new]` works) and
  refuses a name already in use or `None`.
- `add_vertex` raises `ValueError` for a name already used in the graph or a
  vertex that already belongs to a graph, instead of silently shadowing it.
  The default name `#N` is now the first free one, which could previously
  repeat after a removal.
- There is one vertex class, `Vertex`: a vertex has no direction of its own, the
  graph it belongs to decides how an edge is recorded at its ends.  `UVertex`
  and `DVertex` are deprecated names for it (below).  A vertex given to
  `add_vertex` is no longer checked against the kind of graph, because there is
  no kind to mismatch.  An override of the private `_attach`/`_detach` hooks
  moves from a vertex subclass to a `UGraph`/`DGraph` subclass.
- The abstract base class `PGraph` is now `Graph`, and `Graph` and `Vertex` are
  exported from the package.  `Graph` is concrete: `Graph(directed=True)` is a
  `DGraph` and `Graph()` a `UGraph`.
- `Adjacency()` treats a non-zero diagonal element as a self-loop.
- The `str` of an edge between vertices that are not in a graph shows `--`,
  because there is no kind of graph to say whether it is directed.

Deprecated (still work, with a `DeprecationWarning`)

- `Laplacian()` is `laplacian()`: a capital initial denotes a class method that
  builds a graph, like `Adjacency()` and `Dict()`.
- `UVertex` and `DVertex` are `Vertex`.
- `remove()` is `remove_vertex()` or `remove_edge()`, which say which they do.
- The options of `plot()` are named for what they are: `vopt` is `vertex_opt`,
  `eopt` is `edge_opt`, `text` is `labels` and `text_opt`, `colorcomponents` is
  `color_components` and `force2d` is `force_2d`.  The positional order is
  unchanged; giving both an old name and its new one is a `TypeError`.

Added

- `adjacent()` and `isadjacent()`, the graph-theory names, which have one
  spelling.  `neighbours()`, `neighbors()` and `isneighbour()` from 0.6.5 are
  kept as aliases, with `isneighbor()` added, and none of them warn.
- `plot()` draws each edge once.  A directed graph has arrowheads, which
  `arrow_opt` styles; edges that join the same pair of vertices, in either
  direction, are arcs that bow apart; and a loop is a circle at its vertex.  In
  0.6.5 a directed graph looked like an undirected one, parallel edges lay
  exactly on top of each other and a loop was a line of no length.  `plot()`
  does the same in 3D, and `highlight_edge()` follows an arc.
- `Graph(directed=...)`, the `directed` property, and `vertex_class` and
  `edge_class` constructor arguments, so that user subclasses of `Vertex` and
  `Edge` are built by the graph.
- `add_edge()` accepts an `Edge`, and `connect()` an `edge=` argument.
- `edges_to(dest)` lists every edge from a vertex to another, so parallel edges
  can be seen; `edgeto(dest)` is still the cheapest.
- Directed vertices record their incoming edges: `predecessors()`,
  `successors()`, `in_edges()`, `out_edges()`, `in_degree` and `out_degree`.
- `export()` writes a graph as GraphViz dot, Mermaid, GraphML or ELK JSON, or as
  an image rendered by the GraphViz executables, with the layout preserved.  The
  `pgraph.exporters` module can be extended with `register_exporter`.
- `validate()` reports every way in which a graph's records disagree, for tests
  and debugging.
- A graph can be pickled, and `copy()` works for graphs of any size.  `copy()`
  raised `RecursionError` for a path of about 100 vertices.
- `iscyclic()`, `remove_edge()`, `remove_vertex()` and the `dim` argument, which
  declares that every embedded vertex has a coordinate of exactly that length.
- Self-loops and parallel edges are supported throughout: adjacency, incidence,
  degree and Laplacian matrices, and planning, which takes the cheapest of
  parallel edges.
- `SE2Metric(weight)` is the `"SE2"` metric with a weight on the angle, the
  length that a radian is worth, so that position and orientation need not
  count equally.  `"SE2"` is `SE2Metric()`, a weight of 1, so existing results
  are unchanged.  It can be the metric or the heuristic of a graph and is
  pickled with its weight.
- Spatial queries on an embedded graph: `coordinates()` is the matrix of vertex
  coordinates, a column per vertex; `within(box)` the vertices inside a box;
  `near(point, radius)` those within a distance, nearest first; and
  `subgraph(vertices)` the graph that a set of vertices induces, given as
  vertices, names or a boolean mask, as a new graph that shares nothing with
  the original.
- `build_kdtree()` makes an optional k-d tree that `closest()` and `near()` use
  for the built-in metrics, `SE2Metric` included.  It needs scipy, which is
  imported only then and is not a dependency; and it is dropped whenever a
  vertex is added or removed or a coordinate is assigned, so a stale one is
  never used.
- `validate()` also checks that coordinates are finite vectors of one length,
  and with `embedded=True` that every vertex has one.

Faster

- `path_BFS`, `path_UCS` and `path_Astar` scaled badly: they searched lists with
  `in`, and took the minimum of the whole frontier at every step.  Results and
  the verbose output are identical.  On a random graph of 10,000 vertices and
  30,000 edges, the mean of six random queries is 3 ms for breadth-first search
  where 0.6.5 took 172 ms, 18 ms for uniform cost where it took 1,500 ms, and
  13 ms for A* where it took 410 ms.
- The components are found by a union-find over the edges, 4 ms for that graph
  where 0.6.5 took 15 ms after a change.
- Creating a vertex is faster: `Vertex(coord)` takes 0.35 µs where 0.6.5 took
  1.5 µs, and `g.add_vertex(coord)` 1.7 µs where it took 2.2 µs.  Making the
  coordinate with `np.r_` was most of the cost.
- `import pgraph` takes 0.04 s where it took 0.6 s, and loads neither
  matplotlib, spatialmath nor scipy: they are imported by the methods that plot
  or build a spatial index.  A script that only builds graphs and plans paths no
  longer pays for them.
- `str(g)` separates thousands: `DGraph: 4,881 vertices, 8,392 edges, 14
  components`.

Fixed

- `Dict()` always created undirected vertices, even for a `DGraph`; `connect()`
  joined vertices of two different graphs, and now also validates the
  endpoints of a supplied edge; `path_BFS`/`path_UCS`/`path_Astar` checked the
  type of the start vertex twice and never the goal; and `distance()` and the
  planners packed `nan` or `None` into their results for edges with no cost
  instead of raising.
- `nc` never returned for many directed graphs, the smallest a 4-cycle, and
  neither did `component()`, `samecomponent()` or `plot()`, which ask for it.
  Components of a directed graph are weakly connected, as they were meant to
  be, and are numbered from 0 by the first vertex of each, for both kinds of
  graph.  They are also no longer found again by every call once the graph has
  changed.
- `Edge.vertices()` raised `DeprecationWarning` as an exception; it now warns
  and returns the endpoints, as `Edge.endpoints` does.
- `incidence()` returned the wrong matrix for every graph with an edge: a loop
  variable overwrote the vertex index, so edges were marked in the wrong rows,
  and for a directed graph the vertex at the end of an edge was not marked at
  all.  The columns follow `edges()`, which is now in the order the edges were
  added, where 0.6.5 used the order of a set.
- `remove()` crashed for every directed edge; `dotfile()` closed `sys.stdout`
  when called with its default arguments; and several methods raised a bare
  `AttributeError` or `TypeError` instead of a clear message for a vertex with
  no coordinate or no graph.
- `highlight_edge()`, and so `highlight_path()`, raised `AttributeError` for a
  graph of three dimensions, because it drew a flat arrow in the 3D axes (0.6.5
  drew lines), and `highlight_vertex()` put its marker at z = 0.  They draw in
  the dimension of the axes, so a 3D graph plotted with `force_2d` is
  highlighted in 2D.
- Removing a vertex left its `_graph` pointing at the graph; renaming a vertex
  left the name table stale; `g[x]` accepted any type.

Known limitations

- A graph whose metric is a `lambda` or a locally defined function can be copied
  but not pickled; the built-in `"L1"`, `"L2"` and `"SE2"` and module-level
  functions can be.

Build, tests and documentation

- Modern type hints and reST docstrings throughout, most with a runnable
  example, and `mypy` is a CI check.
- The documentation is organized by task, with the data structures explained and
  a policy page on object lifecycle, and builds with `sphinx-pyrunblock` (which
  replaces the unmaintained `sphinx-autorun`) and the matplotlib plot directive.
- The source distribution no longer contains a stale copy of the source
  (`build/lib`, left by a setuptools build before the move to the `src` layout),
  nor the `RELEASE` file, which held `0.6.1` and which nothing read.
- Python 3.9 to 3.14 are tested, the unused `scipy` dependency is dropped, and
  PyPI publishing is automated through GitHub Releases and trusted publishing.
  The workflow checks that the tag is the version in `pyproject.toml` and that
  the tagged commit is on `main`, and installs the built wheel and runs a path
  query before it uploads.
- The documentation site is deployed with the GitHub Pages actions.  It had not
  been updated since January 2025, because the CI pushed to a branch that Pages
  no longer served.
- `llms.txt` and `llms-full.txt` are published with the documentation, for AI
  assistants: a short introduction with a quick start and common mistakes, and the
  whole documentation as one text file.  `AGENTS.md` describes the source for
  assistants that work on it.
- Two notebooks in `examples/`: `roads.ipynb`, the Queensland example, and
  `london.ipynb`, the road network of central London from OpenStreetMap as a
  directed graph with one-way streets, planned through with A*.  There is also
  a benchmark, `examples/benchmark.py`, and the README has its results and an
  installation section.
- The Queensland figure in the README shows the shortest route.  It had been
  drawn with coordinates in metres and distances in kilometres, which made the
  A* heuristic far too large, so the route it found, 1,759 km, was not the
  shortest, 1,659 km.

0.6.3

- runs with Numpy2.x
- changed build system to .toml file
- misc doco and changes
- migrate package to src layout (pgraph under src/pgraph)
- switch build backend to hatchling
- update CI workflow for main branch and docs publish to gh-pages
- update repository links and badges from master to main