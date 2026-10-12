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
  assigns it).  Assigning an endpoint of an edge that was in a graph left the
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

Added

- `adjacent()` and `isadjacent()`, the graph-theory names, which have one
  spelling.  `neighbours()`, `neighbors()` and `isneighbour()` from 0.6.5 are
  kept as aliases, with `isneighbor()` added, and none of them warn.
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

Fixed

- `Dict()` always created undirected vertices, even for a `DGraph`; `connect()`
  joined vertices of two different graphs, and now also validates the
  endpoints of a supplied edge; `path_BFS`/`path_UCS`/`path_Astar` checked the
  type of the start vertex twice and never the goal; and `distance()` and the
  planners packed `nan` or `None` into their results for edges with no cost
  instead of raising.
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
- Removing a vertex left its `_graph` pointing at the graph; renaming a vertex
  left the name table stale; `g[x]` accepted any type.

Known limitations

- `plot()` draws no arrowheads for a directed graph, and parallel edges and
  self-loops are not visible in it.
- A graph whose metric is a `lambda` or a locally defined function can be copied
  but not pickled; the built-in `"L1"`, `"L2"` and `"SE2"` and module-level
  functions can be.

Build, tests and documentation

- Modern type hints and reST docstrings throughout, most with a runnable
  example, and `mypy` is a CI check.
- The documentation is organized by task, with the data structures explained and
  a policy page on object lifecycle, and builds with `sphinx-pyrunblock` (which
  replaces the unmaintained `sphinx-autorun`) and the matplotlib plot directive.
- Python 3.9 to 3.14 are tested, the unused `scipy` dependency is dropped, and
  PyPI publishing is automated through GitHub Releases and trusted publishing.

0.6.3

- runs with Numpy2.x
- changed build system to .toml file
- misc doco and changes
- migrate package to src layout (pgraph under src/pgraph)
- switch build backend to hatchling
- update CI workflow for main branch and docs publish to gh-pages
- update repository links and badges from master to main