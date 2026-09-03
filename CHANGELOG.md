1.0.0

- First 1.0 release: six years of production use since the first commit
  in 2020 (including as a real dependency of both the Robotics Toolbox
  for Python and the Machine Vision Toolbox for Python), and this
  cycle's comprehensive type-hinting, documentation, and bug-fixing
  pass, felt like the right point to commit to API stability.
- Breaking (internal-only): renamed the abstract base classes `PGraph` ->
  `_BaseGraph` and `Vertex` -> `BaseVertex`. Neither was ever exported
  from the package or referenced by name elsewhere in the ecosystem.
- Added `iscyclic()` to test whether a graph contains a cycle.
- Added a `dim` keyword arg to `UGraph`/`DGraph`: declares that every
  embedded vertex must have a coordinate of exactly that length,
  enforced in `add_vertex()`.
- Added `remove_edge()`/`remove_vertex()` as explicit replacements for
  the old, ambiguous `remove()` (deprecated, still works).
- Added modern type hints and reST docstrings throughout, most with a
  runnable, doc-build-verified usage example.
- Fixed a cluster of real bugs found via the type-hinting and testing
  pass: wrong-subclass vertices silently accepted or mangled by
  `add_vertex()`; `Dict()` always creating undirected vertices even for
  `DGraph`; a broken same-type check in `connect()` that never actually
  caught a cross-type or cross-graph connection; `path_BFS`/`path_UCS`/
  `path_Astar` validating the wrong parameter's type; `distance()` and
  the path planners silently packing `None`/`nan` into results for
  costless edges instead of raising; two bugs in `incidence()` that
  scrambled its output; `remove()` unconditionally crashing for any
  directed-graph edge; `dotfile()` closing `sys.stdout` when called
  with its default arguments; and several methods crashing with a bare
  `AttributeError`/`TypeError` instead of a clear message when called on
  a vertex with no coordinate or no graph membership.
- Removed the vertex `connect()` "pass an existing Edge instead of a
  destination vertex" option -- it was completely broken and exercised
  by no test.
- `mypy` added as an informational CI check; the bulk of the
  `Optional`-narrowing gap it found has been closed.
- Documentation now builds with `sphinx-pyrunblock` (replacing the
  unmaintained `sphinx-autorun`) and `matplotlib.sphinxext.plot_directive`
  for real embedded plot images; the doc footer now matches the
  ecosystem-wide convention.
- CI: added Python 3.13/3.14 to the test matrix; PyPI publishing is now
  automated via GitHub Releases and trusted publishing (see
  `.github/workflows/publish.yml`) instead of a manual `twine upload`.

0.6.3

- runs with Numpy2.x
- changed build system to .toml file
- misc doco and changes
- migrate package to src layout (pgraph under src/pgraph)
- switch build backend to hatchling
- update CI workflow for main branch and docs publish to gh-pages
- update repository links and badges from master to main