# pgraph-python — Agent Instructions

> **Audience:** agents working on pgraph's source code. If you're helping someone *use* pgraph, read https://petercorke.github.io/pgraph-python/llms.txt instead (or `llms-full.txt` for the complete documentation).

Part of the RVC ecosystem. **Read [rvc-ecosystem/AGENTS.md](https://github.com/petercorke/rvc-ecosystem/blob/main/AGENTS.md) first** — it defines shared conventions: repo ownership, math invariants, dependency boundaries, git/PR workflow, code standards, tech-debt tracking. This file only adds what's specific to this repo.

| | |
|---|---|
| PyPI package | `pgraph-python` |
| Nickname | pgraph-python |
| Owner | Peter Corke (`petercorke`) |
| Default branch | `main` |
| Contribution model | Branch → PR; `main` is protected: CI must pass and the branch must be up to date (`gh pr update-branch`) |

## Notes specific to this repo

- Minimal graph/path-planning package. Pure Python, builds with Hatch (`hatchling`); Python 3.9+. Depends on `numpy`, `matplotlib` and `spatialmath-python`; `ansitable` and `scipy` arrive through spatialmath. **Do not add scipy to `pyproject.toml`**: it is imported lazily, only inside `Graph.build_kdtree()`.
- Used, unpinned (`"pgraph-python"`), by `robotics-toolbox-python` (`PRMPlanner`, `RRTPlanner`, `LatticePlanner`, `PoseGraph`, which subclass `UVertex`/`DVertex`) and `machinevision-toolbox-python` (`BundleAdjust`). A breaking change needs a deprecation cycle with a working fallback (`warnings.warn(..., DeprecationWarning, stacklevel=2)`); run their pgraph-using tests before a release.
- **Layout.** `src/pgraph/PGraph.py` holds `Graph`, `UGraph`, `DGraph`, `Vertex`, `Edge` and `SE2Metric`; `exporters.py` the `export()` formats. There is one `Vertex` class: a vertex has no direction, the graph decides how an edge is recorded (`UGraph`/`DGraph` implement the hooks `_attach`, `_detach`, `_incoming` and `_endpoints_match`). `UVertex`/`DVertex` are deprecated aliases, by module `__getattr__`.
- **Records that must agree.** A vertex has `_edgelist` (leaving, or incident if undirected) and `_inlist` (entering, directed only); the graph has `_vertexlist`, `_vertexdict` (names) and `_edges` (an insertion-ordered dict). `g.validate()` checks them all; use it in tests of anything that edits the structure.
- **Two staleness flags, consumed independently.** `_connectivitychange` (components, reset by `_graphcolor`) and `_spatialchange` (the k-d tree). Set both wherever a vertex is added or removed; only connectivity cares about edges. Never share one flag.
- **`Vertex.__setattr__` is hot.** It exists to tell the graph that `coord` was assigned. Anything run per vertex (`Vertex.__init__`, `add_vertex`, components) writes `vertex.__dict__` directly to skip it; keep doing that, and declare new attributes at class level so mypy knows them.
- **`import pgraph` loads only numpy.** matplotlib, spatialmath and scipy are imported by the methods that need them (`plot`, `highlight_*`, `_edge_artist`, `build_kdtree`), and the classes that need matplotlib to be defined are in `_plot.py`. `tests/test_lazy_imports.py` runs a fresh interpreter to check it: do not add a module-level import of them.
- **Pickle and `copy` use a flat description** (`__getstate__`/`__setstate__`: vertices, edges by index, replayed through `_attach`); `subgraph` reuses it. Derived or transient state on a graph (the k-d tree) must be excluded there. A bare vertex or edge refuses to be pickled or copied.
- **Tests** are `unittest` classes in `tests/`, run with `pytest`; write expected values by hand or from an oracle in the test, never from the code under test. Import names explicitly (a star import is a Codacy `F405`); route `pickle` through one helper marked `# nosec B301 nosemgrep`. Codecov gates patch coverage, so every changed line needs a test. `mypy src` is a CI check.
- **Performance claims are measured.** `examples/benchmark.py` (`--md`, `--directed`) produces the README tables; regenerate them after a performance change and compare with `v0.6.5` (`git archive v0.6.5 src`), not with a development build.
- **Docs** are Sphinx (`make -C docs html`; `-n` for nitpicky) with `sphinx-pyrunblock` `.. runblock:: pycon` examples that execute at build. `docs/llms_intro.md` is the hand-written part of `llms.txt`: update it when the API changes; `docs/llms_txt.py` builds `llms.txt` and `llms-full.txt` from it and the toctrees (`make html llms`), and CI publishes them with the site.
- **Release.** Version is in `pyproject.toml`, history in `CHANGELOG.md`. Publishing is `.github/workflows/publish.yml` on a GitHub Release (trusted publishing, environment `pypi`): it refuses unless the tag equals the version and the commit is on `main`. Don't tag or upload by hand.
- `examples/london.json` is OpenStreetMap data (© OpenStreetMap contributors, ODbL); keep the attribution with it.
