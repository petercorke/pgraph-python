#!/usr/bin/env python
"""Benchmark the key operations of pgraph on random geometric graphs.

Each graph is N vertices placed uniformly at random in the unit square, each
joined to its 4 nearest neighbours (about 6 edges per vertex once duplicate
edges are merged in the undirected case), like the roadmap a motion planner
builds.  The random number generator is seeded so every run measures the same
graphs and the same path queries.

The table gives, for each operation, the time at each graph size and the
growth exponent ``b`` in ``time ~ N**b`` fitted between the smallest and
largest size, for the quantity as shown (per vertex, per edge, per call, ...):
0 is constant, 1 is linear, 2 is quadratic.

Examples::

    python benchmark.py                 # undirected graphs
    python benchmark.py --directed      # directed graphs
    python benchmark.py --quick         # skip the largest graph
    python benchmark.py --md            # table as MarkDown, to paste into a README
    python benchmark.py --rst           # table as reST, to paste into the docs

The table is made with ``ansitable``, which also writes it as MarkDown or reST.
With ``--md`` or ``--rst`` only the table is written to standard output, so it
can be redirected to a file, and the description of the run goes to standard
error.
"""

from __future__ import annotations

import argparse
import math
import pickle  # nosec B403 nosemgrep - only used to time a round trip of our own data
import platform
import statistics
import subprocess  # nosec B404 - only used to ask macOS for the CPU name
import sys
import time
from collections.abc import Callable
from importlib import metadata
from typing import Any

import numpy as np
from ansitable import ANSITable, Column

from pgraph import DGraph, Graph, UGraph

NEIGHBOURS = 4  # each vertex is joined to this many nearest vertices
NQUERIES = 10  # path queries per graph, the same ones for every planner
MIN_REPS = 3
MAX_REPS = 1000


# ---------------------------------------------------------------------------
# graphs and timing
# ---------------------------------------------------------------------------


def random_graph_data(
    n: int, directed: bool, rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Vertex coordinates and the edges (index pairs) of a k-nearest-neighbour graph"""
    coords = rng.random((n, 2))
    k = min(NEIGHBOURS, n - 1)
    edges: set[tuple[int, int]] = set()
    chunk = max(1, 4_000_000 // n)  # bound the size of the distance matrix
    for start in range(0, n, chunk):
        block = coords[start : start + chunk]
        d2 = ((block[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
        d2[np.arange(len(block)), np.arange(start, start + len(block))] = np.inf
        nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]
        for row, near in enumerate(nearest):
            u = start + row
            for v in near:
                v = int(v)
                edges.add((u, v) if directed else (min(u, v), max(u, v)))
    return coords, sorted(edges)


def build(
    cls: type[Graph], coords: np.ndarray, edges: list[tuple[int, int]] | None = None
) -> Graph:
    g = cls(dim=2)
    vertices = [g.add_vertex(c) for c in coords]
    if edges is not None:
        for a, b in edges:
            g.add_edge(vertices[a], vertices[b])
    return g


def roundtrip(obj: Any) -> Any:
    """Pickle and unpickle: it only ever reads what it has just written"""
    return pickle.loads(pickle.dumps(obj))  # nosec B301 nosemgrep


def measure(
    fn: Callable[[Any], Any],
    setup: Callable[[], Any] | None = None,
    budget: float = 0.2,
) -> float:
    """Median time in seconds of one call of ``fn``

    ``setup()``, if given, makes a fresh argument for each call and is not
    timed.  Calls are repeated until ``budget`` seconds have elapsed in total
    (including setup), at least ``MIN_REPS`` and at most ``MAX_REPS`` times.
    """
    times = []
    started = time.perf_counter()
    while len(times) < MAX_REPS:
        arg = setup() if setup is not None else None
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
        if len(times) >= MIN_REPS and time.perf_counter() - started >= budget:
            break
    return statistics.median(times)


def path_queries(
    g: Graph, rng: np.random.Generator, count: int = NQUERIES
) -> list[tuple[Any, Any]]:
    """Random start/goal pairs that are connected, found with BFS"""
    pairs = []
    for _ in range(50 * count):
        i, j = (int(x) for x in rng.integers(0, g.n, 2))
        if i != j and g.path_BFS(g[i], g[j]) is not None:
            pairs.append((g[i], g[j]))
        if len(pairs) == count:
            break
    return pairs


# ---------------------------------------------------------------------------
# the benchmark
# ---------------------------------------------------------------------------

MICRO, MILLI = 1e6, 1e3  # scale from seconds to the unit shown


def run_size(
    cls: type[Graph], n: int, seed: int, matrix_max: int, with_plot: bool
) -> dict[str, float | None]:
    """All the measurements for one graph size, in the units of ROWS, keyed by name"""
    rng = np.random.default_rng(seed + n)
    coords, edges = random_graph_data(n, cls._directed, rng)
    g = build(cls, coords, edges)
    ne = g.ne
    out: dict[str, float | None] = {}

    # building
    out["add_vertex"] = measure(lambda _: build(cls, coords)) / n
    out["add_edge"] = (
        measure(
            lambda h: [h.add_edge(h[a], h[b]) for a, b in edges],
            setup=lambda: build(cls, coords),
        )
        / ne
    )
    out["copy"] = measure(lambda _: g.copy())
    out["pickle"] = measure(lambda _: roundtrip(g))

    # structure
    def fresh_changed() -> Graph:
        h = g.copy()
        h.add_edge(h[0], h[1])  # a change, so the components are stale
        return h

    out["nc_first"] = measure(lambda h: h.nc, setup=fresh_changed)
    out["nc_cached"] = measure(lambda _: g.nc)
    out["remove_vertex"] = measure(
        lambda h: h.remove_vertex(h[n // 2]), setup=lambda: g.copy()
    )
    queries = rng.random((20, 2))
    out["closest"] = measure(lambda _: [g.closest(q) for q in queries]) / len(queries)

    # matrices are dense, N x N (or N x E)
    if n <= matrix_max:
        out["adjacency"] = measure(lambda _: g.adjacency())
        out["laplacian"] = measure(lambda _: g.laplacian())
        out["incidence"] = measure(lambda _: g.incidence())
    else:
        out["adjacency"] = out["laplacian"] = out["incidence"] = None

    # path planning, median over the same random connected pairs
    pairs = path_queries(g, rng)
    if not pairs:
        raise RuntimeError(f"no connected pairs found in the graph with {n} vertices")
    costs: dict[str, list[float]] = {}
    for name in ("BFS", "UCS", "Astar"):
        method = getattr(g, f"path_{name}")
        times, lengths = [], []
        for s, t in pairs:
            t0 = time.perf_counter()
            result = method(s, t)
            times.append(time.perf_counter() - t0)
            lengths.append(result[1])
        out[f"path_{name}"] = statistics.median(times)
        costs[name] = lengths
    # a guard that the numbers mean something: the two cost-optimal planners
    # must agree on the path cost
    if not np.allclose(costs["UCS"], costs["Astar"]):
        raise RuntimeError("path_UCS and path_Astar disagree on the path cost")

    if with_plot:
        import matplotlib.pyplot as plt

        def plot_and_draw(_: Any) -> None:
            fig = plt.figure()
            g.plot()
            fig.canvas.draw()
            plt.close(fig)

        out["plot"] = measure(plot_and_draw, budget=0.0)
    return out


# name, description, unit scale, unit label
ROWS: list[tuple[str, str, float, str]] = [
    ("add_vertex", "add a vertex", MICRO, "µs / vertex"),
    ("add_edge", "add an edge (cost from the metric)", MICRO, "µs / edge"),
    ("copy", "copy the graph", MILLI, "ms"),
    ("pickle", "pickle and unpickle the graph", MILLI, "ms"),
    ("nc_first", "number of components, after a change", MILLI, "ms"),
    ("nc_cached", "number of components, unchanged", MICRO, "µs"),
    ("remove_vertex", "remove a vertex", MICRO, "µs"),
    ("closest", "closest vertex to a point", MICRO, "µs / query"),
    ("adjacency", "adjacency matrix", MILLI, "ms"),
    ("laplacian", "Laplacian matrix", MILLI, "ms"),
    ("incidence", "incidence matrix", MILLI, "ms"),
    ("path_BFS", "path, breadth-first", MILLI, "ms / query"),
    ("path_UCS", "path, uniform cost", MILLI, "ms / query"),
    ("path_Astar", "path, A*", MILLI, "ms / query"),
    ("plot", "plot() and draw", MILLI, "ms"),
]


def growth(sizes: list[int], values: list[float | None]) -> str:
    """Exponent b of time ~ N**b between the smallest and largest size measured"""
    pts = [(n, v) for n, v in zip(sizes, values) if v]
    if len(pts) < 2 or pts[0][1] <= 0:
        return ""
    (n0, v0), (n1, v1) = pts[0], pts[-1]
    b = math.log(v1 / v0) / math.log(n1 / n0)
    return f"{b:.1f}".replace("-0.0", "0.0")


def cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            r = subprocess.run(  # nosec B603 B607
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def fmt(value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value:.3g}" if value < 1000 else f"{value:,.0f}"


def make_table(
    results: dict[int, dict[str, float | None]], sizes: list[int]
) -> ANSITable:
    table = ANSITable(
        Column("Operation", colalign="<", headalign="<"),
        Column("per", colalign="<", headalign="<"),
        *[Column(f"N={n:,}") for n in sizes],
        Column("growth exponent"),
        border="thin",
    )
    for key, text, scale, unit in ROWS:
        if key not in results[sizes[0]]:
            continue
        vals = [
            None if results[n][key] is None else results[n][key] * scale  # type: ignore[operator]
            for n in sizes
        ]
        table.row(text, unit, *[fmt(v) for v in vals], growth(sizes, vals))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--directed", action="store_true", help="use DGraph, not UGraph")
    parser.add_argument("--quick", action="store_true", help="skip the largest graph")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[100, 1_000, 10_000], help="numbers of vertices"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--matrix-max",
        type=int,
        default=1_000,
        help="largest N for the dense matrix rows (default %(default)s)",
    )
    parser.add_argument("--plot", action="store_true", help="also time plot() (needs matplotlib)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--md", action="store_true", help="print the table as MarkDown")
    output.add_argument("--rst", action="store_true", help="print the table as reST")
    args = parser.parse_args()

    sizes = sorted(args.sizes)[:-1] if args.quick and len(args.sizes) > 1 else sorted(args.sizes)
    cls: type[Graph] = DGraph if args.directed else UGraph

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")

    version = metadata.version("pgraph-python")
    print(
        f"\npgraph {version} benchmark:\n"
        f"  * {cls.__name__}, random geometric graphs, {NEIGHBOURS} nearest neighbours per vertex,\n"
        f"  * {NQUERIES} random connected start/goal pairs per size, median reported,\n"
        f"  * running on {cpu_name()}, Python {platform.python_version()}, "
        f"numpy {np.__version__}.\n",
        file=sys.stderr if (args.md or args.rst) else sys.stdout,
    )

    results: dict[int, dict[str, float | None]] = {}
    for n in sizes:
        print(f"N = {n:,} ", end="", file=sys.stderr, flush=True)
        results[n] = run_size(cls, n, args.seed, args.matrix_max, args.plot)
        print("done", file=sys.stderr)

    table = make_table(results, sizes)
    if args.md:
        print(table.markdown(), end="")
    elif args.rst:
        print(table.rest(), end="")
    else:
        table.print()


if __name__ == "__main__":
    main()
