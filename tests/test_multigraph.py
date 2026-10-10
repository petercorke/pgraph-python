"""
Self-loops and parallel edges.

pgraph does not forbid either, so every quantity derived from the edge set
(degree, the matrix representations, cycle detection, path planning) has to
give a defined, consistent answer when they are present.

The same narrative is run against ``UGraph`` and ``DGraph``:

1. a simple acyclic network, the baseline
2. add a self-loop on a vertex that lies on the planned route
3. remove it, which must restore the baseline exactly
4. add parallel edges, planning must take the cheapest
5. add the self-loop back in on top of the parallel edges

Conventions being tested (see the matrix methods' docstrings):

- a self-loop counts twice towards the degree of a ``UGraph`` vertex, once
  towards the (out-)degree of a ``DGraph`` vertex
- ``adjacency()`` counts edges, so ``A[i, j]`` is the number of edges
  from ``i`` to ``j`` and each row of ``A`` sums to the degree of the vertex
- ``Laplacian()`` is therefore ``degree() - adjacency()``, rows sum to zero
- ``distance()`` and path lengths use the cheapest of any parallel edges
"""

import contextlib
import signal
import unittest

import numpy as np

from pgraph import DGraph, UGraph


@contextlib.contextmanager
def within(seconds=5):
    """Fail, rather than hang the whole suite, if the body does not return."""
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover - not on Windows
        yield
        return

    def handler(signum, frame):
        raise AssertionError(f"did not finish within {seconds}s (infinite loop?)")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class _Multigraph:
    """
    Shared narrative; subclasses set ``Graph`` and the expected values that
    differ between undirected and directed graphs.

    The network is the line A--B--C--D with unit costs.  Vertices are 0.5
    apart so the default Euclidean heuristic stays admissible for A*.
    """

    Graph: type
    # connectivity() / degree-matrix diagonal, in vertex creation order
    deg_base: list
    deg_loop: list
    deg_parallel: list
    deg_both: list
    # incidence column sum of a self-loop (2 undirected, 1 directed)
    loop_colsum: int
    # adjacency A[B, B] for a single self-loop on B
    loop_adjacency: int
    directed: bool

    # ------------------------------------------------------------------ #

    def setUp(self):
        g = self.Graph()
        self.g = g
        self.A = g.add_vertex([0.0, 0], name="A")
        self.B = g.add_vertex([0.5, 0], name="B")
        self.C = g.add_vertex([1.0, 0], name="C")
        self.D = g.add_vertex([1.5, 0], name="D")
        self.AB = self.A.connect(self.B, cost=1.0)
        self.BC = self.B.connect(self.C, cost=1.0)
        self.CD = self.C.connect(self.D, cost=1.0)

    def add_loop(self, at=None):
        v = self.B if at is None else at
        return v.connect(v, cost=2.0)

    def add_parallels(self):
        # AB already exists with cost 1.0; the cheap one is added last on
        # purpose, so "first found" and "last written" both give a wrong answer
        self.A.connect(self.B, cost=3.0)
        self.A.connect(self.B, cost=0.5)

    # ------------------------------------------------------------------ #
    # generic checks

    def check_degrees(self, expected):
        g = self.g
        self.assertEqual(g.connectivity(), expected)
        np.testing.assert_array_equal(np.diag(g.degree()), expected)
        self.assertAlmostEqual(g.average_degree(), sum(expected) / g.n)

    def check_matrices_consistent(self):
        """Relationships that must hold for *any* edge multiset."""
        g = self.g
        deg = np.array(g.connectivity())

        A = g.adjacency()
        np.testing.assert_array_equal(A.sum(axis=1), deg)
        if not self.directed:
            np.testing.assert_array_equal(A, A.T)

        L = g.Laplacian()
        np.testing.assert_array_equal(L, g.degree() - A)
        np.testing.assert_allclose(L.sum(axis=1), 0)

        I = g.incidence()
        self.assertEqual(I.shape, (g.n, g.ne))
        if not self.directed:
            # an undirected incidence row sums to the vertex degree
            np.testing.assert_array_equal(I.sum(axis=1), deg)

    def check_baseline(self):
        g = self.g
        self.assertEqual((g.n, g.ne, g.nc), (4, 3, 1))
        self.assertFalse(g.iscyclic())
        self.check_degrees(self.deg_base)
        self.check_matrices_consistent()

        A = g.adjacency()
        self.assertEqual(A.sum(), 6 if not self.directed else 3)
        self.assertEqual(A.max(), 1)
        self.assertEqual(np.trace(A), 0)
        np.testing.assert_array_equal(g.incidence().sum(axis=0), [2, 2, 2])

        D = g.distance()
        self.assertEqual(D[0, 1], 1.0)
        self.assertEqual(D[1, 2], 1.0)
        self.assertEqual(D[2, 3], 1.0)

        self.check_planning(length=2.0)

    def check_planning(self, length):
        """Every planner must reach C from A via B, with this length."""
        g, A, B, C = self.g, self.A, self.B, self.C
        names = lambda path: [v.name for v in path]

        with within():
            path, cost, _ = g.path_UCS(A, C)
        self.assertEqual(names(path), ["A", "B", "C"])
        self.assertAlmostEqual(cost, length)

        with within():
            path, cost, _ = g.path_Astar(A, C)
        self.assertEqual(names(path), ["A", "B", "C"])
        self.assertAlmostEqual(cost, length)

        with within():
            path, cost = g.path_BFS(A, C)
        self.assertEqual(names(path), ["A", "B", "C"])
        self.assertAlmostEqual(cost, length)

    # ------------------------------------------------------------------ #
    # the narrative

    def test_baseline(self):
        self.check_baseline()

    def test_self_loop(self):
        g = self.g
        loop = self.add_loop()

        self.assertEqual((g.n, g.ne, g.nc), (4, 4, 1))
        self.assertTrue(g.iscyclic())
        self.check_degrees(self.deg_loop)
        self.check_matrices_consistent()

        A = g.adjacency()
        self.assertEqual(A[1, 1], self.loop_adjacency)
        self.assertEqual(np.trace(A), self.loop_adjacency)

        I = g.incidence()
        self.assertEqual(sorted(I.sum(axis=0)), sorted([2, 2, 2, self.loop_colsum]))

        D = g.distance()
        self.assertEqual(D[1, 1], 2.0)  # cost of the loop
        self.assertEqual(D[0, 1], 1.0)  # others unchanged

        # the loop lies on the route A -> B -> C, planning must ignore it
        self.check_planning(length=2.0)

        # the loop is a real edge of the graph
        self.assertIn(loop, g.edges())

    def test_self_loop_removed_restores_baseline(self):
        loop = self.add_loop()
        self.g.remove_edge(loop)
        self.check_baseline()

    def test_self_loop_off_route(self):
        # loops at the start and beyond the goal never mattered, guard them too
        self.add_loop(self.A)
        self.add_loop(self.D)
        self.assertTrue(self.g.iscyclic())
        self.check_matrices_consistent()
        self.check_planning(length=2.0)

    def test_parallel_edges(self):
        g = self.g
        self.add_parallels()

        self.assertEqual((g.n, g.ne, g.nc), (4, 5, 1))
        # parallel edges are a cycle in an undirected graph, but a repeated
        # arrow in a directed graph is not
        self.assertEqual(g.iscyclic(), not self.directed)
        self.check_degrees(self.deg_parallel)
        self.check_matrices_consistent()

        A = g.adjacency()
        self.assertEqual(A[0, 1], 3)  # three A->B edges, not "connected: 1"
        self.assertEqual(A[1, 2], 1)

        D = g.distance()
        self.assertEqual(D[0, 1], 0.5)  # the cheapest, whatever the order

        # edgeto() is what the planners use to turn a path into a length
        self.assertEqual(self.A.edgeto(self.B).cost, 0.5)

        # route A -> B -> C must use the 0.5 edge: 0.5 + 1.0
        self.check_planning(length=1.5)

    def test_parallel_edges_removed_restores_baseline(self):
        before = set(self.g.edges())
        self.add_parallels()
        for e in set(self.g.edges()) - before:
            self.g.remove_edge(e)
        self.check_baseline()

    def test_parallel_edges_and_self_loop(self):
        g = self.g
        self.add_parallels()
        self.add_loop()

        self.assertEqual((g.n, g.ne, g.nc), (4, 6, 1))
        self.assertTrue(g.iscyclic())
        self.check_degrees(self.deg_both)
        self.check_matrices_consistent()

        A = g.adjacency()
        self.assertEqual(A[0, 1], 3)
        self.assertEqual(A[1, 1], self.loop_adjacency)

        D = g.distance()
        self.assertEqual(D[0, 1], 0.5)
        self.assertEqual(D[1, 1], 2.0)

        self.check_planning(length=1.5)


class TestUGraphMultigraph(_Multigraph, unittest.TestCase):
    Graph = UGraph
    directed = False
    deg_base = [1, 2, 2, 1]
    deg_loop = [1, 4, 2, 1]  # a loop contributes 2
    deg_parallel = [3, 4, 2, 1]
    deg_both = [3, 6, 2, 1]
    loop_colsum = 2
    loop_adjacency = 2

    def test_parallel_edges_bfs(self):
        # BFS minimises hops, but must still report the cheap edge's cost
        self.add_parallels()
        self.add_loop()
        with within():
            path, cost = self.g.path_BFS(self.A, self.C)
        self.assertEqual([v.name for v in path], ["A", "B", "C"])
        self.assertAlmostEqual(cost, 1.5)


class TestDGraphMultigraph(_Multigraph, unittest.TestCase):
    Graph = DGraph
    directed = True
    deg_base = [1, 1, 1, 0]  # out-degree only
    deg_loop = [1, 2, 1, 0]  # a loop contributes 1
    deg_parallel = [3, 1, 1, 0]
    deg_both = [3, 2, 1, 0]
    loop_colsum = 1
    loop_adjacency = 1


if __name__ == "__main__":
    unittest.main()
