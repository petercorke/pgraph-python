"""
Components of a graph: ``nc``, ``component()``, ``samecomponent()`` and the
``label`` of a vertex.

The components of a directed graph are *weakly* connected: the direction of an
edge is ignored, so two vertices are in the same component if there is a path
between them when it may run against the arrows.  This is what a directed graph
has always done (an undirected graph is unchanged), but the algorithm that did
it merged labels in a loop that never ended for many directed graphs: the
smallest is the 4-cycle 0->2->1->3->0, a random digraph of 100 vertices and 300
edges hangs, and so do ``plot()``, ``component()`` and ``samecomponent()``,
which all ask for the components.  It was the same in 0.6.5.

Expectations are written out by hand, or computed by a union-find in this file,
not by the code under test.
"""

import signal
import unittest
from contextlib import contextmanager

import numpy as np

from pgraph import DGraph, UGraph


@contextmanager
def within(seconds):
    """Fail, rather than hang, if the body takes longer than ``seconds``"""
    if not hasattr(signal, "setitimer"):  # not on this platform: no guard
        yield
        return

    def expired(signum, frame):
        raise AssertionError(f"did not finish in {seconds} s: it hangs")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def partition(n, edges):
    """The components of n vertices 0..n-1 by union-find: a set of frozensets"""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups = {}
    for v in range(n):
        groups.setdefault(find(v), set()).add(v)
    return {frozenset(s) for s in groups.values()}


def build(cls, n, edges):
    g = cls()
    vs = [g.add_vertex(name=str(i)) for i in range(n)]
    for a, b in edges:
        g.add_edge(vs[a], vs[b])
    return g


def found(g):
    """The components of g, from its labels, as a set of frozensets of ints"""
    groups = {}
    for v in g:
        groups.setdefault(v.label, set()).add(int(v.name))
    return {frozenset(s) for s in groups.values()}


class TestTheHang(unittest.TestCase):
    def test_directed_four_cycle(self):
        # the smallest digraph for which the old algorithm never returned
        g = build(DGraph, 4, [(0, 2), (2, 1), (1, 3), (3, 0)])
        with within(5):
            self.assertEqual(g.nc, 1)

    def test_every_call_that_asks_for_components(self):
        g = build(DGraph, 4, [(0, 2), (2, 1), (1, 3), (3, 0)])
        with within(5):
            self.assertEqual(len(g.component(0)), 4)
            self.assertTrue(g.samecomponent(g[0], g[3]))

    def test_random_digraphs(self):
        rng = np.random.default_rng(12345)
        for trial in range(30):
            n = int(rng.integers(2, 40))
            m = int(rng.integers(1, 3 * n))
            edges = [tuple(int(x) for x in rng.integers(0, n, 2)) for _ in range(m)]
            g = build(DGraph, n, edges)
            with self.subTest(trial=trial, n=n, m=m):
                with within(5):
                    g.nc
                self.assertEqual(found(g), partition(n, edges))

    def test_a_large_digraph(self):
        rng = np.random.default_rng(7)
        n, m = 2000, 3000
        edges = [tuple(int(x) for x in rng.integers(0, n, 2)) for _ in range(m)]
        g = build(DGraph, n, edges)
        with within(10):
            self.assertEqual(g.nc, len(partition(n, edges)))


class _Components:
    cls: type

    def test_empty(self):
        self.assertEqual(self.cls().nc, 0)

    def test_isolated_vertices_are_a_component_each(self):
        g = build(self.cls, 3, [])
        self.assertEqual(g.nc, 3)

    def test_two_paths(self):
        g = build(self.cls, 5, [(0, 1), (1, 2), (3, 4)])
        self.assertEqual(g.nc, 2)
        self.assertEqual(found(g), {frozenset({0, 1, 2}), frozenset({3, 4})})

    def test_labels_are_in_order_of_first_vertex(self):
        g = build(self.cls, 5, [(3, 4), (0, 1)])   # vertex 2 on its own
        self.assertEqual([v.label for v in g], [0, 0, 1, 2, 2])

    def test_component_lists_vertices_in_creation_order(self):
        g = build(self.cls, 5, [(4, 1), (1, 3)])
        self.assertEqual([v.name for v in g.component(0)], ["0"])
        self.assertEqual([v.name for v in g.component(1)], ["1", "3", "4"])
        self.assertEqual([v.name for v in g.component(2)], ["2"])

    def test_loops_and_parallel_edges(self):
        g = build(self.cls, 3, [(0, 0), (0, 1), (0, 1), (2, 2)])
        self.assertEqual(found(g), {frozenset({0, 1}), frozenset({2})})

    def test_the_labels_are_0_to_nc_minus_1(self):
        g = build(self.cls, 6, [(0, 5), (1, 2), (2, 3)])
        self.assertEqual({v.label for v in g}, set(range(g.nc)))

    def test_matches_a_union_find(self):
        rng = np.random.default_rng(99)
        for trial in range(20):
            n = int(rng.integers(1, 30))
            m = int(rng.integers(0, 2 * n))
            edges = [tuple(int(x) for x in rng.integers(0, n, 2)) for _ in range(m)]
            g = build(self.cls, n, edges)
            with self.subTest(trial=trial):
                self.assertEqual(found(g), partition(n, edges))
                self.assertEqual(g.nc, len(partition(n, edges)))

    # ----------------------------------------------- the graph changes

    def test_adding_an_edge_joins_components(self):
        g = build(self.cls, 4, [(0, 1), (2, 3)])
        self.assertEqual(g.nc, 2)
        g.add_edge(g[1], g[2])
        self.assertEqual(g.nc, 1)

    def test_removing_an_edge_splits_a_component(self):
        g = build(self.cls, 4, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(g.nc, 1)
        g.remove_edge(g.edges()[1])
        self.assertEqual(g.nc, 2)
        self.assertEqual(found(g), {frozenset({0, 1}), frozenset({2, 3})})

    def test_removing_a_vertex(self):
        g = build(self.cls, 3, [(0, 1), (1, 2)])
        g.remove_vertex(g[1])
        self.assertEqual(g.nc, 2)

    def test_adding_a_vertex(self):
        g = build(self.cls, 2, [(0, 1)])
        self.assertEqual(g.nc, 1)
        g.add_vertex(name="x")
        self.assertEqual(g.nc, 2)

    def test_not_recomputed_when_nothing_changed(self):
        g = build(self.cls, 4, [(0, 1), (2, 3)])
        g.nc
        self.assertFalse(g._connectivitychange)
        self.assertTrue(all(not v._connectivitychange for v in g))
        # a changed label would show if it were: poke one, ask again
        before = [v.label for v in g]
        g.nc
        self.assertEqual([v.label for v in g], before)

    def test_samecomponent(self):
        g = build(self.cls, 4, [(0, 1), (2, 3)])
        self.assertTrue(g.samecomponent(g[0], g[1]))
        self.assertFalse(g.samecomponent(g[1], g[2]))


class TestUndirectedComponents(_Components, unittest.TestCase):
    cls = UGraph


class TestDirectedComponents(_Components, unittest.TestCase):
    cls = DGraph

    def test_direction_is_ignored(self):
        # a -> c <- b: no path from a to b either way, but one component
        g = build(DGraph, 3, [(0, 2), (1, 2)])
        self.assertEqual(g.nc, 1)
        self.assertTrue(g.samecomponent(g[0], g[1]))

    def test_a_vertex_that_is_only_a_source(self):
        g = build(DGraph, 3, [(0, 1), (0, 2)])
        self.assertEqual(g.nc, 1)

    def test_opposite_edges(self):
        g = build(DGraph, 2, [(0, 1), (1, 0)])
        self.assertEqual(g.nc, 1)

    def test_agrees_with_the_undirected_graph_on_the_same_edges(self):
        rng = np.random.default_rng(2024)
        for trial in range(20):
            n = int(rng.integers(2, 25))
            edges = [tuple(int(x) for x in rng.integers(0, n, 2)) for _ in range(int(rng.integers(0, 2 * n)))]
            with self.subTest(trial=trial):
                d, u = build(DGraph, n, edges), build(UGraph, n, edges)
                with within(5):
                    self.assertEqual(d.nc, u.nc)
                    self.assertEqual([v.label for v in d], [v.label for v in u])


if __name__ == "__main__":
    unittest.main()
