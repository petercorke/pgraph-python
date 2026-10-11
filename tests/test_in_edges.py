"""
A directed vertex knows its incoming edges as well as its outgoing ones.

That gives ``predecessors()`` (to go with ``successors()``, which is
``adjacent()``), ``in_degree`` and ``out_degree``, and makes ``degree`` the
graph-theory degree: in plus out, so a self-loop counts 2 and the degrees of a
graph sum to twice its edges.  For an undirected graph in, out and total
degree are all the same.
"""

import random
import unittest

import numpy as np

from pgraph import DGraph, UGraph


def sample_digraph():
    """
        a -> b (x3, parallel)      b -> c      c -> a      d -> d (loop)
    """
    g = DGraph()
    a, b, c, d = (g.add_vertex([i, 0], name=n) for i, n in enumerate("abcd"))
    for _ in range(3):
        a.connect(b)
    b.connect(c)
    c.connect(a)
    d.connect(d)
    return g, a, b, c, d


class TestDirectedDegrees(unittest.TestCase):
    def setUp(self):
        self.g, self.a, self.b, self.c, self.d = sample_digraph()

    def test_out_degree(self):
        self.assertEqual([v.out_degree for v in self.g], [3, 1, 1, 1])

    def test_in_degree(self):
        self.assertEqual([v.in_degree for v in self.g], [1, 3, 1, 1])

    def test_degree_is_in_plus_out(self):
        self.assertEqual([v.degree for v in self.g], [4, 4, 2, 2])
        for v in self.g:
            self.assertEqual(v.degree, v.in_degree + v.out_degree)

    def test_a_self_loop_counts_two(self):
        self.assertEqual(self.d.in_degree, 1)
        self.assertEqual(self.d.out_degree, 1)
        self.assertEqual(self.d.degree, 2)

    def test_handshake(self):
        # every edge has two ends
        self.assertEqual(sum(v.degree for v in self.g), 2 * self.g.ne)
        self.assertEqual(sum(v.in_degree for v in self.g), self.g.ne)
        self.assertEqual(sum(v.out_degree for v in self.g), self.g.ne)

    def test_graph_degree_matrix_is_the_total(self):
        np.testing.assert_array_equal(np.diag(self.g.degree()), [4, 4, 2, 2])
        self.assertEqual(self.g.connectivity(), [4, 4, 2, 2])

    def test_average_degree(self):
        self.assertAlmostEqual(self.g.average_degree(), 2 * self.g.ne / self.g.n)

    def test_laplacian_still_has_zero_row_sums(self):
        # a directed Laplacian uses the OUT-degree: D_out - A
        L = self.g.Laplacian()
        np.testing.assert_allclose(L.sum(axis=1), 0)
        np.testing.assert_array_equal(
            L, np.diag([v.out_degree for v in self.g]) - self.g.adjacency()
        )

    def test_edges_is_still_outgoing(self):
        self.assertEqual(len(self.a.edges()), 3)
        self.assertEqual(len(self.b.edges()), 1)


class TestPredecessorsSuccessors(unittest.TestCase):
    def setUp(self):
        self.g, self.a, self.b, self.c, self.d = sample_digraph()

    def names(self, vs):
        return sorted(v.name for v in vs)

    def test_successors_are_the_adjacent_vertices(self):
        self.assertEqual(self.a.successors(), self.a.adjacent())
        self.assertEqual(self.names(self.a.successors()), ["b", "b", "b"])

    def test_predecessors(self):
        self.assertEqual(self.names(self.b.predecessors()), ["a", "a", "a"])
        self.assertEqual(self.names(self.a.predecessors()), ["c"])
        self.assertEqual(self.names(self.c.predecessors()), ["b"])

    def test_loop_is_its_own_predecessor_and_successor(self):
        self.assertEqual(self.d.predecessors(), [self.d])
        self.assertEqual(self.d.successors(), [self.d])

    def test_in_edges_and_out_edges(self):
        self.assertEqual(len(self.b.in_edges()), 3)
        self.assertTrue(all(e.v2 is self.b for e in self.b.in_edges()))
        self.assertEqual(len(self.a.out_edges()), 3)
        self.assertTrue(all(e.v1 is self.a for e in self.a.out_edges()))

    def test_results_are_copies(self):
        self.b.predecessors().clear()
        self.b.in_edges().clear()
        self.assertEqual(self.b.in_degree, 3)

    def test_predecessor_of_a_source_is_empty(self):
        g = DGraph()
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        a.connect(b)
        self.assertEqual(a.predecessors(), [])
        self.assertEqual(b.successors(), [])


class TestUndirected(unittest.TestCase):
    def setUp(self):
        self.g = UGraph()
        self.a, self.b, self.c = (self.g.add_vertex(name=n) for n in "abc")
        self.a.connect(self.b)
        self.a.connect(self.b)
        self.b.connect(self.c)
        self.c.connect(self.c)

    def test_in_out_and_total_agree(self):
        for v in self.g:
            self.assertEqual(v.in_degree, v.degree)
            self.assertEqual(v.out_degree, v.degree)
        self.assertEqual([v.degree for v in self.g], [2, 3, 3])  # loop counts 2

    def test_predecessors_and_successors_are_adjacent(self):
        for v in self.g:
            self.assertEqual(v.predecessors(), v.adjacent())
            self.assertEqual(v.successors(), v.adjacent())
        self.assertEqual(len(self.b.in_edges()), 3)
        self.assertEqual(len(self.b.out_edges()), 3)

    def test_handshake(self):
        self.assertEqual(sum(v.degree for v in self.g), 2 * self.g.ne)


class TestInEdgeListStaysConsistent(unittest.TestCase):
    """the in-list is a second record of every edge; it must never drift"""

    def check(self, g):
        out_ids = {id(e) for v in g for e in v._edgelist}
        in_ids = {id(e) for v in g for e in v._inlist}
        graph_ids = {id(e) for e in g.edges()}
        self.assertEqual(out_ids, graph_ids)
        self.assertEqual(in_ids, graph_ids)
        for v in g:
            self.assertTrue(all(e.v1 is v for e in v._edgelist))
            self.assertTrue(all(e.v2 is v for e in v._inlist))
        self.assertEqual(sum(len(v._inlist) for v in g), g.ne)
        self.assertEqual(sum(len(v._edgelist) for v in g), g.ne)

    def test_after_connect_and_remove(self):
        g, a, b, c, d = sample_digraph()
        self.check(g)
        g.remove_edge(a.edges()[0])
        self.check(g)
        g.remove_vertex(b)
        self.check(g)
        self.assertEqual(c.predecessors(), [])  # b is gone
        self.assertEqual(a.successors(), [])

    def test_removing_a_vertex_clears_edges_in_both_directions(self):
        g, a, b, c, d = sample_digraph()
        g.remove_vertex(a)  # has 3 out, 1 in
        self.assertEqual(g.ne, 2)  # b->c and d->d remain
        self.check(g)
        self.assertEqual(b.in_degree, 0)
        self.assertEqual(c.out_degree, 0)

    def test_random_sequences(self):
        rng = random.Random(7)  # nosec B311 - seeded test fuzzing, not security
        for trial in range(25):
            g = DGraph()
            vs = [g.add_vertex([i, 0]) for i in range(8)]
            for _ in range(rng.randrange(0, 40)):
                op = rng.random()
                if op < 0.65 or g.ne == 0:
                    rng.choice(vs).connect(rng.choice(vs))
                elif op < 0.9:
                    g.remove_edge(rng.choice(sorted(g.edges(), key=id)))
                elif g.n > 2:
                    v = rng.choice(list(g))
                    g.remove_vertex(v)
                    vs = [x for x in vs if x is not v]
                self.check(g)
            self.assertEqual(sum(v.degree for v in g), 2 * g.ne)

    def test_deepcopy_keeps_it_consistent(self):
        g, *_ = sample_digraph()
        h = g.copy()
        self.check(h)
        self.assertIsNot(h[0]._inlist, g[0]._inlist)


if __name__ == "__main__":
    unittest.main()
