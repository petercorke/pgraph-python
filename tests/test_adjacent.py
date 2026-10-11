"""
``adjacent()`` is the canonical name for the vertices at the far end of a
vertex's edges, because "adjacent" is the graph-theory term and has one
spelling.  ``neighbours`` and ``neighbors`` (and ``isneighbour`` /
``isneighbor``) remain as silent aliases: they are the same function, not a
second implementation, and they do not warn.
"""

import unittest
import warnings

from pgraph import DGraph, UGraph, DVertex, UVertex
from pgraph.PGraph import BaseVertex


class TestAliases(unittest.TestCase):
    def test_aliases_are_the_same_function(self):
        for cls in (BaseVertex, UVertex, DVertex):
            with self.subTest(cls=cls.__name__):
                self.assertIs(cls.neighbours, cls.adjacent)
                self.assertIs(cls.neighbors, cls.adjacent)
                self.assertIs(cls.isneighbour, cls.isadjacent)
                self.assertIs(cls.isneighbor, cls.isadjacent)

    def test_no_warnings(self):
        g = UGraph()
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        a.connect(b)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning is an error
            for name in ("adjacent", "neighbours", "neighbors"):
                self.assertEqual(getattr(a, name)(), [b])
            for name in ("isadjacent", "isneighbour", "isneighbor"):
                self.assertTrue(getattr(a, name)(b))

    def test_docstring_names_the_aliases(self):
        doc = BaseVertex.adjacent.__doc__
        for name in ("neighbours", "neighbors"):
            self.assertIn(name, doc)


class _Adjacent:
    Graph: type

    def setUp(self):
        self.g = self.Graph()
        self.a = self.g.add_vertex([0, 0], name="a")
        self.b = self.g.add_vertex([1, 0], name="b")
        self.c = self.g.add_vertex([2, 0], name="c")

    def test_simple(self):
        self.a.connect(self.b)
        self.assertEqual(self.a.adjacent(), [self.b])
        self.assertTrue(self.a.isadjacent(self.b))
        self.assertFalse(self.a.isadjacent(self.c))

    def test_returns_a_new_list(self):
        self.a.connect(self.b)
        r = self.a.adjacent()
        r.clear()
        self.assertEqual(self.a.adjacent(), [self.b])

    def test_parallel_edges_listed_once_per_edge(self):
        for _ in range(3):
            self.a.connect(self.b)
        self.assertEqual(self.a.adjacent(), [self.b, self.b, self.b])

    def test_matches_a_row_of_the_adjacency_matrix(self):
        # the point of the name: row i of adjacency() is the vertices adjacent
        # to vertex i, with multiplicity
        self.a.connect(self.b)
        self.a.connect(self.b)
        self.a.connect(self.c)
        A = self.g.adjacency()
        for i, v in enumerate(self.g):
            counts = {w: v.adjacent().count(w) for w in self.g}
            for j, w in enumerate(self.g):
                self.assertEqual(A[i, j], counts[w], (v.name, w.name))


class TestUGraphAdjacent(_Adjacent, unittest.TestCase):
    Graph = UGraph

    def test_both_ends(self):
        self.a.connect(self.b)
        self.assertEqual(self.b.adjacent(), [self.a])
        self.assertTrue(self.b.isadjacent(self.a))

    def test_self_loop_listed_twice(self):
        self.a.connect(self.a)
        self.assertEqual(self.a.adjacent(), [self.a, self.a])


class TestDGraphAdjacent(_Adjacent, unittest.TestCase):
    Graph = DGraph

    def test_outgoing_only(self):
        self.a.connect(self.b)
        self.assertEqual(self.b.adjacent(), [])
        self.assertFalse(self.b.isadjacent(self.a))

    def test_self_loop_listed_once(self):
        self.a.connect(self.a)
        self.assertEqual(self.a.adjacent(), [self.a])


if __name__ == "__main__":
    unittest.main()
