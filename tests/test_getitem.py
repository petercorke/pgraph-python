"""
``graph[x]`` resolves an index, a name or a vertex, and says so clearly when it
cannot -- it must never quietly hand back ``None``.
"""

import unittest

import numpy as np

from pgraph import DGraph, UGraph


class _GetItem:
    Graph: type

    def setUp(self):
        self.g = self.Graph()
        self.a = self.g.add_vertex([0, 0], name="a")
        self.b = self.g.add_vertex([1, 0], name="b")
        self.c = self.g.add_vertex([2, 0], name="c")

    def test_int(self):
        self.assertIs(self.g[0], self.a)
        self.assertIs(self.g[2], self.c)
        self.assertIs(self.g[-1], self.c)

    def test_numpy_integer(self):
        # e.g. the result of np.argmin(), a very common way to get an index
        self.assertIs(self.g[np.int64(1)], self.b)
        self.assertIs(self.g[np.argmin([3.0, 1.0, 2.0])], self.b)

    def test_name(self):
        self.assertIs(self.g["b"], self.b)

    def test_vertex(self):
        self.assertIs(self.g[self.c], self.c)

    def test_unknown_name(self):
        with self.assertRaises(KeyError):
            self.g["zz"]

    def test_index_out_of_range(self):
        with self.assertRaises(IndexError):
            self.g[3]
        with self.assertRaises(IndexError):
            self.g[-4]

    def test_wrong_type(self):
        for bad in (1.5, None, [0], (0,), slice(0, 2), 1 + 2j):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.g[bad]

    def test_membership_and_iteration_unaffected(self):
        self.assertIn("a", self.g)
        self.assertEqual([v.name for v in self.g], ["a", "b", "c"])


class TestUGraphGetItem(_GetItem, unittest.TestCase):
    Graph = UGraph


class TestDGraphGetItem(_GetItem, unittest.TestCase):
    Graph = DGraph


if __name__ == "__main__":
    unittest.main()
