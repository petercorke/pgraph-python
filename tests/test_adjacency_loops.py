"""
A non-zero diagonal element of the matrix given to ``Adjacency()`` is a
self-loop, for undirected and directed graphs alike.  The package permits
self-loops everywhere else (``connect``, ``add_edge``), so the factory must
not be the one place that silently drops, or refuses, them.
"""

import unittest

import numpy as np

from pgraph import DGraph, UGraph


class _AdjacencyLoops:
    Graph: type

    def setUp(self):
        #           0  1  2
        self.A = np.array(
            [
                [0, 1, 0],  # 0 -- 1
                [1, 4, 1],  # 1 has a self-loop of cost 4, and -- 2
                [0, 1, 0],
            ]
        )

    def loops(self, g):
        return [e for e in g.edges() if e.v1 is e.v2]

    def test_diagonal_makes_a_self_loop(self):
        g = self.Graph.Adjacency(self.A)

        self.assertEqual(g.n, 3)
        self.assertEqual(g.ne, self.ne_with_loop)
        loops = self.loops(g)
        self.assertEqual(len(loops), 1)
        self.assertIs(loops[0].v1, g[1])
        self.assertEqual(loops[0].cost, 4)

    def test_no_diagonal_no_loop(self):
        A = self.A.copy()
        A[1, 1] = 0
        g = self.Graph.Adjacency(A)
        self.assertEqual(self.loops(g), [])
        self.assertEqual(g.ne, self.ne_without_loop)

    def test_loop_is_a_real_edge(self):
        g = self.Graph.Adjacency(self.A)
        self.assertTrue(g.iscyclic())
        self.assertEqual(g.distance()[1, 1], 4)

    def test_other_validation_unchanged(self):
        with self.assertRaises(ValueError):
            self.Graph.Adjacency(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            self.Graph.Adjacency(np.zeros((2, 2)), names=["a"])
        with self.assertRaises(ValueError):
            self.Graph.Adjacency(np.zeros((2, 2)), coords=np.zeros((3, 2)))


class TestUGraphAdjacencyLoops(_AdjacencyLoops, unittest.TestCase):
    Graph = UGraph
    # the symmetric matrix is read once, upper triangle: 0--1, 1--2 and 1--1
    ne_with_loop = 3
    ne_without_loop = 2

    def test_lower_triangle_still_ignored(self):
        # documented: only the upper triangle (and now the diagonal) is read
        A = np.array([[0, 0], [5, 0]])
        g = UGraph.Adjacency(A)
        self.assertEqual(g.ne, 0)


class TestDGraphAdjacencyLoops(_AdjacencyLoops, unittest.TestCase):
    Graph = DGraph
    # the whole matrix is read: 0->1, 1->0, 1->2, 2->1 and 1->1
    ne_with_loop = 5
    ne_without_loop = 4

    def test_directed_uses_whole_matrix(self):
        A = np.array([[0, 0], [5, 0]])
        g = DGraph.Adjacency(A)
        self.assertEqual(g.ne, 1)
        self.assertEqual(g[1].neighbours(), [g[0]])


if __name__ == "__main__":
    unittest.main()
