"""``incidence()`` cell by cell.

``I[i, j]`` is 1 where vertex ``i`` is an end of edge ``j``: rows are vertices
in the order they were created, columns are edges in the order they were added
(the order of ``edges()``).  Both ends of an edge are marked whichever way it
points.  An undirected self-loop touches its vertex at both ends, so is 2.

0.6.5 returned the wrong matrix for every graph with an edge.  A loop variable
overwrote the vertex index, so edges were marked in the wrong rows, and for a
directed graph only each vertex's outgoing edges were visited, so the vertex at
the end of an edge was not marked.  The tests that existed checked only column
and row sums, because the edge order was then that of a set, so neither bug was
caught.  Each expectation below is written out by hand, not derived from the
code under test.
"""

import unittest

import numpy as np

from pgraph import DGraph, UGraph


def matrix(rows):
    return np.array(rows, dtype=float)


class _Incidence:
    cls: type

    def graph(self, names):
        g = self.cls()
        return g, [g.add_vertex(name=n) for n in names]

    def assertIncidence(self, g, expected):
        np.testing.assert_array_equal(g.incidence(), matrix(expected))

    # ---------------------------------------------------------------- shape

    def test_empty_graph(self):
        g = self.cls()
        self.assertEqual(g.incidence().shape, (0, 0))

    def test_vertices_without_edges(self):
        g, _ = self.graph("abc")
        self.assertEqual(g.incidence().shape, (3, 0))

    # ------------------------------------------------------ the two 0.6.5 bugs

    def test_path(self):
        # a - b - c: the middle vertex is on both edges
        g, (a, b, c) = self.graph("abc")
        g.add_edge(a, b)
        g.add_edge(b, c)
        self.assertIncidence(g, [[1, 0],
                                 [1, 1],
                                 [0, 1]])

    def test_star(self):
        # a hub with three spokes: the hub row is all ones
        g, (h, x, y, z) = self.graph("hxyz")
        g.add_edge(h, x)
        g.add_edge(h, y)
        g.add_edge(h, z)
        self.assertIncidence(g, [[1, 1, 1],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])

    def test_the_end_of_an_edge_is_marked_whichever_way_it_points(self):
        # edges that run backwards through the alphabet: the vertex at the
        # head of each is marked as well as the one at the tail
        g, (a, b, c) = self.graph("abc")
        g.add_edge(c, b)
        g.add_edge(b, a)
        self.assertIncidence(g, [[0, 1],
                                 [1, 1],
                                 [1, 0]])

    def test_a_vertex_that_is_only_ever_an_end(self):
        # in a directed graph c has no outgoing edge, so a visit to each
        # vertex's own edges never finds it
        g, (a, b, c) = self.graph("abc")
        g.add_edge(a, c)
        g.add_edge(b, c)
        self.assertIncidence(g, [[1, 0],
                                 [0, 1],
                                 [1, 1]])

    # ------------------------------------------------------------- the order

    def test_rows_follow_creation_order_not_names(self):
        g, (z, a, m) = self.graph("zam")
        g.add_edge(z, a)
        g.add_edge(a, m)
        self.assertIncidence(g, [[1, 0],     # z
                                 [1, 1],     # a
                                 [0, 1]])    # m

    def test_columns_follow_the_order_edges_were_added(self):
        g, (a, b, c, d) = self.graph("abcd")
        g.add_edge(c, d)
        g.add_edge(a, b)
        g.add_edge(b, c)
        self.assertIncidence(g, [[0, 1, 0],
                                 [0, 1, 1],
                                 [1, 0, 1],
                                 [1, 0, 0]])

    def test_columns_are_edges_in_edges_order(self):
        g, (a, b, c, d) = self.graph("abcd")
        edges = [g.add_edge(a, b), g.add_edge(c, d), g.add_edge(b, c), g.add_edge(a, d)]
        self.assertEqual(g.edges(), edges)
        I = g.incidence()
        index = {v: i for i, v in enumerate(g)}
        for j, e in enumerate(edges):
            marked = set(np.flatnonzero(I[:, j]))
            self.assertEqual(marked, {index[e.v1], index[e.v2]})

    def test_connect_and_add_edge_agree(self):
        g, (a, b, c) = self.graph("abc")
        a.connect(b)
        g.add_edge(b, c)
        self.assertIncidence(g, [[1, 0],
                                 [1, 1],
                                 [0, 1]])

    # ----------------------------------------------- multigraph and removal

    def test_parallel_edges_have_a_column_each(self):
        g, (a, b, c) = self.graph("abc")
        g.add_edge(a, b)
        g.add_edge(a, b)
        g.add_edge(b, c)
        self.assertIncidence(g, [[1, 1, 0],
                                 [1, 1, 1],
                                 [0, 0, 1]])

    def test_antiparallel_edges_have_a_column_each(self):
        g, (a, b) = self.graph("ab")
        g.add_edge(a, b)
        g.add_edge(b, a)
        self.assertIncidence(g, [[1, 1],
                                 [1, 1]])

    def test_columns_close_up_when_an_edge_is_removed(self):
        g, (a, b, c) = self.graph("abc")
        e1 = g.add_edge(a, b)
        g.add_edge(b, c)
        g.remove_edge(e1)
        self.assertIncidence(g, [[0],
                                 [1],
                                 [1]])

    def test_rows_close_up_when_a_vertex_is_removed(self):
        # a - b - c - d: removing b takes a-b and b-c with it, leaving c-d
        g, (a, b, c, d) = self.graph("abcd")
        g.add_edge(a, b)
        g.add_edge(b, c)
        g.add_edge(c, d)
        g.remove_vertex(b)
        self.assertIncidence(g, [[0],      # a, now isolated
                                 [1],      # c
                                 [1]])     # d

    def test_an_isolated_vertex_is_a_row_of_zeros(self):
        g, (a, b, c) = self.graph("abc")
        g.add_edge(a, c)
        self.assertIncidence(g, [[1],
                                 [0],
                                 [1]])


class TestUndirectedIncidence(_Incidence, unittest.TestCase):
    cls = UGraph

    def test_a_loop_touches_its_vertex_at_both_ends(self):
        g, (a, b) = self.graph("ab")
        g.add_edge(a, a)
        g.add_edge(a, b)
        self.assertIncidence(g, [[2, 1],
                                 [0, 1]])


class TestDirectedIncidence(_Incidence, unittest.TestCase):
    cls = DGraph

    def test_a_loop_is_one_in_its_row(self):
        # the incidence matrix does not say which end is which, so a directed
        # loop is a single 1; the in and out degree of the vertex say more
        g, (a, b) = self.graph("ab")
        g.add_edge(a, a)
        g.add_edge(a, b)
        self.assertIncidence(g, [[1, 1],
                                 [0, 1]])


if __name__ == "__main__":
    unittest.main()
