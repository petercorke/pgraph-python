"""``graph.edges()`` and ``vertex.edges()`` return new lists in order of addition.

In 0.6.5 ``graph.edges()`` returned the graph's own ``set`` (documented as a
list), so it had no order, and either method returned the live container, so
a caller could corrupt the graph by modifying the result.
"""

import unittest

from pgraph import DGraph, UGraph


class _EdgesList:
    cls: type

    def setUp(self):
        self.g = self.cls()
        self.a, self.b, self.c = (self.g.add_vertex() for _ in range(3))
        self.e1 = self.g.add_edge(self.a, self.b)
        self.e2 = self.g.add_edge(self.b, self.c)
        self.e3 = self.g.add_edge(self.a, self.c)

    def test_is_a_list_in_order_of_addition(self):
        edges = self.g.edges()
        self.assertIsInstance(edges, list)
        self.assertEqual(edges, [self.e1, self.e2, self.e3])

    def test_order_survives_removal(self):
        self.g.remove_edge(self.e2)
        e4 = self.g.add_edge(self.b, self.c)
        self.assertEqual(self.g.edges(), [self.e1, self.e3, e4])

    def test_graph_result_is_a_copy(self):
        self.g.edges().clear()
        self.assertEqual(self.g.ne, 3)
        self.assertEqual(self.g.edges(), [self.e1, self.e2, self.e3])
        self.assertEqual(self.g.validate(), [])

    def test_vertex_result_is_a_copy(self):
        self.a.edges().clear()
        self.assertEqual(len(self.a.edges()), 2)
        self.assertEqual(self.g.validate(), [])

    def test_remove_while_iterating_visits_every_edge(self):
        seen = []
        for e in self.g.edges():
            seen.append(e)
            self.g.remove_edge(e)
        self.assertEqual(seen, [self.e1, self.e2, self.e3])
        self.assertEqual(self.g.ne, 0)

    def test_vertex_remove_while_iterating_visits_every_edge(self):
        seen = []
        for e in self.a.edges():
            seen.append(e)
            self.g.remove_edge(e)
        self.assertEqual(seen, [self.e1, self.e3])
        self.assertEqual(self.a.edges(), [])

    def test_parallel_edges_are_both_listed(self):
        e4 = self.g.add_edge(self.a, self.b)
        self.assertEqual(self.g.edges().count(e4), 1)
        self.assertEqual(len(self.g.edges()), 4)
        self.assertEqual(self.g.ne, 4)

    def test_loop_is_one_graph_edge(self):
        loop = self.g.add_edge(self.c, self.c)
        self.assertEqual(self.g.edges().count(loop), 1)
        self.assertEqual(self.g.ne, 4)

    def test_edge_removed_via_edge_leaves_the_list(self):
        self.e1.remove()
        self.assertEqual(self.g.edges(), [self.e2, self.e3])

    def test_vertex_removal_removes_its_edges(self):
        self.g.remove_vertex(self.b)
        self.assertEqual(self.g.edges(), [self.e3])


class TestUGraphEdgesList(_EdgesList, unittest.TestCase):
    cls = UGraph

    def test_loop_is_listed_twice_at_its_vertex(self):
        loop = self.g.add_edge(self.c, self.c)
        self.assertEqual(self.c.edges().count(loop), 2)


class TestDGraphEdgesList(_EdgesList, unittest.TestCase):
    cls = DGraph

    def test_loop_is_listed_once_at_its_vertex(self):
        loop = self.g.add_edge(self.c, self.c)
        self.assertEqual(self.c.edges().count(loop), 1)


if __name__ == "__main__":
    unittest.main()
