"""``edges_to(dest)`` lists every edge from a vertex to ``dest``.

``edgeto(dest)`` returns the one edge a path would use, the cheapest of any
parallel edges.  ``edges_to`` is the whole set, so that parallel edges can be
inspected: in connection order, empty rather than an error when there are none,
and with an undirected self-loop listed once, not once per end.
"""

import unittest

from pgraph import DGraph, UGraph


class _EdgesTo:
    Graph: type

    def setUp(self):
        self.g = self.Graph()
        self.a, self.b, self.c = (self.g.add_vertex(name=n) for n in "abc")

    def test_single_edge(self):
        e = self.a.connect(self.b, cost=3)
        self.assertEqual(self.a.edges_to(self.b), [e])

    def test_parallel_edges_in_connection_order(self):
        e1 = self.a.connect(self.b, cost=5)
        e2 = self.a.connect(self.b, cost=2)
        e3 = self.a.connect(self.b, cost=9)
        self.assertEqual(self.a.edges_to(self.b), [e1, e2, e3])

    def test_edgeto_is_the_cheapest_of_edges_to(self):
        edges = [self.a.connect(self.b, cost=c) for c in (5, 2, 9)]
        self.assertIs(self.a.edgeto(self.b), edges[1])
        self.assertIn(self.a.edgeto(self.b), self.a.edges_to(self.b))

    def test_none_is_an_empty_list_not_an_error(self):
        self.a.connect(self.b)
        self.assertEqual(self.a.edges_to(self.c), [])
        with self.assertRaises(ValueError):
            self.a.edgeto(self.c)  # edgeto has no edge to return

    def test_other_destinations_are_not_included(self):
        e = self.a.connect(self.b)
        self.a.connect(self.c)
        self.assertEqual(self.a.edges_to(self.b), [e])

    def test_self_loop_is_listed_once(self):
        loop = self.a.connect(self.a, cost=1)
        self.assertEqual(self.a.edges_to(self.a), [loop])

    def test_loop_and_parallel_loops(self):
        l1 = self.a.connect(self.a)
        l2 = self.a.connect(self.a)
        self.assertEqual(self.a.edges_to(self.a), [l1, l2])

    def test_is_a_new_list(self):
        self.a.connect(self.b)
        self.a.edges_to(self.b).clear()
        self.assertEqual(len(self.a.edges_to(self.b)), 1)
        self.assertEqual(self.g.validate(), [])

    def test_removed_edge_is_gone(self):
        e1 = self.a.connect(self.b)
        e2 = self.a.connect(self.b)
        self.g.remove_edge(e1)
        self.assertEqual(self.a.edges_to(self.b), [e2])

    def test_unconnected_vertex(self):
        self.assertEqual(self.a.edges_to(self.b), [])


class TestUndirectedEdgesTo(_EdgesTo, unittest.TestCase):
    Graph = UGraph

    def test_found_from_either_end(self):
        e = self.a.connect(self.b)
        self.assertEqual(self.a.edges_to(self.b), [e])
        self.assertEqual(self.b.edges_to(self.a), [e])

    def test_edges_connected_from_the_other_end_are_included(self):
        e1 = self.a.connect(self.b)
        e2 = self.b.connect(self.a)
        self.assertEqual(self.a.edges_to(self.b), [e1, e2])
        self.assertEqual(self.b.edges_to(self.a), [e1, e2])


class TestDirectedEdgesTo(_EdgesTo, unittest.TestCase):
    Graph = DGraph

    def test_only_the_direction_given(self):
        e = self.a.connect(self.b)
        self.assertEqual(self.a.edges_to(self.b), [e])
        self.assertEqual(self.b.edges_to(self.a), [])

    def test_opposite_edges_are_separate(self):
        ab = self.a.connect(self.b)
        ba = self.b.connect(self.a)
        self.assertEqual(self.a.edges_to(self.b), [ab])
        self.assertEqual(self.b.edges_to(self.a), [ba])


if __name__ == "__main__":
    unittest.main()
