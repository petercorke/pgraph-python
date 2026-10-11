"""
``v.connect(w, edge=e)`` with a caller-supplied ``Edge`` must leave the edge
knowing its endpoints, and refuse an edge that already names other vertices.
"""

import unittest

from pgraph import DGraph, Edge, UGraph


class MyEdge(Edge):
    pass


class _ConnectEdge:
    Graph: type

    def setUp(self):
        self.g = self.Graph()
        self.a = self.g.add_vertex([0, 0], name="a")
        self.b = self.g.add_vertex([1, 0], name="b")
        self.c = self.g.add_vertex([2, 0], name="c")
        self.d = self.g.add_vertex([3, 0], name="d")

    def test_bare_edge_gets_endpoints(self):
        e = Edge(cost=7)
        r = self.a.connect(self.b, edge=e)

        self.assertIs(r, e)
        self.assertIs(e.v1, self.a)
        self.assertIs(e.v2, self.b)
        self.assertEqual(self.g.ne, 1)

    def test_bare_edge_is_a_working_edge(self):
        e = MyEdge(cost=7)
        self.a.connect(self.b, edge=e)

        self.assertEqual(self.g.incidence().shape, (4, 1))
        self.assertEqual(self.a.edgeto(self.b).cost, 7)
        self.g.remove_edge(e)  # asserted on v1/v2 not being None
        self.assertEqual(self.g.ne, 0)
        self.assertIsNone(e.v1)

    def test_edge_with_matching_endpoints(self):
        e = MyEdge(self.a, self.b, cost=3)
        self.assertIs(self.a.connect(self.b, edge=e), e)
        self.assertEqual((e.v1, e.v2), (self.a, self.b))

    def test_edge_with_other_endpoints_rejected(self):
        e = Edge(self.a, self.b)
        with self.assertRaises(ValueError):
            self.c.connect(self.d, edge=e)

        # nothing was half-done
        self.assertEqual(self.g.ne, 0)
        self.assertEqual(self.c.edges(), [])
        self.assertEqual(self.d.edges(), [])
        self.assertEqual((e.v1, e.v2), (self.a, self.b))

    def test_edge_with_one_endpoint_matching_rejected(self):
        e = Edge(self.a, self.b)
        with self.assertRaises(ValueError):
            self.a.connect(self.c, edge=e)
        self.assertEqual(self.g.ne, 0)

    def test_edge_connect_method_still_works(self):
        e = MyEdge(cost=2)
        e.connect(self.a, self.b)
        self.assertEqual((e.v1, e.v2), (self.a, self.b))
        self.assertEqual(self.g.ne, 1)


class TestUGraphConnectEdge(_ConnectEdge, unittest.TestCase):
    Graph = UGraph

    def test_reversed_endpoints_are_the_same_undirected_edge(self):
        e = Edge(self.b, self.a)
        self.assertIs(self.a.connect(self.b, edge=e), e)
        self.assertEqual(self.g.ne, 1)
        self.assertIn(self.b, self.a.neighbours())
        self.assertIn(self.a, self.b.neighbours())


class TestDGraphConnectEdge(_ConnectEdge, unittest.TestCase):
    Graph = DGraph

    def test_reversed_endpoints_rejected(self):
        # an arrow b->a is not the arrow a->b
        e = Edge(self.b, self.a)
        with self.assertRaises(ValueError):
            self.a.connect(self.b, edge=e)
        self.assertEqual(self.g.ne, 0)


if __name__ == "__main__":
    unittest.main()
