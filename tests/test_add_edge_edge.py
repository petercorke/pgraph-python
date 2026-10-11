"""
``graph.add_edge()`` accepts either two vertices (or names) or an ``Edge``
that already carries its vertices, mirroring ``add_vertex()`` which accepts
coordinates or a vertex.
"""

import unittest

from pgraph import DGraph, Edge, UGraph, UVertex


class MyEdge(Edge):
    pass


class _AddEdge:
    Graph: type

    def setUp(self):
        self.g = self.Graph()
        self.a = self.g.add_vertex([0, 0], name="a")
        self.b = self.g.add_vertex([3, 4], name="b")
        self.c = self.g.add_vertex([6, 8], name="c")

    def test_two_vertices_unchanged(self):
        e = self.g.add_edge(self.a, self.b)
        self.assertIs(type(e), Edge)
        self.assertEqual((e.v1, e.v2), (self.a, self.b))
        self.assertEqual(e.cost, 5.0)

    def test_two_names_unchanged(self):
        e = self.g.add_edge("a", "b", cost=9)
        self.assertEqual((e.v1, e.v2, e.cost), (self.a, self.b, 9))

    def test_edge_subclass(self):
        e = MyEdge(self.a, self.b, cost=2, data="hello")
        r = self.g.add_edge(e)

        self.assertIs(r, e)
        self.assertIsInstance(r, MyEdge)
        self.assertEqual(self.g.ne, 1)
        self.assertIn(e, self.g.edges())
        self.assertIs(self.a.edgeto(self.b), e)
        self.assertEqual(e.data, "hello")

    def test_edge_cost_computed_at_construction(self):
        r = self.g.add_edge(MyEdge(self.a, self.b))
        self.assertEqual(r.cost, 5.0)

    def test_edge_must_carry_vertices(self):
        with self.assertRaises(ValueError) as cm:
            self.g.add_edge(MyEdge(cost=1))
        self.assertIn("edge=", str(cm.exception))  # points at the alternative
        with self.assertRaises(ValueError):
            self.g.add_edge(MyEdge(self.a))  # only one end
        self.assertEqual(self.g.ne, 0)

    def test_edge_and_second_vertex_is_an_error(self):
        with self.assertRaises(TypeError):
            self.g.add_edge(MyEdge(self.a, self.b), self.c)
        self.assertEqual(self.g.ne, 0)

    def test_edge_with_cost_or_data_kwargs_is_an_error(self):
        # connect() would silently ignore them when given an edge
        e = MyEdge(self.a, self.b)
        with self.assertRaises(TypeError):
            self.g.add_edge(e, cost=3)
        with self.assertRaises(TypeError):
            self.g.add_edge(e, data="x")
        self.assertEqual(self.g.ne, 0)

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            self.g.add_edge(42, self.b)
        with self.assertRaises(TypeError):
            self.g.add_edge(self.a, 42)

    def test_vertices_of_another_graph(self):
        h = self.Graph()
        x = h.add_vertex([0, 0], name="x")
        y = h.add_vertex([1, 1], name="y")
        with self.assertRaises(ValueError):
            self.g.add_edge(MyEdge(x, y))
        with self.assertRaises(ValueError):
            self.g.add_edge(x, y)
        self.assertEqual(self.g.ne, 0)
        self.assertEqual(h.ne, 0)

    def test_edge_between_ungraphed_vertices(self):
        v, w = UVertex([0, 0]), UVertex([1, 1])
        with self.assertRaises(ValueError):
            self.g.add_edge(MyEdge(v, w))


class TestUGraphAddEdge(_AddEdge, unittest.TestCase):
    Graph = UGraph


class TestDGraphAddEdge(_AddEdge, unittest.TestCase):
    Graph = DGraph

    def test_direction_follows_edge(self):
        self.g.add_edge(MyEdge(self.b, self.a))
        self.assertEqual(self.b.neighbours(), [self.a])
        self.assertEqual(self.a.neighbours(), [])


if __name__ == "__main__":
    unittest.main()
