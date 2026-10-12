"""
A graph can be told which vertex and edge classes it builds, so user
subclasses never have to be constructed by hand::

    g = UGraph(vertex_class=MyVertex, edge_class=MyEdge)
    g.add_vertex([0, 0])     # a MyVertex
    g.add_edge(v1, v2)       # a MyEdge
"""

import unittest

import numpy as np

from pgraph import DGraph, Edge, UGraph, Vertex


class MyEdge(Edge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traversals = 0


class OtherVertex(Vertex):
    """A vertex class that is not the one a graph under test is configured for"""


class _Classes:
    Graph: type
    Vertex: type
    OtherVertex: type

    def setUp(self):
        class MyVertex(self.Vertex):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.visited = False

        self.MyVertex = MyVertex
        self.g = self.Graph(vertex_class=MyVertex, edge_class=MyEdge)

    # ------------------------------------------------------------------ #

    def test_defaults(self):
        g = self.Graph()
        self.assertIs(g.vertex_class, self.Vertex)
        self.assertIs(g.edge_class, Edge)

        v = g.add_vertex([0, 0])
        self.assertIs(type(v), self.Vertex)
        self.assertIs(type(g.add_edge(v, g.add_vertex([1, 1]))), Edge)

    def test_properties(self):
        self.assertIs(self.g.vertex_class, self.MyVertex)
        self.assertIs(self.g.edge_class, MyEdge)

    def test_add_vertex_builds_vertex_class(self):
        v = self.g.add_vertex([1, 2], name="p")
        self.assertIs(type(v), self.MyVertex)
        self.assertFalse(v.visited)  # subclass __init__ ran
        self.assertEqual(v.name, "p")
        np.testing.assert_array_equal(v.coord, [1, 2])

        w = self.g.add_vertex()
        self.assertIs(type(w), self.MyVertex)
        self.assertEqual(w.name, "#1")

    def test_add_edge_builds_edge_class(self):
        a = self.g.add_vertex([0, 0])
        b = self.g.add_vertex([3, 4])

        e = self.g.add_edge(a, b)
        self.assertIs(type(e), MyEdge)
        self.assertEqual(e.cost, 5.0)  # still computed from the metric
        self.assertEqual(e.traversals, 0)

        e2 = a.connect(b, cost=9, data="road")
        self.assertIs(type(e2), MyEdge)
        self.assertEqual((e2.cost, e2.data), (9, "road"))

        self.assertTrue(all(type(x) is MyEdge for x in self.g.edges()))

    def test_add_vertex_accepts_instances_of_vertex_class(self):
        v = self.MyVertex([5, 5], name="mine")
        self.assertIs(self.g.add_vertex(v), v)

    def test_add_vertex_rejects_plain_vertex_of_right_kind(self):
        # the graph was configured to hold MyVertex, not just any Vertex
        with self.assertRaises(TypeError):
            self.g.add_vertex(self.Vertex([5, 5]))
        self.assertEqual(self.g.n, 0)

    def test_add_vertex_rejects_a_vertex_of_another_class(self):
        with self.assertRaises(TypeError):
            self.g.add_vertex(self.OtherVertex([5, 5]))
        self.assertEqual(self.g.n, 0)

    def test_connect_rejects_edge_of_wrong_class(self):
        a = self.g.add_vertex([0, 0])
        b = self.g.add_vertex([1, 1])
        with self.assertRaises(TypeError):
            a.connect(b, edge=Edge(cost=1))
        self.assertEqual(self.g.ne, 0)

        mine = MyEdge(cost=1)
        self.assertIs(a.connect(b, edge=mine), mine)

    def test_default_graph_still_accepts_subclasses(self):
        g = self.Graph()
        self.assertIsInstance(g.add_vertex(self.MyVertex([0, 0])), self.MyVertex)
        a, b = g[0], g.add_vertex([1, 1])
        self.assertIsInstance(a.connect(b, edge=MyEdge(a, b)), MyEdge)

    def test_bad_vertex_class(self):
        for bad in (int, "Vertex", self.Vertex([0, 0]), Edge):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.Graph(vertex_class=bad)

    def test_bad_edge_class(self):
        for bad in (Vertex, int, "Edge", Edge(cost=1)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.Graph(edge_class=bad)

    def test_copy_keeps_classes(self):
        a = self.g.add_vertex([0, 0])
        b = self.g.add_vertex([1, 1])
        self.g.add_edge(a, b)

        h = self.g.copy()
        self.assertIs(h.vertex_class, self.MyVertex)
        self.assertIs(h.edge_class, MyEdge)
        self.assertIs(type(h[0]), self.MyVertex)
        c = h.add_vertex([2, 2])
        self.assertIs(type(c), self.MyVertex)
        self.assertIs(type(h.add_edge(h[0], c)), MyEdge)

    def test_constructors_forward_classes(self):
        g = self.Graph.Dict(
            {"b": "a", "c": "a"}, vertex_class=self.MyVertex, edge_class=MyEdge
        )
        self.assertEqual(g.n, 3)
        self.assertTrue(all(type(v) is self.MyVertex for v in g))
        self.assertTrue(all(type(e) is MyEdge for e in g.edges()))

        A = np.array([[0, 1], [1, 0]])
        g = self.Graph.Adjacency(
            A, vertex_class=self.MyVertex, edge_class=MyEdge
        )
        self.assertTrue(all(type(v) is self.MyVertex for v in g))
        self.assertTrue(all(type(e) is MyEdge for e in g.edges()))

    def test_planning_unaffected(self):
        g = self.g
        v = [g.add_vertex([i, 0], name=n) for i, n in enumerate("abc")]
        g.add_edge(v[0], v[1])
        g.add_edge(v[1], v[2])
        path, cost, _ = g.path_UCS(v[0], v[2])
        self.assertEqual([p.name for p in path], ["a", "b", "c"])
        self.assertAlmostEqual(cost, 2.0)


class TestUGraphClasses(_Classes, unittest.TestCase):
    Graph = UGraph
    Vertex = Vertex
    OtherVertex = OtherVertex


class TestDGraphClasses(_Classes, unittest.TestCase):
    Graph = DGraph
    Vertex = Vertex
    OtherVertex = OtherVertex


if __name__ == "__main__":
    unittest.main()
