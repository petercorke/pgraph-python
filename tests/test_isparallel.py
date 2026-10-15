"""Tests for the graph-level parallel-edge predicate."""

import unittest

from pgraph import DGraph, Graph, UGraph


class TestIsParallel(unittest.TestCase):
    @staticmethod
    def vertices(graph):
        a = graph.add_vertex([0, 0], name="a")
        b = graph.add_vertex([1, 0], name="b")
        c = graph.add_vertex([2, 0], name="c")
        return a, b, c

    def test_empty_and_simple_graphs(self):
        for graph_class in (UGraph, DGraph):
            with self.subTest(graph_class=graph_class):
                graph = graph_class()
                self.assertFalse(graph.isparallel())
                a, b, c = self.vertices(graph)
                self.assertFalse(graph.isparallel())
                graph.add_edge(a, b, cost=1)
                graph.add_edge(b, c, cost=2)
                self.assertFalse(graph.isparallel())
                self.assertEqual(graph.validate(), [])

    def test_repeated_edge_and_removal(self):
        for graph_class in (UGraph, DGraph):
            with self.subTest(graph_class=graph_class):
                graph = graph_class()
                a, b, _ = self.vertices(graph)
                first = graph.add_edge(a, b, cost=1)
                second = graph.add_edge(a, b, cost=2)
                self.assertTrue(graph.isparallel())
                self.assertTrue(graph.isparallel())  # read-only predicate
                graph.remove_edge(second)
                self.assertFalse(graph.isparallel())
                self.assertIn(first, graph.edges())
                self.assertEqual(graph.validate(), [])

    def test_opposite_directions(self):
        for graph_class, expected in ((UGraph, True), (DGraph, False)):
            with self.subTest(graph_class=graph_class):
                graph = graph_class()
                a, b, _ = self.vertices(graph)
                graph.add_edge(a, b, cost=1)
                graph.add_edge(b, a, cost=2)
                self.assertEqual(graph.isparallel(), expected)
                self.assertEqual(graph.validate(), [])

    def test_self_loops(self):
        for graph_class in (UGraph, DGraph):
            with self.subTest(graph_class=graph_class):
                graph = graph_class()
                a, b, _ = self.vertices(graph)
                first = graph.add_edge(a, a, cost=1)
                graph.add_edge(b, b, cost=2)
                self.assertFalse(graph.isparallel())
                second = graph.add_edge(a, a, cost=3)
                self.assertTrue(graph.isparallel())
                graph.remove_edge(first)
                self.assertFalse(graph.isparallel())
                self.assertIn(second, graph.edges())
                self.assertEqual(graph.validate(), [])

    def test_factory_graph_direction(self):
        for directed, expected in ((False, True), (True, False)):
            with self.subTest(directed=directed):
                graph = Graph(directed=directed)
                a, b, _ = self.vertices(graph)
                graph.add_edge(a, b)
                graph.add_edge(b, a)
                self.assertEqual(graph.isparallel(), expected)


if __name__ == "__main__":
    unittest.main()
