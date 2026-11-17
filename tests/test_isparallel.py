"""Tests for the graph-level parallel-edge predicate."""

import random
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
                self.assertEqual(graph.parallel_edges(), [])
                a, b, c = self.vertices(graph)
                self.assertFalse(graph.isparallel())
                graph.add_edge(a, b, cost=1)
                graph.add_edge(b, c, cost=2)
                self.assertFalse(graph.isparallel())
                self.assertEqual(graph.parallel_edges(), [])
                self.assertEqual(graph.validate(), [])

    def test_repeated_edge_and_removal(self):
        for graph_class in (UGraph, DGraph):
            with self.subTest(graph_class=graph_class):
                graph = graph_class()
                a, b, _ = self.vertices(graph)
                first = graph.add_edge(a, b, cost=1)
                second = graph.add_edge(a, b, cost=2)
                self.assertTrue(graph.isparallel())
                self.assertEqual(graph.parallel_edges(), [(first, second)])
                self.assertTrue(graph.isparallel())  # read-only predicate
                graph.remove_edge(second)
                self.assertFalse(graph.isparallel())
                self.assertEqual(graph.parallel_edges(), [])
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
                forward = graph.add_edge(a, b)
                reverse = graph.add_edge(b, a)
                self.assertEqual(graph.isparallel(), expected)
                self.assertEqual(
                    graph.parallel_edges(),
                    [(forward, reverse)] if expected else [],
                )

    def test_parallel_groups_are_complete_and_ordered_by_first_edge(self):
        for graph_class in (UGraph, DGraph):
            with self.subTest(graph_class=graph_class):
                graph = graph_class()
                a, b, c = self.vertices(graph)
                first_ab = graph.add_edge(a, b, cost=1)
                first_bc = graph.add_edge(b, c, cost=2)
                second_bc = graph.add_edge(b, c, cost=3)
                second_ab = graph.add_edge(a, b, cost=4)
                third_ab = graph.add_edge(a, b, cost=5)

                # The b-c group completes first, but the a-b group starts first.
                self.assertEqual(
                    graph.parallel_edges(),
                    [(first_ab, second_ab, third_ab), (first_bc, second_bc)],
                )
                self.assertTrue(graph.isparallel())
                self.assertEqual(
                    graph.edges(),
                    [first_ab, first_bc, second_bc, second_ab, third_ab],
                )

                graph.remove_edge(second_ab)
                self.assertEqual(
                    graph.parallel_edges(),
                    [(first_ab, third_ab), (first_bc, second_bc)],
                )

                # Removing the first edge changes the order of surviving groups.
                graph.remove_edge(first_ab)
                self.assertEqual(
                    graph.parallel_edges(), [(first_bc, second_bc)]
                )
                replacement = graph.add_edge(a, b, cost=6)
                self.assertEqual(
                    graph.parallel_edges(),
                    [(first_bc, second_bc), (third_ab, replacement)],
                )
                self.assertEqual(graph.validate(), [])

    def test_reversed_undirected_edge_and_directed_edge(self):
        for graph_class in (UGraph, DGraph):
            with self.subTest(graph_class=graph_class):
                graph = graph_class()
                a, b, _ = self.vertices(graph)
                forward = graph.add_edge(a, b, cost=1)
                reverse = graph.add_edge(b, a, cost=2)
                if graph_class is UGraph:
                    self.assertEqual(graph.parallel_edges(), [(forward, reverse)])
                else:
                    self.assertEqual(graph.parallel_edges(), [])
                    repeated = graph.add_edge(a, b, cost=3)
                    self.assertEqual(graph.parallel_edges(), [(forward, repeated)])
                self.assertEqual(graph.isparallel(), bool(graph.parallel_edges()))

    def test_single_and_double_self_loop_groups(self):
        for graph_class in (UGraph, DGraph):
            with self.subTest(graph_class=graph_class):
                graph = graph_class()
                a, b, _ = self.vertices(graph)
                first = graph.add_edge(a, a, cost=1)
                graph.add_edge(b, b, cost=2)
                self.assertEqual(graph.parallel_edges(), [])
                second = graph.add_edge(a, a, cost=3)
                self.assertEqual(graph.parallel_edges(), [(first, second)])
                graph.remove_edge(first)
                self.assertEqual(graph.parallel_edges(), [])
                self.assertFalse(graph.isparallel())

    def test_random_graphs_match_independent_pairwise_reference(self):
        # Reproducible test data only; no security-sensitive randomness.
        rng = random.Random(20261030)  # nosec B311

        def same_endpoints(left, right, directed):
            same = left.v1 is right.v1 and left.v2 is right.v2
            reverse = left.v1 is right.v2 and left.v2 is right.v1
            return same or (not directed and reverse)

        for graph_class in (UGraph, DGraph):
            for trial in range(250):
                with self.subTest(graph_class=graph_class, trial=trial):
                    graph = graph_class()
                    vertices = [
                        graph.add_vertex([i, 0], name=f"v{i}")
                        for i in range(5)
                    ]
                    for _ in range(rng.randrange(31)):
                        a = rng.choice(vertices)
                        b = rng.choice(vertices)
                        graph.add_edge(a, b, cost=1)

                    edges = graph.edges()
                    used = set()
                    expected = []
                    for i, edge in enumerate(edges):
                        if i in used:
                            continue
                        indices = [
                            j for j in range(i, len(edges))
                            if same_endpoints(
                                edge, edges[j], graph_class is DGraph
                            )
                        ]
                        used.update(indices)
                        if len(indices) >= 2:
                            expected.append(tuple(edges[j] for j in indices))

                    self.assertEqual(graph.parallel_edges(), expected)
                    self.assertEqual(graph.isparallel(), bool(expected))


if __name__ == "__main__":
    unittest.main()
