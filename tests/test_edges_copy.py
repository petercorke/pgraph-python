"""
``vertex.edges()`` and ``graph.edges()`` return the caller a new container.

They used to hand back the object's own edge list/set, so ``v.edges().clear()``
emptied the vertex while the graph still counted the edges, leaving the graph
inconsistent.  A caller can now do what it likes with the result.
"""

import unittest

from pgraph import DGraph, UGraph


class _EdgesCopy:
    Graph: type

    def setUp(self):
        self.g = self.Graph()
        self.a = self.g.add_vertex([0, 0], name="a")
        self.b = self.g.add_vertex([1, 0], name="b")
        self.c = self.g.add_vertex([2, 0], name="c")
        self.ab = self.a.connect(self.b)
        self.bc = self.b.connect(self.c)
        self.loop = self.b.connect(self.b)

    # -- vertex.edges() -------------------------------------------------- #

    def test_vertex_edges_is_a_list(self):
        self.assertIsInstance(self.b.edges(), list)

    def test_vertex_edges_is_not_the_internal_list(self):
        self.assertIsNot(self.b.edges(), self.b._edgelist)
        self.assertEqual(self.b.edges(), self.b._edgelist)  # same content

    def test_each_call_is_a_new_list(self):
        self.assertIsNot(self.b.edges(), self.b.edges())

    def test_clearing_the_result_leaves_the_vertex_intact(self):
        before = list(self.b._edgelist)
        self.b.edges().clear()
        self.assertEqual(self.b._edgelist, before)
        self.assertEqual(self.b.degree, len(before))

    def test_appending_to_the_result_leaves_the_vertex_intact(self):
        n = len(self.b._edgelist)
        self.b.edges().append(self.ab)
        self.assertEqual(len(self.b._edgelist), n)

    def test_removing_while_iterating_the_result_is_safe(self):
        # iterating v.edges() while removing from the graph used to skip edges
        for e in self.b.edges():
            if e in self.g.edges():
                self.g.remove_edge(e)
        self.assertEqual(self.b.edges(), [])
        self.assertEqual(self.g.ne, self.expected_ne_after)

    def test_degree_unchanged(self):
        # degree is built on edges(); the loop counts twice only if undirected
        self.assertEqual(self.b.degree, self.expected_b_degree)

    # -- graph.edges() --------------------------------------------------- #

    def test_graph_edges_is_a_set_and_not_the_internal_set(self):
        edges = self.g.edges()
        self.assertIsInstance(edges, set)
        self.assertIsNot(edges, self.g._edgelist)
        self.assertEqual(edges, self.g._edgelist)

    def test_clearing_graph_edges_leaves_the_graph_intact(self):
        self.g.edges().clear()
        self.assertEqual(self.g.ne, 3)
        self.assertEqual(len(self.g.edges()), 3)

    def test_adding_to_graph_edges_leaves_the_graph_intact(self):
        extra = type(self.ab)(self.a, self.c)
        self.g.edges().add(extra)
        self.assertEqual(self.g.ne, 3)
        self.assertNotIn(extra, self.g.edges())


class TestUGraphEdgesCopy(_EdgesCopy, unittest.TestCase):
    Graph = UGraph
    expected_b_degree = 4  # ab, bc, and the loop's two ends
    expected_ne_after = 0  # all three edges are at b

    def test_edges_at_a_vertex_are_the_incident_ones(self):
        self.assertEqual(self.a.edges(), [self.ab])
        self.assertEqual(self.c.edges(), [self.bc])

    def test_a_self_loop_is_listed_twice(self):
        self.assertEqual(self.b.edges().count(self.loop), 2)


class TestDGraphEdgesCopy(_EdgesCopy, unittest.TestCase):
    Graph = DGraph
    expected_b_degree = 2  # bc and the loop; ab enters b
    expected_ne_after = 1  # a->b enters b, so it is not one of b's edges

    def test_edges_at_a_vertex_are_the_outgoing_ones(self):
        self.assertEqual(self.a.edges(), [self.ab])
        self.assertEqual(self.c.edges(), [])
        self.assertNotIn(self.ab, self.b.edges())

    def test_a_self_loop_is_listed_once(self):
        self.assertEqual(self.b.edges().count(self.loop), 1)


if __name__ == "__main__":
    unittest.main()
