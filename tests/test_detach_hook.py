"""
``_attach`` records an edge in the vertices' edge lists when it is connected;
``_detach`` is its twin, undoing exactly that when the edge is removed, so each
vertex kind states how an edge enters and leaves its lists once, and
``remove_edge`` / ``remove_vertex`` need no knowledge of which kind they hold.
"""

import unittest

from pgraph import DGraph, DVertex, Edge, UGraph, UVertex
from pgraph.PGraph import BaseVertex


class TestHookIsDefinedPerKind(unittest.TestCase):
    def test_kinds_supply_it(self):
        self.assertIsNot(UVertex._detach, BaseVertex._detach)
        self.assertIsNot(DVertex._detach, BaseVertex._detach)
        self.assertIsNot(UVertex._detach, DVertex._detach)

    def test_base_hook_is_not_implemented(self):
        a, b = BaseVertex(name="a"), BaseVertex(name="b")
        with self.assertRaises(NotImplementedError):
            a._detach(Edge(a, b), b)


class _Detach:
    Graph: type
    Vertex: type

    def setUp(self):
        self.calls = []
        outer = self

        class Recording(self.Vertex):
            def _detach(self, edge, dest):
                # at this point the edge still knows its endpoints
                outer.calls.append((self, edge, dest, edge.v1, edge.v2))
                super()._detach(edge, dest)

        self.g = self.Graph(vertex_class=Recording)
        self.a = self.g.add_vertex([0, 0], name="a")
        self.b = self.g.add_vertex([1, 0], name="b")
        self.c = self.g.add_vertex([2, 0], name="c")

    def consistent(self):
        """every edge held by a vertex is in the graph, and vice versa"""
        held = {id(e) for v in self.g for e in v.edges()}
        return held <= {id(e) for e in self.g.edges()} and all(
            e.v1 is not None and e.v2 is not None for e in self.g.edges()
        )

    def test_remove_edge_calls_the_hook_once(self):
        e = self.a.connect(self.b)
        self.g.remove_edge(e)
        self.assertEqual(len(self.calls), 1)
        v, edge, dest, v1, v2 = self.calls[0]
        self.assertIs(v, self.a)
        self.assertIs(edge, e)
        self.assertIs(dest, self.b)
        self.assertEqual((v1, v2), (self.a, self.b))
        self.assertEqual(self.g.ne, 0)
        self.assertIsNone(e.v1)

    def test_refused_removal_detaches_nothing(self):
        other = self.Graph()
        x, y = other.add_vertex(name="x"), other.add_vertex(name="y")
        e = x.connect(y)
        with self.assertRaises(ValueError):
            self.g.remove_edge(e)
        self.assertEqual(self.calls, [])

    def test_removing_one_of_parallel_edges(self):
        e1 = self.a.connect(self.b, cost=1)
        e2 = self.a.connect(self.b, cost=2)
        self.g.remove_edge(e1)
        self.assertEqual(self.a.edges(), [e2])
        self.assertTrue(self.consistent())

    def test_remove_vertex_removes_its_edges_through_the_hook(self):
        self.a.connect(self.b)
        self.b.connect(self.c)
        self.c.connect(self.a)
        self.g.remove_vertex(self.b)
        self.assertEqual(self.g.n, 2)
        self.assertEqual(self.g.ne, 1)  # c-a survives
        self.assertTrue(self.consistent())
        self.assertEqual(len(self.calls), 2)  # a-b and b-c

    def test_removed_vertex_no_longer_points_at_the_graph(self):
        self.a.connect(self.b)
        self.g.remove_vertex(self.a)
        self.assertIsNone(self.a._graph)
        self.assertEqual(self.a.edges(), [])
        with self.assertRaises(ValueError):
            self.a.remove()  # not in a graph any more, a clear error
        # and it can join another graph
        self.Graph(vertex_class=type(self.a)).add_vertex(self.a)
        self.assertEqual(self.g.n, 2)


class TestUndirectedDetach(_Detach, unittest.TestCase):
    Graph, Vertex = UGraph, UVertex

    def test_both_ends_are_cleared(self):
        e = self.a.connect(self.b)
        self.g.remove_edge(e)
        self.assertEqual((self.a.edges(), self.b.edges()), ([], []))

    def test_self_loop_is_cleared_from_both_entries(self):
        e = self.a.connect(self.a)
        self.assertEqual(self.a.edges(), [e, e])
        self.g.remove_edge(e)
        self.assertEqual(self.a.edges(), [])


class TestDirectedDetach(_Detach, unittest.TestCase):
    Graph, Vertex = DGraph, DVertex

    def test_only_the_source_held_it(self):
        e = self.a.connect(self.b)
        self.g.remove_edge(e)
        self.assertEqual((self.a.edges(), self.b.edges()), ([], []))

    def test_removing_a_vertex_removes_edges_that_point_at_it(self):
        self.a.connect(self.b)  # a -> b, held by a only
        self.g.remove_vertex(self.b)
        self.assertEqual(self.a.edges(), [])
        self.assertEqual(self.g.ne, 0)
        self.assertTrue(self.consistent())


if __name__ == "__main__":
    unittest.main()
