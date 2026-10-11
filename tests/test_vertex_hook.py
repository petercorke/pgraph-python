"""
``BaseVertex.connect()`` is the one public method; what differs between an
undirected and a directed vertex is a single bookkeeping hook, ``_attach``,
that ``connect()`` calls once the connection has been validated and the edge
registered with the graph.

This pins the design: ``connect`` is defined once, the hook is what a vertex
kind supplies, and nothing is attached when validation fails.
"""

import unittest

from pgraph import DGraph, Edge, UGraph, DVertex, UVertex
from pgraph.PGraph import BaseVertex


class TestConnectIsDefinedOnce(unittest.TestCase):
    def test_subclasses_inherit_connect(self):
        self.assertIs(UVertex.connect, BaseVertex.connect)
        self.assertIs(DVertex.connect, BaseVertex.connect)

    def test_subclasses_supply_the_hook(self):
        self.assertIsNot(UVertex._attach, BaseVertex._attach)
        self.assertIsNot(DVertex._attach, BaseVertex._attach)
        self.assertIsNot(UVertex._attach, DVertex._attach)

    def test_base_hook_is_not_implemented(self):
        a, b = BaseVertex(name="a"), BaseVertex(name="b")
        with self.assertRaises(NotImplementedError):
            a._attach(Edge(a, b), b)


class TestBareBaseVertex(unittest.TestCase):
    """BaseVertex stays constructible (see policy.rst) but can never join a graph"""

    def test_instantiable(self):
        v = BaseVertex([1, 2], name="v")
        self.assertEqual(v.name, "v")

    def test_cannot_connect_or_be_added(self):
        a, b = BaseVertex(name="a"), BaseVertex(name="b")
        with self.assertRaises(ValueError):
            a.connect(b)
        with self.assertRaises(TypeError):
            UGraph().add_vertex(a)
        with self.assertRaises(TypeError):
            DGraph().add_vertex(a)


class _Hook:
    Graph: type
    Vertex: type

    def setUp(self):
        self.calls = []
        outer = self

        class Recording(self.Vertex):
            def _attach(self, edge, dest):
                # the edge is already registered with the graph at this point
                outer.calls.append(
                    (self, edge, dest, edge in self._graph._edgelist)
                )
                super()._attach(edge, dest)

        self.g = self.Graph(vertex_class=Recording)
        self.a = self.g.add_vertex([0, 0], name="a")
        self.b = self.g.add_vertex([1, 0], name="b")

    def test_attach_called_once_with_the_edge(self):
        e = self.a.connect(self.b, cost=3)
        self.assertEqual(len(self.calls), 1)
        v, edge, dest, registered = self.calls[0]
        self.assertIs(v, self.a)
        self.assertIs(edge, e)
        self.assertIs(dest, self.b)
        self.assertTrue(registered)

    def test_returns_the_edge(self):
        e = self.a.connect(self.b, cost=3)
        self.assertIsInstance(e, Edge)
        self.assertEqual(e.cost, 3)
        self.assertIn(e, self.g.edges())

    def test_nothing_attached_when_validation_fails(self):
        other = self.Graph().add_vertex([5, 5], name="x")
        with self.assertRaises(ValueError):
            self.a.connect(other)  # different graph
        with self.assertRaises(ValueError):
            self.a.connect(self.b, edge=Edge(self.b, self.b))  # wrong endpoints
        ungraphed = self.Vertex([9, 9])
        with self.assertRaises(ValueError):
            self.a.connect(ungraphed)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.g.ne, 0)
        self.assertEqual((self.a.edges(), self.b.edges()), ([], []))

    def test_add_edge_and_edge_connect_go_through_the_hook(self):
        self.g.add_edge(self.a, self.b)
        Edge(cost=1).connect(self.a, self.b)
        self.assertEqual(len(self.calls), 2)


class TestUndirectedHook(_Hook, unittest.TestCase):
    Graph, Vertex = UGraph, UVertex

    def test_edge_is_held_by_both_ends(self):
        e = self.a.connect(self.b)
        self.assertEqual(self.a.edges(), [e])
        self.assertEqual(self.b.edges(), [e])

    def test_self_loop_is_held_twice_by_its_vertex(self):
        e = self.a.connect(self.a)
        self.assertEqual(self.a.edges(), [e, e])
        self.assertEqual(self.a.degree, 2)


class TestDirectedHook(_Hook, unittest.TestCase):
    Graph, Vertex = DGraph, DVertex

    def test_edge_is_held_by_the_source_only(self):
        e = self.a.connect(self.b)
        self.assertEqual(self.a.edges(), [e])
        self.assertEqual(self.b.edges(), [])

    def test_self_loop_is_held_once(self):
        e = self.a.connect(self.a)
        self.assertEqual(self.a.edges(), [e])
        self.assertEqual(self.a.degree, 1)


class TestHookIsTheExtensionPoint(unittest.TestCase):
    def test_custom_bookkeeping(self):
        # a user kind that records the edge on the source only, though it is
        # built on the undirected vertex: only the hook has to change
        class OneSided(UVertex):
            def _attach(self, edge, dest):
                self._edgelist.append(edge)

        g = UGraph(vertex_class=OneSided)
        a = g.add_vertex([0, 0], name="a")
        b = g.add_vertex([1, 0], name="b")
        e = a.connect(b)
        self.assertEqual(a.edges(), [e])
        self.assertEqual(b.edges(), [])
        self.assertEqual(g.ne, 1)


if __name__ == "__main__":
    unittest.main()
