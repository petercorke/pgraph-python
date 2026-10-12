"""
``Vertex.connect()`` is the one public method.  A vertex has no direction of
its own, so what differs between an undirected and a directed graph is a single
bookkeeping hook of the *graph*, ``_attach``, that ``connect()`` calls once the
connection has been validated and the edge registered with the graph.

This pins the design: ``connect`` is defined once, on the vertex, the hook is
what a graph kind supplies, and nothing is attached when validation fails.
"""

import unittest

from pgraph import DGraph, Edge, Graph, UGraph, Vertex


class TestConnectIsDefinedOnce(unittest.TestCase):
    def test_there_is_one_vertex_class_and_it_has_no_hook(self):
        self.assertFalse(hasattr(Vertex, "_attach"))
        self.assertFalse(hasattr(Vertex, "_detach"))

    def test_graph_kinds_supply_the_hook(self):
        self.assertIsNot(UGraph._attach, Graph._attach)
        self.assertIsNot(DGraph._attach, Graph._attach)
        self.assertIsNot(UGraph._attach, DGraph._attach)

    def test_base_hook_is_not_implemented(self):
        g = UGraph()
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        with self.assertRaises(NotImplementedError):
            Graph._attach(g, Edge(a, b))


class TestStandaloneVertex(unittest.TestCase):
    """A vertex is constructible alone (see policy.rst) and joins either kind"""

    def test_instantiable(self):
        v = Vertex([1, 2], name="v")
        self.assertEqual(v.name, "v")

    def test_cannot_connect_until_added(self):
        a, b = Vertex(name="a"), Vertex(name="b")
        with self.assertRaises(ValueError):
            a.connect(b)

    def test_can_be_added_to_either_kind(self):
        for kind in (UGraph, DGraph):
            with self.subTest(kind=kind.__name__):
                g = kind()
                a, b = Vertex(name="a"), Vertex(name="b")
                g.add_vertex(a)
                g.add_vertex(b)
                self.assertIsInstance(a.connect(b), Edge)
                self.assertEqual(g.ne, 1)


class _Hook:
    Graph: type

    def setUp(self):
        self.calls = []
        outer = self

        class Recording(self.Graph):
            def _attach(self, edge):
                # the edge is already registered with the graph at this point
                outer.calls.append((edge, edge in self._edges))
                super()._attach(edge)

        self.g = Recording()
        self.a = self.g.add_vertex([0, 0], name="a")
        self.b = self.g.add_vertex([1, 0], name="b")

    def test_attach_called_once_with_the_edge(self):
        e = self.a.connect(self.b, cost=3)
        self.assertEqual(len(self.calls), 1)
        edge, registered = self.calls[0]
        self.assertIs(edge, e)
        self.assertIs(edge.v1, self.a)
        self.assertIs(edge.v2, self.b)
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
        ungraphed = Vertex([9, 9])
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
    Graph = UGraph

    def test_edge_is_held_by_both_ends(self):
        e = self.a.connect(self.b)
        self.assertEqual(self.a.edges(), [e])
        self.assertEqual(self.b.edges(), [e])

    def test_self_loop_is_held_twice_by_its_vertex(self):
        e = self.a.connect(self.a)
        self.assertEqual(self.a.edges(), [e, e])
        self.assertEqual(self.a.degree, 2)


class TestDirectedHook(_Hook, unittest.TestCase):
    Graph = DGraph

    def test_edge_is_held_by_the_source_only(self):
        e = self.a.connect(self.b)
        self.assertEqual(self.a.edges(), [e])
        self.assertEqual(self.b.edges(), [])

    def test_self_loop_is_held_once(self):
        e = self.a.connect(self.a)
        self.assertEqual(self.a.edges(), [e])
        self.assertEqual(self.a.out_degree, 1)  # held once as an outgoing edge
        self.assertEqual(self.a.degree, 2)  # and once as an incoming one


class TestHookIsTheExtensionPoint(unittest.TestCase):
    def test_custom_bookkeeping(self):
        # a user graph that records the edge on the start vertex only, though
        # it is built on the undirected graph: only the hook has to change
        class OneSided(UGraph):
            def _attach(self, edge):
                edge.v1._edgelist.append(edge)

        g = OneSided()
        a = g.add_vertex([0, 0], name="a")
        b = g.add_vertex([1, 0], name="b")
        e = a.connect(b)
        self.assertEqual(a.edges(), [e])
        self.assertEqual(b.edges(), [])
        self.assertEqual(g.ne, 1)


if __name__ == "__main__":
    unittest.main()
