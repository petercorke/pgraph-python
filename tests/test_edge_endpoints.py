"""
``Edge.v1`` and ``Edge.v2`` are read-only.

They used to be plain attributes, so a caller could write ``e.v1 = None`` on an
edge that was in a graph and leave it wedged: ``remove_edge`` and
``remove_vertex`` then raised and the edge could not be removed.  Only the
library binds an edge to its vertices -- the constructor, ``connect()`` and
``remove_edge()`` -- so an edge in a graph always has both of its ends.
"""

import copy
import unittest

from pgraph import DGraph, Edge, UGraph


class TestReadOnly(unittest.TestCase):
    def setUp(self):
        self.g = UGraph()
        self.a = self.g.add_vertex([0, 0], name="a")
        self.b = self.g.add_vertex([1, 0], name="b")
        self.c = self.g.add_vertex([2, 0], name="c")
        self.e = self.a.connect(self.b)

    def test_cannot_assign(self):
        for name, value in (("v1", None), ("v1", self.c), ("v2", None), ("v2", self.c)):
            with self.subTest(name=name, value=value):
                with self.assertRaises(AttributeError) as cm:
                    setattr(self.e, name, value)
                # the message says what to do instead
                self.assertIn("connect", str(cm.exception))

    def test_cannot_delete(self):
        with self.assertRaises(AttributeError):
            del self.e.v1

    def test_failed_assignment_changes_nothing(self):
        with self.assertRaises(AttributeError):
            self.e.v1 = self.c
        self.assertIs(self.e.v1, self.a)
        self.assertIs(self.e.v2, self.b)
        self.assertEqual(self.a.edges(), [self.e])
        self.assertEqual(self.c.edges(), [])

    def test_the_graph_can_always_remove_its_edges(self):
        for name in ("v1", "v2"):
            try:
                setattr(self.e, name, None)
            except AttributeError:
                pass
        self.g.remove_edge(self.e)  # not wedged
        self.assertEqual(self.g.ne, 0)
        self.g.remove_vertex(self.a)
        self.assertEqual(self.g.n, 2)


class TestLibraryStillBindsThem(unittest.TestCase):
    def test_constructor(self):
        g = DGraph()
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        e = Edge(a, b, cost=3)
        self.assertIs(e.v1, a)
        self.assertIs(e.v2, b)

    def test_standalone_edge_has_no_vertices(self):
        e = Edge(cost=1)
        self.assertIsNone(e.v1)
        self.assertIsNone(e.v2)

    def test_connect_binds_a_bare_edge(self):
        g = UGraph()
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        e = Edge(cost=1)
        a.connect(b, edge=e)
        self.assertEqual((e.v1, e.v2), (a, b))

    def test_edge_connect_binds(self):
        g = UGraph()
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        e = Edge(cost=1)
        e.connect(a, b)
        self.assertEqual((e.v1, e.v2), (a, b))

    def test_removal_unlinks_and_the_edge_can_be_reused(self):
        g = UGraph()
        a, b, c = (g.add_vertex(name=n) for n in "abc")
        e = a.connect(b)
        g.remove_edge(e)
        self.assertIsNone(e.v1)
        self.assertIsNone(e.v2)
        with self.assertRaises(ValueError):
            g.add_edge(e)  # unlinked, so add_edge(edge) refuses it
        b.connect(c, edge=e)  # but connect() re-binds it
        self.assertEqual((e.v1, e.v2), (b, c))
        self.assertEqual(g.ne, 1)

    def test_subclass_that_calls_super(self):
        class Observation(Edge):
            def __init__(self, v1, v2, uv):
                super().__init__(v1, v2, cost=0)
                self.uv = uv

        g = UGraph()
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        o = Observation(a, b, (1, 2))
        a.connect(b, edge=o)
        self.assertEqual((o.v1, o.v2, o.uv), (a, b, (1, 2)))

    def test_repr_shows_the_endpoints_and_an_edge_alone_is_not_copied(self):
        g = UGraph()
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        e = a.connect(b, cost=2)
        self.assertIn("a", str(e))
        # an edge has no meaning apart from its graph: copy the graph instead
        for f in (copy.copy, copy.deepcopy):
            with self.assertRaises(TypeError):
                f(e)
        dup = g.copy().edges()[0]
        self.assertEqual(dup.cost, 2)
        self.assertEqual((dup.v1.name, dup.v2.name), ("a", "b"))


if __name__ == "__main__":
    unittest.main()
