"""Pickling and copying a graph.

A graph is a web of mutual references, which pickle and deepcopy follow
recursively: before the graph described itself flatly, ``copy()`` overflowed
the stack on a path of about 100 vertices, and a graph using a built-in metric
could not be pickled at all because the metric was a local function.
"""

import copy
import pickle
import unittest

import numpy as np

from pgraph import DGraph, DVertex, Edge, UGraph, UVertex


def manhattan(v):
    return float(np.abs(v).sum())


class TaggedVertex(UVertex):
    def __init__(self, tag, coord=None, name=None):  # a different signature
        super().__init__(coord=coord, name=name)
        self.tag = tag


class TaggedDVertex(DVertex):
    def __init__(self, tag, coord=None, name=None):
        super().__init__(coord=coord, name=name)
        self.tag = tag


class WeightedEdge(Edge):
    def __init__(self, v1=None, v2=None, cost=None, data=None):
        super().__init__(v1, v2, cost=cost, data=data)
        self.weight = 7


def chain(n, cls=UGraph):
    g = cls()
    vs = [g.add_vertex([i, 0]) for i in range(n)]
    for a, b in zip(vs, vs[1:]):
        g.add_edge(a, b)
    return g


def snapshot(g):
    """Everything observable about a graph, by name, including order"""
    return {
        "kind": type(g),
        "vertices": [(v.name, type(v), v.coord.tolist() if v.coord is not None else None) for v in g],
        "edges": [(e.v1.name, e.v2.name, e.cost, type(e)) for e in g.edges()],
        "adjacent": {v.name: [w.name for w in v.adjacent()] for v in g},
        "out": {v.name: [(e.v1.name, e.v2.name) for e in v.out_edges()] for v in g},
        "in": {v.name: [(e.v1.name, e.v2.name) for e in v.in_edges()] for v in g},
    }


class _RoundTrip:
    cls: type

    def make(self):
        g = self.cls(dim=2)
        a, b, c, d = (g.add_vertex([i, i * i], name=n) for i, n in enumerate("abcd"))
        g.add_edge(a, b)
        g.add_edge(b, c)
        g.add_edge(a, b, cost=9.0)  # parallel
        g.add_edge(c, c)  # loop
        g.add_edge(d, a)
        g.add_edge(c, a)
        g.remove_edge(g.edges()[1])  # leave a gap in the order
        g.add_edge(b, c, cost=2.5)
        a.data = {"note": "hello"}
        return g

    def check(self, g, h):
        self.assertIsNot(g, h)
        self.assertEqual(snapshot(g), snapshot(h))
        self.assertEqual(h.validate(), [])
        self.assertEqual((g.n, g.ne), (h.n, h.ne))
        for v in g:
            w = h[v.name]
            self.assertIsNot(v, w)
            self.assertIs(w._graph, h)
            self.assertEqual((v.degree, v.in_degree, v.out_degree), (w.degree, w.in_degree, w.out_degree))
        for e in h.edges():
            self.assertIs(e.v1._graph, h)
            self.assertIn(e, e.v1.out_edges())
        np.testing.assert_array_equal(g.adjacency(), h.adjacency())
        np.testing.assert_array_equal(g.incidence(), h.incidence())

    def test_pickle(self):
        g = self.make()
        self.check(g, pickle.loads(pickle.dumps(g)))

    def test_pickle_every_protocol(self):
        g = self.make()
        for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.check(g, pickle.loads(pickle.dumps(g, protocol)))

    def test_copy(self):
        g = self.make()
        h = g.copy()
        self.check(g, h)
        self.assertIsNot(h["a"].coord, g["a"].coord)
        self.assertEqual(h["a"].data, {"note": "hello"})
        self.assertIsNot(h["a"].data, g["a"].data)

    def test_copy_is_independent(self):
        g = self.make()
        h = g.copy()
        h.remove_vertex(h["a"])
        h.add_vertex([5, 5], name="z")
        self.assertEqual(g.n, 4)
        self.assertIn("a", [v.name for v in g])
        self.assertEqual(g.validate(), [])
        self.assertEqual(h.validate(), [])

    def test_restored_graph_is_usable(self):
        h = pickle.loads(pickle.dumps(self.make()))
        e = h.add_edge(h["d"], h["c"])
        self.assertEqual(h.ne, self.make().ne + 1)
        h.remove_edge(e)
        v = h.add_vertex([9, 9])
        self.assertEqual(v.name, "#4")
        self.assertEqual(h.validate(), [])
        path, length, _ = h.path_Astar(h["a"], h["c"])
        self.assertEqual(path[0], h["a"])

    def test_settings_survive(self):
        g = self.cls(metric="L1", heuristic="SE2", dim=3, verbose=False)
        h = pickle.loads(pickle.dumps(g))
        self.assertEqual(h.metric([3, -4, 0]), 7)
        with self.assertRaises(ValueError):
            h.add_vertex([1, 2])  # dim is still 3
        self.assertIs(h.heuristic, g.heuristic)

    def test_builtin_metrics_pickle_by_name(self):
        for name, value in (("L1", 7.0), ("L2", 5.0)):
            with self.subTest(metric=name):
                g = self.cls(metric=name, heuristic=name)
                h = pickle.loads(pickle.dumps(g))
                self.assertEqual(h.metric(np.r_[3.0, -4.0]), value)
                self.assertIs(h.metric, g.metric)
                self.assertIs(h.heuristic, g.heuristic)

    def test_default_metric_pickles(self):
        g = self.cls()
        h = pickle.loads(pickle.dumps(g))
        self.assertAlmostEqual(h.metric(np.r_[3.0, 4.0]), 5.0)

    def test_se2_metric_pickles(self):
        g = self.cls(metric="SE2")
        h = pickle.loads(pickle.dumps(g))
        self.assertIs(h.metric, g.metric)

    def test_user_module_level_metric(self):
        g = self.cls(metric=manhattan)
        h = pickle.loads(pickle.dumps(g))
        self.assertIs(h.metric, manhattan)

    def test_lambda_metric_cannot_be_pickled_but_can_be_copied(self):
        g = self.cls(metric=lambda v: 1.0)
        with self.assertRaises((pickle.PicklingError, AttributeError)):
            pickle.dumps(g)
        self.assertEqual(g.copy().metric(np.r_[1.0]), 1.0)

    def test_empty_graph(self):
        g = self.cls()
        h = pickle.loads(pickle.dumps(g))
        self.assertEqual((h.n, h.ne), (0, 0))

    def test_long_chain_pickles_and_copies(self):
        g = chain(5000, self.cls)
        for h in (pickle.loads(pickle.dumps(g)), g.copy(), copy.deepcopy(g)):
            self.assertEqual((h.n, h.ne), (5000, 4999))
            self.assertEqual(h[2500].adjacent()[0].name, g[2500].adjacent()[0].name)
        self.assertEqual(h.validate(), [])

    def test_vertex_alone_is_refused(self):
        g = self.make()
        for f in (pickle.dumps, copy.deepcopy, copy.copy):
            with self.subTest(f=f.__name__):
                with self.assertRaises(TypeError):
                    f(g["a"])

    def test_edge_alone_is_refused(self):
        g = self.make()
        for f in (pickle.dumps, copy.deepcopy, copy.copy):
            with self.subTest(f=f.__name__):
                with self.assertRaises(TypeError):
                    f(g.edges()[0])

    def test_unknown_version_is_refused(self):
        state = self.make().__getstate__()
        state["_flat"]["version"] = 99
        g = self.cls.__new__(self.cls)
        with self.assertRaises(ValueError):
            g.__setstate__(state)

    def test_data_referring_to_a_vertex_is_refused(self):
        g = self.make()
        g["a"].data = g["b"]
        with self.assertRaises(TypeError):
            pickle.dumps(g)


class TestUGraphPickle(_RoundTrip, unittest.TestCase):
    cls = UGraph

    def test_loop_listed_twice_at_vertex(self):
        h = pickle.loads(pickle.dumps(self.make()))
        loop = [e for e in h.edges() if e.v1 is e.v2][0]
        self.assertEqual(h["c"].edges().count(loop), 2)


class TestDGraphPickle(_RoundTrip, unittest.TestCase):
    cls = DGraph

    def test_in_edge_order_survives(self):
        g = self.make()
        h = pickle.loads(pickle.dumps(g))
        for v in g:
            self.assertEqual(
                [(e.v1.name, e.v2.name, e.cost) for e in v.in_edges()],
                [(e.v1.name, e.v2.name, e.cost) for e in h[v.name].in_edges()],
            )


class TestSubclasses(unittest.TestCase):
    def test_vertex_and_edge_classes(self):
        for gcls, vcls in ((UGraph, TaggedVertex), (DGraph, TaggedDVertex)):
            with self.subTest(graph=gcls.__name__):
                g = gcls(vertex_class=vcls, edge_class=WeightedEdge)
                a = g.add_vertex([0, 0], name="a")
                b = g.add_vertex([1, 1], name="b")
                a.tag, b.tag = "first", "second"
                e = g.add_edge(a, b)
                e.weight = 11
                for h in (pickle.loads(pickle.dumps(g)), g.copy()):
                    self.assertIsInstance(h["a"], vcls)
                    self.assertEqual((h["a"].tag, h["b"].tag), ("first", "second"))
                    self.assertIsInstance(h.edges()[0], WeightedEdge)
                    self.assertEqual(h.edges()[0].weight, 11)
                    self.assertIs(h.vertex_class, vcls)
                    self.assertIs(h.edge_class, WeightedEdge)
                    self.assertEqual(h.validate(), [])


if __name__ == "__main__":
    unittest.main()
