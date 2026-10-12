"""
``Graph.within`` (the vertices in a box), ``Graph.near`` (the vertices within
a distance of a point) and ``Graph.subgraph`` (the graph induced by a set of
vertices).

Expectations are written out by hand.
"""

import pickle  # nosec B403 - only used to round-trip graphs made in this file
import unittest

import numpy as np

from pgraph import DGraph, Edge, UGraph, Vertex


def line(cls=UGraph, n=5, **kwargs):
    """Vertices v0 .. v(n-1) at (0, 0), (1, 0), ... with no edges"""
    g = cls(dim=2, **kwargs)
    return g, [g.add_vertex([float(i), 0.0], name=f"v{i}") for i in range(n)]


def names(vertices):
    return [v.name for v in vertices]


class TestWithin(unittest.TestCase):
    def test_closed_box(self):
        g, _ = line()
        # the corners and edges of the box are inside
        self.assertEqual(names(g.within([[1, -1], [3, 1]])), ["v1", "v2", "v3"])
        self.assertEqual(names(g.within([[1, 0], [3, 0]])), ["v1", "v2", "v3"])

    def test_nothing_inside(self):
        g, _ = line()
        self.assertEqual(g.within([[1.2, -1], [1.8, 1]]), [])
        self.assertEqual(g.within([[0, 1], [4, 2]]), [])

    def test_a_box_of_no_size(self):
        g, _ = line()
        self.assertEqual(names(g.within([[2, 0], [2, 0]])), ["v2"])

    def test_graph_order_not_distance_order(self):
        g = UGraph(dim=2)
        for name, x in [("far", 9.0), ("near", 1.0), ("mid", 5.0)]:
            g.add_vertex([x, 0.0], name=name)
        self.assertEqual(names(g.within([[0, 0], [10, 0]])), ["far", "near", "mid"])

    def test_three_dimensions(self):
        g = UGraph(dim=3)
        for name, c in [("a", (0, 0, 0)), ("b", (1, 1, 1)), ("c", (1, 1, 5))]:
            g.add_vertex(list(c), name=name)
        self.assertEqual(names(g.within([[0, 0, 0], [2, 2, 2]])), ["a", "b"])
        self.assertEqual(names(g.within([[0.5, 0.5, 0.5], [2, 2, 6]])), ["b", "c"])

    def test_vertices_without_a_coordinate_are_skipped(self):
        g = UGraph()
        g.add_vertex(name="nowhere")
        g.add_vertex([1.0, 1.0], name="here")
        self.assertEqual(names(g.within([[0, 0], [2, 2]])), ["here"])

    def test_the_box_must_be_two_by_n(self):
        g, _ = line()
        for box in ([0, 0, 1, 1], [[0, 0], [1, 1], [2, 2]], [[0], [1], [2]]):
            with self.subTest(box=box):
                with self.assertRaises(ValueError):
                    g.within(box)

    def test_the_upper_corner_must_not_be_below_the_lower(self):
        g, _ = line()
        with self.assertRaises(ValueError):
            g.within([[3, 0], [1, 0]])

    def test_wrong_dimension(self):
        g, _ = line()  # 2-D coordinates
        with self.assertRaises(ValueError):
            g.within([[0, 0, 0], [1, 1, 1]])

    def test_does_not_change_the_graph(self):
        g, vs = line()
        g.within([[0, 0], [2, 0]])
        self.assertEqual(g.n, 5)
        self.assertEqual(names(g), ["v0", "v1", "v2", "v3", "v4"])


class TestNear(unittest.TestCase):
    def test_sorted_by_distance(self):
        g, _ = line()
        found = g.near([2.2, 0], 1.5)
        self.assertEqual(names(v for v, _ in found), ["v2", "v3", "v1"])
        np.testing.assert_allclose([d for _, d in found], [0.2, 0.8, 1.2])

    def test_the_radius_is_inclusive(self):
        g, _ = line()
        self.assertEqual(names(v for v, _ in g.near([0, 0], 2)), ["v0", "v1", "v2"])

    def test_nothing_near(self):
        g, _ = line()
        self.assertEqual(g.near([10, 10], 1), [])

    def test_equal_distances_in_graph_order(self):
        g = UGraph(dim=2)
        for name, c in [("east", (1, 0)), ("west", (-1, 0)), ("north", (0, 1))]:
            g.add_vertex(list(c), name=name)
        found = g.near([0, 0], 1)
        self.assertEqual(names(v for v, _ in found), ["east", "west", "north"])

    def test_the_closest_vertex_is_the_first(self):
        g, _ = line()
        near = g.near([2.2, 0], 10)
        closest, distance = g.closest([2.2, 0])
        self.assertIs(near[0][0], closest)
        self.assertAlmostEqual(near[0][1], distance)

    def test_uses_the_metric_of_the_graph(self):
        for metric, expected in (("L2", ["a", "b"]), ("L1", ["a"])):
            with self.subTest(metric=metric):
                g = UGraph(dim=2, metric=metric)
                g.add_vertex([0.0, 0.0], name="a")
                g.add_vertex([1.0, 1.0], name="b")  # 1.41 by L2, 2 by L1
                self.assertEqual(names(v for v, _ in g.near([0, 0], 1.5)), expected)

    def test_vertices_without_a_coordinate_are_skipped(self):
        g = UGraph()
        g.add_vertex(name="nowhere")
        g.add_vertex([0.0, 0.0], name="here")
        self.assertEqual(names(v for v, _ in g.near([0, 0], 1)), ["here"])

    def test_negative_radius(self):
        g, _ = line()
        with self.assertRaises(ValueError):
            g.near([0, 0], -1)


class MyVertex(Vertex):
    pass


class MyEdge(Edge):
    pass


class _Subgraph:
    cls: type

    def graph(self):
        """v0-v1, v1=v2 (two edges, costs 9 and 4), a loop on v2, v2-v3, v3-v4"""
        g, v = line(self.cls)
        g.add_edge(v[0], v[1])
        g.add_edge(v[1], v[2], cost=9)
        g.add_edge(v[1], v[2], cost=4)
        g.add_edge(v[2], v[2])
        g.add_edge(v[2], v[3])
        g.add_edge(v[3], v[4])
        return g, v

    @staticmethod
    def description(g):
        return [(e.v1.name, e.v2.name, e.cost) for e in g.edges()]

    def test_the_edges_between_the_vertices_are_kept(self):
        g, v = self.graph()
        h = g.subgraph([v[1], v[2], v[3]])
        self.assertIsInstance(h, self.cls)
        self.assertEqual(h.n, 3)
        # the edges v0-v1 and v3-v4 cross the border: removed; the parallel
        # edges and the loop stay, in the order they were added
        self.assertEqual(
            self.description(h),
            [("v1", "v2", 9), ("v1", "v2", 4), ("v2", "v2", 0.0), ("v2", "v3", 1.0)],
        )

    def test_the_original_is_unchanged(self):
        g, v = self.graph()
        before = self.description(g)
        g.subgraph([v[1], v[2]])
        self.assertEqual(g.n, 5)
        self.assertEqual(self.description(g), before)
        self.assertEqual(g.validate(), [])

    def test_the_result_is_a_valid_graph(self):
        g, v = self.graph()
        self.assertEqual(g.subgraph([v[1], v[2], v[3]]).validate(), [])

    def test_the_new_graph_is_independent(self):
        g, v = self.graph()
        h = g.subgraph([v[1], v[2]])
        self.assertIsNot(h["v1"], g["v1"])
        h["v1"].coord[0] = 99.0
        self.assertEqual(g["v1"].coord[0], 1.0)
        h.edges()[0].cost = 123
        self.assertEqual(g.edges()[1].cost, 9)
        h.add_vertex([5.0, 5.0], name="new")
        self.assertEqual(g.n, 5)
        self.assertNotIn("new", g)

    def test_names_are_kept_and_order_is_the_graphs(self):
        g, v = self.graph()
        h = g.subgraph([v[3], v[1], v[2]])  # asked for in another order
        self.assertEqual(names(h), ["v1", "v2", "v3"])
        np.testing.assert_array_equal(h["v2"].coord, g["v2"].coord)

    def test_vertices_by_name(self):
        g, v = self.graph()
        h = g.subgraph(["v2", v[3]])
        self.assertEqual(names(h), ["v2", "v3"])
        self.assertEqual(h.ne, 2)  # the loop and v2-v3

    def test_a_cost_that_was_given_is_kept(self):
        g, v = self.graph()
        h = g.subgraph([v[1], v[2]])
        # 9 is not the distance between the vertices, which is 1
        self.assertEqual([e.cost for e in h.edges()][:2], [9, 4])

    def test_nothing(self):
        g, _ = self.graph()
        h = g.subgraph([])
        self.assertIsInstance(h, self.cls)
        self.assertEqual((h.n, h.ne, h.nc), (0, 0, 0))

    def test_everything_is_a_copy(self):
        g, v = self.graph()
        h = g.subgraph(g)
        self.assertEqual(names(h), names(g))
        self.assertEqual(self.description(h), self.description(g))

    def test_components_are_those_of_the_new_graph(self):
        g, v = self.graph()
        self.assertEqual(g.nc, 1)  # make the original's cached components stale
        h = g.subgraph([v[0], v[1], v[3], v[4]])  # v0-v1 and v3-v4
        self.assertEqual(h.nc, 2)
        self.assertEqual(h["v0"].label, h["v1"].label)
        self.assertEqual(h["v3"].label, h["v4"].label)
        self.assertNotEqual(h["v1"].label, h["v3"].label)
        self.assertEqual(g.nc, 1)

    def test_a_vertex_from_another_graph(self):
        g, _ = self.graph()
        other, w = line(self.cls)
        with self.assertRaises(ValueError):
            g.subgraph([w[0]])

    def test_a_vertex_that_was_removed(self):
        g, v = self.graph()
        g.remove_vertex(v[4])
        with self.assertRaises(ValueError):
            g.subgraph([v[4]])

    def test_unknown_name_and_wrong_type(self):
        g, _ = self.graph()
        with self.assertRaises(KeyError):
            g.subgraph(["nobody"])
        with self.assertRaises(TypeError):
            g.subgraph([3])

    def test_classes_metric_and_attributes_are_kept(self):
        g = self.cls(dim=2, vertex_class=MyVertex, edge_class=MyEdge, metric="L1")
        a = g.add_vertex([0.0, 0.0], name="a")
        b = g.add_vertex([1.0, 1.0], name="b")
        g.add_vertex([5.0, 5.0], name="c")
        e = g.add_edge(a, b)
        e.data = {"road": ["x"]}
        a.note = "start"

        h = g.subgraph([a, b])
        self.assertIsInstance(h["a"], MyVertex)
        self.assertIsInstance(h.edges()[0], MyEdge)
        self.assertEqual(h["a"].note, "start")
        self.assertEqual(h.edges()[0].cost, 2.0)  # L1
        self.assertEqual(h.metric(np.array([1.0, 1.0])), 2.0)
        self.assertIs(h.vertex_class, MyVertex)
        self.assertIs(h.edge_class, MyEdge)
        # the data are copies, not shared with the original
        self.assertEqual(h.edges()[0].data, {"road": ["x"]})
        h.edges()[0].data["road"].append("y")
        self.assertEqual(e.data, {"road": ["x"]})

    def test_with_within(self):
        g, v = self.graph()
        h = g.subgraph(g.within([[1, -1], [3, 1]]))
        self.assertEqual(names(h), ["v1", "v2", "v3"])
        self.assertEqual(h.ne, 4)

    def test_can_be_pickled_and_modified(self):
        g, v = self.graph()
        h = g.subgraph([v[1], v[2], v[3]])
        h2 = pickle.loads(pickle.dumps(h))  # nosec B301
        self.assertEqual(self.description(h2), self.description(h))
        h.add_edge(h["v1"], h["v3"])
        self.assertEqual(h.ne, 5)
        self.assertEqual(h.validate(), [])


class TestUndirectedSubgraph(_Subgraph, unittest.TestCase):
    cls = UGraph

    def test_an_edge_is_found_from_either_end(self):
        g, v = self.graph()
        h = g.subgraph([v[2], v[3]])
        self.assertEqual(names(h["v2"].adjacent()), ["v2", "v2", "v3"])
        self.assertEqual(names(h["v3"].adjacent()), ["v2"])


class TestDirectedSubgraph(_Subgraph, unittest.TestCase):
    cls = DGraph

    def test_direction_is_kept(self):
        g, v = self.graph()
        h = g.subgraph([v[1], v[2], v[3]])
        self.assertEqual(names(h["v1"].successors()), ["v2", "v2"])
        self.assertEqual(h["v1"].predecessors(), [])  # v0 -> v1 crossed the border
        self.assertEqual(names(h["v3"].predecessors()), ["v2"])
        self.assertEqual(h["v3"].successors(), [])  # v3 -> v4 crossed it too


if __name__ == "__main__":
    unittest.main()
