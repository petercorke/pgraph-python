"""
A vertex fills its attributes directly in ``__init__``, and ``Graph.add_vertex``
sets its name and graph directly, because the ``__setattr__`` of a vertex only
exists to tell the graph that a coordinate moved, and a vertex that is not in
a graph yet has no graph to tell.  These tests check that nothing else changed:
the attributes, the name table, and that moving a vertex that is in a graph is
still seen.
"""

import unittest

import numpy as np

from pgraph import DGraph, UGraph, Vertex


class TestConstructor(unittest.TestCase):
    def test_attributes_of_a_new_vertex(self):
        v = Vertex(coord=[1, 2], name="a")
        np.testing.assert_array_equal(v.coord, [1, 2])
        self.assertEqual(v.name, "a")
        self.assertIsNone(v.label)
        self.assertEqual(v._edgelist, [])
        self.assertEqual(v._inlist, [])
        self.assertIsNone(v._graph)
        self.assertTrue(v._connectivitychange)

    def test_no_coordinate(self):
        v = Vertex()
        self.assertIsNone(v.coord)
        self.assertIsNone(v.name)

    def test_the_coordinate_is_a_float_array_copy(self):
        c = np.array([1.0, 2.0])
        v = Vertex(coord=c)
        c[0] = 99.0
        self.assertEqual(v.coord[0], 1.0)  # not the caller's array
        self.assertIsInstance(v.coord, np.ndarray)

    def test_the_lists_are_not_shared_between_vertices(self):
        a, b = Vertex(), Vertex()
        self.assertIsNot(a._edgelist, b._edgelist)
        self.assertIsNot(a._inlist, b._inlist)
        self.assertIsNot(a._edgelist, a._inlist)

    def test_subclass_with_its_own_init(self):
        class Landmark(Vertex):
            def __init__(self, coord=None, name=None, color=None):
                super().__init__(coord, name)
                self.color = color  # an attribute of its own, set after

        v = Landmark([0.0, 0.0], name="l", color="red")
        self.assertEqual((v.name, v.color), ("l", "red"))
        g = UGraph(vertex_class=Landmark)
        g.add_vertex(v)
        self.assertEqual(v.color, "red")
        self.assertIs(g["l"], v)


class TestInAGraph(unittest.TestCase):
    def test_name_table_and_back_pointer(self):
        for cls in (UGraph, DGraph):
            with self.subTest(cls=cls.__name__):
                g = cls()
                v = g.add_vertex([0.0, 0.0], name="v")
                w = g.add_vertex([1.0, 0.0])  # default name
                self.assertIs(v._graph, g)
                self.assertIs(g["v"], v)
                self.assertEqual(w.name, "#1")
                self.assertIs(g["#1"], w)
                self.assertEqual(g.validate(), [])

    def test_name_from_the_vertex_is_kept(self):
        g = UGraph()
        v = Vertex(name="given")
        g.add_vertex(v)
        self.assertEqual(v.name, "given")
        self.assertIs(g["given"], v)

    def test_renaming_still_updates_the_table(self):
        g = UGraph()
        v = g.add_vertex(name="old")
        v.name = "new"
        self.assertIs(g["new"], v)
        self.assertNotIn("old", g)

    def test_duplicate_name_is_refused_and_the_vertex_stays_free(self):
        g = UGraph()
        g.add_vertex(name="a")
        v = Vertex(name="b")
        with self.assertRaises(ValueError):
            g.add_vertex(v, name="a")
        self.assertIsNone(v._graph)
        self.assertEqual(v.name, "b")

    def test_a_vertex_in_a_graph_cannot_be_added_to_another(self):
        g, h = UGraph(), UGraph()
        v = g.add_vertex()
        with self.assertRaises(ValueError):
            h.add_vertex(v)

    def test_assigning_a_coordinate_is_still_seen(self):
        g = UGraph(dim=2)
        v = g.add_vertex([0.0, 0.0])
        g._spatialchange = False
        v.coord = np.r_[1.0, 1.0]
        self.assertTrue(g._spatialchange)
        g._spatialchange = False
        v.coord += np.r_[1.0, 0.0]
        self.assertTrue(g._spatialchange)

    def test_assigning_other_attributes_is_not_a_change(self):
        g = UGraph(dim=2)
        v = g.add_vertex([0.0, 0.0])
        g._spatialchange = False
        v.data = "anything"
        self.assertFalse(g._spatialchange)


if __name__ == "__main__":
    unittest.main()
