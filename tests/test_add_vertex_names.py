"""
A graph's vertices have unique names (the name is the key ``g[name]`` looks
them up by), and a vertex belongs to at most one graph.  ``add_vertex`` used to
overwrite the earlier vertex's table entry on a clash, and the default
``#N`` names could clash after a removal; it now refuses, and chooses a default
name that is free.
"""

import unittest

from pgraph import DGraph, UGraph, UVertex


class _AddVertex:
    Graph: type
    Vertex: type

    def setUp(self):
        self.g = self.Graph()

    def snapshot(self, g):
        return (g.n, sorted(g._vertexdict), [v.name for v in g])

    def test_duplicate_name_is_refused(self):
        a = self.g.add_vertex(name="a")
        before = self.snapshot(self.g)
        with self.assertRaises(ValueError) as cm:
            self.g.add_vertex(name="a")
        self.assertIn("'a'", str(cm.exception))
        self.assertEqual(self.snapshot(self.g), before)
        self.assertIs(self.g["a"], a)
        self.assertEqual(self.g.validate(), [])

    def test_duplicate_through_a_vertex_instance(self):
        self.g.add_vertex(name="a")
        clash = self.Vertex([1, 1], name="a")
        with self.assertRaises(ValueError):
            self.g.add_vertex(clash)
        self.assertIsNone(clash._graph)  # untouched: not half added
        self.assertEqual(self.g.n, 1)

    def test_name_argument_overriding_the_instance_name_is_checked_too(self):
        self.g.add_vertex(name="taken")
        v = self.Vertex([0, 0], name="free")
        with self.assertRaises(ValueError):
            self.g.add_vertex(v, name="taken")
        self.assertEqual(v.name, "free")  # a refused add does not rename it

    def test_default_names_are_unique_after_removals(self):
        vs = [self.g.add_vertex() for _ in range(4)]  # #0 .. #3
        self.g.remove_vertex(vs[0])
        self.g.remove_vertex(vs[2])
        more = [self.g.add_vertex() for _ in range(3)]
        names = [v.name for v in self.g]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.g.n, 5)
        self.assertEqual(self.g.validate(), [])
        for v in more:
            self.assertIs(self.g[v.name], v)

    def test_default_name_skips_one_given_explicitly(self):
        self.g.add_vertex(name="#1")
        a = self.g.add_vertex()
        b = self.g.add_vertex()
        self.assertEqual(len({"#1", a.name, b.name}), 3)
        self.assertEqual(self.g.validate(), [])

    def test_default_names_still_count_from_zero(self):
        self.assertEqual([self.g.add_vertex().name for _ in range(3)], ["#0", "#1", "#2"])

    def test_a_vertex_in_this_graph_cannot_be_added_again(self):
        a = self.g.add_vertex(name="a")
        with self.assertRaises(ValueError):
            self.g.add_vertex(a)
        self.assertEqual(self.g.n, 1)
        self.assertEqual(self.g.validate(), [])

    def test_a_vertex_in_another_graph_cannot_be_added(self):
        other = self.Graph()
        v = other.add_vertex(name="v")
        with self.assertRaises(ValueError):
            self.g.add_vertex(v)
        self.assertIs(v._graph, other)  # still where it was
        self.assertEqual(other.validate(), [])
        self.assertEqual(self.g.n, 0)

    def test_a_removed_vertex_can_be_added_again_or_elsewhere(self):
        a = self.g.add_vertex(name="a")
        self.g.remove_vertex(a)
        self.assertIs(self.g.add_vertex(a), a)
        self.g.remove_vertex(a)
        other = self.Graph()
        self.assertIs(other.add_vertex(a), a)
        self.assertEqual(other.validate(), [])

    def test_wrong_dimension_changes_nothing(self):
        g = self.Graph(dim=3)
        v = self.Vertex([1, 2], name="short")
        with self.assertRaises(ValueError):
            g.add_vertex(v)
        self.assertEqual(g.n, 0)
        self.assertIsNone(v._graph)


class TestUGraph(_AddVertex, unittest.TestCase):
    Graph, Vertex = UGraph, UVertex


class TestDGraph(_AddVertex, unittest.TestCase):
    from pgraph import DVertex as Vertex

    Graph = DGraph


if __name__ == "__main__":
    unittest.main()
