"""
Names say what is API and what is not, and the public ones cannot be used to
corrupt a graph.

- ``label`` is the component number, computed by the graph: read-only
- ``name`` is the key in the graph's name table: renaming a vertex that is in a
  graph updates the table, and refuses a name that is already taken
- ``Graph`` is the public name of the common base class (it appears in
  every signature and in users' type hints)
- the exporters use the public vertex API, and look formats up through a
  function rather than reaching into a private table
"""

import unittest

from pgraph import Graph, Vertex, DGraph, UGraph, exporters


class TestLabelIsReadOnly(unittest.TestCase):
    def setUp(self):
        self.g = UGraph()
        self.a, self.b, self.c = (self.g.add_vertex(name=n) for n in "abc")
        self.a.connect(self.b)  # c is a second component

    def test_cannot_assign(self):
        self.g.nc
        with self.assertRaises(AttributeError):
            self.a.label = 7

    def test_still_computed(self):
        self.assertEqual(self.g.nc, 2)
        self.assertEqual(self.a.label, self.b.label)
        self.assertNotEqual(self.a.label, self.c.label)
        self.assertEqual({v.label for v in self.g}, {0, 1})

    def test_none_until_the_components_are_found(self):
        self.assertIsNone(Vertex(name="loose").label)

    def test_directed_graph_too(self):
        g = DGraph()
        a, b, c = (g.add_vertex(name=n) for n in "abc")
        a.connect(b)
        self.assertEqual(g.nc, 2)
        with self.assertRaises(AttributeError):
            b.label = 0
        self.assertEqual(a.label, b.label)


class TestRename(unittest.TestCase):
    def setUp(self):
        self.g = UGraph()
        self.a, self.b = self.g.add_vertex(name="a"), self.g.add_vertex(name="b")
        self.a.connect(self.b, cost=1)

    def test_rename_updates_the_name_table(self):
        self.a.name = "alpha"
        self.assertIs(self.g["alpha"], self.a)
        self.assertNotIn("a", self.g)
        self.assertIn("alpha", self.g)
        self.assertEqual(self.g.validate(), [])

    def test_graph_methods_follow_the_new_name(self):
        self.a.name = "alpha"
        path, cost, _ = self.g.path_UCS("alpha", "b")
        self.assertEqual([v.name for v in path], ["alpha", "b"])

    def test_name_already_taken(self):
        with self.assertRaises(ValueError):
            self.a.name = "b"
        self.assertEqual(self.a.name, "a")  # unchanged
        self.assertIs(self.g["a"], self.a)
        self.assertEqual(self.g.validate(), [])

    def test_same_name_is_a_no_op(self):
        self.a.name = "a"
        self.assertIs(self.g["a"], self.a)
        self.assertEqual(self.g.validate(), [])

    def test_cannot_clear_the_name_of_a_vertex_in_a_graph(self):
        with self.assertRaises(ValueError):
            self.a.name = None
        self.assertEqual(self.a.name, "a")

    def test_before_adding_is_just_assignment(self):
        v = Vertex([0, 0])
        v.name = "free"
        v.name = None
        v.name = "again"
        self.assertEqual(v.name, "again")
        g = UGraph()
        g.add_vertex(v)
        self.assertIs(g["again"], v)

    def test_after_removal_is_just_assignment(self):
        self.g.remove_vertex(self.a)
        self.a.name = "b"  # no longer in the graph, so no clash
        self.assertEqual(self.a.name, "b")
        self.assertEqual(self.g.validate(), [])


class TestValidateAndRename(unittest.TestCase):
    """validate() is what notices a name table that is out of step"""

    def test_name_changed_behind_the_graph_s_back(self):
        # renaming through .name updates the table; writing the private field
        # does not, and validate() says so
        g = UGraph()
        a = g.add_vertex(name="a")
        g.add_vertex(name="b")
        a.name = "renamed"
        self.assertEqual(g.validate(), [])  # the public route stays valid
        a._name = "sneaky"
        msg = " | ".join(g.validate())
        self.assertTrue(msg)
        self.assertIn("sneaky", msg)
        self.assertIn("renamed", msg)



class TestPublicBaseNames(unittest.TestCase):
    def test_basegraph_is_exported(self):
        self.assertTrue(issubclass(UGraph, Graph))
        self.assertTrue(issubclass(DGraph, Graph))
        self.assertIsInstance(UGraph(), Graph)

    def test_vertex_is_exported(self):
        self.assertIs(UGraph().add_vertex([0, 0]).__class__, Vertex)
        self.assertIs(DGraph().add_vertex([0, 0]).__class__, Vertex)

    def test_the_unreleased_private_name_is_gone(self):
        import pgraph.PGraph as module

        self.assertFalse(hasattr(module, "_BaseGraph"))

    def test_the_class_reports_its_public_name(self):
        self.assertEqual(Graph.__name__, "Graph")

    def test_a_private_attribute_is_not_documented_as_api(self):
        self.assertNotIn("_edgelist", Vertex.__doc__ or "")


class TestExporters(unittest.TestCase):
    def test_get_exporter(self):
        for name in ("dot", "mermaid", "graphml", "elk"):
            self.assertTrue(callable(exporters.get_exporter(name)))

    def test_unknown_exporter(self):
        with self.assertRaises(ValueError) as cm:
            exporters.get_exporter("nonesuch")
        self.assertIn("dot", str(cm.exception))  # lists what is available

    def test_export_uses_it(self):
        g = UGraph()
        g.add_edge(g.add_vertex(name="a"), g.add_vertex(name="b"))
        self.assertEqual(g.export(format="dot"), exporters.get_exporter("dot")(g))

    def test_edge_order_follows_the_public_api(self):
        # the order is vertices in creation order, each vertex's out_edges()
        g = DGraph()
        a, b, c = (g.add_vertex(name=n) for n in "abc")
        e1 = b.connect(c)
        e2 = a.connect(b)
        e3 = a.connect(c)
        listed = [e for e, _, _ in exporters._ordered_edges(g)]
        self.assertEqual(listed, [e2, e3, e1])
        self.assertEqual(listed, [e for v in g for e in v.out_edges()])


if __name__ == "__main__":
    unittest.main()
