"""One vertex class, and a graph that is directed or not.

A vertex has no direction of its own: what differs between an undirected and a
directed graph is how an edge is recorded at its ends, which is the graph's
decision.  So there is one ``Vertex``; ``Graph(directed=...)`` chooses the kind;
``UGraph`` and ``DGraph`` remain the classes; and ``UVertex`` / ``DVertex`` are
deprecated names for ``Vertex``.
"""

import copy
import pickle  # nosec B403 nosemgrep - the tests pickle their own graphs
import unittest
import warnings

import numpy as np

import pgraph
from pgraph import DGraph, Edge, Graph, UGraph, Vertex


class TestGraphChoosesTheKind(unittest.TestCase):
    def test_default_is_undirected(self):
        g = Graph()
        self.assertIsInstance(g, UGraph)
        self.assertFalse(g.directed)

    def test_directed_true_is_a_dgraph(self):
        g = Graph(directed=True)
        self.assertIsInstance(g, DGraph)
        self.assertTrue(g.directed)

    def test_directed_false_is_a_ugraph(self):
        self.assertIsInstance(Graph(directed=False), UGraph)

    def test_the_kind_can_be_a_variable(self):
        for flag in (False, True):
            with self.subTest(directed=flag):
                self.assertEqual(Graph(directed=flag).directed, flag)

    def test_other_arguments_pass_through(self):
        g = Graph(directed=True, metric="L1", dim=3)
        self.assertEqual(g.metric(np.r_[1.0, -2.0, 3.0]), 6.0)
        with self.assertRaises(ValueError):
            g.add_vertex([1, 2])

    def test_graph_arguments_are_keyword_only(self):
        with self.assertRaises(TypeError):
            Graph(6)
        with self.assertRaises(TypeError):
            Graph(True)

    def test_the_classes_report_their_kind(self):
        self.assertFalse(UGraph().directed)
        self.assertTrue(DGraph().directed)

    def test_directed_may_repeat_the_class(self):
        self.assertTrue(DGraph(directed=True).directed)
        self.assertFalse(UGraph(directed=False).directed)

    def test_directed_may_not_contradict_the_class(self):
        with self.assertRaises(ValueError):
            UGraph(directed=True)
        with self.assertRaises(ValueError):
            DGraph(directed=False)

    def test_directed_is_read_only(self):
        g = Graph()
        with self.assertRaises(AttributeError):
            g.directed = True


class TestOneVertexClass(unittest.TestCase):
    def test_both_kinds_build_the_same_class(self):
        self.assertIs(type(UGraph().add_vertex([0, 0])), Vertex)
        self.assertIs(type(DGraph().add_vertex([0, 0])), Vertex)

    def test_a_vertex_joins_either_kind(self):
        for kind in (UGraph, DGraph):
            with self.subTest(kind=kind.__name__):
                g = kind()
                v = Vertex([1, 2], name="v")
                self.assertIs(g.add_vertex(v), v)
                self.assertIs(v._graph, g)

    def test_a_removed_vertex_may_join_the_other_kind(self):
        u, d = UGraph(), DGraph()
        v = u.add_vertex([0, 0], name="v")
        u.remove_vertex(v)
        self.assertIs(d.add_vertex(v), v)

    def test_user_subclass_serves_both_kinds(self):
        class Mine(Vertex):
            pass

        for kind in (UGraph, DGraph):
            with self.subTest(kind=kind.__name__):
                g = kind(vertex_class=Mine)
                self.assertIs(type(g.add_vertex([0, 0])), Mine)

    def test_vertex_class_must_be_a_vertex(self):
        for bad in (int, "Vertex", Edge, Vertex([0, 0])):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    Graph(vertex_class=bad)

    def test_the_kind_is_the_graphs_decision(self):
        # the same two vertices, joined in a graph of each kind
        for kind in (UGraph, DGraph):
            g = kind()
            a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
            e = a.connect(b)
            self.assertEqual(a.edges(), [e])
            self.assertEqual(b.edges(), [e] if not g.directed else [])
            self.assertEqual(b.in_edges(), [e])
            self.assertEqual(a.in_edges(), [] if g.directed else [e])


class TestDeprecatedNames(unittest.TestCase):
    def test_the_old_names_are_vertex(self):
        for name in ("UVertex", "DVertex"):
            with self.subTest(name=name):
                with self.assertWarns(DeprecationWarning) as cm:
                    cls = getattr(pgraph, name)
                self.assertIs(cls, Vertex)
                self.assertIn("Vertex", str(cm.warning))

    def test_importing_warns_and_works(self):
        with self.assertWarns(DeprecationWarning):
            from pgraph import UVertex

            self.assertIs(UVertex, Vertex)
        with self.assertWarns(DeprecationWarning):
            from pgraph import DVertex

            self.assertIs(DVertex, Vertex)

    def test_subclassing_an_old_name_still_works(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            class Mine(pgraph.UVertex):
                def __init__(self, coord=None, name=None):
                    super().__init__(coord, name=name)
                    self.extra = 1

        for kind in (UGraph, DGraph):
            g = kind(vertex_class=Mine)
            self.assertEqual(g.add_vertex([0, 0]).extra, 1)

    def test_other_missing_names_are_still_errors(self):
        with self.assertRaises(AttributeError):
            pgraph.NoSuchThing

    def test_the_old_names_are_not_advertised(self):
        self.assertNotIn("UVertex", pgraph.__all__)
        self.assertNotIn("DVertex", pgraph.__all__)

    def test_what_is_advertised_does_not_warn(self):
        # this is what ``from pgraph import *`` fetches
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for name in pgraph.__all__:
                getattr(pgraph, name)


class TestConstructorsAreKindAware(unittest.TestCase):
    def test_adjacency_on_graph(self):
        A = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        u = Graph.Adjacency(A)
        d = Graph.Adjacency(A, directed=True)
        self.assertIsInstance(u, UGraph)
        self.assertIsInstance(d, DGraph)
        self.assertEqual(d.ne, 3)
        self.assertEqual(u.ne, 2)  # only the upper triangle is used
        np.testing.assert_array_equal(d.adjacency(), A)

    def test_dict_on_graph(self):
        d = {"b": "a", "c": "a"}
        h = Graph.Dict(d, directed=True)
        self.assertIsInstance(h, DGraph)
        self.assertEqual(h.ne, 2)
        self.assertEqual([v.name for v in h["a"].successors()], ["b", "c"])
        self.assertIsInstance(Graph.Dict(d), UGraph)

    def test_copy_and_pickle_keep_the_kind(self):
        for flag in (False, True):
            with self.subTest(directed=flag):
                g = Graph(directed=flag)
                a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
                g.add_edge(a, b)
                for h in (g.copy(), copy.deepcopy(g), pickle.loads(pickle.dumps(g))):  # nosec nosemgrep
                    self.assertEqual(h.directed, flag)
                    self.assertEqual(type(h), type(g))
                    self.assertEqual(h.validate(), [])


class TestNotInstantiatedDirectly(unittest.TestCase):
    """Graph is concrete only through the kind it dispatches to"""

    def test_a_subclass_must_say_its_kind(self):
        class Vague(Graph):
            pass

        with self.assertRaises(TypeError):
            Vague()

    def test_a_subclass_of_a_kind_is_fine(self):
        class Mine(DGraph):
            pass

        self.assertTrue(Mine().directed)
        self.assertTrue(Mine(directed=True).directed)

    def test_the_base_hooks_say_which_kinds_provide_them(self):
        g = UGraph()
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        e = Edge(a, b)
        for call in (
            lambda: Graph._attach(g, e),
            lambda: Graph._detach(g, e),
            lambda: Graph._incoming(g, a),
            lambda: Graph._graphcolor(g),
        ):
            with self.assertRaises(NotImplementedError):
                call()

    def test_the_base_endpoint_test_is_directional(self):
        g = DGraph()
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        e = Edge(a, b)
        self.assertTrue(Graph._endpoints_match(g, e, a, b))
        self.assertFalse(Graph._endpoints_match(g, e, b, a))


class TestModuleLevelDeprecatedNames(unittest.TestCase):
    """The names are also reachable as attributes of pgraph.PGraph"""

    def test_alias_in_the_module(self):
        import pgraph.PGraph as module

        with self.assertWarns(DeprecationWarning):
            self.assertIs(module.UVertex, Vertex)
        with self.assertWarns(DeprecationWarning):
            self.assertIs(module.DVertex, Vertex)

    def test_other_names_in_the_module_are_errors(self):
        import pgraph.PGraph as module

        with self.assertRaises(AttributeError):
            module.NoSuchThing


class TestEdgeArrow(unittest.TestCase):
    def test_arrow_follows_the_graph(self):
        for flag, arrow in ((False, "--"), (True, "->")):
            g = Graph(directed=flag)
            a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
            self.assertIn(arrow, str(g.add_edge(a, b)))

    def test_a_standalone_edge_has_no_arrow_to_show(self):
        e = Edge(Vertex(name="a"), Vertex(name="b"), cost=1)
        self.assertIn("--", str(e))


if __name__ == "__main__":
    unittest.main()
