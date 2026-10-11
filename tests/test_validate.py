"""
``graph.validate()`` checks that a graph's records agree with each other and
returns a list of problems, empty if it is consistent.

A graph keeps the same facts in several places: the set of edges, each
vertex's lists of outgoing (and, if directed, incoming) edges, each edge's two
vertices, each vertex's pointer back to its graph, and the name table.  The
library keeps them in step, so a valid graph is the normal case; ``validate``
is for tests, debugging, and code that reaches into the internals.
"""

import random
import unittest

from pgraph import DGraph, Edge, UGraph, UVertex


def poison(edge, name, value):
    """Break an edge's endpoint by hand, whether v1/v2 are plain attributes or
    read-only properties over _v1/_v2 -- the point is to corrupt the record."""
    try:
        setattr(edge, name, value)
    except AttributeError:
        setattr(edge, "_" + name, value)


def build(Graph):
    g = Graph()
    a, b, c, d = (g.add_vertex([i, 0], name=n) for i, n in enumerate("abcd"))
    a.connect(b)
    a.connect(b)  # parallel
    b.connect(c)
    c.connect(c)  # self-loop
    if Graph is DGraph:
        c.connect(a)
    return g, a, b, c, d


class TestValidGraphs(unittest.TestCase):
    def test_returns_a_list_of_strings(self):
        for Graph in (UGraph, DGraph):
            r = build(Graph)[0].validate()
            self.assertIsInstance(r, list)
            self.assertEqual(r, [])

    def test_empty_graph(self):
        self.assertEqual(UGraph().validate(), [])
        self.assertEqual(DGraph().validate(), [])

    def test_loops_and_parallel_edges_are_valid(self):
        for Graph in (UGraph, DGraph):
            g, *_ = build(Graph)
            self.assertEqual(g.validate(), [], Graph.__name__)

    def test_stays_valid_through_removals(self):
        for Graph in (UGraph, DGraph):
            g, a, b, c, d = build(Graph)
            g.remove_edge(a.edges()[0])
            self.assertEqual(g.validate(), [])
            g.remove_vertex(c)
            self.assertEqual(g.validate(), [])
            g.remove_vertex(a)
            self.assertEqual(g.validate(), [])

    def test_random_edit_sequences(self):
        rng = random.Random(3)
        for Graph in (UGraph, DGraph):
            g = Graph()
            vs = [g.add_vertex([i, 0]) for i in range(7)]
            for _ in range(80):
                if rng.random() < 0.6 or g.ne == 0:
                    rng.choice(vs).connect(rng.choice(vs))
                else:
                    g.remove_edge(rng.choice(sorted(g.edges(), key=id)))
                self.assertEqual(g.validate(), [])

    def test_user_vertex_and_edge_classes(self):
        class V(UVertex):
            pass

        class E(Edge):
            pass

        g = UGraph(vertex_class=V, edge_class=E)
        a, b = g.add_vertex([0, 0]), g.add_vertex([1, 0])
        g.add_edge(a, b)
        self.assertEqual(g.validate(), [])

    def test_does_not_change_the_graph(self):
        g, *_ = build(DGraph)
        before = (g.n, g.ne, [list(v._edgelist) for v in g], [list(v._inlist) for v in g])
        g.validate()
        g.validate()
        after = (g.n, g.ne, [list(v._edgelist) for v in g], [list(v._inlist) for v in g])
        self.assertEqual(before, after)


class TestFindsCorruption(unittest.TestCase):
    """each case breaks one record by hand and expects it to be named"""

    def problems(self, g):
        r = g.validate()
        self.assertTrue(r, "expected a problem to be reported")
        self.assertTrue(all(isinstance(p, str) for p in r))
        return " | ".join(r)

    def test_edge_missing_from_a_vertex_list(self):
        for Graph in (UGraph, DGraph):
            g, a, b, c, d = build(Graph)
            a._edgelist.pop()
            self.assertIn("missing", self.problems(g), Graph.__name__)

    def test_edge_held_by_a_vertex_but_not_in_the_graph(self):
        for Graph in (UGraph, DGraph):
            g, a, b, c, d = build(Graph)
            ghost = Edge(a, d)
            a._edgelist.append(ghost)
            self.assertIn("not in the graph", self.problems(g), Graph.__name__)

    def test_edge_in_graph_without_vertices(self):
        g, a, b, c, d = build(UGraph)
        e = a.edges()[0]
        poison(e, "v1", None)
        self.assertIn("no vertices", self.problems(g))

    def test_edge_whose_vertices_disagree_with_the_lists(self):
        g, a, b, c, d = build(UGraph)
        e = a.edges()[0]
        poison(e, "v1", d)  # now claims d, but a's list still holds it
        msg = self.problems(g)
        self.assertIn("'d'", msg)

    def test_directed_incoming_list_out_of_step(self):
        g, a, b, c, d = build(DGraph)
        b._inlist.pop()
        self.assertIn("missing", self.problems(g))

    def test_undirected_vertex_must_not_use_the_incoming_list(self):
        g, a, b, c, d = build(UGraph)
        b._inlist.append(a.edges()[0])
        self.assertIn("incoming", self.problems(g))

    def test_vertex_does_not_point_back(self):
        g, a, b, c, d = build(UGraph)
        a._graph = None
        self.assertIn("'a'", self.problems(g))

    def test_vertex_missing_from_the_name_table(self):
        g, a, b, c, d = build(UGraph)
        del g._vertexdict["d"]
        self.assertIn("'d'", self.problems(g))

    def test_vertex_of_the_wrong_class(self):
        class V(UVertex):
            pass

        g = UGraph(vertex_class=V)
        g.add_vertex([0, 0], name="a")
        g._vertexlist.append(UVertex([1, 0], name="plain"))
        g._vertexlist[-1]._graph = g
        g._vertexdict["plain"] = g._vertexlist[-1]
        self.assertIn("plain", self.problems(g))

    def test_wrong_coordinate_length(self):
        g = UGraph(dim=3)
        v = g.add_vertex([0, 0, 0], name="a")
        v.coord = v.coord[:2]
        self.assertIn("dim", self.problems(g))

    def test_edge_to_a_vertex_in_another_graph(self):
        g, a, b, c, d = build(UGraph)
        h = UGraph()
        x = h.add_vertex(name="x")
        poison(a.edges()[0], "v2", x)
        self.assertIn("'x'", self.problems(g))

    def test_junk_in_the_containers_is_reported_not_raised(self):
        # validate() is for finding corruption, so it must survive it
        g, a, b, c, d = build(UGraph)
        g._vertexlist.append("junk")
        g._edgelist.add(42)
        a._edgelist.append("junk")
        msg = self.problems(g)
        for word in ("junk", "42", "not a vertex", "not an edge"):
            self.assertIn(word, msg)

    def test_junk_in_a_directed_incoming_list(self):
        g, a, b, c, d = build(DGraph)
        b._inlist.append(3)
        self.assertIn("incoming", self.problems(g))

    def test_vertex_listed_twice(self):
        g, a, b, c, d = build(UGraph)
        g._vertexlist.append(a)
        self.assertIn("'a' is listed 2 times", self.problems(g))

    def test_name_table_entries_are_named(self):
        g, a, b, c, d = build(UGraph)
        g._vertexdict["zz"] = a  # a filed under a second, wrong name
        self.assertIn("'zz'", self.problems(g))
        del g._vertexdict["zz"]
        g._vertexdict["x"] = UGraph().add_vertex(name="x")  # not in this graph
        self.assertIn("'x'", self.problems(g))

    def test_edge_of_the_wrong_class(self):
        class MyEdge(Edge):
            pass

        g = UGraph(edge_class=MyEdge)
        a, b = g.add_vertex(name="a"), g.add_vertex(name="b")
        e = Edge(a, b)
        g._edgelist.add(e)
        a._edgelist.append(e)
        b._edgelist.append(e)
        self.assertIn("MyEdge", self.problems(g))

    def test_vertex_list_entry_for_an_edge_removed_from_the_graph(self):
        g, a, b, c, d = build(UGraph)
        g._edgelist.discard(a.edges()[0])
        self.assertIn("not in the graph", self.problems(g))

    def test_self_loop_held_once_instead_of_twice(self):
        g, a, b, c, d = build(UGraph)
        c._edgelist.pop()
        self.assertIn("missing", self.problems(g))

    def test_reports_every_problem_not_just_the_first(self):
        g, a, b, c, d = build(UGraph)
        a._edgelist.pop()
        del g._vertexdict["d"]
        self.assertGreaterEqual(len(g.validate()), 2)


if __name__ == "__main__":
    unittest.main()
