"""
``Graph.coordinates``, the optional spatial index of ``Graph.build_kdtree``
that ``closest`` and ``near`` use, and what ``validate`` says about
coordinates.

The index needs scipy, which pgraph does not depend on, so its tests are
skipped where it is absent; the test that scipy is only imported on demand
is not.  Expectations are computed here by brute force, not by the code
under test.
"""

import ast
import copy
import importlib
import importlib.util
import pathlib
import pickle  # nosec B403 nosemgrep - only used to round-trip graphs made in this file
import unittest
import unittest.mock

import numpy as np

import pgraph
from pgraph import DGraph, SE2Metric, UGraph

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None


def roundtrip(obj):
    """Pickle and unpickle: these tests only read what they have just written"""
    return pickle.loads(pickle.dumps(obj))  # nosec B301 nosemgrep


def points_graph(cls=UGraph, n=200, dim=2, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    g = cls(dim=dim, **kwargs)
    P = rng.uniform(-10, 10, (n, dim))
    if dim == 3:
        P[:, 2] = rng.uniform(-np.pi, np.pi, n)
    for i, p in enumerate(P):
        g.add_vertex(p, name=f"v{i}")
    return g, P


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def se2(D, weight):
    return np.sqrt(D[:, 0] ** 2 + D[:, 1] ** 2 + (weight * wrap(D[:, 2])) ** 2)


class TestCoordinates(unittest.TestCase):
    def test_one_column_per_vertex_in_order(self):
        g = UGraph()
        g.add_vertex([1, 2], name="a")
        g.add_vertex([3, 4], name="b")
        g.add_vertex([5, 6], name="c")
        C = g.coordinates()
        self.assertEqual(C.shape, (2, 3))
        np.testing.assert_array_equal(C, [[1, 3, 5], [2, 4, 6]])
        np.testing.assert_array_equal(C[:, 1], g["b"].coord)
        self.assertEqual(C.dtype, np.float64)

    def test_is_a_view_with_each_coordinate_contiguous(self):
        g, _ = points_graph(n=10)
        C = g.coordinates()
        self.assertTrue(C.T.flags.c_contiguous)  # a row per vertex underneath
        self.assertIsNotNone(C.base)
        self.assertTrue(C[:, 3].flags.c_contiguous)

    def test_vertex_without_coordinate_is_nan(self):
        g = UGraph()
        g.add_vertex([1.0, 2.0], name="a")
        g.add_vertex(name="nowhere")
        g.add_vertex([5.0, 6.0], name="c")
        C = g.coordinates()
        self.assertEqual(C.shape, (2, 3))
        self.assertTrue(np.isnan(C[:, 1]).all())
        np.testing.assert_array_equal(C[:, [0, 2]], [[1, 5], [2, 6]])

    def test_no_coordinates_at_all(self):
        g = UGraph()
        g.add_vertex()
        g.add_vertex()
        self.assertEqual(g.coordinates().shape, (0, 2))
        self.assertEqual(UGraph().coordinates().shape, (0, 0))

    def test_dim_of_the_graph_gives_the_rows(self):
        g = UGraph(dim=3)
        g.add_vertex()
        self.assertEqual(g.coordinates().shape, (3, 1))

    def test_different_lengths(self):
        g = UGraph()
        g.add_vertex([1.0, 2.0], name="a")
        g.add_vertex([1.0, 2.0, 3.0], name="b")
        with self.assertRaises(ValueError) as cm:
            g.coordinates()
        self.assertIn("'b'", str(cm.exception))

    def test_a_snapshot(self):
        g, _ = points_graph(n=5)
        C = g.coordinates()
        before = C.copy()
        g["v0"].coord = np.r_[99.0, 99.0]
        np.testing.assert_array_equal(C, before)  # not changed by moving a vertex
        C[:] = 0
        self.assertNotEqual(g["v1"].coord[0], 0.0)  # nor the vertices by it

    def test_columns_follow_removal(self):
        g = UGraph()
        for i in range(3):
            g.add_vertex([float(i), 0.0], name=f"v{i}")
        g.remove_vertex(g["v1"])
        np.testing.assert_array_equal(g.coordinates(), [[0, 2], [0, 0]])


@unittest.skipUnless(HAVE_SCIPY, "scipy is not installed")
class TestIndexAgreesWithTheLoop(unittest.TestCase):
    def queries(self, dim, seed=1, n=60):
        rng = np.random.default_rng(seed)
        Q = rng.uniform(-12, 12, (n, dim))
        if dim == 3:
            Q[:, 2] = rng.uniform(-4, 4, n)
        return Q

    def check(self, cls, metric, expected_distances, dim):
        g, P = points_graph(cls, dim=dim, metric=metric)
        g.build_kdtree()
        for q in self.queries(dim):
            d = expected_distances(P - q)
            vertex, dist = g.closest(q)
            self.assertAlmostEqual(dist, d.min(), places=9)
            self.assertAlmostEqual(d[int(vertex.name[1:])], d.min(), places=9)
            for radius in (0.0, 1.0, 4.0):
                found = g.near(q, radius)
                self.assertEqual(
                    sorted(int(v.name[1:]) for v, _ in found),
                    sorted(np.flatnonzero(d <= radius).tolist()),
                )
                np.testing.assert_allclose([x for _, x in found], np.sort(d[d <= radius]), atol=1e-9)

    def test_L2(self):
        for cls in (UGraph, DGraph):
            self.check(cls, "L2", lambda D: np.linalg.norm(D, axis=1), 2)

    def test_L1(self):
        self.check(UGraph, "L1", lambda D: np.abs(D).sum(axis=1), 2)

    def test_L2_in_three_dimensions(self):
        self.check(UGraph, "L2", lambda D: np.linalg.norm(D, axis=1), 3)

    def test_SE2_with_weights(self):
        for weight in (0.0, 0.3, 1.0, 4.0):
            with self.subTest(weight=weight):
                self.check(UGraph, SE2Metric(weight), lambda D, w=weight: se2(D, w), 3)

    def test_the_named_SE2(self):
        self.check(UGraph, "SE2", lambda D: se2(D, 1.0), 3)

    def test_a_query_across_the_angle_wrap(self):
        g = UGraph(dim=3, metric=SE2Metric(2.0))
        g.add_vertex([0.0, 0.0, np.pi - 0.05], name="a")
        g.add_vertex([0.0, 0.1, 0.0], name="b")
        g.build_kdtree()
        v, d = g.closest([0.0, 0.0, -np.pi + 0.05])  # 0.1 rad from a, weight 2
        self.assertEqual(v.name, "a")
        self.assertAlmostEqual(d, 0.2)

    def test_a_query_far_outside_is_still_right(self):
        g, P = points_graph(UGraph, dim=3, metric=SE2Metric(1.0))
        g.build_kdtree()
        q = np.r_[1e9, -1e9, 0.0]
        v, d = g.closest(q)
        self.assertAlmostEqual(d, se2(P - q, 1.0).min(), delta=1e-3)

    def test_vertices_without_coordinates_are_skipped(self):
        g = UGraph(dim=2)
        g.add_vertex(name="nowhere")
        g.add_vertex([5.0, 5.0], name="a")
        g.add_vertex(name="also nowhere")
        g.add_vertex([0.0, 0.0], name="b")
        g.build_kdtree()
        self.assertEqual(g.closest([4.0, 4.0])[0].name, "a")
        self.assertEqual([v.name for v, _ in g.near([0, 0], 1)], ["b"])


@unittest.skipUnless(HAVE_SCIPY, "scipy is not installed")
class TestIndexLifetime(unittest.TestCase):
    def graph(self):
        g = UGraph(dim=2)
        g.add_vertex([0.0, 0.0], name="a")
        g.add_vertex([10.0, 0.0], name="b")
        g.build_kdtree()
        return g

    def test_a_vertex_added_is_found(self):
        g = self.graph()
        self.assertEqual(g.closest([9.0, 0.0])[0].name, "b")
        g.add_vertex([9.0, 0.0], name="new")
        self.assertEqual(g.closest([9.0, 0.0])[0].name, "new")
        self.assertEqual({v.name for v, _ in g.near([9.0, 0.0], 1.5)}, {"b", "new"})

    def test_a_vertex_removed_is_not(self):
        g = self.graph()
        g.remove_vertex(g["b"])
        self.assertEqual(g.closest([9.0, 0.0])[0].name, "a")

    def test_assigning_a_coordinate(self):
        g = self.graph()
        g["a"].coord = np.r_[9.0, 0.0]
        self.assertEqual(g.closest([9.0, 0.0]), (g["a"], 0.0))

    def test_augmented_assignment_of_a_coordinate(self):
        g = self.graph()
        g["a"].coord += np.r_[9.0, 0.0]
        self.assertEqual(g.closest([9.0, 0.0]), (g["a"], 0.0))

    def test_adding_an_edge_keeps_it(self):
        g = self.graph()
        g.add_edge(g["a"], g["b"])
        self.assertIsNotNone(g._kdtree)  # white box: edges do not move vertices
        self.assertEqual(g.closest([9.0, 0.0])[0].name, "b")

    def test_a_coordinate_changed_in_place_is_not_seen(self):
        # the documented limitation: the index is kept and gives the old answer
        g = self.graph()
        g["a"].coord[0] = 9.0
        self.assertEqual(g.closest([9.0, 0.0])[0].name, "b")  # a is really nearer
        g.build_kdtree()  # as the documentation says to do
        self.assertEqual(g.closest([9.0, 0.0])[0].name, "a")

    def test_not_in_copies_or_pickles(self):
        g = self.graph()
        for h in (g.copy(), copy.deepcopy(g), roundtrip(g), g.subgraph(g)):
            self.assertIsNone(h._kdtree)
            self.assertEqual(h.closest([9.0, 0.0])[0].name, "b")

    def test_a_metric_of_your_own(self):
        g = UGraph(dim=2, metric=lambda v: float(np.abs(v).max()))
        g.add_vertex([0.0, 0.0])
        with self.assertRaises(ValueError):
            g.build_kdtree()
        self.assertEqual(g.closest([1.0, 0.5])[1], 1.0)  # still works, by the loop

    def test_se2_needs_three_elements(self):
        g = UGraph(dim=2, metric="SE2")
        g.add_vertex([0.0, 0.0])
        with self.assertRaises(ValueError):
            g.build_kdtree()

    def test_no_coordinates(self):
        g = UGraph()
        g.add_vertex()
        with self.assertRaises(ValueError):
            g.build_kdtree()


class TestVectorizedAndLoopPaths(unittest.TestCase):
    """near() without an index: built-in metrics in one operation, others in a loop"""

    def test_SE2_without_an_index(self):
        for weight in (0.0, 0.5, 2.0):
            with self.subTest(weight=weight):
                g, P = points_graph(UGraph, dim=3, metric=SE2Metric(weight))
                for q in ([0.0, 0.0, 3.0], [4.0, -2.0, -3.0]):
                    d = se2(P - np.asarray(q), weight)
                    found = g.near(q, 5.0)
                    self.assertEqual(
                        sorted(int(v.name[1:]) for v, _ in found),
                        sorted(np.flatnonzero(d <= 5.0).tolist()),
                    )
                    np.testing.assert_allclose([x for _, x in found], np.sort(d[d <= 5.0]), atol=1e-9)

    def test_metric_of_your_own(self):
        g = UGraph(dim=2, metric=lambda v: float(np.abs(v).max()))  # Chebyshev
        for name, c in [("a", (0, 0)), ("b", (2, 2)), ("c", (3, 0)), ("d", (0, 5))]:
            g.add_vertex(list(c), name=name)
        g.add_vertex(name="nowhere")
        found = g.near([0.0, 0.0], 3.0)
        self.assertEqual([(v.name, d) for v, d in found], [("a", 0.0), ("b", 2.0), ("c", 3.0)])

    def test_vertices_without_coordinates(self):
        g = UGraph()
        g.add_vertex(name="nowhere")
        g.add_vertex(name="nowhere else")
        self.assertEqual(g.within([[0, 0], [1, 1]]), [])
        self.assertEqual(g.near([0, 0], 5), [])

    def test_closest_skips_vertices_without_coordinates(self):
        g = UGraph(dim=2)
        g.add_vertex(name="nowhere")
        g.add_vertex([3.0, 4.0], name="a")
        vertex, d = g.closest([0.0, 0.0])  # no index: the loop
        self.assertEqual((vertex.name, d), ("a", 5.0))


@unittest.skipUnless(HAVE_SCIPY, "scipy is not installed")
class TestIndexGuards(unittest.TestCase):
    def test_a_query_that_is_not_finite_falls_back(self):
        g, _ = points_graph(UGraph, n=20)
        g.build_kdtree()
        vertex, d = g.closest([np.nan, 0.0])  # the loop: nothing is nearer than nan
        self.assertIsNone(vertex)
        self.assertEqual(d, np.inf)

    def test_a_query_of_the_wrong_length_with_SE2(self):
        g, _ = points_graph(UGraph, n=20, dim=3, metric="SE2")
        g.build_kdtree()
        with self.assertRaises(ValueError):  # the metric itself says so
            g.closest([0.0, 0.0])

    def test_coordinates_that_are_none_finite(self):
        g = UGraph(dim=2)
        g.add_vertex([np.nan, np.nan])
        with self.assertRaises(ValueError):
            g.build_kdtree()


class TestScipyIsOnlyImportedOnDemand(unittest.TestCase):
    def test_no_module_of_pgraph_imports_scipy_at_import_time(self):
        # spatialmath, which pgraph depends on, imports scipy itself, so
        # whether scipy is loaded cannot be tested; what pgraph does can
        package = pathlib.Path(pgraph.__file__).parent
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in tree.body:  # module level only
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    self.assertFalse(
                        name.split(".")[0] == "scipy", f"{path.name} imports {name}"
                    )

    def test_without_scipy_there_is_a_clear_error(self):
        g = UGraph(dim=2)
        g.add_vertex([0.0, 0.0])
        real = importlib.import_module

        def refuse(name, *args, **kwargs):
            if name.startswith("scipy"):
                raise ImportError("No module named 'scipy'")
            return real(name, *args, **kwargs)

        with unittest.mock.patch("importlib.import_module", refuse):
            with self.assertRaises(ImportError) as cm:
                g.build_kdtree()
        self.assertIn("scipy", str(cm.exception))
        self.assertIn("pip install scipy", str(cm.exception))
        self.assertEqual(g.closest([1.0, 1.0])[1], np.sqrt(2))  # the loop works


class TestValidateCoordinates(unittest.TestCase):
    def test_a_good_graph(self):
        g, _ = points_graph(n=5)
        self.assertEqual(g.validate(), [])
        self.assertEqual(g.validate(embedded=True), [])

    def test_a_graph_with_no_coordinates_is_valid(self):
        g = UGraph()
        g.add_vertex()
        self.assertEqual(g.validate(), [])

    def test_presence_only_when_asked_for(self):
        g = UGraph()
        g.add_vertex([1.0, 2.0], name="a")
        g.add_vertex(name="b")
        self.assertEqual(g.validate(), [])
        problems = g.validate(embedded=True)
        self.assertEqual(len(problems), 1)
        self.assertIn("'b'", problems[0])

    def test_lengths_must_agree_when_dim_was_not_given(self):
        g = UGraph()
        g.add_vertex([1.0, 2.0], name="a")
        g.add_vertex([1.0, 2.0, 3.0], name="b")
        problems = g.validate()
        self.assertEqual(len(problems), 1)
        self.assertIn("'b'", problems[0])

    def test_not_finite(self):
        g = UGraph()
        g.add_vertex([1.0, 2.0], name="a")
        g["a"].coord = np.r_[np.nan, 2.0]
        g.add_vertex([np.inf, 0.0], name="b")
        problems = g.validate()
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("not finite" in p for p in problems))

    def test_not_a_vector(self):
        g = UGraph()
        g.add_vertex([1.0, 2.0], name="a")
        g["a"].coord = np.zeros((2, 2))
        self.assertIn("not a vector", g.validate()[0])

    def test_not_numeric(self):
        g = UGraph()
        g.add_vertex([1.0, 2.0], name="a")
        g["a"].coord = np.array(["x", "y"])
        self.assertIn("not numeric", g.validate()[0])

    def test_length_of_dim_still_checked(self):
        g = UGraph(dim=2)
        g.add_vertex([1.0, 2.0], name="a")
        g["a"].coord = np.r_[1.0, 2.0, 3.0]
        self.assertEqual(len(g.validate()), 1)


if __name__ == "__main__":
    unittest.main()
