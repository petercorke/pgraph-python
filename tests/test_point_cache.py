"""
The matrix of coordinates that ``closest`` and ``near`` use when there is no
spatial index, and that ``add_vertex`` keeps up to date.

Expectations are computed here, by hand, from the coordinates that were put
in; never by the code under test.  The cache is private, so what is checked
is that answers are the same as the pass over the vertices would give,
through every way the graph can change.
"""

import copy
import pickle  # nosec B403 nosemgrep - only used to round-trip graphs made in this file
import unittest

import numpy as np

from pgraph import SE2Metric, UGraph, Vertex


def roundtrip(obj):
    """Pickle and unpickle: these tests only read what they have just written"""
    return pickle.loads(pickle.dumps(obj))  # nosec B301 nosemgrep


def oracle(points, q, p=2.0):
    """Index and distance of the nearest of ``points`` to ``q``, first on ties"""
    best, best_d = None, np.inf
    for i, x in enumerate(points):
        if x is None:
            continue
        d = float(np.sum(np.abs(np.asarray(x, float) - np.asarray(q, float)) ** p) ** (1 / p))
        if d < best_d:
            best, best_d = i, d
    return best, best_d


class TestClosestMatrix(unittest.TestCase):
    def check(self, g, points, q, p=2.0):
        v, d = g.closest(q)
        i, d_expected = oracle(points, q, p)
        if i is None:
            self.assertIsNone(v)
            self.assertEqual(d, np.inf)
        else:
            self.assertIs(v, g[i])
            self.assertAlmostEqual(d, d_expected, places=12)

    def test_matches_oracle(self):
        rng = np.random.default_rng(1)
        pts = rng.uniform(-5, 5, (100, 2))
        g = UGraph()
        for x in pts:
            g.add_vertex(coord=x)
        for q in rng.uniform(-6, 6, (20, 2)):
            self.check(g, pts, q)

    def test_interleaved_add_and_query(self):
        """The planners' pattern, over enough vertices to grow the buffer"""
        rng = np.random.default_rng(2)
        g = UGraph()
        pts = []
        for _ in range(300):
            x = rng.uniform(-5, 5, 2)
            g.add_vertex(coord=x)
            pts.append(x)
            self.check(g, pts, rng.uniform(-5, 5, 2))
        g.validate()

    def test_l1_and_se2_metrics(self):
        rng = np.random.default_rng(3)
        pts = rng.uniform(-3, 3, (60, 2))
        g = UGraph(metric="L1")
        for x in pts:
            g.add_vertex(coord=x)
        for q in rng.uniform(-3, 3, (10, 2)):
            self.check(g, pts, q, p=1.0)

        # SE2: planar distance plus weighted, wrapped, heading difference
        w = 2.0
        poses = np.column_stack([rng.uniform(-3, 3, (40, 2)), rng.uniform(-3, 3, 40)])
        h = UGraph(metric=SE2Metric(weight=w))
        for x in poses:
            h.add_vertex(coord=x)
        for q in poses[:5] + 0.01:
            v, d = h.closest(q)
            best = np.inf
            for i, x in enumerate(poses):
                th = (x[2] - q[2] + np.pi) % (2 * np.pi) - np.pi
                best = min(best, np.hypot(np.hypot(x[0] - q[0], x[1] - q[1]), w * th))
            self.assertAlmostEqual(d, best, places=12)

    def test_first_wins_a_tie(self):
        g = UGraph()
        g.add_vertex(coord=[1, 0], name="a")
        g.add_vertex(coord=[-1, 0], name="b")
        self.assertEqual(g.closest([0, 0])[0].name, "a")

    def test_vertex_without_coordinate_is_skipped(self):
        g = UGraph()
        g.add_vertex(name="bare")
        self.assertEqual(g.closest([0, 0]), (None, np.inf))  # nothing embedded
        g.add_vertex(coord=[3, 4], name="p")
        g.add_vertex(name="bare2")
        g.add_vertex(coord=[1, 1], name="q")
        v, d = g.closest([0, 0])
        self.assertEqual(v.name, "q")
        self.assertAlmostEqual(d, np.sqrt(2))
        # built before the unembedded vertex was added, and after
        self.assertEqual(g.closest([3, 4])[0].name, "p")

    def test_remove_vertex(self):
        pts = [[0, 0], [1, 0], [2, 0]]
        g = UGraph()
        for i, x in enumerate(pts):
            g.add_vertex(coord=x, name=f"v{i}")
        self.assertEqual(g.closest([0.1, 0])[0].name, "v0")
        g.remove_vertex(g["v0"])
        v, d = g.closest([0.1, 0])
        self.assertEqual(v.name, "v1")
        self.assertAlmostEqual(d, 0.9)
        g.validate()

    def test_assigning_a_coordinate(self):
        g = UGraph()
        a = g.add_vertex(coord=[0, 0], name="a")
        g.add_vertex(coord=[5, 5], name="b")
        self.assertEqual(g.closest([0.1, 0])[0].name, "a")
        a.coord = np.array([9.0, 9.0])
        self.assertEqual(g.closest([0.1, 0])[0].name, "b")
        a.coord += 1  # also seen
        self.assertEqual(g.closest([10, 10])[0].name, "a")

    def test_in_place_change_is_not_seen(self):
        """The documented limit, shared with the k-d tree"""
        g = UGraph()
        a = g.add_vertex(coord=[0.0, 0.0], name="a")
        g.add_vertex(coord=[5.0, 5.0], name="b")
        g.closest([0, 0])  # build
        a.coord[0] = 9.0
        self.assertEqual(g.closest([0.1, 0])[0].name, "a")  # the stale answer

    def test_nonfinite_coordinates_are_left_out(self):
        g = UGraph()
        g.add_vertex(coord=[np.nan, 0], name="nan")
        g.add_vertex(coord=[1, 1], name="ok")
        self.assertEqual(g.closest([0, 0])[0].name, "ok")  # built without it
        g.add_vertex(coord=[np.inf, 0], name="inf")  # an append that is refused
        self.assertEqual(g.closest([0, 0])[0].name, "ok")
        h = UGraph()
        h.add_vertex(coord=[np.nan, np.nan])
        self.assertEqual(h.closest([0, 0]), (None, np.inf))

    def test_non_finite_query_and_wrong_length_query(self):
        g = UGraph()
        g.add_vertex(coord=[0, 0], name="a")
        g.add_vertex(coord=[1, 1], name="b")
        self.assertEqual(g.closest([0, 0])[0].name, "a")
        v, d = g.closest([np.nan, 0])  # as the pass over the vertices does
        self.assertIsNone(v)
        with self.assertRaises(ValueError):
            g.closest([0, 0, 0])

    def test_a_vertex_of_another_length_is_reported(self):
        g = UGraph()
        g.add_vertex(coord=[0, 0])
        g.closest([0, 0])  # build
        g.add_vertex(coord=[1, 2, 3])  # does not fit; the matrix is dropped
        with self.assertRaises(ValueError):
            g.closest([0, 0])
        h = UGraph()
        h.add_vertex(coord=[0, 0])
        h.add_vertex(coord=[1, 2, 3])
        with self.assertRaises(ValueError):  # ragged when first built
            h.closest([0, 0])

    def test_non_numeric_coordinate_falls_back(self):
        g = UGraph()
        g.add_vertex(coord=["a", "b"])
        with self.assertRaises(TypeError):  # from the pass over the vertices
            g.closest([0, 0])

    def test_coordinate_that_is_not_a_vector_is_left_to_the_pass(self):
        g = UGraph()
        g.add_vertex(coord=np.ones((2, 2)))  # a matrix
        v, d = g.closest([1, 1])  # broadcast against it, as it always was
        self.assertIs(v, g[0])
        self.assertEqual(d, 0.0)
        self.assertIsNone(g._pts)
        h = UGraph()
        h.add_vertex(coord=[])  # no elements
        self.assertEqual(h.closest([])[1], 0.0)
        self.assertIsNone(h._pts)

    def test_user_metric_is_applied_to_each_vertex(self):
        calls = []

        def metric(v):
            calls.append(1)
            return float(np.abs(v).max())

        g = UGraph(metric=metric)
        for x in ([0, 0], [3, 1], [1, 3]):
            g.add_vertex(coord=x)
        v, d = g.closest([2, 0])
        self.assertEqual(d, 1.0)
        self.assertGreaterEqual(len(calls), 3)

    def test_near(self):
        rng = np.random.default_rng(4)
        pts = rng.uniform(-5, 5, (80, 2))
        g = UGraph()
        for x in pts:
            g.add_vertex(coord=x)
        g.add_vertex(name="bare")
        q, r = np.array([0.5, -0.5]), 2.5
        got = g.near(q, r)
        want = sorted(
            [(i, float(np.hypot(*(x - q)))) for i, x in enumerate(pts) if np.hypot(*(x - q)) <= r],
            key=lambda t: t[1],
        )
        self.assertEqual([g._vertexlist.index(v) for v, _ in got], [i for i, _ in want])
        for (_, d), (_, e) in zip(got, want):
            self.assertAlmostEqual(d, e, places=12)
        # after a vertex is added, with the matrix already built
        g.add_vertex(coord=q + 0.01, name="new")
        self.assertEqual(g.near(q, r)[0][0].name, "new")

    def test_pickle_and_copy_do_not_carry_it(self):
        g = UGraph()
        for x in ([0, 0], [4, 4], [1, 1]):
            g.add_vertex(coord=x)
        g.closest([0, 0])
        for h in (roundtrip(g), copy.deepcopy(g), g.subgraph([True, True, False])):
            self.assertIsNone(h._pts)
            self.assertIs(h.closest([4, 4])[0], h[1])
            h.add_vertex(coord=[4.1, 4.1], name="x")
            self.assertEqual(h.closest([4.2, 4.2])[0].name, "x")
            h.validate()

    def test_vertex_instance_added(self):
        g = UGraph()
        g.add_vertex(coord=[0, 0])
        g.closest([0, 0])
        g.add_vertex(Vertex(coord=[2, 2], name="inst"))
        g.add_vertex(Vertex(name="bare"))
        self.assertEqual(g.closest([2.1, 2.1])[0].name, "inst")


if __name__ == "__main__":
    unittest.main()
