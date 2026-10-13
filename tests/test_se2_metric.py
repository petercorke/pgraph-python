"""
The ``SE2Metric`` of a pose difference (x, y, theta) and its weight on the
angle, in graphs: edge costs, ``closest``, the A* heuristic, copying and
pickling.

Expectations are worked out by hand.
"""

import copy
import pickle  # nosec B403 - only used to round-trip objects made in this file
import unittest

import numpy as np

from pgraph import DGraph, SE2Metric, UGraph


class TestMetric(unittest.TestCase):
    def test_default_weight_is_the_string_se2(self):
        v = np.r_[3.0, 4.0, 1.0]
        self.assertEqual(SE2Metric()(v), UGraph(metric="SE2").metric(v))
        self.assertAlmostEqual(SE2Metric()(v), np.sqrt(26))  # 9 + 16 + 1

    def test_weight_scales_the_angle_only(self):
        v = np.r_[3.0, 4.0, np.pi / 2]
        self.assertAlmostEqual(SE2Metric(1)(v), np.sqrt(25 + (np.pi / 2) ** 2))
        self.assertAlmostEqual(SE2Metric(2)(v), np.sqrt(25 + np.pi**2))
        self.assertAlmostEqual(SE2Metric(0.5)(v), np.sqrt(25 + (np.pi / 4) ** 2))
        self.assertAlmostEqual(SE2Metric(0)(v), 5.0)

    def test_angle_is_wrapped_before_weighting(self):
        for turns in (-2, -1, 0, 1, 2):
            with self.subTest(turns=turns):
                v = np.r_[0.0, 0.0, 0.5 + turns * 2 * np.pi]
                self.assertAlmostEqual(SE2Metric(3)(v), 1.5)
        # just past pi is a turn the other way
        self.assertAlmostEqual(SE2Metric(1)(np.r_[0.0, 0.0, np.pi + 0.1]), np.pi - 0.1)

    def test_a_pure_turn_is_worth_the_weight_times_the_angle(self):
        self.assertAlmostEqual(SE2Metric(0.25)(np.r_[0.0, 0.0, 2.0]), 0.5)

    def test_the_argument_is_not_changed(self):
        v = np.r_[1.0, 2.0, 7.0]  # needs wrapping
        before = v.copy()
        SE2Metric(0.5)(v)
        np.testing.assert_array_equal(v, before)

    def test_a_list_is_accepted(self):
        self.assertAlmostEqual(SE2Metric(2)([0, 0, 1]), 2.0)

    def test_needs_three_elements(self):
        for v in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0]):
            with self.subTest(length=len(v)):
                with self.assertRaises(ValueError):
                    SE2Metric()(np.array(v))

    def test_bad_weights(self):
        for weight in (-1, np.inf, np.nan):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    SE2Metric(weight)

    def test_repr_equality_and_hash(self):
        self.assertEqual(repr(SE2Metric(0.5)), "SE2Metric(weight=0.5)")
        self.assertEqual(SE2Metric(0.5), SE2Metric(0.5))
        self.assertNotEqual(SE2Metric(0.5), SE2Metric(2))
        self.assertEqual(len({SE2Metric(0.5), SE2Metric(0.5), SE2Metric(2)}), 2)
        self.assertEqual(SE2Metric(), UGraph(metric="SE2").metric)


class TestInGraphs(unittest.TestCase):
    cls = UGraph

    def test_edge_cost_uses_the_weight(self):
        g = self.cls(dim=3, metric=SE2Metric(weight=2))
        a = g.add_vertex([0.0, 0.0, 0.0])
        b = g.add_vertex([0.0, 0.0, 0.5])
        self.assertAlmostEqual(g.add_edge(a, b).cost, 1.0)  # 2 * 0.5

    def test_closest_depends_on_the_weight(self):
        # p is 1 away in the plane and no turn; q is at the same place but
        # turned by 1 rad: nearer than p only if turning is cheap enough
        def nearest(weight):
            g = self.cls(dim=3, metric=SE2Metric(weight=weight))
            g.add_vertex([2.0, 0.0, 0.0], name="p")  # 2 away, no turn
            g.add_vertex([0.0, 0.0, 1.5], name="q")  # no distance, 1.5 rad
            return g.closest([0.0, 0.0, 0.0])

        v, d = nearest(1.0)  # q is 1.5 away, p is 2 away
        self.assertEqual((v.name, round(d, 6)), ("q", 1.5))
        v, d = nearest(2.0)  # q is 3 away, p is 2 away
        self.assertEqual((v.name, round(d, 6)), ("p", 2.0))
        v, d = nearest(0.0)  # q is 0 away
        self.assertEqual((v.name, round(d, 6)), ("q", 0.0))

    def test_metric_can_be_set_after(self):
        g = self.cls(dim=3)
        g.metric = SE2Metric(weight=3)
        self.assertAlmostEqual(g.metric(np.r_[0.0, 0.0, 1.0]), 3.0)

    def test_astar_with_the_weighted_metric_is_optimal(self):
        # heuristic defaults to the metric, which makes the edge costs and the
        # heuristic consistent; compare with the uniform cost search
        rng = np.random.default_rng(5)
        for weight in (0.0, 0.3, 1.0, 4.0):
            with self.subTest(weight=weight):
                g = self.cls(dim=3, metric=SE2Metric(weight=weight), heuristic=SE2Metric(weight=weight))
                vs = [g.add_vertex([*rng.uniform(0, 10, 2), rng.uniform(-3, 3)]) for _ in range(30)]
                for i in range(30):
                    for j in rng.choice(30, 3, replace=False):
                        if i != j:
                            g.add_edge(vs[i], vs[int(j)])
                for s, t in [(0, 29), (3, 17), (10, 25)]:
                    ucs = g.path_UCS(vs[s], vs[t])
                    astar = g.path_Astar(vs[s], vs[t])
                    self.assertEqual(ucs is None, astar is None)
                    if ucs is not None:
                        self.assertAlmostEqual(astar[1], ucs[1])

    def test_pickle_and_copy_keep_the_weight(self):
        g = self.cls(dim=3, metric=SE2Metric(weight=0.25), heuristic=SE2Metric(weight=0))
        g.add_vertex([0.0, 0.0, 0.0])
        for h in (pickle.loads(pickle.dumps(g)), g.copy(), copy.deepcopy(g)):  # nosec B301
            self.assertEqual(h.metric, SE2Metric(0.25))
            self.assertEqual(h.heuristic, SE2Metric(0))
        h = pickle.loads(pickle.dumps(self.cls(metric="SE2")))  # nosec B301
        self.assertEqual(h.metric, SE2Metric())

    def test_the_named_metric_is_still_pickled_by_name(self):
        g = self.cls(metric="SE2")
        self.assertIs(pickle.loads(pickle.dumps(g)).metric, g.metric)  # nosec B301


class TestInDirectedGraphs(TestInGraphs):
    cls = DGraph


if __name__ == "__main__":
    unittest.main()
