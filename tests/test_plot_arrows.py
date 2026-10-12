"""
plot() draws each edge once, as an arc patch: a directed graph with
arrowheads, parallel and antiparallel edges bowed apart, and loops.

In 0.6.5 every vertex drew a straight line to each of its neighbours, so a
directed graph looked like an undirected one, an undirected edge was drawn
twice, parallel edges were drawn on top of each other, and loops were not
visible at all.
"""

import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ArrowStyle, ConnectionStyle, FancyArrowPatch

from pgraph import DGraph, UGraph
from pgraph.PGraph import _Arrow3D, _Loop3D


def build(cls, edges, loops=(), dim=2):
    """Four vertices a-d on a unit square, and edges named by their letters"""
    g = cls()
    xy = dict(a=(0, 0), b=(2, 0), c=(2, 2), d=(0, 2))
    for name, c in xy.items():
        g.add_vertex(list(c) + [0] * (dim - 2), name=name)
    made = [g.add_edge(u, v) for u, v in edges]
    return g, made


class _Plot(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def draw(self, g, **kwargs):
        fig, ax = plt.subplots()
        g.plot(ax=ax, block=None, **kwargs)
        fig.canvas.draw()
        return ax

    def arrows(self, ax):
        return [p for p in ax.patches if isinstance(p, FancyArrowPatch)]


class TestEdgesAreDrawnOnce(_Plot):
    def test_one_patch_per_edge_undirected(self):
        g, _ = build(UGraph, [("a", "b"), ("b", "c"), ("c", "d")])
        self.assertEqual(len(self.arrows(self.draw(g))), 3)  # was 6: each edge twice

    def test_one_patch_per_edge_directed(self):
        g, _ = build(DGraph, [("a", "b"), ("b", "c"), ("c", "d")])
        self.assertEqual(len(self.arrows(self.draw(g))), 3)

    def test_parallel_and_loops_each_have_a_patch(self):
        g, _ = build(UGraph, [("a", "b"), ("a", "b"), ("c", "c"), ("c", "c"), ("d", "a")])
        self.assertEqual(len(self.arrows(self.draw(g))), g.ne)


class TestArrowheads(_Plot):
    def style(self, patch):
        return patch.get_arrowstyle()

    def test_directed_edges_have_heads(self):
        g, _ = build(DGraph, [("a", "b"), ("b", "c")])
        for p in self.arrows(self.draw(g)):
            self.assertIsInstance(self.style(p), ArrowStyle.CurveFilledB)

    def test_undirected_edges_have_none(self):
        g, _ = build(UGraph, [("a", "b"), ("b", "c")])
        for p in self.arrows(self.draw(g)):
            self.assertIsInstance(self.style(p), ArrowStyle.Curve)

    def test_head_is_at_the_end_vertex(self):
        g, _ = build(DGraph, [("a", "b")])
        (p,) = self.arrows(self.draw(g))
        # the shaft starts at a (0, 0) and the head is out at b (2, 0): the
        # path has the head's vertices, so the end is its largest x
        path = p.get_path().vertices
        self.assertLess(path[0][0], 0.5)
        self.assertGreater(path[:, 0].max(), 1.5)
        self.assertLess(path[:, 0].max(), 2.0)  # stops short of the vertex centre

    def test_edge_stops_at_the_marker_not_the_vertex_centre(self):
        g, _ = build(DGraph, [("a", "b")])
        (p,) = self.arrows(self.draw(g))
        self.assertGreater(p.shrinkA, 0)   # points: half the marker, and a little
        self.assertGreater(p.shrinkB, 0)

    def test_arrow_opt_overrides_the_style(self):
        g, _ = build(DGraph, [("a", "b")])
        (p,) = self.arrows(self.draw(g, arrow_opt=dict(arrowstyle="->", mutation_scale=30)))
        self.assertIsInstance(self.style(p), ArrowStyle.CurveB)
        self.assertEqual(p.get_mutation_scale(), 30)

    def test_arrow_opt_can_give_an_undirected_graph_heads(self):
        g, _ = build(UGraph, [("a", "b")])
        (p,) = self.arrows(self.draw(g, arrow_opt=dict(arrowstyle="<->")))
        self.assertIsInstance(self.style(p), ArrowStyle.CurveAB)

    def test_arrow_opt_none_is_the_default(self):
        g, _ = build(DGraph, [("a", "b")])
        (p,) = self.arrows(self.draw(g, arrow_opt=None))
        self.assertIsInstance(self.style(p), ArrowStyle.CurveFilledB)


class TestParallelEdges(_Plot):
    def rad(self, patch):
        style = patch.get_connectionstyle()
        self.assertIsInstance(style, ConnectionStyle.Arc3)
        return style.rad

    def side(self, patch):
        """+1 or -1: which side of the straight line between its ends the arc bows to"""
        v = patch.get_path().vertices
        start, end = v[0], v[-1]
        mid = v.mean(axis=0)
        d, m = end - start, mid - start
        return float(np.sign(d[0] * m[1] - d[1] * m[0]))

    def test_a_single_edge_is_straight(self):
        g, _ = build(DGraph, [("a", "b")])
        (p,) = self.arrows(self.draw(g))
        self.assertEqual(self.rad(p), 0)

    def test_parallel_edges_bow_to_either_side(self):
        g, _ = build(UGraph, [("a", "b"), ("a", "b")])
        p, q = self.arrows(self.draw(g))
        self.assertEqual(self.rad(p), -self.rad(q))
        self.assertNotEqual(self.rad(p), 0)
        self.assertNotEqual(self.side(p), self.side(q))

    def test_three_parallel_edges_one_is_straight(self):
        g, _ = build(UGraph, [("a", "b"), ("a", "b"), ("a", "b")])
        rads = sorted(self.rad(p) for p in self.arrows(self.draw(g)))
        self.assertAlmostEqual(rads[1], 0)
        self.assertAlmostEqual(rads[0], -rads[2])

    def test_antiparallel_edges_are_separate_arcs(self):
        # a->b and b->a join the same pair: two arcs, not one on the other
        g, _ = build(DGraph, [("a", "b"), ("b", "a")])
        p, q = self.arrows(self.draw(g))
        self.assertNotEqual(self.side(p), self.side(q))
        mid_p = p.get_path().vertices.mean(axis=0)
        mid_q = q.get_path().vertices.mean(axis=0)
        self.assertGreater(np.linalg.norm(mid_p - mid_q), 0.1)

    def test_other_pairs_are_unaffected(self):
        g, _ = build(UGraph, [("a", "b"), ("a", "b"), ("c", "d")])
        arrows = self.arrows(self.draw(g))
        self.assertEqual([self.rad(p) for p in arrows][2], 0)


class TestLoops(_Plot):
    def test_a_loop_is_drawn(self):
        g, _ = build(UGraph, [("a", "a")])
        (p,) = self.arrows(self.draw(g))
        # the path is in points from the vertex: a circle that passes through
        # it, so it comes within a marker radius of the origin and is about a
        # loop radius across
        verts = p.get_path().vertices
        self.assertGreater(len(verts), 4)
        self.assertLess(np.linalg.norm(verts, axis=1).min(), 10)
        self.assertGreater(np.ptp(verts[:, 0]) + np.ptp(verts[:, 1]), 30)

    def test_a_directed_loop_has_a_head(self):
        g, _ = build(DGraph, [("a", "a")])
        (p,) = self.arrows(self.draw(g))
        self.assertIsInstance(p.get_arrowstyle(), ArrowStyle.CurveFilledB)

    def test_loops_at_one_vertex_are_different(self):
        g, _ = build(UGraph, [("a", "a"), ("a", "a"), ("a", "a")])
        paths = [p.get_path().vertices for p in self.arrows(self.draw(g))]
        centres = {tuple(np.round(v.mean(axis=0), 2)) for v in paths}
        self.assertEqual(len(centres), 3)

    def test_a_loop_is_not_clipped_by_the_axes(self):
        g, _ = build(UGraph, [("a", "a")])
        (p,) = self.arrows(self.draw(g))
        self.assertFalse(p.get_clip_on())


class TestOptions(_Plot):
    def setUp(self):
        self.g, _ = build(UGraph, [("a", "b"), ("c", "d")])

    def test_edge_opt_linewidth(self):
        for p in self.arrows(self.draw(self.g, edge_opt=dict(linewidth=7))):
            self.assertEqual(p.get_linewidth(), 7)

    def test_default_edge_linewidth(self):
        for p in self.arrows(self.draw(self.g)):
            self.assertEqual(p.get_linewidth(), 3)

    def test_edge_opt_that_is_not_an_edge_option_is_reported(self):
        with self.assertWarns(UserWarning):
            self.draw(self.g, edge_opt=dict(marker="x"))

    def test_vertex_opt_markersize(self):
        ax = self.draw(self.g, vertex_opt=dict(markersize=5))
        self.assertEqual({line.get_markersize() for line in ax.lines}, {5})

    def test_labels_default_on(self):
        ax = self.draw(self.g)
        self.assertEqual(sorted(t.get_text().strip() for t in ax.texts), list("abcd"))

    def test_labels_false(self):
        self.assertEqual(len(self.draw(self.g, labels=False).texts), 0)

    def test_text_opt(self):
        ax = self.draw(self.g, text_opt=dict(color="r", fontsize=17))
        self.assertEqual({t.get_color() for t in ax.texts}, {"r"})
        self.assertEqual({t.get_fontsize() for t in ax.texts}, {17})

    def test_color_components_colours_each_component(self):
        ax = self.draw(self.g, color_components=True)
        colors = {tuple(p.get_edgecolor()) for p in self.arrows(ax)}
        self.assertEqual(len(colors), 2)  # a-b and c-d are two components

    def test_without_component_colors_everything_is_one_color(self):
        ax = self.draw(self.g, color_components=False)
        self.assertEqual(len({tuple(p.get_edgecolor()) for p in self.arrows(ax)}), 1)
        self.assertEqual(len({line.get_color() for line in ax.lines}), 1)

    def test_force_2d_with_3d_coordinates(self):
        g, _ = build(UGraph, [("a", "b")], dim=3)
        fig = plt.figure()
        ax = fig.add_subplot()
        g.plot(ax=ax, force_2d=True, block=None)
        self.assertEqual(len(self.arrows(ax)), 1)


class TestOldNames(_Plot):
    """The 0.6.5 names still work, with a DeprecationWarning that names the new"""

    def setUp(self):
        self.g, _ = build(UGraph, [("a", "b"), ("c", "d")])

    def deprecated(self, old, new, **kwargs):
        with self.assertWarns(DeprecationWarning) as cm:
            ax = self.draw(self.g, **kwargs)
        self.assertIn(old, str(cm.warning))
        self.assertIn(new, str(cm.warning))
        self.assertEqual(cm.filename, __file__.replace(".pyc", ".py"))  # blames the caller
        return ax

    def test_vopt(self):
        ax = self.deprecated("vopt", "vertex_opt", vopt=dict(markersize=4))
        self.assertEqual({line.get_markersize() for line in ax.lines}, {4})

    def test_eopt(self):
        ax = self.deprecated("eopt", "edge_opt", eopt=dict(linewidth=9))
        self.assertEqual({p.get_linewidth() for p in self.arrows(ax)}, {9})

    def test_text_false(self):
        self.assertEqual(len(self.deprecated("text", "labels", text=False).texts), 0)

    def test_text_dict(self):
        ax = self.deprecated("text", "text_opt", text=dict(color="r"))
        self.assertEqual({t.get_color() for t in ax.texts}, {"r"})

    def test_text_none_and_true_are_default_labels(self):
        for value in (None, True):
            with self.subTest(text=value):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    ax = self.draw(self.g, text=value)
                self.assertEqual(len(ax.texts), 4)

    def test_colorcomponents(self):
        ax = self.deprecated("colorcomponents", "color_components", colorcomponents=False)
        self.assertEqual(len({tuple(p.get_edgecolor()) for p in self.arrows(ax)}), 1)

    def test_force2d(self):
        g, _ = build(UGraph, [("a", "b")], dim=3)
        fig = plt.figure()
        ax = fig.add_subplot()
        with self.assertWarns(DeprecationWarning):
            g.plot(ax=ax, force2d=True, block=None)
        self.assertEqual(len(self.arrows(ax)), 1)

    def test_positional_arguments_still_mean_what_they_did(self):
        # plot(colorcomponents, force2d, vopt, eopt, text, block, grid, ax)
        fig, ax = plt.subplots()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.g.plot(False, False, dict(markersize=3), None, False, None, True, ax)
        self.assertEqual({line.get_markersize() for line in ax.lines}, {3})
        self.assertEqual(len(ax.texts), 0)

    def test_old_and_new_together_is_an_error(self):
        for kwargs in (
            dict(vopt={}, vertex_opt={}),
            dict(eopt={}, edge_opt={}),
            dict(colorcomponents=True, color_components=True),
            dict(force2d=True, force_2d=True),
            dict(text=False, labels=True),
            dict(text={}, text_opt={}),
        ):
            with self.subTest(**{k: type(v).__name__ for k, v in kwargs.items()}):
                with self.assertRaises(TypeError):
                    self.g.plot(block=None, **kwargs)


class Test3D(_Plot):
    def draw3d(self, g, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        g.plot(ax=ax, block=None, **kwargs)
        fig.canvas.draw()
        return fig, ax

    def artists(self, ax, kind):
        return [a for a in ax.get_children() if isinstance(a, kind)]

    def test_edges_are_3d_arrows(self):
        g, _ = build(DGraph, [("a", "b"), ("b", "c")], dim=3)
        fig, ax = self.draw3d(g)
        arrows = self.artists(ax, _Arrow3D)
        self.assertEqual(len(arrows), 2)
        for a in arrows:
            self.assertIsInstance(a.get_arrowstyle(), ArrowStyle.CurveFilledB)

    def test_undirected_have_no_head(self):
        g, _ = build(UGraph, [("a", "b")], dim=3)
        fig, ax = self.draw3d(g)
        (a,) = self.artists(ax, _Arrow3D)
        self.assertIsInstance(a.get_arrowstyle(), ArrowStyle.Curve)

    def test_loops_and_parallel_edges(self):
        g, _ = build(DGraph, [("a", "b"), ("a", "b"), ("c", "c")], dim=3)
        fig, ax = self.draw3d(g)
        self.assertEqual(len(self.artists(ax, _Arrow3D)), 2)
        self.assertEqual(len(self.artists(ax, _Loop3D)), 1)
        rads = sorted(a.get_connectionstyle().rad for a in self.artists(ax, _Arrow3D))
        self.assertAlmostEqual(rads[0], -rads[1])

    def test_follows_a_rotation(self):
        g, _ = build(DGraph, [("a", "b")], dim=3)
        fig, ax = self.draw3d(g)
        (a,) = self.artists(ax, _Arrow3D)
        before = np.array(a.get_path().vertices)
        ax.view_init(elev=60, azim=40)
        fig.canvas.draw()
        after = np.array(a.get_path().vertices)
        self.assertFalse(np.allclose(before, after))

    def test_four_coordinates_use_the_first_three(self):
        g, _ = build(UGraph, [("a", "b")], dim=4)
        fig, ax = self.draw3d(g)
        self.assertEqual(len(self.artists(ax, _Arrow3D)), 1)


class TestHighlight(_Plot):
    def test_highlight_follows_the_arc(self):
        g, (e1, e2) = build(UGraph, [("a", "b"), ("a", "b")])
        ax = self.draw(g)
        before = len(self.arrows(ax))
        g.highlight_edge(e2)
        arrows = self.arrows(ax)
        self.assertEqual(len(arrows), before + 1)
        highlight, edge = arrows[-1], arrows[1]
        np.testing.assert_allclose(
            highlight.get_path().vertices.mean(axis=0),
            edge.get_path().vertices.mean(axis=0),
            atol=0.02,
        )

    def test_highlight_a_loop(self):
        g, (e,) = build(DGraph, [("a", "a")])
        self.draw(g)
        g.highlight_edge(e)  # does not raise

    def test_highlight_path_through_parallel_edges_takes_the_cheapest(self):
        g, _ = build(DGraph, [("a", "b"), ("a", "b")])
        g.edges()[0].cost = 5
        g.edges()[1].cost = 1
        ax = self.draw(g)
        g.highlight_path([g["a"], g["b"]], block=None)
        highlight = self.arrows(ax)[-2]  # the edge; the vertices follow as markers
        np.testing.assert_allclose(
            highlight.get_path().vertices.mean(axis=0),
            self.arrows(ax)[1].get_path().vertices.mean(axis=0),
            atol=0.02,
        )


if __name__ == "__main__":
    unittest.main()
