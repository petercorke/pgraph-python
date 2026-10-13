"""
Highlighting in three dimensions: ``highlight_edge``, ``highlight_path`` and
``highlight_vertex`` draw in the axes that ``plot()`` made.

``highlight_edge`` assumed two dimensions, so on a 3D graph it added a flat
patch to a 3D axes, which raised ``AttributeError`` when the figure was drawn,
and ``highlight_vertex`` drew its marker at z = 0.  In 0.6.5 ``highlight_path``
worked in 3D.
"""

import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pgraph import DGraph, UGraph
from pgraph._plot import _Arrow3D, _Loop3D


def graph(cls, dim=3):
    g = cls(dim=dim)
    coords = [[0, 0, 0], [1, 1, 0], [2, 0, 1], [0, 2, 2]]
    v = [g.add_vertex(c[:dim], name=n) for n, c in zip("abcd", coords)]
    edges = [g.add_edge(v[p], v[q]) for p, q in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1), (2, 2)]]
    return g, v, edges


class TestHighlight3D(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        warnings.simplefilter("ignore")

    def draws(self):
        fig = plt.gcf()
        fig.canvas.draw()  # the failure was here, not when the patch was added
        return fig.axes[0]

    def test_highlight_edge(self):
        for cls in (UGraph, DGraph):
            with self.subTest(cls=cls.__name__):
                plt.close("all")
                g, v, edges = graph(cls)
                g.plot(block=None)
                ax = plt.gca()
                before = len(ax.patches)
                g.highlight_edge(edges[1])
                self.assertEqual(len(ax.patches), before + 1)
                self.assertIsInstance(ax.patches[-1], _Arrow3D)
                self.draws()

    def test_highlight_a_loop(self):
        g, v, edges = graph(DGraph)
        g.plot(block=None)
        g.highlight_edge(edges[5])  # the loop on c
        self.assertIsInstance(plt.gca().patches[-1], _Loop3D)
        self.draws()

    def test_highlight_a_parallel_edge(self):
        g, v, edges = graph(UGraph)
        g.plot(block=None)
        g.highlight_edge(edges[4])  # the second edge from a to b
        self.assertIsInstance(plt.gca().patches[-1], _Arrow3D)
        self.draws()

    def test_highlight_path(self):
        g, v, edges = graph(UGraph)
        g.plot(block=None)
        g.highlight_path([v[0], v[1], v[2]], block=None)
        ax = self.draws()
        # the 5 edges that are not a loop, and the 2 of the path over them
        self.assertEqual(sum(isinstance(p, _Arrow3D) for p in ax.patches), 5 + 2)
        self.assertEqual(sum(isinstance(p, _Loop3D) for p in ax.patches), 1)

    def test_highlight_vertex_is_at_the_z_of_the_vertex(self):
        g, v, edges = graph(UGraph)
        g.plot(block=None)
        g.highlight_vertex(v[3])
        ax = self.draws()
        x, y, z = ax.lines[-1].get_data_3d()
        np.testing.assert_allclose([x[0], y[0], z[0]], [0, 2, 2])

    def test_highlight_several_vertices_by_name(self):
        g, v, edges = graph(UGraph)
        g.plot(block=None)
        n = len(plt.gca().lines)
        g.highlight_vertex(["a", v[2]])
        self.assertEqual(len(plt.gca().lines), n + 2)
        self.draws()

    def test_a_3d_graph_plotted_in_2d_is_highlighted_in_2d(self):
        g, v, edges = graph(UGraph)
        g.plot(block=None, force_2d=True)
        g.highlight_path([v[0], v[1], v[2]], block=None)
        g.highlight_vertex(v[1])
        ax = self.draws()
        self.assertNotEqual(getattr(ax, "name", None), "3d")
        np.testing.assert_allclose(ax.lines[-1].get_xydata(), [[1, 1]])


class TestHighlight2DUnchanged(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        warnings.simplefilter("ignore")

    def test_vertex_marker(self):
        g, v, edges = graph(DGraph, dim=2)
        g.plot(block=None)
        g.highlight_vertex(v[2], color="g", scale=1.5, alpha=0.25)
        line = plt.gca().lines[-1]
        np.testing.assert_allclose(line.get_xydata(), [[2, 0]])
        self.assertEqual(line.get_color(), "g")
        self.assertEqual(line.get_markersize(), 12 * 1.5)
        self.assertEqual(line.get_alpha(), 0.25)

    def test_edge_patch_is_flat(self):
        g, v, edges = graph(DGraph, dim=2)
        g.plot(block=None)
        g.highlight_edge(edges[0])
        self.assertNotIsInstance(plt.gca().patches[-1], _Arrow3D)
        plt.gcf().canvas.draw()


if __name__ == "__main__":
    unittest.main()
