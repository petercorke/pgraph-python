"""
plot() / highlight_*() argument handling and the errors they give for graphs
that cannot be drawn.  These were found by mypy: ``text=None`` and
``text=True`` crashed although the docstring documents ``None`` as "default
formatting", and unplottable graphs gave a bare TypeError/IndexError.
"""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgraph import DGraph, UGraph


class TestPlotArgs(unittest.TestCase):
    def setUp(self):
        self.g = UGraph()
        a = self.g.add_vertex([0, 0], name="a")
        b = self.g.add_vertex([1, 1], name="b")
        self.edge = self.g.add_edge(a, b)

    def tearDown(self):
        plt.close("all")

    def texts(self):
        return [t.get_text().strip() for t in plt.gca().texts]

    def test_text_default_labels_vertices(self):
        self.g.plot(block=None)
        self.assertEqual(sorted(self.texts()), ["a", "b"])

    def test_text_none_means_default_formatting(self):
        self.g.plot(text=None, block=None)
        self.assertEqual(sorted(self.texts()), ["a", "b"])

    def test_text_true_means_default_formatting(self):
        self.g.plot(text=True, block=None)
        self.assertEqual(sorted(self.texts()), ["a", "b"])

    def test_text_false_means_no_labels(self):
        self.g.plot(text=False, block=None)
        self.assertEqual(self.texts(), [])

    def test_text_dict_is_the_format(self):
        self.g.plot(text={"color": "r"}, block=None)
        self.assertEqual({t.get_color() for t in plt.gca().texts}, {"r"})

    def test_vopt_eopt_none(self):
        self.g.plot(vopt=None, eopt=None, block=None)

    def test_options_are_not_shared_between_calls(self):
        # a mutable default argument must not accumulate state
        self.g.plot(vopt={"markersize": 3}, block=None)
        plt.close("all")
        self.g.plot(block=None)
        sizes = {line.get_markersize() for line in plt.gca().lines}
        self.assertNotIn(3, sizes)

    def test_3d_text_none(self):
        g = UGraph()
        a = g.add_vertex([0, 0, 0], name="a")
        b = g.add_vertex([1, 1, 1], name="b")
        g.add_edge(a, b)
        g.plot(text=None, block=None)
        self.assertEqual(sorted(self.texts()), ["a", "b"])

    def test_3d_vertices_plotted_at_their_z(self):
        # without component colouring the vertex marker dropped z (plotted at 0)
        g = UGraph()
        a = g.add_vertex([0, 0, 5], name="a")
        b = g.add_vertex([1, 1, 7], name="b")
        g.add_edge(a, b)
        g.plot(colorcomponents=False, block=None)
        zs = set()
        for line in plt.gca().lines:
            x, y, z = line.get_data_3d()
            if len(x) == 1:  # a vertex marker, not an edge
                zs.add(float(z[0]))
        self.assertEqual(zs, {5.0, 7.0})

    def test_plot_graph_without_coordinates(self):
        g = UGraph()
        g.add_edge(g.add_vertex(), g.add_vertex())
        with self.assertRaises(ValueError):
            g.plot(block=None)

    def test_plot_graph_with_one_coordinate(self):
        g = UGraph()
        g.add_vertex([0])
        with self.assertRaises(ValueError):
            g.plot(block=None)

    def test_plot_empty_graph(self):
        with self.assertRaises(ValueError):
            DGraph().plot(block=None)

    def test_highlight_edge(self):
        self.g.plot(block=None)
        self.g.highlight_edge(self.edge)

    def test_highlight_removed_edge(self):
        self.g.plot(block=None)
        self.g.remove_edge(self.edge)
        with self.assertRaises(ValueError):
            self.g.highlight_edge(self.edge)


if __name__ == "__main__":
    unittest.main()
