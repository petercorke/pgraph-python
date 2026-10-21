"""Regression tests for the three-state block= plotting contract."""

import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgraph import UGraph


class TestPlotBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = UGraph()
        a = self.graph.add_vertex([0.0, 0.0], name="a")
        b = self.graph.add_vertex([1.0, 1.0], name="b")
        self.graph.add_edge(a, b)
        self.path = [a, b]

    def tearDown(self) -> None:
        plt.close("all")

    def test_default_plot_and_highlight_do_not_call_show(self) -> None:
        with patch.object(plt, "show") as show:
            self.graph.plot()
            axes = plt.gca()
            before = len(axes.patches)
            self.graph.highlight_path(self.path)
            show.assert_not_called()
        self.assertIs(plt.gca(), axes)
        self.assertGreater(len(axes.patches), before)

    def test_explicit_plot_show_modes(self) -> None:
        for block in (False, True):
            with self.subTest(block=block):
                with patch.object(plt, "show") as show:
                    self.graph.plot(block=block)
                    show.assert_called_once_with(block=block)

    def test_explicit_highlight_show_modes(self) -> None:
        self.graph.plot()
        for block in (False, True):
            with self.subTest(block=block):
                with patch.object(plt, "show") as show:
                    self.graph.highlight_path(self.path, block=block)
                    show.assert_called_once_with(block=block)


if __name__ == "__main__":
    unittest.main()
