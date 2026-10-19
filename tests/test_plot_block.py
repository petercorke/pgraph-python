"""Regression tests for the three-state block= plotting contract."""

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgraph import UGraph


def make_graph() -> tuple[UGraph, list]:
    graph = UGraph()
    a = graph.add_vertex([0.0, 0.0], name="a")
    b = graph.add_vertex([1.0, 1.0], name="b")
    graph.add_edge(a, b)
    return graph, [a, b]


def test_default_plot_and_highlight_do_not_call_show() -> None:
    graph, path = make_graph()
    try:
        with patch.object(plt, "show") as show:
            graph.plot()
            axes = plt.gca()
            before = len(axes.patches)
            graph.highlight_path(path)
            show.assert_not_called()
        assert plt.gca() is axes
        assert len(axes.patches) > before
    finally:
        plt.close("all")


def test_explicit_plot_show_modes() -> None:
    graph, _ = make_graph()
    try:
        for block in (False, True):
            with patch.object(plt, "show") as show:
                graph.plot(block=block)
                show.assert_called_once_with(block=block)
    finally:
        plt.close("all")


def test_explicit_highlight_show_modes() -> None:
    graph, path = make_graph()
    try:
        graph.plot()
        for block in (False, True):
            with patch.object(plt, "show") as show:
                graph.highlight_path(path, block=block)
                show.assert_called_once_with(block=block)
    finally:
        plt.close("all")
