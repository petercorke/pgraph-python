"""Graph constructor arguments are keyword-only.

In 0.6.5 the first positional argument was accepted and ignored, so
``UGraph(6)`` looked as though it set a dimension but did nothing.
"""

import unittest

from pgraph import DGraph, UGraph


class _Keywords:
    cls: type

    def test_positional_rejected(self):
        for arg in (6, "L2", None, True):
            with self.subTest(arg=arg):
                with self.assertRaises(TypeError):
                    self.cls(arg)

    def test_keywords_accepted(self):
        g = self.cls(dim=6, metric="L1", heuristic="L2", verbose=False)
        self.assertEqual(g.n, 0)
        with self.assertRaises(ValueError):
            g.add_vertex([1, 2])

    def test_no_arguments(self):
        self.assertEqual(self.cls().n, 0)


class TestUGraphKeywords(_Keywords, unittest.TestCase):
    cls = UGraph


class TestDGraphKeywords(_Keywords, unittest.TestCase):
    cls = DGraph


if __name__ == "__main__":
    unittest.main()
