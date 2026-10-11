"""``laplacian()`` is canonical; ``Laplacian()`` is a deprecated alias."""

import unittest
import warnings

import numpy as np

from pgraph import DGraph, UGraph


class _Name:
    cls: type

    def setUp(self):
        self.g = self.cls()
        a, b, c = (self.g.add_vertex() for _ in range(3))
        self.g.add_edge(a, b)
        self.g.add_edge(b, c)

    def test_lowercase_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            L = self.g.laplacian()
        self.assertEqual(L.shape, (3, 3))

    def test_capitalized_warns_and_agrees(self):
        with self.assertWarns(DeprecationWarning) as cm:
            L = self.g.Laplacian()
        self.assertIn("laplacian()", str(cm.warning))
        self.assertEqual(cm.filename, __file__)  # stacklevel points at caller
        np.testing.assert_array_equal(L, self.g.laplacian())


class TestUGraphName(_Name, unittest.TestCase):
    cls = UGraph


class TestDGraphName(_Name, unittest.TestCase):
    cls = DGraph


if __name__ == "__main__":
    unittest.main()
