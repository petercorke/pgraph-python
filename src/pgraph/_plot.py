"""
Matplotlib pieces of the plot of a graph (private module)

Imported by :meth:`pgraph.Graph.plot` and the methods that draw an edge, so that
``import pgraph`` does not load matplotlib: this module is the only part of
the package whose classes need it to be defined at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, ScaledTranslation
from mpl_toolkits.mplot3d import proj3d
from numpy.typing import ArrayLike


def _loop_path(angle: float, radius: float, markersize: float) -> Path:
    """
    The path of a loop at a vertex, in points from the vertex (private function)

    A circle of ``radius`` whose centre is that far from the vertex, in the
    direction ``angle`` degrees, less the part inside the vertex's marker.  It
    is counter-clockwise, so a directed loop has its head where it returns.
    """
    m = markersize / 2
    phi = np.degrees(np.arccos(np.clip(1 - m * m / (2 * radius * radius), -1, 1)))
    arc = Path.arc(angle + 180 + phi, angle + 540 - phi)
    centre = radius * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    return Path(np.asarray(arc.vertices) * radius + centre, arc.codes)


class _Arrow3D(FancyArrowPatch):
    """An edge between two points of a 3D plot, which follows a rotation"""

    def __init__(self, p1: ArrayLike, p2: ArrayLike, **kwargs: Any):
        super().__init__((0, 0), (0, 0), **kwargs)
        self._p1, self._p2 = np.asarray(p1, float), np.asarray(p2, float)

    def do_3d_projection(self, renderer: Any = None) -> float:
        ax: Any = self.axes
        xs, ys, zs = proj3d.proj_transform(
            [self._p1[0], self._p2[0]],
            [self._p1[1], self._p2[1]],
            [self._p1[2], self._p2[2]],
            ax.M,
        )
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))


class _Loop3D(FancyArrowPatch):
    """A loop at a vertex of a 3D plot, which follows a rotation"""

    def __init__(self, anchor: ArrayLike, path: Path, **kwargs: Any):
        super().__init__(path=path, **kwargs)
        self._anchor = np.asarray(anchor, float)

    def do_3d_projection(self, renderer: Any = None) -> float:
        ax: Any = self.axes
        x, y, z = proj3d.proj_transform(*self._anchor, ax.M)
        self.set_transform(
            Affine2D().scale(1 / 72)
            + ax.figure.dpi_scale_trans
            + ScaledTranslation(x, y, ax.transData)
        )
        return float(z)
