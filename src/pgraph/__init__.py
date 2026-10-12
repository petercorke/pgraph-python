import warnings
from typing import Any

from pgraph.PGraph import (
    Graph,
    Vertex,
    UGraph,
    DGraph,
    Edge,
)

__all__ = [
    "Graph",
    "Vertex",
    "UGraph",
    "DGraph",
    "Edge",
]


# UVertex and DVertex are deprecated names for Vertex, see pgraph.PGraph
def __getattr__(name: str) -> Any:
    if name in ("UVertex", "DVertex"):
        warnings.warn(
            f"pgraph.{name} is deprecated, use pgraph.Vertex instead: a vertex "
            "has no direction of its own, the graph it belongs to decides",
            DeprecationWarning,
            stacklevel=2,
        )
        return Vertex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
