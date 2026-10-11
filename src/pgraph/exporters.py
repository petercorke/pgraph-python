"""
Export a graph as text (GraphViz DOT, Mermaid, GraphML, ELK JSON) or as a
rendered image (via the GraphViz command line tools).

This module is reached through :meth:`pgraph.PGraph.BaseGraph.export`, which
dispatches on the format name or the file extension.  Each text exporter is a
plain function ``f(graph, **options) -> str``, registered with
:func:`register_exporter`, so a user can add a format without touching the
package::

    from pgraph.exporters import register_exporter

    @register_exporter("gexf", ".gexf")
    def gexf(graph, *, version="1.3") -> str:
        ...

None of the text formats needs a dependency beyond the standard library.  Only
the rendered formats (:data:`RENDER_FORMATS`) need the GraphViz executables.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404 - GraphViz is run without a shell, see render()
from typing import IO, TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .PGraph import BaseVertex, Edge, BaseGraph

TextExporter = Callable[..., str]

_TEXT: dict[str, TextExporter] = {}
_EXTENSIONS: dict[str, str] = {}

#: GraphViz layout programs that may be run to render an image
ENGINES = ("dot", "neato", "fdp", "sfdp", "twopi", "circo", "osage", "patchwork")

#: image formats produced by running the GraphViz ``dot``/``neato`` executable,
#: mapped to whether the result is text (``True``) or binary (``False``)
RENDER_FORMATS: dict[str, bool] = {
    "svg": True,
    "ps": True,
    "pdf": False,
    "png": False,
    "jpg": False,
    "jpeg": False,
    "gif": False,
}


def register_exporter(
    name: str, *extensions: str
) -> Callable[[TextExporter], TextExporter]:
    """
    Register a text exporter (decorator)

    :param name: format name used as ``export(format=name)``
    :param extensions: file extensions, with the leading dot, that select this
        format when ``format`` is not given
    :return: the decorator

    The decorated function is called as ``f(graph, **options)`` and returns the
    text.  Declare its options as keyword-only parameters so that an unknown
    option is a ``TypeError``.
    """

    def decorator(func: TextExporter) -> TextExporter:
        _TEXT[name] = func
        for ext in extensions:
            _EXTENSIONS[ext.lower()] = name
        return func

    return decorator


def formats() -> list[str]:
    """
    Names of all supported export formats

    :return: text formats followed by rendered image formats
    :rtype: list of str
    """
    return sorted(_TEXT) + sorted(RENDER_FORMATS)


def format_from_filename(filename: str | os.PathLike | IO | None) -> str | None:
    """
    Infer an export format from a file name's extension

    :param filename: path, or a file object with a ``name`` attribute
    :return: the format name, or None if it cannot be decided
    :rtype: str or None

    Compound extensions such as ``.elk.json`` are tried before the plain
    suffix.
    """
    if isinstance(filename, (str, os.PathLike)):
        name = os.fspath(filename)
    else:
        name = getattr(filename, "name", None)
    if not isinstance(name, str):
        return None
    lower = name.lower()
    for ext in sorted(_EXTENSIONS, key=len, reverse=True):
        if lower.endswith(ext):
            return _EXTENSIONS[ext]
    suffix = os.path.splitext(lower)[1].lstrip(".")
    return suffix if suffix in RENDER_FORMATS else None


# --------------------------------------------------------------------------- #
# helpers shared by the exporters


def _is_directed(graph: BaseGraph) -> bool:
    from .PGraph import DGraph  # not at module level: PGraph imports this module

    return isinstance(graph, DGraph)


def _ordered_edges(graph: BaseGraph) -> list[tuple[Edge, BaseVertex, BaseVertex]]:
    """
    Edges in a reproducible order

    ``graph.edges()`` is a set, so its order changes from run to run and the
    exported text would not be stable (useless for diffs and golden files).
    Instead walk the vertices in creation order and each vertex's own edge
    list, keeping the first sight of every edge.  Identity is by object, not by
    endpoints, so parallel edges are all kept, and an undirected edge (held by
    both ends, twice for a self-loop) is listed once.

    Each item is ``(edge, start, end)``.
    """
    seen: set[int] = set()
    edges: list[tuple[Edge, BaseVertex, BaseVertex]] = []
    for vertex in graph:
        for e in vertex._edgelist:
            if id(e) in seen or e.v1 is None or e.v2 is None:
                continue
            seen.add(id(e))
            edges.append((e, e.v1, e.v2))
    return edges


def _component_colors(graph: BaseGraph) -> dict[BaseVertex, str]:
    """
    Map each vertex to a ``#rrggbb`` colour for its component, as plot() does
    """
    import matplotlib.pyplot as plt
    import numpy as np

    nc = graph.nc  # refreshes vertex.label
    cmap = plt.cm.coolwarm(np.linspace(0, 1, max(nc, 1)))
    colors = {}
    for vertex in graph:
        if vertex.label is not None:
            r, g, b, _ = cmap[vertex.label]
            colors[vertex] = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )
    return colors


def _xml_text(value: Any) -> str:
    """
    Escape a value for XML element content or a double-quoted attribute

    Characters that XML 1.0 cannot represent at all (most control characters)
    are replaced with U+FFFD, since escaping cannot make them valid.
    """
    text = str(value)
    text = "".join(
        c if c in "\t\n\r" or ord(c) >= 0x20 else "\ufffd" for c in text
    )
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _dot_quote(text: str) -> str:
    """Quote a string for use as a DOT identifier or attribute value"""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped.replace("\r\n", "\\n").replace("\n", "\\n") + '"'


# --------------------------------------------------------------------------- #
# text exporters


@register_exporter("dot", ".dot", ".gv")
def dot(
    graph: BaseGraph,
    *,
    direction: str | None = None,
    colorcomponents: bool = False,
    costs: bool = True,
) -> str:
    """
    GraphViz DOT text

    :param direction: layout direction, ``rankdir`` in DOT, eg. ``"LR"``
    :param colorcomponents: fill vertices by connected component
    :param costs: label edges with their cost
    """
    directed = _is_directed(graph)
    link = "->" if directed else "--"
    colors = _component_colors(graph) if colorcomponents else {}

    lines = ["digraph {" if directed else "graph {"]
    if direction is not None:
        lines.append(f"  rankdir={_dot_quote(direction)}")

    for vertex in graph:
        attrs = []
        if vertex.coord is not None and len(vertex.coord) >= 2:
            # the "!" pins the position when the neato engine is used
            attrs.append(
                'pos="{:g},{:g}!"'.format(vertex.coord[0], vertex.coord[1])
            )
        if vertex in colors:
            attrs.append('style="filled"')
            attrs.append(f'fillcolor="{colors[vertex]}"')
        suffix = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f"  {_dot_quote(str(vertex.name))}{suffix}")

    for e, v1, v2 in _ordered_edges(graph):
        attrs = []
        if costs and e.cost is not None:
            attrs.append(f'label="{e.cost:.4g}"')
        suffix = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(
            f"  {_dot_quote(str(v1.name))} {link} {_dot_quote(str(v2.name))}{suffix}"
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


@register_exporter("mermaid", ".mmd", ".mermaid")
def mermaid(
    graph: BaseGraph,
    *,
    direction: str = "TB",
    colorcomponents: bool = False,
    costs: bool = True,
    fence: bool = False,
) -> str:
    """
    Mermaid flowchart text

    :param direction: ``"TB"``, ``"BT"``, ``"LR"`` or ``"RL"``
    :param colorcomponents: fill vertices by connected component
    :param costs: label edges with their cost
    :param fence: wrap in a Markdown ```` ```mermaid ```` code fence

    Mermaid flowcharts are laid out automatically and cannot pin node
    positions, so vertex coordinates are not represented.  Vertices are given
    generated ids (``n0``, ``n1``, ...) with the real name as the label, since
    names are arbitrary strings but Mermaid ids are not.
    """
    if direction not in ("TB", "TD", "BT", "LR", "RL"):
        raise ValueError(f"direction must be TB, BT, LR or RL, got {direction!r}")

    arrow = "-->" if _is_directed(graph) else "---"
    ids: dict[BaseVertex, str] = {}
    lines = [f"flowchart {direction}"]
    for i, vertex in enumerate(graph):
        ids[vertex] = f"n{i}"
        label = str(vertex.name).replace('"', "#quot;")
        lines.append(f'    n{i}["{label}"]')

    for e, v1, v2 in _ordered_edges(graph):
        if costs and e.cost is not None:
            lines.append(f'    {ids[v1]} {arrow}|"{e.cost:.4g}"| {ids[v2]}')
        else:
            lines.append(f"    {ids[v1]} {arrow} {ids[v2]}")

    if colorcomponents:
        colors = _component_colors(graph)
        for vertex, color in colors.items():
            lines.append(f"    style {ids[vertex]} fill:{color}")

    text = "\n".join(lines) + "\n"
    if fence:
        text = f"```mermaid\n{text}```\n"
    return text


@register_exporter("graphml", ".graphml")
def graphml(graph: BaseGraph) -> str:
    """
    GraphML XML text

    Name, component, coordinate (``x``, ``y`` and the full ``coord`` string for
    higher dimensions) and edge cost are written as ``<data>`` elements with
    ``<key>`` declarations.  ``edgedefault`` follows the graph kind.  GraphML
    has no standard node position outside the proprietary yFiles extension,
    which is not used.
    """
    graph.nc  # refresh vertex.label
    has_coords = any(v.coord is not None for v in graph)
    edges = _ordered_edges(graph)
    has_cost = any(e.cost is not None for e, _, _ in edges)

    def key(key_id: str, domain: str, name: str, type_: str) -> str:
        return (
            f'  <key id="{key_id}" for="{domain}" '
            f'attr.name="{name}" attr.type="{type_}"/>'
        )

    kind = "directed" if _is_directed(graph) else "undirected"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        key("node_label", "node", "label", "string"),
        key("component", "node", "component", "int"),
    ]
    if has_coords:
        lines += [
            key("x", "node", "x", "double"),
            key("y", "node", "y", "double"),
            key("coord", "node", "coord", "string"),
        ]
    if has_cost:
        lines.append(key("edge_cost", "edge", "cost", "double"))
    lines.append(f'  <graph id="G" edgedefault="{kind}">')

    ids: dict[BaseVertex, str] = {}
    for i, vertex in enumerate(graph):
        ids[vertex] = f"n{i}"
        lines.append(f'    <node id="n{i}">')
        lines.append(f'      <data key="node_label">{_xml_text(vertex.name)}</data>')
        if vertex.label is not None:
            lines.append(f'      <data key="component">{vertex.label}</data>')
        if vertex.coord is not None:
            if len(vertex.coord) >= 2:
                lines.append(f'      <data key="x">{vertex.coord[0]:.6g}</data>')
                lines.append(f'      <data key="y">{vertex.coord[1]:.6g}</data>')
            coord = ",".join(f"{c:.6g}" for c in vertex.coord)
            lines.append(f'      <data key="coord">{coord}</data>')
        lines.append("    </node>")

    for i, (e, v1, v2) in enumerate(edges):
        lines.append(f'    <edge id="e{i}" source="{ids[v1]}" target="{ids[v2]}">')
        if e.cost is not None:
            lines.append(f'      <data key="edge_cost">{e.cost:.6g}</data>')
        lines.append("    </edge>")

    lines += ["  </graph>", "</graphml>"]
    return "\n".join(lines) + "\n"


@register_exporter("elk", ".elk.json", ".elk")
def elk(
    graph: BaseGraph,
    *,
    size: tuple[float, float] = (60, 40),
    layout_options: dict[str, str] | None = None,
) -> str:
    """
    Eclipse Layout Kernel (ELK) JSON graph text

    :param size: width and height given to every node, which ELK requires
    :param layout_options: ELK layout options, merged over the defaults

    Unlike GraphML, ELK has native ``x`` and ``y`` node fields.  If any vertex
    has a coordinate the ``fixed`` algorithm is selected so that a consumer
    keeps the positions; otherwise ``layered`` computes a layout.
    """
    graph.nc  # refresh vertex.label
    has_coords = any(v.coord is not None for v in graph)

    ids: dict[BaseVertex, str] = {}
    children = []
    for i, vertex in enumerate(graph):
        ids[vertex] = f"n{i}"
        node: dict[str, Any] = {
            "id": f"n{i}",
            "labels": [{"text": str(vertex.name)}],
            "width": size[0],
            "height": size[1],
        }
        properties: dict[str, Any] = {}
        if vertex.coord is not None:
            if len(vertex.coord) >= 2:
                node["x"] = float(vertex.coord[0])
                node["y"] = float(vertex.coord[1])
            properties["coord"] = [float(c) for c in vertex.coord]
        if vertex.label is not None:
            properties["component"] = vertex.label
        if properties:
            node["properties"] = properties
        children.append(node)

    edges = []
    for i, (e, v1, v2) in enumerate(_ordered_edges(graph)):
        edge: dict[str, Any] = {
            "id": f"e{i}",
            "sources": [ids[v1]],
            "targets": [ids[v2]],
        }
        if e.cost is not None:
            edge["labels"] = [{"text": f"{e.cost:.4g}"}]
            edge["properties"] = {"cost": e.cost}
        edges.append(edge)

    options = {
        "org.eclipse.elk.algorithm": "fixed" if has_coords else "layered",
        "org.eclipse.elk.direction": "DOWN" if _is_directed(graph) else "UNDEFINED",
    }
    options.update(layout_options or {})
    return (
        json.dumps(
            {
                "id": "root",
                "layoutOptions": options,
                "children": children,
                "edges": edges,
            },
            indent=2,
        )
        + "\n"
    )


# --------------------------------------------------------------------------- #
# rendered images


def render(
    graph: BaseGraph,
    fmt: str,
    *,
    engine: str | None = None,
    direction: str | None = None,
    colorcomponents: bool = False,
    costs: bool = True,
) -> bytes:
    """
    Render the graph to an image with the GraphViz executables

    :param fmt: one of :data:`RENDER_FORMATS`
    :param engine: GraphViz layout program, one of :data:`ENGINES`.  Defaults
        to ``neato`` if any vertex has a coordinate, since it honours pinned
        positions, else ``dot``
    :return: the image
    :rtype: bytes
    :raises ValueError: ``fmt`` or ``engine`` is not a known name
    :raises RuntimeError: the layout program is not installed, or failed

    Both names are checked against fixed lists and the program is run with an
    argument list and no shell, so neither can carry a command.
    """
    if fmt not in RENDER_FORMATS:
        raise ValueError(
            f"unknown image format {fmt!r}, expected one of {sorted(RENDER_FORMATS)}"
        )
    if engine is None:
        engine = "neato" if any(v.coord is not None for v in graph) else "dot"
    if engine not in ENGINES:
        raise ValueError(f"unknown GraphViz engine {engine!r}, expected one of {ENGINES}")
    exe = shutil.which(engine)
    if exe is None:
        raise RuntimeError(
            f"GraphViz layout program {engine!r} was not found on PATH; "
            "install GraphViz (https://graphviz.org/download/) to render "
            f"{fmt} output, or export the text with format='dot'"
        )

    text = dot(graph, direction=direction, colorcomponents=colorcomponents, costs=costs)
    result = subprocess.run(  # nosec B603 nosemgrep - allow-listed program and format, no shell
        [exe, f"-T{fmt}"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{engine} failed with exit code {result.returncode}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.stdout
