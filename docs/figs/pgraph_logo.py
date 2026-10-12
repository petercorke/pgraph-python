"""
Generate the pgraph logo.

A snake follows a path through a small graph, its head acting as the arrowhead
pointing at the goal vertex, beside the "pgraph" wordmark. The style matches
the snake logos used across the RVC toolboxes (RTB, MVTB, bdsim, SMTB): a
grey-to-teal body, yellow eyes and a magenta forked tongue.

Renders ``pgraph_logo.png`` (2x scale, white background) next to this
script, using ``rsvg-convert`` (from librsvg) to rasterize the SVG.

The wordmark is SVG text in Roboto Medium, so Roboto must be installed for
the PNG to render correctly. The PNG is the committed artifact used by the
README.

Usage::

    python docs/figs/pgraph_logo.py
"""

import math
import shutil
import subprocess
from pathlib import Path

# palette
TEAL = "#2bb39a"  # snake head, end of body gradient
GREY_TAIL = "#5c5c5c"  # start of body gradient
GREY_TEXT = "#4d4d4d"
GREY_EDGE = "#9a9a9a"
GREY_NODE = "#555555"
EYE = "#ffee00"
TONGUE = "#d14fc4"

WIDTH, HEIGHT = 580, 210
NODE_RADIUS = 11
BODY_WIDTH = 14
HEAD_SCALE = 1.3

# graph: vertex coordinates, edges, and the path the snake follows
VERTICES: dict[int, tuple[float, float]] = {
    1: (35, 165),
    2: (85, 55),
    3: (125, 150),
    4: (180, 90),
    5: (225, 175),
    6: (255, 40),
}
EDGES: list[tuple[int, int]] = [
    (1, 2), (1, 3), (2, 3), (2, 4), (3, 4),
    (3, 5), (4, 5), (4, 6), (2, 6), (5, 6),
]  # fmt: skip
PATH: list[int] = [1, 3, 4, 6]


def snake_head(x: float, y: float, angle: float, scale: float = 1.0) -> str:
    """
    Snake head as an SVG group.

    :param x: x-coordinate of the neck
    :param y: y-coordinate of the neck
    :param angle: heading of the head, in radians
    :param scale: scale factor applied to the head
    :returns: SVG ``<g>`` element

    The head is defined with its neck at the origin and nose pointing along +x,
    then translated and rotated into place.
    """
    return f"""<g transform="translate({x:.1f},{y:.1f}) rotate({math.degrees(angle):.1f}) scale({scale})">
  <path d="M 22 0 l 9 0 l 5 -4 m -5 4 l 5 4" stroke="{TONGUE}" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <path d="M -4 -7 C 6 -12, 18 -11, 23 -3 C 25 0, 25 0, 23 3 C 18 11, 6 12, -4 7 Z" fill="{TEAL}"/>
  <circle cx="13" cy="-4.5" r="3.2" fill="{EYE}"/>
  <circle cx="13" cy="4.5" r="3.2" fill="{EYE}"/>
</g>"""


def logo_svg(font_weight: int = 500) -> str:
    """
    Build the logo as an SVG document.

    :param font_weight: CSS font weight of the "pgraph" wordmark
    :returns: SVG document
    """
    pts = [VERTICES[v] for v in PATH]
    (x0, y0), (xN, yN) = pts[0], pts[-1]

    # body gradient runs from the tail to near the neck, so the neck
    # blends into the solid-colored head
    gx = x0 + 0.8 * (xN - x0)
    gy = y0 + 0.8 * (yN - y0)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<defs><linearGradient id="body" gradientUnits="userSpaceOnUse" x1="{x0}" y1="{y0}" x2="{gx:.1f}" y2="{gy:.1f}">'
        f'<stop offset="0" stop-color="{GREY_TAIL}"/><stop offset="1" stop-color="{TEAL}"/>'
        "</linearGradient></defs>",
    ]

    # graph edges not on the path, drawn thin
    on_path = {frozenset(e) for e in zip(PATH, PATH[1:])}
    for a, b in EDGES:
        if frozenset((a, b)) in on_path:
            continue
        (xa, ya), (xb, yb) = VERTICES[a], VERTICES[b]
        out.append(
            f'<line x1="{xa}" y1="{ya}" x2="{xb}" y2="{yb}" '
            f'stroke="{GREY_EDGE}" stroke-width="3" stroke-linecap="round"/>'
        )

    # the body stops short of the goal vertex, leaving room for head and tongue
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    heading = math.atan2(yb - ya, xb - xa)
    tongue_len, head_len = 15, 30
    back = NODE_RADIUS + tongue_len + head_len + 2
    hx, hy = xb - back * math.cos(heading), yb - back * math.sin(heading)

    # thin tail poking out behind the start vertex
    x1, y1 = pts[1]
    t = math.atan2(y0 - y1, x0 - x1)
    tx, ty = x0 + 22 * math.cos(t), y0 + 22 * math.sin(t)
    out.append(
        f'<path d="M {tx:.1f} {ty:.1f} L {x0} {y0}" stroke="url(#body)" '
        'stroke-width="6" stroke-linecap="round"/>'
    )

    body = f"M {x0} {y0} " + " ".join(f"L {x} {y}" for x, y in pts[1:-1])
    body += f" L {hx:.1f} {hy:.1f}"
    out.append(
        f'<path d="{body}" stroke="url(#body)" stroke-width="{BODY_WIDTH}" '
        'fill="none" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    out.append(snake_head(hx, hy, heading, HEAD_SCALE))

    # vertices drawn last, over the edges and the snake's body
    for x, y in VERTICES.values():
        out.append(
            f'<circle cx="{x}" cy="{y}" r="{NODE_RADIUS}" fill="white" '
            f'stroke="{GREY_NODE}" stroke-width="3"/>'
        )

    out.append(
        f'<text x="290" y="140" font-family="Roboto" font-weight="{font_weight}" '
        f'font-size="84" fill="{GREY_TEXT}">pgraph</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    if shutil.which("rsvg-convert") is None:
        raise SystemExit("rsvg-convert not found (brew install librsvg)")
    png = Path(__file__).parent / "pgraph_logo.png"
    subprocess.run(
        ["rsvg-convert", "-z", "2", "-b", "white", "-o", str(png)],
        input=logo_svg().encode(),
        check=True,
    )
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
