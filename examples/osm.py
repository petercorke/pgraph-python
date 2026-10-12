"""Fetch a road network from OpenStreetMap, for the ``london.ipynb`` example.

Everything that is about the OpenStreetMap service rather than about graphs
lives here, so that the notebook can concentrate on pgraph:

* one request to the Overpass API, with the ``User-Agent`` that the service
  insists upon and a few retries when it is busy,
* the choice of which ``highway`` classes a car may use, and which ways are
  closed to it,
* the rules for which direction a way may be driven in,
* a projection from latitude and longitude to metres,
* a compact JSON file as a cache, so the data is fetched once and the notebook
  then runs offline.

The standard library and numpy only.

The data are © OpenStreetMap contributors, available under the Open Database
License (https://www.openstreetmap.org/copyright).  An extract saved by this
module is a derived database, so keep this notice with it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "pgraph-python-example/1.0 (https://github.com/petercorke/pgraph-python)"

# the classes of road that a car can use
DRIVABLE = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street"
    "|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
)
CLOSED = {"no", "private"}  # values of access tags that exclude a car


def _direction(tags: dict[str, str]) -> int:
    """Which way may a car drive along this way?

    :return: 1 forward (in the order of the nodes), -1 backward, 0 both
    """
    oneway = tags.get("oneway")
    if oneway in ("yes", "true", "1"):
        return 1
    if oneway in ("-1", "reverse"):
        return -1
    if oneway == "no":
        return 0
    # no explicit tag: some things imply a direction
    if tags.get("junction") in ("roundabout", "circular"):
        return 1
    if tags.get("highway") in ("motorway", "motorway_link"):
        return 1
    return 0


def _open_to_cars(tags: dict[str, str]) -> bool:
    return not (
        tags.get("access") in CLOSED
        or tags.get("motor_vehicle") in CLOSED
        or tags.get("vehicle") in CLOSED
    )


def _query(bbox: tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    return (
        "[out:json][timeout:90];"
        f'way["highway"~"^({DRIVABLE})$"]({south},{west},{north},{east});'
        "(._;>;);out body;"
    )


def _post(query: str, retries: int = 4) -> dict[str, Any]:
    body = urllib.parse.urlencode({"data": query}).encode()
    request = urllib.request.Request(
        OVERPASS, data=body, headers={"User-Agent": USER_AGENT, "Accept": "*/*"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            # 429 too many requests, 504 gateway timeout: the service is busy
            if err.code not in (429, 502, 503, 504) or attempt == retries - 1:
                raise
        time.sleep(5 * 2**attempt)
    raise RuntimeError("unreachable")


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _to_lines(data: dict[str, Any]) -> str:
    """JSON with one record per line, for a file that is read by people and diffs

    A dict of records is written a key per line, and a list of lists an item
    per line, so that a change to the data is a change to a few lines, and a
    record can be found with ``grep``.  It is about 10% bigger than the
    compact form and the same JSON.
    """
    lines = ["{"]
    keys = list(data)
    for k, key in enumerate(keys):
        value = data[key]
        tail = "," if k < len(keys) - 1 else ""
        label = json.dumps(key)
        if isinstance(value, dict) and value:
            lines.append(f" {label}: {{")
            items = list(value.items())
            for i, (name, record) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""
                lines.append(f"  {json.dumps(name)}: {_compact(record)}{comma}")
            lines.append(" }" + tail)
        elif isinstance(value, list) and value and all(isinstance(r, list) for r in value):
            lines.append(f" {label}: [")
            for i, record in enumerate(value):
                comma = "," if i < len(value) - 1 else ""
                lines.append(f"  {_compact(record)}{comma}")
            lines.append(" ]" + tail)
        else:
            lines.append(f" {label}: {_compact(value)}{tail}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def fetch_roads(
    bbox: tuple[float, float, float, float],
    cache: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Roads that a car can use inside a bounding box

    :param bbox: ``(south, west, north, east)`` in degrees
    :param cache: JSON file that holds the result, read if it exists.  It is
        written with one record (node, way) per line
    :param refresh: fetch again even if ``cache`` exists
    :return: ``{"bbox": ..., "ways": [...], "nodes": {id: [lat, lon]}}`` where
        each way is ``[name, direction, [node ids]]`` and direction is 1 if the
        way is one-way in the order of its nodes, -1 if against it, 0 if both.
        The ways come first in the file, since they are what a reader looks at
    """
    cache = Path(cache)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())

    elements = _post(_query(bbox))["elements"]
    coords = {e["id"]: [round(e["lat"], 6), round(e["lon"], 6)] for e in elements if e["type"] == "node"}

    ways, used = [], set()
    for e in elements:
        if e["type"] != "way" or not _open_to_cars(e.get("tags", {})):
            continue
        tags = e.get("tags", {})
        ways.append([tags.get("name", ""), _direction(tags), e["nodes"]])
        used.update(e["nodes"])

    data = {
        "source": "OpenStreetMap, © OpenStreetMap contributors, ODbL "
        "https://www.openstreetmap.org/copyright",
        "bbox": list(bbox),
        "ways": ways,
        "nodes": {str(i): coords[i] for i in sorted(used)},
    }
    cache.write_text(_to_lines(data), encoding="utf-8")
    return data


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------

_A = 6378137.0  # WGS84 semi-major axis, m
_F = 1 / 298.257223563  # WGS84 flattening


def local_projection(
    lat0: float, lon0: float
) -> Callable[[ArrayLike, ArrayLike], NDArray]:
    r"""A map projection to metres, centred on a point

    :param lat0: latitude of the origin, degrees
    :param lon0: longitude of the origin, degrees, which is also the central
        meridian
    :return: a function ``project(lat, lon)`` that returns metres east and
        north of the origin, as an array with a trailing dimension of 2

    This is a function that makes a function, so the constants of the
    projection are worked out once.  The projection is the transverse Mercator
    on the WGS84 ellipsoid, with a scale of 1 on the central meridian, by the
    series in the third flattening :math:`n` of Krüger, to fourth order
    (Karney, "Transverse Mercator with an accuracy of a few nanometers", 2011).
    Its error is well below a millimetre, and the scale is within 1e-6 of 1,
    for a few tens of kilometres around the origin, so that distances in it are
    distances on the ground.

    Some people use the UTM grid instead.  It is the same projection, but with
    the central meridian at the centre of a six degree zone and a scale of
    0.9996 there, so distances in London are wrong by about 1 part in 10,000,
    which is half a metre over 5 km.
    """
    n = _F / (2 - _F)
    n2, n3, n4 = n**2, n**3, n**4
    rect = _A / (1 + n) * (1 + n2 / 4 + n4 / 64)  # rectifying radius
    alpha = (
        n / 2 - 2 * n2 / 3 + 5 * n3 / 16 + 41 * n4 / 180,
        13 * n2 / 48 - 3 * n3 / 5 + 557 * n4 / 1440,
        61 * n3 / 240 - 103 * n4 / 140,
        49561 * n4 / 161280,
    )
    e = 2 * np.sqrt(n) / (1 + n)  # the first eccentricity-like term
    lam0 = np.radians(lon0)

    def conformal(phi: NDArray) -> NDArray:
        """Tangent of the conformal latitude, as a sinh"""
        s = np.sin(phi)
        return np.sinh(np.arctanh(s) - e * np.arctanh(e * s))

    def xi_eta(lat: ArrayLike, lon: ArrayLike) -> tuple[NDArray, NDArray]:
        phi = np.radians(np.asarray(lat, dtype=float))
        dlam = np.radians(np.asarray(lon, dtype=float)) - lam0
        t = conformal(phi)
        xi = np.arctan2(t, np.cos(dlam))
        eta = np.arctanh(np.sin(dlam) / np.sqrt(1 + t**2))
        xi_s, eta_s = xi.copy(), eta.copy()
        for j, a in enumerate(alpha, start=1):
            xi_s = xi_s + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
            eta_s = eta_s + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        return xi_s, eta_s

    xi0 = float(xi_eta(lat0, lon0)[0])

    def project(lat: ArrayLike, lon: ArrayLike) -> NDArray:
        xi, eta = xi_eta(lat, lon)
        return np.stack([rect * eta, rect * (xi - xi0)], axis=-1)

    return project
