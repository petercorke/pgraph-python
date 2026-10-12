"""Fetch a road network from OpenStreetMap, for the ``london.ipynb`` example.

Everything that is about the OpenStreetMap service rather than about graphs
lives here, so that the notebook can concentrate on pgraph:

* one request to the Overpass API, with the ``User-Agent`` that the service
  insists upon and a few retries when it is busy,
* the choice of which ``highway`` classes a car may use, and which ways are
  closed to it,
* the rules for which direction a way may be driven in,
* a compact JSON file as a cache, so the data is fetched once and the notebook
  then runs offline.

Standard library only.

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
from typing import Any

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


def fetch_roads(
    bbox: tuple[float, float, float, float],
    cache: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Roads that a car can use inside a bounding box

    :param bbox: ``(south, west, north, east)`` in degrees
    :param cache: JSON file that holds the result, read if it exists
    :param refresh: fetch again even if ``cache`` exists
    :return: ``{"bbox": ..., "nodes": {id: [lat, lon]}, "ways": [...]}`` where
        each way is ``[name, direction, [node ids]]`` and direction is 1 if the
        way is one-way in the order of its nodes, -1 if against it, 0 if both
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
        "nodes": {str(i): coords[i] for i in sorted(used)},
        "ways": ways,
    }
    cache.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    return data
