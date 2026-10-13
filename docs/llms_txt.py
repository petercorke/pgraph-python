"""Write llms.txt and llms-full.txt for the documentation site.

``llms.txt`` (see https://llmstxt.org) is a short Markdown file that tells AI assistants
what the package is, how to use it, and where its documentation lives. ``llms-full.txt``
is the whole documentation as plain text, so an assistant can read it in a single fetch.

The package is small, so ``llms.txt`` is mostly prose: the hand-written ``llms_intro.md``
(what it does, a quick start, key points, common mistakes) followed by a generated list
of links. ``llms-full.txt`` is this Sphinx site rendered by Sphinx's ``text`` builder, so
the autodoc API reference is included, in the order of the toctrees, which are followed
from ``index.rst`` through the nested pages. Run after the HTML build::

    make html llms          # from docs/

The files are written into ``build/html`` so they are published with the site.
"""

from __future__ import annotations

import os
import re
import shutil

# Used only to re-invoke `sphinx-build`, resolved via shutil.which() below; this is a
# local/CI docs-build script, not code that runs against untrusted input.
import subprocess  # nosec B404
import sys
from importlib.metadata import metadata
from pathlib import Path

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "source"
BUILD = DOCS / "build"
TEXT = BUILD / "text"
HTML = BUILD / "html"
INTRO = DOCS / "llms_intro.md"

# One-line descriptions of the pages listed in the toctrees
PAGE_NOTES = {
    "index": "overview",
    "intro": "tutorial: building a graph, vertices, edges, plotting, path planning, directed graphs",
    "api": "the API, in three parts: the common methods, vertices and edges, and the policy page",
    "common": "the methods that UGraph and DGraph share, by task",
    "construction": "UGraph, DGraph, Graph, constructors from a dict or an adjacency matrix, distance metrics",
    "modification": "adding and removing vertices and edges",
    "analysis": "counts, adjacency, degree, components, coordinates, spatial queries, subgraphs",
    "representations": "adjacency, incidence, degree and Laplacian matrices",
    "visualization": "plot(), its options, highlighting vertices, edges and paths",
    "export": "writing a graph as GraphViz dot, Mermaid, GraphML, ELK or an image",
    "planning": "path_BFS, path_UCS and path_Astar",
    "edges_vertices": "the data structures: Vertex and Edge, and the edge lists of a vertex",
    "policy": "object lifecycle: what may be done with a vertex or edge outside a graph",
}

# Links that belong in llms.txt but are not pages of this site
EXTRA_LINKS = [
    ("Source code", "https://github.com/petercorke/pgraph-python", "GitHub repository"),
    ("PyPI", "https://pypi.org/project/pgraph-python/", "install with pip install pgraph-python"),
    ("Changelog", "https://github.com/petercorke/pgraph-python/blob/main/CHANGELOG.md", "release history and how to migrate from 0.6"),
    ("Examples", "https://github.com/petercorke/pgraph-python/tree/main/examples", "Queensland and central London route-planning notebooks, and a benchmark"),
    ("RVC ecosystem", "https://github.com/petercorke/rvc-ecosystem", "how pgraph relates to the Robotics and Machine Vision Toolboxes"),
    ("Contributing", "https://github.com/petercorke/pgraph-python/blob/main/AGENTS.md", "repo conventions and workflow, for agents working on pgraph's source code"),
]


def toctree_entries(document: Path) -> list[str]:
    """Document names listed in the toctrees of one document, in order.

    :param document: an ``.rst`` file
    :return: document names such as ``"common"``
    """
    entries, in_toctree = [], False
    for line in document.read_text().splitlines():
        if line.startswith(".. toctree::"):
            in_toctree = True
        elif in_toctree and line.strip() and not line.startswith((" ", "\t")):
            in_toctree = False  # the directive's indented body has ended
        elif in_toctree and line.strip() and not line.strip().startswith(":"):
            entries.append(line.strip())
    return entries


def toctree_pages(index: Path) -> list[str]:
    """All the documents below a master document, depth first, in toctree order.

    The pages of this site are nested: ``index`` lists ``api``, which lists
    ``common``, which lists the pages of the common methods.

    :param index: the master document, ``index.rst``
    :return: document names such as ``"construction"``, each once
    """
    pages: list[str] = []

    def walk(document: Path) -> None:
        for name in toctree_entries(document):
            if name not in pages:
                pages.append(name)
                child = document.parent / f"{name}.rst"
                if child.exists():
                    walk(child)

    walk(index)
    return pages


def title(text: str) -> str:
    """The page title: the first non-blank line that isn't an overline of punctuation."""
    for line in text.splitlines():
        if line.strip() and not re.fullmatch(r"[*=\-~#^]+", line.strip()):
            return line.strip()
    return ""


def main() -> None:
    # Read the name, summary and documentation URL from the installed package's metadata
    # (built from pyproject.toml). Reading pyproject.toml directly would need tomllib,
    # which is Python 3.11+, and the Pages workflow builds the docs on 3.9.
    meta = metadata("pgraph-python")
    urls = dict(u.split(", ", 1) for u in meta.get_all("Project-URL"))
    base = urls["Documentation"].rstrip("/") + "/"
    name, summary = meta["Name"], meta["Summary"]

    # SPHINXBUILD matches the override convention of docs/Makefile's
    # `SPHINXBUILD ?= sphinx-build` -- resolved via shutil.which() to a full path.
    sphinxbuild_name = os.environ.get("SPHINXBUILD", "sphinx-build")
    sphinxbuild = shutil.which(sphinxbuild_name)
    if sphinxbuild is None:
        raise SystemExit(f"sphinx-build not found: {sphinxbuild_name!r}")
    subprocess.run(  # nosec B603 -- sphinxbuild resolved via shutil.which() above  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        [sphinxbuild, "-q", "-b", "text", str(SOURCE), str(TEXT)],  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
        check=True,
    )
    pages = ["index"] + toctree_pages(SOURCE / "index.rst")
    texts = {p: (TEXT / f"{p}.txt").read_text() for p in pages}

    # llms.txt: prose introduction, then links
    index = [
        f"# {name}",
        "",
        f"> {summary}.",
        "",
        INTRO.read_text().strip(),
        "",
        "## Documentation",
        "",
        f"The whole documentation, including the API reference, is available as a single text file: [llms-full.txt]({base}llms-full.txt).",
        "",
        *(f"- [{title(texts[p])}]({base}{p}.html): {PAGE_NOTES.get(p, '')}" for p in pages),
        "",
        "## Optional",
        "",
        *(f"- [{label}]({url}): {note}" for label, url, note in EXTRA_LINKS),
        "",
    ]
    HTML.mkdir(parents=True, exist_ok=True)
    (HTML / "llms.txt").write_text("\n".join(index))

    # llms-full.txt: the prose introduction, then every page
    def block(url: str, body: str) -> str:
        return f"\n{'=' * 72}\nPage: {url}\n{'=' * 72}\n\n{body.strip()}\n"

    parts = [f"# {name}: {summary}\n\nDocumentation: {base}\n\n{INTRO.read_text().strip()}\n"]
    parts += [block(f"{base}{p}.html", texts[p]) for p in pages]
    full = re.sub(r"\n{3,}", "\n\n", "".join(parts))  # the text builder leaves runs of blank lines
    (HTML / "llms-full.txt").write_text(full)
    print(f"llms.txt: {len(pages)} pages; llms-full.txt: {len(full) // 1000} kB", file=sys.stderr)


if __name__ == "__main__":
    main()
