"""
Importing pgraph does not load matplotlib, spatialmath or scipy: they are
imported by the methods that plot (matplotlib, spatialmath) or build a spatial
index (scipy), so a script that only builds graphs and plans paths does not pay
for them.

Each check runs a new interpreter, since the modules of this one are loaded.
"""

import json
import subprocess  # nosec B404 nosemgrep - runs this interpreter on a fixed snippet
import sys
import unittest


def loaded(code: str, *modules: str) -> dict[str, bool]:
    """Run ``code``, then say which of ``modules`` are in sys.modules"""
    probe = (
        code
        + "\nimport sys, json\n"
        + f"print(json.dumps({{m: m in sys.modules for m in {list(modules)!r}}}))"
    )
    out = subprocess.run(  # nosec B603 nosemgrep
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


HEAVY = ("matplotlib", "matplotlib.pyplot", "mpl_toolkits", "spatialmath", "scipy")


class TestLazyImports(unittest.TestCase):
    def test_import_pgraph(self):
        got = loaded("import pgraph", *HEAVY)
        self.assertEqual({m for m, v in got.items() if v}, set())

    def test_building_and_planning_does_not_import_them(self):
        got = loaded(
            """
import pgraph
g = pgraph.DGraph(dim=2)
a = g.add_vertex([0, 0], name="a")
b = g.add_vertex([3, 4], name="b")
g.add_edge(a, b)
g.path_Astar("a", "b"); g.path_BFS("a", "b"); g.path_UCS("a", "b")
g.closest([1, 1]); g.within([[0, 0], [5, 5]]); g.near([0, 0], 10)
g.subgraph([a, b]).copy(); g.adjacency(); g.laplacian(); g.incidence()
g.export(format="dot")
import pickle; pickle.loads(pickle.dumps(g))
""",
            *HEAVY,
        )
        self.assertEqual({m for m, v in got.items() if v}, set())

    def test_plotting_imports_matplotlib(self):
        got = loaded(
            """
import matplotlib
matplotlib.use("Agg")
import pgraph
g = pgraph.UGraph(dim=2)
a = g.add_vertex([0, 0]); b = g.add_vertex([1, 1]); g.add_edge(a, b)
g.plot(block=None)
""",
            "matplotlib.pyplot",
        )
        self.assertTrue(got["matplotlib.pyplot"])

    def test_plotting_works_when_matplotlib_is_imported_late(self):
        # the whole point: the import happens inside plot(), not before
        got = loaded(
            """
import pgraph
import sys
assert "matplotlib" not in sys.modules
g = pgraph.DGraph(dim=3)
a = g.add_vertex([0, 0, 0]); b = g.add_vertex([1, 1, 1]); g.add_edge(a, b); g.add_edge(b, a); g.add_edge(a, a)
import matplotlib
matplotlib.use("Agg")
g.plot(block=None)
g.highlight_path([a, b], block=None)
""",
            "matplotlib",
        )
        self.assertTrue(got["matplotlib"])


if __name__ == "__main__":
    unittest.main()
