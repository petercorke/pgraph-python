"""
Graph export: DOT, Mermaid, GraphML, ELK JSON, and rendered images.

Real I/O boundary: the image formats run the GraphViz executables.  Those tests
use the real ``dot``/``neato`` and are skipped when GraphViz is not installed;
the error path for a missing program is tested without needing it.
"""

import io
import json
import pathlib
import shutil
import tempfile
import unittest
import unittest.mock
import xml.etree.ElementTree as ET  # nosec B405 nosemgrep - parses our own generated output


from pgraph import DGraph, UGraph, exporters

HAVE_DOT = shutil.which("dot") is not None
NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


def line_graph(Graph):
    """a(0,0) -- b(1,0) -- c(2,0) with costs 1 and 2"""
    g = Graph()
    a = g.add_vertex([0, 0], name="a")
    b = g.add_vertex([1, 0], name="b")
    c = g.add_vertex([2, 0], name="c")
    a.connect(b, cost=1)
    b.connect(c, cost=2)
    return g, a, b, c


class TestRegistry(unittest.TestCase):
    def test_formats(self):
        names = exporters.formats()
        for name in ("dot", "mermaid", "graphml", "elk", "svg", "pdf", "png"):
            self.assertIn(name, names)

    def test_format_from_filename(self):
        f = exporters.format_from_filename
        self.assertEqual(f("a.dot"), "dot")
        self.assertEqual(f("a.GV"), "dot")
        self.assertEqual(f("a.mmd"), "mermaid")
        self.assertEqual(f("a.graphml"), "graphml")
        self.assertEqual(f("a.elk.json"), "elk")
        self.assertEqual(f(pathlib.Path("x/y.svg")), "svg")
        self.assertEqual(f("a.pdf"), "pdf")
        self.assertIsNone(f("a.unknown"))
        self.assertIsNone(f("noextension"))
        self.assertIsNone(f(None))
        self.assertIsNone(f(io.StringIO()))

    def test_file_object_with_name(self):
        f = io.StringIO()
        f.name = "out.mmd"
        self.assertEqual(exporters.format_from_filename(f), "mermaid")

    def test_user_registered_exporter(self):
        @exporters.register_exporter("names", ".names")
        def names(graph, *, sep=",") -> str:
            return sep.join(v.name for v in graph)

        try:
            g, *_ = line_graph(UGraph)
            self.assertEqual(g.export(format="names"), "a,b,c")
            self.assertEqual(g.export(format="names", sep="|"), "a|b|c")
            self.assertEqual(exporters.format_from_filename("x.names"), "names")
        finally:
            exporters._TEXT.pop("names", None)
            exporters._EXTENSIONS.pop(".names", None)


class TestExportDispatch(unittest.TestCase):
    def setUp(self):
        self.g, *_ = line_graph(UGraph)

    def test_returns_text_without_filename(self):
        text = self.g.export(format="dot")
        self.assertIsInstance(text, str)
        self.assertTrue(text.startswith("graph {"))

    def test_format_is_case_insensitive(self):
        self.assertEqual(self.g.export(format="DOT"), self.g.export(format="dot"))

    def test_cannot_infer(self):
        with self.assertRaises(ValueError) as cm:
            self.g.export()
        self.assertIn("format", str(cm.exception))
        with self.assertRaises(ValueError):
            self.g.export("noextension")

    def test_unknown_format(self):
        with self.assertRaises(ValueError):
            self.g.export(format="bogus")

    def test_unknown_option(self):
        with self.assertRaises(TypeError):
            self.g.export(format="dot", nonsense=1)
        with self.assertRaises(TypeError):
            self.g.export(format="graphml", direction="LR")

    def test_image_needs_filename(self):
        with self.assertRaises(ValueError):
            self.g.export(format="svg")

    def test_path_written_and_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "g.mmd"
            self.assertIsNone(self.g.export(p))
            self.assertEqual(p.read_text(), self.g.export(format="mermaid"))
            q = pathlib.Path(d) / "g.txt"
            self.assertIsNone(self.g.export(str(q), format="dot"))
            self.assertEqual(q.read_text(), self.g.export(format="dot"))

    def test_file_object_not_closed(self):
        f = io.StringIO()
        self.assertIsNone(self.g.export(f, format="graphml"))
        self.assertFalse(f.closed)
        self.assertIn("<graphml", f.getvalue())


class TestDot(unittest.TestCase):
    def test_undirected(self):
        g, *_ = line_graph(UGraph)
        self.assertEqual(
            g.export(format="dot"),
            'graph {\n'
            '  "a" [pos="0,0!"]\n'
            '  "b" [pos="1,0!"]\n'
            '  "c" [pos="2,0!"]\n'
            '  "a" -- "b" [label="1"]\n'
            '  "b" -- "c" [label="2"]\n'
            "}\n",
        )

    def test_directed(self):
        g, *_ = line_graph(DGraph)
        text = g.export(format="dot")
        self.assertTrue(text.startswith("digraph {"))
        self.assertIn('"a" -> "b"', text)
        self.assertNotIn(" -- ", text)

    def test_options(self):
        g, *_ = line_graph(UGraph)
        text = g.export(format="dot", direction="LR", costs=False)
        self.assertIn('rankdir="LR"', text)
        self.assertNotIn("label=", text)

    def test_no_coordinates_no_pos(self):
        g = UGraph()
        g.add_edge(g.add_vertex(name="p"), g.add_vertex(name="q"))
        text = g.export(format="dot")
        self.assertNotIn("pos=", text)
        self.assertIn('"p" -- "q"', text)

    def test_three_dimensional_uses_first_two(self):
        g = UGraph()
        g.add_vertex([1, 2, 3], name="p")
        self.assertIn('pos="1,2!"', g.export(format="dot"))

    def test_escaping(self):
        g = UGraph()
        a = g.add_vertex(name='say "hi"')
        b = g.add_vertex(name="back\\slash\nnewline")
        g.add_edge(a, b)
        text = g.export(format="dot")
        self.assertIn('"say \\"hi\\""', text)
        self.assertIn('"back\\\\slash\\nnewline"', text)
        # every line is balanced: no raw newline inside a quoted name
        self.assertEqual(len(text.strip().splitlines()), 5)

    def test_color_components(self):
        g = UGraph()
        a, b, c = (g.add_vertex(name=n) for n in "abc")
        g.add_edge(a, b)  # c is a second component
        text = g.export(format="dot", color_components=True)
        colors = {
            line.split('fillcolor="')[1].split('"')[0]
            for line in text.splitlines()
            if "fillcolor" in line
        }
        self.assertEqual(len(colors), 2)


class TestMermaid(unittest.TestCase):
    def test_undirected(self):
        g, *_ = line_graph(UGraph)
        self.assertEqual(
            g.export(format="mermaid"),
            "flowchart TB\n"
            '    n0["a"]\n'
            '    n1["b"]\n'
            '    n2["c"]\n'
            '    n0 ---|"1"| n1\n'
            '    n1 ---|"2"| n2\n',
        )

    def test_directed_arrow_and_options(self):
        g, *_ = line_graph(DGraph)
        text = g.export(format="mermaid", direction="LR", costs=False)
        self.assertTrue(text.startswith("flowchart LR"))
        self.assertIn("n0 --> n1", text)
        self.assertNotIn("|", text)

    def test_fence(self):
        g, *_ = line_graph(UGraph)
        text = g.export(format="mermaid", fence=True)
        self.assertTrue(text.startswith("```mermaid\nflowchart"))
        self.assertTrue(text.endswith("```\n"))

    def test_bad_direction(self):
        g, *_ = line_graph(UGraph)
        with self.assertRaises(ValueError):
            g.export(format="mermaid", direction="sideways")

    def test_awkward_names_are_safe(self):
        g = UGraph()
        g.add_edge(g.add_vertex(name='a "b" c'), g.add_vertex(name="x y-z]"))
        text = g.export(format="mermaid")
        self.assertIn('n0["a #quot;b#quot; c"]', text)
        self.assertIn('n1["x y-z]"]', text)
        self.assertIn("n0 --- n1", text)

    def test_color_components(self):
        g = UGraph()
        g.add_vertex(name="a")
        g.add_vertex(name="b")
        text = g.export(format="mermaid", color_components=True)
        self.assertEqual(sum(1 for l in text.splitlines() if "fill:#" in l), 2)


class TestGraphML(unittest.TestCase):
    def parse(self, g):
        return ET.fromstring(g.export(format="graphml"))  # nosec B314

    def test_structure(self):
        g, *_ = line_graph(UGraph)
        root = self.parse(g)
        graph = root.find("g:graph", NS)
        self.assertEqual(graph.get("edgedefault"), "undirected")
        self.assertEqual(len(graph.findall("g:node", NS)), 3)
        edges = graph.findall("g:edge", NS)
        self.assertEqual(len(edges), 2)
        self.assertEqual([(e.get("source"), e.get("target")) for e in edges],
                         [("n0", "n1"), ("n1", "n2")])
        costs = [e.find("g:data", NS).text for e in edges]
        self.assertEqual(costs, ["1", "2"])

    def test_directed(self):
        g, *_ = line_graph(DGraph)
        graph = self.parse(g).find("g:graph", NS)
        self.assertEqual(graph.get("edgedefault"), "directed")

    def test_node_data(self):
        g, *_ = line_graph(UGraph)
        node = self.parse(g).find("g:graph/g:node[@id='n1']", NS)
        data = {d.get("key"): d.text for d in node.findall("g:data", NS)}
        self.assertEqual(data["node_label"], "b")
        self.assertEqual((data["x"], data["y"]), ("1", "0"))
        self.assertEqual(data["coord"], "1,0")
        self.assertEqual(data["component"], "0")

    def test_special_characters_roundtrip(self):
        g = UGraph()
        g.add_vertex(name="<&>\"'")
        node = self.parse(g).find("g:graph/g:node", NS)
        self.assertEqual(node.find("g:data", NS).text, "<&>\"'")

    def test_unrepresentable_control_characters(self):
        # XML 1.0 cannot hold most control characters even escaped; the output
        # must still be well formed, with the character replaced
        g = UGraph()
        g.add_vertex(name="a\x00b\x1fc\td")
        node = self.parse(g).find("g:graph/g:node", NS)
        self.assertEqual(node.find("g:data", NS).text, "a\ufffdb\ufffdc\td")

    def test_no_cost_key_when_no_costs(self):
        g = UGraph()
        g.add_edge(g.add_vertex(name="a"), g.add_vertex(name="b"))
        text = g.export(format="graphml")
        self.assertNotIn("edge_cost", text)

    def test_xml_declaration(self):
        g, *_ = line_graph(UGraph)
        self.assertTrue(
            g.export(format="graphml").startswith('<?xml version="1.0" encoding="UTF-8"?>')
        )


class TestElk(unittest.TestCase):
    def test_structure(self):
        g, *_ = line_graph(DGraph)
        d = json.loads(g.export(format="elk"))
        self.assertEqual(d["id"], "root")
        self.assertEqual([c["id"] for c in d["children"]], ["n0", "n1", "n2"])
        self.assertEqual(d["children"][1]["x"], 1.0)
        self.assertEqual((d["children"][0]["width"], d["children"][0]["height"]), (60, 40))
        self.assertEqual(
            [(e["sources"], e["targets"]) for e in d["edges"]],
            [(["n0"], ["n1"]), (["n1"], ["n2"])],
        )
        self.assertEqual(d["edges"][1]["properties"]["cost"], 2)
        self.assertEqual(d["layoutOptions"]["org.eclipse.elk.algorithm"], "fixed")
        self.assertEqual(d["layoutOptions"]["org.eclipse.elk.direction"], "DOWN")

    def test_layered_without_coordinates(self):
        g = UGraph()
        g.add_edge(g.add_vertex(name="p"), g.add_vertex(name="q"))
        d = json.loads(g.export(format="elk"))
        self.assertEqual(d["layoutOptions"]["org.eclipse.elk.algorithm"], "layered")
        self.assertNotIn("x", d["children"][0])

    def test_options(self):
        g, *_ = line_graph(UGraph)
        d = json.loads(
            g.export(
                format="elk",
                size=(10, 5),
                layout_options={"org.eclipse.elk.direction": "RIGHT", "k": "v"},
            )
        )
        self.assertEqual(d["children"][0]["width"], 10)
        self.assertEqual(d["layoutOptions"]["org.eclipse.elk.direction"], "RIGHT")
        self.assertEqual(d["layoutOptions"]["k"], "v")
        self.assertEqual(d["layoutOptions"]["org.eclipse.elk.algorithm"], "fixed")

    def test_names_roundtrip(self):
        g = UGraph()
        g.add_vertex(name='q"uote\\')
        d = json.loads(g.export(format="elk"))
        self.assertEqual(d["children"][0]["labels"][0]["text"], 'q"uote\\')


class TestMultigraph(unittest.TestCase):
    """loops and parallel edges are permitted, so every format must cope"""

    def build(self, Graph):
        g, a, b, c = line_graph(Graph)
        a.connect(b, cost=3)  # parallel to a-b
        a.connect(b, cost=0.5)  # and again
        b.connect(b, cost=7)  # self-loop
        if Graph is DGraph:
            b.connect(a, cost=9)  # antiparallel
        return g

    def expected_edges(self, Graph):
        # a-b, b-c, two more a-b, the loop (+ b->a for the directed graph)
        return 5 if Graph is UGraph else 6

    def test_counts(self):
        for Graph in (UGraph, DGraph):
            with self.subTest(Graph=Graph.__name__):
                g = self.build(Graph)
                self.assertEqual(g.ne, self.expected_edges(Graph))
                n = self.expected_edges(Graph)

                dot = g.export(format="dot")
                link = " -> " if Graph is DGraph else " -- "
                self.assertEqual(sum(link in l for l in dot.splitlines()), n)
                self.assertIn(f'"b"{link}"b"', dot)

                mm = g.export(format="mermaid")
                self.assertEqual(sum("|" in l for l in mm.splitlines()), n)

                gml = ET.fromstring(g.export(format="graphml"))  # nosec B314
                edges = gml.findall("g:graph/g:edge", NS)
                self.assertEqual(len(edges), n)
                self.assertEqual(len({e.get("id") for e in edges}), n)
                loops = [e for e in edges if e.get("source") == e.get("target")]
                self.assertEqual(len(loops), 1)

                elk = json.loads(g.export(format="elk"))
                self.assertEqual(len(elk["edges"]), n)
                self.assertEqual(len({e["id"] for e in elk["edges"]}), n)
                costs = sorted(e["properties"]["cost"] for e in elk["edges"])
                self.assertIn(0.5, costs)
                self.assertIn(3, costs)

    def test_reproducible(self):
        # edges() is a set; the export must not depend on its iteration order
        for fmt in ("dot", "mermaid", "graphml", "elk"):
            with self.subTest(format=fmt):
                outputs = {self.build(DGraph).export(format=fmt) for _ in range(5)}
                self.assertEqual(len(outputs), 1)

    def test_order_follows_vertices_then_edge_creation(self):
        g = self.build(UGraph)
        lines = [l for l in g.export(format="dot").splitlines() if "--" in l]
        self.assertEqual(
            lines,
            [
                '  "a" -- "b" [label="1"]',
                '  "a" -- "b" [label="3"]',
                '  "a" -- "b" [label="0.5"]',
                '  "b" -- "c" [label="2"]',  # created before the loop
                '  "b" -- "b" [label="7"]',
            ],
        )


class TestWrappers(unittest.TestCase):
    def test_dotfile_prints_to_console(self):
        g, *_ = line_graph(UGraph)
        with unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            g.dotfile()
        self.assertEqual(out.getvalue(), g.export(format="dot", costs=False))
        self.assertNotIn("label=", out.getvalue())
        # writing to the console must leave it usable (0.6.5 closed it)
        self.assertFalse(out.closed)

    def test_dotfile_to_path_and_stream(self):
        g, *_ = line_graph(UGraph)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "g.dot"
            g.dotfile(str(p), direction="LR")
            self.assertIn('rankdir="LR"', p.read_text())
        f = io.StringIO()
        g.dotfile(f)
        self.assertFalse(f.closed)
        self.assertTrue(f.getvalue().startswith("graph {"))

    def test_showgraph_missing_graphviz(self):
        g, *_ = line_graph(UGraph)
        with unittest.mock.patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as cm:
                g.showgraph(display=False)
        self.assertIn("GraphViz", str(cm.exception))


class TestRender(unittest.TestCase):
    def test_engine_not_installed(self):
        g, *_ = line_graph(UGraph)
        with unittest.mock.patch("shutil.which", return_value=None):
            with self.assertRaises(RuntimeError) as cm:
                g.export(io.BytesIO(), format="png", engine="neato")
        self.assertIn("neato", str(cm.exception))

    def test_engine_must_be_a_graphviz_program(self):
        # the name is run as a program, so it is checked against a fixed list
        g, *_ = line_graph(UGraph)
        for bad in ("rm", "../dot", "dot; rm -rf /", "no-such-engine"):
            with self.subTest(engine=bad):
                with self.assertRaises(ValueError):
                    g.export(io.BytesIO(), format="png", engine=bad)

    def test_unknown_image_format(self):
        g, *_ = line_graph(UGraph)
        with self.assertRaises(ValueError):
            exporters.render(g, "bmp")

    @unittest.skipUnless(HAVE_DOT, "GraphViz not installed")
    def test_svg_to_path_and_text_stream(self):
        g, *_ = line_graph(UGraph)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "g.svg"
            g.export(p)
            self.assertIn("<svg", p.read_text())
        f = io.StringIO()
        g.export(f, format="svg")
        self.assertIn("<svg", f.getvalue())

    @unittest.skipUnless(HAVE_DOT, "GraphViz not installed")
    def test_pdf_and_png_are_real_files(self):
        g, *_ = line_graph(DGraph)
        pdf, png = io.BytesIO(), io.BytesIO()
        g.export(pdf, format="pdf", direction="LR")
        g.export(png, format="png", color_components=True)
        self.assertEqual(pdf.getvalue()[:5], b"%PDF-")
        self.assertEqual(png.getvalue()[:4], b"\x89PNG")

    @unittest.skipUnless(HAVE_DOT, "GraphViz not installed")
    def test_binary_format_to_text_file_is_an_error(self):
        g, *_ = line_graph(UGraph)
        with self.assertRaises(TypeError):
            g.export(io.StringIO(), format="pdf")

    @unittest.skipUnless(HAVE_DOT, "GraphViz not installed")
    def test_positions_are_respected(self):
        # neato with pinned positions: the svg node centres keep the x order
        # and spacing of the coordinates, which dot would discard
        g = UGraph()
        a = g.add_vertex([0, 0], name="a")
        b = g.add_vertex([100, 0], name="b")
        g.add_edge(a, b)
        f = io.StringIO()
        g.export(f, format="svg")
        import re

        xs = [float(x) for x in re.findall(r"<ellipse[^>]* cx=\"(-?[\d.]+)\"", f.getvalue())]
        self.assertEqual(len(xs), 2)
        self.assertGreater(abs(xs[1] - xs[0]), 50)

    @unittest.skipUnless(HAVE_DOT, "GraphViz not installed")
    def test_multigraph_renders(self):
        g, a, b, c = line_graph(UGraph)
        a.connect(b, cost=3)
        b.connect(b, cost=7)
        f = io.BytesIO()
        g.export(f, format="png")
        self.assertEqual(f.getvalue()[:4], b"\x89PNG")

    @unittest.skipUnless(HAVE_DOT, "GraphViz not installed")
    def test_showgraph_display_false_returns_pdf_path(self):
        g, *_ = line_graph(UGraph)
        with unittest.mock.patch("pgraph.PGraph.webbrowser.open") as opened:
            path = g.showgraph(display=False)
        opened.assert_not_called()
        self.assertEqual(pathlib.Path(path).read_bytes()[:5], b"%PDF-")

    @unittest.skipUnless(HAVE_DOT, "GraphViz not installed")
    def test_showgraph_opens_browser(self):
        g, *_ = line_graph(UGraph)
        with unittest.mock.patch("pgraph.PGraph.webbrowser.open") as opened:
            self.assertIsNone(g.showgraph())
        opened.assert_called_once()
        self.assertTrue(opened.call_args[0][0].startswith("file://"))


if __name__ == "__main__":
    unittest.main()
