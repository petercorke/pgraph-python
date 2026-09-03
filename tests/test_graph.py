
import unittest
import numpy as np
import numpy.testing as nt

from pgraph import *

class TestUGraph(unittest.TestCase):

    def test_constructor(self):

        g = UGraph()

        v1 = g.add_vertex()
        v2 = g.add_vertex()
        self.assertEqual(g.n, 2)
        self.assertIsInstance(v1, UVertex)
        self.assertIsInstance(v2, UVertex)

    def test_constructor2(self):

        g = UGraph()

        v = g.add_vertex([1,2,3])
        self.assertIsInstance(v, UVertex)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)

    def test_str(self):
        g = UGraph()

        v0 = g.add_vertex([1,2,3])
        v1 = g.add_vertex([1,2,3])
        v0.connect(v1)

        self.assertEqual(str(g), "UGraph: 2 vertices, 1 edge, 1 component")

        g = DGraph()

        v0 = g.add_vertex([1,2,3])
        v1 = g.add_vertex([1,2,3])
        v0.connect(v1)

        self.assertEqual(str(g), "DGraph: 2 vertices, 1 edge, 1 component")


        s = repr(g)
        self.assertIsInstance(s, str)
        lines = s.split('\n')
        self.assertEqual(len(lines), 3)  # class-name header + one line per vertex
        self.assertEqual(lines[0], "DGraph:")

    def test_attr(self):

        g = UGraph()

        v1 = g.add_vertex(name='v1')
        self.assertEqual(v1.name, 'v1')

        v1 = g.add_vertex(coord=[1,2,3])
        self.assertIsInstance(v1.coord, np.ndarray)
        self.assertEqual(v1.coord.shape, (3,))
        self.assertEqual(list(v1.coord), [1,2,3])

    def test_constructor3(self):

        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()

        self.assertIs(g[0], v1)
        self.assertIs(g[0], v1)
        self.assertIs(g[0], v1)

        class MyNode(UVertex):
            def __init__(self, a):
                super().__init__()
                self.a = a

        v1 = g.add_vertex(MyNode(1))
        v2 = g.add_vertex(MyNode(2))

        self.assertIsInstance(v1, MyNode)
        v1.connect(v2)
        self.assertEqual(v1.neighbours()[0].a, 2)

    def test_neighbours(self):
        g = UGraph()
        v1 = g.add_vertex(name='v1')
        v2 = g.add_vertex(name='v2')
        v3 = g.add_vertex(name='v3')
        v4 = g.add_vertex(name='v4')
        v1.connect(v2)
        v1.connect(v3)

        n = v1.neighbours()
        self.assertTrue(len(n) == 2)
        self.assertTrue(v2 in n)
        self.assertTrue(v3 in n)
        self.assertFalse(v1 in n)
        self.assertFalse(v4 in n)

        n = v2.neighbours()
        self.assertTrue(len(n) == 1)
        self.assertTrue(v1 in n)
        self.assertFalse(v2 in n)
        self.assertFalse(v3 in n)
        self.assertFalse(v4 in n)


        g = UGraph()
        v1 = g.add_vertex(name='v1')
        v2 = g.add_vertex(name='v2')
        v3 = g.add_vertex(name='v3')
        v1.connect(v2)

        self.assertTrue(v1.isneighbour(v2))
        self.assertTrue(v2.isneighbour(v1))
        self.assertFalse(v1.isneighbour(v3))
        self.assertFalse(v3.isneighbour(v1))

        g = DGraph()
        v1 = g.add_vertex(name='v1')
        v2 = g.add_vertex(name='v2')
        v3 = g.add_vertex(name='v3')
        v4 = g.add_vertex(name='v4')
        v1.connect(v2)
        v1.connect(v3)

        n = v1.neighbours()
        self.assertTrue(len(n) == 2)
        self.assertTrue(v2 in n)
        self.assertTrue(v3 in n)
        self.assertFalse(v1 in n)
        self.assertFalse(v4 in n)

        n = v2.neighbours()
        self.assertTrue(len(n) == 0)
        self.assertFalse(v1 in n)
        self.assertFalse(v2 in n)
        self.assertFalse(v3 in n)
        self.assertFalse(v4 in n)

        g = DGraph()
        v1 = g.add_vertex(name='v1')
        v2 = g.add_vertex(name='v2')
        v3 = g.add_vertex(name='v3')
        v1.connect(v2)
        self.assertTrue(v1.isneighbour(v2))
        self.assertFalse(v2.isneighbour(v1))
        self.assertFalse(v1.isneighbour(v3))
        self.assertFalse(v3.isneighbour(v1))

    def test_adjacent_deprecated(self):
        # adjacent() must still work (same result as neighbours()) but
        # emit a DeprecationWarning, not a hard failure
        g = UGraph()
        v1 = g.add_vertex(name='v1')
        v2 = g.add_vertex(name='v2')
        v1.connect(v2)

        with self.assertWarns(DeprecationWarning):
            n = v1.adjacent()
        self.assertEqual(n, v1.neighbours())

    def test_getitem(self):
        g = UGraph()
        v1 = g.add_vertex(name='v1')
        v2 = g.add_vertex(name='v2')
        v3 = g.add_vertex(name='v3')

        self.assertIs(g[0], v1)
        self.assertIs(g[1], v2)
        self.assertIs(g[2], v3)

        self.assertIs(g['v1'], v1)
        self.assertIs(g['v2'], v2)
        self.assertIs(g['v3'], v3)

        self.assertIs(g[v1], v1)
        self.assertIs(g[v2], v2)
        self.assertIs(g[v3], v3)

        v = [v for v in g]
        self.assertEqual(len(v), 3)
        self.assertEqual(v, [v1, v2, v3])

    def test_connect(self):

        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()
        e12 = v1.connect(v2)
        e13 = v1.connect(v3)
        
        self.assertEqual(g.n, 3)
        self.assertEqual(g.ne, 2)

        self.assertIsInstance(e12, Edge)
        self.assertIsInstance(e12, Edge)

        self.assertTrue(e12 in v1.edges())
        self.assertTrue(e12 in v2.edges())
        self.assertFalse(e12 in v3.edges())

        self.assertTrue(e13 in v1.edges())
        self.assertTrue(e13 in v3.edges())
        self.assertFalse(e13 in v2.edges())


    def test_remove_node(self):

        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()
        e12 = v1.connect(v2)
        e13 = v1.connect(v3)

        g.remove_vertex(v1)
        self.assertEqual(g.n, 2)
        self.assertEqual(g.ne, 0)
        self.assertEqual(g.nc, 2)

        self.assertFalse(e12 in v2.edges())
        self.assertFalse(e12 in v3.edges())

        self.assertFalse(e13 in v3.edges())
        self.assertFalse(e13 in v2.edges())

    def test_remove_edge(self):

        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()
        e12 = v1.connect(v2)
        e13 = v1.connect(v3)

        g.remove_edge(e12)
        self.assertEqual(g.n, 3)
        self.assertEqual(g.ne, 1)
        self.assertEqual(g.nc, 2)

        self.assertFalse(e12 in v1.edges())
        self.assertFalse(e12 in v2.edges())

        self.assertTrue(e13 in v1.edges())
        self.assertTrue(e13 in v3.edges())

    def test_edge1(self):

        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()
        v4 = g.add_vertex()
        v1.connect(v2)
        v1.connect(v3)

        self.assertEqual(len(v1.edges()), 2)
        self.assertEqual(len(v2.edges()), 1)
        self.assertEqual(len(v3.edges()), 1)

        self.assertEqual(len(v1.neighbours()), 2)
        self.assertEqual(len(v2.neighbours()), 1)
        self.assertEqual(len(v3.neighbours()), 1)

        self.assertTrue(v2 in v1.neighbours())
        self.assertTrue(v3 in v1.neighbours())
        self.assertFalse(v4 in v1.neighbours())

    def test_edge2(self):

        g = UGraph()
        v1 = g.add_vertex(name='n1')
        v2 = g.add_vertex(name='n2')
        v3 = g.add_vertex(name='n3')
        v4 = g.add_vertex(name='n4')

        g.add_edge('n1', 'n2')
        g.add_edge('n1', 'n3')

        self.assertEqual(len(v1.edges()), 2)
        self.assertEqual(len(v2.edges()), 1)
        self.assertEqual(len(v3.edges()), 1)

        self.assertEqual(len(v1.neighbours()), 2)
        self.assertEqual(len(v2.neighbours()), 1)
        self.assertEqual(len(v3.neighbours()), 1)

        self.assertTrue(v2 in v1.neighbours())
        self.assertTrue(v3 in v1.neighbours())
        self.assertFalse(v4 in v1.neighbours())

    def test_edge_vertices_deprecated(self):
        # vertices() must still work (same result as endpoints) but emit
        # a DeprecationWarning, not raise DeprecationWarning as a hard
        # failure (raising it aborts the call exactly like a missing
        # method would -- no backward compatibility at all)
        g = UGraph()
        v1 = g.add_vertex(name='v1')
        v2 = g.add_vertex(name='v2')
        e = v1.connect(v2)

        with self.assertWarns(DeprecationWarning):
            verts = e.vertices()
        self.assertEqual(verts, e.endpoints)

    def test_edge3(self):

        g = UGraph()
        v1 = g.add_vertex(name='n1')
        v2 = g.add_vertex(name='n2')
        v3 = g.add_vertex(name='n3')
        v4 = g.add_vertex(name='n4')

        g.add_edge(v1, v2)
        g.add_edge(v1, v3)

        self.assertEqual(len(v1.edges()), 2)
        self.assertEqual(len(v2.edges()), 1)
        self.assertEqual(len(v3.edges()), 1)

        self.assertEqual(len(v1.neighbours()), 2)
        self.assertEqual(len(v2.neighbours()), 1)
        self.assertEqual(len(v3.neighbours()), 1)

        self.assertTrue(v2 in v1.neighbours())
        self.assertTrue(v3 in v1.neighbours())
        self.assertFalse(v4 in v1.neighbours())


    def test_edgeto(self):

        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()
        e12 = v1.connect(v2)
        e13 = v1.connect(v3)

        self.assertIsInstance(v1.edgeto(v2), Edge)
        self.assertIs(v1.edgeto(v2), e12)
        self.assertIs(v1.edgeto(v2), e12)

    def test_add_vertex(self):

        g = UGraph()

        v = UVertex()

        g.add_vertex(v)
        self.assertEqual(v.name, '#0')
        self.assertTrue(v in g)
        self.assertTrue(v._graph, g)

    def test_dim(self):

        # unconstrained by default: any length, or no coord at all
        g = UGraph()
        g.add_vertex(coord=[1, 2])
        g.add_vertex(coord=[1, 2, 3, 4])
        g.add_vertex()
        self.assertEqual(g.n, 3)

        # dim enforces every embedded vertex has exactly that length
        g = UGraph(dim=6)
        g.add_vertex(coord=[0, 0, 0, 0, 0, 0], name='pose1')
        g.add_vertex(name='untyped')  # no coord: not checked
        with self.assertRaises(ValueError):
            g.add_vertex(coord=[1, 2, 3], name='bad')
        self.assertEqual(g.n, 2)

        # dim must be a positive integer
        with self.assertRaises(ValueError):
            UGraph(dim=0)
        with self.assertRaises(ValueError):
            UGraph(dim=-1)

    def test_properties(self):

        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()
        v4 = g.add_vertex()
        e12 = v1.connect(v2)
        e13 = v1.connect(v3)

        self.assertEqual(g.n, 4)
        self.assertEqual(g.ne, 2)
        self.assertEqual(g.average_degree(), 1.0)

        g = DGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()
        v4 = g.add_vertex()
        e12 = v1.connect(v2)
        e13 = v1.connect(v3)

        self.assertEqual(g.n, 4)
        self.assertEqual(g.ne, 2)
        self.assertEqual(g.average_degree(), 0.5)

    def test_contains(self):

        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()

        self.assertTrue('#0' in g)
        self.assertFalse('#2' in g)

        self.assertTrue(v1 in g)
        g2 = UGraph()
        self.assertFalse(v1 in g2)

    def test_Dict(self):

        v1 = UVertex(name='v1')
        v2 = UVertex(name='v2')
        v3 = UVertex(name='v3')
        v4 = UVertex(name='v4')

        parent = {}
        parent[v2] = v1
        parent[v3] = v1
        parent[v4] = v3

        g = UGraph.Dict(parent)

        self.assertIsInstance(g, UGraph)
        self.assertEqual(g.n, 4)
        self.assertEqual(g.ne, 3)
        self.assertTrue('v1' in g)
        self.assertTrue('v2' in g)
        self.assertTrue('v3' in g)
        self.assertTrue('v4' in g)

        self.assertTrue(g['v2'] in g['v1'].neighbours())
        self.assertTrue(g['v3'] in g['v1'].neighbours())
        self.assertTrue(g['v4'] in g['v3'].neighbours())

    def test_Adjacency(self):
        A = np.zeros((5, 5))
        A[1,2] = 5  # 1 <--> 2
        A[3,4] = 2  # 3 <--> 4
        
        coords = np.random.rand(5, 3)
        names = "zero one two three four".split(" ")

        g = UGraph.Adjacency(A, coords, names)
        self.assertIsInstance(g, UGraph)
        self.assertEqual(g.n, 5)
        self.assertEqual(g.ne, 2)
        self.assertTrue(g['two'] in g['one'].neighbours())
        self.assertTrue(g['three'] in g['four'].neighbours())
        e = g['two'].edgeto(g['one'])
        self.assertEqual(e.cost, 5)
        e = g['three'].edgeto(g['four'])
        self.assertEqual(e.cost, 2)
        nt.assert_almost_equal(g['two'].coord, coords[2,:])

        A = np.zeros((5, 5))
        A[1,2] = 5  # 1 --> 2
        A[3,4] = 2  # 3 --> 4
        
        coords = np.random.rand(5, 3)
        names = "zero one two three four".split(" ")

        g = DGraph.Adjacency(A, coords, names)
        self.assertIsInstance(g, DGraph)
        self.assertEqual(g.n, 5)
        self.assertEqual(g.ne, 2)
        self.assertTrue(g['two'] in g['one'].neighbours())
        self.assertTrue(g['four'] in g['three'].neighbours())
        e = g['one'].edgeto(g['two'])
        self.assertEqual(e.cost, 5)
        e = g['three'].edgeto(g['four'])
        self.assertEqual(e.cost, 2)
        nt.assert_almost_equal(g['two'].coord, coords[2,:])


    def test_remove_edge_via_edge(self):
        # Edge.remove(), as opposed to g.remove(edge) covered above

        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()
        e12 = v1.connect(v2)
        e13 = v1.connect(v3)

        self.assertEqual(g.nc, 1)
        e12.remove()

        self.assertEqual(g.nc, 2)
        self.assertEqual(g.ne, 1)
        self.assertNotIn(e12, g.edges())

        self.assertEqual(len(v1.edges()), 1)
        self.assertEqual(len(v2.edges()), 0)
        self.assertEqual(len(v3.edges()), 1)

        self.assertEqual(len(v1.neighbours()), 1)
        self.assertEqual(len(v2.neighbours()), 0)
        self.assertEqual(len(v3.neighbours()), 1)

        with self.assertRaises(ValueError):
            e12.remove()  # already removed, no longer connected to a graph

    def test_remove_deprecated(self):
        # remove() must still work (dispatching to remove_edge()/
        # remove_vertex()) but emit a DeprecationWarning, not a hard failure
        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()
        e12 = v1.connect(v2)
        v1.connect(v3)

        with self.assertWarns(DeprecationWarning):
            g.remove(e12)
        self.assertEqual(g.ne, 1)

        with self.assertWarns(DeprecationWarning):
            g.remove(v1)
        self.assertEqual(g.n, 2)

        with self.assertRaises(TypeError):
            g.remove(42)

    def test_components(self):

        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()

        self.assertEqual(g.nc, 3)
        v1.connect(v2)
        self.assertEqual(g.nc, 2)
        v1.connect(v3)
        self.assertEqual(g.nc, 1)

    def test_iscyclic(self):

        # tree: acyclic
        g = UGraph()
        v = [g.add_vertex(name=str(i)) for i in range(4)]
        g.add_edge(v[0], v[1])
        g.add_edge(v[1], v[2])
        g.add_edge(v[2], v[3])
        self.assertFalse(g.iscyclic())

        # one extra edge closes a cycle
        g.add_edge(v[3], v[0])
        self.assertTrue(g.iscyclic())

        # two disconnected trees: still acyclic
        g2 = UGraph()
        a = [g2.add_vertex(name=f'a{i}') for i in range(3)]
        b = [g2.add_vertex(name=f'b{i}') for i in range(2)]
        g2.add_edge(a[0], a[1])
        g2.add_edge(a[1], a[2])
        g2.add_edge(b[0], b[1])
        self.assertFalse(g2.iscyclic())

    def test_matrices(self):
        g = UGraph()
        # coordinates are needed so edge cost auto-computes, otherwise
        # distance() would have nothing to put in the distance matrix
        v1 = g.add_vertex(coord=[0, 0])
        v2 = g.add_vertex(coord=[1, 0])
        v3 = g.add_vertex(coord=[0, 1])
        v4 = g.add_vertex(coord=[1, 1])  # deliberately isolated
        e12 = v1.connect(v2)
        e13 = v1.connect(v3)

        # vertex order is deterministic (insertion order), so adjacency,
        # degree and Laplacian -- all indexed by vertex only -- can be
        # checked against exact expected matrices
        A = g.adjacency()
        self.assertIsInstance(A, np.ndarray)
        expected_A = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        nt.assert_array_equal(A, expected_A)
        self.assertTrue(np.array_equal(A, A.T))  # symmetric for UGraph

        D = g.degree()
        nt.assert_array_equal(D, np.diag([2, 1, 1, 0]))

        L = g.Laplacian()
        expected_L = np.diag([2, 1, 1, 0]) - expected_A
        nt.assert_array_equal(L, expected_L)

        # edge order (columns) is not deterministic -- self.edges() is a
        # set -- so check incidence structurally rather than by position:
        # each column (edge) touches exactly 2 vertices, and each row
        # (vertex)'s total matches its degree
        I = g.incidence()
        self.assertEqual(I.shape, (g.n, g.ne))
        nt.assert_array_equal(I.sum(axis=0), np.full(g.ne, 2))
        nt.assert_array_equal(I.sum(axis=1), [2, 1, 1, 0])

        A = g.distance()
        self.assertIsInstance(A, np.ndarray)
        self.assertEqual(A.shape, (g.n, g.n))
        self.assertFalse(np.any(np.isnan(A)))
        self.assertEqual(A[0, 1], 1.0)  # v1 to v2
        self.assertTrue(np.array_equal(A, A.T))  # symmetric for UGraph

    def test_distance_requires_cost(self):
        # regression test: distance() used to silently pack None/nan into
        # the matrix for edges with no cost, instead of raising
        g = UGraph()
        v1 = g.add_vertex()  # no coord -> edge cost can't auto-compute
        v2 = g.add_vertex()
        v1.connect(v2)
        with self.assertRaises(ValueError):
            g.distance()

    def test_metric(self):
        g = UGraph()
        v1 = g.add_vertex([1,2,3])
        p = [7,6,6]
        self.assertAlmostEqual(v1.distance(p), np.sqrt(61))

        g = UGraph(metric='L2')
        v1 = g.add_vertex([1,2,3])
        p = [7,6,6]
        self.assertAlmostEqual(v1.distance(p), np.sqrt(61))

        g = UGraph(metric='L1')
        v1 = g.add_vertex([1,2,3])
        p = [7,6,6]
        self.assertEqual(v1.distance(p), 13)

        g = UGraph(metric='SE2')
        v1 = g.add_vertex([1,2,0])
        p = [7,6,0]
        self.assertAlmostEqual(v1.distance(p), np.sqrt(52))
        p = [7,6,2*np.pi]
        self.assertAlmostEqual(v1.distance(p), np.sqrt(52))
        p = [7,6,-2*np.pi]
        self.assertAlmostEqual(v1.distance(p), np.sqrt(52))
        p = [7,6,4*np.pi]
        self.assertAlmostEqual(v1.distance(p), np.sqrt(52))
        p = [7,6,-4*np.pi]
        self.assertAlmostEqual(v1.distance(p), np.sqrt(52))

        p = [7,6,np.pi]
        self.assertAlmostEqual(v1.distance(p), np.sqrt(52+np.pi**2))

        v2 = g.add_vertex([1,2,np.pi/2])
        p = [7,6,np.pi/2]
        self.assertAlmostEqual(v2.distance(p), np.sqrt(52))
        p = [7,6,-np.pi/2]
        self.assertAlmostEqual(v2.distance(p), np.sqrt(52+np.pi**2))

    def test_heuristic(self):
        g = UGraph()
        p = [2, 3, 4]
        self.assertAlmostEqual(g.heuristic(p), np.sqrt(29))

        g = UGraph(heuristic='L2')
        p = [2, 3, 4]
        self.assertAlmostEqual(g.heuristic(p), np.sqrt(29))

        g = UGraph(heuristic='L1')
        p = [2, 3, 4]
        self.assertAlmostEqual(g.heuristic(p), 9)


    def test_closest(self):
        g = UGraph()
        v1 = g.add_vertex([1,2,3])
        v2 = g.add_vertex([4,5,6])
        v, d = g.closest([4, 5, 7])
        self.assertIs(v, v2)
        self.assertEqual(d, 1)

    def test_BFS(self):
        g = UGraph()
        v1 = g.add_vertex(coord=[0,0], name='v1')
        v2 = g.add_vertex(coord=[1,1], name='v2')
        v3 = g.add_vertex(coord=[2,2], name='v3')
        v4 = g.add_vertex(coord=[1,3], name='v4')
        v5 = g.add_vertex(coord=[0,4], name='v5')
        v6 = g.add_vertex(coord=[-5,2], name='v6')
        v7 = g.add_vertex(coord=[0,6], name='v7')

        v1.connect(v2)
        v2.connect(v3)
        v3.connect(v4)
        v4.connect(v5)
        v1.connect(v6)
        e = v6.connect(v5)

        p = g.path_UCS(v1, v7)
        self.assertIsNone(p)

        p, length = g.path_BFS(v1, v5, verbose=True, summary=True)
        self.assertIsInstance(p, list)
        self.assertEqual(len(p), 3)
        self.assertEqual(p, [v1, v6, v5])

    def test_UCS(self):
        g = UGraph()
        v1 = g.add_vertex(coord=[0,0], name='v1')
        v2 = g.add_vertex(coord=[1,1], name='v2')
        v3 = g.add_vertex(coord=[2,2], name='v3')
        v4 = g.add_vertex(coord=[1,3], name='v4')
        v5 = g.add_vertex(coord=[0,4], name='v5')
        v6 = g.add_vertex(coord=[-5,2], name='v6')
        v7 = g.add_vertex(coord=[0,6], name='v7')

        v1.connect(v2)
        v2.connect(v3)
        v3.connect(v4)
        v4.connect(v5)
        v1.connect(v6)
        e = v6.connect(v5)

        p = g.path_UCS(v1, v7)
        self.assertIsNone(p)

        p, length, parent = g.path_UCS(v1, v5, verbose=True, summary=True)
        self.assertIsInstance(p, list)
        self.assertEqual(len(p), 5)
        self.assertEqual(p, [v1, v2, v3, v4, v5])
        self.assertIsInstance(length, float)
        self.assertAlmostEqual(length, 5.656854249492381)
        self.assertIsInstance(parent, dict)
        self.assertEqual(parent[v2.name], v1.name)
        self.assertEqual(parent[v3.name], v2.name)
        self.assertEqual(parent[v4.name], v3.name)
        self.assertEqual(parent[v5.name], v4.name)

    def test_Astar(self):
        g = UGraph()
        v1 = g.add_vertex(coord=[0,0], name='v1')
        v2 = g.add_vertex(coord=[1,1], name='v2')
        v3 = g.add_vertex(coord=[2,2], name='v3')
        v4 = g.add_vertex(coord=[1,3], name='v4')
        v5 = g.add_vertex(coord=[0,4], name='v5')
        v6 = g.add_vertex(coord=[-5,2], name='v6')
        v7 = g.add_vertex(coord=[0,6], name='v7')

        v1.connect(v2)
        v2.connect(v3)
        v3.connect(v4)
        v4.connect(v5)
        v1.connect(v6)
        e = v6.connect(v5)

        p = g.path_Astar(v1, v7)
        self.assertIsNone(p)

        p, length, parent = g.path_Astar(v1, v5, verbose=True, summary=True)
        self.assertIsInstance(p, list)
        self.assertEqual(len(p), 5)
        self.assertEqual(p, [v1, v2, v3, v4, v5])
        self.assertIsInstance(length, float)
        self.assertAlmostEqual(length, 5.656854249492381)

        self.assertIsInstance(parent, dict)
        self.assertEqual(parent[v2.name], v1.name)
        self.assertEqual(parent[v3.name], v2.name)
        self.assertEqual(parent[v4.name], v3.name)
        self.assertEqual(parent[v5.name], v4.name)

    def test_plot(self):

        g = UGraph()
        v1 = g.add_vertex(coord=[0,0], name='v1')
        v2 = g.add_vertex(coord=[1,1], name='v2')
        v3 = g.add_vertex(coord=[2,2], name='v3')
        v4 = g.add_vertex(coord=[1,3], name='v4')
        v5 = g.add_vertex(coord=[0,4], name='v5')
        v6 = g.add_vertex(coord=[-5,2], name='v6')
        v7 = g.add_vertex(coord=[0,6], name='v7')

        v1.connect(v2)
        v2.connect(v3)
        v3.connect(v4)
        v4.connect(v5)
        v1.connect(v6)
        e = v6.connect(v5)

        g.plot()

        p, length, parent = g.path_Astar(v1, v5, verbose=True, summary=True)
        g.highlight_path(p)

    def test_dotfile(self):
        import pathlib
        import unittest.mock

        g = UGraph()
        v1 = g.add_vertex(coord=[0,0], name='v1')
        v2 = g.add_vertex(coord=[1,1], name='v2')
        v3 = g.add_vertex(coord=[2,2], name='v3')
        v4 = g.add_vertex(coord=[1,3], name='v4')
        v5 = g.add_vertex(coord=[0,4], name='v5')
        v6 = g.add_vertex(coord=[-5,2], name='v6')
        v7 = g.add_vertex(coord=[0,6], name='v7')

        path = pathlib.Path('./dotfile.dot')
        g.dotfile(str(path))
        self.assertTrue(path.is_file())

        # showgraph() pops a PDF up in a browser via webbrowser.open() --
        # mock that so the test runs headless, but still exercise the real
        # dot->PDF rendering and check the result is an actual PDF file.
        with unittest.mock.patch('pgraph.PGraph.webbrowser.open') as mock_open:
            g.showgraph()

        mock_open.assert_called_once()
        pdf_url = mock_open.call_args[0][0]
        pdf_path = pathlib.Path(pdf_url.removeprefix('file://'))
        self.assertTrue(pdf_path.is_file())
        self.assertEqual(pdf_path.read_bytes()[:5], b'%PDF-')

class TestDGraph(unittest.TestCase):

    def test_constructor(self):

        g = DGraph()

        v1 = g.add_vertex()
        v2 = g.add_vertex()
        self.assertEqual(g.n, 2)
        self.assertIsInstance(v1, DVertex)
        self.assertIsInstance(v2, DVertex)

    def test_Dict(self):
        # regression test: Dict() used to hardcode UVertex() regardless of
        # the graph type, silently producing wrong-typed vertices in a DGraph
        g = DGraph.Dict({'child': 'parent'})

        self.assertIsInstance(g, DGraph)
        self.assertEqual(g.n, 2)
        for v in g:
            self.assertIsInstance(v, DVertex)

    def test_iscyclic(self):

        # diamond DAG: shared descendant, but acyclic -- a classic false
        # positive for a naive/buggy cycle detector
        g = DGraph()
        v = [g.add_vertex(name=str(i)) for i in range(4)]
        g.add_edge(v[0], v[1])
        g.add_edge(v[0], v[2])
        g.add_edge(v[1], v[3])
        g.add_edge(v[2], v[3])
        self.assertFalse(g.iscyclic())

        # a back-edge closes an actual directed cycle
        g.add_edge(v[3], v[0])
        self.assertTrue(g.iscyclic())

        # disconnected DAG components: still acyclic
        g2 = DGraph()
        x = [g2.add_vertex(name=f'x{i}') for i in range(2)]
        y = [g2.add_vertex(name=f'y{i}') for i in range(2)]
        g2.add_edge(x[0], x[1])
        g2.add_edge(y[0], y[1])
        self.assertFalse(g2.iscyclic())

        # pure 3-cycle, no self-loops
        g3 = DGraph()
        t = [g3.add_vertex(name=str(i)) for i in range(3)]
        g3.add_edge(t[0], t[1])
        g3.add_edge(t[1], t[2])
        g3.add_edge(t[2], t[0])
        self.assertTrue(g3.iscyclic())

    def test_matrices(self):
        # 3-4-5 triangle, directed 0->1, 0->2, 1->2 -- deliberately
        # asymmetric so directed-specific behaviour (out-degree only,
        # non-symmetric adjacency) actually gets exercised
        g = DGraph()
        v0 = g.add_vertex(coord=[0, 0], name='0')
        v1 = g.add_vertex(coord=[3, 0], name='1')
        v2 = g.add_vertex(coord=[3, 4], name='2')
        g.add_edge(v0, v1)
        g.add_edge(v0, v2)
        g.add_edge(v1, v2)

        A = g.adjacency()
        expected_A = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ])
        nt.assert_array_equal(A, expected_A)
        self.assertFalse(np.array_equal(A, A.T))  # not symmetric, directed

        # out-degree only: v0 has 2 outgoing, v1 has 1, v2 has 0 (even
        # though v2 has two *incoming* edges)
        D = g.degree()
        nt.assert_array_equal(D, np.diag([2, 1, 0]))

        L = g.Laplacian()
        expected_L = np.diag([2, 1, 0]) - expected_A
        nt.assert_array_equal(L, expected_L)

        # incidence marks both endpoints regardless of direction, so unlike
        # degree() every vertex here touches exactly 2 edges
        I = g.incidence()
        self.assertEqual(I.shape, (g.n, g.ne))
        nt.assert_array_equal(I.sum(axis=0), np.full(g.ne, 2))
        nt.assert_array_equal(I.sum(axis=1), [2, 2, 2])

        A = g.distance()
        expected_D = np.array([
            [0, 3, 5],
            [0, 0, 4],
            [0, 0, 0],
        ])
        nt.assert_array_almost_equal(A, expected_D)
        self.assertFalse(np.array_equal(A, A.T))  # not symmetric, directed

    def test_remove_edge(self):
        # regression test: removing a directed edge used to always crash,
        # since it assumed both endpoints track the edge in their own
        # _edgelist, but a DVertex only tracks outgoing edges
        g = DGraph()
        v1 = g.add_vertex(name='1')
        v2 = g.add_vertex(name='2')
        e = g.add_edge(v1, v2)

        self.assertIn(e, v1._edgelist)  # source tracks it
        self.assertNotIn(e, v2._edgelist)  # target does not

        g.remove_edge(e)
        self.assertEqual(g.ne, 0)
        self.assertEqual(g.n, 2)
        self.assertIsNone(e.v1)
        self.assertIsNone(e.v2)

        with self.assertRaises(ValueError):
            g.remove_edge(e)  # already removed

    def test_remove_vertex(self):
        # regression test: removing a vertex used to only clean up its
        # *outgoing* edges (via vertex._edgelist), leaving any incoming
        # edge dangling -- referencing a removed vertex, still present in
        # the graph's own edge set and the source vertex's edgelist
        g = DGraph()
        v0 = g.add_vertex(name='0')
        v1 = g.add_vertex(name='1')
        v2 = g.add_vertex(name='2')
        g.add_edge(v0, v1)  # incoming to v1
        g.add_edge(v1, v2)  # outgoing from v1

        self.assertEqual(g.n, 3)
        self.assertEqual(g.ne, 2)

        g.remove_vertex(v1)

        self.assertEqual(g.n, 2)
        self.assertEqual(g.ne, 0)  # both incoming and outgoing edges gone
        self.assertEqual(len(v0._edgelist), 0)
        self.assertEqual(len(v2._edgelist), 0)

        with self.assertRaises(ValueError):
            g.remove_vertex(v1)  # no longer belongs to this graph

    def test_remove_via_instance_methods(self):
        # Edge.remove() / BaseVertex.remove(), the ergonomic wrappers
        g = DGraph()
        v0 = g.add_vertex(name='0')
        v1 = g.add_vertex(name='1')
        v2 = g.add_vertex(name='2')
        g.add_edge(v0, v1)
        e12 = g.add_edge(v1, v2)

        e12.remove()
        self.assertEqual(g.ne, 1)

        v1.remove()  # cascades: removes the remaining v0->v1 edge too
        self.assertEqual(g.n, 2)
        self.assertEqual(g.ne, 0)

        with self.assertRaises(ValueError):
            v1.remove()  # no longer connected to a graph

class TestGraph(unittest.TestCase):

    def test_print(self):
        pass

    def test_plot(self):
        pass


# ========================================================================== #

if __name__ == "__main__":
    unittest.main()