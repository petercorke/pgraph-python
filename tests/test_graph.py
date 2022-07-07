
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
        self.assertEqual(len(s.split('\n')), 2)

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

        g.remove(v1)
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

        g.remove(e12)
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


    # def test_remove_edge(self):

    #     g = UGraph()
    #     v1 = g.add_vertex()
    #     v2 = g.add_vertex()
    #     v3 = g.add_vertex()
    #     e12 = v1.connect(v2)
    #     e13 = v1.connect(v3)

    #     self.assertEqual(g.nc, 1)
    #     e12.remove()

    #     self.assertEqual(g.nc, 2)

    #     self.assertEqual(len(v1.edges()), 1)
    #     self.assertEqual(len(v2.edges()), 0)
    #     self.assertEqual(len(v3.edges()), 1)

    #     self.assertEqual(len(v1.neighbours()), 1)
    #     self.assertEqual(len(v2.neighbours()), 0)
    #     self.assertEqual(len(v3.neighbours()), 1)

    # def test_remove_vertex(self):

    #     g = UGraph()
    #     v1 = g.add_vertex()
    #     v2 = g.add_vertex()
    #     v3 = g.add_vertex()
    #     e12 = v1.connect(v2)
    #     e13 = v1.connect(v3)

    #     self.assertEqual(g.n, 3)
    #     self.assertEqual(g.nc, 1)
    #     g.remove(v1)

    #     self.assertEqual(g.n, 2)
    #     self.assertEqual(g.nc, 2)

    #     self.assertEqual(len(v1.edges()), 0)
    #     self.assertEqual(len(v2.edges()), 0)
    #     self.assertEqual(len(v3.edges()), 0)

    #     self.assertEqual(len(v1.neighbours()), 0)
    #     self.assertEqual(len(v2.neighbours()), 0)
    #     self.assertEqual(len(v3.neighbours()), 0)

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

    def test_matrices(self):
        g = UGraph()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        v3 = g.add_vertex()
        v4 = g.add_vertex()
        e12 = v1.connect(v2)
        e13 = v1.connect(v3)

        A = g.adjacency()
        self.assertIsInstance(A, np.ndarray)
        self.assertEqual(A.shape, (g.n, g.n))

        A = g.Laplacian()
        self.assertIsInstance(A, np.ndarray)
        self.assertEqual(A.shape, (g.n, g.n))

        A = g.incidence()
        self.assertIsInstance(A, np.ndarray)
        self.assertEqual(A.shape, (g.n, g.ne))

        A = g.distance()
        self.assertIsInstance(A, np.ndarray)
        self.assertEqual(A.shape, (g.n, g.n))

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

        g.showgraph()

class TestDGraph(unittest.TestCase):

    def test_constructor(self):

        g = DGraph()

        v1 = g.add_vertex()
        v2 = g.add_vertex()
        self.assertEqual(g.n, 2)
        self.assertIsInstance(v1, DVertex)
        self.assertIsInstance(v2, DVertex)

class TestGraph(unittest.TestCase):

    def test_print(self):
        pass

    def test_plot(self):
        pass


# ========================================================================== #

if __name__ == "__main__":
    unittest.main()