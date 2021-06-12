
import unittest
import numpy as np

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


    def test_bfs(self):
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

        p, length = g.path_BFS(v1, v5)
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

        p, length, parent = g.path_Astar(v1, v5)
        self.assertIsInstance(p, list)
        self.assertEqual(len(p), 5)
        self.assertEqual(p, [v1, v2, v3, v4, v5])
        self.assertIsInstance(length, float)
        self.assertAlmostEqual(length, 5.656854249492381)
        self.assertIsInstance(parent, dict)
        self.assertEqual(parent[v2], v1)
        self.assertEqual(parent[v3], v2)
        self.assertEqual(parent[v4], v3)
        self.assertEqual(parent[v5], v4)

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

        p, length, parent = g.path_Astar(v1, v5)
        self.assertIsInstance(p, list)
        self.assertEqual(len(p), 5)
        self.assertEqual(p, [v1, v2, v3, v4, v5])
        self.assertIsInstance(length, float)
        self.assertAlmostEqual(length, 5.656854249492381)
        self.assertIsInstance(parent, dict)
        self.assertEqual(parent[v2], v1)
        self.assertEqual(parent[v3], v2)
        self.assertEqual(parent[v4], v3)
        self.assertEqual(parent[v5], v4)


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