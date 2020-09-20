from abc import ABC, abstractmethod
import sys
import numpy as np
import matplotlib.pyplot as plt

class PGraph(ABC):

    @abstractmethod
    def add_node(self, coord=None, name=None):
        pass

    # @abstractmethod
    # def add_edge(self, v1, v2, cost=None):
    #     pass

    def __init__(self):
        self._nodelist = []
        self._nodedict = {}

    def new_node(self, node):
        if node.name is None:
            node.name = f"#{len(self._nodelist)}"
        self._nodelist.append(node)
        self._nodedict[node.name] = node

    def add_edge(self, v1, v2, **kwargs):
        v1 = self[v1]
        v2 = self[v2]
        v1.connect(v2, **kwargs)

    @property
    def n(self):
        """
        Number of vertices

        :return: Number of vertices
        :rtype: int
        """
        return len(self._nodedict)

    @property
    def nc(self):
        """
        Number of components

        :return: Number of components
        :rtype: int

        :notes:
        - Components are labeled from 0 to ``g.nc-1``.
        - A graph coloring algorithm is run if the graph 
        """
        self._graphcolor()
        return self._ncomponents

    def __repr__(self):
        s = []
        for node in self:
            s.append(f"{node.name} at {node.coord}, label={node.label}")
        return '\n'.join(s)


    def __getitem__(self, i):
        """
        [summary]

        :param i: vertex description, index or string
        :type i: int or str
        :return: the referenced vertex
        :rtype: Vertex subclass

        Th
        -``g[i]`` is the i'th node in the graph.  This reflects the order of 
         addition to the graph.
        -``g[s]`` is node named ``s``
        -``g[v]`` is ``v`` where ``v`` is a ``Vertex`` subclass

        This method also supports iteration over the vertices in a graph::

            for v in g:
                print(v)
        """
        if isinstance(i, int):
            return self._nodelist[i]
        elif isinstance(i, str):
            return self._nodedict[i]
        elif isinstance(i, Vertex):
            return i

    def edges(self):
        e = set()
        for node in self:
            e = e.union(node._edges)
        return e
            
    def plot(self, block=True):
        nc = self.nc
        print(nc, ' components')
        color = plt.cm.coolwarm(np.linspace(0, 1, nc))
        fig = plt.figure()
        for c in range(self.nc):
            # for each component
            for node in self.component(c):
                plt.text(node.x, node.y, "  " + node.name)
                plt.plot(node.x, node.y, 'o', color=color[c,:], markersize=12)
                for v in node.neighbours():
                    plt.plot([node.x, v.x], [node.y, v.y], color=color[c,:], linewidth=3)

        # if nc > 1:
        #     # add a colorbar
        #     plt.colorbar()
        plt.grid(True)
        if block:
            plt.show()

    def highlight_path(self, path, block=True, **kwargs):
        for i in range(len(path)):
            if i < len(path) - 1:
                e = path[i].edgeto(path[i+1])
                self.highlight_edge(e, **kwargs)
            self.highlight_node(path[i], **kwargs)
        plt.show(block=block)

    def highlight_edge(self, edge, scale=1.5, color='r'):
        p1 = edge.v1
        p2 = edge.v2
        plt.plot([p1.x, p2.x], [p1.y, p2.y], color=color, linewidth=3 * scale)

    def highlight_node(self, node, scale=1.5, color='r'):
        plt.plot(node.x, node.y, 'o', color=color, markersize=12 * scale)

    def dotfile(self, file=None):
        """
        Create a GraphViz dot file

        :param file: filename to save graph to, defaults to None
        :type file: str, optional

        ``g.dotfile()`` creates the specified file which contains the
        GraphViz code to represent the embedded graph.  By default output
        is to the console

        :notes:
        - The graph is undirected if it is a subclass of ``UGraph``
        - The graph is directed if it is a subclass of ``DGraph``
        - Use ``neato`` rather than dot to get the embedded layout
        """
     
        if file is not None:
            f = open('file', 'w')
        else:
            f = sys.stdout

        if isinstance(self, DGraph):
            print("digraph {", file=f)
        else:
            print("graph {", file=f)

        # add the nodes including name and position
        for node in self:
            print('  "{:s}" [pos="{:.5g},{:.5g}"]'.format(node.name, node.coord[0], node.coord[1]), file=f)
        print(file=f)
        # add the edges
        for e in self.edges():
            if isinstance(self, DGraph):
                print('  "{:s}" -> "{:s}"'.format(e.v1.name, e.v2.name), file=f)
            else:
                print('  "{:s}" -- "{:s}"'.format(e.v1.name, e.v2.name), file=f)

        print('}', file=f);

        if file is not None:
            f.close()

    def iscyclic(self):
        pass

    def average_degree(self):
        r"""
        Average degree of the graph

        :return: average degree
        :rtype: float

        Average degree is :math:`2 E / N` for an undirected graph and 
        :math:`E / N` for a directed graph where :math:`E` is the total number of
        edges and :math:`N` is the number of vertices.

        """
        if isinstance(self, DGraph):
            return len(self.edges() / self.n)
        elif isinstance(self, UGraph):
            return 2 * len(self.edges() / self.n)

# --------------------------------------------------------------------------- #

    ## MATRIX REPRESENTATIONS
    
    def Laplacian(self):
        """
        Laplacian matrix for the graph

        :return: Laplacian matrix
        :rtype: NumPy ndarray

        ``g.Laplacian()`` is the Laplacian matrix (NxN) of the graph where N
        is the number of vertices.

        :notes:
        - Laplacian is always positive-semidefinite.
        - Laplacian has at least one zero eigenvalue.
        - The number of zero-valued eigenvalues is the number of connected 
          components in the graph.
        
        :seealso: :func:`adjacency`, :func:`incidence`, :func:`degree`
        """
        return self.degree() - (g.adjacency() > 0)

    def connectivity(self):
        """
        Graph connectivity

        :return: [description]
        :rtype: [type]

        % C = G.connectivity() is a vector (Nx1) with the number of edges per
        % vertex.
        %
        % The average vertex connectivity is
        %         mean(g.connectivity())
        %
        % and the minimum vertex connectivity is
        %         min(g.connectivity())
        """

        c = []
        for n in self:
            c.append(len(n._edges))
        return c

    def degree(self):
        """
        %Pgraph.degree Degree matrix of graph
        %
        % D = G.degree() is a diagonal matrix (NxN) where element D(i,i) is the number
        % of edges connected to vertex id i.
        %
        % See also PGraph.adjacency, PGraph.incidence, PGraph.laplacian.
        """
        
        return np.diag( self.connectivity() );

    def adjacency(self):
        """
        %Pgraph.adjacency Adjacency matrix of graph
        %
        % A = G.adjacency() is a matrix (NxN) where element A(i,j) is the cost
        % of moving from vertex i to vertex j.
        %
        % Notes::
        % - Matrix is symmetric.
        % - Eigenvalues of A are real and are known as the spectrum of the graph.
        % - The element A(I,J) can be considered the number of walks of one
        %   edge from vertex I to vertex J (either zero or one).  The element (I,J)
        %   of A^N are the number of walks of length N from vertex I to vertex J.
        %
        % See also PGraph.degree, PGraph.incidence, PGraph.laplacian.
        """
        # create a dict mapping node to an id
        vdict = {}
        for i, vert in enumerate(self):
            vdict[vert] = i

        A = np.zeros((self.n, self.n))
        for node in self:
            for n in node.neighbours():
                A[vdict[node], vdict[n]] = 1
        return A

    def incidence(self):
        """
        %Pgraph.degree Incidence matrix of graph
        %
        % IN = G.incidence() is a matrix (NxNE) where element IN(i,j) is
        % non-zero if vertex id i is connected to edge id j.
        %
        % See also PGraph.adjacency, PGraph.degree, PGraph.laplacian.
        """
        edges = self.edges()
        I = np.zeros((self.n, len(edges)))

        # create a dict mapping edge to an id
        edict = {}
        for i, edge in enumerate(edges):
            edict[edge] = i

        for i, node in enumerate(self):
            for e in enumerate(node.edges()):
                I[i, edict[e]] = 1

    ## GRAPH COMPONENTS
        
    def _graphcolor(self):
        """
        Color the graph

        Performs a depth-first labeling operation, assigning the ``label`` 
        attribute of every vertex with a sequential integer starting from 0.

        This method checks the ``_connectivitychange`` attribute of all nodes
        and if any are True it will perform the coloring operation. This flag
        is set True by any operation that adds or removes a node or edge.

        :seealso: :func:`nc`
        """
        if any([n._connectivitychange for n in self]):

            # color the graph
            
            # clear all the labels
            for node in self:
                node.label = None
                node._connectivitychange = False
            
            def color_component(v, l):
                v.label = l
                for n in v.neighbours():
                    if n.label is None:
                        color_component(n, l)
            
            lastlabel = None
            for label in range(self.n):
                for v in self:
                    # find first vertex with no label
                    if v.label is None:
                        color_component(v, label)
                        lastlabel = label
                        break

            self._ncomponents = lastlabel + 1
    
    def component(self, c):
        """
        All nodes in specified graph component
        
        ``graph.component(c)`` is a list of all vertices in graph component ``c``.
        """
        self._graphcolor()  # ensure labels are uptodate
        return [v for v in self if v.label == c]

    def samecomponent(self, v1, v2):
        """
        %PGraph.component Graph component
        %
        % C = G.component(V) is the id of the graph component that contains vertex
        % V.
        """
        self._graphcolor()  # ensure labels are uptodate

        return v1.label == v2.label

    def remove(self, v):
        # remove edges from neighbour's edge list
        for e in v.edges():
            next = e.next(v)
            next._edges.remove(e)
            next._connectivitychange = True  

        # remove references from the graph
        self._nodelist.remove(v)
        for key, value in self._nodedict.items():
            if value is v:
                del self._nodedict[key]
                break

        v._edges = []  # remove all references to edges
# --------------------------------------------------------------------------- #

    def BFS(self, S, G):
        """
        Breadth-first search for path

        :param S: start vertex
        :type S: Vertex subclass
        :param G: goal vertex
        :type G: Vertex subclass
        :return: list of vertices from S to G inclusive
        :rtype: list of Vertex subclass

        :notes:
        - Returns None
        """
        S = self[S]
        G = self[G]
        frontier = set([S])
        explored = set()
        evaluation = [None for i in range(self.n)]
        parent = {}

        while frontier:
            x = frontier.pop()
            if x is G:
                break
            # expand the node
            for n in x.neighbours():
                if n not in frontier and n not in explored:
                    # add it to the frontier
                    frontier.add(n)
                    parent[n] = x
            explored.add(x)
        else:
            # no path
            return None

        # reconstruct the path from start to goal
        path = []
        x = G
        while True:
            path.insert(0, x)
            x = parent[x]
            if x not in parent:
                path.insert(0, x)
                break
        
        return path

    def Astar(self, S, G):
        S = self[S]
        G = self[G]
        frontier = [S]
        explored = set()
        parent = {}
        g = {S: 0} # cost to come
        f = {S: 0} # evaluation function

        def h(v1, v2):  # heuristic
            return np.linalg.norm(v1.coord - v2.coord)

        while frontier:
            i = np.argmin([f[n] for n in frontier])  # minimum f in frontier
            x = frontier.pop(i)
            if x is G:
                break
            # expand the node
            for n, e in x.incidences():
                if n not in frontier and n not in explored:
                    # add it to the frontier
                    frontier.append(n)
                    parent[n] = x
                    g[n] = g[x] + e.cost
                    f[n] = g[n] + h(n, G)
                elif n in frontier:
                    # neighbour is already in the frontier
                    gnew = g[x] + e.cost
                    if gnew < g[n]:
                        # cost of path via x is lower that previous, reparent it
                        g[n] = gnew
                        f[n] = g[n] + h(n, G)
                        parent[n] = x

            explored.add(x)
        else:
            # no path
            return None

        # reconstruct the path from start to goal
        x = G
        path = [x]

        while x is not S:
            x = parent[x]
            path.insert(0, x)

        
        return path

# -------------------------------------------------------------------------- #

class UGraph(PGraph):

    def add_node(self, coord=None, name=None):
        if isinstance(coord, UVertex):
            node = coord
        else:
            node = UVertex(coord, name)
        super().new_node(node)
        return node

class DGraph(PGraph):

    def add_node(self, coord=None, name=None):
        if isinstance(coord, DVertex):
            node = coord
        else:
            node = DVertex(coord, name)
        super().new_node(node)
        return node


# ========================================================================== #

class Edge:
    """
    Edge class

    Is used to represent directed directed and undirected edges.

    Each edge has:
    - ``cost`` cost of traversing this edge, required for planning methods
    - ``data`` reference to arbitrary data associated with the edge
    - ``v1`` first vertex, start vertex for a directed edge
    - ``v2`` second vertex, end vertex for a directed edge

    :notes:
    - An undirected graph is created by having a single edge object in the 
      edgelist of _each_ vertex.
    - This class can be inherited to provide user objects with graph capability.
    - Inheritance is an alternative to providing arbitrary user data.
    """

    def __init__(self, v1, v2, cost=None, data=None):
        if cost is None:
            try:
                self.cost = np.linalg.norm(v1.coord - v2.coord)
            except TypeError:
                self.cost = None
        else:
            self.cost = cost
        self.data = data
        self.v1 = v1
        self.v2 = v2

    def __repr__(self):
        return str(self)

    def __str__(self):

        return f"edge {self.v1} -- {self.v2}, cost={self.cost}, data={self.data}"

    def next(self, vertex):
        """
        Return other end of an edge

        :param vertex: one vertex on the edge
        :type vertex: Vertex subclass
        :raises ValueError: ``vertex`` is not on the edge
        :return: the other vertex on the edge
        :rtype: Vertex subclass

        ``e.next(v1)`` is the vertex at the other end of edge ``e``, ie. the 
        vertex that is not ``v1``.
        """
        
        if self.v1 is vertex:
            return self.v2
        elif self.v2 is vertex:
            return self.v1
        else:
            raise ValueError('shouldnt happen')

    def remove(self):
        """
        Remove edge from graph

        ``e.remove()`` removes ``e`` from the graph, but does not delete the
        edge object.
        """
        # remove this edge from the edge list of both end nodes
        if self in self.v1._edges:
            self.v1._edges.remove(self)
        if self in self.v2._edges:
            self.v2._edges.remove(self)

        # indicate that connectivity has changed
        self.v1._connectivitychange = True
        self.v2._connectivitychange = True

        # remove references to the nodes
        self.v1 = None
        self.v2 = None

# ========================================================================== #

class Vertex:
    """
    Superclass for vertices of directed and non-directed graphs.

    Each vertex has:
        - ``name``
        - ``label`` an int indicating which graph component contains it
        - ``_edges`` a list of edge objects that connect this vertex to others
        - ``coord`` the coordinate in an embedded graph (optional)
    """
    def __init__(self, coord=None, name=None):
        self._edges = []
        self.coord = np.r_[coord]
        self.name = name
        self.label = None
        self._connectivitychange = True
        self._edges = []
        print('Vertex init', type(self))

    def __str__(self):
        return f"[{self.name:s}]"

    def __repr__(self):
        return f"Vertex(name={self.name:s}, coord={self.coord})"

    def neighbours(self):
        """
        Neighbours of a vertex

        ``v.neighbours()`` is a list of neighbour of the vertex object ``v``.
        """
        return [e.next(self) for e in self._edges]

    def incidences(self):
        """
        Neighbours and edges of a vertex

        ``v.incidences()`` is a generator that returns a list of incidences, 
        tuples of (vertex, edge) for all neighbours of the vertex ``v``.
        """
        return [(e.next(self), e) for e in self._edges]

    def connect(self, dest, edge=None, cost=None, edgedata=None):
        """
        Connect two vertices with an edge

        :param dest: The node to connect to
        :type dest: ``Vertex`` subclass
        :param edge: Use this as the edge object, otherwise a new ``Edge``
        object is created, defaults to None
        :type edge: ``Edge`` subclass, optional
        :param cost: the cost to traverse this edge, required for planning methods, defaults to None
        :type cost: float, optional
        :param edgedata: reference to arbitrary data associated with the edge, defaults to None
        :type edgedata: Any, optional
        :raises TypeError: vertex types are different subclasses
        :return: the edge connecting the nodes
        :rtype: Edge

        ``v1.connect(v2)`` connects vertex ``v1`` to vertex ``v2``.

        :notes:
        - If the vertices subclass ``UVertex`` the edge is undirected, and if
          they subclass ``DVertex`` the edge is directed.
        - Vertices must both be of the same ``Vertex`` subclass
        """

        if not type(dest) is type(self):
            raise TypeError('must connect vertices of same type')
        elif isinstance(edge, Edge):
            e = edge
        else:
            e = Edge(self, dest, cost=cost, data=edgedata)
        self._connectivitychange = True
        
        return e

    def edgeto(self, dest):
        """
        Edge connecting vertex to specific neighbour

        :param dest: a neigbouring node
        :type dest: ``Vertex`` subclass
        :raises ValueError: ``dest`` is not a neighbour
        :return: the edge from this node to ``dest``
        :rtype: Edge

        :notes:
        - For a directed graph ``dest`` must be at the arrow end of the edge
        """
        for (n, e) in self.incidences():
            if n is dest:
                return e
        raise ValueError('dest is not a neighbour')

    def edges(self):
        """
        All outgoing edges of vertex

        :return: List of all edges leaving this node
        :rtype: list of Edge

        :notes:
        - For a directed graph the edges are those leaving this vertex
        - For a non-directed graph the edges are those leaving or entering
          this vertex
        """
        return self._edges

    @property
    def x(self):
        """
        The x-coordinate of an embedded vertex

        :return: The x-coordinate
        :rtype: float
        """
        return self.coord[0]

    @property
    def y(self):
        """
        The y-coordinate of an embedded vertex

        :return: The y-coordinate
        :rtype: float
        """
        return self.coord[1]

    @property
    def z(self):
        """
        The z-coordinate of an embedded vertex

        :return: The z-coordinate
        :rtype: float
        """
        return self.coord[2]

class UVertex(Vertex):
    """
    Vertex subclass for undirected graphs

    This class can be inherited to provide user objects with graph capability.
    """


    def connect(self, other, **kwargs):
        e = super().connect(other, **kwargs)
        
        self._edges.append(e)
        other._edges.append(e)
        return e



class DVertex(Vertex):
    """
    Vertex subclass for directed graphs

    This class can be inherited to provide user objects with graph capability.
    """

    def connect(self, other, **kwargs):
        e = super().connect(other, **kwargs)
        
        self._edges.append(e)
        return e

    def remove(self):
        self._edges = None  # remove all references to edges

# ========================================================================== #


def rand():
    return np.random.rand(2)

if __name__ == "__main__":


    import unittest

    class TestUGraph(unittest.TestCase):

        def test_constructor(self):

            g = UGraph()

            v1 = g.add_node()
            v2 = g.add_node()
            self.assertEqual(g.n, 2)
            self.assertIsInstance(v1, UVertex)
            self.assertIsInstance(v2, UVertex)

        def test_constructor2(self):

            g = UGraph()

            v = g.add_node([1,2,3])
            self.assertIsInstance(v, UVertex)
            self.assertEqual(v.x, 1)
            self.assertEqual(v.y, 2)
            self.assertEqual(v.z, 3)

        def test_attr(self):

            g = UGraph()

            v1 = g.add_node(name='v1')
            self.assertEqual(v1.name, 'v1')

            v1 = g.add_node(coord=[1,2,3])
            self.assertIsInstance(v1.coord, np.ndarray)
            self.assertEqual(v1.coord.shape, (3,))
            self.assertEqual(list(v1.coord), [1,2,3])

        def test_constructor3(self):

            g = UGraph()
            v1 = g.add_node()
            v2 = g.add_node()
            v3 = g.add_node()

            self.assertIs(g[0], v1)
            self.assertIs(g[0], v1)
            self.assertIs(g[0], v1)

            class MyNode(UVertex):
                def __init__(self, a):
                    super().__init__()
                    self.a = a

            v1 = g.add_node(MyNode(1))
            v2 = g.add_node(MyNode(2))

            self.assertIsInstance(v1, MyNode)
            v1.connect(v2)
            self.assertEqual(v1.neighbours()[0].a, 2)

        def test_getitem(self):
            g = UGraph()
            v1 = g.add_node(name='v1')
            v2 = g.add_node(name='v2')
            v3 = g.add_node(name='v3')

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
            v1 = g.add_node()
            v2 = g.add_node()
            v3 = g.add_node()
            e12 = v1.connect(v2)
            e13 = v1.connect(v3)
            
            self.assertIsInstance(e12, Edge)
            self.assertIsInstance(e12, Edge)

            self.assertTrue(e12 in v1.edges())
            self.assertTrue(e12 in v2.edges())
            self.assertFalse(e12 in v3.edges())

            self.assertTrue(e13 in v1.edges())
            self.assertTrue(e13 in v3.edges())
            self.assertFalse(e13 in v2.edges())

        def test_edge1(self):

            g = UGraph()
            v1 = g.add_node()
            v2 = g.add_node()
            v3 = g.add_node()
            v4 = g.add_node()
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
            v1 = g.add_node(name='n1')
            v2 = g.add_node(name='n2')
            v3 = g.add_node(name='n3')
            v4 = g.add_node(name='n4')

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
            v1 = g.add_node(name='n1')
            v2 = g.add_node(name='n2')
            v3 = g.add_node(name='n3')
            v4 = g.add_node(name='n4')

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
            v1 = g.add_node()
            v2 = g.add_node()
            v3 = g.add_node()
            e12 = v1.connect(v2)
            e13 = v1.connect(v3)

            self.assertIsInstance(v1.edgeto(v2), Edge)
            self.assertIs(v1.edgeto(v2), e12)
            self.assertIs(v1.edgeto(v2), e12)

        def test_remove_edge(self):

            g = UGraph()
            v1 = g.add_node()
            v2 = g.add_node()
            v3 = g.add_node()
            e12 = v1.connect(v2)
            e13 = v1.connect(v3)

            self.assertEqual(g.nc, 1)
            e12.remove()

            self.assertEqual(g.nc, 2)

            self.assertEqual(len(v1.edges()), 1)
            self.assertEqual(len(v2.edges()), 0)
            self.assertEqual(len(v3.edges()), 1)

            self.assertEqual(len(v1.neighbours()), 1)
            self.assertEqual(len(v2.neighbours()), 0)
            self.assertEqual(len(v3.neighbours()), 1)

        def test_remove_vertex(self):

            g = UGraph()
            v1 = g.add_node()
            v2 = g.add_node()
            v3 = g.add_node()
            e12 = v1.connect(v2)
            e13 = v1.connect(v3)

            self.assertEqual(g.n, 3)
            self.assertEqual(g.nc, 1)
            g.remove(v1)

            self.assertEqual(g.n, 2)
            self.assertEqual(g.nc, 2)

            self.assertEqual(len(v1.edges()), 0)
            self.assertEqual(len(v2.edges()), 0)
            self.assertEqual(len(v3.edges()), 0)

            self.assertEqual(len(v1.neighbours()), 0)
            self.assertEqual(len(v2.neighbours()), 0)
            self.assertEqual(len(v3.neighbours()), 0)

        def test_components(self):

            g = UGraph()
            v1 = g.add_node()
            v2 = g.add_node()
            v3 = g.add_node()

            self.assertEqual(g.nc, 3)
            v1.connect(v2)
            self.assertEqual(g.nc, 2)
            v1.connect(v3)
            self.assertEqual(g.nc, 1)

        def test_bfs(self):
            g = UGraph()
            v1 = g.add_node()
            v2 = g.add_node()
            v3 = g.add_node()
            v4 = g.add_node()
            v5 = g.add_node()
            v6 = g.add_node()

            v1.connect(v2)
            v2.connect(v3)
            v1.connect(v2)
            v1.connect(v4)
            v4.connect(v5)
            v5.connect(v3)

            p = g.BFS(v1, v6)
            self.assertIsNone(p)

            p = g.BFS(v1, v3)
            self.assertIsInstance(p, list)
            self.assertEqual(len(p), 3)
            self.assertEqual(p, [v1, v2, v3])

        def test_Astar(self):
            g = UGraph()
            v1 = g.add_node()
            v2 = g.add_node()
            v3 = g.add_node()
            v4 = g.add_node()
            v5 = g.add_node()
            v6 = g.add_node()

            v1.connect(v2, cost=1)
            v2.connect(v3, cost=1)
            v1.connect(v2, cost=1)
            v1.connect(v4, cost=1)
            v4.connect(v5, cost=1)
            v5.connect(v3, cost=1)

            p = g.Astar(v1, v6)
            self.assertIsNone(p)

            p = g.BFS(v1, v3)
            self.assertIsInstance(p, list)
            self.assertEqual(len(p), 3)
            self.assertEqual(p, [v1, v2, v3])


    class TestDGraph(unittest.TestCase):

        def test_constructor(self):

            g = DGraph()

            v1 = g.add_node()
            v2 = g.add_node()
            self.assertEqual(g.n, 2)
            self.assertIsInstance(v1, DVertex)
            self.assertIsInstance(v2, DVertex)

    class TestGraph(unittest.TestCase):

        def test_print(self):
            pass
    
        def test_plot(self):
            pass

    unittest.main()
    # v3 = g.add_node(rand())
    # v4 = g.add_node(rand())
    # v5 = g.add_node(rand())

    # v6 = g.add_node(rand())
    # v7 = g.add_node(rand())

    # print(g)

    # v1.connect(v2)
    # v1.connect(v3)
    # v1.connect(v4)
    # v2.connect(v3)
    # v2.connect(v4)
    # e = v4.connect(v5)
    # print(e)

    # v6.connect(v7)

    # print(list(v6.neighbours()))

    # g._graphcolor()

    # # print(g.BFS(v5, v3))
    # # g.plot()

    # # print(g)
    # # print('number of components', g.nc)
    # # print(g.component(0))
    # # print(g.component(1))
    # # print(g.samecomponent(v1, v2), g.samecomponent(v1, v7))
    # # print(g.n)


    # # for n in g:
    # #     print(n)

    # # e = g.edges()
    # # print(e)
    # # print(len(e))
    # print(g.connectivity())
    # A = g.incidence()
    # print(A)
    # # print(np.trace(A))
    # # print(A  @ A)
    # # print(np.trace(A @ A))
    # # print(A @ A @ A)
    # # print(np.trace(A @ A @ A))