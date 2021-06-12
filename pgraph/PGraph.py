from abc import ABC, abstractmethod
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections.abc import Iterable
import tempfile
import subprocess
import webbrowser
class PGraph(ABC):

    @abstractmethod
    def add_vertex(self, coord=None, name=None):
        pass

    # @abstractmethod
    # def add_edge(self, v1, v2, cost=None):
    #     pass

    def __init__(self, arg=None, metric=None, verbose=False):
        self._vertexlist = []
        self._vertexdict = {}
        self._edges = set()
        self._verbose = verbose
        if metric is not None and callable(metric):
            self._metric = metric
        else:
            self._metric = np.linalg.norm

    def copy(self, g):
        """
        Deepcopy of graph

        :param g: A graph
        :type g: PGraph
        :return: deep copy
        :rtype: PGraph
        """
        return copy.deepcopy(g)
            
    def add_vertex(self, vertex):
        """
        Add a vertex to the graph (superclass method)

        :param vertex: vertex to add
        :type vertex: Vertex subclass

        ``G.add_vertex(v)`` add vertex ``v`` to the graph ``G``.

        If the vertex has no name give it a default name ``#N`` where ``N``
        is a consecutive integer.

        The vertex is placed into a dictionary with a key equal to its name.
        """
        if vertex.name is None:
            vertex.name = f"#{len(self._vertexlist)}"
        self._vertexlist.append(vertex)
        self._vertexdict[vertex.name] = vertex
        if self._verbose:
            print(f"New vertex {vertex.name}: {vertex.coord}")
        vertex._graph = self
        
    def add_edge(self, v1, v2, **kwargs):
        """
        Add an edge to the graph (superclass method)

        :param v1: first vertex (start if a directed graph)
        :type v1: Vertex subclass
        :param v2: second vertex (end if a directed graph)
        :type v2: Vertex subclass
        :param kwargs: optional arguments to pass to ``connect``
        :return: edge
        :rtype: Edge

        .. note:: This is a graph centric way of creating an edge.  The 
            alternative is the ``connect`` method of a vertex.
    
        :seealso: :func:`Edge.connect`
        """
        v1 = self[v1]
        v2 = self[v2]
        if self._verbose:
            print(f"New edge from {v1.name} to {v2.name}")
        return v1.connect(v2, **kwargs)

    def remove(self, x):
        """
        Remove element from graph (superclass method)

        :param x: element to remove from graph
        :type x: Edge or Vertex subclass
        :raises TypeError: unknown type

        The edge or vertex is removed, and all references and lists are
        updated.

        .. warning:: The connectivity of the network may be changed.
        """
        if isinstance(x, Edge):
            # remove an edge

            # remove edge from the edgelist of connected vertices
            x.v1._edges.remove(x)
            x.v2._edges.remove(x)

            # indicate that connectivity has changed
            x.v1._connectivitychange = True
            x.v2._connectivitychange = True

            # remove references to the nodes
            x.v1 = None
            x.v2 = None

            # remove from list of all edges
            self._edges.remove(x)

        elif isinstance(x, Vertex):
            # remove a vertex

            # remove all edges of this vertex
            for edge in copy.copy(x._edges):
                self.remove(edge)
            
            # remove from list and dict of all edges
            self._vertexlist.remove(x)
            del self._vertexdict[x.name]
        else:
            raise TypeError('expecting Edge or Vertex')

    def show(self):
        print('vertices:')
        for v in self._vertexlist:
            print('  ' + str(v))
        print('edges:')
        for e in self._edges:
            print('  ' + str(e))

    @property
    def n(self):
        """
        Number of vertices

        :return: Number of vertices
        :rtype: int
        """
        return len(self._vertexdict)

    @property
    def ne(self):
        """
        Number of edges

        :return: Number of vertices
        :rtype: int
        """
        return len(self._edges)

    @property
    def nc(self):
        """
        Number of components

        :return: Number of components
        :rtype: int

        .. note::

            - Components are labeled from 0 to ``g.nc-1``.
            - A graph coloring algorithm is run if the graph connectivity
              has changed.

        .. note:: A lazy approach is used, and if a connectivity changing 
            operation has been performed since the last call, the graph
            coloring algorithm is run which is potentially expensive for 
            a large graph.
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
        Get vertex (superclass method)

        :param i: vertex description
        :type i: int or str
        :return: the referenced vertex
        :rtype: Vertex subclass

        Retrieve a vertex by index or name:

        -``g[i]`` is the i'th node in the graph.  This reflects the order of 
         addition to the graph.
        -``g[s]`` is node named ``s``
        -``g[v]`` is ``v`` where ``v`` is a ``Vertex`` subclass

        This method also supports iteration over the vertices in a graph::

            for v in g:
                print(v)

        will iterate over all the vertices.
        """
        if isinstance(i, int):
            return self._vertexlist[i]
        elif isinstance(i, str):
            return self._vertexdict[i]
        elif isinstance(i, Vertex):
            return i

    def edges(self):
        """
        Get all edges in graph (superclass method)

        :return: All edges in the graph
        :rtype: list of Edge references

        We can iterate over all edges in the graph by::

            for e in g.edges():
                print(e)

        .. note:: The ``edges()`` of a Vertex is a list of all edges connected
            to that vertex.

        :seealso: :func:`Vertex.edges`
        """
        return self._edgelist
            
    def plot(self, vertex=None, edge=None, text=None, color=None, block=True):
        """
        Plot the graph

        :param vertex: vertex format, defaults to 12pt o-marker
        :type vertex: dict, optional
        :param edge: edge format, defaults to None
        :type edge: dict, optional
        :param text: [description], defaults to None
        :type text: [type], optional
        :param color: [description], defaults to None
        :type color: [type], optional
        :param block: block until figure is dismissed, defaults to True
        :type block: bool, optional

        The graph is plotted using matplotlib.

        .. todo::

            - more options
            - save options for use by highlight
        """

        if vertex is None:
            vertex = {"marker": 'o', "markersize": 12}
        else:
            if "marker" not in vertex:
                vertex["marker"] = 'o'  # default circular marker
        if edge is None:
            edge = {"linewidth": 3}
        if text is None:
            text = {}

        if color is None:
            color = plt.cm.coolwarm(np.linspace(0, 1, self.nc))

        fig = plt.figure()
        for c in range(self.nc):
            # for each component
            for node in self.component(c):
                plt.text(node.x, node.y, "  " + node.name, **text)
                plt.plot(node.x, node.y, color=color[c,:], **vertex)
                for v in node.neighbours():
                    plt.plot([node.x, v.x], [node.y, v.y], color=color[c,:], **edge)

        # if nc > 1:
        #     # add a colorbar
        #     plt.colorbar()
        plt.grid(True)
        if block:
            plt.show()

    def highlight_path(self, path, block=True, **kwargs):
        """
        Highlight a path through the graph

        :param path: [description]
        :type path: [type]
        :param block: [description], defaults to True
        :type block: bool, optional

        The nodes and edges along the path are overwritten with a different
        size/width and color.

        :seealso: :func:`highlight_vertex`, :func:`highlight_edge`
        """
        for i in range(len(path)):
            if i < len(path) - 1:
                e = path[i].edgeto(path[i+1])
                self.highlight_edge(e, **kwargs)
            self.highlight_vertex(path[i], **kwargs)
        plt.show(block=block)

    def highlight_edge(self, edge, scale=1.5, color='r'):
        """
        Highlight an edge in the graph

        :param edge: The edge to highlight
        :type edge: Edge subclass
        :param scale: Overwrite with a line this much bigger than the original,
                      defaults to 1.5
        :type scale: float, optional
        :param color: Overwrite with a line in this color, defaults to 'r'
        :type color: str, optional
        """
        p1 = edge.v1
        p2 = edge.v2
        plt.plot([p1.x, p2.x], [p1.y, p2.y], color=color, linewidth=3 * scale)

    def highlight_vertex(self, node, scale=1.5, color='r'):
        """
        Highlight a vertex in the graph

        :param edge: The vertex to highlight
        :type edge: Vertex subclass
        :param scale: Overwrite with a line this much bigger than the original,
                      defaults to 1.5
        :type scale: float, optional
        :param color: Overwrite with a line in this color, defaults to 'r'
        :type color: str, optional
        """
        plt.plot(node.x, node.y, 'o', color=color, markersize=12 * scale)

    def dotfile(self, filename=None, direction=None):
        """
        Create a GraphViz dot file

        :param filename: filename to save graph to, defaults to None
        :type filename: str, optional

        ``g.dotfile()`` creates the specified file which contains the
        GraphViz code to represent the embedded graph.  By default output
        is to the console

        .. note::

            - The graph is undirected if it is a subclass of ``UGraph``
            - The graph is directed if it is a subclass of ``DGraph``
            - Use ``neato`` rather than dot to get the embedded layout

        .. note:: If ``filename`` is a file object then the file will *not*
            be closed after the GraphViz model is written.
        """
        
        if filename is None:
            f = sys.stdout
        elif isinstance(filename, str):
            f = open(filename, "w")
        else:
            f = filename

        if isinstance(self, DGraph):
            print("digraph {", file=f)
        else:
            print("graph {", file=f)

        if direction is not None:
            print(f"rankdir = {direction}", file=f)

        # add the nodes including name and position
        for node in self:
            if node.coord is None:
                print('  "{:s}"'.format(node.name), file=f)
            else:
                print('  "{:s}" [pos="{:.5g},{:.5g}"]'.format(node.name, node.coord[0], node.coord[1]), file=f)
        print(file=f)
        # add the edges
        for e in self.edges():
            if isinstance(self, DGraph):
                print('  "{:s}" -> "{:s}"'.format(e.v1.name, e.v2.name), file=f)
            else:
                print('  "{:s}" -- "{:s}"'.format(e.v1.name, e.v2.name), file=f)

        print('}', file=f);

        if filename is None or isinstance(filename, str):
            f.close()  # noqa

    def showgraph(self, **kwargs):
        """
        Display a link transform graph in browser
        :param etsbox: Put the link ETS in a box, otherwise an edge label
        :type etsbox: bool
        :param jtype: Arrowhead to node indicates revolute or prismatic type
        :type jtype: bool
        :param static: Show static joints in blue and bold
        :type static: bool
        ``robot.showgraph()`` displays a graph of the robot's link frames
        and the ETS between them.  It uses GraphViz dot.
        The nodes are:
            - Base is shown as a grey square.  This is the world frame origin,
              but can be changed using the ``base`` attribute of the robot.
            - Link frames are indicated by circles
            - ETS transforms are indicated by rounded boxes
        The edges are:
            - an arrow if `jtype` is False or the joint is fixed
            - an arrow with a round head if `jtype` is True and the joint is
              revolute
            - an arrow with a box head if `jtype` is True and the joint is
              prismatic
        Edge labels or nodes in blue have a fixed transformation to the
        preceding link.
        Example::
            >>> import roboticstoolbox as rtb
            >>> panda = rtb.models.URDF.Panda()
            >>> panda.showgraph()
        .. image:: ../figs/panda-graph.svg
            :width: 600
        :seealso: :func:`dotfile`
        """

        # create the temporary dotfile
        dotfile = tempfile.TemporaryFile(mode="w")
        self.dotfile(dotfile, **kwargs)

        # rewind the dot file, create PDF file in the filesystem, run dot
        dotfile.seek(0)
        pdffile = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        subprocess.run("dot -Tpdf", shell=True, stdin=dotfile, stdout=pdffile)

        # open the PDF file in browser (hopefully portable), then cleanup
        webbrowser.open(f"file://{pdffile.name}")
        # time.sleep(1)
        # os.remove(pdffile.name)

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

        .. note::

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
        Adjacency matrix of graph
        
        ``g.adjacency()`` is a matrix (NxN) where N is the number of vertices.
        Element A[i,j] is 1 if vertex i is connected to vertex j, else 0.
        
        .. note::

            - vertices are numbered in their order of creation. A vertex index
              can be resolved to a vertex reference by ``graph[i]``.
            - Matrix is symmetric for an undirected graph
            - Eigenvalues of A are real and are known as the spectrum of the graph.
            - The element A[i,j] can be considered the number of walks of one
              edge from vertex i to vertex j (either zero or one).  The element (i,j)
              of A^N are the number of walks of length N from vertex i to vertex j.

        :seealso: :func:`Laplacian`, :func:`incidence`, :func:`degree`
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
        Incidence matrix of graph
        
        ``g.incidence()`` is a matrix (NxE) where N is the number of vertices
        and E is the number of edges.  Element (i,j) is 1 if vertex i is 
        connected to edge  j.

        .. note::

            - vertices are numbered in their order of creation. A vertex index
              can be resolved to a vertex reference by ``graph[i]``.
            - edges are numbered in the order they appear in ``graph.edges()``.

        :seealso: :func:`Laplacian`, :func:`adjacency`, :func:`degree`
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
            
            lastlabel = None
            for label in range(self.n):
                assignment = False
                for v in self:
                    # find first vertex with no label
                    if v.label is None:
                        # do BFS
                        q = [v]
                        while len(q) > 0:
                            v = q.pop()
                            v.label = label
                            for n in v.neighbours():
                                if n.label is None:
                                    q.append(n)
                        lastlabel = label
                        assignment = True
                        break
                if not assignment:
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

    # def remove(self, v):
    #     # remove edges from neighbour's edge list
    #     for e in v.edges():
    #         next = e.next(v)
    #         next._edges.remove(e)
    #         next._connectivitychange = True  

    #     # remove references from the graph
    #     self._vertexlist.remove(v)
    #     for key, value in self._vertexdict.items():
    #         if value is v:
    #             del self._vertexdict[key]
    #             break

    #     v._edges = []  # remove all references to edges
# --------------------------------------------------------------------------- #

    def path_BFS(self, S, G):
        """
        Breadth-first search for path

        :param S: start vertex
        :type S: Vertex subclass
        :param G: goal vertex
        :type G: Vertex subclass
        :return: list of vertices from S to G inclusive
        :rtype: list of Vertex subclass

        Returns a list of vertices that form a path from vertex ``S`` to
        vertex ``G`` if possible, otherwise return None.
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

    def path_Astar(self, S, G):
        """
        A* search for path

        :param S: start vertex
        :type S: Vertex subclass
        :param G: goal vertex
        :type G: Vertex subclass
        :return: list of vertices from S to G inclusive
        :rtype: list of Vertex subclass

        Returns a list of vertices that form a path from vertex ``S`` to
        vertex ``G`` if possible, otherwise return None.

        The heuristic is the distance metric of the graph, which defaults to
        Euclidean distance.
        """
        S = self[S]
        G = self[G]
        frontier = [S]
        explored = set()
        parent = {}
        g = {S: 0} # cost to come
        f = {S: 0} # evaluation function

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
                    f[n] = g[n] + n.distance(G)  # heuristic
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
    """
    Class for undirected graphs

    .. inheritance-diagram:: UGraph
    """

    def add_vertex(self, coord=None, name=None):
        """
        Add vertex to undirected graph

        :param coord: coordinate for an embedded graph, defaults to None
        :type coord: array-like, optional
        :param name: node name, defaults to "#i"
        :type name: str, optional
        :return: new vertex
        :rtype: UVertex

        - ``g.add_vertex()`` creates a new vertex with optional ``coord`` and
          ``name``.
        - ``g.add_vertex(v)`` takes an instance or subclass of UVertex and adds
          it to the graph
        """
        if isinstance(coord, UVertex):
            node = coord
        else:
            node = UVertex(coord, name)
        super().add_vertex(node)
        return node

class DGraph(PGraph):
    """
    Class for directed graphs

    .. inheritance-diagram:: DGraph
    """

    def add_vertex(self, coord=None, name=None):
        """
        Add vertex to directed graph

        :param coord: coordinate for an embedded graph, defaults to None
        :type coord: array-like, optional
        :param name: node name, defaults to "#i"
        :type name: str, optional
        :return: new vertex
        :rtype: DVertex

        - ``g.add_vertex()`` creates a new vertex with optional ``coord`` and
          ``name``.
        - ``g.add_vertex(v)`` takes an instance or subclass of DVertex and adds
          it to the graph
        """
        if isinstance(coord, DVertex):
            node = coord
        else:
            node = DVertex(coord, name)
        super().add_vertex(node)
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

    .. note::

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

    # def remove(self):
    #     """
    #     Remove edge from graph

    #     ``e.remove()`` removes ``e`` from the graph, but does not delete the
    #     edge object.
    #     """
    #     # remove this edge from the edge list of both end nodes
    #     if self in self.v1._edges:
    #         self.v1._edges.remove(self)
    #     if self in self.v2._edges:
    #         self.v2._edges.remove(self)

    #     # indicate that connectivity has changed
    #     self.v1._connectivitychange = True
    #     self.v2._connectivitychange = True

    #     # remove references to the nodes
    #     self.v1 = None
    #     self.v2 = None

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
        self._graph = None  # reference to owning graph
        # print('Vertex init', type(self))

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
        :param cost: the cost to traverse this edge, required for planning 
                     methods, defaults to None
        :type cost: float, optional
        :param edgedata: reference to arbitrary data associated with the edge,
                         defaults to None
        :type edgedata: Any, optional
        :raises TypeError: vertex types are different subclasses
        :return: the edge connecting the nodes
        :rtype: Edge

        ``v1.connect(v2)`` connects vertex ``v1`` to vertex ``v2``.

        .. note::

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

        self._graph._edges.add(e)
        return e

    def edgeto(self, dest):
        """
        Get edge connecting vertex to specific neighbour

        :param dest: a neigbouring node
        :type dest: ``Vertex`` subclass
        :raises ValueError: ``dest`` is not a neighbour
        :return: the edge from this node to ``dest``
        :rtype: Edge

        .. note::

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

        .. note::

            - For a directed graph the edges are those leaving this vertex
            - For a non-directed graph the edges are those leaving or entering
                this vertex
        """
        return self._edges

    def distance(self, v2):
        return self._graph._metric(v1.coord, v2.coord)

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


    .. inheritance-diagram:: UVertex

    """


    def connect(self, other, **kwargs):
        e = super().connect(other, **kwargs)
        
        self._edges.append(e)
        other._edges.append(e)
        self._graph._edges.add(e)

        return e



class DVertex(Vertex):
    """
    Vertex subclass for directed graphs

    This class can be inherited to provide user objects with graph capability.

    .. inheritance-diagram:: DVertex

    """

    def connect(self, other, **kwargs):
        e = super().connect(other, **kwargs)
        
        self._edges.add(e)
        return e

    def remove(self):
        self._edges = None  # remove all references to edges


