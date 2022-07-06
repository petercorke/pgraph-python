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
        pass  # pragma: no cover

    # @abstractmethod
    # def add_edge(self, v1, v2, cost=None):
    #     pass

    def __init__(self, arg=None, metric=None, heuristic=None, verbose=False):
        # we use a list and a dict, the list respects the order of adding
        self._vertexlist = []
        self._vertexdict = {}
        self._edgelist = set()
        self._verbose = verbose
        if metric is None:
            self.metric = 'L2'
        else:
            self.metric = metric
        if heuristic is None:
            self.heuristic = self.metric
        else:
            self.heuristic = heuristic

    def __str__(self):
        s = f"{self.__class__.__name__}: {self.n} vertices, {self.ne} edges, {self.nc} components"
        return s

    @classmethod
    def Dict(cls, d, reverse=False, copy=False):
        """
        Create graph from parent/child dictionary

        :param d: dictionary that maps from ``Vertex`` subclass to ``Vertex`` subclass
        :type d: dict
        :param reverse: reverse link direction, defaults to False
        :type reverse: bool, optional
        :return: graph
        :rtype: UGraph or DGraph

        Behaves like a constructor for a ``DGraph`` or ``UGraph`` from a
        dictionary that maps vertices to parents.  From this information it
        can create a tree graph.

        By default parent nodes are linked their children. If ``reverse`` is
        True then children are linked to their parents.
        """

        g = cls()

        for vertex, parent in d.items():
            if vertex.name not in g:
                if copy:
                    vertex = vertex.copy(cls)
                g.add_vertex(vertex, name=vertex.name)
            if parent.name not in g:
                if copy:
                    parent = parent.copy(cls)
                g.add_vertex(parent, name=parent.name)

            if reverse:
                g.add_edge(vertex.name, parent.name)
            else:
                g.add_edge(parent.name, vertex.name)

        return g

    @classmethod
    def Adjacency(cls, A, coords=None, names=None):
        """
        Create graph from adjacency matrix

        :param A: adjacency matrix
        :type A: ndarray(N,N)
        :param coords: coordinates of vertices, defaults to None
        :type coords: ndarray(N,M), optional
        :param names: names of vertices, defaults to None
        :type names: list(N) of str, optional

        :return: [description]
        :rtype: [type]

        Create a directed or undirected graph where non-zero elements ``A[i,j]``
        correspond to edges from vertex ``i`` to vertex ``j``.

        .. warning:: For undirected graph ``A`` should be symmetric but this
            is not checked.  Only the upper triangular part is used.
        """

        if A.shape[0] != A.shape[1]:
            raise ValueError('Adjacency matrix must be square')
        if names is not None and len(names) != A.shape[0]:
            raise ValueError('length of names must match dimension of adjacency matrix')
        if coords is not None and coords.shape[0] != A.shape[0]:
                raise ValueError('coords must have same number of rows as adjacency matrix')

        g = cls()

        name = None
        coord = None
        for i in range(A.shape[0]):
            if names is not None:
                name = names[i]
            if coords is not None:
                coord = coords[i, :]
            g.add_vertex(name=name, coord=coord)

        if isinstance(g, UGraph):
            # undirected graph
            for i in range(A.shape[0]):
                for j in range(i+1, A.shape[1]):
                    if A[i, j] > 0:
                        g[i].connect(g[j], cost=A[i,j])
        else:
            # directed graph
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if A[i, j] > 0:
                        if i == j:
                            raise ValueError('loops in graph not supported')
                        g[i].connect(g[j], cost=A[i,j])

        return g


    def copy(self):
        """
        Deepcopy of graph

        :param g: A graph
        :type g: PGraph
        :return: deep copy
        :rtype: PGraph
        """
        return copy.deepcopy(self)

    def add_vertex(self, vertex, name=None):
        """
        Add a vertex to the graph (superclass method)

        :param vertex: vertex to add
        :type vertex: Vertex subclass
        :param name: name of vertex
        :type name: str

        ``G.add_vertex(v)`` add vertex ``v`` to the graph ``G``.

        If the vertex has no name and ``name`` is None give it a default name
        ``#N`` where ``N`` is a consecutive integer.

        The vertex is placed into a dictionary with a key equal to its name.
        """
        if name is None:
            name = vertex.name
        if name is None:
            name = f"#{len(self._vertexlist)}"
        vertex.name = name
        self._vertexlist.append(vertex)
        self._vertexdict[vertex.name] = vertex
        if self._verbose:
            print(f"New vertex {vertex.name}: {vertex.coord}")
        vertex._graph = self
        self._connectivitychange = True

    def add_edge(self, v1, v2, **kwargs):
        """
        Add an edge to the graph (superclass method)

        :param v1: first vertex (start if a directed graph)
        :type v1: Vertex subclass
        :param v2: second vertex (end if a directed graph)
        :type v2: Vertex subclass
        :param kwargs: optional arguments to pass to ``Vertex.connect``
        :return: edge
        :rtype: Edge

        Create an edge between a vertex pair and adds it to the graph.

        This is a graph centric way of creating an edge.  The 
        alternative is the ``connect`` method of a vertex.

        :seealso: :meth:`Edge.connect` :meth:`Vertex.connect`
        """
        if isinstance(v1, str):
            v1 = self[v1]
        elif not isinstance(v1, Vertex):
            raise TypeError('v1 must be Vertex subclass or string name')
        if isinstance(v2, str):
            v2 = self[v2]
        elif not isinstance(v2, Vertex):
            raise TypeError('v2 must be Vertex subclass or string name')
        
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
            x.v1._edgelist.remove(x)
            x.v2._edgelist.remove(x)

            # indicate that connectivity has changed
            x.v1._connectivitychange = True
            x.v2._connectivitychange = True
            self._connectivitychange = True

            # remove references to the vertices
            x.v1 = None
            x.v2 = None

            # remove from list of all edges
            self._edgelist.remove(x)

        elif isinstance(x, Vertex):
            # remove a vertex

            # remove all edges of this vertex
            for edge in copy.copy(x._edgelist):
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
        for e in self._edgelist:
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
        return len(self._edgelist)

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

    def _metricfunc(self, metric):

        def L1(v):
            return np.linalg.norm(v, 1)

        def L2(v):
            return np.linalg.norm(v)
        
        def SE2(v):
            # wrap angle to range [-pi, pi)
            v[2] = (v[2] + np.pi) % (2 * np.pi) - np.pi
            return np.linalg.norm(v)

        if callable(metric):
            return metric
        elif isinstance(metric, str):
            if metric == 'L1':
                return L1
            elif metric == 'L2':
                return L2
            elif metric == 'SE2':
                return SE2
        else:
            raise ValueError('unknown metric')

    @property
    def metric(self):
        """
        Get the distance metric for graph

        :return: distance metric
        :rtype: callable

        This is a function of a vector and returns a scalar.
        """
        return self._metric

    @metric.setter
    def metric(self, metric):
        r"""
        Set the distance metric for graph

        :param metric: distance metric
        :type metric: callable or str

        This is a function of a vector and returns a scalar.  It can be 
        user defined function or a string:

        - 'L1' is the norm :math:`L_1 = \Sigma_i | v_i |`
        - 'L2' is the norm :math:`L_2 = \sqrt{ \Sigma_i v_i^2}`
        - 'SE2' is a mixed norm for vectors :math:`(x, y, \theta)` and
            is :math:`\sqrt{x^2 + y^2 + \bar{\theta}^2}` where :math:`\bar{\theta}`
            is :math:`\theta` wrapped to the interval :math:`[-\pi, \pi)`
        
        The metric is used by :meth:`closest` and :meth:`distance`
        """
        self._metric = self._metricfunc(metric)

    @property
    def heuristic(self):
        """
        Get the heuristic distance metric for graph

        :return: heuristic distance metric
        :rtype: callable

        This is a function of a vector and returns a scalar.
        """
        return self._heuristic

    @heuristic.setter
    def heuristic(self, heuristic):
        r"""
        Set the heuristic distance metric for graph

        :param metric: heuristic distance metric
        :type metric: callable or str

        This is a function of a vector and returns a scalar.  It can be 
        user defined function or a string:

        - 'L1' is the norm :math:`L_1 = \Sigma_i | v_i |`
        - 'L2' is the norm :math:`L_2 = \sqrt{ \Sigma_i v_i^2}`
        - 'SE2' is a mixed norm for vectors :math:`(x, y, \theta)` and
            is :math:`\sqrt{x^2 + y^2 + \bar{\theta}^2}` where :math:`\bar{\theta}`
            is :math:`\theta` wrapped to the interval :math:`[-\pi, \pi)`

        The heuristic distance is only used by the A* planner :meth:`path_Astar`.
        """
        self._heuristic = self._metricfunc(heuristic)

    def __repr__(self):
        s = []
        for node in self:
            ss = f"{node.name} at {node.coord}"
            if node.label is not None:
                ss += " component={node.label}"
            s.append(ss)
        return '\n'.join(s)

    def __getitem__(self, i):
        """
        Get vertex (superclass method)

        :param i: vertex description
        :type i: int or str
        :return: the referenced vertex
        :rtype: Vertex subclass

        Retrieve a vertex by index or name:

        -``g[i]`` is the i'th vertex in the graph.  This reflects the order of 
         addition to the graph.
        -``g[s]`` is vertex named ``s``
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

    def __contains__(self, item):
        """
        Test if vertex in graph

        :param item: vertex or name of vertex
        :type item: Vertex subclass or str
        :return: true if vertex exists in the graph
        :rtype: bool

        - ``'name' in graph`` is true if a vertex named ``'name'`` exists in the
          graph.
        - ``v in graph`` is true if the vertex reference ``v`` exists in the
          graph.

        """
        if isinstance(item, str):
            return item in self._vertexdict
        elif isinstance(item, Vertex):
            return item in self._vertexdict.values()

    def closest(self, coord):
        """
        Vertex closest to point

        :param coord: coordinates of a point
        :type coord: ndarray(n)
        :return: closest vertex
        :rtype: Vertex subclass

        Returns the vertex closest to the given point. Distance is computed
        according to the graph's metric.

        :seealso: :meth:`metric`
        """
        min_dist = np.Inf

        for vertex in self:
            d = self.metric(vertex.coord - coord)
            if d < min_dist:
                min_dist = d
                min_which = vertex
        
        return min_which, min_dist

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

        :seealso: :meth:`Vertex.edges`
        """
        return self._edgelist

    def plot(self, colorcomponents=True, vertex=None, edge=None, text={}, block=False, ax=None):
        """
        Plot the graph

        :param vertex: vertex format, defaults to 12pt o-marker
        :type vertex: dict, optional
        :param edge: edge format, defaults to None
        :type edge: dict, optional
        :param text: text label format, defaults to None
        :type text: False or dict, optional
        :param colorcomponents: color nodes and edges by component, defaults to None
        :type color: bool, optional
        :param block: block until figure is dismissed, defaults to True
        :type block: bool, optional

        The graph is plotted using matplotlib.

        If ``colorcomponents`` is True then each component is assigned a unique
        color.  ``vertex`` and ``edge`` cannot include a color keyword.

        If ``text`` is a dict it is used to format text labels for the vertices
        which are the vertex names.  If ``text`` is None default formatting is
        used.  If ``text`` is False no labels are added.
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

        if colorcomponents:
            color = plt.cm.coolwarm(np.linspace(0, 1, self.nc))

        if ax is None:
            ax = plt.axes()
        for c in range(self.nc):
            # for each component
            for node in self.component(c):
                if text is not False:
                    ax.text(node.x, node.y, "  " + node.name, **text)
                if colorcomponents:
                    ax.plot(node.x, node.y, color=color[c, :], **vertex)
                    for v in node.neighbours():
                        ax.plot([node.x, v.x], [node.y, v.y],
                                color=color[c, :], **edge)
                else:
                    x.plot(node.x, node.y, **vertex)
                    for v in node.neighbours():
                        ax.plot([node.x, v.x], [node.y, v.y], **edge)
        # if nc > 1:
        #     # add a colorbar
        #     plt.colorbar()
        ax.grid(True)
        plt.show(block=block)

    def highlight_path(self, path, block=False, **kwargs):
        """
        Highlight a path through the graph

        :param path: [description]
        :type path: [type]
        :param block: [description], defaults to True
        :type block: bool, optional

        The vertices and edges along the path are overwritten with a different
        size/width and color.

        :seealso: :meth:`highlight_vertex` :meth:`highlight_edge`
        """
        for i in range(len(path)):
            if i < len(path) - 1:
                e = path[i].edgeto(path[i+1])
                self.highlight_edge(e, **kwargs)
            self.highlight_vertex(path[i], **kwargs)
        plt.show(block=block)

    def highlight_edge(self, edge, scale=2, color='r', alpha=0.5):
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
        plt.plot([p1.x, p2.x], [p1.y, p2.y], color=color,
                 linewidth=3 * scale, alpha=alpha)

    def highlight_vertex(self, vertex, scale=2, color='r', alpha=0.5):
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
        if isinstance(vertex, Iterable):
            for n in vertex:
                if isinstance(n, str):
                    n = self[n]
                plt.plot(n.x, n.y, 'o', color=color,
                         markersize=12 * scale, alpha=alpha)
        else:
            plt.plot(vertex.x, vertex.y, 'o', color=color,
                     markersize=12 * scale, alpha=alpha)

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

        # add the vertices including name and position
        for vertex in self:
            if vertex.coord is None:
                print('  "{:s}"'.format(vertex.name), file=f)
            else:
                print('  "{:s}" [pos="{:.5g},{:.5g}"]'.format(
                    vertex.name, vertex.coord[0], vertex.coord[1]), file=f)
        print(file=f)
        # add the edges
        for e in self.edges():
            if isinstance(self, DGraph):
                print('  "{:s}" -> "{:s}"'.format(e.v1.name, e.v2.name), file=f)
            else:
                print('  "{:s}" -- "{:s}"'.format(e.v1.name, e.v2.name), file=f)

        print('}', file=f)

        if filename is None or isinstance(filename, str):
            f.close()  # noqa

    def showgraph(self, **kwargs):
        """
        Display a link transform graph in browser
        :param etsbox: Put the link ETS in a box, otherwise an edge label
        :type etsbox: bool
        :param jtype: Arrowhead to vertex indicates revolute or prismatic type
        :type jtype: bool
        :param static: Show static joints in blue and bold
        :type static: bool
        ``robot.showgraph()`` displays a graph of the robot's link frames
        and the ETS between them.  It uses GraphViz dot.
        The vertices are:
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
        Edge labels or vertices in blue have a fixed transformation to the
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
            return len(self.edges()) / self.n
        elif isinstance(self, UGraph):
            return 2 * len(self.edges()) / self.n

# --------------------------------------------------------------------------- #

    # MATRIX REPRESENTATIONS

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

        :seealso: :meth:`adjacency` :meth:`incidence` :meth:`degree`
        """
        return self.degree() - (self.adjacency() > 0)

    def connectivity(self, vertices=None):
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
        if vertices is None:
            vertices = self
        for n in vertices:
            c.append(len(n._edgelist))
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

        return np.diag(self.connectivity())

    def adjacency(self):
        """
        Adjacency matrix of graph

        :returns: adjacency matrix
        :rtype: ndarray(n,n)

        The elements of the adjacency matrix ``A[i,j]`` are 1 if vertex ``i`` is
        connected to vertex ``j``, else 0.

        .. note::

            - vertices are numbered in their order of creation. A vertex index
              can be resolved to a vertex reference by ``graph[i]``.
            - for an undirected graph the matrix is symmetric
            - Eigenvalues of ``A`` are real and are known as the spectrum of the graph.
            - The element ``A[i,j]`` can be considered the number of walks of length one
              edge from vertex ``i`` to vertex ``j`` (either zero or one).
            - If ``Ak = A ** k`` the element ``Ak[i,j]`` is the number of
              walks of length ``k`` from vertex ``i`` to vertex ``j``.

        :seealso: :meth:`Laplacian` :meth:`incidence` :meth:`degree`
        """
        # create a dict mapping vertex to an id
        vdict = {}
        for i, vert in enumerate(self):
            vdict[vert] = i

        A = np.zeros((self.n, self.n))
        for vertex in self:
            for n in vertex.neighbours():
                A[vdict[vertex], vdict[n]] = 1
        return A

    def incidence(self):
        """
        Incidence matrix of graph

        :returns: incidence matrix
        :rtype: ndarray(n,ne)

        The elements of the incidence matrix ``I[i,j]`` are 1 if vertex ``i`` is
        connected to edge ``j``, else 0.

        .. note::

            - vertices are numbered in their order of creation. A vertex index
              can be resolved to a vertex reference by ``graph[i]``.
            - edges are numbered in the order they appear in ``graph.edges()``.

        :seealso: :meth:`Laplacian` :meth:`adjacency` :meth:`degree`
        """
        edges = self.edges()
        I = np.zeros((self.n, len(edges)))

        # create a dict mapping edge to an id
        edict = {}
        for i, edge in enumerate(edges):
            edict[edge] = i

        for i, vertex in enumerate(self):
            for i, e in enumerate(vertex.edges()):
                I[i, edict[e]] = 1

        return I

    def distance(self):
        """
        Distance matrix of graph

        :return: distance matrix
        :rtype: ndarray(n,n)

        The elements of the distance matrix ``D[i,j]`` is the edge cost of moving
        from vertex ``i`` to vertex ``j``. It is zero if the vertices are not
        connected.
        """
        # create a dict mapping vertex to an id
        vdict = {}
        for i, vert in enumerate(self):
            vdict[vert] = i

        A = np.zeros((self.n, self.n))
        for v1 in self:
            for v2, edge in v1.incidences():
                A[vdict[v1], vdict[v2]] = edge.cost
        return A

    # GRAPH COMPONENTS

    def _graphcolor(self):
        """
        Color the graph

        Performs a depth-first labeling operation, assigning the ``label`` 
        attribute of every vertex with a sequential integer starting from 0.

        This method checks the ``_connectivitychange`` attribute of all nodes
        and if any are True it will perform the coloring operation. This flag
        is set True by any operation that adds or removes a node or edge.

        :seealso: :meth:`nc`
        """
        if self._connectivitychange or any([n._connectivitychange for n in self]):

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
        All vertices in specified graph component

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
    #         next._edgelist.remove(e)
    #         next._connectivitychange = True

    #     # remove references from the graph
    #     self._vertexlist.remove(v)
    #     for key, value in self._vertexdict.items():
    #         if value is v:
    #             del self._vertexdict[key]
    #             break

    #     v._edgelist = []  # remove all references to edges
# --------------------------------------------------------------------------- #

    def path_BFS(self, S, G, verbose=False, summary=False):
        """
        Breadth-first search for path

        :param S: start vertex
        :type S: Vertex subclass
        :param G: goal vertex
        :type G: Vertex subclass
        :return: list of vertices from S to G inclusive, path length
        :rtype: list of Vertex subclass, float

        Returns a list of vertices that form a path from vertex ``S`` to
        vertex ``G`` if possible, otherwise return None.

        """
        if isinstance(S, str):
            S = self[S]
        elif not isinstance(S, Vertex):
            raise TypeError('start must be Vertex subclass or string name')
        if isinstance(G, str):
            G = self[G]
        elif not isinstance(S, Vertex):
            raise TypeError('goal must be Vertex subclass or string name')

        # we use lists not sets since the order is instructive in verbose
        # mode, really need ordered sets...
        frontier = [S]
        explored = []
        evaluation = [None for i in range(self.n)]
        parent = {}
        done = False

        while frontier:
            if verbose:
                print()
                print('FRONTIER:', ", ".join([v.name for v in frontier]))
                print('EXPLORED:', ", ".join([v.name for v in explored]))

            x = frontier.pop(0)
            if verbose:
                print('   expand', x.name)

            # expand the vertex
            for n in x.neighbours():
                if n is G:
                    if verbose:
                        print('     goal', n.name, 'reached')
                    parent[n] = x
                    done = True
                    break
                if n not in frontier and n not in explored:
                    # add it to the frontier
                    frontier.append(n)
                    if verbose:
                        print('      add', n.name, 'to the frontier')
                    parent[n] = x
            if done:
                break
            explored.append(x)
            if verbose:
                print('     move', x.name, ' to the explored list')
        else:
            # no path
            return None

        # reconstruct the path from start to goal
        x = G
        path = [x]
        length = 0

        while x is not S:
            p = parent[x]
            length += x.edgeto(p).cost
            path.insert(0, p)
            x = p

        if summary or verbose:
            print(
                f"{len(explored)} vertices explored, {len(frontier)} remaining on the frontier")

        return path, length

    def path_UCS(self, S, G, verbose=False, summary=False):
        """
        Uniform cost search for path

        :param S: start vertex
        :type S: Vertex subclass
        :param G: goal vertex
        :type G: Vertex subclass
        :return: list of vertices from S to G inclusive, path length, tree
        :rtype: list of Vertex subclass, float, dict

        Returns a list of vertices that form a path from vertex ``S`` to
        vertex ``G`` if possible, otherwise return None.

        The search tree is returned as dict that maps a vertex to its parent.

        The heuristic is the distance metric of the graph, which defaults to
        Euclidean distance.
        """
        if isinstance(S, str):
            S = self[S]
        elif not isinstance(S, Vertex):
            raise TypeError('start must be Vertex subclass or string name')
        if isinstance(G, str):
            G = self[G]
        elif not isinstance(S, Vertex):
            raise TypeError('goal must be Vertex subclass or string name')

        frontier = [S]
        explored = []
        parent = {}
        f = {S: 0}  # evaluation function

        while frontier:
            if verbose:
                print()
                print('FRONTIER:', ", ".join(
                    [f"{v.name}({f[v]:.0f})" for v in frontier]))
                print('EXPLORED:', ", ".join([v.name for v in explored]))

            i = np.argmin([f[n] for n in frontier])  # minimum f in frontier
            x = frontier.pop(i)
            if verbose:
                print('   expand', x.name)
            if x is G:
                break
            # expand the vertex
            for n, e in x.incidences():
                fnew = f[x] + e.cost
                if n not in frontier and n not in explored:
                    # add it to the frontier
                    parent[n] = x
                    f[n] = fnew
                    frontier.append(n)
                    if verbose:
                        print('      add', n.name, 'to the frontier')

                elif n in frontier:
                    # neighbour is already in the frontier
                    # cost of path via x is lower that previous, reparent it
                    if fnew < f[n]:
                        if verbose:
                            print(
                                f" reparent {n.name}: cost {fnew} via {x.name} is less than cost {f[n]} via {parent[n].name}, change parent from {parent[n].name} to {x.name} ")
                        f[n] = fnew
                        parent[n] = x

            explored.append(x)
            if verbose:
                print('     move', x.name, ' to the explored list')
        else:
            # no path
            return None

        # reconstruct the path from start to goal
        x = G
        path = [x]
        length = 0

        while x is not S:
            p = parent[x]
            length += p.edgeto(x).cost
            path.insert(0, p)
            x = p

        if summary or verbose:
            print(
                f"{len(explored)} vertices explored, {len(frontier)} remaining on the frontier")

        return path, length, parent

    def path_Astar(self, S, G, verbose=False, summary=False):
        """
        A* search for path

        :param S: start vertex
        :type S: Vertex subclass
        :param G: goal vertex
        :type G: Vertex subclass
        :return: list of vertices from S to G inclusive, path length, tree
        :rtype: list of Vertex subclass, float, dict

        Returns a list of vertices that form a path from vertex ``S`` to
        vertex ``G`` if possible, otherwise return None.

        The search tree is returned as dict that maps a vertex to its parent.

        The heuristic is the distance metric of the graph, which defaults to
        Euclidean distance.

        :seealso: :meth:`heuristic`
        """
        if isinstance(S, str):
            S = self[S]
        elif not isinstance(S, Vertex):
            raise TypeError('start must be Vertex subclass or string name')
        if isinstance(G, str):
            G = self[G]
        elif not isinstance(S, Vertex):
            raise TypeError('goal must be Vertex subclass or string name')

        frontier = [S]
        explored = []
        parent = {}
        g = {S: 0}  # cost to come
        f = {S: 0}  # evaluation function

        while frontier:
            if verbose:
                print()
                print('FRONTIER:', ", ".join(
                    [f"{v.name}({f[v]:.0f})" for v in frontier]))
                print('EXPLORED:', ", ".join([v.name for v in explored]))

            i = np.argmin([f[n] for n in frontier])  # minimum f in frontier
            x = frontier.pop(i)
            if verbose:
                print('   expand', x.name)
            if x is G:
                break
            # expand the vertex
            for n, e in x.incidences():
                if n not in frontier and n not in explored:
                    # add it to the frontier
                    frontier.append(n)
                    parent[n] = x
                    g[n] = g[x] + e.cost  # update cost to come
                    f[n] = g[n] + n.heuristic_distance(G)  # heuristic
                    if verbose:
                        print('      add', n.name, 'to the frontier')
                elif n in frontier:
                    # neighbour is already in the frontier
                    gnew = g[x] + e.cost
                    if gnew < g[n]:
                        # cost of path via x is lower that previous, reparent it
                        if verbose:
                            print(
                                f" reparent {n.name}: cost {gnew} via {x.name} is less than cost {g[n]} via {parent[n].name}, change parent from {parent[n].name} to {x.name} ")
                        g[n] = gnew
                        f[n] = g[n] + n.heuristic_distance(G)  # heuristic

                        parent[n] = x  # reparent

            explored.append(x)
            if verbose:
                print('     move', x.name, ' to the explored list')

        else:
            # no path
            return None

        # reconstruct the path from start to goal
        x = G
        path = [x]
        length = 0

        while x is not S:
            p = parent[x]
            length += p.edgeto(x).cost
            path.insert(0, p)
            x = p

        if summary or verbose:
            print(
                f"{len(explored)} vertices explored, {len(frontier)} remaining on the frontier")

        return path, length, parent


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
        :param name: vertex name, defaults to "#i"
        :type name: str, optional
        :return: new vertex
        :rtype: UVertex

        - ``g.add_vertex()`` creates a new vertex with optional ``coord`` and
          ``name``.
        - ``g.add_vertex(v)`` takes an instance or subclass of UVertex and adds
          it to the graph
        """
        if isinstance(coord, UVertex):
            vertex = coord
        else:
            vertex = UVertex(coord)
        super().add_vertex(vertex, name=name)
        return vertex

    @classmethod
    def vertex_copy(self, vertex):
        return DVertex(coord=vertex.coord, name=vertex.name)
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
        :param name: vertex name, defaults to "#i"
        :type name: str, optional
        :return: new vertex
        :rtype: DVertex

        - ``g.add_vertex()`` creates a new vertex with optional ``coord`` and
          ``name``.
        - ``g.add_vertex(v)`` takes an instance or subclass of DVertex and adds
          it to the graph
        """
        if isinstance(coord, DVertex):
            vertex = coord
        else:
            vertex = DVertex(coord)
        super().add_vertex(vertex, name=name)
        return vertex

    @classmethod
    def vertex_copy(self, vertex):
        return DVertex(coord=vertex.coord, name=vertex.name)

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

    An Edge points to a pair of vertices.  At ``connect`` time the vertices
    get references back to the Edge object.

    ``graph.add_edge(v1, v2)`` calls ``v1.connect(v2)``
    """

    def __init__(self, v1=None, v2=None, cost=None, data=None):
        """
        Create an edge object

        :param v1: start of the edge, defaults to None
        :type v1: Vertex subclass, optional
        :param v2: end of the edge, defaults to None
        :type v2: Vertex subclass, optional
        :param cost: edge cost, defaults to None
        :type cost: any, optional
        :param data: edge data, defaults to None
        :type data: any, optional

        Creates an edge but does not connect it to the vertices or add it to the
        graph.

        If vertices are given, and have associated coordinates, the edge cost
        will be computed according to the distance measure associated with the
        graph.

        ``data`` is a way of associating any object with the edge, its value
        can be found as the ``.data`` attribute of the edge.  An alternative
        approach is to subclass the ``Edge`` class.

        .. note:: To compute edge cost from the vertices, the vertices must have
            been added to the graph.

        :seealso: :meth:`Edge.connect` :meth:`Vertex.connect`
        """
        self.v1 = v1
        self.v2 = v2

        self.data = data

        # try to compute edge cost as metric distance if not given
        if cost is not None:
            self.cost = cost
        elif not (v1 is None or v1.coord is None or v2 is None or v2.coord is None):
            self.cost = v1._graph.metric(v1.coord - v2.coord)
        else:
            self.cost = None


    def __repr__(self):
        return str(self)

    def __str__(self):

        s = f"Edge {self.v1} -- {self.v2}, cost={self.cost}"
        if self.data is not None:
            s += f" data={self.data}"
        return s

    def connect(self, v1, v2):
        """
        Add edge to the graph

        :param v1: start of the edge
        :type v1: Vertex subclass
        :param v2: end of the edge
        :type v2: Vertex subclass

        The edge is added to the graph and connects vertices ``v1`` and ``v2``.
        
        .. note:: The vertices must already be added to the graph.
        """

        if v1._graph is None:
            raise ValueError('vertex v1 does not belong to a graph')
        if v2._graph is None:
            raise ValueError('vertex v2 does not belong to a graph')
        if not v1._graph is v2._graph:
            raise ValueError('vertices must belong to the same graph')

        # connect edge to its vertices
        self.v1 = v1
        self.v2 = v2

        # tell the vertices to add edge to their edgelists as appropriate for
        # DGraph or UGraph
        v1.connect(v2, edge=self)


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

    def vertices(self):
        raise DeprecationWarning('use endpoints instead')

    @property
    def endpoints(self):
        return [self.v1, self.v2]

    # def remove(self):
    #     """
    #     Remove edge from graph

    #     ``e.remove()`` removes ``e`` from the graph, but does not delete the
    #     edge object.
    #     """
    #     # remove this edge from the edge list of both end vertices
    #     if self in self.v1._edgelist:
    #         self.v1._edgelist.remove(self)
    #     if self in self.v2._edgelist:
    #         self.v2._edgelist.remove(self)

    #     # indicate that connectivity has changed
    #     self.v1._connectivitychange = True
    #     self.v2._connectivitychange = True

    #     # remove references to the vertices
    #     self.v1 = None
    #     self.v2 = None

# ========================================================================== #


class Vertex:
    """
    Superclass for vertices of directed and non-directed graphs.

    Each vertex has:
        - ``name``
        - ``label`` an int indicating which graph component contains it
        - ``_edgelist`` a list of edge objects that connect this vertex to others
        - ``coord`` the coordinate in an embedded graph (optional)
    """

    def __init__(self, coord=None, name=None):
        self._edgelist = []
        if coord is None:
            self.coord = None
        else:
            self.coord = np.r_[coord]
        self.name = name
        self.label = None
        self._connectivitychange = True
        self._edgelist = []
        self._graph = None  # reference to owning graph
        # print('Vertex init', type(self))

    def __str__(self):
        return f"[{self.name:s}]"

    def __repr__(self):
        coord = ', '.join([f"{x:.4g}" for x in self.coord])
        return f"{self.__class__.__name__}[{self.name:s}, coord=({coord})]"

    def copy(self, cls=None):
        if cls is not None:
            return cls.vertex_copy(self)
        else:
            return self.__class__(coord=self.coord, name=self.name)

    def neighbours(self):
        """
        Neighbours of a vertex

        ``v.neighbours()`` is a list of neighbour of the vertex object ``v``.

        .. note:: For a directed graph the neighbours are those on edges leaving this vertex
        """
        return [e.next(self) for e in self._edgelist]

    def isneighbour(self, vertex):
        """
        Test if vertex is a neigbour

        :param vertex: vertex reference
        :type vertex: Vertex subclass
        :return: true if a neighbour
        :rtype: bool

        For a directed graph this is true only if the edge is from ``self`` to
        ``vertex``.
        """
        return vertex in [e.next(self) for e in self._edgelist]

    def incidences(self):
        """
        Neighbours and edges of a vertex

        ``v.incidences()`` is a generator that returns a list of incidences, 
        tuples of (vertex, edge) for all neighbours of the vertex ``v``.

        .. note:: For a directed graph the edges are those leaving this vertex
        """
        return [(e.next(self), e) for e in self._edgelist]

    def connect(self, dest, edge=None, cost=None, data=None):
        """
        Connect two vertices with an edge

        :param dest: The vertex to connect to
        :type dest: ``Vertex`` subclass
        :param edge: Use this as the edge object, otherwise a new ``Edge``
                     object is created from the vertices being connected,
                     and the ``cost`` and ``edge`` parameters, defaults to None
        :type edge: ``Edge`` subclass, optional
        :param cost: the cost to traverse this edge, defaults to None
        :type cost: float, optional
        :param data: reference to arbitrary data associated with the edge,
                     defaults to None
        :type data: Any, optional
        :raises TypeError: vertex types are different subclasses
        :return: the edge connecting the vertices
        :rtype: Edge

        ``v1.connect(v2)`` connects vertex ``v1`` to vertex ``v2``.

        .. note::

            - If the vertices subclass ``UVertex`` the edge is undirected, and if
              they subclass ``DVertex`` the edge is directed.
            - Vertices must both be of the same ``Vertex`` subclass

        :seealso: :meth:`Edge`
        """

        if not dest.__class__.__bases__[0] is self.__class__.__bases__[0]:
            raise TypeError('must connect vertices of same type')
        elif isinstance(edge, Edge):
            e = edge
        else:
            e = Edge(self, dest, cost=cost, data=data)

        self._graph._edgelist.add(e)
        self._graph._connectivitychange = True
        self._connectivitychange = True

        return e

    def edgeto(self, dest):
        """
        Get edge connecting vertex to specific neighbour

        :param dest: a neigbouring vertex
        :type dest: ``Vertex`` subclass
        :raises ValueError: ``dest`` is not a neighbour
        :return: the edge from this vertex to ``dest``
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

        :return: List of all edges leaving this vertex
        :rtype: list of Edge

        .. note::

            - For a directed graph the edges are those leaving this vertex
            - For a non-directed graph the edges are those leaving or entering
                this vertex
        """
        return self._edgelist

    def heuristic_distance(self, v2):
        return self._graph.heuristic(self.coord - v2.coord)

    def distance(self, coord):
        """
        Distance from vertex to point

        :param coord: coordinates of the point
        :type coord: ndarray(n)
        :return: distance
        :rtype: float

        Distance is computed according to the graph's metric.

        :seealso: :meth:`metric`
        """
        return self._graph.metric(self.coord - coord)

    @property
    def degree(self):
        """
        Degree of vertex

        :return: degree of the vertex
        :rtype: int

        Returns the number of edges connected to the vertex.

        .. note:: For a ``DGraph`` only outgoing edges are considered.

        :seealso: :meth:`edges`
        """
        return len(self.edges())

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

        if isinstance(other, Vertex):
            e = super().connect(other, **kwargs)
        elif isinstance(other, Edge):
            e = super().connect(edge=other)
        else:
            raise TypeError('bad argument')

        # e = super().connect(other, **kwargs)

        self._edgelist.append(e)
        other._edgelist.append(e)
        self._graph._edgelist.add(e)

        return e


class DVertex(Vertex):
    """
    Vertex subclass for directed graphs

    This class can be inherited to provide user objects with graph capability.

    .. inheritance-diagram:: DVertex

    """

    def connect(self, other, **kwargs):
        if isinstance(other, Vertex):
            e = super().connect(other, **kwargs)
        elif isinstance(other, Edge):
            e = super().connect(edge=other)
        else:
            raise TypeError('bad argument')

        self._edgelist.append(e)
        return e

    def remove(self):
        self._edgelist = None  # remove all references to edges
