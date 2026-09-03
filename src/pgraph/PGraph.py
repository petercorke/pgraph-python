from __future__ import annotations

from abc import ABC, abstractmethod
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections.abc import Iterable, Iterator
import tempfile
import subprocess
import webbrowser
from typing import Any, Callable, ClassVar
from numpy.typing import ArrayLike, NDArray

from spatialmath.base.graphics import axes_logic


class _BaseGraph(ABC):

    #: concrete vertex type for this graph kind, provided by :class:`UGraph`
    #: and :class:`DGraph` -- lets :meth:`add_vertex` and :meth:`vertex_copy`
    #: be defined once here rather than duplicated per subclass.
    _vertex_cls: ClassVar[type[BaseVertex]]

    def __init__(
        self,
        metric: Callable[[NDArray], float] | str | None = None,
        heuristic: Callable[[NDArray], float] | str | None = None,
        verbose: bool = False,
    ):
        """
        Abstract base class for graphs

        :param metric: distance metric, defaults to "L2"
        :type metric: callable or str, optional
        :param heuristic: heuristic distance metric for A*, defaults to the
            same as ``metric``
        :type heuristic: callable or str, optional
        :param verbose: print diagnostic information as vertices/edges are
            added, defaults to False

        This is the common base class of :class:`UGraph` and :class:`DGraph`
        and should not be instantiated directly.

        :seealso: :class:`UGraph` :class:`DGraph`
        """
        # we use a list and a dict, the list respects the order of adding
        self._vertexlist: list[BaseVertex] = []
        self._vertexdict: dict[str, BaseVertex] = {}
        self._edgelist: set[Edge] = set()
        self._verbose = verbose
        self._ncomponents = 0
        self._connectivitychange = False
        if metric is None:
            self.metric = "L2"
        else:
            self.metric = metric
        if heuristic is None:
            self.heuristic = self.metric
        else:
            self.heuristic = heuristic

    def __str__(self) -> str:
        """
        Human-readable summary of the graph

        :return: one-line summary of vertex/edge/component counts
        :rtype: str

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v3 = g.add_vertex(coord=[2,2], name='v3')
            >>> g.add_edge(v1, v2)
            >>> g.add_edge(v2, v3)
            >>> str(g)

        :seealso: :meth:`show`
        """
        s = f"{self.__class__.__name__}: {self.n} {'vertex' if self.n==1 else 'vertices'}, {self.ne} edge{'s'[:self.ne^1]}, {self.nc} component{'s'[:self.nc^1]}"
        return s

    def __repr__(self) -> str:
        # NOTE: this is shadowed by the __repr__ defined further below in
        # this class, which is the one actually used -- kept as-is here to
        # avoid changing behaviour as part of a typing-only pass.
        return str(self)

    @classmethod
    def Dict(cls, d: dict, reverse: bool = False) -> _BaseGraph:
        """
        Create graph from parent/child dictionary

        :param d: dictionary that maps from ``BaseVertex`` subclass to ``BaseVertex`` subclass
        :type d: dict
        :param reverse: reverse link direction, defaults to False
        :return: graph
        :rtype: UGraph or DGraph

        Behaves like a constructor for a ``DGraph`` or ``UGraph`` from a
        dictionary that maps vertices to parents.  From this information it
        can create a tree graph.

        By default parent vertices are linked their children. If ``reverse`` is
        True then children are linked to their parents.

        :seealso: :meth:`Adjacency`
        """

        g = cls()

        for vertex, parent in d.items():
            if isinstance(vertex, str):
                vertex_name = vertex
            else:
                vertex_name = vertex.name

            if vertex_name in g:
                vertex = g[vertex_name]
            else:
                vertex = g.add_vertex(name=vertex_name)

            if isinstance(parent, str):
                parent_name = parent
            else:
                parent_name = parent.name
            if parent_name in g:
                parent = g[parent_name]
            else:
                parent = g.add_vertex(name=parent_name)

            if reverse:
                g.add_edge(vertex, parent)
            else:
                g.add_edge(parent, vertex)

        return g

    @classmethod
    def Adjacency(
        cls,
        A: NDArray,
        coords: NDArray | None = None,
        names: list[str] | None = None,
    ) -> _BaseGraph:
        """
        Create graph from adjacency matrix

        :param A: adjacency matrix
        :type A: ndarray(N,N)
        :param coords: coordinates of vertices, defaults to None
        :type coords: ndarray(N,M), optional
        :param names: names of vertices, defaults to None
        :type names: list(N) of str, optional
        :return: graph
        :rtype: UGraph or DGraph

        Create a directed or undirected graph where non-zero elements ``A[i,j]``
        correspond to edges from vertex ``i`` to vertex ``j``.

        .. warning:: For undirected graph ``A`` should be symmetric but this
            is not checked.  Only the upper triangular part is used.

        :seealso: :meth:`Dict` :meth:`adjacency`
        """

        if A.shape[0] != A.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if names is not None and len(names) != A.shape[0]:
            raise ValueError("length of names must match dimension of adjacency matrix")
        if coords is not None and coords.shape[0] != A.shape[0]:
            raise ValueError("coords must have same number of rows as adjacency matrix")

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
                for j in range(i + 1, A.shape[1]):
                    if A[i, j] > 0:
                        g[i].connect(g[j], cost=A[i, j])
        else:
            # directed graph
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if A[i, j] > 0:
                        if i == j:
                            raise ValueError("loops in graph not supported")
                        g[i].connect(g[j], cost=A[i, j])

        return g

    def copy(self) -> _BaseGraph:
        """
        Deepcopy of graph

        :return: deep copy
        :rtype: _BaseGraph
        """
        return copy.deepcopy(self)

    def add_vertex(
        self, coord: ArrayLike | BaseVertex | None = None, name: str | None = None
    ) -> BaseVertex:
        """
        Add a vertex to the graph

        :param coord: coordinate for an embedded graph, or an existing vertex
            of this graph's own kind (``UVertex`` for :class:`UGraph`,
            ``DVertex`` for :class:`DGraph`) to add as-is, defaults to None
        :type coord: array-like or BaseVertex subclass, optional
        :param name: name of vertex, defaults to "#i"
        :type name: str, optional
        :raises TypeError: ``coord`` is a ``BaseVertex`` of the wrong kind
        :return: the added vertex
        :rtype: BaseVertex subclass

        - ``g.add_vertex()`` creates a new vertex with optional ``coord`` and
          ``name``.
        - ``g.add_vertex(v)`` takes an instance or subclass of this graph's
          own vertex kind and adds it to the graph.

        If the vertex has no name and ``name`` is None give it a default name
        ``#N`` where ``N`` is a consecutive integer.

        The vertex is placed into a dictionary with a key equal to its name.

        This single implementation, shared by :class:`UGraph` and
        :class:`DGraph`, is parameterized by each subclass's
        :attr:`_vertex_cls` rather than duplicated per subclass -- see
        :doc:`policy` for why that matters.

        :seealso: :meth:`vertex_copy`
        """
        if isinstance(coord, self._vertex_cls):
            vertex = coord
        elif isinstance(coord, BaseVertex):
            raise TypeError(
                f"expecting {self._vertex_cls.__name__} or coordinate data, "
                f"got {type(coord).__name__}"
            )
        else:
            vertex = self._vertex_cls(coord, name=name)

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
        return vertex

    @classmethod
    def vertex_copy(cls, vertex: BaseVertex) -> BaseVertex:
        """
        Copy a vertex for use in a new graph of this kind

        :param vertex: vertex to copy
        :type vertex: BaseVertex subclass
        :return: new, unconnected vertex with the same coordinate and name
        :rtype: BaseVertex subclass

        A vertex can only belong to a single graph, so this method is used to
        create a new vertex with the same name and coordinates for inclusion
        in a new graph -- of ``cls``'s own vertex kind, per :attr:`_vertex_cls`.

        :seealso: :meth:`BaseVertex.copy`
        """
        return cls._vertex_cls(coord=vertex.coord, name=vertex.name)

    def _resolve_vertex(self, v: BaseVertex | str, label: str) -> BaseVertex:
        """
        Resolve a vertex given by reference or name (private method)

        :param v: vertex, or the name of a vertex in this graph
        :type v: BaseVertex subclass or str
        :param label: parameter name to use in the error message, e.g. "start"
        :raises TypeError: ``v`` is neither a ``BaseVertex`` nor a string
        :return: the resolved vertex
        :rtype: BaseVertex subclass

        Used by :meth:`add_edge`, :meth:`path_BFS`, :meth:`path_UCS` and
        :meth:`path_Astar` for their vertex-or-name parameters -- previously
        each method duplicated this check inline, which had let a
        copy-paste mistake (checking the wrong parameter's type in the
        error-raising branch) slip into all three path-finding methods
        unnoticed.
        """
        if isinstance(v, str):
            return self[v]
        elif isinstance(v, BaseVertex):
            return v
        else:
            raise TypeError(f"{label} must be BaseVertex subclass or string name")

    def _require_cost(self, edge: Edge) -> float:
        """
        Get an edge's cost, raising clearly if it hasn't been set (private method)

        :param edge: the edge
        :raises ValueError: ``edge.cost`` is None
        :return: the edge's cost
        :rtype: float

        ``Edge.cost`` is None when it could not be auto-computed (the edge
        was created outside a graph, or without vertex coordinates) and no
        explicit cost was given. Every method that does arithmetic with edge
        costs -- :meth:`distance`, :meth:`path_BFS`, :meth:`path_UCS`,
        :meth:`path_Astar` -- calls this rather than reading ``edge.cost``
        directly, so a missing cost fails clearly at the point of use
        instead of with a bare ``TypeError`` deep inside a search loop.

        If you want an edge that is present but deliberately unusable for
        path planning or distance calculations, set its cost to
        ``float("inf")`` explicitly -- ``None`` means "not set", not
        "infinite".

        :seealso: :meth:`Edge`
        """
        if edge.cost is None:
            raise ValueError(
                f"{edge} has no cost -- set an explicit cost, or "
                "float('inf') to mark it unusable for path planning"
            )
        return edge.cost

    def add_edge(self, v1: BaseVertex | str, v2: BaseVertex | str, **kwargs: Any) -> Edge:
        """
        Add an edge to the graph (base class method)

        :param v1: first vertex (start if a directed graph)
        :type v1: BaseVertex subclass or str
        :param v2: second vertex (end if a directed graph)
        :type v2: BaseVertex subclass or str
        :param kwargs: optional arguments to pass to ``BaseVertex.connect``
        :return: edge
        :rtype: Edge

        Create an edge between a vertex pair and adds it to the graph.

        This is a graph centric way of creating an edge.  The
        alternative is the ``connect`` method of a vertex.

        :seealso: :meth:`Edge.connect` :meth:`BaseVertex.connect`
        """
        v1 = self._resolve_vertex(v1, "v1")
        v2 = self._resolve_vertex(v2, "v2")

        if self._verbose:
            print(f"New edge from {v1.name} to {v2.name}")
        return v1.connect(v2, **kwargs)

    def remove_edge(self, edge: Edge) -> None:
        """
        Remove an edge from the graph

        :param edge: edge to remove
        :raises ValueError: ``edge`` does not belong to this graph

        The edge is removed from this graph's own edge collection and from
        the edge lists of its connected vertices, and ``edge.v1``/``edge.v2``
        are cleared to ``None``.

        .. warning:: The connectivity of the network may be changed.

        .. note:: A directed edge is tracked only by its source vertex's
            edge list, not its target's (see :attr:`BaseVertex.edges`), so
            membership is checked per endpoint rather than assumed for both
            -- removing a directed edge from an undirected-only
            implementation would otherwise raise ``ValueError`` on the
            target side.

        :seealso: :meth:`remove_vertex` :meth:`Edge.remove`
        """
        if edge not in self._edgelist:
            raise ValueError("edge does not belong to this graph")
        assert edge.v1 is not None and edge.v2 is not None

        if edge in edge.v1._edgelist:
            edge.v1._edgelist.remove(edge)
        if edge in edge.v2._edgelist:
            edge.v2._edgelist.remove(edge)

        edge.v1._connectivitychange = True
        edge.v2._connectivitychange = True
        self._connectivitychange = True

        edge.v1 = None
        edge.v2 = None

        self._edgelist.remove(edge)

    def remove_vertex(self, vertex: BaseVertex) -> None:
        """
        Remove a vertex, and all its edges, from the graph

        :param vertex: vertex to remove
        :raises ValueError: ``vertex`` does not belong to this graph

        Every edge touching ``vertex`` -- incoming or outgoing -- is removed
        via :meth:`remove_edge`, then the vertex itself is removed.

        .. warning:: The connectivity of the network may be changed.

        .. note:: This scans the graph's own edge set for edges touching
            ``vertex``, rather than iterating ``vertex.edges()`` --
            for a ``DGraph`` vertex that only ever reports outgoing edges
            (see :attr:`BaseVertex.edges`), so incoming edges would
            otherwise be missed and left dangling.

        :seealso: :meth:`remove_edge` :meth:`BaseVertex.remove`
        """
        if vertex._graph is not self:
            raise ValueError("vertex does not belong to this graph")
        assert vertex.name is not None

        for edge in [e for e in self._edgelist if e.v1 is vertex or e.v2 is vertex]:
            self.remove_edge(edge)

        self._vertexlist.remove(vertex)
        del self._vertexdict[vertex.name]

    def remove(self, x: Edge | BaseVertex) -> None:
        """
        Remove element from graph (deprecated)

        :param x: element to remove from graph
        :type x: Edge or BaseVertex subclass
        :raises TypeError: unknown type

        .. deprecated:: use :meth:`remove_edge` or :meth:`remove_vertex`
            instead -- this dispatched on ``type(x)`` to two operations with
            very different blast radii (detach one edge, vs. cascade-remove
            everything touching a vertex) hidden behind one ambiguous name.

        :seealso: :meth:`remove_edge` :meth:`remove_vertex`
        """
        warnings.warn(
            "remove() is deprecated, use remove_edge() or remove_vertex() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(x, Edge):
            self.remove_edge(x)
        elif isinstance(x, BaseVertex):
            self.remove_vertex(x)
        else:
            raise TypeError("expecting Edge or BaseVertex")

    def show(self) -> None:
        """
        Print a summary of all vertices and edges to stdout

        :seealso: :meth:`__str__`
        """
        print("vertices:")
        for v in self._vertexlist:
            print("  " + str(v))
        print("edges:")
        for e in self._edgelist:
            print("  " + str(e))

    @property
    def n(self) -> int:
        """
        Number of vertices

        :return: Number of vertices
        :rtype: int
        """
        return len(self._vertexdict)

    @property
    def ne(self) -> int:
        """
        Number of edges

        :return: Number of vertices
        :rtype: int
        """
        return len(self._edgelist)

    @abstractmethod
    def _graphcolor(self) -> int | None:
        """
        Color the graph (subclass method)

        Concrete graph coloring algorithm, provided by :meth:`UGraph._graphcolor`
        and :meth:`DGraph._graphcolor`.

        """

    @property
    def nc(self) -> int:
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
        n = self._graphcolor()
        if n is not None:
            self._ncomponents = n

        return self._ncomponents

    def _metricfunc(self, metric: Callable[[NDArray], float] | str) -> Callable[[NDArray], float]:
        """
        Resolve a metric name or callable to a callable (private method)

        :param metric: distance metric, a callable or one of "L1", "L2", "SE2"
        :raises ValueError: ``metric`` is a string other than "L1"/"L2"/"SE2",
            or is neither callable nor a string
        :return: the resolved distance metric callable
        :rtype: callable

        The returned callable takes a single coordinate-difference vector
        (shape ``(n,)``) and returns a scalar distance -- never a list or
        array of multiple vectors. That vector is always the difference
        between one vertex's ``coord`` and either another vertex's ``coord``
        or an arbitrary point supplied by the caller (see :meth:`closest` and
        :meth:`BaseVertex.distance`).

        If ``metric`` is already a callable matching this signature, it is
        returned unchanged. Otherwise it must be one of the built-in names
        "L1", "L2", "SE2" (see :meth:`metric` for their definitions).

        :seealso: :meth:`metric` :meth:`heuristic`
        """

        def L1(v):
            return np.linalg.norm(v, 1)

        def L2(v):
            return np.linalg.norm(v)

        def SE2(v):
            if len(v) != 3:
                raise ValueError(
                    f"SE2 metric requires a 3-element (x, y, theta) vector, got length {len(v)}"
                )
            # wrap angle to range [-pi, pi)
            v[2] = (v[2] + np.pi) % (2 * np.pi) - np.pi
            return np.linalg.norm(v)

        if callable(metric):
            return metric
        elif isinstance(metric, str):
            if metric == "L1":
                return L1
            elif metric == "L2":
                return L2
            elif metric == "SE2":
                return SE2
            else:
                raise ValueError(f"unknown metric {metric!r}")
        else:
            raise ValueError("unknown metric")

    @property
    def metric(self) -> Callable[[NDArray], float]:
        """
        Get the distance metric for graph

        :return: distance metric
        :rtype: callable

        This is a function of a single coordinate-difference vector (shape
        ``(n,)``), returning a scalar distance.
        """
        return self._metric

    @metric.setter
    def metric(self, metric: Callable[[NDArray], float] | str) -> None:
        r"""
        Set the distance metric for graph

        :param metric: distance metric
        :type metric: callable or str

        This is a function that takes a single coordinate-difference vector
        (shape ``(n,)``, not a list/array of multiple vectors) and returns a
        scalar distance.  It can be a user defined function or a string:

        - 'L1' is the norm :math:`L_1 = \Sigma_i | v_i |`
        - 'L2' is the norm :math:`L_2 = \sqrt{ \Sigma_i v_i^2}`
        - 'SE2' is a mixed norm for vectors :math:`(x, y, \theta)` and
            is :math:`\sqrt{x^2 + y^2 + \bar{\theta}^2}` where :math:`\bar{\theta}`
            is :math:`\theta` wrapped to the interval :math:`[-\pi, \pi)`.
            Requires every coordinate involved -- vertex ``coord`` and any
            point passed to :meth:`closest`/:meth:`BaseVertex.distance` -- to be
            exactly 3 elements; raises :exc:`ValueError` otherwise.

        The metric is used by :meth:`closest` and :meth:`distance`
        """
        self._metric = self._metricfunc(metric)

    @property
    def heuristic(self) -> Callable[[NDArray], float]:
        """
        Get the heuristic distance metric for graph

        :return: heuristic distance metric
        :rtype: callable

        This is a function of a single coordinate-difference vector (shape
        ``(n,)``), returning a scalar distance.
        """
        return self._heuristic

    @heuristic.setter
    def heuristic(self, heuristic: Callable[[NDArray], float] | str) -> None:
        r"""
        Set the heuristic distance metric for graph

        :param metric: heuristic distance metric
        :type metric: callable or str

        This is a function that takes a single coordinate-difference vector
        (shape ``(n,)``, not a list/array of multiple vectors) and returns a
        scalar distance.  It can be a user defined function or a string:

        - 'L1' is the norm :math:`L_1 = \Sigma_i | v_i |`
        - 'L2' is the norm :math:`L_2 = \sqrt{ \Sigma_i v_i^2}`
        - 'SE2' is a mixed norm for vectors :math:`(x, y, \theta)` and
            is :math:`\sqrt{x^2 + y^2 + \bar{\theta}^2}` where :math:`\bar{\theta}`
            is :math:`\theta` wrapped to the interval :math:`[-\pi, \pi)`.
            Requires every coordinate involved -- vertex ``coord`` and any
            point passed to :meth:`closest`/:meth:`BaseVertex.distance` -- to be
            exactly 3 elements; raises :exc:`ValueError` otherwise.

        The heuristic distance is only used by the A* planner :meth:`path_Astar`.
        """
        self._heuristic = self._metricfunc(heuristic)

    def __repr__(self) -> str:  # type: ignore[no-redef]
        """
        Detailed representation of the graph, one line per vertex

        :return: one line per vertex showing its name, coordinate and component
        :rtype: str

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v3 = g.add_vertex(coord=[2,2], name='v3')
            >>> g.add_edge(v1, v2)
            >>> g.add_edge(v2, v3)
            >>> repr(g)

        """
        s = [f"{self.__class__.__name__}:"]
        for vertex in self:
            ss = f"  {vertex.name} at {vertex.coord}"
            if vertex.label is not None:
                ss += f" component={vertex.label}"
            s.append(ss)
        return "\n".join(s)

    def __getitem__(self, i: int | str | BaseVertex) -> BaseVertex:
        """
        Get vertex (base class method)

        :param i: vertex description
        :type i: int or str
        :return: the referenced vertex
        :rtype: BaseVertex subclass

        Retrieve a vertex by index or name:

        -``g[i]`` is the i'th vertex in the graph.  This reflects the order of
         addition to the graph.
        -``g[s]`` is vertex named ``s``
        -``g[v]`` is ``v`` where ``v`` is a ``BaseVertex`` subclass

        This method also supports iteration over the vertices in a graph::

            for v in g:
                print(v)

        will iterate over all the vertices.
        """
        if isinstance(i, int):
            return self._vertexlist[i]
        elif isinstance(i, str):
            return self._vertexdict[i]
        elif isinstance(i, BaseVertex):
            return i

    def __iter__(self) -> Iterator[BaseVertex]:
        """
        Iterate over the vertices of the graph

        :return: iterator over vertices, in order of addition
        :rtype: iterator of BaseVertex subclass

        .. runblock:: pycon

        
            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v3 = g.add_vertex(coord=[2,2], name='v3')
            >>> for v in g:
            ...     print(v)

        :seealso: :meth:`__getitem__`
        """
        return iter(self._vertexlist)

    def __contains__(self, item: BaseVertex | str) -> bool:
        """
        Test if vertex in graph

        :param item: vertex or name of vertex
        :type item: BaseVertex subclass or str
        :return: true if vertex exists in the graph
        :rtype: bool

        - ``'name' in graph`` is true if a vertex named ``'name'`` exists in the
          graph.
        - ``v in graph`` is true if the vertex reference ``v`` exists in the
          graph.

        """
        if isinstance(item, str):
            return item in self._vertexdict
        elif isinstance(item, BaseVertex):
            return item in self._vertexdict.values()

    def closest(self, coord: ArrayLike) -> tuple[BaseVertex | None, float]:
        """
        BaseVertex closest to point

        :param coord: coordinates of a point
        :type coord: ndarray(n)
        :return: closest vertex and its distance, or ``(None, inf)`` if no
            vertex in the graph has a coordinate
        :rtype: BaseVertex subclass or None, float

        Returns the vertex closest to the given point. Distance is computed
        according to the graph's metric. Vertices without a coordinate
        (``coord`` is None) are skipped -- they have no position to compare.

        :seealso: :meth:`metric`
        """
        min_dist = np.inf
        min_which: BaseVertex | None = None

        for vertex in self:
            if vertex.coord is None:
                continue
            d = self.metric(vertex.coord - coord)
            if d < min_dist:
                min_dist = d
                min_which = vertex

        return min_which, min_dist

    def edges(self) -> set[Edge]:
        """
        Get all edges in graph (base class method)

        :return: All edges in the graph
        :rtype: set of Edge references

        We can iterate over all edges in the graph by::

            for e in g.edges():
                print(e)

        .. note:: Unlike :meth:`BaseVertex.edges`, which returns a ``list`` in
            connection order, this returns a ``set`` with no defined
            iteration order.

        :seealso: :meth:`BaseVertex.edges`
        """
        return self._edgelist

    def plot(
        self,
        colorcomponents: bool = True,
        force2d: bool = False,
        vopt: dict = {},
        eopt: dict = {},
        text: dict | bool = {},
        block: bool = False,
        grid: bool = True,
        ax: Any = None,
    ) -> None:
        """
        Plot the graph

        :param vopt: vertex format, defaults to 12pt o-marker
        :type vopt: dict, optional
        :param eopt: edge format, defaults to None
        :type eopt: dict, optional
        :param text: text label format, defaults to None
        :type text: False or dict, optional
        :param colorcomponents: color vertices and edges by component, defaults to None
        :type color: bool, optional
        :param block: block until figure is dismissed, defaults to True
        :type block: bool, optional

        The graph is plotted using matplotlib.

        If ``colorcomponents`` is True then each component is assigned a unique
        color.  ``vertex`` and ``edge`` cannot include a color keyword.

        If ``text`` is a dict it is used to format text labels for the vertices
        which are the vertex names.  If ``text`` is None default formatting is
        used.  If ``text`` is False no labels are added.

        :seealso: :meth:`highlight_path`
        """
        vopt = {**dict(marker="o", markersize=12), **vopt}
        eopt = {**dict(linewidth=3), **eopt}

        if colorcomponents:
            color = plt.cm.coolwarm(np.linspace(0, 1, self.nc))

        if len(self[0].coord) == 2 or force2d:
            # 2D plotting
            if ax is None:
                ax = axes_logic(ax, 2)
            for c in range(self.nc):
                # for each component
                for vertex in self.component(c):
                    if text is not False:
                        ax.text(vertex.x, vertex.y, "  " + vertex.name, **text)
                    if colorcomponents:
                        ax.plot(vertex.x, vertex.y, color=color[c, :], **vopt)
                        for v in vertex.neighbours():
                            ax.plot(
                                [vertex.x, v.x],
                                [vertex.y, v.y],
                                color=color[c, :],
                                **eopt,
                            )
                    else:
                        ax.plot(vertex.x, vertex.y, **vopt)
                        for v in vertex.neighbours():
                            ax.plot([vertex.x, v.x], [vertex.y, v.y], **eopt)
        else:
            # 3D or higher plotting, just do (x, y, z)
            if ax is None:
                ax = axes_logic(ax, 3)
            for c in range(self.nc):
                # for each component
                for vertex in self.component(c):
                    if text is not False:
                        ax.text(
                            vertex.x, vertex.y, vertex.z, "  " + vertex.name, **text
                        )
                    if colorcomponents:
                        ax.plot(
                            vertex.x,
                            vertex.y,
                            vertex.z,
                            **{**dict(color=color[c, :]), **vopt},
                        )
                        for v in vertex.neighbours():
                            ax.plot(
                                [vertex.x, v.x],
                                [vertex.y, v.y],
                                [vertex.z, v.z],
                                **{**dict(color=color[c, :]), **eopt},
                            )
                    else:
                        ax.plot(vertex.x, vertex.y, **vopt)
                        for v in vertex.neighbours():
                            ax.plot(
                                [vertex.x, v.x],
                                [vertex.y, v.y],
                                [vertex.z, v.z],
                                **eopt,
                            )
        # if nc > 1:
        #     # add a colorbar
        #     plt.colorbar()
        ax.grid(grid)
        if block is not None:
            plt.show(block=block)

    def highlight_path(self, path: list[BaseVertex], block: bool = False, **kwargs: Any) -> None:
        """
        Highlight a path through the graph

        :param path: sequence of vertices forming a path
        :type path: list of BaseVertex subclass
        :param block: block until figure is dismissed, defaults to False
        :param kwargs: arguments passed to :meth:`highlight_edge` and
            :meth:`highlight_vertex`

        The vertices and edges along the path are overwritten with a different
        size/width and color.

        :seealso: :meth:`highlight_vertex` :meth:`highlight_edge`
        """
        for i in range(len(path)):
            if i < len(path) - 1:
                e = path[i].edgeto(path[i + 1])
                self.highlight_edge(e, **kwargs)
            self.highlight_vertex(path[i], **kwargs)
        if block is not None:
            plt.show(block=block)

    def highlight_edge(
        self, edge: Edge, scale: float = 2, color: str = "r", alpha: float = 0.5
    ) -> None:
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
        plt.plot(
            [p1.x, p2.x], [p1.y, p2.y], color=color, linewidth=3 * scale, alpha=alpha
        )

    def highlight_vertex(
        self,
        vertex: BaseVertex | Iterable[BaseVertex | str],
        scale: float = 2,
        color: str = "r",
        alpha: float = 0.5,
    ) -> None:
        """
        Highlight a vertex in the graph

        :param edge: The vertex to highlight
        :type edge: BaseVertex subclass
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
                plt.plot(n.x, n.y, "o", color=color, markersize=12 * scale, alpha=alpha)
        else:
            plt.plot(
                vertex.x, vertex.y, "o", color=color, markersize=12 * scale, alpha=alpha
            )

    def dotfile(self, filename: str | Any | None = None, direction: str | None = None) -> None:
        """
        Export graph as a GraphViz dot file

        :param filename: filename to save graph to, defaults to None
        :type filename: str, optional

        ``g.dotfile()`` creates the specified file which contains the `GraphViz
        <https://graphviz.org>`_ code to represent the embedded graph.  By default
        output is to the console.

        .. note::

            - The graph is undirected if it is a subclass of ``UGraph``
            - The graph is directed if it is a subclass of ``DGraph``

        The ``graphviz`` formatters ``dot`` and ``neato`` can be used to render the
        graph in various formats including PDF, PNG, JPG, GIF, SVG, etc.  The formmatters are
        similar, but ``dot`` is better suited for directed graphs while ``neato`` is better
        suited for undirected graphs.

        .. note:: If ``filename`` is a file object then the file will *not*
            be closed after the GraphViz model is written.

        :seealso: :func:`showgraph`
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
                print(
                    '  "{:s}" [pos="{:.5g},{:.5g}"]'.format(
                        vertex.name, vertex.coord[0], vertex.coord[1]
                    ),
                    file=f,
                )
        print(file=f)
        # add the edges
        for e in self.edges():
            assert e.v1 is not None and e.v2 is not None
            if isinstance(self, DGraph):
                print('  "{:s}" -> "{:s}"'.format(e.v1.name, e.v2.name), file=f)
            else:
                print('  "{:s}" -- "{:s}"'.format(e.v1.name, e.v2.name), file=f)

        print("}", file=f)

        if filename is None or isinstance(filename, str):
            f.close()  # noqa

    def showgraph(self, **kwargs: Any) -> None:
        """
        Display graph in a browser tab

        :param kwargs: arguments passed to :meth:`dotfile`

        ``g.showgraph()`` renders and displays the graph in a browser tab.  The
        graph is exported in `GraphViz <https://graphviz.org>`_ format, rendered to
        PDF using ``dot`` and then displayed in a browser tab.

        :seealso: :func:`dotfile`
        """
        # create the temporary dotfile
        dotfile = tempfile.TemporaryFile(mode="w")
        self.dotfile(dotfile, **kwargs)

        # rewind the dot file, create PDF file in the filesystem, run dot
        dotfile.seek(0)
        pdffile = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        result = subprocess.run("dot -Tpdf", shell=True, stdin=dotfile, stdout=pdffile)

        if result.returncode == 0:
            # dot ran happily
            # open the PDF file in browser (hopefully portable), then cleanup
            webbrowser.open(f"file://{pdffile.name}")
            # time.sleep(1)
            # os.remove(pdffile.name)

    def iscyclic(self) -> bool:
        """
        Test if graph is cyclic

        :return: true if the graph contains at least one cycle
        :rtype: bool

        For an undirected graph this is a simple count: a forest (acyclic)
        has exactly ``n - nc`` edges, one per component-connecting edge with
        no redundancy, so any excess indicates a cycle. This is O(1) given
        :meth:`n`, :meth:`ne` and :meth:`nc`, which are already cached.

        For a directed graph this runs `Kahn's algorithm
        <https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm>`_:
        repeatedly remove vertices with no remaining incoming edges. If every
        vertex can eventually be removed, the graph is a DAG (acyclic); if
        some are never freed, they form a cycle. This is O(V+E), using
        vertices' outgoing edges via :meth:`BaseVertex.neighbours` the same
        way :meth:`path_BFS` and friends do.

        .. note:: A matrix-based test also exists in theory -- a digraph is
            acyclic iff its adjacency matrix is nilpotent (all eigenvalues
            zero) -- but that needs an O(N^3) eigendecomposition plus a
            floating-point zero-tolerance judgement call, to answer a
            question Kahn's algorithm answers exactly with integers in
            O(V+E). Not used here for that reason.

        :seealso: :meth:`adjacency`
        """
        if isinstance(self, UGraph):
            return self.ne > self.n - self.nc

        # DGraph: Kahn's algorithm
        indegree = {vertex: 0 for vertex in self}
        for e in self.edges():
            assert e.v2 is not None
            indegree[e.v2] += 1

        frontier = [vertex for vertex in self if indegree[vertex] == 0]
        removed = 0
        while frontier:
            vertex = frontier.pop()
            removed += 1
            for n in vertex.neighbours():
                indegree[n] -= 1
                if indegree[n] == 0:
                    frontier.append(n)

        return removed != self.n

    def average_degree(self) -> float:
        r"""
        Average degree of the graph

        :return: average degree
        :rtype: float

        Average degree is :math:`2 E / N` for an undirected graph and
        :math:`E / N` for a directed graph where :math:`E` is the total number of
        edges and :math:`N` is the number of vertices.

        :seealso: :meth:`degree`
        """
        if isinstance(self, DGraph):
            return len(self.edges()) / self.n
        elif isinstance(self, UGraph):
            return 2 * len(self.edges()) / self.n
        else:
            raise TypeError(f"unsupported graph type {type(self).__name__}")

    # --------------------------------------------------------------------------- #

    # MATRIX REPRESENTATIONS

    def Laplacian(self) -> NDArray:
        """
        Laplacian matrix for the graph

        :return: Laplacian matrix
        :rtype: NumPy ndarray

        ``g.Laplacian()`` is the Laplacian matrix (NxN) of the graph where N
        is the number of vertices.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2))
            ...
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j])
            ...
            >>> L = g.Laplacian()
            >>> print(L)

        .. note::

            - Laplacian always has at least one zero eigenvalue (each row of
              ``degree() - adjacency()`` sums to zero, so the all-ones
              vector is always a right null vector).
            - For an **undirected** graph specifically: the Laplacian is
              symmetric and positive-semidefinite, and the number of
              zero-valued eigenvalues equals the number of connected
              components.
            - For a **directed** graph, this computes the out-degree
              Laplacian (a real, recognized construction, e.g. in directed
              consensus/synchronization literature) -- but it is generally
              *not* symmetric, its eigenvalues can be complex, and the
              zero-eigenvalue/component-count relationship above does not
              hold. For example a simple weakly-connected out-tree (one
              component) already has two zero eigenvalues, not one.

        :seealso: :meth:`adjacency` :meth:`incidence` :meth:`degree`
        """
        return self.degree() - (self.adjacency() > 0)

    def connectivity(self, vertices: Iterable[BaseVertex] | None = None) -> list[int]:
        """
        Graph connectivity

        :param vertices: vertices to report connectivity for, defaults to all
            vertices in the graph
        :type vertices: iterable of BaseVertex subclass, optional
        :return: a list with the number of edges per vertex
        :rtype: list

        The average vertex connectivity is::

            mean(g.connectivity())

        and the minimum vertex connectivity is::

            min(g.connectivity())

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2))
            ...
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j])
            ...
            >>> c = g.connectivity()
            >>> print(c)


        :seealso: :meth:`degree`
        """

        c = []
        if vertices is None:
            vertices = self
        for n in vertices:
            c.append(len(n._edgelist))
        return c

    def degree(self) -> NDArray:
        """
        Degree matrix of graph

        :return: degree matrix
        :rtype: ndarray(N,N)

        This is a diagonal matrix  where element ``[i,i]`` is the number
        of edges connected to vertex id ``i``.

        .. note:: For a ``DGraph`` only outgoing edges are counted, matching
            :attr:`BaseVertex.degree`.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2))
            ...
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j])
            ...
            >>> d = g.degree()
            >>> print(d)

        :seealso: :meth:`adjacency` :meth:`incidence` :meth:`laplacian`
        """

        return np.diag(self.connectivity())

    def adjacency(self) -> NDArray:
        """
        Adjacency matrix of graph

        :returns: adjacency matrix
        :rtype: ndarray(N,N)

        The elements of the adjacency matrix ``[i,j]`` are 1 if vertex ``i`` is
        connected to vertex ``j``, else 0.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2))
            ...
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j])
            ...
            >>> A = g.adjacency()
            >>> print(A)

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

    def incidence(self) -> NDArray:
        """
        Incidence matrix of graph

        :returns: incidence matrix
        :rtype: ndarray(n,ne)

        The elements of the incidence matrix ``I[i,j]`` are 1 if vertex ``i`` is
        connected to edge ``j``, else 0.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2))
            ...
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j])
            ...
            >>> I = g.incidence()
            >>> print(I)

        .. note::
            - vertices are numbered in their order of creation. A vertex index
              can be resolved to a vertex reference by ``graph[i]``.
            - edges are numbered in the order they appear in ``graph.edges()``.
            - Both endpoints of every edge are marked, regardless of
              direction -- for a ``DGraph`` this means a vertex that is only
              ever a target (never a source) still appears here, unlike
              :meth:`degree`/:attr:`BaseVertex.degree`, which count outgoing
              edges only. Iterating each vertex's own
              :meth:`BaseVertex.edges` instead would silently drop such
              vertices for a directed graph.

        :seealso: :meth:`Laplacian` :meth:`adjacency` :meth:`degree`
        """
        edges = self.edges()
        I = np.zeros((self.n, len(edges)))

        vdict = {}
        for i, vertex in enumerate(self):
            vdict[vertex] = i

        for j, e in enumerate(edges):
            assert e.v1 is not None and e.v2 is not None
            I[vdict[e.v1], j] = 1
            I[vdict[e.v2], j] = 1

        return I

    def distance(self) -> NDArray:
        """
        Distance matrix of graph

        :return: distance matrix
        :rtype: ndarray(n,n)

        The elements of the distance matrix ``D[i,j]`` is the edge cost of moving
        from vertex ``i`` to vertex ``j``. It is zero if the vertices are not
        connected.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2))
            ...
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j])
            ...
            >>> d = g.distance()
            >>> print(d)

        :seealso: :meth:`BaseVertex.distance`
        """
        # create a dict mapping vertex to an id
        vdict = {}
        for i, vert in enumerate(self):
            vdict[vert] = i

        A = np.zeros((self.n, self.n))
        for v1 in self:
            for v2, edge in v1.incidences():
                A[vdict[v1], vdict[v2]] = self._require_cost(edge)
        return A

    # GRAPH COMPONENTS

    def component(self, c: int) -> list[BaseVertex]:
        """
        All vertices in specified graph component

        :param c: component index
        :return: vertices belonging to component ``c``
        :rtype: list of BaseVertex subclass

        ``graph.component(c)`` is a list of all vertices in graph component ``c``.

        :seealso: :meth:`nc` :meth:`samecomponent`
        """
        self._graphcolor()  # ensure labels are uptodate
        return [v for v in self if v.label == c]

    def samecomponent(self, v1: BaseVertex, v2: BaseVertex) -> bool:
        """
        Test if vertices belong to same graph component

        :param v1: vertex
        :type v1: BaseVertex subclass
        :param v2: vertex
        :type v2: BaseVertex subclass
        :return: true if vertices belong to same graph component
        :rtype: bool

        Test whether vertices belong to the same component.  For a:

        - directed graph this implies a path between them
        - undirected graph there is not necessarily a path between them

        :seealso: :meth:`component`
        """
        self._graphcolor()  # ensure labels are uptodate

        return v1.label == v2.label

    # --------------------------------------------------------------------------- #

    def path_BFS(
        self, S: BaseVertex | str, G: BaseVertex | str, verbose: bool = False, summary: bool = False
    ) -> tuple[list[BaseVertex], float] | None:
        """
        Breadth-first search for path

        :param S: start vertex
        :type S: BaseVertex subclass
        :param G: goal vertex
        :type G: BaseVertex subclass
        :param verbose: print search progress, defaults to False
        :param summary: print a one-line search summary, defaults to False
        :return: list of vertices from S to G inclusive, path length
        :rtype: list of BaseVertex subclass, float

        Returns a list of vertices that form a path from vertex ``S`` to
        vertex ``G`` if possible, otherwise return None.

        :seealso: :meth:`path_UCS` :meth:`path_Astar`
        """
        S = self._resolve_vertex(S, "start")
        G = self._resolve_vertex(G, "goal")

        # we use lists not sets since the order is instructive in verbose
        # mode, really need ordered sets...
        frontier: list[BaseVertex] = [S]
        explored: list[BaseVertex] = []
        parent: dict[BaseVertex, BaseVertex] = {}
        done = False

        while frontier:
            if verbose:
                print()
                print("FRONTIER:", ", ".join([str(v.name) for v in frontier]))
                print("EXPLORED:", ", ".join([str(v.name) for v in explored]))

            x = frontier.pop(0)
            if verbose:
                print("   expand", x.name)

            # expand the vertex
            for n in x.neighbours():
                if n is G:
                    if verbose:
                        print("     goal", n.name, "reached")
                    parent[n] = x
                    done = True
                    break
                if n not in frontier and n not in explored:
                    # add it to the frontier
                    frontier.append(n)
                    if verbose:
                        print("      add", n.name, "to the frontier")
                    parent[n] = x
            if done:
                break
            explored.append(x)
            if verbose:
                print("     move", x.name, "to the explored list")
        else:
            # no path
            return None

        # reconstruct the path from start to goal
        x = G
        path = [x]
        length = 0.0

        while x is not S:
            p = parent[x]
            length += self._require_cost(x.edgeto(p))
            path.insert(0, p)
            x = p

        if summary or verbose:
            print(
                f"{len(explored)} vertices explored, {len(frontier)} remaining on the frontier"
            )

        return path, length

    def path_UCS(
        self, S: BaseVertex | str, G: BaseVertex | str, verbose: bool = False, summary: bool = False
    ) -> tuple[list[BaseVertex], float, dict[str, str]] | None:
        """
        Uniform cost search for path

        :param S: start vertex
        :type S: BaseVertex subclass
        :param G: goal vertex
        :type G: BaseVertex subclass
        :param verbose: print search progress, defaults to False
        :param summary: print a one-line search summary, defaults to False
        :return: list of vertices from S to G inclusive, path length, tree
        :rtype: list of BaseVertex subclass, float, dict

        Returns a list of vertices that form a path from vertex ``S`` to
        vertex ``G`` if possible, otherwise return None.

        The search tree is returned as dict that maps a vertex to its parent.

        The heuristic is the distance metric of the graph, which defaults to
        Euclidean distance.

        :seealso: :meth:`path_BFS` :meth:`path_Astar`
        """
        S = self._resolve_vertex(S, "start")
        G = self._resolve_vertex(G, "goal")

        frontier: list[BaseVertex] = [S]
        explored: list[BaseVertex] = []
        parent: dict[BaseVertex, BaseVertex] = {}
        f: dict[BaseVertex, float] = {S: 0}  # evaluation function

        while frontier:
            if verbose:
                print()
                print(
                    "FRONTIER:", ", ".join([f"{v.name}({f[v]:.0f})" for v in frontier])
                )
                print("EXPLORED:", ", ".join([str(v.name) for v in explored]))

            i = np.argmin([f[n] for n in frontier])  # minimum f in frontier
            x = frontier.pop(i)
            if verbose:
                print("   expand", x.name)
            if x is G:
                break
            # expand the vertex
            for n, e in x.incidences():
                fnew = f[x] + self._require_cost(e)
                if n not in frontier and n not in explored:
                    # add it to the frontier
                    parent[n] = x
                    f[n] = fnew
                    frontier.append(n)
                    if verbose:
                        print("      add", n.name, "to the frontier")

                elif n in frontier:
                    # neighbour is already in the frontier
                    # cost of path via x is lower that previous, reparent it
                    if fnew < f[n]:
                        if verbose:
                            print(
                                f" reparent {n.name}: cost {fnew} via {x.name} is less than cost {f[n]} via {parent[n].name}, change parent from {parent[n].name} to {x.name} "
                            )
                        f[n] = fnew
                        parent[n] = x

            explored.append(x)
            if verbose:
                print("     move", x.name, "to the explored list")
        else:
            # no path
            return None

        # reconstruct the path from start to goal
        x = G
        path = [x]
        length = 0.0

        while x is not S:
            p = parent[x]
            length += self._require_cost(p.edgeto(x))
            path.insert(0, p)
            x = p

        parent_names: dict[str, str] = {}
        for v, p in parent.items():
            assert v.name is not None and p.name is not None
            parent_names[v.name] = p.name

        if summary or verbose:
            print(
                f"{len(explored)} vertices explored, {len(frontier)} remaining on the frontier"
            )

        return path, length, parent_names

    def path_Astar(
        self, S: BaseVertex | str, G: BaseVertex | str, verbose: bool = False, summary: bool = False
    ) -> tuple[list[BaseVertex], float, dict[str, str]] | None:
        """
        A* search for path

        :param S: start vertex
        :type S: BaseVertex subclass
        :param G: goal vertex
        :type G: BaseVertex subclass
        :param verbose: print search progress, defaults to False
        :param summary: print a one-line search summary, defaults to False
        :return: list of vertices from S to G inclusive, path length, tree
        :rtype: list of BaseVertex subclass, float, dict

        Returns a list of vertices that form a path from vertex ``S`` to
        vertex ``G`` if possible, otherwise return None.

        The search tree is returned as dict that maps a vertex to its parent.

        The heuristic is the distance metric of the graph, which defaults to
        Euclidean distance.

        :seealso: :meth:`heuristic` :meth:`path_BFS` :meth:`path_UCS`
        """
        S = self._resolve_vertex(S, "start")
        G = self._resolve_vertex(G, "goal")

        frontier: list[BaseVertex] = [S]
        explored: list[BaseVertex] = []
        parent: dict[BaseVertex, BaseVertex] = {}
        g: dict[BaseVertex, float] = {S: 0}  # cost to come
        f: dict[BaseVertex, float] = {S: 0}  # evaluation function

        while frontier:
            if verbose:
                print()
                print(
                    "FRONTIER:", ", ".join([f"{v.name}({f[v]:.0f})" for v in frontier])
                )
                print("EXPLORED:", ", ".join([str(v.name) for v in explored]))

            i = np.argmin([f[n] for n in frontier])  # minimum f in frontier
            x = frontier.pop(i)
            if verbose:
                print("   expand", x.name)
            if x is G:
                break
            # expand the vertex
            for n, e in x.incidences():
                if n not in frontier and n not in explored:
                    # add it to the frontier
                    frontier.append(n)
                    parent[n] = x
                    g[n] = g[x] + self._require_cost(e)  # update cost to come
                    f[n] = g[n] + n.heuristic_distance(G)  # heuristic
                    if verbose:
                        print("      add", n.name, "to the frontier")
                elif n in frontier:
                    # neighbour is already in the frontier
                    gnew = g[x] + self._require_cost(e)
                    if gnew < g[n]:
                        # cost of path via x is lower that previous, reparent it
                        if verbose:
                            print(
                                f" reparent {n.name}: cost {gnew} via {x.name} is less than cost {g[n]} via {parent[n].name}, change parent from {parent[n].name} to {x.name} "
                            )
                        g[n] = gnew
                        f[n] = g[n] + n.heuristic_distance(G)  # heuristic

                        parent[n] = x  # reparent

            explored.append(x)
            if verbose:
                print("     move", x.name, "to the explored list")

        else:
            # no path
            return None

        # reconstruct the path from start to goal
        x = G
        path = [x]
        length = 0.0

        while x is not S:
            p = parent[x]
            length += self._require_cost(p.edgeto(x))
            path.insert(0, p)
            x = p

        parent_names: dict[str, str] = {}
        for v, p in parent.items():
            assert v.name is not None and p.name is not None
            parent_names[v.name] = p.name

        if summary or verbose:
            print(
                f"{len(explored)} vertices explored, {len(frontier)} remaining on the frontier"
            )

        return path, length, parent_names


# -------------------------------------------------------------------------- #


class UGraph(_BaseGraph):
    """
    Class for undirected graphs

    .. inheritance-diagram:: UGraph

    :seealso: :class:`_BaseGraph` :class:`DGraph`
    """

    def _graphcolor(self) -> None:
        """
        Color the graph

        Performs a depth-first labeling operation, assigning the ``label``
        attribute of every vertex with a sequential integer starting from 0.

        This method checks the ``_connectivitychange`` attribute of all vertices
        and if any are True it will perform the coloring operation. This flag
        is set True by any operation that adds or removes a vertex or edge.

        :seealso: :meth:`nc`
        """
        if self._connectivitychange or any([n._connectivitychange for n in self]):

            # color the graph

            # clear all the labels
            for vertex in self:
                vertex.label = None
                vertex._connectivitychange = False

            lastlabel: int | None = None
            for label in range(self.n):
                assignment = False
                for v in self:
                    # find first vertex with no label
                    if v.label is None:
                        # do BFS
                        q = [v]  # initialize frontier
                        while len(q) > 0:
                            v = q.pop()  # expand v
                            v.label = label
                            for n in v.neighbours():
                                if n.label is None:
                                    q.append(n)
                        lastlabel = label
                        assignment = True
                        break
                if not assignment:
                    break

            self._ncomponents = 0 if lastlabel is None else lastlabel + 1


class DGraph(_BaseGraph):
    """
    Class for directed graphs

    .. inheritance-diagram:: DGraph

    :seealso: :class:`_BaseGraph` :class:`UGraph`
    """

    def _graphcolor(self) -> int | None:
        """
        Color the graph

        Performs a depth-first labeling operation, assigning the ``label``
        attribute of every vertex with a sequential integer starting from 0.

        This method checks the ``_connectivitychange`` attribute of all vertices
        and if any are True it will perform the coloring operation. This flag
        is set True by any operation that adds or removes a vertex or edge.

        :seealso: :meth:`nc`
        """
        if self._connectivitychange or any([n._connectivitychange for n in self]):

            # color the graph

            # clear all the labels
            for vertex in self:
                vertex.label = None
                vertex._connectivitychange = False

            # initial labeling pass
            merge: dict[int, int] = {}
            nextlabel = 1
            for v in self:
                if v.label is None:
                    # no label, try to inherit one from a neighbour
                    for n in v.neighbours():
                        if n.label is not None:
                            # neighbour has a label
                            v.label = n.label
                            break

                if v.label is None:
                    # still not labeled, assign a new label
                    v.label = nextlabel
                    nextlabel += 1

                label = v.label
                assert label is not None

                # now look for clashes
                for n in v.neighbours():
                    if n.label is None:
                        # neighbour has no label, give it this one
                        n.label = label
                    elif label != n.label:
                        # label clash, note it for merging
                        assert n.label is not None
                        merge[n.label] = label

            # merge labels and find unique labels
            unique: set[int] = set()
            for v in self:
                vlabel = v.label
                assert vlabel is not None
                while vlabel in merge:
                    vlabel = merge[vlabel]
                v.label = vlabel
                unique.add(vlabel)

            final = {u: i for i, u in enumerate(unique)}
            for v in self:
                vlabel = v.label
                assert vlabel is not None
                v.label = final[vlabel]

            return len(unique)
        else:
            # no coloring performed
            return None


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

    :seealso: :class:`BaseVertex`
    """

    def __init__(
        self,
        v1: BaseVertex | None = None,
        v2: BaseVertex | None = None,
        cost: float | None = None,
        data: Any = None,
    ):
        """
        Create an edge object

        :param v1: start of the edge, defaults to None
        :type v1: BaseVertex subclass, optional
        :param v2: end of the edge, defaults to None
        :type v2: BaseVertex subclass, optional
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

        .. note:: To compute edge cost from the vertices, both vertices must
            have already been added to the same graph -- otherwise ``cost``
            is left as None rather than raising, since this constructor is
            also used standalone, independent of any graph.

        :seealso: :meth:`Edge.connect` :meth:`BaseVertex.connect`
        """
        self.v1 = v1
        self.v2 = v2

        self.data = data

        # try to compute edge cost as metric distance if not given
        self.cost: float | None
        if cost is not None:
            self.cost = cost
        elif (
            v1 is not None
            and v2 is not None
            and v1.coord is not None
            and v2.coord is not None
            and v1._graph is not None
            and v1._graph is v2._graph
        ):
            self.cost = v1._graph.metric(v1.coord - v2.coord)
        else:
            self.cost = None

    def __repr__(self) -> str:
        """
        Detailed representation of the edge

        :return: same as :meth:`__str__`
        :rtype: str

        .. runblock:: pycon

            >>> from pgraph import UVertex, Edge
            >>> v1 = UVertex(coord=[1,2], name="A")
            >>> v2 = UVertex(coord=[3,4], name="B")
            >>> e = Edge(v1, v2, cost=5.0, data="A to B")
            >>> repr(e)

        :seealso: :meth:`__str__`
        """
        return str(self)

    def __str__(self) -> str:
        """
        Human-readable summary of the edge

        :return: endpoints, cost and optional data
        :rtype: str

        .. runblock:: pycon

            >>> from pgraph import UVertex, Edge
            >>> v1 = UVertex(coord=[1,2], name="A")
            >>> v2 = UVertex(coord=[3,4], name="B")
            >>> e = Edge(v1, v2, cost=5.0, data="A to B")
            >>> str(e)

        :seealso: :meth:`__repr__`
        """
        arrow = "->" if isinstance(self.v1, DVertex) else "--"
        cost_str = "None" if self.cost is None else f"{self.cost:.4g}"
        s = f"{self.__class__.__name__}{{{self.v1} {arrow} {self.v2}, cost={cost_str}}}"
        if self.data is not None:
            s += f" data={self.data}"
        return s

    def connect(self, v1: BaseVertex, v2: BaseVertex) -> None:
        """
        Attach this edge to a pair of vertices

        :param v1: start of the edge
        :type v1: BaseVertex subclass
        :param v2: end of the edge
        :type v2: BaseVertex subclass

        The edge connects vertices ``v1`` and ``v2``, and is added to the
        graph that those vertices belong to.

        .. runblock:: pycon

            >>> from pgraph import UVertex, Edge, UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> e = Edge(cost=1.414)
            >>> e.connect(v1, v2)
            >>> print(e)

        .. note:: The vertices must already be added to the graph.

        :seealso: :meth:`BaseVertex.connect`
        """

        if v1._graph is None:
            raise ValueError("vertex v1 does not belong to a graph")
        if v2._graph is None:
            raise ValueError("vertex v2 does not belong to a graph")
        if not v1._graph is v2._graph:
            raise ValueError("vertices must belong to the same graph")

        # connect edge to its vertices
        self.v1 = v1
        self.v2 = v2

        # tell the vertices to add edge to their edgelists as appropriate for
        # DGraph or UGraph
        v1.connect(v2, edge=self)

    def next(self, vertex: BaseVertex) -> BaseVertex:
        """
        Return other end of an edge

        :param vertex: one vertex on the edge
        :type vertex: BaseVertex subclass
        :raises ValueError: ``vertex`` is not on the edge
        :return: the other vertex on the edge
        :rtype: BaseVertex subclass

        ``e.next(v1)`` is the vertex at the other end of edge ``e``, ie. the
        vertex that is not ``v1``.

        .. runblock:: pycon

            >>> from pgraph import UVertex, Edge
            >>> v1 = UVertex(coord=[1,2], name="A")
            >>> v2 = UVertex(coord=[3,4], name="B")
            >>> e = Edge(v1, v2, cost=5.0, data="A to B")
            >>> print(e)
            >>> e.next(v1)
            >>> e.next(v2)

        """

        if self.v1 is vertex:
            assert self.v2 is not None
            return self.v2
        elif self.v2 is vertex:
            assert self.v1 is not None
            return self.v1
        else:
            raise ValueError("shouldnt happen")

    def vertices(self) -> list[BaseVertex]:
        """
        Vertices of an edge (deprecated)

        :return: the two vertices of this edge
        :rtype: list of BaseVertex subclass

        .. deprecated:: use :attr:`endpoints` instead
        """
        warnings.warn(
            "vertices() is deprecated, use endpoints instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.endpoints

    @property
    def endpoints(self) -> list[BaseVertex]:
        """
        The two vertices of this edge

        :return: start and end vertex
        :rtype: list of BaseVertex subclass

        .. runblock:: pycon

            >>> from pgraph import UVertex, Edge
            >>> v1 = UVertex(coord=[1,2], name="A")
            >>> v2 = UVertex(coord=[3,4], name="B")
            >>> e = Edge(v1, v2, cost=5.0, data="A to B")
            >>> print(e)
            >>> print(e.endpoints)

        :seealso: :meth:`vertices`
        """
        assert self.v1 is not None and self.v2 is not None
        return [self.v1, self.v2]

    def remove(self) -> None:
        """
        Remove this edge from its graph

        :raises ValueError: the edge is not connected to a graph

        ``e.remove()`` removes ``e`` from its graph's own edge collection and
        from its connected vertices' edge lists (both, for an undirected
        edge; the source only, for a directed one -- see
        :meth:`_BaseGraph.remove_edge`), and clears ``e.v1``/``e.v2`` to
        ``None``. The ``Edge`` object itself is not deleted -- it becomes an
        orphaned, disconnected shell, and calling ``remove()`` on it again
        raises ``ValueError``.

        .. note:: This is a thin convenience wrapper around
            :meth:`_BaseGraph.remove_edge`.

        :seealso: :meth:`_BaseGraph.remove_edge` :meth:`BaseVertex.remove`
        """
        if self.v1 is None or self.v1._graph is None:
            raise ValueError("edge is not connected to a graph")
        self.v1._graph.remove_edge(self)


# ========================================================================== #


class BaseVertex:
    """
    Base class for vertices of directed and non-directed graphs.

    Each vertex has:
        - ``name``
        - ``label`` an int indicating which graph component contains it
        - ``_edgelist`` a list of edge objects that connect this vertex to others
        - ``coord`` the coordinate in an embedded graph (optional)

    :seealso: :class:`UVertex` :class:`DVertex` :class:`Edge`
    """

    def __init__(self, coord: ArrayLike | None = None, name: str | None = None):
        """
        Create a vertex object

        :param coord: coordinate of the vertex for an embedded graph, defaults to None
        :type coord: array-like, optional
        :param name: vertex name, defaults to None
        :type name: str, optional

        Creates a vertex but does not add it to a graph -- use
        :meth:`_BaseGraph.add_vertex` for that.

        .. runblock:: pycon

            >>> from pgraph import UVertex
            >>> v1 = UVertex(coord=[0,0], name='v1')
            >>> print(v1)

        :seealso: :meth:`_BaseGraph.add_vertex`
        """
        self._edgelist: list[Edge] = []
        if coord is None:
            self.coord = None
        else:
            self.coord = np.r_[coord]
        self.name = name
        self.label: int | None = None
        self._connectivitychange = True
        self._edgelist = []
        self._graph: _BaseGraph | None = None  # reference to owning graph
        # print('BaseVertex init', type(self))

    def __str__(self) -> str:
        """
        Compact representation of the vertex

        :return: the vertex name in square brackets
        :rtype: str

        .. runblock:: pycon

            >>> from pgraph import UVertex
            >>> v = UVertex(coord=[1,2], name="A")
            >>> str(v)
        """
        return f"[{self.name:s}]"

    def __repr__(self) -> str:
        """
        Detailed representation of the vertex

        :return: class name, vertex name and coordinate
        :rtype: str

        .. runblock:: pycon

            >>> from pgraph import UVertex
            >>> v = UVertex(coord=[1,2], name="A")
            >>> repr(v)
        """
        if self.coord is None:
            coord = "?"
        else:
            coord = ", ".join([f"{x:.4g}" for x in self.coord])
        return f"{self.__class__.__name__}[{self.name:s}, coord=({coord})]"

    def copy(self, cls: type[_BaseGraph] | None = None) -> BaseVertex:
        """
        Copy a vertex

        :param cls: graph class whose ``vertex_copy`` method should be used to
            create the copy, defaults to None
        :type cls: UGraph or DGraph subclass, optional
        :return: a new, unconnected vertex with the same coordinate and name
        :rtype: BaseVertex subclass

        If ``cls`` is given, ``cls.vertex_copy(self)`` is used to create a
        vertex of the appropriate subclass for that graph type, otherwise a
        vertex of the same class as ``self`` is created directly.

        :seealso: :meth:`UGraph.vertex_copy` :meth:`DGraph.vertex_copy`
        """
        if cls is not None:
            return cls.vertex_copy(self)
        else:
            return self.__class__(coord=self.coord, name=self.name)

    def neighbours(self) -> list[BaseVertex]:
        """
        Neighbours of a vertex

        ``v.neighbours()`` is a list of neighbours of this vertex.

        .. note:: For a directed graph the neighbours are those on edges leaving this vertex

        :seealso: :meth:`neighbors` :meth:`incidences`
        """
        return [e.next(self) for e in self._edgelist]

    def neighbors(self) -> list[BaseVertex]:
        """
        Neighbors of a vertex

        ``v.neighbors()`` is a list of neighbors of this vertex.

        .. note:: For a directed graph the neighbours are those on edges leaving this vertex

        :seealso: :meth:`neighbours`
        """
        return [e.next(self) for e in self._edgelist]

    def adjacent(self) -> list[BaseVertex]:
        """
        Neighbours of a vertex (deprecated)

        :return: a list of neighbours of this vertex
        :rtype: list of BaseVertex subclass

        .. deprecated:: use :meth:`neighbours` instead
        """
        warnings.warn(
            "adjacent() is deprecated, use neighbours() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.neighbours()

    def isneighbour(self, vertex: BaseVertex) -> bool:
        """
        Test if vertex is a neigbour

        :param vertex: vertex reference
        :type vertex: BaseVertex subclass
        :return: true if a neighbour
        :rtype: bool

        For a directed graph this is true only if the edge is from ``self`` to
        ``vertex``.

        :seealso: :meth:`neighbours`
        """
        return vertex in [e.next(self) for e in self._edgelist]

    def incidences(self) -> list[tuple[BaseVertex, Edge]]:
        """
        Neighbours and edges of a vertex

        ``v.incidences()`` is a generator that returns a list of incidences,
        tuples of (vertex, edge) for all neighbours of the vertex ``v``.

        .. note:: For a directed graph the edges are those leaving this vertex

        :seealso: :meth:`neighbours` :meth:`edges`
        """
        return [(e.next(self), e) for e in self._edgelist]

    def connect(
        self,
        dest: BaseVertex,
        edge: Edge | None = None,
        cost: float | None = None,
        data: Any = None,
    ) -> Edge:
        """
        Connect two vertices with an edge

        :param dest: The vertex to connect to
        :type dest: ``BaseVertex`` subclass
        :param edge: Use this as the edge object, otherwise a new ``Edge``
                     object is created from the vertices being connected,
                     and the ``cost`` and ``edge`` parameters, defaults to None
        :type edge: ``Edge`` subclass, optional
        :param cost: the cost to traverse this edge, defaults to None
        :type cost: float, optional
        :param data: reference to arbitrary data associated with the edge,
                     defaults to None
        :type data: Any, optional
        :raises ValueError: either vertex has not been added to a graph, or
            the vertices belong to different graphs
        :return: the edge connecting the vertices
        :rtype: Edge

        ``v1.connect(v2)`` connects vertex ``v1`` to vertex ``v2``.

        .. note::

            - If the vertices subclass ``UVertex`` the edge is undirected, and if
              they subclass ``DVertex`` the edge is directed.
            - Both vertices must already have been added to the same graph,
              e.g. via :meth:`PGraph.add_vertex` -- since a graph only ever
              accepts its own vertex subclass (see :meth:`UGraph.add_vertex`,
              :meth:`DGraph.add_vertex`), this also rules out connecting a
              ``UVertex`` to a ``DVertex``.

        :seealso: :meth:`Edge` :meth:`Edge.connect`
        """

        if self._graph is None or dest._graph is None:
            raise ValueError(
                "both vertices must be added to a graph before being connected"
            )
        elif self._graph is not dest._graph:
            raise ValueError("vertices must belong to the same graph")
        elif isinstance(edge, Edge):
            e = edge
        else:
            e = Edge(self, dest, cost=cost, data=data)

        self._graph._edgelist.add(e)
        self._graph._connectivitychange = True
        self._connectivitychange = True

        return e

    def edgeto(self, dest: BaseVertex) -> Edge:
        """
        Get edge connecting vertex to specific neighbour

        :param dest: a neigbouring vertex
        :type dest: ``BaseVertex`` subclass
        :raises ValueError: ``dest`` is not a neighbour
        :return: the edge from this vertex to ``dest``
        :rtype: Edge

        .. note::

            - For a directed graph ``dest`` must be at the arrow end of the edge
        """
        for n, e in self.incidences():
            if n is dest:
                return e
        raise ValueError("dest is not a neighbour")

    def edges(self) -> list[Edge]:
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

    def heuristic_distance(self, v2: BaseVertex) -> float:
        """
        Heuristic distance to another vertex

        :param v2: the other vertex
        :type v2: BaseVertex subclass
        :return: heuristic distance between this vertex and ``v2``
        :rtype: float

        Distance is computed according to the graph's heuristic, see
        :meth:`_BaseGraph.heuristic`. The vertex must belong to a graph,
        since the heuristic is a property of the graph, not the vertex.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[1, 2], name='v1')
            >>> v2 = g.add_vertex(coord=[4, 3], name='v2')
            >>> print(v1.heuristic_distance(v2))

        :seealso: :meth:`distance` :meth:`_BaseGraph.heuristic`
        """
        if self._graph is None:
            raise ValueError("vertex is not connected to a graph")
        if self.coord is None or v2.coord is None:
            raise ValueError("both vertices must have a coordinate")
        return self._graph.heuristic(self.coord - v2.coord)

    def distance(self, coord: ArrayLike | BaseVertex) -> float:
        """
        Distance from vertex to point

        :param coord: coordinates of the point
        :type coord: ndarray(n) or BaseVertex
        :return: distance
        :rtype: float

        Distance is computed according to the graph's metric. The vertex
        must belong to a graph, since the metric is a property of the
        graph, not the vertex.

        :seealso: :meth:`metric`
        """
        if self._graph is None:
            raise ValueError("vertex is not connected to a graph")
        target = coord.coord if isinstance(coord, BaseVertex) else coord
        if self.coord is None or target is None:
            raise ValueError("vertex, and coord if given as a vertex, must have a coordinate")
        return self._graph.metric(self.coord - target)

    @property
    def degree(self) -> int:
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
    def x(self) -> float:
        """
        The x-coordinate of an embedded vertex

        :raises ValueError: the vertex has no coordinate
        :return: The x-coordinate
        :rtype: float
        """
        if self.coord is None:
            raise ValueError("vertex has no coordinate")
        return self.coord[0]

    @property
    def y(self) -> float:
        """
        The y-coordinate of an embedded vertex

        :raises ValueError: the vertex has no coordinate
        :return: The y-coordinate
        :rtype: float
        """
        if self.coord is None:
            raise ValueError("vertex has no coordinate")
        return self.coord[1]

    @property
    def z(self) -> float:
        """
        The z-coordinate of an embedded vertex

        :raises ValueError: the vertex has no coordinate
        :return: The z-coordinate
        :rtype: float
        """
        if self.coord is None:
            raise ValueError("vertex has no coordinate")
        return self.coord[2]

    def closest(self) -> tuple[BaseVertex | None, float]:
        """
        BaseVertex closest to this vertex

        :return: closest vertex and its distance
        :rtype: BaseVertex subclass or None, float

        Equivalent to ``self._graph.closest(self.coord)``. The vertex must
        belong to a graph, since ``closest()`` searches that graph's other
        vertices.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v3 = g.add_vertex(coord=[2,3], name='v3')
            >>> v4 = g.add_vertex(coord=[4,3], name='v4')
            >>> print(v1.closest())

        :seealso: :meth:`_BaseGraph.closest`
        """
        if self._graph is None:
            raise ValueError("vertex is not connected to a graph")
        if self.coord is None:
            raise ValueError("vertex must have a coordinate")
        return self._graph.closest(self.coord)

    def remove(self) -> None:
        """
        Remove this vertex, and all its edges, from its graph

        :raises ValueError: the vertex is not connected to a graph

        ``v.remove()`` removes ``v``, and every edge touching it (incoming
        or outgoing), from its graph. The ``BaseVertex`` object itself is
        not deleted.

        .. note:: This is a thin convenience wrapper around
            :meth:`_BaseGraph.remove_vertex`.

        :seealso: :meth:`_BaseGraph.remove_vertex` :meth:`Edge.remove`
        """
        if self._graph is None:
            raise ValueError("vertex is not connected to a graph")
        self._graph.remove_vertex(self)


class UVertex(BaseVertex):
    """
    BaseVertex subclass for undirected graphs

    This class can be inherited to provide user objects with graph capability.


    .. inheritance-diagram:: UVertex

    :seealso: :class:`BaseVertex` :class:`DVertex`
    """

    def connect(
        self,
        dest: BaseVertex,
        edge: Edge | None = None,
        cost: float | None = None,
        data: Any = None,
    ) -> Edge:
        """
        Connect this vertex to another with an undirected edge

        :param dest: vertex to connect to
        :type dest: BaseVertex subclass
        :param edge: Use this as the edge object, otherwise a new ``Edge``
                     object is created, defaults to None
        :type edge: ``Edge`` subclass, optional
        :param cost: the cost to traverse this edge, defaults to None
        :type cost: float, optional
        :param data: reference to arbitrary data associated with the edge,
                     defaults to None
        :type data: Any, optional
        :return: the edge connecting the vertices
        :rtype: Edge

        Unlike the directed-graph counterpart, the new edge is added to
        *both* vertices' edge lists, so it is discovered from either end.

        .. runblock:: pycon

            >>> from pgraph import UVertex, Edge, UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v1.connect(v2, cost=1.414)
            >>> print(v1.edges())
            >>> print(g.edges())
            >>> print(g)

        :seealso: :meth:`BaseVertex.connect` :meth:`DVertex.connect`
        """

        e = super().connect(dest, edge=edge, cost=cost, data=data)

        self._edgelist.append(e)
        dest._edgelist.append(e)

        return e


class DVertex(BaseVertex):
    """
    BaseVertex subclass for directed graphs

    This class can be inherited to provide user objects with graph capability.

    .. inheritance-diagram:: DVertex

    :seealso: :class:`BaseVertex` :class:`UVertex`
    """

    def connect(
        self,
        dest: BaseVertex,
        edge: Edge | None = None,
        cost: float | None = None,
        data: Any = None,
    ) -> Edge:
        """
        Connect this vertex to another with a directed edge

        :param dest: vertex to connect to
        :type dest: BaseVertex subclass
        :param edge: Use this as the edge object, otherwise a new ``Edge``
                     object is created, defaults to None
        :type edge: ``Edge`` subclass, optional
        :param cost: the cost to traverse this edge, defaults to None
        :type cost: float, optional
        :param data: reference to arbitrary data associated with the edge,
                     defaults to None
        :type data: Any, optional
        :return: the edge connecting the vertices
        :rtype: Edge

        Unlike the undirected-graph counterpart, the new edge is added only
        to *this* vertex's edge list -- it is only discoverable from the
        start of the directed edge.

        .. runblock:: pycon

            >>> from pgraph import DVertex, Edge, DGraph
            >>> g = DGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v1.connect(v2, cost=1.414)
            >>> print(v1.edges())
            >>> print(g.edges())
            >>> print(g)

        :seealso: :meth:`BaseVertex.connect` :meth:`UVertex.connect`
        """
        e = super().connect(dest, edge=edge, cost=cost, data=data)

        self._edgelist.append(e)
        return e


# UGraph/DGraph declare their _vertex_cls here, rather than in their own
# class body, because UVertex/DVertex are defined later in this file --
# _vertex_cls is a real runtime assignment (unlike a type annotation), so it
# can't rely on `from __future__ import annotations` to defer it.
UGraph._vertex_cls = UVertex
DGraph._vertex_cls = DVertex


if __name__ == "__main__":

    g = UGraph()
    print(g)

    for i in range(10):
        g.add_vertex()

    g.add_edge(g[0], g[1])

    print(g)
