from __future__ import annotations

import math
import os
import sys
import warnings
import numpy as np
import copy
import heapq
import importlib
from collections import deque
from collections.abc import Iterable, Iterator
import tempfile
import webbrowser
from typing import IO, Any, Callable, ClassVar, overload
from numpy.typing import ArrayLike, NDArray

# matplotlib and spatialmath are imported by the methods that plot, and by
# pgraph._plot, so that importing pgraph does not load them.


def _L1(v: NDArray) -> float:
    """L1 (Manhattan) norm, the built-in ``"L1"`` metric"""
    return float(np.linalg.norm(v, 1))


def _L2(v: NDArray) -> float:
    """L2 (Euclidean) norm, the built-in ``"L2"`` metric"""
    return float(np.linalg.norm(v))


class _KDTree:
    """A spatial index of a graph's vertices and how to query it (private)"""

    def __init__(self, tree: Any, index: NDArray, p: float, prepare: Any):
        self.tree = tree  # scipy.spatial.cKDTree
        self.index = index  # position in the graph's vertex list of each point
        self.p = p  # the Minkowski norm that the tree uses
        self.prepare = prepare  # point -> point for the tree, or None if out of range


class _PointCache:
    """
    The coordinates of a graph's embedded vertices, as rows of a matrix (private)

    Row ``i`` is the coordinate of ``vertices[i]``, in the order of the graph.
    The buffer has spare rows so that adding a vertex is an append, which is
    what lets :meth:`Graph.closest` stay fast while a planner interleaves
    ``add_vertex`` with queries.
    """

    def __init__(self, points: NDArray, vertices: list[Vertex]):
        n, dim = points.shape
        self.buf = np.empty((n + n // 2 + 16, dim))
        self.buf[:n] = points
        self.n = n
        self.vertices = vertices

    @property
    def dim(self) -> int:
        return self.buf.shape[1]

    @property
    def points(self) -> NDArray:
        """The ``n`` x dim matrix of coordinates (a view, not a copy)"""
        return self.buf[: self.n]

    def append(self, vertex: Vertex, coord: NDArray) -> None:
        if self.n == self.buf.shape[0]:
            grown = np.empty((2 * self.n, self.dim))
            grown[: self.n] = self.buf
            self.buf = grown
        self.buf[self.n] = coord
        self.n += 1
        self.vertices.append(vertex)


class SE2Metric:
    r"""
    Distance between planar poses, a weighted mixed norm

    :param weight: length that one radian of turn counts for, defaults to 1
    :raises ValueError: ``weight`` is negative or not finite

    The metric of a vector :math:`(x, y, \theta)`, the difference between two
    poses, is

    .. math::

        d = \sqrt{x^2 + y^2 + (w \bar{\theta})^2}

    where :math:`\bar{\theta}` is :math:`\theta` wrapped to the interval
    :math:`[-\pi, \pi)` and :math:`w` is the ``weight``.

    Distance and angle have different units, so their relative importance is
    a choice.  The weight is the conversion: it is how much length a turn of one
    radian is worth, so a turn of :math:`\theta` costs as much as moving
    :math:`w \theta`.  A weight of 1 treats a radian as a metre, which is what
    the string ``"SE2"`` means.  A characteristic length of the vehicle, such as
    its turning radius, is a natural choice.  With a weight of 0 the
    orientation is ignored and this is the distance in the plane.

    An instance is a metric for a graph, and can be used as its heuristic too::

        g = UGraph(dim=3, metric=SE2Metric(weight=0.5))

    .. runblock:: pycon

        >>> from pgraph import SE2Metric
        >>> import numpy as np
        >>> v = np.r_[3, 4, np.pi / 2]   # 3 east, 4 north, a quarter turn
        >>> SE2Metric()(v)
        >>> SE2Metric(weight=0.5)(v)
        >>> SE2Metric(weight=0)(v)

    A metric with a weight is still a metric, so it is a consistent heuristic
    for A* when the edge costs are computed with it.  If edge costs come from
    somewhere else, use a weight low enough, or zero, for the heuristic never to
    exceed the cost to go.

    :seealso: :meth:`Graph.metric` :meth:`Graph.heuristic`
    """

    def __init__(self, weight: float = 1.0):
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"weight must be finite and not negative, got {weight!r}")
        self.weight = weight

    def __call__(self, v: NDArray) -> float:
        if len(v) != 3:
            raise ValueError(
                f"SE2 metric requires a 3-element (x, y, theta) vector, got length {len(v)}"
            )
        # wrap the angle to the range [-pi, pi), without changing the argument
        theta = (float(v[2]) + math.pi) % (2 * math.pi) - math.pi
        return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + (self.weight * theta) ** 2)

    def __repr__(self) -> str:
        return f"SE2Metric(weight={self.weight:g})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SE2Metric) and other.weight == self.weight

    def __hash__(self) -> int:
        return hash(("SE2Metric", self.weight))


#: the built-in metrics by name.  They are module level, and pickled by name,
#: so that a graph using one can be pickled.
_METRICS: dict[str, Callable[[NDArray], float]] = {
    "L1": _L1,
    "L2": _L2,
    "SE2": SE2Metric(),
}


#: curvature, as the ``rad`` of an ``arc3`` connection style, between successive
#: edges that join the same pair of vertices
_ARC_STEP = 0.35
#: radius in points of the first loop at a vertex, and the growth for each next
_LOOP_RADIUS = 20.0
_LOOP_GROWTH = 6.0
#: angle in degrees at which the first loop leaves a vertex, and the turn for each next
_LOOP_ANGLE = 60.0
_LOOP_TURN = 75.0


def _renamed(
    func: str, old_name: str, old: Any, new_name: str, new: Any, stacklevel: int = 3
) -> Any:
    """
    Resolve an option that has been renamed (private function)

    Returns ``new`` if the old name was not used.  Using the old name is still
    accepted, with a :exc:`DeprecationWarning` that names its replacement.
    Using both is a :exc:`TypeError`, since it is not clear which is meant.
    """
    if old is None:
        return new
    if new is not None:
        raise TypeError(
            f"{func}() got both {old_name} and {new_name}; "
            f"{old_name} is deprecated, use {new_name} only"
        )
    warnings.warn(
        f"{func}(): {old_name} is deprecated, use {new_name}",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return old


class Graph:
    """
    Base class for all graphs

    A graph is a set of :class:`Vertex` objects joined by :class:`Edge`
    objects, which are directed or not according to the kind of graph:
    :class:`UGraph` is undirected and :class:`DGraph` is directed.  This
    class holds everything they share, and ``Graph(directed=flag)`` makes one
    or the other.  The constructor arguments are described below.

    :seealso: :class:`UGraph` :class:`DGraph` :class:`Vertex` :class:`Edge`
    """

    #: whether the edges of this kind of graph are directed, set by
    #: :class:`UGraph` and :class:`DGraph`
    _directed: ClassVar[bool]

    def __new__(cls, *args: Any, directed: bool | None = None, **kwargs: Any) -> Graph:
        # Graph(directed=True) is a DGraph and Graph() a UGraph; the classes
        # underneath stay honest, so isinstance and the hooks work as usual
        if cls is Graph:
            cls = DGraph if directed else UGraph
        elif not hasattr(cls, "_directed"):
            raise TypeError(
                f"{cls.__name__} does not say whether its edges are directed; "
                "subclass UGraph or DGraph"
            )
        return super().__new__(cls)

    def __init__(
        self,
        *,
        metric: Callable[[NDArray], float] | str | None = None,
        heuristic: Callable[[NDArray], float] | str | None = None,
        verbose: bool = False,
        dim: int | None = None,
        vertex_class: type[Vertex] | None = None,
        edge_class: type[Edge] | None = None,
        directed: bool | None = None,
    ):
        """
        Graph, undirected or directed

        :param metric: distance metric, defaults to "L2"
        :type metric: callable or str, optional
        :param heuristic: heuristic distance metric for A*, defaults to the
            same as ``metric``
        :type heuristic: callable or str, optional
        :param verbose: print diagnostic information as vertices/edges are
            added, defaults to False
        :param dim: required length of every vertex's ``coord``, defaults to
            None (unconstrained -- vertices may have coordinates of any
            length, or none at all)
        :type dim: int, optional
        :param vertex_class: class of the vertices this graph creates, defaults
            to :class:`Vertex`
        :type vertex_class: subclass of :class:`Vertex`, optional
        :param edge_class: class of the edges this graph creates, defaults to
            :class:`Edge`
        :type edge_class: subclass of ``Edge``, optional
        :raises ValueError: ``dim`` is given but is not a positive integer
        :param directed: whether the edges are directed, defaults to False
            for ``Graph`` and is fixed by the class for :class:`UGraph` and
            :class:`DGraph`
        :type directed: bool, optional
        :raises ValueError: ``directed`` contradicts the class, for example
            ``UGraph(directed=True)``
        :raises TypeError: ``vertex_class`` is not a subclass of
            :class:`Vertex`, or ``edge_class`` is not a subclass of ``Edge``

        ``Graph(directed=True)`` is a :class:`DGraph` and ``Graph()`` a
        :class:`UGraph`, so the kind can be chosen by a variable::

            g = Graph(directed=is_directed)

        :class:`UGraph` and :class:`DGraph` remain, and are what
        ``isinstance`` and the documentation refer to.  Everything else, the
        vertices included, is the same for both kinds, so there is one vertex
        class, :class:`Vertex`.  A vertex has no direction of its own; the
        graph it belongs to decides how an edge is recorded at its ends.

        All arguments are keyword-only, so ``UGraph(dim=6)`` is correct and
        ``UGraph(6)`` raises :exc:`TypeError`.

        .. versionchanged:: 1.0
            Arguments are keyword-only.  In 0.6.5 the first positional
            argument was accepted and ignored, so ``UGraph(6)`` silently did
            nothing.

        ``vertex_class`` and ``edge_class`` let user subclasses be built by the
        graph, so they never need to be constructed by hand::

            g = UGraph(vertex_class=MyVertex, edge_class=MyEdge)
            v = g.add_vertex([0, 0])  # a MyVertex
            e = g.add_edge(v, w)      # a MyEdge

        The classes are called as ``vertex_class(coord, name=name)`` and
        ``edge_class(v1, v2, cost=cost, data=data)``, so a subclass that
        overrides ``__init__`` must accept and pass on those arguments.
        A graph given a ``vertex_class`` only accepts vertices of that class
        in :meth:`add_vertex`, and one given an ``edge_class`` only accepts
        edges of that class.

        :seealso: :class:`UGraph` :class:`DGraph` :meth:`add_vertex`
            :meth:`add_edge` :attr:`vertex_class` :attr:`edge_class`
        """
        if dim is not None and dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        if directed is not None and bool(directed) != type(self)._directed:
            raise ValueError(
                f"{type(self).__name__} is "
                f"{'directed' if type(self)._directed else 'undirected'}, "
                f"but directed={directed!r} was given"
            )
        if vertex_class is None:
            vertex_class = Vertex
        elif not (isinstance(vertex_class, type) and issubclass(vertex_class, Vertex)):
            raise TypeError(f"vertex_class must be a subclass of Vertex, got {vertex_class!r}")
        if edge_class is None:
            edge_class = Edge
        elif not (isinstance(edge_class, type) and issubclass(edge_class, Edge)):
            raise TypeError(f"edge_class must be a subclass of Edge, got {edge_class!r}")
        self._vertex_class: type[Vertex] = vertex_class
        self._edge_class: type[Edge] = edge_class
        # we use a list and a dict, the list respects the order of adding
        self._vertexlist: list[Vertex] = []
        self._vertexdict: dict[str, Vertex] = {}
        # an insertion-ordered set: the keys are the edges, hashed by identity
        # (Edge defines no __eq__), the values are unused.  This gives O(1)
        # removal and a defined order.
        self._edges: dict[Edge, None] = {}
        self._verbose = verbose
        self._dim = dim
        self._ncomponents = 0
        self._connectivitychange = False
        # the optional spatial index of build_kdtree(), and whether the
        # vertices have changed since it was built
        self._kdtree: _KDTree | None = None
        self._spatialchange = False
        # the coordinates of the embedded vertices as a matrix, built when
        # first needed and kept up to date by add_vertex; None if not built or
        # invalidated by a removal or an assignment to a ``coord``
        self._pts: _PointCache | None = None
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

        Counts of 1,000 or more have thousands separators.

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
        s = f"{self.__class__.__name__}: {self.n:,} {'vertex' if self.n==1 else 'vertices'}, {self.ne:,} edge{'s'[:self.ne^1]}, {self.nc:,} component{'s'[:self.nc^1]}"
        return s

    def __repr__(self) -> str:
        # NOTE: this is shadowed by the __repr__ defined further below in
        # this class, which is the one actually used -- kept as-is here to
        # avoid changing behaviour as part of a typing-only pass.
        return str(self)

    @classmethod
    def Dict(cls, d: dict, reverse: bool = False, **kwargs: Any) -> Graph:
        """
        Create graph from parent/child dictionary

        :param d: dictionary that maps from ``Vertex`` subclass to ``Vertex`` subclass
        :type d: dict
        :param reverse: reverse link direction, defaults to False
        :param kwargs: optional arguments passed to the graph constructor, for
            example ``vertex_class`` or ``edge_class``
        :raises TypeError: ``vertex_class`` or ``edge_class`` is not a
            suitable class, see :class:`~pgraph.PGraph.Graph`
        :return: graph
        :rtype: UGraph or DGraph

        Behaves like a constructor for a ``DGraph`` or ``UGraph`` from a
        dictionary that maps vertices to parents.  From this information it
        can create a graph that is a tree or forest.

        By default parent vertices are linked their children. If ``reverse`` is
        True then children are linked to their parents.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> d = {'b': 'a', 'c': 'a', 'd': 'b'}
            >>> g = UGraph.Dict(d)
            >>> print(g)

        :seealso: :meth:`Adjacency`
        """

        g = cls(**kwargs)

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
        **kwargs: Any,
    ) -> Graph:
        """
        Create graph from adjacency matrix

        :param A: adjacency matrix
        :type A: ndarray(N,N)
        :param coords: coordinates of vertices, defaults to None
        :type coords: ndarray(N,M), optional
        :param names: names of vertices, defaults to None
        :type names: list(N) of str, optional
        :param kwargs: optional arguments passed to the graph constructor, for
            example ``vertex_class`` or ``edge_class``
        :raises ValueError: ``A`` is not square, or the length of ``names`` or
            the number of rows of ``coords`` does not match ``A``
        :raises TypeError: ``vertex_class`` or ``edge_class`` is not a
            suitable class, see :class:`~pgraph.PGraph.Graph`
        :return: graph
        :rtype: UGraph or DGraph

        Behaves like a constructor for a ``DGraph`` or ``UGraph`` creating the graph from an adjacency matrix ``A``.
        Here non-zero elements ``A[i,j]``
        correspond to edges from vertex ``i`` to vertex ``j``.

        A non-zero diagonal element ``A[i,i]`` is a self-loop on vertex ``i``,
        for both kinds of graph.

        .. warning:: For an undirected graph ``A`` should be symmetric but this
            is not checked.  Only the upper triangular part, including the
            diagonal, is used. Elements are costs, not edge counts: a non-zero
            element is a single edge with that element as its cost, so
            parallel edges cannot be created this way.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
            >>> g = UGraph.Adjacency(A)
            >>> print(g)

        :seealso: :meth:`Dict` :meth:`adjacency`
        """

        if A.shape[0] != A.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if names is not None and len(names) != A.shape[0]:
            raise ValueError("length of names must match dimension of adjacency matrix")
        if coords is not None and coords.shape[0] != A.shape[0]:
            raise ValueError("coords must have same number of rows as adjacency matrix")

        g = cls(**kwargs)

        name = None
        coord = None
        for i in range(A.shape[0]):
            if names is not None:
                name = names[i]
            if coords is not None:
                coord = coords[i, :]
            g.add_vertex(name=name, coord=coord)

        if not g.directed:
            # undirected graph: upper triangle including the diagonal
            for i in range(A.shape[0]):
                for j in range(i, A.shape[1]):
                    if A[i, j] > 0:
                        g[i].connect(g[j], cost=A[i, j])
        else:
            # directed graph: whole matrix, a diagonal element is a self-loop
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if A[i, j] > 0:
                        g[i].connect(g[j], cost=A[i, j])

        return g

    @property
    def directed(self) -> bool:
        """
        Test if the graph is directed

        :return: True for a :class:`DGraph`, False for a :class:`UGraph`
        :rtype: bool

        An undirected edge is recorded at both of its vertices, and is
        followed in either direction.  A directed edge is recorded as leaving
        its start vertex and entering its end vertex.

        .. runblock:: pycon

            >>> from pgraph import Graph
            >>> Graph().directed
            >>> Graph(directed=True).directed

        :seealso: :class:`UGraph` :class:`DGraph`
        """
        return self._directed

    # the hooks: the one place where undirected and directed graphs differ in
    # how an edge is held.  A vertex has no direction of its own, so its
    # connect() and the graph's remove_edge() ask the graph.

    def _attach(self, edge: Edge) -> None:
        """
        Record a new edge in the edge lists of its vertices (private method)

        :param edge: the edge just created or supplied to
            :meth:`Vertex.connect`, with ``edge.v1`` and ``edge.v2`` set and
            already part of the graph
        :raises NotImplementedError: not provided by this graph kind

        This is the hook that :meth:`Vertex.connect` calls once the connection
        has been validated and the edge registered with the graph, so it is
        never called for a connection that is refused.  :class:`UGraph`
        records the edge at both ends and :class:`DGraph` as leaving its start
        and entering its end.  It is defined by the graph kind, not by
        :meth:`Vertex.connect`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define how an edge is attached; "
            "subclass UGraph or DGraph"
        )

    def _detach(self, edge: Edge) -> None:
        """
        Remove an edge from the edge lists of its vertices (private method)

        :param edge: the edge being removed, with ``edge.v1`` and ``edge.v2``
            still set
        :raises NotImplementedError: not provided by this graph kind

        The twin of :meth:`_attach`: it undoes exactly what that recorded.
        Called by :meth:`remove_edge` before the edge's endpoints are
        cleared.  An edge that is absent from a list is skipped rather than
        treated as an error.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define how an edge is detached; "
            "subclass UGraph or DGraph"
        )

    def _incoming(self, vertex: Vertex) -> list[Edge]:
        """
        The edges that end at a vertex (private method)

        :param vertex: a vertex of this graph
        :raises NotImplementedError: not provided by this graph kind
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define the incoming edges; "
            "subclass UGraph or DGraph"
        )

    def _endpoints_match(self, edge: Edge, v1: Vertex, v2: Vertex) -> bool:
        """
        Test if an edge already connects two vertices (private method)

        A directed edge must run from ``v1`` to ``v2``; see
        :meth:`UGraph._endpoints_match` for the undirected relaxation.
        """
        return edge.v1 is v1 and edge.v2 is v2

    @property
    def vertex_class(self) -> type[Vertex]:
        """
        Class of the vertices this graph creates

        :return: vertex class
        :rtype: subclass of ``Vertex``

        This is read-only.  It is set by the ``vertex_class`` argument of the
        graph constructor, see :class:`Graph`, and is :class:`Vertex` if not
        given.

        .. runblock:: pycon

            >>> from pgraph import UGraph, Vertex
            >>> class MyVertex(Vertex):
            ...     pass
            >>> g = UGraph(vertex_class=MyVertex)
            >>> print(g.vertex_class.__name__)

        :seealso: :meth:`add_vertex` :attr:`edge_class`
        """
        return self._vertex_class

    @property
    def edge_class(self) -> type[Edge]:
        """
        Class of the edges this graph creates

        :return: edge class
        :rtype: subclass of ``Edge``

        This is read-only.  It is set by the ``edge_class`` argument of the
        graph constructor, see :class:`Graph`, and is :class:`Edge` if not
        given.

        .. runblock:: pycon

            >>> from pgraph import UGraph, Edge
            >>> class MyEdge(Edge):
            ...     pass
            >>> g = UGraph(edge_class=MyEdge)
            >>> print(g.edge_class.__name__)

        :seealso: :meth:`add_edge` :attr:`vertex_class`
        """
        return self._edge_class

    def validate(self, embedded: bool = False) -> list[str]:
        """
        Check that the graph's records agree with each other

        :param embedded: also report a vertex that has no coordinate, defaults
            to False
        :return: a description of each inconsistency found, empty if the graph
            is consistent
        :rtype: list of str

        A graph keeps the same facts in several places: its set of edges, each
        vertex's list of outgoing edges (and, for a directed graph, of incoming
        ones), each edge's two vertices, each vertex's pointer back to its
        graph, and the table of vertex names.  The library keeps them in step,
        so a graph built and edited through the public methods is always
        valid.  This is a check for tests, for debugging, and for code that
        reaches into the internals.

        It does not change the graph, and takes time proportional to the
        number of vertices plus edges.  It reports every inconsistency, not
        only the first.

        The checks are:

        - the vertex list holds only vertices, once each, and the edge set
          holds only edges
        - every edge in the graph has two vertices, both in this graph, and is
          an instance of :attr:`edge_class`
        - every vertex points back at this graph, is an instance of
          :attr:`vertex_class`, is in the name table under its own name, and
          has a coordinate of the required length if ``dim`` was given
        - every coordinate is a finite one-dimensional numeric vector, and if
          ``dim`` was not given they all have the same length
        - with ``embedded=True``, every vertex has a coordinate.  A graph with
          no coordinates is valid, but one that is half embedded rarely is
        - every entry of the name table is a vertex of this graph and is
          filed under its own name
        - each edge is in the right vertex lists, the right number of times:
          both ends for an undirected edge (twice at one vertex for a
          self-loop), the source's outgoing and the target's incoming list for
          a directed edge
        - no vertex list holds an edge that is not in the graph

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(name='v1')
            >>> v2 = g.add_vertex(name='v2')
            >>> g.add_edge(v1, v2)
            >>> print(g.validate())

        :seealso: :meth:`remove_edge` :meth:`remove_vertex`
        """
        from collections import Counter

        problems: list[str] = []
        directed = self.directed

        # ---- the containers themselves must hold the right kind of thing; a
        # stray object is reported and set aside so the checks below can run
        vertices: list[Vertex] = []
        for entry in self._vertexlist:
            if isinstance(entry, Vertex):
                vertices.append(entry)
            else:
                problems.append(f"the vertex list holds {entry!r}, which is not a vertex")
        graph_edges: list[Edge] = []
        for member in self._edges:
            if isinstance(member, Edge):
                graph_edges.append(member)
            else:
                problems.append(f"the graph's edges hold {member!r}, which is not an edge")

        # ---- vertices
        members = {id(v) for v in vertices}
        for vid, n in Counter(id(v) for v in vertices).items():
            if n > 1:
                v = next(x for x in vertices if id(x) == vid)
                problems.append(f"vertex {v.name!r} is listed {n} times in the vertex list")
        for key, v in self._vertexdict.items():
            if not isinstance(v, Vertex) or id(v) not in members:
                problems.append(
                    f"name table entry {key!r} refers to a vertex that is not in the graph"
                )
            elif v.name != key:
                problems.append(
                    f"name table entry {key!r} refers to vertex {v.name!r}, "
                    "which has a different name"
                )
        length: int | None = None  # of the coordinates, if dim was not given
        for v in vertices:
            if v._graph is not self:
                problems.append(f"vertex {v.name!r} does not point back at this graph")
            if v.name is None or self._vertexdict.get(v.name) is not v:
                problems.append(
                    f"vertex {v.name!r} is not in the name table under its own name"
                )
            if not isinstance(v, self._vertex_class):
                problems.append(
                    f"vertex {v.name!r} is a {type(v).__name__}, not a "
                    f"{self._vertex_class.__name__}"
                )
            if (
                self._dim is not None
                and v.coord is not None
                and len(v.coord) != self._dim
            ):
                problems.append(
                    f"vertex {v.name!r} has a coordinate of length "
                    f"{len(v.coord)} but the graph requires dim={self._dim}"
                )
            if v.coord is None:
                if embedded:
                    problems.append(f"vertex {v.name!r} has no coordinate")
            else:
                try:
                    coord = np.asarray(v.coord, dtype=float)
                except (TypeError, ValueError):
                    problems.append(f"vertex {v.name!r} has a coordinate that is not numeric")
                else:
                    if coord.ndim != 1:
                        problems.append(
                            f"vertex {v.name!r} has a coordinate of shape "
                            f"{coord.shape}, not a vector"
                        )
                    elif not np.all(np.isfinite(coord)):
                        problems.append(
                            f"vertex {v.name!r} has a coordinate that is not finite"
                        )
                    elif self._dim is None:
                        if length is None:
                            length = len(coord)
                        elif len(coord) != length:
                            problems.append(
                                f"vertex {v.name!r} has a coordinate of length "
                                f"{len(coord)}, but others have {length}"
                            )
            if not directed and v._inlist:
                problems.append(
                    f"vertex {v.name!r} has {len(v._inlist)} incoming edges "
                    "recorded, but an undirected vertex does not use that list"
                )

        # ---- edges, and what each vertex list should then contain
        expected_out: Counter[tuple[int, int]] = Counter()
        expected_in: Counter[tuple[int, int]] = Counter()
        edges: dict[int, Edge] = {}
        in_graph = {id(e) for e in graph_edges}
        for e in graph_edges:
            edges[id(e)] = e
            if e.v1 is None or e.v2 is None:
                problems.append(f"edge {e!r} is in the graph but has no vertices")
                continue
            for end in (e.v1, e.v2):
                if id(end) not in members:
                    problems.append(
                        f"edge {e!r} joins vertex {end.name!r}, which is not in this graph"
                    )
            if not isinstance(e, self._edge_class):
                problems.append(
                    f"edge {e!r} is a {type(e).__name__}, not a {self._edge_class.__name__}"
                )
            expected_out[(id(e.v1), id(e))] += 1
            if directed:
                expected_in[(id(e.v2), id(e))] += 1
            else:
                expected_out[(id(e.v2), id(e))] += 1

        actual_out: Counter[tuple[int, int]] = Counter()
        actual_in: Counter[tuple[int, int]] = Counter()
        owner = {id(v): v for v in vertices}
        for v in vertices:
            for listname, held, counter in (
                ("outgoing" if directed else "edge", v._edgelist, actual_out),
                ("incoming", v._inlist, actual_in),
            ):
                for e in held:
                    if not isinstance(e, Edge):
                        problems.append(
                            f"vertex {v.name!r} holds {e!r} in its {listname} list, "
                            "which is not an edge"
                        )
                        continue
                    counter[(id(v), id(e))] += 1
                    edges.setdefault(id(e), e)

        def report(expected: Counter, actual: Counter, listname: str) -> None:
            for key in expected.keys() | actual.keys():
                vid, eid = key
                want, have = expected[key], actual[key]
                if want == have:
                    continue
                v, e = owner.get(vid), edges[eid]
                vname = v.name if v is not None else "?"
                if eid not in in_graph:
                    problems.append(
                        f"vertex {vname!r} holds edge {e!r} in its {listname} "
                        "list, but that edge is not in the graph"
                    )
                elif have < want:
                    problems.append(
                        f"edge {e!r} is missing from the {listname} list of "
                        f"vertex {vname!r} (found {have}, expected {want})"
                    )
                else:
                    problems.append(
                        f"edge {e!r} appears {have} times in the {listname} list "
                        f"of vertex {vname!r}, expected {want}"
                    )

        report(expected_out, actual_out, "outgoing" if directed else "edge")
        if directed:
            report(expected_in, actual_in, "incoming")

        return problems

    def copy(self) -> Graph:
        """
        Deepcopy of graph

        :return: deep copy
        :rtype: Graph

        The copy has its own vertices and edges, which keep their class, name,
        coordinate, cost and any attributes the user has added, and in the
        same order, so it can be modified without affecting the original.

        A graph can also be pickled, and copied with :func:`copy.deepcopy`,
        which this method calls.  Both work for a graph of any size.  A
        vertex or an edge cannot be pickled or copied on its own, since it has
        no meaning apart from its graph: a :exc:`TypeError` is raised, and the
        graph should be copied instead.

        .. note::

            - A metric or heuristic given by name (``"L1"``, ``"L2"``,
              ``"SE2"``) is pickled by name, and an :class:`SE2Metric` with
              its weight.  A function you supply must be
              importable at module level to be pickled; a ``lambda`` or a
              function defined inside another function can be copied but not
              pickled.
            - Attributes that you add to vertices or edges must not refer to
              other vertices or edges of the graph, since those cannot be
              pickled on their own.
            - A pickle is not a long-term storage format.  Use :meth:`export`
              for that.

        .. versionchanged:: 1.0
            Pickling a graph works, and copying one no longer fails with a
            ``RecursionError`` for a graph with a path of about a hundred
            vertices.

        Copies the graph, its vertices and edges, but not any external data that may be
        attached to them.  The copy is a new graph of the same kind, with the same
        vertices and edges, but the vertices and edges are new objects.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> g2 = g.copy()
            >>> g2 is g
            >>> g2[0] is g[0]

        :seealso: :meth:`vertex_copy` :meth:`export`
        """
        return copy.deepcopy(self)

    def subgraph(self, vertices: Iterable[Vertex | str] | NDArray) -> Graph:
        """
        Subgraph induced by a set of vertices

        :param vertices: vertices of this graph, or their names, or a boolean
            mask with one element per vertex
        :type vertices: iterable of Vertex subclass or str, or ndarray(n) of bool
        :raises ValueError: a vertex is not in this graph, or a mask is not
            the length of the number of vertices
        :raises KeyError: a name is not that of a vertex in this graph
        :raises TypeError: an item is neither a vertex nor a string
        :return: a new graph of the same kind
        :rtype: Graph

        The new graph has the given vertices and every edge of this graph that
        has *both* its end vertices among them.  An edge that has one end
        outside is removed, so the edges that cross the border of a region
        are dropped.  Parallel edges and loops are kept.

        Like :meth:`copy` the new graph is independent of this one: its
        vertices and edges are new objects that keep their class, name,
        coordinate, cost and any attributes that were added, and the graph
        keeps its metric, heuristic and vertex and edge classes.  They are in
        the order of this graph, however ``vertices`` was ordered.  Names are
        kept, so the vertex ``g[name]`` has its counterpart ``h[name]``.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v = [g.add_vertex(coord=[i, 0], name=f"v{i}") for i in range(4)]
            >>> for i in range(3):
            ...     g.add_edge(v[i], v[i + 1])
            >>> h = g.subgraph([v[1], v[2]])
            >>> print(h)
            >>> h.edges()

        It is a single pass over the graph.  The vertices might be a component,
        the vertices that satisfy a test, or those in a region of an embedded
        graph from :meth:`within`.

        A boolean mask, such as the result of a comparison on
        :meth:`coordinates`, selects the vertices by their position in the
        graph, the ``True`` elements being kept::

            h = g.subgraph(g.coordinates()[0] > 0)   # the vertices with x > 0

        The mask must be of boolean type (a list of ``bool`` also works) and of
        length ``g.n``.  A vector of 0 and 1 of integer type is not accepted, since
        it could as well be a list of indices.

        :seealso: :meth:`copy` :meth:`within` :meth:`component`
        """
        mask = None
        if isinstance(vertices, np.ndarray) and vertices.dtype == np.bool_:
            mask = vertices
        elif (
            isinstance(vertices, (list, tuple))
            and len(vertices) > 0
            and all(isinstance(x, (bool, np.bool_)) for x in vertices)
        ):
            mask = np.array(vertices, dtype=bool)

        if mask is not None:
            if mask.shape != (len(self._vertexlist),):
                raise ValueError(
                    f"a mask must have one element per vertex, {len(self._vertexlist)}, "
                    f"not shape {mask.shape}"
                )
            keep = [int(i) for i in np.flatnonzero(mask)]
        else:
            wanted = set()
            items: Iterable[Any] = vertices
            for item in items:
                v = self._resolve_vertex(item, "vertex")
                if v._graph is not self:
                    raise ValueError(f"vertex {v.name!r} is not in this graph")
                wanted.add(id(v))
            keep = [i for i, v in enumerate(self._vertexlist) if id(v) in wanted]

        state = self.__getstate__()
        flat = state["_flat"]
        renumber = {old: new for new, old in enumerate(keep)}
        flat["vertices"] = [flat["vertices"][i] for i in keep]
        flat["edges"] = [
            (cls, renumber[i], renumber[j], attrs)
            for cls, i, j, attrs in flat["edges"]
            if i in renumber and j in renumber
        ]

        h = type(self).__new__(type(self))
        h.__setstate__(copy.deepcopy(state))
        h._connectivitychange = True  # the components are those of the new graph
        return h

    # pickling and copying

    #: the graph's own containers, which are saved as a flat description
    _STRUCTURE = ("_vertexlist", "_vertexdict", "_edges", "_kdtree", "_pts")

    #: the references a vertex holds, which are rebuilt rather than saved
    _VERTEX_LINKS = ("_edgelist", "_inlist", "_graph")

    def __getstate__(self) -> dict[str, Any]:
        """
        State for pickling and deep copying (private method)

        The vertices and edges form a web of references, and following them
        recursively overflows the stack for a graph with a path of a hundred
        or so vertices.  Instead the graph is described flatly: a list of
        vertices, and a list of edges that refer to their end vertices by
        index.  The edges are listed in the order they were added, so
        replaying them rebuilds each vertex's edge lists in their original
        order.  A built-in metric is saved by name.
        """
        state = {k: v for k, v in self.__dict__.items() if k not in self._STRUCTURE}
        for key in ("_metric", "_heuristic"):
            for name, func in _METRICS.items():
                if state[key] is func:
                    state[key] = ("builtin", name)
        index = {id(v): i for i, v in enumerate(self._vertexlist)}
        state["_flat"] = {
            "version": 1,
            "vertices": [
                (
                    type(v),
                    {k: x for k, x in v.__dict__.items() if k not in self._VERTEX_LINKS},
                )
                for v in self._vertexlist
            ],
            "edges": [
                (
                    type(e),
                    index[id(e.v1)],
                    index[id(e.v2)],
                    {k: x for k, x in e.__dict__.items() if k not in ("_v1", "_v2")},
                )
                for e in self._edges
            ],
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Rebuild the graph from its flat description (private method)

        The vertex and edge objects are made without calling ``__init__``, so
        subclasses with their own constructor signature are restored as they
        were.  Each edge is attached with the vertex's own hook, exactly as
        :meth:`Vertex.connect` does.
        """
        state = dict(state)
        flat = state.pop("_flat")
        if flat["version"] != 1:
            raise ValueError(f"unknown pickled graph version {flat['version']!r}")
        for key in ("_metric", "_heuristic"):
            value = state[key]
            if isinstance(value, tuple) and value[0] == "builtin":
                state[key] = _METRICS[value[1]]
        self.__dict__.update(state)
        self._kdtree = None
        self._pts = None
        self._spatialchange = False

        self._vertexlist = []
        self._vertexdict = {}
        for cls, attrs in flat["vertices"]:
            v = cls.__new__(cls)
            v.__dict__.update(attrs)
            v._edgelist = []
            v._inlist = []
            v._graph = self
            self._vertexlist.append(v)
            if v._name is not None:
                self._vertexdict[v._name] = v

        self._edges = {}
        for cls, i, j, attrs in flat["edges"]:
            e = cls.__new__(cls)
            e.__dict__.update(attrs)
            v1, v2 = self._vertexlist[i], self._vertexlist[j]
            e._set_endpoints(v1, v2)
            self._edges[e] = None
            self._attach(e)

    def add_vertex(
        self, coord: ArrayLike | Vertex | None = None, name: str | None = None
    ) -> Vertex:
        """
        Add a vertex to the graph

        :param coord: coordinate for an embedded graph, or an existing vertex
            (an instance of this graph's :attr:`vertex_class`, not belonging
            to a graph) to add as-is, defaults to None
        :type coord: array-like or Vertex subclass, optional
        :param name: name of vertex, defaults to "#i"
        :type name: str, optional
        :raises TypeError: ``coord`` is a ``Vertex`` that is not an
            instance of this graph's :attr:`vertex_class`
        :raises ValueError: the name is already used by another vertex in the
            graph, or ``coord`` is a vertex that already belongs to a graph
        :raises ValueError: the graph was constructed with ``dim``, and
            ``coord`` is given but its length doesn't match
        :return: the added vertex
        :rtype: Vertex subclass

        This is a graph-centric way of creating a vertex.

        - ``g.add_vertex()`` creates a new vertex with optional ``coord`` and
          ``name``.
        - ``g.add_vertex(v)`` takes an instance of this graph's
          :attr:`vertex_class` (by default any :class:`Vertex`) and adds it
          to the graph.

        If the vertex has no name and ``name`` is None give it a default name
        ``#N`` where ``N`` is the number of vertices, or the next integer above
        it that is not already used by a name in the graph.

        Names are unique within a graph, since they are how ``g[name]`` finds a
        vertex, and a vertex belongs to one graph at a time.

        .. versionchanged:: 1.0
            Adding a vertex whose name is already used raises ``ValueError``.
            It used to replace the earlier vertex in the name table, leaving it
            in the graph but unreachable by name.  A default name could also
            clash with an existing one after a vertex had been removed.

        The vertex is placed into the graph's vertex dictionary with a key equal to its name.

        .. runblock:: pycon

            >>> from pgraph import UGraph, Vertex
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0])
            >>> repr(v1)
            >>> print(v1)
            >>> print(v1.name)
            >>> v2 = g.add_vertex(Vertex(coord=[1,1], name='v2'))
            >>> print(v2.name)
            >>> g['v2']

        If the graph was constructed with a required ``dim`` (see
        :class:`Graph`), every embedded vertex must have a
        coordinate of exactly that length -- adding one of the wrong length
        raises ``ValueError``:

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph(dim=6)
            >>> v1 = g.add_vertex(coord=[0, 0, 0, 0, 0, 0], name='pose1')
            >>> print(v1)

        .. note:: This single implementation, shared by :class:`UGraph` and
            :class:`DGraph`, is parameterized by the graph's :attr:`vertex_class`
            rather than duplicated per subclass -- see :doc:`policy` for why that
            matters.

        :seealso: :meth:`vertex_copy`
        """
        if isinstance(coord, self._vertex_class):
            vertex = coord
            if vertex._graph is not None:
                raise ValueError(
                    f"vertex {vertex.name!r} already belongs to a graph; "
                    "remove it from that graph first"
                )
        elif isinstance(coord, Vertex):
            raise TypeError(
                f"expecting {self._vertex_class.__name__} or coordinate data, "
                f"got {type(coord).__name__}"
            )
        else:
            vertex = self._vertex_class(coord, name=name)

        if (
            self._dim is not None
            and vertex.coord is not None
            and len(vertex.coord) != self._dim
        ):
            raise ValueError(
                f"vertex coord has length {len(vertex.coord)}, "
                f"but this graph requires dim={self._dim}"
            )

        if name is None:
            name = vertex.name
        if name is None:
            # the first #N not already used: a removal makes len() reuse a name
            n = len(self._vertexlist)
            while f"#{n}" in self._vertexdict:
                n += 1
            name = f"#{n}"
        elif name in self._vertexdict:
            raise ValueError(f"the graph already has a vertex named {name!r}")
        # the vertex is in no graph yet, so the name table and the spatial index
        # have nothing to be told; skip the property and ``__setattr__``
        vertex.__dict__["_name"] = name
        self._vertexlist.append(vertex)
        self._vertexdict[name] = vertex
        if self._verbose:
            print(f"New vertex {name}: {vertex.coord}")
        vertex.__dict__["_graph"] = self
        self._connectivitychange = True
        self._spatialchange = True
        pts = self._pts
        if pts is not None and vertex.coord is not None:
            # keep the matrix of coordinates up to date; one that does not
            # fit it (another length, not finite) is left for _points to
            # judge, by dropping the matrix
            coord = vertex.coord
            if coord.ndim == 1 and coord.shape[0] == pts.dim and np.isfinite(coord).all():
                pts.append(vertex, coord)
            else:
                self._pts = None
        return vertex

    @classmethod
    def vertex_copy(cls, vertex: Vertex) -> Vertex:
        """
        Copy a vertex for use in a new graph of this kind

        :param vertex: vertex to copy
        :type vertex: Vertex subclass
        :return: new, unconnected vertex with the same coordinate and name
        :rtype: Vertex subclass

        A vertex can only belong to a single graph, so this method is used to
        create a new vertex with the same name and coordinates for inclusion
        in a new graph.  A vertex is the same for undirected and directed graphs.

        .. runblock:: pycon

            >>> from pgraph import UGraph, DGraph
            >>> g = UGraph()
            >>> v = g.add_vertex(coord=[1,2], name='v1')
            >>> newv = DGraph.vertex_copy(v)
            >>> print(newv)

        :seealso: :meth:`Vertex.copy`
        """
        return Vertex(coord=vertex.coord, name=vertex.name)

    def _resolve_vertex(self, v: Vertex | str, label: str) -> Vertex:
        """
        Resolve a vertex given by reference or name (private method)

        :param v: vertex, or the name of a vertex in this graph
        :type v: Vertex subclass or str
        :param label: parameter name to use in the error message, e.g. "start"
        :raises TypeError: ``v`` is neither a ``Vertex`` nor a string
        :return: the resolved vertex
        :rtype: Vertex subclass

        Used by :meth:`add_edge`, :meth:`path_BFS`, :meth:`path_UCS` and
        :meth:`path_Astar` for their vertex-or-name parameters -- previously
        each method duplicated this check inline, which had let a
        copy-paste mistake (checking the wrong parameter's type in the
        error-raising branch) slip into all three path-finding methods
        unnoticed.
        """
        if isinstance(v, str):
            return self[v]
        elif isinstance(v, Vertex):
            return v
        else:
            raise TypeError(f"{label} must be Vertex subclass or string name")

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

    @overload
    def add_edge(self, v1: Edge, **kwargs: Any) -> Edge: ...

    @overload
    def add_edge(
        self, v1: Vertex | str, v2: Vertex | str, **kwargs: Any
    ) -> Edge: ...

    def add_edge(
        self,
        v1: Edge | Vertex | str,
        v2: Vertex | str | None = None,
        **kwargs: Any,
    ) -> Edge:
        """
        Add an edge to the graph (base class method)

        :param v1: first vertex (start if a directed graph), or an existing edge that
            already carries its two vertices
        :type v1: Vertex subclass, str or Edge
        :param v2: second vertex (end if a directed graph), must be omitted if ``v1`` is
            an edge
        :type v2: Vertex subclass or str, optional
        :param kwargs: optional arguments to pass to ``Vertex.connect``, not allowed
            if ``v1`` is an edge
        :raises TypeError: ``v1`` or ``v2`` is of the wrong type, or ``v2``, ``cost`` or
            ``data`` is given along with an edge
        :raises ValueError: the edge has no vertices, or the vertices do not belong to
            this graph
        :return: edge
        :rtype: Edge

        This is a graph-centric way of creating an edge.  The alternative is the
        ``connect`` method of a vertex.

        - ``g.add_edge(v1, v2)`` creates an edge between a vertex pair and adds it to
          the graph.
        - ``g.add_edge(e)`` takes an edge, or an instance of an ``Edge`` subclass, that
          was constructed with its two vertices and adds it to the graph, just as
          ``g.add_vertex(v)`` takes an existing vertex.

        .. runblock:: pycon
            :linenos:

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v3 = g.add_vertex(coord=[2,2], name='v3')
            >>> e = g.add_edge(v1, v2)
            >>> print(e)
            >>> e2 = g.add_edge('v2', 'v3', cost=99)
            >>> print(e2)

        Note that at line 9 the vertices are specified by their string names.

        Adding an instance of a user-defined ``Edge`` subclass:

        .. runblock:: pycon
            :linenos:

            >>> from pgraph import UGraph, Edge
            >>> class MyEdge(Edge):
            ...     pass
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> e = g.add_edge(MyEdge(v1, v2, data="a road"))
            >>> print(type(e).__name__, e.data)

        .. note:: Vertices have coordinates and the edge ``cost`` defaults to the distance between them, but if the vertices have no coordinates the edge cost must be set explicitly, otherwise it will be ``None`` and path planning will fail with a ``ValueError``.
      
        .. note:: To supply an edge that does not yet have vertices use
            ``g.add_edge(v1, v2, edge=e)`` or ``e.connect(v1, v2)``.

        :seealso: :meth:`add_vertex` :meth:`Edge.connect` :meth:`Vertex.connect`
        """
        if isinstance(v1, Edge):
            if v2 is not None:
                raise TypeError("v2 must not be given when adding an Edge")
            if kwargs:
                raise TypeError(
                    "add_edge(edge) takes no other arguments, got "
                    + ", ".join(kwargs)
                    + " -- set them on the edge itself"
                )
            edge = v1
            if edge.v1 is None or edge.v2 is None:
                raise ValueError(
                    "edge does not have both vertices; use "
                    "add_edge(v1, v2, edge=e) or e.connect(v1, v2)"
                )
            start, end = edge.v1, edge.v2
            kwargs = {"edge": edge}
        else:
            if v2 is None:
                raise TypeError("add_edge() requires two vertices, or an Edge")
            start = self._resolve_vertex(v1, "v1")
            end = self._resolve_vertex(v2, "v2")

        if start._graph is not self or end._graph is not self:
            raise ValueError("vertices must belong to this graph")

        if self._verbose:
            print(f"New edge from {start.name} to {end.name}")
        return start.connect(end, **kwargs)

    def remove_edge(self, edge: Edge) -> None:
        """
        Remove an edge from the graph

        :param edge: edge to remove
        :raises ValueError: ``edge`` does not belong to this graph

        The edge is removed from this graph's own edge collection and from
        the edge lists of its connected vertices, and ``edge.v1``/``edge.v2``
        are cleared to ``None``.

        .. warning:: The connectivity of the network may be changed.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> e = g.add_edge(v1, v2)
            >>> g.remove_edge(e)
            >>> print(g)

        .. note:: A directed edge is tracked only by its source vertex's
            edge list, not its target's (see :attr:`Vertex.edges`), so
            membership is checked per endpoint rather than assumed for both
            -- removing a directed edge from an undirected-only
            implementation would otherwise raise ``ValueError`` on the
            target side.

        :seealso: :meth:`remove_vertex` :meth:`Edge.remove`
        """
        if edge not in self._edges:
            raise ValueError("edge does not belong to this graph")
        v1, v2 = edge.v1, edge.v2
        if v1 is None or v2 is None:
            raise ValueError("edge does not connect two vertices")

        # how the edge is held differs between undirected and directed
        # vertices, so the vertex kind undoes what its _attach did
        self._detach(edge)

        v1._connectivitychange = True
        v2._connectivitychange = True
        self._connectivitychange = True

        edge._set_endpoints(None, None)

        del self._edges[edge]

    def remove_vertex(self, vertex: Vertex) -> None:
        """
        Remove a vertex, and all its edges, from the graph

        :param vertex: vertex to remove
        :raises ValueError: ``vertex`` does not belong to this graph

        Every edge touching ``vertex`` -- incoming or outgoing -- is removed
        via :meth:`remove_edge`, then the vertex itself is removed.

        .. warning:: The connectivity of the network may be changed.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> g.add_edge(v1, v2)
            >>> g.remove_vertex(v2)
            >>> print(g)

        .. note:: This scans the graph's own edge set for edges touching
            ``vertex``, rather than iterating ``vertex.edges()`` --
            for a ``DGraph`` vertex that only ever reports outgoing edges
            (see :attr:`Vertex.edges`), so incoming edges would
            otherwise be missed and left dangling.

        :seealso: :meth:`remove_edge` :meth:`Vertex.remove`
        """
        if vertex._graph is not self:
            raise ValueError("vertex does not belong to this graph")
        assert vertex.name is not None

        # its own lists name every edge at it, so no need to search the graph
        for edge in vertex._incident_edges():
            self.remove_edge(edge)

        self._vertexlist.remove(vertex)
        del self._vertexdict[vertex.name]
        vertex._graph = None  # it is not in any graph now
        vertex._label = None  # and so not in a component
        self._connectivitychange = True
        self._spatialchange = True
        self._pts = None

    def remove(self, x: Edge | Vertex) -> None:
        """
        Remove element from graph (deprecated)

        :param x: element to remove from graph
        :type x: Edge or Vertex subclass
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
        elif isinstance(x, Vertex):
            self.remove_vertex(x)
        else:
            raise TypeError("expecting Edge or Vertex")

    def show(self) -> None:
        """
        Print a summary of all vertices and edges to stdout

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> g.add_edge(v1, v2)
            >>> g.show()

        :seealso: :meth:`__str__`
        """
        print("vertices:")
        for v in self._vertexlist:
            print("  " + str(v))
        print("edges:")
        for e in self._edges:
            print("  " + str(e))

    @property
    def n(self) -> int:
        """
        Number of vertices

        :return: Number of vertices
        :rtype: int

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> g.add_vertex(name='v1')
            >>> g.add_vertex(name='v2')
            >>> print(g.n)
        """
        return len(self._vertexdict)

    @property
    def ne(self) -> int:
        """
        Number of edges

        :return: Number of vertices
        :rtype: int

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(name='v1')
            >>> v2 = g.add_vertex(name='v2')
            >>> g.add_edge(v1, v2)
            >>> print(g.ne)
        """
        return len(self._edges)

    def _graphcolor(self) -> None:
        """
        Label the components of the graph (private method)

        Assigns every vertex a ``label``, the number of its component, from 0
        in the order of the first vertex of each component, and sets the
        number of components.  Components are *weakly* connected: the
        direction of an edge is ignored, so a vertex is in the component of
        every vertex that it has a path to or from.

        This is a single pass over the edges with a union-find, nearly O(E),
        and is only done again when the graph has changed since, which every
        operation that adds or removes a vertex or edge records in the graph's
        ``_connectivitychange``.

        Looking at the two ends of each edge, rather than walking the
        neighbours of each vertex, makes the direction of an edge irrelevant
        and avoids a call per incident edge to find the far end, which is most
        of the time of a traversal.  The root of a component is its first
        vertex, so numbering the roots in order of the vertices gives the
        numbering described above.

        :seealso: :meth:`nc` :meth:`component`
        """
        if not self._connectivitychange:
            return

        vertices = self._vertexlist
        index = {v: i for i, v in enumerate(vertices)}
        parent = list(range(len(vertices)))

        for edge in self._edges:
            a = index[edge._v1]  # type: ignore[index]
            b = index[edge._v2]  # type: ignore[index]
            while parent[a] != a:  # find the root, halving the path as we go
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a < b:  # the smaller index is the root, so the first vertex
                parent[b] = a
            elif b < a:
                parent[a] = b

        count = 0
        label = [0] * len(vertices)
        for i, vertex in enumerate(vertices):
            root = i
            while parent[root] != root:
                parent[root] = parent[parent[root]]
                root = parent[root]
            if root == i:  # the first vertex of a component
                label[i] = count
                count += 1
            # the graph-level flag is what is consulted; writing the
            # attributes directly avoids the hook of ``Vertex.__setattr__``
            attrs = vertex.__dict__
            attrs["_label"] = label[root]
            attrs["_connectivitychange"] = False

        self._ncomponents = count
        self._connectivitychange = False

    @property
    def nc(self) -> int:
        """
        Number of components

        :return: Number of components
        :rtype: int

        Number of components in the graph. Components are labeled from 0 to ``g.nc-1``.

        For a directed graph the components are *weakly* connected: the
        direction of the edges is ignored, so two vertices are in the same
        component if there is a path between them that may run against the
        arrows.  A graph with no vertices has no components.

        .. versionchanged:: 1.0
            Does not hang.  For many directed graphs, the smallest being the
            4-cycle, the algorithm of 0.6.5 never finished, which also stopped
            :meth:`component`, :meth:`samecomponent` and :meth:`plot`.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(name='v1')
            >>> v2 = g.add_vertex(name='v2')
            >>> v3 = g.add_vertex(name='v3')
            >>> g.add_edge(v1, v2)
            >>> print(g.nc)

        .. note:: A lazy approach is used, and if a connectivity changing
            operation has been performed since the last call, the graph
            coloring algorithm is run which is potentially expensive for
            a large graph.

        :seealso: :meth:`component` :meth:`samecomponent`
        """
        self._graphcolor()
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
        :meth:`Vertex.distance`).

        If ``metric`` is already a callable matching this signature, it is
        returned unchanged. Otherwise it must be one of the built-in names
        "L1", "L2", "SE2" (see :meth:`metric` for their definitions).

        :seealso: :meth:`metric` :meth:`heuristic`
        """

        if callable(metric):
            return metric
        elif isinstance(metric, str):
            if metric in _METRICS:
                return _METRICS[metric]
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
            point passed to :meth:`closest`/:meth:`Vertex.distance` -- to be
            exactly 3 elements; raises :exc:`ValueError` otherwise.  Radians
            and metres are weighted equally; to change that pass
            :class:`SE2Metric` with a ``weight``, as in
            ``g.metric = SE2Metric(weight=0.5)``.

        The metric is used by :meth:`closest` and :meth:`distance`

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> g.metric = 'L1'
            >>> print(g.metric(np.r_[3, -4]))

        A pose metric whose orientation counts for less than its position

        .. runblock:: pycon

            >>> from pgraph import UGraph, SE2Metric
            >>> import numpy as np
            >>> g = UGraph(dim=3)
            >>> g.metric = SE2Metric(weight=0.5)
            >>> print(g.metric(np.r_[3, 4, np.pi / 2]))
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
            point passed to :meth:`closest`/:meth:`Vertex.distance` -- to be
            exactly 3 elements; raises :exc:`ValueError` otherwise.  Radians
            and metres are weighted equally; to change that pass
            :class:`SE2Metric` with a ``weight``, as in
            ``g.metric = SE2Metric(weight=0.5)``.

        The heuristic distance is only used by the A* planner :meth:`path_Astar`.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> g.heuristic = 'L2'
            >>> print(g.heuristic(np.r_[3, 4]))

        The heuristic of A* must not exceed the true cost to go.  For poses
        whose costs are the planar distance travelled, ignore the orientation
        with ``SE2Metric(weight=0)``.
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

    def __getitem__(self, i: int | str | Vertex) -> Vertex:
        """
        Get vertex (base class method)

        :param i: vertex description
        :type i: int, str or Vertex subclass
        :raises IndexError: ``i`` is an integer outside the range of vertices
        :raises KeyError: ``i`` is a name that is not in the graph
        :raises TypeError: ``i`` is none of the types above
        :return: the referenced vertex
        :rtype: Vertex subclass

        Retrieve a vertex by index or name:

        - ``g[i]`` is the i'th vertex in the graph.  This reflects the order of
          addition to the graph.  ``i`` can be any integer, including a NumPy
          integer such as the result of ``np.argmin()``, and a negative
          ``i`` counts from the last vertex.
        - ``g[s]`` is vertex named ``s``
        - ``g[v]`` is ``v`` where ``v`` is a ``Vertex`` subclass

        This method also supports iteration over the vertices in a graph::

            for v in g:
                print(v)

        will iterate over all the vertices.
        """
        if isinstance(i, (int, np.integer)):
            return self._vertexlist[int(i)]
        elif isinstance(i, str):
            return self._vertexdict[i]
        elif isinstance(i, Vertex):
            return i
        else:
            raise TypeError(
                f"vertex must be an int, str or Vertex, got {type(i).__name__}"
            )

    def __iter__(self) -> Iterator[Vertex]:
        """
        Iterate over the vertices of the graph

        :return: iterator over vertices, in order of addition
        :rtype: iterator of Vertex subclass

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

    def __contains__(self, item: Vertex | str) -> bool:
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

    def coordinates(self) -> NDArray:
        """
        Coordinates of all the vertices as a matrix

        :return: coordinates, one column per vertex in the order they were added
        :rtype: ndarray(dim, n)
        :raises ValueError: the coordinates of two vertices differ in length

        Column ``i`` is the coordinate of ``g[i]``, the same order as the
        rows and columns of :meth:`adjacency`.  A vertex that has no
        coordinate has a column of NaN, so the matrix always has one column
        per vertex.  The number of rows is ``dim`` if the graph was given one,
        otherwise the length of the first coordinate, and 0 if no vertex has
        one.  Elements are float.

        The result is a *view* of a new array that has one row per vertex, so
        the coordinate of each vertex is contiguous in memory and ``C[:, i]``
        is a contiguous slice, while a row such as ``C[0]`` is strided.  Use
        :func:`numpy.ascontiguousarray` if you need the rows to be contiguous.

        .. note:: This is a snapshot, made each time: changing it does not
            move the vertices, and moving a vertex does not change a matrix
            that was already returned.

        With NumPy this selects vertices in a single operation, as
        :meth:`within` and :meth:`near` do, or it can be given to a spatial
        index of your own.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> for i in range(3):
            ...     g.add_vertex(coord=[i, 10 * i])
            >>> g.coordinates()

        :seealso: :meth:`within` :meth:`near` :meth:`closest` :meth:`adjacency`
        """
        n = len(self._vertexlist)
        dim = self._dim
        if dim is None:
            dim = next((len(v.coord) for v in self._vertexlist if v.coord is not None), 0)
        out = np.full((n, dim), np.nan)
        for i, vertex in enumerate(self._vertexlist):
            if vertex.coord is None:
                continue
            if len(vertex.coord) != dim:
                raise ValueError(
                    f"vertex {vertex.name!r} has a coordinate of length "
                    f"{len(vertex.coord)}, but the others have {dim}"
                )
            out[i] = vertex.coord
        return out.T

    def build_kdtree(self) -> None:
        """
        Build a spatial index of the vertices, used by :meth:`closest` and :meth:`near`

        :raises ValueError: the metric is not one of the built-in ``"L1"``,
            ``"L2"`` or ``"SE2"``, or ``"SE2"`` and the coordinates are not
            3 elements, or no vertex has a coordinate
        :raises ImportError: scipy is not installed

        Without an index, finding the nearest vertex to a point measures the
        distance to every vertex, which is the cost of a query on a large graph
        that is queried often.  A k-d tree answers in a time that grows more
        slowly, at the cost of building it once, which pays for a graph that is
        built and then queried.  A graph that is queried while it is growing
        does better without it, since adding a vertex discards the index.
        :meth:`closest` and :meth:`near` use the index if it has been built,
        and measure every vertex if it has not or if the graph has changed.

        The index is for the built-in metrics ``"L1"``, ``"L2"`` and
        ``"SE2"`` (with any weight, see :class:`SE2Metric`), and results
        are the same as without it, except that when two vertices are exactly
        the same distance the one returned may differ.

        .. warning:: The index is dropped when a vertex is added or removed
            or when ``vertex.coord`` is assigned (including ``+=``), and
            :meth:`closest` is then correct but slow, until it is built again.
            It cannot see a coordinate that is changed in place,
            ``vertex.coord[0] = x``.  Do not change vertices in place after
            building, or build again.

        The index is not copied or pickled with the graph.

        .. note:: This needs `scipy <https://scipy.org>`_.  pgraph does not
            list it as a dependency and imports it only when this method is
            called, but `spatialmath-python`, which pgraph does depend on,
            installs it.

        :seealso: :meth:`closest` :meth:`near` :meth:`coordinates`
        """
        metric = self._metric
        weight: float | None = None
        if metric is _METRICS["L1"]:
            p = 1
        elif metric is _METRICS["L2"]:
            p = 2
        elif isinstance(metric, SE2Metric):
            p, weight = 2, metric.weight
        else:
            raise ValueError(
                "build_kdtree is for the built-in metrics 'L1', 'L2' and 'SE2', "
                f"not {metric!r}"
            )
        try:
            spatial = importlib.import_module("scipy.spatial")
        except ImportError as err:
            raise ImportError(
                "build_kdtree needs scipy, which pgraph does not itself require: "
                "pip install scipy"
            ) from err

        points = self.coordinates().T  # one row per vertex
        if points.shape[1] == 0:
            raise ValueError("no vertex has a coordinate")
        index = np.flatnonzero(np.isfinite(points).all(axis=1))
        if len(index) == 0:
            raise ValueError("no vertex has a coordinate")
        points = points[index]

        prepare: Any = None
        if weight is None:
            tree = spatial.cKDTree(points)
        else:
            if points.shape[1] != 3:
                raise ValueError(
                    f"the SE2 metric needs 3-element coordinates, not {points.shape[1]}"
                )
            # scipy has periodic dimensions for the angle, but a periodic box
            # must be finite and the data in it, so put the plane inside a box
            # much larger than the data and wrap the angle, scaled by its weight
            span = max(float(np.ptp(points[:, :2], axis=0).max()), 1.0)
            size = 1e3 * span
            shift = points[:, :2].min(axis=0) - size / 4
            period = 2 * np.pi * weight

            def prepare(q: NDArray) -> NDArray | None:  # noqa: F811
                xy = q[:2] - shift
                if not np.all((xy >= 0) & (xy < size)):
                    return None  # far outside the data: let the caller do it
                if weight == 0:
                    return xy
                return np.r_[xy, np.mod(weight * q[2], period)]

            if weight == 0:
                tree = spatial.cKDTree(points[:, :2] - shift, boxsize=[size, size])
            else:
                data = np.column_stack(
                    [points[:, :2] - shift, np.mod(weight * points[:, 2], period)]
                )
                tree = spatial.cKDTree(data, boxsize=[size, size, period])

        self._kdtree = _KDTree(tree, index, p, prepare)
        self._spatialchange = False

    def _index(self) -> _KDTree | None:
        """The spatial index, if it is built and still valid (private method)"""
        if self._kdtree is not None and self._spatialchange:
            self._kdtree = None  # vertices were added or removed: no longer right
        return self._kdtree

    def _index_point(self, index: _KDTree, coord: ArrayLike) -> NDArray | None:
        """The point to query the index with, or None if it cannot be (private)"""
        q = np.asarray(coord, dtype=float)
        if q.ndim != 1 or not np.all(np.isfinite(q)):
            return None
        if index.prepare is None:
            return q if q.shape[0] == index.tree.m else None
        if q.shape[0] != 3:
            return None
        return index.prepare(q)

    def _metric_columns(self, diff: NDArray) -> NDArray | None:
        """
        The metric of each column of a matrix, or None (private method)

        Only for the built-in metrics, which can be applied to all columns in
        one operation; the metric of a user is a function of one vector.
        """
        metric = self._metric
        if metric is _METRICS["L1"]:
            return np.asarray(np.abs(diff).sum(axis=0))
        if metric is _METRICS["L2"]:
            return np.asarray(np.sqrt((diff**2).sum(axis=0)))
        if isinstance(metric, SE2Metric) and diff.shape[0] == 3:
            theta = (diff[2] + np.pi) % (2 * np.pi) - np.pi
            return np.asarray(
                np.sqrt(diff[0] ** 2 + diff[1] ** 2 + (metric.weight * theta) ** 2)
            )
        return None

    def _points(self) -> _PointCache | None:
        """
        The coordinates of the embedded vertices as a matrix, or None (private)

        Built on first use and kept by :meth:`add_vertex`.  It is None, so
        that the caller makes its pass over the vertices, if no vertex has a
        coordinate or the coordinates are not all vectors of the same length,
        which that pass reports.  A vertex whose coordinate is not finite is
        left out, as it is never the closest.
        """
        if self._pts is not None:
            return self._pts
        vertices = [v for v in self._vertexlist if v.coord is not None]
        if not vertices:
            return None
        try:
            points = np.array([v.coord for v in vertices], dtype=float)
        except (ValueError, TypeError):
            return None  # not all the same length, or not numbers
        if points.ndim != 2 or points.shape[1] == 0:
            return None
        finite = np.isfinite(points).all(axis=1)
        if not finite.all():
            points = points[finite]
            vertices = [v for v, ok in zip(vertices, finite) if ok]
            if not vertices:
                return None
        self._pts = _PointCache(points, vertices)
        return self._pts

    def closest(self, coord: ArrayLike) -> tuple[Vertex | None, float]:
        """
        Vertex closest to point

        :param coord: coordinates of a point
        :type coord: ndarray(n)
        :return: closest vertex and its distance, or ``(None, inf)`` if no
            vertex in the graph has a coordinate
        :rtype: Vertex subclass or None, float

        Returns the vertex closest to the given point. Distance is computed
        according to the graph's metric. Vertices without a coordinate
        (``coord`` is None) are skipped -- they have no position to compare.

        For the built-in metrics the distance to every vertex is found in one
        operation on a matrix of the coordinates that the graph keeps, which
        adding a vertex only appends to, so this stays fast when vertices are
        added between queries, as a planner does.  A spatial index built with
        :meth:`build_kdtree`, if it is still valid, is faster again for a graph
        that is not changing.  A metric of your own is applied to each vertex
        in turn.

        .. note:: The matrix is a copy, so a coordinate changed in place,
            ``vertex.coord[0] = x``, is not seen.  Assigning
            ``vertex.coord = x`` (or ``+=``) is.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[10,10], name='v2')
            >>> vertex, d = g.closest([1, 1])
            >>> print(vertex, d)

        :seealso: :meth:`metric` :meth:`near` :meth:`build_kdtree`
        """
        index = self._index()
        if index is not None:
            q = self._index_point(index, coord)
            if q is not None:
                d, i = index.tree.query(q, p=index.p)
                if np.isfinite(d):
                    return self._vertexlist[int(index.index[i])], float(d)

        pts = self._points()
        if pts is not None:
            q = np.asarray(coord, dtype=float)
            if q.shape == (pts.dim,) and np.isfinite(q).all():
                d = self._metric_columns((pts.points - q).T)
                if d is not None:  # a built-in metric, applied to all at once
                    i = int(np.argmin(d))
                    return pts.vertices[i], float(d[i])

        min_dist = np.inf
        min_which: Vertex | None = None

        for vertex in self:
            if vertex.coord is None:
                continue
            d = self.metric(vertex.coord - coord)
            if d < min_dist:
                min_dist = d
                min_which = vertex

        return min_which, min_dist

    def within(self, box: ArrayLike) -> list[Vertex]:
        """
        Vertices inside a box

        :param box: lower and upper corners of the box
        :type box: ndarray(2,n)
        :raises ValueError: ``box`` is not 2 x n, its upper corner is below its
            lower corner in some dimension, or the coordinates of the vertices
            do not have n elements
        :return: the vertices inside the box, in the order of the graph
        :rtype: list of Vertex subclass

        The first row of ``box`` is the lower corner, the second row the upper
        corner, so in two dimensions it is ``[[xmin, ymin], [xmax, ymax]]``.
        The box is closed: a vertex on its boundary is inside.  Vertices
        without a coordinate are skipped, as they are by :meth:`closest`.

        The test is made for all the vertices at once on :meth:`coordinates`.
        Together with :meth:`subgraph` it selects the part of an embedded
        graph in a region

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(coord=[i, i], name=f"v{i}")
            >>> g.within([[1, 1], [3, 3]])
            >>> g.subgraph(g.within([[1, 1], [3, 3]]))

        :seealso: :meth:`near` :meth:`closest` :meth:`subgraph` :meth:`coordinates`
        """
        box = np.asarray(box, dtype=float)
        if box.ndim != 2 or box.shape[0] != 2:
            raise ValueError(f"box must be 2 x n, not {box.shape}")
        lower, upper = box
        if np.any(upper < lower):
            raise ValueError("the upper corner of the box is below the lower corner")

        C = self.coordinates()
        if C.shape[0] == 0 or C.shape[1] == 0:
            return []  # no vertex has a coordinate
        if C.shape[0] != lower.size:
            raise ValueError(
                f"the vertices have coordinates of {C.shape[0]} elements, "
                f"the box has {lower.size}"
            )
        inside = np.all((C >= lower[:, None]) & (C <= upper[:, None]), axis=0)
        return [self._vertexlist[i] for i in np.flatnonzero(inside)]

    def near(self, coord: ArrayLike, radius: float) -> list[tuple[Vertex, float]]:
        """
        Vertices within a distance of a point

        :param coord: coordinates of a point
        :type coord: ndarray(n)
        :param radius: greatest distance, inclusive
        :raises ValueError: ``radius`` is negative
        :return: the vertices within ``radius`` of the point with their
            distances, nearest first
        :rtype: list of (Vertex subclass, float)

        Distance is computed according to the graph's metric, as it is by
        :meth:`closest`, of which this is the generalization to every vertex
        within a radius.  Vertices without a coordinate are skipped, and
        vertices at equal distance are in the order of the graph.

        For the built-in metrics the distance to every vertex is found in one
        operation on the matrix of coordinates that :meth:`closest` uses, or
        by the spatial index if :meth:`build_kdtree` was called; a metric of
        your own is applied to each vertex in turn.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(coord=[i, 0], name=f"v{i}")
            >>> [(v.name, d) for v, d in g.near([2.2, 0], 1.5)]

        :seealso: :meth:`closest` :meth:`within` :meth:`metric` :meth:`build_kdtree`
        """
        if radius < 0:
            raise ValueError(f"radius must not be negative, got {radius!r}")

        candidates: list[Vertex] | None = None
        index = self._index()
        if index is not None:
            q = self._index_point(index, coord)
            if q is not None:
                # the index finds the candidates, the metric gives the distance
                hits = index.tree.query_ball_point(q, radius * (1 + 1e-9) + 1e-12, p=index.p)
                candidates = [self._vertexlist[int(index.index[i])] for i in sorted(hits)]

        if candidates is None:
            pts = self._points()
            point = np.asarray(coord, dtype=float)
            if pts is not None and point.shape == (pts.dim,):
                d = self._metric_columns((pts.points - point).T)
                if d is not None:
                    hit = np.flatnonzero(d <= radius)
                    hit = hit[np.argsort(d[hit], kind="stable")]
                    return [(pts.vertices[i], float(d[i])) for i in hit]
            candidates = [v for v in self._vertexlist if v.coord is not None]

        found = []
        for vertex in candidates:
            assert vertex.coord is not None  # the candidates all have a position
            dist = float(self.metric(vertex.coord - coord))
            if dist <= radius:
                found.append((vertex, dist))
        found.sort(key=lambda pair: pair[1])
        return found

    def edges(self) -> list[Edge]:
        """
        All edges in the graph, in the order they were added

        :return: All edges in the graph
        :rtype: list of Edge references

        The edges are in the order they were added to the graph.  This is a
        new list, so it can be modified or sorted without affecting the graph,
        and edges can be removed from the graph while iterating over it.

        We can iterate over all edges in the graph by::

            for e in g.edges():
                print(e)

        and select one by index, for example the second edge added is
        ``g.edges()[1]``.

        .. note:: Like :meth:`Vertex.edges` this is a ``list``, but where
            that lists the edges attached to one vertex, this lists every edge
            in the graph once.  A self-loop is one edge here, but appears
            twice in the :meth:`Vertex.edges` of an undirected vertex
            because it has two ends there.

        .. versionchanged:: 1.0
            Returns a ``list`` in order of addition.  In 0.6.5 this was the
            graph's own ``set``, with no defined order, although the
            documentation said ``list``.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> g.add_edge(v1, v2)
            >>> print(g.edges())

        :seealso: :meth:`Vertex.edges` :attr:`ne`
        """
        return list(self._edges)

    def _edge_layout(self) -> tuple[dict[Edge, float], dict[Edge, int]]:
        """
        How each edge is to be drawn, apart from its end points (private method)

        :return: the curvature of every edge that joins two vertices, and the
            number of every loop among the loops at its vertex

        Edges that join the same pair of vertices, whichever way they point,
        are drawn as arcs that bow to either side, so that parallel and
        antiparallel edges can all be seen.  A single edge is straight, with
        curvature 0.  Loops are numbered from 0 at each vertex.
        """
        index = {v: i for i, v in enumerate(self)}
        groups: dict[frozenset[int], list[Edge]] = {}
        loops: dict[Vertex, list[Edge]] = {}
        for e in self._edges:
            assert e.v1 is not None and e.v2 is not None
            if e.v1 is e.v2:
                loops.setdefault(e.v1, []).append(e)
            else:
                groups.setdefault(frozenset((index[e.v1], index[e.v2])), []).append(e)
        rads: dict[Edge, float] = {}
        for edges in groups.values():
            n = len(edges)
            for k, e in enumerate(edges):
                assert e.v1 is not None and e.v2 is not None
                offset = (k - (n - 1) / 2) * _ARC_STEP
                # the same side in space for an edge drawn the other way round
                rads[e] = offset if index[e.v1] < index[e.v2] else -offset
        loopno = {e: k for edges in loops.values() for k, e in enumerate(edges)}
        return rads, loopno

    def _edge_artist(
        self,
        ax: Any,
        edge: Edge,
        rad: float,
        loopno: int,
        markersize: float,
        ndim: int,
        **kwargs: Any,
    ) -> Any:
        """
        The matplotlib artist for one edge (private method)

        :param rad: curvature of an edge that is not a loop, see
            :meth:`_edge_layout`
        :param loopno: number of this edge among the loops at its vertex
        :param markersize: size in points of a vertex marker, so that the edge
            stops at the rim of the marker, not at the center of the vertex
        :param ndim: 2 for a 2D axes, 3 for a 3D axes
        :param kwargs: options of the patch: color, linewidth, arrowstyle, ...
        :return: the artist, to be added to ``ax``
        """
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.transforms import Affine2D, ScaledTranslation
        from ._plot import _Arrow3D, _Loop3D, _loop_path
        v1, v2 = edge.v1, edge.v2
        assert v1 is not None and v2 is not None
        shrink = markersize / 2 + 1
        if v1 is v2:
            radius = _LOOP_RADIUS + _LOOP_GROWTH * loopno
            path = _loop_path(_LOOP_ANGLE + _LOOP_TURN * loopno, radius, markersize)
            # a loop is a size in points, so it can reach beyond the limits of
            # the axes, which are in data units: do not clip it there
            options = {**dict(shrinkA=0, shrinkB=0, clip_on=False), **kwargs}
            if ndim == 3:
                return _Loop3D((v1.x, v1.y, v1.z), path, **options)
            transform = (
                Affine2D().scale(1 / 72)
                + ax.figure.dpi_scale_trans
                + ScaledTranslation(v1.x, v1.y, ax.transData)
            )
            return FancyArrowPatch(path=path, transform=transform, **options)

        options = {
            **dict(
                shrinkA=shrink, shrinkB=shrink, connectionstyle=f"arc3,rad={rad}"
            ),
            **kwargs,
        }
        if ndim == 3:
            return _Arrow3D((v1.x, v1.y, v1.z), (v2.x, v2.y, v2.z), **options)
        return FancyArrowPatch((v1.x, v1.y), (v2.x, v2.y), **options)

    def plot(
        self,
        colorcomponents: bool | None = None,
        force2d: bool | None = None,
        vopt: dict | None = None,
        eopt: dict | None = None,
        text: dict | bool | None = None,
        block: bool | None = False,
        grid: bool = True,
        ax: Any = None,
        *,
        vertex_opt: dict | None = None,
        edge_opt: dict | None = None,
        arrow_opt: dict | None = None,
        text_opt: dict | None = None,
        labels: bool | None = None,
        color_components: bool | None = None,
        force_2d: bool | None = None,
    ) -> None:
        """
        Plot graph as a vertex/edge diagram using Matplotlib

        :param vertex_opt: vertex format, a dict of Matplotlib ``plot``
            options, defaults to a 12 point circle marker
        :type vertex_opt: dict, optional
        :param edge_opt: edge format, a dict of Matplotlib patch options such as
            ``color``, ``linewidth`` and ``linestyle``, defaults to a line width
            of 3
        :type edge_opt: dict, optional
        :param arrow_opt: arrowhead and path format, a dict of Matplotlib
            ``FancyArrowPatch`` options such as ``arrowstyle``,
            ``mutation_scale`` and ``connectionstyle``, defaults to
            ``dict(arrowstyle="-|>", mutation_scale=15)`` for a directed graph
            and ``dict(arrowstyle="-")``, no head, for an undirected graph
        :type arrow_opt: dict, optional
        :param labels: label each vertex with its name, defaults to True
        :type labels: bool, optional
        :param text_opt: format of the labels, a dict of Matplotlib ``text``
            options
        :type text_opt: dict, optional
        :param color_components: color vertices and edges by component,
            defaults to True
        :type color_components: bool, optional
        :param force_2d: plot only the first two coordinates of each vertex
            even if they have more, defaults to False
        :type force_2d: bool, optional
        :param grid: show a grid, defaults to True
        :type grid: bool, optional
        :param ax: axes to draw in, defaults to the current axes or a new figure
        :type ax: Axes or Axes3D, optional
        :param block: block until figure is dismissed, defaults to False. If
            None the figure is not shown at all, which is useful when
            adding to the plot or saving it.
        :type block: bool or None, optional
        :raises ValueError: the graph is empty, or a vertex has no coordinate
            or fewer than two coordinate values
        :raises TypeError: an option is given by both its old and new name

        The graph is plotted using Matplotlib according to the dimensionality of the vertex
        coordinates (set by the `dim` parameter to graph constructor).  If the dimensionality is

        - 2, then a 2D plot is generated.
        - 3, then a 3D plot is generated.
        - 4 or more, then a 3D plot is generated using the first three coordinates.

        If a vertex has no coordinate or fewer than two values, a ValueError is raised.

        Every edge is drawn once.  In a directed graph it ends in an arrowhead
        at its end vertex.  Edges that join the same pair of vertices, whichever
        way they point, are drawn as arcs that bow to either side so that
        parallel and antiparallel edges can all be seen, and a loop is drawn as
        a circle at its vertex.

        If ``color_components`` is True then all edges and vertices within a component are assigned a unique
        color.

        .. note:: Only embedded graphs can be plotted: every vertex must have
            a coordinate of at least two values.  A coordinate of more than
            two values is drawn in 3D, using the first three.

        .. versionchanged:: 1.0
            A directed graph is drawn with arrowheads.  In 0.6.5 it was
            indistinguishable from an undirected one, parallel edges were
            drawn on top of each other and loops were not visible.  The
            options are named ``vertex_opt``, ``edge_opt``, ``arrow_opt`` and
            ``text_opt``.  ``vopt``, ``eopt``, ``text``, ``colorcomponents``
            and ``force2d`` are deprecated: they still work and give a
            :exc:`DeprecationWarning`.  ``text=False`` is ``labels=False`` and
            a ``text`` dict is ``text_opt``.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> g.plot(block=None)

        .. plot::

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> g.plot(block=None)

        A directed graph, with a pair of antiparallel edges, a pair of parallel
        edges and a loop

        .. plot::

            >>> from pgraph import DGraph
            >>> g = DGraph()
            >>> for name, xy in dict(A=(0, 1), B=(2, 1.7), C=(3.3, 0.3), D=(1.2, -0.5)).items():
            ...     g.add_vertex(xy, name=name);
            >>> for a, b in ["AB", "AB", "AD", "BC", "CB", "DC", "CC"]:
            ...     g.add_edge(a, b);
            >>> g.plot(block=None)

        :seealso: :meth:`highlight_path` :meth:`highlight_edge`
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        from spatialmath.base.graphics import axes_logic
        # the options that have been renamed
        vertex_opt = _renamed("plot", "vopt", vopt, "vertex_opt", vertex_opt)
        edge_opt = _renamed("plot", "eopt", eopt, "edge_opt", edge_opt)
        color_components = _renamed(
            "plot", "colorcomponents", colorcomponents, "color_components", color_components
        )
        force_2d = _renamed("plot", "force2d", force2d, "force_2d", force_2d)
        if text is not None:
            if labels is not None or text_opt is not None:
                raise TypeError(
                    "plot() got both text and labels or text_opt; "
                    "text is deprecated, use labels and text_opt only"
                )
            warnings.warn(
                "plot(): text is deprecated, use labels=False or text_opt=dict(...)",
                DeprecationWarning,
                stacklevel=2,
            )
            if text is False:
                labels = False
            elif isinstance(text, dict):
                text_opt = text
        if color_components is None:
            color_components = True
        if force_2d is None:
            force_2d = False
        if labels is None:
            labels = True

        if self.n == 0:
            raise ValueError("cannot plot an empty graph")
        ndim = 0  # number of coordinate values, taken from the first vertex
        for i, vertex in enumerate(self):
            if vertex.coord is None or len(vertex.coord) < 2:
                raise ValueError(
                    f"cannot plot: vertex {vertex.name} needs a coordinate "
                    "of at least two values"
                )
            if i == 0:
                ndim = len(vertex.coord)

        # without component colors every vertex and edge is the first color of
        # the cycle, rather than each in turn
        same = {} if color_components else dict(color="C0")
        vertex_opt = {**dict(marker="o", markersize=12), **same, **(vertex_opt or {})}
        arrow_opt = {
            **dict(
                arrowstyle="-|>" if self.directed else "-", mutation_scale=15
            ),
            **(arrow_opt or {}),
        }
        # the edge options are Line2D options in 0.6.5: keep those that a patch
        # also has, and say so about the rest
        patch_opt: dict[str, Any] = {}
        for key, value in {**dict(linewidth=3), **same, **(edge_opt or {})}.items():
            if hasattr(FancyArrowPatch, "set_" + key):
                patch_opt[key] = value
            else:
                warnings.warn(
                    f"plot(): edge_opt {key!r} does not apply to an edge and is ignored",
                    UserWarning,
                    stacklevel=2,
                )
        edge_kwargs = {**patch_opt, **arrow_opt}

        if color_components:
            color = plt.cm.coolwarm(np.linspace(0, 1, self.nc))

        draw3d = not (ndim == 2 or force_2d)
        if draw3d:
            ax = axes_logic(ax, 3)
        else:
            ax = axes_logic(ax, 2)

        # the vertices, and their labels
        for vertex in self:
            coords = (vertex.x, vertex.y, vertex.z) if draw3d else (vertex.x, vertex.y)
            opt = dict(vertex_opt)
            if color_components:
                opt = {**dict(color=color[vertex.label, :]), **opt}
            ax.plot(*coords, **opt)
            if labels:
                ax.text(*coords, f"  {vertex.name}", **(text_opt or {}))

        # the edges, each once
        rads, loopno = self._edge_layout()
        for e in self._edges:
            assert e.v1 is not None
            kwargs = dict(edge_kwargs)
            if color_components:
                kwargs = {**dict(color=color[e.v1.label, :]), **kwargs}
            artist = self._edge_artist(
                ax,
                e,
                rads.get(e, 0.0),
                loopno.get(e, 0),
                vertex_opt["markersize"],
                3 if draw3d else 2,
                **kwargs,
            )
            if draw3d:
                ax.add_artist(artist)
            else:
                ax.add_patch(artist)

        if loopno and not draw3d:
            # a loop is a size in points from its vertex, which matplotlib does
            # not count in the limits of the axes, so make room for the largest
            ax.autoscale_view()
            reach = 2 * (_LOOP_RADIUS + _LOOP_GROWTH * max(loopno.values()))
            pixels = (reach + vertex_opt["markersize"]) * ax.figure.dpi / 72
            (x0, y0), (x1, y1) = ax.transData.inverted().transform(
                [(0, 0), (pixels, pixels)]
            )
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            for e in loopno:
                assert e.v1 is not None
                ax.update_datalim(
                    [(e.v1.x - dx, e.v1.y - dy), (e.v1.x + dx, e.v1.y + dy)]
                )
            ax.autoscale_view()

        ax.grid(grid)
        if block is not None:
            plt.show(block=block)

    def highlight_path(
        self, path: list[Vertex], block: bool | None = False, **kwargs: Any
    ) -> None:
        """
        Highlight a path through the graph

        :param path: sequence of vertices forming a path
        :type path: list of Vertex subclass
        :param block: block until figure is dismissed, defaults to False. If
            None the figure is not shown.
        :type block: bool or None, optional
        :param kwargs: arguments passed to :meth:`highlight_edge` and
            :meth:`highlight_vertex`

        The vertices and edges along the path are overwritten with a different
        size/width and color.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> g.plot(block=None)
            >>> path, length, _ = g.path_Astar(g[2], g[4])
            >>> g.highlight_path(path, block=None)

        .. plot::

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> g.plot(block=None)
            >>> path, length, _ = g.path_Astar(g[2], g[4])
            >>> g.highlight_path(path, block=None)

        :seealso: :meth:`highlight_vertex` :meth:`highlight_edge`
        """
        import matplotlib.pyplot as plt
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
        Highlight an edge in a plot of the graph

        :param edge: The edge to highlight
        :type edge: Edge subclass
        :param scale: Overwrite with a line that is thicker than the original,
                      defaults to 1.5
        :type scale: float, optional
        :param color: Overwrite with a line in this color, defaults to 'r'
        :type color: str, optional
        :param alpha: Transparency of the highlight, defaults to 0.5
        :type alpha: float, optional
        :raises ValueError: ``edge`` does not connect two vertices, for
            example because it has been removed from the graph
        :rtype: None

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> np.random.seed(0)

            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> edges = []
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     edges.append(g.add_edge(g[i], g[j]));
            >>> g.plot(block=None)
            >>> g.highlight_edge(edges[2])

        .. plot::

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> edges = []
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     edges.append(g.add_edge(g[i], g[j]));
            >>> g.plot(block=None)
            >>> g.highlight_edge(edges[2])

        :seealso: :meth:`plot` :meth:`highlight_vertex` :meth:`highlight_path`
        """
        import matplotlib.pyplot as plt
        if edge.v1 is None or edge.v2 is None:
            raise ValueError("edge does not connect two vertices")
        # over the edge as plot() draws it, so that it follows an arc, and in
        # three dimensions if that is what the axes are
        ax = plt.gca()
        ndim = 3 if getattr(ax, "name", None) == "3d" else 2
        rads, loopno = self._edge_layout()
        ax.add_patch(
            self._edge_artist(
                ax,
                edge,
                rads.get(edge, 0.0),
                loopno.get(edge, 0),
                12,
                ndim,
                arrowstyle="-",
                color=color,
                linewidth=3 * scale,
                alpha=alpha,
            )
        )

    def highlight_vertex(
        self,
        vertex: Vertex | Iterable[Vertex | str],
        scale: float = 2,
        color: str = "r",
        alpha: float = 0.5,
    ) -> None:
        """
        Highlight a vertex in the graph

        :param vertex: The vertex (or vertices, or vertex names) to highlight
        :type vertex: Vertex subclass, or iterable of Vertex subclass or str
        :param scale: Overwrite with a marker that is larger than the original,
                      defaults to 1.5
        :type scale: float, optional
        :param color: Overwrite with a line in this color, defaults to 'r'
        :type color: str, optional
        :param alpha: Transparency of the highlight, defaults to 0.5
        :type alpha: float, optional
        :rtype: None

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> g.plot(block=None)
            >>> g.highlight_vertex(g[2])

        .. plot::

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> np.random.seed(0)
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> g.plot(block=None)
            >>> g.highlight_vertex(g[2])

        """
        import matplotlib.pyplot as plt

        ax = plt.gca()
        is3d = getattr(ax, "name", None) == "3d"
        vertices = vertex if isinstance(vertex, Iterable) else [vertex]
        for n in vertices:
            if isinstance(n, str):
                n = self[n]
            style: dict[str, Any] = dict(
                color=color, markersize=12 * scale, alpha=alpha
            )
            if is3d:
                ax.plot([n.x], [n.y], [n.z], "o", **style)
            else:
                ax.plot(n.x, n.y, "o", **style)

    def export(
        self,
        filename: str | os.PathLike | IO | None = None,
        format: str | None = None,  # pylint: disable=redefined-builtin
        **options: Any,
    ) -> str | None:
        """
        Convert and save the graph in a graph export or image format

        :param filename: destination path or open file, defaults to None
        :type filename: str, path-like or file object, optional
        :param format: ``"dot"``, ``"mermaid"``, ``"graphml"``, ``"elk"`` or an
            image format ``"svg"``, ``"pdf"``, ``"png"`` ..., inferred from
            the extension of ``filename`` if not given
        :type format: str, optional
        :param options: format specific options, see below
        :raises ValueError: the format is unknown or cannot be inferred, or an
            image format is requested without a ``filename``
        :raises TypeError: an option is not valid for the format
        :raises RuntimeError: an image format is requested and GraphViz is not
            installed, or fails
        :return: the text if ``filename`` is None, otherwise None
        :rtype: str or None

        .. list-table::
            :header-rows: 1

            * - ``format``
              - extensions
              - positions
              - options
            * - ``"dot"``
              - ``.dot`` ``.gv``
              - ``pos="x,y!"``
              - ``direction``, ``color_components``, ``costs``
            * - ``"mermaid"``
              - ``.mmd`` ``.mermaid``
              - no
              - ``direction``, ``color_components``, ``costs``, ``fence``
            * - ``"graphml"``
              - ``.graphml``
              - ``x``, ``y``, ``coord``
              - none
            * - ``"elk"``
              - ``.elk.json`` ``.elk``
              - ``x``, ``y``
              - ``size``, ``layout_options``
            * - ``"svg"`` ``"pdf"`` ``"png"`` ...
              - the same
              - yes, via ``neato``
              - ``engine``, ``direction``, ``color_components``, ``costs``

        The text formats need nothing beyond the standard library.  The image
        formats run the GraphViz ``dot`` or ``neato`` executable.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> g.add_edge(v1, v2, cost=2)
            >>> print(g.export(format="mermaid", direction="LR"))

        .. note::

            - Output is reproducible: vertices are written in creation order
              and edges in the order of their vertices, not in set order.
            - A file object passed as ``filename`` is written but not closed.
              A path is opened, and closed even if writing fails.
            - Names are escaped for each format, and Mermaid, GraphML and ELK
              use generated ids so any vertex name is valid.

        .. rubric:: For graphs with parallel edges or loops

        Every edge is written, each with its own cost, so parallel edges give
        repeated lines (DOT, Mermaid) or distinct ``id`` elements (GraphML,
        ELK).  A self-loop is written once, as an edge from a vertex to itself.

        :seealso: :meth:`dotfile` :meth:`showgraph` :mod:`pgraph.exporters`
        """
        from . import exporters

        if format is None:
            format = exporters.format_from_filename(filename)
            if format is None:
                raise ValueError(
                    "cannot infer the export format, give format= or a filename "
                    f"with a known extension; formats: {exporters.formats()}"
                )
        format = format.lower()

        if format in exporters.RENDER_FORMATS:
            if filename is None:
                raise ValueError(f"format {format!r} is an image and needs a filename")
            image = exporters.render(self, format, **options)
            if isinstance(filename, (str, os.PathLike)):
                with open(filename, "wb") as f:
                    f.write(image)
            else:
                try:
                    filename.write(image)
                except TypeError:  # a text mode file, fine for svg and ps
                    if not exporters.RENDER_FORMATS[format]:
                        raise
                    filename.write(image.decode("utf-8"))
            return None

        text = exporters.get_exporter(format)(self, **options)

        if filename is None:
            return text
        if isinstance(filename, (str, os.PathLike)):
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            filename.write(text)
        return None

    def dotfile(
        self,
        filename: str | os.PathLike | IO | None = None,
        direction: str | None = None,
    ) -> None:
        """
        Export graph as a GraphViz dot file

        :param filename: filename to save graph to, defaults to None
        :type filename: str, path-like or file object, optional
        :param direction: layout direction, eg. ``"LR"``, defaults to None
        :type direction: str, optional

        ``g.dotfile()`` creates the specified file which contains the `GraphViz
        <https://graphviz.org>`_ code to represent the embedded graph.  By default
        output is to the console.

        This is :meth:`export` with ``format="dot"`` and edge costs left
        unlabelled, as it has always been.  Use :meth:`export` for other
        formats or for edge labels.

        .. note::

            - The graph is undirected if it is a :class:`UGraph`
            - The graph is directed if it is a :class:`DGraph`

        .. note:: The ``graphviz`` formatters ``dot`` and ``neato`` can be used to render the
            graph in various formats including PDF, PNG, JPG, GIF, SVG, etc.  The formatters are
            similar, but ``dot`` is better suited for directed graphs while ``neato`` is better
            suited for undirected graphs.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> g.add_edge(v1, v2)
            >>> g.dotfile()

        .. note:: Edge costs are not included in the output.  Use :meth:`export` with
            ``costs=True`` to include them.

        .. note:: If ``filename`` is a file object then the file will *not*
            be closed after the GraphViz model is written. Safe for use with ``sys.stdout`` or ``sys.stderr``.

        :seealso: :meth:`export` :meth:`showgraph`
        """
        from . import exporters

        text = exporters.dot(self, direction=direction, costs=False)
        if filename is None:
            sys.stdout.write(text)
        elif isinstance(filename, (str, os.PathLike)):
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            filename.write(text)

    def showgraph(self, display: bool = True, **options: Any) -> str | None:
        """
        Display graph as a vertex/edge diagram in a browser tab

        :param display: open the rendered PDF in a browser tab, defaults to True
        :param options: ``engine``, ``direction``, ``color_components`` and
            ``costs``, passed to :meth:`export`
        :raises RuntimeError: GraphViz is not installed, or fails
        :return: path of the rendered PDF if ``display`` is False, else None
        :rtype: str or None

        ``g.showgraph()`` renders the graph to PDF with `GraphViz
        <https://graphviz.org>`_ and displays it in a browser tab.  With
        ``display=False`` the PDF is rendered but not opened and its path is
        returned, which is useful for tests and notebooks.

        .. note:: Vertices do not have to have coordinates for this to work, but if they do the coordinates are used to position the vertices in the output.  If no coordinates are given, GraphViz will choose a layout.

        :seealso: :meth:`export` :meth:`dotfile`
        """
        pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.close()
        self.export(pdf.name, format="pdf", **options)
        if display:
            webbrowser.open(f"file://{pdf.name}")
            return None
        return pdf.name

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
        vertices' outgoing edges via :meth:`Vertex.adjacent` the same
        way :meth:`path_BFS` and friends do.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(name='v1')
            >>> v2 = g.add_vertex(name='v2')
            >>> v3 = g.add_vertex(name='v3')
            >>> g.add_edge(v1, v2)
            >>> g.add_edge(v2, v3)
            >>> print(g.iscyclic())
            >>> g.add_edge(v3, v1)
            >>> print(g.iscyclic())

        .. note:: A self-loop is a cycle of length one, so it makes either kind
            of graph cyclic. Two parallel edges between the same pair of
            vertices are a cycle in an undirected graph, but a repeated
            arrow in a directed graph is not.

        .. note:: A matrix-based test also exists in theory -- a digraph is
            acyclic iff its adjacency matrix is nilpotent (all eigenvalues
            zero) -- but that needs an O(N^3) eigendecomposition plus a
            floating-point zero-tolerance judgement call, to answer a
            question Kahn's algorithm answers exactly with integers in
            O(V+E). Not used here for that reason.
            
        :seealso: :meth:`adjacency`
        """
        if not self.directed:
            return self.ne > self.n - self.nc

        # DGraph: Kahn's algorithm
        indegree = {vertex: vertex.in_degree for vertex in self}

        frontier = [vertex for vertex in self if indegree[vertex] == 0]
        removed = 0
        while frontier:
            vertex = frontier.pop()
            removed += 1
            for n in vertex.adjacent():
                indegree[n] -= 1
                if indegree[n] == 0:
                    frontier.append(n)

        return removed != self.n

    def average_degree(self) -> float:
        r"""
        Average degree of the graph

        :return: average degree
        :rtype: float

        Average degree is :math:`2 E / N` where :math:`E` is the total number
        of edges and :math:`N` is the number of vertices, for an undirected or a
        directed graph.  It is the mean of :attr:`Vertex.degree`, and every
        edge has two ends, so the degrees sum to :math:`2 E`.

        Parallel edges and self-loops are each counted as one edge in
        :math:`E`; a self-loop contributes 2 to the degree sum.

        .. versionchanged:: 1.0
            For a ``DGraph`` this was :math:`E / N`, the mean *out*-degree
            (which equals the mean in-degree).  It is now :math:`2 E / N`, the
            mean of the degree, which is twice the old value.  Use
            ``g.ne / g.n`` for the previous number.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(name='v1')
            >>> v2 = g.add_vertex(name='v2')
            >>> v3 = g.add_vertex(name='v3')
            >>> g.add_edge(v1, v2);
            >>> g.add_edge(v2, v3);
            >>> print(g.average_degree())

        :seealso: :meth:`degree` :meth:`connectivity` :attr:`Vertex.degree`
        """
        return 2 * self.ne / self.n

    # --------------------------------------------------------------------------- #

    # MATRIX REPRESENTATIONS

    def laplacian(self) -> NDArray:
        """
        Convert graph to a Laplacian matrix

        :return: Laplacian matrix
        :rtype: NumPy ndarray

        ``g.laplacian()`` is the Laplacian matrix (NxN) of the graph where N
        is the number of vertices.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> L = g.laplacian()
            >>> print(L)

        .. note::

            - Laplacian always has at least one zero eigenvalue (each row of
              ``D_out - adjacency()`` sums to zero, so the all-ones
              vector is always a right null vector).
            - For an **undirected** graph specifically: the Laplacian is
              symmetric and positive-semidefinite, and the number of
              zero-valued eigenvalues equals the number of connected
              components.
            - For a **directed** graph, this computes the out-degree
              Laplacian (used in directed
              consensus/synchronization literature) -- but it is generally
              *not* symmetric, its eigenvalues can be complex, and the
              zero-eigenvalue/component-count relationship above does not
              hold.

        .. rubric:: For graphs with parallel edges or loops

        Both the out-degree and :meth:`adjacency` count edges, so the
        identity ``L = D_out - adjacency()`` and the zero row sums hold
        unchanged.  A self-loop adds the same amount to both and cancels out
        of the Laplacian entirely.

        .. note:: ``D_out`` is the diagonal matrix of
            :attr:`Vertex.out_degree`.  For an undirected graph that is
            :meth:`degree`, but for a directed graph :meth:`degree` is in plus
            out, so the Laplacian does *not* use it.

        :seealso: :meth:`adjacency` :meth:`incidence` :meth:`degree`
        """
        return np.diag([v.out_degree for v in self]) - self.adjacency()

    def Laplacian(self) -> NDArray:
        """
        Laplacian matrix for the graph (deprecated)

        .. deprecated:: 1.0
            Use :meth:`laplacian` instead.  A capitalized name denotes a class
            method that constructs a graph, like :meth:`Adjacency` and
            :meth:`Dict`; this is an instance method that returns a matrix.

        :seealso: :meth:`laplacian`
        """
        warnings.warn(
            "Laplacian() is deprecated, use laplacian() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.laplacian()

    def connectivity(self, vertices: Iterable[Vertex] | None = None) -> list[int]:
        """
        Graph connectivity

        :param vertices: vertices to report connectivity for, defaults to all
            vertices in the graph
        :type vertices: iterable of Vertex subclass, optional
        :return: a list with the number of edges per vertex
        :rtype: list

        This is the :attr:`Vertex.degree` of each vertex: the number of
        edge *ends* at it, which is the number of edges connected to it except
        that a self-loop has two ends and so counts twice.  For a directed
        graph it counts the edges entering the vertex as well as those leaving.

        .. versionchanged:: 1.0
            For a ``DGraph`` this counted only the edges *leaving* each vertex.
            Use ``[v.out_degree for v in g]`` for the previous values.

        The average vertex connectivity is::

            mean(g.connectivity())

        and the minimum vertex connectivity is::

            min(g.connectivity())

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> c = g.connectivity()
            >>> print(c)


        .. note:: This is the diagonal of :meth:`degree`.  Parallel edges are
            counted individually.

        :seealso: :meth:`degree` :meth:`average_degree` :attr:`Vertex.degree`
        """

        c = []
        if vertices is None:
            vertices = self
        for n in vertices:
            c.append(n.degree)
        return c

    def degree(self) -> NDArray:
        """
        Convert graph to a degree matrix

        :return: degree matrix
        :rtype: ndarray(N,N)

        This is a diagonal matrix where element ``[i,i]`` is the
        :attr:`Vertex.degree` of vertex ``i``: the number of edge ends at
        it, so a self-loop counts twice.  For a directed graph that is the
        edges entering plus the edges leaving.

        .. versionchanged:: 1.0
            For a ``DGraph`` this was the out-degree matrix, counting only the
            edges leaving each vertex.  It is now the total, in plus out.  The
            old matrix is ``np.diag([v.out_degree for v in g])``, which is what
            :meth:`laplacian` uses.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> d = g.degree()
            >>> print(d)

        .. rubric:: For graphs with parallel edges or loops

        Every edge is counted, so ``k`` parallel edges between two vertices
        add ``k`` to the degree of each end.  A self-loop adds 2, for both
        kinds of graph, since it has two ends.

        :seealso: :meth:`adjacency` :meth:`incidence` :meth:`laplacian`
            :meth:`connectivity` :meth:`average_degree` :attr:`Vertex.degree`
        """

        return np.diag(self.connectivity())

    def adjacency(self) -> NDArray:
        """
        Convert graph to an adjacency matrix

        :returns: adjacency matrix
        :rtype: ndarray(N,N)

        The elements of the adjacency matrix ``[i,j]`` are 1 if vertex ``i`` is
        connected to vertex ``j``, else 0.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> A = g.adjacency()
            >>> print(A)

        .. note::

            - vertices are numbered in their order of creation. A vertex index
              can be resolved to a vertex reference by ``graph[i]``.
            - for an undirected graph the matrix is symmetric
            - Eigenvalues of ``A`` are real and are known as the spectrum of the graph.
            - The element ``A[i,j]`` is the number of walks of length one
              from vertex ``i`` to vertex ``j``, which is the number of
              edges between them (zero or one for a simple graph).
            - If ``Ak = A ** k`` the element ``Ak[i,j]`` is the number of
              walks of length ``k`` from vertex ``i`` to vertex ``j``.

        .. rubric:: For graphs with parallel edges or loops

        The matrix holds edge *counts*, so ``k`` parallel edges from ``i`` to
        ``j`` give ``A[i,j] = k`` and ``Ak = A ** k`` still counts walks
        correctly. A self-loop adds to the diagonal: 2 for a ``UGraph``
        (consistent with :meth:`degree`) and 1 for a ``DGraph``. In every case
        each row of ``A`` sums to the degree of that vertex.

        :seealso: :meth:`laplacian` :meth:`incidence` :meth:`degree`
        """
        # create a dict mapping vertex to an id
        vdict = {}
        for i, vert in enumerate(self):
            vdict[vert] = i

        A = np.zeros((self.n, self.n))
        for vertex in self:
            for n in vertex.adjacent():
                A[vdict[vertex], vdict[n]] += 1
        return A

    def incidence(self) -> NDArray:
        """
        Convert graph to an incidence matrix

        :returns: incidence matrix
        :rtype: ndarray(n,ne)

        The elements of the incidence matrix ``I[i,j]`` are 1 if vertex ``i`` is
        connected to edge ``j``, else 0.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> for i in range(5):
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> I = g.incidence()
            >>> print(I)

        .. note::
            - Vertices are numbered in their order of creation. A vertex index
              can be resolved to a vertex reference by ``graph[i]``.
            - Edges are numbered in the order they appear in :meth:`edges`, which
              is the order they were added.
            - Both endpoints of every edge are marked, regardless of
              direction -- for a ``DGraph`` this means a vertex that is only
              ever a target (never a source) still appears here.  Iterating
              each vertex's own :meth:`Vertex.edges`, which are its
              outgoing edges, would silently drop such vertices for a
              directed graph.

        .. rubric:: For graphs with parallel edges or loops

        Parallel edges each get their own column. For a ``UGraph`` the column
        of a self-loop holds a single 2 (it touches its vertex at both ends,
        so row sums still equal :meth:`degree`); for a ``DGraph`` it holds a
        single 1.

        :seealso: :meth:`laplacian` :meth:`adjacency` :meth:`degree`
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
            if e.v1 is e.v2 and not self.directed:
                I[vdict[e.v1], j] = 2  # self-loop, counted at both ends

        return I

    def distance(self) -> NDArray:
        """
        Convert graph to a distance matrix

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
            ...     g.add_vertex(np.random.rand(2));
            >>> for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
            ...     g.add_edge(g[i], g[j]);
            >>> d = g.distance()
            >>> print(d)

        .. rubric:: For graphs with parallel edges or loops

        If there are parallel edges the element is the cost of the *cheapest*
        of them. A self-loop gives a non-zero diagonal element equal to its
        cost.

        :seealso: :meth:`Vertex.distance`
        """
        # create a dict mapping vertex to an id
        vdict = {}
        for i, vert in enumerate(self):
            vdict[vert] = i

        A = np.full((self.n, self.n), np.inf)
        for v1 in self:
            for v2, edge in v1.incidences():
                i, j = vdict[v1], vdict[v2]
                A[i, j] = min(A[i, j], self._require_cost(edge))
        A[np.isinf(A)] = 0  # not connected
        return A

    # GRAPH COMPONENTS

    def component(self, c: int) -> list[Vertex]:
        """
        All vertices in specified graph component

        :param c: component index
        :return: vertices belonging to component ``c``
        :rtype: list of Vertex subclass

        ``graph.component(c)`` is a list of all vertices in graph component ``c``.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(name='v1')
            >>> v2 = g.add_vertex(name='v2')
            >>> v3 = g.add_vertex(name='v3')
            >>> g.add_edge(v1, v2);
            >>> print(g)
            >>> print(g.component(0))

        :seealso: :attr:`nc` :meth:`samecomponent`
        """
        self._graphcolor()  # ensure labels are up to date
        return [v for v in self if v.label == c]

    def samecomponent(self, v1: Vertex, v2: Vertex) -> bool:
        """
        Test if vertices belong to same graph component

        :param v1: vertex
        :type v1: Vertex subclass
        :param v2: vertex
        :type v2: Vertex subclass
        :return: true if vertices belong to same graph component
        :rtype: bool

        Test whether vertices belong to the same component.  A component is a
        set of vertices that are connected by a path of edges, ignoring their
        direction, so for a directed graph this does not mean that there is a
        path from one to the other: ``a -> c <- b`` is one component, but there
        is no path from ``a`` to ``b``.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(name='v1')
            >>> v2 = g.add_vertex(name='v2')
            >>> v3 = g.add_vertex(name='v3')
            >>> g.add_edge(v1, v2)
            >>> print(g.samecomponent(v1, v2))
            >>> print(g.samecomponent(v1, v3))

        :seealso: :meth:`component` :attr:`nc`
        """
        self._graphcolor()  # ensure labels are up to date

        return v1.label == v2.label

    # --------------------------------------------------------------------------- #

    def path_BFS(
        self, S: Vertex | str, G: Vertex | str, verbose: bool = False, summary: bool = False
    ) -> tuple[list[Vertex], float] | None:
        """
        Breadth-first search for path

        :param S: start vertex
        :type S: Vertex subclass
        :param G: goal vertex
        :type G: Vertex subclass
        :param verbose: print search progress, defaults to False
        :param summary: print a one-line search summary, defaults to False
        :raises TypeError: ``S`` or ``G`` is neither a vertex nor a string
        :raises KeyError: ``S`` or ``G`` is a name not in the graph
        :raises ValueError: an edge on the path has no cost
        :return: list of vertices from S to G inclusive and path length, or
            None if there is no path
        :rtype: (list of Vertex subclass, float) or None

        Compute a path from vertex ``S`` to vertex ``G`` using breadth-first search.  The
        path is guaranteed to be the shortest in terms of number of edges, but not
        necessarily the cheapest in terms of edge costs.  This search is not suited
        to graphs with edge weights, and is inefficient in that it explores all vertices.

        Returns a list of vertices that starts with vertex ``S`` and ends with
        vertex ``G`` if possible, otherwise returns None -- so check the
        result before unpacking it.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,0], name='v2')
            >>> v3 = g.add_vertex(coord=[2,0], name='v3')
            >>> g.add_edge(v1, v2)
            >>> g.add_edge(v2, v3)
            >>> path, length = g.path_BFS(v1, v3)
            >>> print(path)
            >>> print(length)

        .. note:: Self-loops are ignored. If there are parallel edges the
            cheapest is used for the path length.

        :seealso: :meth:`path_UCS` :meth:`path_Astar`
        """
        S = self._resolve_vertex(S, "start")
        G = self._resolve_vertex(G, "goal")

        # we use lists not sets since the order is instructive in verbose
        # mode, really need ordered sets...
        # the frontier is a queue; ``seen`` is everything that has been on it,
        # so that "frontier or explored" is a set lookup, not a list scan
        frontier: deque[Vertex] = deque([S])
        seen: set[Vertex] = {S}
        explored: list[Vertex] = []
        parent: dict[Vertex, Vertex] = {}
        done = False

        while frontier:
            if verbose:
                print()
                print("FRONTIER:", ", ".join([str(v.name) for v in frontier]))
                print("EXPLORED:", ", ".join([str(v.name) for v in explored]))

            x = frontier.popleft()
            if verbose:
                print("   expand", x.name)

            # expand the vertex
            for n in x.adjacent():
                if n is x:
                    continue  # self-loop, never part of a shortest path
                if n is G:
                    if verbose:
                        print("     goal", n.name, "reached")
                    parent[n] = x
                    done = True
                    break
                if n not in seen:
                    # add it to the frontier
                    seen.add(n)
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
            length += self._require_cost(p.edgeto(x))
            path.append(p)
            x = p
        path.reverse()

        if summary or verbose:
            print(
                f"{len(explored)} vertices explored, {len(frontier)} remaining on the frontier"
            )

        return path, length

    def path_UCS(
        self, S: Vertex | str, G: Vertex | str, verbose: bool = False, summary: bool = False
    ) -> tuple[list[Vertex], float, dict[str, str]] | None:
        """
        Uniform cost search for path

        :param S: start vertex
        :type S: Vertex subclass
        :param G: goal vertex
        :type G: Vertex subclass
        :param verbose: print search progress, defaults to False
        :param summary: print a one-line search summary, defaults to False
        :raises TypeError: ``S`` or ``G`` is neither a vertex nor a string
        :raises KeyError: ``S`` or ``G`` is a name not in the graph
        :raises ValueError: an edge examined during the search has no cost
        :return: list of vertices from S to G inclusive, path length and
            search tree, or None if there is no path
        :rtype: (list of Vertex subclass, float, dict) or None

        Compute the lowest-cost path from vertex ``S`` to vertex ``G`` using uniform cost search.  The
        path is guaranteed to be the shortest in terms of traversed edge cost, but is inefficient in that it explores all vertices.

        Returns a list of vertices that starts with vertex ``S`` and ends with
        vertex ``G`` if possible, otherwise returns None -- so check the
        result before unpacking it.

        The search tree is returned as dict that maps a vertex to its parent.

        The heuristic is the distance metric of the graph, which defaults to
        Euclidean distance.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,0], name='v2')
            >>> v3 = g.add_vertex(coord=[2,0], name='v3')
            >>> g.add_edge(v1, v2)
            >>> g.add_edge(v2, v3)
            >>> path, length, tree = g.path_UCS(v1, v3)
            >>> print(path)
            >>> print(length)

        .. note:: Self-loops are ignored. If there are parallel edges the
            cheapest is used for the path length.

        :seealso: :meth:`path_BFS` :meth:`path_Astar`
        """
        S = self._resolve_vertex(S, "start")
        G = self._resolve_vertex(G, "goal")

        # The frontier is a dict, used as an ordered set: its order is the order
        # vertices first joined it, which breaks ties between equal ``f``, and
        # its value is that position.  ``heap`` finds the minimum ``f``; a
        # vertex that is reparented is pushed again, and the older entry is
        # recognised as stale when it is popped.
        frontier: dict[Vertex, int] = {S: 0}
        heap: list[tuple[float, int, Vertex]] = [(0, 0, S)]
        explored: dict[Vertex, None] = {}
        parent: dict[Vertex, Vertex] = {}
        f: dict[Vertex, float] = {S: 0}  # evaluation function

        while frontier:
            if verbose:
                print()
                print(
                    "FRONTIER:", ", ".join([f"{v.name}({f[v]:.0f})" for v in frontier])
                )
                print("EXPLORED:", ", ".join([str(v.name) for v in explored]))

            while True:  # minimum f in frontier, skipping stale heap entries
                fx, _, x = heapq.heappop(heap)
                if x in frontier and f[x] == fx:
                    break
            del frontier[x]
            if verbose:
                print("   expand", x.name)
            if x is G:
                break
            # expand the vertex
            for n, e in x.incidences():
                if n is x:
                    continue  # self-loop, never part of a shortest path
                fnew = f[x] + self._require_cost(e)
                if n not in frontier and n not in explored:
                    # add it to the frontier
                    parent[n] = x
                    f[n] = fnew
                    frontier[n] = len(parent)  # one entry per vertex added so far
                    heapq.heappush(heap, (fnew, frontier[n], n))
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
                        heapq.heappush(heap, (fnew, frontier[n], n))

            explored[x] = None
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
            path.append(p)
            x = p
        path.reverse()

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
        self, S: Vertex | str, G: Vertex | str, verbose: bool = False, summary: bool = False
    ) -> tuple[list[Vertex], float, dict[str, str]] | None:
        """
        A* search for path

        :param S: start vertex
        :type S: Vertex subclass
        :param G: goal vertex
        :type G: Vertex subclass
        :param verbose: print search progress, defaults to False
        :param summary: print a one-line search summary, defaults to False
        :raises TypeError: ``S`` or ``G`` is neither a vertex nor a string
        :raises KeyError: ``S`` or ``G`` is a name not in the graph
        :raises ValueError: an edge examined during the search has no cost
        :return: list of vertices from S to G inclusive, path length and
            search tree, or None if there is no path
        :rtype: (list of Vertex subclass, float, dict) or None

        Compute the lowest-cost path from vertex ``S`` to vertex ``G`` using A*t search.  The
        path is guaranteed to be the shortest in terms of traversed edge cost, and the search is efficient in
        terms of the number of vertices explored.

        Returns a list of vertices that starts with vertex ``S`` and ends with
        vertex ``G`` if possible, otherwise returns None -- so check the
        result before unpacking it.

        The search tree is returned as dict that maps a vertex to its parent.

        The heuristic is the distance metric of the graph, which defaults to
        Euclidean distance.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,0], name='v2')
            >>> v3 = g.add_vertex(coord=[2,0], name='v3')
            >>> g.add_edge(v1, v2)
            >>> g.add_edge(v2, v3)
            >>> path, length, tree = g.path_Astar(v1, v3)
            >>> print(path)
            >>> print(length)

        .. note:: Self-loops are ignored. If there are parallel edges the
            cheapest is used for the path length.

        :seealso: :meth:`heuristic` :meth:`path_BFS` :meth:`path_UCS`
        """
        S = self._resolve_vertex(S, "start")
        G = self._resolve_vertex(G, "goal")

        # see path_UCS for the frontier and heap
        frontier: dict[Vertex, int] = {S: 0}
        heap: list[tuple[float, int, Vertex]] = [(0, 0, S)]
        explored: dict[Vertex, None] = {}
        parent: dict[Vertex, Vertex] = {}
        g: dict[Vertex, float] = {S: 0}  # cost to come
        f: dict[Vertex, float] = {S: 0}  # evaluation function

        while frontier:
            if verbose:
                print()
                print(
                    "FRONTIER:", ", ".join([f"{v.name}({f[v]:.0f})" for v in frontier])
                )
                print("EXPLORED:", ", ".join([str(v.name) for v in explored]))

            while True:  # minimum f in frontier, skipping stale heap entries
                fx, _, x = heapq.heappop(heap)
                if x in frontier and f[x] == fx:
                    break
            del frontier[x]
            if verbose:
                print("   expand", x.name)
            if x is G:
                break
            # expand the vertex
            for n, e in x.incidences():
                if n is x:
                    continue  # self-loop, never part of a shortest path
                if n not in frontier and n not in explored:
                    # add it to the frontier
                    parent[n] = x
                    frontier[n] = len(parent)  # one entry per vertex added so far
                    g[n] = g[x] + self._require_cost(e)  # update cost to come
                    f[n] = g[n] + n.heuristic_distance(G)  # heuristic
                    heapq.heappush(heap, (f[n], frontier[n], n))
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
                        heapq.heappush(heap, (f[n], frontier[n], n))

            explored[x] = None
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
            path.append(p)
            x = p
        path.reverse()

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


class UGraph(Graph):
    """
    Class for undirected graphs

    Vertices are :class:`Vertex` objects, or instances of the subclass given
    as ``vertex_class``. Edges are :class:`Edge` objects, or instances of the
    subclass given as ``edge_class``; see :class:`Graph` for the
    constructor arguments and the exceptions they raise.

    An edge is recorded at *both* of its vertices, so it is found from either
    end, and an edge supplied to :meth:`Vertex.connect` may have its endpoints
    in either order.

    .. inheritance-diagram:: UGraph

    :seealso: :class:`Graph` :class:`DGraph`
    """

    _directed = False

    def _attach(self, edge: Edge) -> None:
        """
        Record an undirected edge at both ends (private method)

        The edge is added to *both* vertices' edge lists, so it is discovered
        from either end.  A self-loop is therefore held twice by its vertex,
        which is why it counts 2 towards the degree.
        """
        assert edge.v1 is not None and edge.v2 is not None
        edge.v1._edgelist.append(edge)
        edge.v2._edgelist.append(edge)

    def _detach(self, edge: Edge) -> None:
        """
        Remove an undirected edge from both ends (private method)

        A self-loop was recorded twice by :meth:`_attach`, so it is removed
        twice.
        """
        for vertex in (edge.v1, edge.v2):
            if vertex is not None and edge in vertex._edgelist:
                vertex._edgelist.remove(edge)

    def _incoming(self, vertex: Vertex) -> list[Edge]:
        """The edges that end here: for an undirected graph, all of them"""
        return vertex._edgelist

    def _endpoints_match(self, edge: Edge, v1: Vertex, v2: Vertex) -> bool:
        """
        Test if an edge already connects two vertices (private method)

        An undirected edge has no direction, so either order is accepted.
        """
        return (edge.v1 is v1 and edge.v2 is v2) or (edge.v1 is v2 and edge.v2 is v1)


class DGraph(Graph):
    """
    Class for directed graphs

    Vertices are :class:`Vertex` objects, or instances of the subclass given
    as ``vertex_class``. Edges are :class:`Edge` objects, or instances of the
    subclass given as ``edge_class``; see :class:`Graph` for the
    constructor arguments and the exceptions they raise.

    An edge is recorded as leaving its start vertex and entering its end
    vertex, and an edge supplied to :meth:`Vertex.connect` must run exactly
    from that vertex to the other.

    .. inheritance-diagram:: DGraph

    :seealso: :class:`Graph` :class:`UGraph`
    """

    _directed = True

    def _attach(self, edge: Edge) -> None:
        """
        Record a directed edge at both ends (private method)

        The edge goes in the start vertex's list of outgoing edges and in the
        end vertex's list of incoming ones.
        """
        assert edge.v1 is not None and edge.v2 is not None
        edge.v1._edgelist.append(edge)
        edge.v2._inlist.append(edge)

    def _detach(self, edge: Edge) -> None:
        """
        Remove a directed edge from its start and from its end (private method)
        """
        if edge.v1 is not None and edge in edge.v1._edgelist:
            edge.v1._edgelist.remove(edge)
        if edge.v2 is not None and edge in edge.v2._inlist:
            edge.v2._inlist.remove(edge)

    def _incoming(self, vertex: Vertex) -> list[Edge]:
        """The edges that end at the vertex"""
        return vertex._inlist


# ========================================================================== #


_ENDPOINT_MESSAGE = (
    "Edge.{0} is read-only. Give the vertices to the constructor, Edge(v1, v2), "
    "or connect the edge with v1.connect(v2, edge=e) or e.connect(v1, v2)"
)


class Edge:
    """
    Edge class

    Is used to represent directed and undirected edges.

    Each edge has a cost, arbitrary data, and two vertices :attr:`v1` and
    :attr:`v2`, which are read-only: the start and end vertex for a directed
    edge.

    .. attribute:: cost
        :type: float | None

        cost of traversing this edge, required for planning methods

    .. attribute:: data
        :type: Any

        reference to arbitrary data associated with the edge

    .. note::

        - An undirected graph is created by having a single edge object in the
          edgelist of _each_ vertex.
        - This class can be inherited to provide user objects with graph capability.
        - Inheritance is an alternative to providing arbitrary user data.
        - Pass the subclass as ``edge_class`` when constructing the graph and
          the graph will create instances of it itself, see
          :class:`Graph`.

    An Edge points to a pair of vertices.  At ``connect`` time the vertices
    get references back to the Edge object.

    ``graph.add_edge(v1, v2)`` calls ``v1.connect(v2)``

    :seealso: :class:`Vertex`
    """

    def __init__(
        self,
        v1: Vertex | None = None,
        v2: Vertex | None = None,
        cost: float | None = None,
        data: Any = None,
    ):
        """
        Create an edge object

        :param v1: start of the edge, defaults to None
        :type v1: :class:`Vertex`, optional
        :param v2: end of the edge, defaults to None
        :type v2: :class:`Vertex`, optional
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

        :seealso: :meth:`Edge.connect` :meth:`Vertex.connect`
        """
        self._v1 = v1
        self._v2 = v2

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

    @property
    def v1(self) -> Vertex | None:
        """
        First vertex of the edge, the start vertex for a directed edge

        :return: the vertex, or None if the edge is not connected
        :rtype: Vertex subclass or None

        .. versionchanged:: 1.0
            Read-only.  It was a plain attribute, so assigning to it on an edge
            that was in a graph left the graph inconsistent: the vertices'
            edge lists no longer agreed with the edge, and the edge could not
            be removed.  Give the vertices to the constructor, ``Edge(v1,
            v2)``, or to :meth:`connect` or :meth:`Vertex.connect`.

        :seealso: :attr:`v2` :meth:`connect`
        """
        return self._v1

    @v1.setter
    def v1(self, value: Vertex | None) -> None:
        raise AttributeError(_ENDPOINT_MESSAGE.format("v1"))

    @property
    def v2(self) -> Vertex | None:
        """
        Second vertex of the edge, the end vertex for a directed edge

        :return: the vertex, or None if the edge is not connected
        :rtype: Vertex subclass or None

        Read-only, see :attr:`v1`.

        :seealso: :attr:`v1` :meth:`connect`
        """
        return self._v2

    @v2.setter
    def v2(self, value: Vertex | None) -> None:
        raise AttributeError(_ENDPOINT_MESSAGE.format("v2"))

    def _set_endpoints(self, v1: Vertex | None, v2: Vertex | None) -> None:
        """
        Bind the edge to its vertices, or unbind it (private method)

        Only the library calls this: the constructor's own assignment, then
        :meth:`connect`, :meth:`Vertex.connect` for an edge built without
        vertices, and :meth:`Graph.remove_edge`, which clears them.
        """
        self._v1 = v1
        self._v2 = v2

    def __reduce__(self) -> Any:
        """
        Refuse to pickle or copy an edge on its own

        An edge is held by its vertices and its graph, so it has no meaning
        apart from them.  Pickle or copy the graph instead.
        """
        raise TypeError(
            "an Edge cannot be pickled or copied on its own, "
            "pickle or copy its graph instead"
        )

    def __repr__(self) -> str:
        """
        Detailed representation of the edge

        :return: same as :meth:`__str__`
        :rtype: str

        .. runblock:: pycon

            >>> from pgraph import Vertex, Edge
            >>> v1 = Vertex(coord=[1,2], name="A")
            >>> v2 = Vertex(coord=[3,4], name="B")
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

            >>> from pgraph import Vertex, Edge
            >>> v1 = Vertex(coord=[1,2], name="A")
            >>> v2 = Vertex(coord=[3,4], name="B")
            >>> e = Edge(v1, v2, cost=5.0, data="A to B")
            >>> str(e)

        :seealso: :meth:`__repr__`
        """
        graph = self.v1._graph if self.v1 is not None else None
        arrow = "->" if graph is not None and graph.directed else "--"
        cost_str = "None" if self.cost is None else f"{self.cost:.4g}"
        s = f"{self.__class__.__name__}{{{self.v1} {arrow} {self.v2}, cost={cost_str}}}"
        if self.data is not None:
            s += f" data={self.data}"
        return s

    def connect(self, v1: Vertex, v2: Vertex) -> None:
        """
        Attach this edge to a pair of vertices

        :param v1: start of the edge
        :type v1: Vertex subclass
        :param v2: end of the edge
        :type v2: Vertex subclass
        :raises ValueError: either vertex has not been added to a graph, or
            the vertices belong to different graphs
        :raises TypeError: this edge is not an instance of the graph's
            :attr:`Graph.edge_class`

        The edge connects vertices ``v1`` and ``v2``, and is added to the
        graph that those vertices belong to.

        .. warning:: Intended for an edge that is not yet part of a graph. The
            edge's ``v1`` and ``v2`` are overwritten without checking, so
            connecting an edge that is already in a graph leaves it
            referenced from its old vertices as well.

        .. runblock:: pycon

            >>> from pgraph import Vertex, Edge, UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> e = Edge(cost=1.414)
            >>> e.connect(v1, v2)
            >>> print(e)

        .. note:: The vertices must already be added to the graph.

        :seealso: :meth:`Vertex.connect`
        """

        if v1._graph is None:
            raise ValueError("vertex v1 does not belong to a graph")
        if v2._graph is None:
            raise ValueError("vertex v2 does not belong to a graph")
        if not v1._graph is v2._graph:
            raise ValueError("vertices must belong to the same graph")

        # connect edge to its vertices
        self._set_endpoints(v1, v2)

        # tell the vertices to add edge to their edgelists as appropriate for
        # DGraph or UGraph
        v1.connect(v2, edge=self)

    def next(self, vertex: Vertex) -> Vertex:
        """
        Return other end of an edge

        :param vertex: one vertex on the edge
        :type vertex: Vertex subclass
        :raises ValueError: ``vertex`` is not on the edge
        :return: the other vertex on the edge
        :rtype: Vertex subclass

        ``e.next(v1)`` is the vertex at the other end of edge ``e``, ie. the
        vertex that is not ``v1``.

        .. runblock:: pycon

            >>> from pgraph import Vertex, Edge
            >>> v1 = Vertex(coord=[1,2], name="A")
            >>> v2 = Vertex(coord=[3,4], name="B")
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
            raise ValueError("should not happen")

    def vertices(self) -> list[Vertex]:
        """
        Vertices of an edge (deprecated)

        :return: the two vertices of this edge
        :rtype: list of Vertex subclass

        .. deprecated:: use :attr:`endpoints` instead
        """
        warnings.warn(
            "vertices() is deprecated, use endpoints instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.endpoints

    @property
    def endpoints(self) -> list[Vertex]:
        """
        The two vertices of this edge

        :return: start and end vertex
        :rtype: list of Vertex subclass

        .. runblock:: pycon

            >>> from pgraph import Vertex, Edge
            >>> v1 = Vertex(coord=[1,2], name="A")
            >>> v2 = Vertex(coord=[3,4], name="B")
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
        :meth:`Graph.remove_edge`), and clears ``e.v1``/``e.v2`` to
        ``None``. The ``Edge`` object itself is not deleted -- it becomes an
        orphaned, disconnected shell, and calling ``remove()`` on it again
        raises ``ValueError``.

        .. note:: This is a thin convenience wrapper around
            :meth:`Graph.remove_edge`.

        :rtype: None

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> e = g.add_edge(v1, v2)
            >>> e.remove()
            >>> print(g)

        :seealso: :meth:`Graph.remove_edge` :meth:`Vertex.remove`
        """
        if self.v1 is None or self.v1._graph is None:
            raise ValueError("edge is not connected to a graph")
        self.v1._graph.remove_edge(self)


# ========================================================================== #


class Vertex:
    """
    A vertex of a graph, directed or undirected.

    A vertex has no direction of its own: how an edge is recorded at its ends
    is decided by the graph it belongs to (see :class:`UGraph` and
    :class:`DGraph`).  So there is one vertex class, which can be subclassed
    to give user objects graph capability and passed to the graph as
    ``vertex_class``.

    .. versionchanged:: 1.0
        This replaces ``UVertex`` and ``DVertex``, which are deprecated names
        for it.  They differed only in how an edge was recorded, which is now
        the graph's decision.

    Each vertex has:
        - ``name`` which is its key in the graph, so renaming a vertex that is
          in a graph updates the graph
        - ``label`` an int indicating which graph component contains it,
          read-only: the graph assigns it
        - ``coord`` the coordinate in an embedded graph (optional)

    and its edges are available from :meth:`edges`, :meth:`out_edges` and
    :meth:`in_edges`.

    :seealso: :class:`Graph` :class:`Edge`
    """

    coord: NDArray | None
    _name: str | None
    _label: int | None
    _connectivitychange: bool
    _edgelist: list[Edge]
    _inlist: list[Edge]
    _graph: Graph | None

    def __init__(self, coord: ArrayLike | None = None, name: str | None = None):
        """
        Create a vertex object

        :param coord: coordinate of the vertex for an embedded graph, defaults to None
        :type coord: array-like, optional
        :param name: vertex name, defaults to None
        :type name: str, optional

        Creates a vertex but does not add it to a graph -- use
        :meth:`Graph.add_vertex` for that.

        .. runblock:: pycon

            >>> from pgraph import Vertex
            >>> v1 = Vertex(coord=[0,0], name='v1')
            >>> print(v1)

        :seealso: :meth:`Graph.add_vertex`
        """
        # A vertex that is being made is in no graph, so there is no spatial
        # index (Graph.build_kdtree) for the assignment of its coordinate to
        # invalidate; writing the attributes directly avoids the cost of
        # ``__setattr__`` on each of them.
        self.__dict__.update(
            # np.array is a tenth of the cost of np.r_ for the same result here
            coord=None if coord is None else np.array(coord, ndmin=1),
            _name=name,
            _label=None,
            _connectivitychange=True,
            _edgelist=[],  # edges leaving the vertex, or incident on it if undirected
            # edges ending at this vertex.  Only a directed vertex fills it: an
            # undirected edge is already in ``_edgelist`` at both of its ends.
            _inlist=[],
            _graph=None,  # reference to owning graph
        )

    def __setattr__(self, name: str, value: Any) -> None:
        # a vertex that moves makes its graph's spatial index (build_kdtree)
        # stale; this sees ``v.coord = x`` and ``v.coord += x`` but not
        # ``v.coord[0] = x``, which is why that is documented
        if name == "coord":
            graph = self.__dict__.get("_graph")
            if graph is not None:
                graph._spatialchange = True
                graph._pts = None
        object.__setattr__(self, name, value)

    def __reduce__(self) -> Any:
        """
        Refuse to pickle or copy a vertex on its own

        A vertex is held by its graph and by its edges, so it has no meaning
        apart from them.  Pickle or copy the graph instead.
        """
        raise TypeError(
            "a vertex cannot be pickled or copied on its own, "
            "pickle or copy its graph instead"
        )

    def __str__(self) -> str:
        """
        Compact representation of the vertex

        :return: the vertex name in square brackets
        :rtype: str

        .. runblock:: pycon

            >>> from pgraph import Vertex
            >>> v = Vertex(coord=[1,2], name="A")
            >>> str(v)
        """
        return f"[{self.name:s}]"

    def __repr__(self) -> str:
        """
        Detailed representation of the vertex

        :return: class name, vertex name and coordinate
        :rtype: str

        .. runblock:: pycon

            >>> from pgraph import Vertex
            >>> v = Vertex(coord=[1,2], name="A")
            >>> repr(v)
        """
        if self.coord is None:
            coord = "?"
        else:
            coord = ", ".join([f"{x:.4g}" for x in self.coord])
        return f"{self.__class__.__name__}[{self.name:s}, coord=({coord})]"

    def copy(self, cls: type[Graph] | None = None) -> Vertex:
        """
        Copy a vertex

        :param cls: graph class whose ``vertex_copy`` method should be used to
            create the copy, defaults to None
        :type cls: UGraph or DGraph subclass, optional
        :return: a new, unconnected vertex with the same coordinate and name
        :rtype: Vertex subclass

        If ``cls`` is given, ``cls.vertex_copy(self)`` is used to create a
        vertex of the appropriate subclass for that graph type, otherwise a
        vertex of the same class as ``self`` is created directly.

        .. runblock:: pycon

            >>> from pgraph import Vertex, DGraph
            >>> v = Vertex(coord=[1,2], name='v1')
            >>> v2 = v.copy(cls=DGraph)
            >>> print(v2)

        :seealso: :meth:`Graph.vertex_copy`
        """
        if cls is not None:
            return cls.vertex_copy(self)
        else:
            return self.__class__(coord=self.coord, name=self.name)

    def adjacent(self) -> list[Vertex]:
        """
        Vertices adjacent to this vertex

        :return: the vertices at the far end of this vertex's edges
        :rtype: list of Vertex subclass

        ``v.adjacent()`` is a list of the vertices that this vertex is joined
        to by an edge.  "Adjacent" is the graph-theory term, and it is the
        relation that :meth:`Graph.adjacency` tabulates: row ``i`` of the
        adjacency matrix is non-zero exactly at the columns of the vertices
        returned by ``g[i].adjacent()``.

        .. note::

            - For a directed graph these are the vertices at the head of the
              edges leaving this vertex, the *successors*.
            - There is one entry per edge, so a vertex joined by ``k`` parallel
              edges appears ``k`` times.  In an undirected graph a self-loop
              lists the vertex itself twice, once for each end.
            - ``neighbours`` and ``neighbors`` are aliases, the same method
              under the British and American spellings, and are not deprecated.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v3 = g.add_vertex(coord=[2,2], name='v3')
            >>> g.add_edge(v1, v2)
            >>> g.add_edge(v1, v3)
            >>> print(v1.adjacent())

        :seealso: :meth:`isadjacent` :meth:`successors` :meth:`predecessors`
            :meth:`incidences` :meth:`out_edges` :meth:`edges`
            :meth:`Graph.adjacency`
        """
        return [e.next(self) for e in self._edgelist]

    #: alias of :meth:`adjacent`
    neighbours = adjacent
    #: alias of :meth:`adjacent`
    neighbors = adjacent

    def isadjacent(self, vertex: Vertex) -> bool:
        """
        Test if a vertex is adjacent to this vertex

        :param vertex: vertex reference
        :type vertex: Vertex subclass
        :return: true if ``vertex`` is adjacent
        :rtype: bool

        For a directed graph this is true only if there is an edge from
        ``self`` to ``vertex``.

        ``isneighbour`` and ``isneighbor`` are aliases.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v3 = g.add_vertex(coord=[2,2], name='v3')
            >>> g.add_edge(v1, v2)
            >>> print(v1.isadjacent(v2))
            >>> print(v1.isadjacent(v3))

        :seealso: :meth:`adjacent`
        """
        return vertex in [e.next(self) for e in self._edgelist]

    #: alias of :meth:`isadjacent`
    isneighbour = isadjacent
    #: alias of :meth:`isadjacent`
    isneighbor = isadjacent

    def incidences(self) -> list[tuple[Vertex, Edge]]:
        """
        Adjacent vertices paired with the edges that lead to them

        :return: a ``(vertex, edge)`` pair for each edge leaving this vertex,
            in the order the edges were connected
        :rtype: list of (Vertex subclass, Edge)

        ``v.incidences()`` is :meth:`adjacent` and :meth:`out_edges` taken
        together: the first element of each pair is the vertex at the far end
        of the edge, the second is the edge itself, so its :attr:`Edge.cost`
        and :attr:`Edge.data` are at hand.  It is equivalent to::

            list(zip(v.adjacent(), v.out_edges()))

        but is built in a single pass, which is what the path-planning methods
        want when they expand a vertex.

        .. note::

            - In graph-theory terms :meth:`adjacent` is a vertex's *adjacency
              list*, :meth:`out_edges` its *incidence list*, and this is the
              two joined.
            - For a directed graph the pairs are for the edges *leaving* this
              vertex.  The ones entering it are :meth:`in_edges`, and the
              vertices at their tails are :meth:`predecessors`.
            - There is one pair per edge, so parallel edges to the same vertex
              give repeated vertices with different edges, and an undirected
              self-loop gives the pair twice, once for each end.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v3 = g.add_vertex(coord=[2,2], name='v3')
            >>> g.add_edge(v1, v2, cost=1.5)
            >>> g.add_edge(v1, v3, cost=2.5)
            >>> for vertex, edge in v1.incidences():
            ...     print(vertex.name, edge.cost)

        :seealso: :meth:`adjacent` :meth:`isadjacent` :meth:`successors`
            :meth:`out_edges` :meth:`in_edges` :meth:`predecessors`
            :meth:`edgeto` :meth:`edges_to`
            :meth:`edges` :meth:`edgeto` :attr:`degree`
            :meth:`Graph.adjacency`
        """
        return [(e.next(self), e) for e in self._edgelist]

    def connect(
        self,
        dest: Vertex,
        edge: Edge | None = None,
        cost: float | None = None,
        data: Any = None,
    ) -> Edge:
        """
        Connect two vertices with an edge

        :param dest: The vertex to connect to
        :type dest: ``Vertex`` subclass
        :param edge: Use this as the edge object, otherwise a new object of the
                     graph's :attr:`Graph.edge_class` is created from the
                     vertices being connected, and the ``cost`` and ``data``
                     parameters, defaults to None
        :type edge: ``Edge`` subclass, optional
        :param cost: the cost to traverse this edge, defaults to None
        :type cost: float, optional
        :param data: reference to arbitrary data associated with the edge,
                     defaults to None
        :type data: Any, optional
        :raises ValueError: either vertex has not been added to a graph, the
            vertices belong to different graphs, or ``edge`` already connects
            other vertices
        :raises TypeError: ``edge`` is not an instance of the graph's
            :attr:`Graph.edge_class`
        :return: the edge connecting the vertices
        :rtype: Edge

        ``v1.connect(v2)`` connects vertex ``v1`` to vertex ``v2``.

        .. note::

            - In an undirected graph the edge is recorded in the edge list of
              *both* vertices, so it is found from either end. In a directed
              graph it is recorded as leaving this, the start, vertex and
              entering ``dest``.
            - A supplied ``edge`` built without vertices has its ``v1`` and
              ``v2`` set to the vertices being connected. One built with
              vertices must agree with them: in either order if undirected,
              exactly ``self`` then ``dest`` if directed.
            - Both vertices must already have been added to the same graph,
              e.g. via :meth:`Graph.add_vertex`.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> e = v1.connect(v2, cost=1.414)
            >>> print(e)

        :seealso: :meth:`Edge` :meth:`Edge.connect`
        """

        if self._graph is None or dest._graph is None:
            raise ValueError(
                "both vertices must be added to a graph before being connected"
            )
        elif self._graph is not dest._graph:
            raise ValueError("vertices must belong to the same graph")
        elif isinstance(edge, Edge):
            if not isinstance(edge, self._graph._edge_class):
                raise TypeError(
                    f"expecting an edge of class {self._graph._edge_class.__name__}, "
                    f"got {type(edge).__name__}"
                )
            e = edge
            self._bind_edge(e, dest)
        else:
            e = self._graph._edge_class(self, dest, cost=cost, data=data)

        self._graph._edges[e] = None
        self._graph._connectivitychange = True
        self._connectivitychange = True

        # the one thing that differs between undirected and directed graphs
        self._graph._attach(e)

        return e

    def _bind_edge(self, edge: Edge, dest: Vertex) -> None:
        """
        Make a caller-supplied edge connect this vertex to ``dest`` (private method)

        :param edge: edge passed to :meth:`connect`
        :param dest: vertex being connected to
        :raises ValueError: ``edge`` already connects other vertices

        An edge built without vertices has its ``v1`` and ``v2`` set here.
        One built with vertices is left alone provided they agree with the
        connection being made; otherwise it is rejected before anything is
        changed, rather than leaving the graph holding an edge whose
        endpoints disagree with the vertices' edge lists.
        """
        graph = self._graph
        assert graph is not None  # connect() has checked
        if edge.v1 is None and edge.v2 is None:
            edge._set_endpoints(self, dest)
        elif not graph._endpoints_match(edge, self, dest):
            raise ValueError(
                f"edge already connects {edge.v1} and {edge.v2}, "
                f"not {self} and {dest}"
            )

    def edgeto(self, dest: Vertex) -> Edge:
        """
        The edge from this vertex to an adjacent vertex

        :param dest: an adjacent vertex
        :type dest: ``Vertex`` subclass
        :raises ValueError: ``dest`` is not adjacent to this vertex
        :return: the edge from this vertex to ``dest``
        :rtype: Edge

        .. note::

            - For a directed graph ``dest`` must be at the arrow end of the edge
            - If there are parallel edges to ``dest`` the cheapest is returned
              (the first, if none of them has a cost)

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> g.add_edge(v1, v2, cost=1.414)
            >>> print(v1.edgeto(v2))

        :seealso: :meth:`edges_to` :meth:`incidences` :meth:`adjacent`
        """
        edges = self.edges_to(dest)
        if not edges:
            raise ValueError("dest is not a neighbour")
        costed = [e for e in edges if e.cost is not None]
        if costed:
            return min(costed, key=lambda e: e.cost)  # type: ignore[arg-type,return-value]
        return edges[0]

    def edges_to(self, dest: Vertex) -> list[Edge]:
        """
        All the edges from this vertex to another vertex

        :param dest: the vertex at the far end
        :type dest: ``Vertex`` subclass
        :return: every edge from this vertex to ``dest``, in the order they
            were connected, which is empty if there are none
        :rtype: list of Edge

        This is how to see parallel edges.  :meth:`edgeto` returns the one
        edge that a path would use, the cheapest, whereas this returns them
        all.

        .. note::

            - For a directed graph only the edges leaving this vertex for
              ``dest`` are included, so ``a.edges_to(b)`` and
              ``b.edges_to(a)`` are different lists
            - For an undirected graph the edges are found from either end,
              whichever vertex they were connected from
            - A self-loop is listed once, although its vertex holds it twice
              (see :meth:`edges`)
            - Unlike :meth:`edgeto`, there being no edge is not an error, the
              list is empty
            - The list is a new one, so changing it does not change the graph

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> g.add_edge(v1, v2, cost=3)
            >>> g.add_edge(v1, v2, cost=1)
            >>> print(v1.edges_to(v2))
            >>> print(v1.edgeto(v2))

        :seealso: :meth:`edgeto` :meth:`edges` :meth:`out_edges`
            :meth:`incidences` :meth:`adjacent`
        """
        found: list[Edge] = []
        for vertex, edge in self.incidences():
            # a self-loop is held twice by its vertex, so is met twice
            if vertex is dest and not any(edge is e for e in found):
                found.append(edge)
        return found

    def edges(self) -> list[Edge]:
        """
        The edges at the vertex, in the order they were connected

        :return: the edges at this vertex
        :rtype: list of ``Edge``

        For a directed graph these are the edges leaving this vertex.  For an
        undirected graph they are the edges incident on it, whichever end it
        is.  A self-loop in an undirected graph is listed twice, once for each
        end.

        The list is a new one on every call, in the order the edges were
        connected, so changing it does not change the graph, and edges can be
        removed from the graph while iterating over it.  Use
        :meth:`Graph.remove_edge` to remove one.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> g.add_edge(v1, v2)
            >>> print(v1.edges())

        :seealso: :meth:`out_edges` :meth:`in_edges` :meth:`adjacent`
            :meth:`incidences` :meth:`edges_to` :attr:`degree`
            :meth:`Graph.edges`
        """
        return list(self._edgelist)

    def heuristic_distance(self, v2: Vertex) -> float:
        """
        Heuristic distance to another vertex

        :param v2: the other vertex
        :type v2: Vertex subclass
        :return: heuristic distance between this vertex and ``v2``
        :rtype: float

        Distance is computed according to the graph's heuristic, see
        :meth:`Graph.heuristic`. The vertex must belong to a graph,
        since the heuristic is a property of the graph, not the vertex.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[1, 2], name='v1')
            >>> v2 = g.add_vertex(coord=[4, 3], name='v2')
            >>> print(v1.heuristic_distance(v2))

        :seealso: :meth:`distance` :meth:`Graph.heuristic`
        """
        if self._graph is None:
            raise ValueError("vertex is not connected to a graph")
        if self.coord is None or v2.coord is None:
            raise ValueError("both vertices must have a coordinate")
        return self._graph.heuristic(self.coord - v2.coord)

    def distance(self, coord: ArrayLike | Vertex) -> float:
        """
        Distance from vertex to point

        :param coord: coordinates of the point
        :type coord: ndarray(n) or Vertex
        :return: distance
        :rtype: float

        Distance is computed according to the graph's metric. The vertex
        must belong to a graph, since the metric is a property of the
        graph, not the vertex.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[3,4], name='v2')
            >>> print(v1.distance(v2))

        :seealso: :attr:`Graph.metric`
        """
        if self._graph is None:
            raise ValueError("vertex is not connected to a graph")
        target = coord.coord if isinstance(coord, Vertex) else coord
        if self.coord is None or target is None:
            raise ValueError("vertex, and coord if given as a vertex, must have a coordinate")
        return self._graph.metric(self.coord - target)

    @property
    def degree(self) -> int:
        """
        Degree of vertex

        :return: degree of the vertex
        :rtype: int

        The number of edge ends at this vertex: the edges leaving it plus the
        edges entering it, so a self-loop counts twice.  The degrees of all the
        vertices of a graph sum to twice its number of edges.

        For an undirected graph this is the number of incident edges.  For a
        directed graph it is :attr:`in_degree` + :attr:`out_degree`.

        .. versionchanged:: 1.0
            For a ``DGraph`` this used to count only the edges *leaving* the
            vertex.  It is now the graph-theory degree, in plus out.  Use
            :attr:`out_degree` for the previous value.

        .. runblock:: pycon

            >>> from pgraph import DGraph
            >>> g = DGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v3 = g.add_vertex(coord=[2,2], name='v3')
            >>> g.add_edge(v1, v2)
            >>> g.add_edge(v1, v3)
            >>> g.add_edge(v3, v1)
            >>> print(v1.degree, v1.in_degree, v1.out_degree)

        .. rubric:: For graphs with parallel edges or loops

        Every edge is counted, so ``k`` parallel edges to the same vertex add
        ``k``.  A self-loop adds 2, for both kinds of graph: it has two ends.

        :seealso: :attr:`in_degree` :attr:`out_degree` :meth:`edges`
            :meth:`Graph.degree` :meth:`Graph.connectivity`
        """
        return len(self._edgelist) + len(self._inlist)

    @property
    def out_degree(self) -> int:
        """
        Number of edges leaving the vertex

        :return: out-degree of the vertex
        :rtype: int

        For an undirected graph every incident edge both leaves and enters, so
        this equals :attr:`degree`.

        :seealso: :attr:`in_degree` :attr:`degree` :meth:`out_edges`
        """
        return len(self._edgelist)

    @property
    def in_degree(self) -> int:
        """
        Number of edges entering the vertex

        :return: in-degree of the vertex
        :rtype: int

        For an undirected graph every incident edge both leaves and enters, so
        this equals :attr:`degree`.

        :seealso: :attr:`out_degree` :attr:`degree` :meth:`in_edges`
        """
        return len(self._incoming())

    def _incoming(self) -> list[Edge]:
        """The edges that end here (private method)"""
        if self._graph is None:
            return self._inlist
        return self._graph._incoming(self)

    def out_edges(self) -> list[Edge]:
        """
        The edges leaving the vertex

        :return: the outgoing edges, in the order they were connected
        :rtype: list of ``Edge``

        This is a new list, and is the same as :meth:`edges`: for an undirected
        graph every incident edge is outgoing.

        :seealso: :meth:`in_edges` :meth:`edges` :meth:`adjacent` :meth:`successors`
            :meth:`incidences` :meth:`edges_to` :attr:`out_degree`
        """
        return list(self._edgelist)

    def in_edges(self) -> list[Edge]:
        """
        The edges entering the vertex

        :return: the incoming edges, in the order they were connected
        :rtype: list of ``Edge``

        This is a new list.  For an undirected graph every incident edge is
        incoming, so it equals :meth:`edges`.

        :seealso: :meth:`out_edges` :meth:`edges` :meth:`predecessors` :attr:`in_degree`
        """
        return list(self._incoming())

    def successors(self) -> list[Vertex]:
        """
        Vertices that this vertex has an edge to

        :return: the vertices at the head of the outgoing edges
        :rtype: list of Vertex subclass

        This is :meth:`adjacent`, under its directed-graph name.

        :seealso: :meth:`predecessors` :meth:`adjacent` :meth:`incidences`
            :meth:`out_edges`
        """
        return self.adjacent()

    def predecessors(self) -> list[Vertex]:
        """
        Vertices that have an edge to this vertex

        :return: the vertices at the tail of the incoming edges
        :rtype: list of Vertex subclass

        There is one entry per edge, so a vertex joined by ``k`` parallel edges
        appears ``k`` times.  For an undirected graph this is :meth:`adjacent`.

        .. runblock:: pycon

            >>> from pgraph import DGraph
            >>> g = DGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> v3 = g.add_vertex(coord=[2,2], name='v3')
            >>> g.add_edge(v1, v3)
            >>> g.add_edge(v2, v3)
            >>> print(v3.predecessors())

        :seealso: :meth:`successors` :meth:`adjacent` :meth:`in_edges`
            :attr:`in_degree`
        """
        return [e.next(self) for e in self._incoming()]

    def _incident_edges(self) -> list[Edge]:
        """Every distinct edge at this vertex, either end (private method)"""
        seen: dict[int, Edge] = {}
        for e in (*self._edgelist, *self._incoming()):
            seen.setdefault(id(e), e)
        return list(seen.values())

    @property
    def name(self) -> str | None:
        """
        Name of the vertex

        :return: the name
        :rtype: str or None

        The name is the vertex's key in its graph: ``g[name]`` and ``name in
        g`` look it up.  A vertex that is not in a graph can be given any name,
        or none.  For one that is in a graph:

        - assigning a new name updates the graph's name table, so the vertex
          is then found under the new name and not the old
        - a name already used by another vertex in the graph raises
          ``ValueError`` and changes nothing
        - the name cannot be cleared

        .. versionchanged:: 1.0
            Renaming a vertex in a graph used to leave the graph's name table
            stale, so lookups by the new name failed.

        :seealso: :meth:`Graph.add_vertex`
        """
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        graph = self._graph
        old = self._name
        if (
            graph is not None
            and old is not None
            and graph._vertexdict.get(old) is self
        ):
            if value is None:
                raise ValueError("a vertex in a graph must have a name")
            if value != old:
                if value in graph._vertexdict:
                    raise ValueError(
                        f"the graph already has a vertex named {value!r}"
                    )
                del graph._vertexdict[old]
                graph._vertexdict[value] = self
        self._name = value

    @property
    def label(self) -> int | None:
        """
        Number of the graph component that contains the vertex

        :return: component number, counted from 0, or None if the vertex is
            not in a graph
        :rtype: int or None

        Assigned by the graph, which finds its components when this is read
        after the graph has changed, as it does for :attr:`Graph.nc`,
        :meth:`Graph.component` and :meth:`Graph.samecomponent`.  The numbering
        is by the first vertex of each component, in the order the vertices
        were added.  It is read-only.

        .. versionchanged:: 1.0
            Read-only.  It was a plain attribute, but it is computed from the
            graph's structure and meaningless to assign.  Reading it finds the
            components; in 0.6.5 it was ``None`` until something else had.

        :seealso: :attr:`Graph.nc` :meth:`Graph.component`
        """
        if self._graph is not None:
            self._graph._graphcolor()
        return self._label

    @property
    def x(self) -> float:
        """
        The x-coordinate of an embedded vertex

        :raises ValueError: the vertex has no coordinate
        :return: The x-coordinate
        :rtype: float

        .. runblock:: pycon

            >>> from pgraph import Vertex
            >>> v = Vertex(coord=[1,2], name='v1')
            >>> print(v.x)
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

        .. runblock:: pycon

            >>> from pgraph import Vertex
            >>> v = Vertex(coord=[1,2], name='v1')
            >>> print(v.y)
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

        .. runblock:: pycon

            >>> from pgraph import Vertex
            >>> v = Vertex(coord=[1,2,3], name='v1')
            >>> print(v.z)
        """
        if self.coord is None:
            raise ValueError("vertex has no coordinate")
        return self.coord[2]

    def closest(self) -> tuple[Vertex | None, float]:
        """
        Return the closest vertex to this vertex

        :return: closest vertex and its distance
        :rtype: Vertex subclass or None, float

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

        :seealso: :meth:`Graph.closest`
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
        or outgoing), from its graph. The ``Vertex`` object itself is
        not deleted.

        .. note:: This is a thin convenience wrapper around
            :meth:`Graph.remove_vertex`.

        :rtype: None

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[1,1], name='v2')
            >>> g.add_edge(v1, v2)
            >>> v2.remove()
            >>> print(g)

        :seealso: :meth:`Graph.remove_vertex` :meth:`Edge.remove`
        """
        if self._graph is None:
            raise ValueError("vertex is not connected to a graph")
        self._graph.remove_vertex(self)


# UVertex and DVertex were two vertex classes that differed only in how an edge
# was recorded at them, which is the graph's decision.  They remain as names
# for Vertex, so that existing code that subclasses or constructs them runs.
def __getattr__(name: str) -> Any:
    if name in ("UVertex", "DVertex"):
        warnings.warn(
            f"{name} is deprecated, use Vertex instead: a vertex has no "
            "direction of its own, the graph it belongs to decides",
            DeprecationWarning,
            stacklevel=2,
        )
        return Vertex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
