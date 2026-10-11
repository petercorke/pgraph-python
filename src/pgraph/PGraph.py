from __future__ import annotations

from abc import ABC, abstractmethod
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections.abc import Iterable, Iterator
import tempfile
import webbrowser
from typing import IO, Any, Callable, ClassVar, overload
from numpy.typing import ArrayLike, NDArray

from spatialmath.base.graphics import axes_logic


def _L1(v: NDArray) -> float:
    """L1 (Manhattan) norm, the built-in ``"L1"`` metric"""
    return float(np.linalg.norm(v, 1))


def _L2(v: NDArray) -> float:
    """L2 (Euclidean) norm, the built-in ``"L2"`` metric"""
    return float(np.linalg.norm(v))


def _SE2(v: NDArray) -> float:
    """Mixed norm for (x, y, theta), the built-in ``"SE2"`` metric"""
    if len(v) != 3:
        raise ValueError(
            f"SE2 metric requires a 3-element (x, y, theta) vector, got length {len(v)}"
        )
    # wrap angle to range [-pi, pi)
    v[2] = (v[2] + np.pi) % (2 * np.pi) - np.pi
    return float(np.linalg.norm(v))


#: the built-in metrics by name.  They are module level, and pickled by name,
#: so that a graph using one can be pickled.
_METRICS: dict[str, Callable[[NDArray], float]] = {
    "L1": _L1,
    "L2": _L2,
    "SE2": _SE2,
}


class Graph(ABC):

    #: concrete vertex type for this graph kind, provided by :class:`UGraph`
    #: and :class:`DGraph` -- lets :meth:`add_vertex` and :meth:`vertex_copy`
    #: be defined once here rather than duplicated per subclass.
    _vertex_cls: ClassVar[type[Vertex]]

    def __init__(
        self,
        *,
        metric: Callable[[NDArray], float] | str | None = None,
        heuristic: Callable[[NDArray], float] | str | None = None,
        verbose: bool = False,
        dim: int | None = None,
        vertex_class: type[Vertex] | None = None,
        edge_class: type[Edge] | None = None,
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
        :param dim: required length of every vertex's ``coord``, defaults to
            None (unconstrained -- vertices may have coordinates of any
            length, or none at all)
        :type dim: int, optional
        :param vertex_class: class of the vertices this graph creates, defaults
            to ``UVertex`` for a :class:`UGraph` and ``DVertex`` for a
            :class:`DGraph`
        :type vertex_class: subclass of ``UVertex`` or ``DVertex``, optional
        :param edge_class: class of the edges this graph creates, defaults to
            :class:`Edge`
        :type edge_class: subclass of ``Edge``, optional
        :raises ValueError: ``dim`` is given but is not a positive integer
        :raises TypeError: ``vertex_class`` is not a subclass of this graph's
            own vertex kind, or ``edge_class`` is not a subclass of ``Edge``

        This is the common base class of :class:`UGraph` and :class:`DGraph`
        and should not be instantiated directly.

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
        kind = type(self)._vertex_cls
        if vertex_class is None:
            vertex_class = kind
        elif not (isinstance(vertex_class, type) and issubclass(vertex_class, kind)):
            raise TypeError(
                f"vertex_class must be a subclass of {kind.__name__} for a "
                f"{type(self).__name__}, got {vertex_class!r}"
            )
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
    def Dict(cls, d: dict, reverse: bool = False, **kwargs: Any) -> Graph:
        """
        Create graph from parent/child dictionary

        :param d: dictionary that maps from ``Vertex`` subclass to ``Vertex`` subclass
        :type d: dict
        :param reverse: reverse link direction, defaults to False
        :param kwargs: optional arguments passed to the graph constructor, for
            example ``vertex_class`` or ``edge_class``
        :raises TypeError: ``vertex_class`` or ``edge_class`` is not a
            suitable class, see :meth:`Graph.__init__`
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
            suitable class, see :meth:`Graph.__init__`
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

        if isinstance(g, UGraph):
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
    def vertex_class(self) -> type[Vertex]:
        """
        Class of the vertices this graph creates

        :return: vertex class
        :rtype: subclass of ``UVertex`` or ``DVertex``

        This is read-only.  It is set by the ``vertex_class`` argument of the
        graph constructor, see :meth:`__init__`, and is ``UVertex`` for a
        :class:`UGraph` and ``DVertex`` for a :class:`DGraph` if not given.

        .. runblock:: pycon

            >>> from pgraph import UGraph, UVertex
            >>> class MyVertex(UVertex):
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
        graph constructor, see :meth:`__init__`, and is :class:`Edge` if not
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

    def validate(self) -> list[str]:
        """
        Check that the graph's records agree with each other

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
        directed = isinstance(self, DGraph)

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
              ``"SE2"``) is pickled by name.  A function you supply must be
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

    # pickling and copying

    #: the graph's own containers, which are saved as a flat description
    _STRUCTURE = ("_vertexlist", "_vertexdict", "_edges")

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
            v1._attach(e, v2)

    def add_vertex(
        self, coord: ArrayLike | Vertex | None = None, name: str | None = None
    ) -> Vertex:
        """
        Add a vertex to the graph

        :param coord: coordinate for an embedded graph, or an existing vertex
            of this graph's own kind (``UVertex`` for :class:`UGraph`,
            ``DVertex`` for :class:`DGraph`) to add as-is, defaults to None
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
          :attr:`vertex_class` (by default any ``UVertex`` for a
          :class:`UGraph`, any ``DVertex`` for a :class:`DGraph`) and adds it
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

            >>> from pgraph import UGraph, UVertex
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0])
            >>> repr(v1)
            >>> print(v1)
            >>> print(v1.name)
            >>> v2 = g.add_vertex(UVertex(coord=[1,1], name='v2'))
            >>> print(v2.name)
            >>> g['v2']

        If the graph was constructed with a required ``dim`` (see
        :meth:`Graph.__init__`), every embedded vertex must have a
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
        vertex.name = name
        self._vertexlist.append(vertex)
        self._vertexdict[vertex.name] = vertex
        if self._verbose:
            print(f"New vertex {vertex.name}: {vertex.coord}")
        vertex._graph = self
        self._connectivitychange = True
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
        in a new graph -- of ``cls``'s own vertex kind, per :attr:`_vertex_cls`.

        .. runblock:: pycon

            >>> from pgraph import UGraph, DGraph
            >>> g = UGraph()
            >>> v = g.add_vertex(coord=[1,2], name='v1')
            >>> newv = DGraph.vertex_copy(v)
            >>> print(newv)

        :seealso: :meth:`Vertex.copy`
        """
        return cls._vertex_cls(coord=vertex.coord, name=vertex.name)

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

        .. note:: Vertices have coordinates and the edge ``cost`` defaults to the distance between them, but if the vertices have no coordinates the edge cost must be set explicitly, otherwise it will be ``None`` and path planning will fail with a ``ValueError``.  See :meth:`_require_cost` for details.
      
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
        v1._detach(edge, v2)

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
        self._connectivitychange = True

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

        Number of components in the graph. Components are labeled from 0 to ``g.nc-1``.

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
            exactly 3 elements; raises :exc:`ValueError` otherwise.

        The metric is used by :meth:`closest` and :meth:`distance`

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> g.metric = 'L1'
            >>> print(g.metric(np.r_[3, -4]))
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
            exactly 3 elements; raises :exc:`ValueError` otherwise.

        The heuristic distance is only used by the A* planner :meth:`path_Astar`.

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> import numpy as np
            >>> g = UGraph()
            >>> g.heuristic = 'L2'
            >>> print(g.heuristic(np.r_[3, 4]))
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

        .. runblock:: pycon

            >>> from pgraph import UGraph
            >>> g = UGraph()
            >>> v1 = g.add_vertex(coord=[0,0], name='v1')
            >>> v2 = g.add_vertex(coord=[10,10], name='v2')
            >>> vertex, d = g.closest([1, 1])
            >>> print(vertex, d)

        :seealso: :meth:`metric`
        """
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

    def edges(self) -> list[Edge]:
        """
        Return a set of all edges in graph (base class method)

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

    def plot(
        self,
        colorcomponents: bool = True,
        force2d: bool = False,
        vopt: dict | None = None,
        eopt: dict | None = None,
        text: dict | bool | None = None,
        block: bool | None = False,
        grid: bool = True,
        ax: Any = None,
    ) -> None:
        """
        Plot graph as a vertex/edge diagram using Matplotlib

        :param vopt: vertex format, defaults to 12pt o-marker
        :type vopt: dict, optional
        :param eopt: edge format, defaults to 3pt line width
        :type eopt: dict, optional
        :param text: text label format, defaults to None
        :type text: dict, bool or None, optional
        :param colorcomponents: color vertices and edges by component, defaults to True
        :type colorcomponents: bool, optional
        :param force2d: plot only the first two coordinates of each vertex
            even if they have more, defaults to False
        :type force2d: bool, optional
        :param block: block until figure is dismissed, defaults to False. If
            None the figure is not shown at all, which is useful when
            adding to the plot or saving it.
        :type block: bool or None, optional
        :raises ValueError: the graph is empty, or a vertex has no coordinate
            or fewer than two coordinate values

        The graph is plotted using Matplotlib according to the dimensionality of the vertex
        coordinates (set by the `dim` parameter to graph constructor).  If the dimensionality is
         
        - 2, then a 2D plot is generated.
        - 3, then a 3D plot is generated.
        - 4 or more, then a 3D plot is generated using the first three coordinates.

        If a vertex has no coordinate or fewer than two values, a ValueError is raised.

        If ``colorcomponents`` is True then all edges and vertices within a component are assigned a unique
        color.

        If ``text`` is a dict it is used to format text labels for the vertices
        which are the vertex names.  If ``text`` is None or True default
        formatting is used.  If ``text`` is False no labels are added.

        .. note:: Only embedded graphs can be plotted: every vertex must have
            a coordinate of at least two values.  A coordinate of more than
            two values is drawn in 3D, using the first three.

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

        :seealso: :meth:`highlight_path`
        """
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

        vopt = {**dict(marker="o", markersize=12), **(vopt or {})}
        eopt = {**dict(linewidth=3), **(eopt or {})}
        # text labels: None or True is default formatting, False is no labels
        textopt: dict | None = None
        if text is not False:
            textopt = text if isinstance(text, dict) else {}

        if colorcomponents:
            color = plt.cm.coolwarm(np.linspace(0, 1, self.nc))

        if ndim == 2 or force2d:
            # 2D plotting
            if ax is None:
                ax = axes_logic(ax, 2)
            for c in range(self.nc):
                # for each component
                for vertex in self.component(c):
                    if textopt is not None:
                        ax.text(vertex.x, vertex.y, f"  {vertex.name}", **textopt)
                    if colorcomponents:
                        ax.plot(vertex.x, vertex.y, color=color[c, :], **vopt)
                        for v in vertex.adjacent():
                            ax.plot(
                                [vertex.x, v.x],
                                [vertex.y, v.y],
                                color=color[c, :],
                                **eopt,
                            )
                    else:
                        ax.plot(vertex.x, vertex.y, **vopt)
                        for v in vertex.adjacent():
                            ax.plot([vertex.x, v.x], [vertex.y, v.y], **eopt)
        else:
            # 3D or higher plotting, just do (x, y, z)
            if ax is None:
                ax = axes_logic(ax, 3)
            for c in range(self.nc):
                # for each component
                for vertex in self.component(c):
                    if textopt is not None:
                        ax.text(
                            vertex.x, vertex.y, vertex.z, f"  {vertex.name}", **textopt
                        )
                    if colorcomponents:
                        ax.plot(
                            vertex.x,
                            vertex.y,
                            vertex.z,
                            **{**dict(color=color[c, :]), **vopt},
                        )
                        for v in vertex.adjacent():
                            ax.plot(
                                [vertex.x, v.x],
                                [vertex.y, v.y],
                                [vertex.z, v.z],
                                **{**dict(color=color[c, :]), **eopt},
                            )
                    else:
                        ax.plot(vertex.x, vertex.y, vertex.z, **vopt)
                        for v in vertex.adjacent():
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
        p1 = edge.v1
        p2 = edge.v2
        if p1 is None or p2 is None:
            raise ValueError("edge does not connect two vertices")
        plt.plot(
            [p1.x, p2.x], [p1.y, p2.y], color=color, linewidth=3 * scale, alpha=alpha
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
        if isinstance(vertex, Iterable):
            for n in vertex:
                if isinstance(n, str):
                    n = self[n]
                plt.plot(n.x, n.y, "o", color=color, markersize=12 * scale, alpha=alpha)
        else:
            plt.plot(
                vertex.x, vertex.y, "o", color=color, markersize=12 * scale, alpha=alpha
            )

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
              - ``direction``, ``colorcomponents``, ``costs``
            * - ``"mermaid"``
              - ``.mmd`` ``.mermaid``
              - no
              - ``direction``, ``colorcomponents``, ``costs``, ``fence``
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
              - ``engine``, ``direction``, ``colorcomponents``, ``costs``

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

            - The graph is undirected if it is a subclass of ``UGraph``
            - The graph is directed if it is a subclass of ``DGraph``

        .. note:: The ``graphviz`` formatters ``dot`` and ``neato`` can be used to render the
            graph in various formats including PDF, PNG, JPG, GIF, SVG, etc.  The formmatters are
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
        :param options: ``engine``, ``direction``, ``colorcomponents`` and
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
        if isinstance(self, UGraph):
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
        Convert the graph to a Laplacian matrix

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
            if e.v1 is e.v2 and isinstance(self, UGraph):
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
        self._graphcolor()  # ensure labels are uptodate
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

        Test whether vertices belong to the same component.  For a:

        - directed graph this implies a path between them
        - undirected graph there is not necessarily a path between them

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
        self._graphcolor()  # ensure labels are uptodate

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
        frontier: list[Vertex] = [S]
        explored: list[Vertex] = []
        parent: dict[Vertex, Vertex] = {}
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
            for n in x.adjacent():
                if n is x:
                    continue  # self-loop, never part of a shortest path
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
            length += self._require_cost(p.edgeto(x))
            path.insert(0, p)
            x = p

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

        frontier: list[Vertex] = [S]
        explored: list[Vertex] = []
        parent: dict[Vertex, Vertex] = {}
        f: dict[Vertex, float] = {S: 0}  # evaluation function

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
                if n is x:
                    continue  # self-loop, never part of a shortest path
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

        frontier: list[Vertex] = [S]
        explored: list[Vertex] = []
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

            i = np.argmin([f[n] for n in frontier])  # minimum f in frontier
            x = frontier.pop(i)
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


class UGraph(Graph):
    """
    Class for undirected graphs

    Vertices are :class:`UVertex` objects, or instances of the subclass given
    as ``vertex_class``. Edges are :class:`Edge` objects, or instances of the
    subclass given as ``edge_class``; see :class:`Graph` for the
    constructor arguments and the exceptions they raise.

    .. inheritance-diagram:: UGraph

    :seealso: :class:`Graph` :class:`DGraph`
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
                vertex._label = None
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
                            v._label = label
                            for n in v.adjacent():
                                if n.label is None:
                                    q.append(n)
                        lastlabel = label
                        assignment = True
                        break
                if not assignment:
                    break

            self._ncomponents = 0 if lastlabel is None else lastlabel + 1


class DGraph(Graph):
    """
    Class for directed graphs

    Vertices are :class:`DVertex` objects, or instances of the subclass given
    as ``vertex_class``. Edges are :class:`Edge` objects, or instances of the
    subclass given as ``edge_class``; see :class:`Graph` for the
    constructor arguments and the exceptions they raise.

    .. inheritance-diagram:: DGraph

    :seealso: :class:`Graph` :class:`UGraph`
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
                vertex._label = None
                vertex._connectivitychange = False

            # initial labeling pass
            merge: dict[int, int] = {}
            nextlabel = 1
            for v in self:
                if v.label is None:
                    # no label, try to inherit one from a neighbour
                    for n in v.adjacent():
                        if n.label is not None:
                            # neighbour has a label
                            v._label = n.label
                            break

                if v.label is None:
                    # still not labeled, assign a new label
                    v._label = nextlabel
                    nextlabel += 1

                label = v.label
                assert label is not None

                # now look for clashes
                for n in v.adjacent():
                    if n.label is None:
                        # neighbour has no label, give it this one
                        n._label = label
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
                v._label = vlabel
                unique.add(vlabel)

            final = {u: i for i, u in enumerate(unique)}
            for v in self:
                vlabel = v.label
                assert vlabel is not None
                v._label = final[vlabel]

            return len(unique)
        else:
            # no coloring performed
            return None


# ========================================================================== #


_ENDPOINT_MESSAGE = (
    "Edge.{0} is read-only. Give the vertices to the constructor, Edge(v1, v2), "
    "or connect the edge with v1.connect(v2, edge=e) or e.connect(v1, v2)"
)


class Edge:
    """
    Edge class

    Is used to represent directed directed and undirected edges.

    Each edge has:
    - ``cost`` cost of traversing this edge, required for planning methods
    - ``data`` reference to arbitrary data associated with the edge
    - ``v1`` first vertex, start vertex for a directed edge (read-only)
    - ``v2`` second vertex, end vertex for a directed edge (read-only)

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

            >>> from pgraph import UVertex, Edge, UGraph
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
    Base class for vertices of directed and non-directed graphs.

    Each vertex has:
        - ``name`` which is its key in the graph, so renaming a vertex that is
          in a graph updates the graph
        - ``label`` an int indicating which graph component contains it,
          read-only: the graph assigns it
        - ``coord`` the coordinate in an embedded graph (optional)

    and its edges are available from :meth:`edges`, :meth:`out_edges` and
    :meth:`in_edges`.

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
        :meth:`Graph.add_vertex` for that.

        .. runblock:: pycon

            >>> from pgraph import UVertex
            >>> v1 = UVertex(coord=[0,0], name='v1')
            >>> print(v1)

        :seealso: :meth:`Graph.add_vertex`
        """
        self._edgelist: list[Edge] = []
        if coord is None:
            self.coord = None
        else:
            self.coord = np.r_[coord]
        self._name = name
        self._label: int | None = None
        self._connectivitychange = True
        self._edgelist = []
        #: edges ending at this vertex.  Only a directed vertex fills it: an
        #: undirected edge is already in ``_edgelist`` at both of its ends.
        self._inlist: list[Edge] = []
        self._graph: Graph | None = None  # reference to owning graph
        # print('Vertex init', type(self))

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

            >>> from pgraph import UVertex, DGraph
            >>> v = UVertex(coord=[1,2], name='v1')
            >>> v2 = v.copy(cls=DGraph)
            >>> print(v2)

        :seealso: :meth:`UGraph.vertex_copy` :meth:`DGraph.vertex_copy`
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

            - If the vertices subclass ``UVertex`` the edge is undirected: it is
              recorded in the edge list of *both* vertices, so it is found from
              either end. If they subclass ``DVertex`` the edge is directed: it
              is recorded only in the edge list of this, the source, vertex.
            - A supplied ``edge`` built without vertices has its ``v1`` and
              ``v2`` set to the vertices being connected. One built with
              vertices must agree with them: in either order if undirected,
              exactly ``self`` then ``dest`` if directed.
            - Both vertices must already have been added to the same graph,
              e.g. via :meth:`Graph.add_vertex` -- since a graph only ever
              accepts its own vertex subclass (see :meth:`UGraph.add_vertex`,
              :meth:`DGraph.add_vertex`), this also rules out connecting a
              ``UVertex`` to a ``DVertex``.

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

        # the one thing that differs between undirected and directed vertices
        self._attach(e, dest)

        return e

    def _attach(self, edge: Edge, dest: Vertex) -> None:
        """
        Record a new edge in the vertices' edge lists (private method)

        :param edge: the edge just created or supplied by :meth:`connect`, with
            ``edge.v1`` and ``edge.v2`` set and already part of the graph
        :param dest: the vertex ``connect`` was called with
        :raises NotImplementedError: not provided by this vertex kind

        This is the hook that :meth:`connect` calls once the connection has
        been validated and the edge registered with the graph, so it is never
        called for a connection that is refused.  :class:`UVertex` records the
        edge at both ends and :class:`DVertex` at the source only.  A vertex
        kind of your own supplies this, not :meth:`connect`, which is defined
        once.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define how an edge is attached; "
            "subclass UVertex or DVertex"
        )

    def _detach(self, edge: Edge, dest: Vertex) -> None:
        """
        Remove an edge from the vertices' edge lists (private method)

        :param edge: the edge being removed, with ``edge.v1`` and ``edge.v2``
            still set
        :param dest: the vertex at the other end, ``edge.v2``
        :raises NotImplementedError: not provided by this vertex kind

        The twin of :meth:`_attach`: it undoes exactly what that recorded.
        Called by :meth:`Graph.remove_edge` before the edge's endpoints
        are cleared.  An edge that is absent from a list is skipped rather
        than treated as an error.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define how an edge is detached; "
            "subclass UVertex or DVertex"
        )

    def _endpoints_match(self, edge: Edge, dest: Vertex) -> bool:
        """
        Test if an edge already connects this vertex to ``dest`` (private method)

        A directed edge must run from this vertex to ``dest``; see
        :meth:`UVertex._endpoints_match` for the undirected relaxation.
        """
        return edge.v1 is self and edge.v2 is dest

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
        if edge.v1 is None and edge.v2 is None:
            edge._set_endpoints(self, dest)
        elif not self._endpoints_match(edge, dest):
            raise ValueError(
                f"edge already connects {edge.v1} and {edge.v2}, "
                f"not {self} and {dest}"
            )

    def edgeto(self, dest: Vertex) -> Edge:
        """
        Get edge connecting vertex to specific neighbour

        :param dest: a neigbouring vertex
        :type dest: ``Vertex`` subclass
        :raises ValueError: ``dest`` is not a neighbour
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
        """
        edges = [e for n, e in self.incidences() if n is dest]
        if not edges:
            raise ValueError("dest is not a neighbour")
        costed = [e for e in edges if e.cost is not None]
        if costed:
            return min(costed, key=lambda e: e.cost)  # type: ignore[arg-type,return-value]
        return edges[0]

    def edges(self) -> list[Edge]:
        """
        Return a list of the edges for vertex

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
            :meth:`incidences` :attr:`degree` :meth:`Graph.edges`
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

        :seealso: :meth:`metric`
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
        return self._inlist

    def out_edges(self) -> list[Edge]:
        """
        The edges leaving the vertex

        :return: the outgoing edges, in the order they were connected
        :rtype: list of ``Edge``

        This is a new list, and is the same as :meth:`edges`: for an undirected
        graph every incident edge is outgoing.

        :seealso: :meth:`in_edges` :meth:`edges` :meth:`adjacent` :meth:`successors`
            :meth:`incidences` :attr:`out_degree`
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

        :return: component number, counted from 0, or None if the components
            have not been found yet
        :rtype: int or None

        Assigned by the graph when its components are found, which happens when
        :attr:`Graph.nc`, :meth:`Graph.component` or
        :meth:`Graph.samecomponent` is used after the graph has changed.
        It is read-only.

        .. versionchanged:: 1.0
            Read-only.  It was a plain attribute, but it is computed from the
            graph's structure and meaningless to assign.

        :seealso: :attr:`Graph.nc` :meth:`Graph.component`
        """
        return self._label

    @property
    def x(self) -> float:
        """
        The x-coordinate of an embedded vertex

        :raises ValueError: the vertex has no coordinate
        :return: The x-coordinate
        :rtype: float

        .. runblock:: pycon

            >>> from pgraph import UVertex
            >>> v = UVertex(coord=[1,2], name='v1')
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

            >>> from pgraph import UVertex
            >>> v = UVertex(coord=[1,2], name='v1')
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

            >>> from pgraph import UVertex
            >>> v = UVertex(coord=[1,2,3], name='v1')
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


class UVertex(Vertex):
    """
    Vertex subclass for undirected graphs

    This class can be inherited to provide user objects with graph capability.

    :meth:`~Vertex.connect` adds the new edge to *both* vertices' edge
    lists, so it is discovered from either end, and accepts a supplied edge
    whose endpoints are the two vertices in either order.

    .. runblock:: pycon

        >>> from pgraph import UGraph
        >>> g = UGraph()
        >>> v1 = g.add_vertex(coord=[0,0], name='v1')
        >>> v2 = g.add_vertex(coord=[1,1], name='v2')
        >>> v1.connect(v2, cost=1.414)
        >>> print(v1.edges())
        >>> print(v2.edges())

    .. inheritance-diagram:: UVertex

    :seealso: :class:`Vertex` :class:`DVertex`
    """

    def _attach(self, edge: Edge, dest: Vertex) -> None:
        """
        Record an undirected edge at both ends (private method)

        The edge is added to *both* vertices' edge lists, so it is discovered
        from either end.  A self-loop is therefore held twice by its vertex,
        which is why it counts 2 towards the degree.
        """
        self._edgelist.append(edge)
        dest._edgelist.append(edge)

    def _incoming(self) -> list[Edge]:
        """The edges that end here: for an undirected vertex, all of them"""
        return self._edgelist

    def _detach(self, edge: Edge, dest: Vertex) -> None:
        """
        Remove an undirected edge from both ends (private method)

        A self-loop was recorded twice by :meth:`_attach`, so it is removed
        twice.
        """
        for vertex in (self, dest):
            if edge in vertex._edgelist:
                vertex._edgelist.remove(edge)

    def _endpoints_match(self, edge: Edge, dest: Vertex) -> bool:
        """
        Test if an edge already connects this vertex and ``dest`` (private method)

        An undirected edge has no direction, so either order is accepted.
        """
        return (edge.v1 is self and edge.v2 is dest) or (
            edge.v1 is dest and edge.v2 is self
        )


class DVertex(Vertex):
    """
    Vertex subclass for directed graphs

    This class can be inherited to provide user objects with graph capability.

    :meth:`~Vertex.connect` adds the new edge only to *this* vertex's edge
    list, so it is discoverable only from the start of the directed edge, and a
    supplied edge must run exactly from this vertex to the other.

    .. runblock:: pycon

        >>> from pgraph import DGraph
        >>> g = DGraph()
        >>> v1 = g.add_vertex(coord=[0,0], name='v1')
        >>> v2 = g.add_vertex(coord=[1,1], name='v2')
        >>> v1.connect(v2, cost=1.414)
        >>> print(v1.edges())
        >>> print(v2.edges())

    .. inheritance-diagram:: DVertex

    :seealso: :class:`Vertex` :class:`UVertex`
    """

    def _attach(self, edge: Edge, dest: Vertex) -> None:
        """
        Record a directed edge at both ends (private method)

        The edge goes in this vertex's list of outgoing edges and in
        ``dest``'s list of incoming ones.
        """
        self._edgelist.append(edge)
        dest._inlist.append(edge)

    def _detach(self, edge: Edge, dest: Vertex) -> None:
        """
        Remove a directed edge from its source and from where it ends
        (private method)
        """
        if edge in self._edgelist:
            self._edgelist.remove(edge)
        if edge in dest._inlist:
            dest._inlist.remove(edge)


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
