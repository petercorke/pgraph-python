from pgraph import UGraph, UVertex, Edge
import itertools


class Frame(UVertex):
    def __str__(self):
        return f"Frame: {self.name}"

    def __repr__(self):
        return self.__str__()

    def neighbours(self):
        """
        Neighbours of a vertex

        ``v.neighbours()`` is a list of neighbours of this vertex.

        .. note:: For a directed graph the neighbours are those on edges leaving this vertex
        """
        return [e.next(self) for e in self._edgelist]


class Transform(Edge):

    def __str__(self):
        return f"Transform: {self.v1.name}:{self.v2.name} # [{self.data.strline()}]"

    def set(self, x):
        self.data = x


class PoseGraph(UGraph):

    def __init__(self):
        super().__init__()
        self.frames = {}
        self.transforms = {}
        self.cache = {}
        self.dirty = False

    def add_frame(self, name):
        f = Frame(name=name)
        self.frames[name] = f
        self.add_vertex(f)
        return f

    def set_transform(self, x, f1, f2):
        if isinstance(f1, str):
            f1 = self.frames[f1]
        if isinstance(f2, str):
            f2 = self.frames[f2]

        T = Transform(f1, f2, cost=1, data=x)
        f1.connect(f2, edge=T)
        return T

    def transform(self, start, end):
        # key = (frm, to)
        # if key not in self.cache:
        #     self.cache[key] = self.compute_transform(frm, to)
        # return self.cache[key]
        vertices, len = self.path_BFS(start, end)
        edges = []
        for first, next in itertools.pairwise(vertices):
            e = first.edgeto(next)
            if e.v1 == next:
                edges.append((e, False))
            elif e.v2 == next:
                edges.append((e, True))

        print(vertices)
        print(edges)

    def map(self, frm, to):
        pass


if __name__ == "__main__":

    from spatialmath import SE3

    pg = PoseGraph()
    f1 = pg.add_frame("frame1")
    f2 = pg.add_frame("frame2")
    f3 = pg.add_frame("frame3")
    f4 = pg.add_frame("frame4")

    print(pg)

    x = SE3()
    print(x)

    t12 = pg.set_transform(x, f1, f2)
    t13 = pg.set_transform(x, "frame1", "frame3")

    print(pg)

    print(f1)
    print(t12)

    t12.set(SE3.Trans(1, 2, 3))

    x = pg.transform(start=f2, end=f3)  # keeps a cache of paths
    print(x)
    # P3 = g.map(p1, from=f1, to=f
