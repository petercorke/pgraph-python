from pgraph import DGraph
g = DGraph()
for name, xy in dict(A=(0, 1), B=(2, 1.7), C=(3.3, 0.3), D=(1.2, -0.5)).items():
    g.add_vertex(xy, name=name);
for a, b in ["AB", "AB", "AD", "BC", "CB", "DC", "CC"]:
    g.add_edge(a, b);
g.plot(block=None)
