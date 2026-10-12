from pgraph import UGraph
import numpy as np
np.random.seed(0)
g = UGraph()
for i in range(5):
    g.add_vertex(np.random.rand(2));
edges = []
for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (3, 4)]:
    edges.append(g.add_edge(g[i], g[j]));
g.plot(block=None)
g.highlight_edge(edges[2])
