import pgraph
import numpy as np
g = pgraph.UGraph()
np.random.seed(0) # ensure repeatable results
for i in range(5):
  g.add_vertex(np.random.rand(2))
g.add_edge(g[0], g[1])
g.add_edge(g[0], g[2])
g.add_edge(g[0], g[3])
g.add_edge(g[1], g[2])
g.add_edge(g[1], g[3])
g.add_edge(g[3], g[4])
g.plot()
