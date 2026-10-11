from pgraph import UGraph
g = UGraph()
v1 = g.add_vertex(coord=[0,0], name='v1')
v2 = g.add_vertex(coord=[1,1], name='v2')
v3 = g.add_vertex(coord=[2,2], name='v3')
g.add_edge(v1, v2)
g.add_edge(v2, v3)
g.plot(block=None)
g.highlight_path([v1, v2, v3], block=None)