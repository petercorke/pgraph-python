import json
from pgraph import UGraph, DGraph

# load the JSON file
with open('queensland.json', 'r') as f:
    data = json.loads(f.read())

# create an undirected graph
g = UGraph()

# create a vertex for every place, by providing the coordinate the
# resulting graph will be embedded
for name, info in data['places'].items():
    g.add_vertex(name=name, coord=info["utm"])

# create an edge for every route, and the cost is the driving distance
for route in data['routes']:
    g.add_edge(route['start'], route['end'], cost=route['distance'])

for route in data['routes']:
    g.add_edge(route['start'], route['end'], 
               cost=route['distance'] / route['speed'])

# print the graph in tabular form
print(g)
print(g.edges())

# Heuristic for minimum time problem
# g.heuristic = lambda x: np.linalg.norm(x) / 100

p, length, parents = g.path_Astar('Hughenden', 'Brisbane', 
                                  verbose=True, summary=True)
# compute the path using A*, the result is a list of UVertex objects

print(f"shortest path has length {length:.1f}:", 
      '->'.join([str(x.name) for x in p]))

# plot the graph, and overlay the path
g.plot()
g.highlight_path(p, alpha=0.5, scale=2)

# turn the vertex parent information into a tree, it's a dict that maps a
# vertex to its parent.
tree = DGraph.Dict(parents)
tree.showgraph()  # display it via the browser
