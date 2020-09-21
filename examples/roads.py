import json
from pgraph import *

# load the JSON files
with open('places.json', 'r') as f:
    places = json.loads(f.read())

with open('routes.json', 'r') as f:
    routes = json.loads(f.read())

# create an undirected graph
g = UGraph()

# create a vertex for every place, by providing the coordinate the
# resulting graph will be embedded
for name, info in places.items():
    g.add_vertex(name=name, coord=info["utm"])

# create an edge for every route, and the cost is the driving distance
for route in routes:
    g.add_edge(route[0], route[1], cost=route[2])

# print the graph in tabular form
print(g)

# compute the path using A*, the result is a list of UVertex objects
p = g.path_Astar('Hughenden', 'Brisbane')
print(' '.join([str(x) for x in p]))

# plot the graph, and overlay the path
g.plot(block=False)
g.highlight_path(p)