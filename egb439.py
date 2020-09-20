import json
from PGraph import *

with open('places.json', 'r') as f:
    places = json.loads(f.read())

with open('routes.json', 'r') as f:
    routes = json.loads(f.read())

# print(x)

# print(x['Brisbane'])

g = UGraph()

for name, info in places.items():
    g.add_node(name=name, coord=info["utm"])

for route in routes:
    g.add_edge(route[0], route[1], cost=route[2])

print(g)
# g.plot()

p = g.Astar('Hughenden', 'Brisbane')
print(' '.join([str(x) for x in p]))

# g.dotfile()

g.plot(block=False)
g.highlight_path(p)

print(g.adjacency)