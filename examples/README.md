# Examples

This folder contains two JSON files that describe towns and major roads in Queensland Australia.

- `places.json` describes place names, UTM coordinates
- `routes.json` describes routes between towns as a set of tuples (start town, end town, distance, road type) where road type is 1 for a major road, and 2 for a minor road (worse quality, lower average speed)

`roads.py` loads these files, creates an embedded graph, and find an 
`benchmark.py` times the key operations (building, copying, pickling, components,
`closest`, the matrix forms and the three path planners) on random geometric graphs
of 100, 1 000 and 10 000 vertices and prints a table, including how each cost grows
with the size of the graph.  `--directed` uses `DGraph`, `--quick` skips the largest
graph, and `--rst` prints the table as reST for the documentation.  Timings depend
on the machine, so compare runs made on the same one.
