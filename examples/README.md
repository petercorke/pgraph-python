# Examples

## Central London

A real-world example: a road network taken from OpenStreetMap, a directed graph with one-way streets, and the shortest drive between two landmarks.

- `london.ipynb` a Jupyter notebook that builds a directed, embedded graph of the roads that a car may use in a 5.5 km by 2.2 km area from Buckingham Palace to the Tower of London, keeps the part of it that can be driven around in, and finds the shortest route between landmarks with A\*, there and back.  The two routes differ because of the one-way streets.  It also shows why `plot` is not the tool for a map, and how to select the part of a graph in a region.
- `osm.py` helper module with the parts that are about OpenStreetMap rather than about graphs: the request to the Overpass API, which kinds of way a car may use and in which direction, a transverse Mercator projection from latitude and longitude to metres, and the cache.  It needs only the standard library and numpy.
- `london.json` data file, the ways and nodes of the area saved from OpenStreetMap, one record per line.  Each way is a name, a flag for the direction in which a car may drive along it (1 along its nodes, -1 against them, 0 both), and a list of node numbers; each node is a latitude and longitude.  The notebook reads this file, so it runs without a network connection.  Delete the file, or call `fetch_roads(..., refresh=True)`, to fetch it again, which takes about 20 seconds.

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), available under the [Open Database License](https://opendatacommons.org/licenses/odbl/).  `london.json` is a derived database of that data, so it carries the same licence and attribution.

## Queensland outback

This folder contains an example published in Robotics, Vision & Control (3rd edition) §5.3:

- `roads.ipynb` a Jupyter notebook that creates an embedded graph, and finds the shortest path from Hughenden to Brisbane

- `roads.py` loads these files, creates an embedded graph, and finds the shortest path from 
Hughenden to Brisbane

- `places.json` data file, describes 20 cities/towns in Queensland with place names and UTM coordinates
- `routes.json` data file describes routes between these places as a set of tuples (start town, end town, distance, road type) where road type is 1 for a major road, and 2 for a minor road (worse quality, lower average speed)

## Benchmark

- `benchmark.py` times the key operations of the package and prints a table: adding vertices and edges, copying, pickling, finding the components, removing a vertex, finding the closest vertex to a point, forming the adjacency, Laplacian and incidence matrices, and the three path planners (breadth-first, uniform cost and A\*).  The graphs are random geometric graphs in the plane, each vertex joined to its 4 nearest neighbours, of 100, 1,000 and 10,000 vertices, and the same graphs and path queries are used every run.  The last column of the table, the growth exponent, is the exponent *b* in time ∝ N<sup>*b*</sup> between the smallest and largest graph for the quantity shown: 0 is constant, 1 is linear and 2 is quadratic.

  ```
  python benchmark.py             # undirected graphs, about 15 seconds
  python benchmark.py --directed  # directed graphs
  python benchmark.py --quick     # without the largest graph
  python benchmark.py --md        # the table as MarkDown, to paste into a README
  python benchmark.py --rst       # the table as reST, to paste into the documentation
  ```

  Also `--sizes`, `--seed` and `--matrix-max` (the matrices are dense, so they are only formed up to this number of vertices), and `--plot` to time `plot()` as well, which needs a working matplotlib backend.  The table is made with `ansitable`, which `spatialmath-python` installs, and with `--md` or `--rst` only the table goes to standard output, so it can be redirected to a file.  Timings depend on the machine, so compare runs made on the same one.
