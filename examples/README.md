# Examples

This folder contains two JSON files that describe towns and major roads in Queensland Australia.

- `places.json` describes place names, UTM coordinates
- `routes.json` describes routes between towns as a set of tuples (start town, end town, distance, road type) where road type is 1 for a major road, and 2 for a minor road (worse quality, lower average speed)

`roads.py` loads these files, creates an embedded graph, and find an 