import json

# load the JSON files
with open('places.json', 'r') as f:
    places = json.loads(f.read())

    # list of dicts

with open('routes.json', 'r') as f:
    json_routes = json.loads(f.read())
    # list of tuples

for place, info in places.items():
    del info['x']
    del info['y']
    del info['dbne']
    del info['utmzone']
    try:
        del info['dgoal']
    except:
        pass
    info['utm'] = [info['utm'][0] / 1000, info['utm'][1] / 1000]

routes = []
for r in json_routes:
    route = {}
    route['start'] = r[0]
    route['end'] = r[1]
    route['distance'] = r[2]
    route['speed'] = 100 if r[3] == 1 else 50
    routes.append(route)

data = {'places': places, 'routes': routes}
with open('queensland.json', 'w', encoding='utf-8') as f:
  f.write(json.dumps(data, ensure_ascii=False))


# %     start        end       routelength  quality (1=hwy,2=minor)
# routes = {
#     {"Camooweal", "Mount Isa", 188, 1},
#     {"Mount Isa","Cloncurry", 118, 1},
#     {"Cloncurry","Hughenden", 401, 1},
#     {"Cloncurry","Winton", 350, 1},
#     {"Hughenden","Charters Towers", 248, 1},
#     {"Charters Towers","Townsville", 135, 1},
#     {"Townsville","Mackay", 388, 1},
#     {"Mackay","Rockhampton", 334, 1},
#     {"Mackay","Emerald", 384, 1},
#     {"Rockhampton","Brisbane", 682, 1},
#     {"Rockhampton","Emerald", 270, 1},
#     {"Brisbane","Roma", 482, 1},
#     {"Roma","Charleville", 266, 1},
#     {"Charleville","Blackall", 305, 1},
#     {"Blackall","Barcaldine",106, 1},
#     {"Blackall","Windorah", 530, 2},
#     {"Barcaldine","Emerald", 307, 1, 1},
#     {"Barcaldine","Hughenden", 500, 2},
#     {"Barcaldine","Longreach", 106, 1},
#     {"Longreach","Windorah", 311, 2},
#     {"Longreach","Winton", 180, 1},
#     {"Winton","Hughenden", 216, 1},
#     {"Winton","Boulia", 363, 2},
#     {"Winton","Windorah", 487, 2},
#     {"Boulia","Mount Isa", 304, 2},
#     {"Boulia","Bedourie", 194 2},
#     {"Bedourie","Birdsville", 193, 2},
#     {"Birdsville","Windorah", 380, 2},
#     {"Bedourie","Windorah", 411, 2}
#     };