# Step 6a: Make a small India state map for the dashboard
#
# Input: DataMeet's state boundaries (CC BY 4.0), downloaded from
#     https://github.com/datameet/maps/tree/master/States  (Admin2.shp and its
#     .dbf, .shx, .prj, .cpg files) into data/raw/datameet/
# The original file is 17 MB, far too big for a web page, so we simplify the
# shapes (fewer points along each border) and save a small GeoJSON file.
#
# Two changes so the map matches our data:
#   - Jammu & Kashmir and Ladakh are joined into one shape, because our data
#     merges them (see DATA_PROBLEMS.md #3)
#   - state names are written the same way as in our clean tables
#
# Run from the Project folder:
#     .venv\Scripts\python src\06_india_map.py
# Needs: pip install pyshp shapely   (only for this script, not for the app)

import json
import shapefile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

RAW = "data/raw/datameet/Admin2.shp"
OUT = "data/india_states.geojson"
TOLERANCE = 0.03   # in degrees (about 3 km); bigger = smaller file, rougher borders


def our_name(name):
    # "Andaman & Nicobar" -> "Andaman And Nicobar Islands", and so on
    name = name.replace("&", "and").title()
    if name == "Andaman And Nicobar":
        name = "Andaman And Nicobar Islands"
    if name == "Jammu And Kashmir" or name == "Ladakh":
        name = "Jammu And Kashmir And Ladakh"
    return name


def round_coordinates(item):
    # 3 decimals (about 100 m) is plenty for a state map and keeps the file small
    if isinstance(item, float):
        return round(item, 3)
    rounded = []
    for part in item:
        rounded.append(round_coordinates(part))
    return rounded


# 1. Read every state and give it our name
reader = shapefile.Reader(RAW)
shapes_by_state = {}
for i in range(len(reader)):
    name = our_name(reader.record(i)[0])
    geometry = shape(reader.shape(i).__geo_interface__)
    if name not in shapes_by_state:
        shapes_by_state[name] = []
    shapes_by_state[name].append(geometry)

# 2. Join shapes with the same name (J&K + Ladakh), then simplify
features = []
for name in sorted(shapes_by_state):
    joined = unary_union(shapes_by_state[name])
    simple = joined.simplify(TOLERANCE, preserve_topology=True)
    geometry = mapping(simple)
    features.append({
        "type": "Feature",
        "id": name,
        "properties": {"state": name},
        "geometry": {"type": geometry["type"], "coordinates": round_coordinates(geometry["coordinates"])},
    })
    print(name.ljust(42), geometry["type"])

with open(OUT, "w", encoding="utf-8") as out_file:
    json.dump({"type": "FeatureCollection", "features": features}, out_file, separators=(",", ":"))
print("\nSaved", OUT, "with", len(features), "states")

# 3. Draw the outline, to check the borders against the official map of India
fig, ax = plt.subplots(figsize=(8, 9))
for feature in features:
    outline = shape(feature["geometry"])
    polygons = [outline]
    if outline.geom_type == "MultiPolygon":
        polygons = list(outline.geoms)
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.fill(x, y, color="#cde2fb", edgecolor="#1c5cab", linewidth=0.5)
ax.set_aspect("equal")
ax.set_title("Simplified state map (check against the official map of India)")
fig.tight_layout()
fig.savefig("outputs/india_map_check.png", dpi=110)
