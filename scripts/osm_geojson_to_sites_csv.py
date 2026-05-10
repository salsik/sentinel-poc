from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
#INPUT = ROOT / "data" / "osm_sites.geojson"
INPUT = ROOT / "data" / "osm_corridor.geojson"
OUTPUT = ROOT / "data" / "sites_from_osm.csv"

def get_lon_lat(feature):
    geom = feature.get("geometry", {})
    props = feature.get("properties", {})

    # For node features
    if geom.get("type") == "Point":
        lon, lat = geom["coordinates"]
        return lon, lat

    # For way/relation features exported with center
    if "center" in props:
        return props["center"]["lon"], props["center"]["lat"]

    # Fallback: use centroid-like average for polygons/lines
    coords = geom.get("coordinates")
    if not coords:
        return None, None

    flat = []

    def flatten(x):
        if isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
            flat.append(x)
        elif isinstance(x, list):
            for y in x:
                flatten(y)

    flatten(coords)
    if not flat:
        return None, None

    lon = sum(p[0] for p in flat) / len(flat)
    lat = sum(p[1] for p in flat) / len(flat)
    return lon, lat

def map_site_type(amenity):
    if amenity == "school":
        return "school"
    if amenity in ["clinic", "hospital", "doctors"]:
        return "clinic"
    if amenity == "community_centre":
        return "community"
    return "community"

def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    rows = []
    for i, feature in enumerate(geojson.get("features", []), start=1):
        props = feature.get("properties", {})
        lon, lat = get_lon_lat(feature)

        if lon is None or lat is None:
            continue

        amenity = props.get("amenity", "community")
        name = props.get("name") or props.get("name:en") or props.get("name:ar") or f"OSM site {i}"

        rows.append({
            "site_id": f"osm_{i:03d}",
            "site_name": name,
            "site_type": map_site_type(amenity),
            "latitude": lat,
            "longitude": lon,
            "source": "OpenStreetMap",
            "validation_status": "unvalidated_public_osm"
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No usable sites found in GeoJSON. Try a larger AOI or add manual demo points.")

    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} sites to {OUTPUT}")

if __name__ == "__main__":
    main()