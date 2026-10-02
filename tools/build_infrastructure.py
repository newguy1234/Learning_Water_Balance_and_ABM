"""Build the district's infrastructure layer and parcel membership (cyclic/bundled/).

    python tools/build_infrastructure.py "<path to Okura et al. 2025 PDF>" [gsi_cache.json]

Writes
    cyclic/bundled/infrastructure.geojson   drainage channels from OpenStreetMap and GSI channel
                                            centrelines, each with a role (main_canal /
                                            connected_drain / other_drain / river), plus the
                                            pumping station, structures and documented notes
    cyclic/bundled/okura_fig3_canal_depth.csv  daily main drainage canal depth (North / South
                                            ADCP sites) digitised from Okura et al. Fig. 3
    cyclic/bundled/paddy_elevation.csv      each member parcel's median ground elevation (GSI 5 m DEM)
    cyclic/bundled/district_membership.csv  every MAFF Fude paddy (田) parcel in the box, keyed
                                            by polygon_uuid, classed from Okura et al. Fig. 1a
                                            as north_cyclic / south_cyclic / non_cyclic /
                                            outside / unknown (hidden under the figure's legend,
                                            compass or scale bar)

Sources are listed in cyclic/bundled/SOURCES.md. The figure itself is not stored; only the
per-parcel classes read from it and the registration used.

How the figure is read. Fig. 1a (panel a of the first figure) is a north-up map. It is
registered onto the parcels by a similarity transform (scale, small rotation, shift) about
the pumping station, chosen to put the most parcel centroids on drawn parcels. The fitted
scale (~101 px/km) is checked against the figure's own scale bar (253 px = 2.5 km). Each
parcel is then classed from the colours around its centroid (7 x 7 px, ~70 m, for the
cyclic colours; 5 x 5 px for the non-cyclic grey, which otherwise bleeds past the district
edge): gold = North cyclic, magenta = South cyclic, light grey fill = non-cyclic, nothing =
outside.
Black lines (parcel outlines, the zoom box and its connector lines) are ignored.
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BUNDLED = os.path.join(ROOT, "cyclic", "bundled")
DATA = os.path.join(ROOT, "data")

# Hakusan-Jimbee (白山甚兵衛) pumping station: the round discharge tank (OSM way 1082034661,
# D = 26.0 m per MAFF), confirmed on GSI seamless aerial imagery.
STATION = (140.26372, 35.80755)
# The OSM channel along the lake dyke through the station: Okura Fig. 1b's main drainage canal.
MAIN_CANAL_WAYS = (60622667,)
BOX = (35.740, 140.215, 35.880, 140.310)          # south, west, north, east
# Channel ends this close are joined. Drains pass under farm roads through short culverts
# the maps leave out (road + verges < 25 m); connectivity in the cyclic area rises from
# 29 km at 15 m to 45 km at 25 m and then levels off (46 km at 50 m).
SNAP_M = 25.0

OVERPASS = f"""[out:json][timeout:150];
(
  way["waterway"~"^(river|stream|ditch|drain|canal)$"]({BOX[0]},{BOX[1]},{BOX[2]},{BOX[3]});
  way["natural"="water"]["name"~"印旛沼"]({BOX[0]},{BOX[1]},{BOX[2]},{BOX[3]});
);
out tags geom;"""


OVERPASS_ENDPOINTS = ("https://overpass-api.de/api/interpreter",
                      "https://overpass.kumi.systems/api/interpreter",
                      "https://overpass.private.coffee/api/interpreter")


def overpass(query, attempts=2):
    """Run an Overpass query, trying the public mirrors in turn when one is busy."""
    import time

    error = None
    for attempt in range(attempts):
        for url in OVERPASS_ENDPOINTS:
            request = urllib.request.Request(
                url, data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
                headers={"User-Agent": "LearningWaterBalance-research/1.0", "Accept": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=200) as response:
                    return json.loads(response.read())
            except Exception as exc:  # noqa: BLE001  (busy mirror: try the next)
                error = exc
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"all Overpass mirrors failed: {error}")


def metres(lon, lat):
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    return ((lon - STATION[0]) * 111_320.0 * math.cos(math.radians(STATION[1])),
            (lat - STATION[1]) * 110_540.0)


def lonlat(x, y):
    """Inverse of ``metres``."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return (STATION[0] + x / (111_320.0 * math.cos(math.radians(STATION[1]))),
            STATION[1] + y / 110_540.0)


# --- GSI centrelines ---------------------------------------------------------------------------

GSI_TILE = "https://cyberjapandata.gsi.go.jp/xyz/{layer}/16/{x}/{y}.geojson"
# Box around the district (south, west, north, east) for the GSI tiles.
GSI_BOX = (35.755, 140.228, 35.872, 140.300)


def gsi_lines(layer, cache=None):
    """Every feature of a GSI z16 vector layer over the district box: ``experimental_rvrcl``
    (river / channel centrelines, incl. narrow channels 細河川) or ``experimental_rdcl``
    (road centrelines). ``cache``: a JSON file of {layer: features} from an earlier fetch."""
    from concurrent.futures import ThreadPoolExecutor

    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as src:
            stored = json.load(src)
        if layer in stored:
            return stored[layer]

    def tile(lat, lon):
        n = 2 ** 16
        return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)

    x0, y1 = tile(GSI_BOX[0], GSI_BOX[1])
    x1, y0 = tile(GSI_BOX[2], GSI_BOX[3])

    def fetch(xy):
        url = GSI_TILE.format(layer=layer, x=xy[0], y=xy[1])
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LearningWaterBalance-research/1.0"})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())["features"]
        except Exception:  # noqa: BLE001  (no tile = nothing there)
            return []

    feats, seen = [], set()
    with ThreadPoolExecutor(8) as pool:
        for tile_feats in pool.map(fetch, [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]):
            for f in tile_feats:
                key = (f["properties"].get("rID"), json.dumps(f["geometry"]["coordinates"])[:80])
                if key not in seen:
                    seen.add(key)
                    feats.append(f)
    return feats


def _parts(feature):
    g = feature["geometry"]
    return [g["coordinates"]] if g["type"] == "LineString" else list(g["coordinates"])


def cyclic_footprint(buffer_m=250.0):
    """Area around the cyclic parcels (Okura Fig. 1a classes), in local metres. Only
    channels here can belong to the pool's network: non-cyclic blocks drain out of the
    scheme (Okura Fig. 2b)."""
    from shapely.geometry import MultiPoint

    member = pd.read_csv(os.path.join(BUNDLED, "district_membership.csv"), comment="#")
    cyc = member[member["fig1_class"].isin(["north_cyclic", "south_cyclic"])]
    x, y = metres(cyc["lon"], cyc["lat"])
    return MultiPoint(np.column_stack([x, y])).buffer(buffer_m)


# --- network ---------------------------------------------------------------------------------

def build_network(gsi_cache=None):
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    from shapely.geometry import Polygon

    raw = overpass(OVERPASS)
    ways, lakes = [], []
    for e in raw["elements"]:
        tags = e.get("tags", {})
        if e["type"] != "way" or not e.get("geometry"):
            continue
        if tags.get("natural") == "water":
            ring = [(p["lon"], p["lat"]) for p in e["geometry"]]
            if len(ring) >= 4:
                lakes.append(Polygon(np.column_stack(metres(*zip(*ring)))).buffer(0))
            continue
        coords = [(p["lon"], p["lat"]) for p in e["geometry"]]
        if len(coords) < 2:
            continue
        ways.append(dict(osm_id=int(e["id"]), source="osm", kind=tags["waterway"], name=tags.get("name", ""),
                         coords=coords, culvert=tags.get("tunnel") == "culvert"))
    # GSI channel centrelines (国土基本情報): many farm and collector drains OSM lacks.
    for f in gsi_lines("experimental_rvrcl", cache=gsi_cache):
        p = f["properties"]
        kind = ("river" if p.get("rivCtg") == "一級河川" else
                "narrow_channel" if str(p.get("type", "")).startswith("細河川") else "channel")
        for part in _parts(f):
            if len(part) >= 2:
                ways.append(dict(osm_id=None, gsi_id=p.get("rID"), source="gsi", kind=kind,
                                 name=p.get("name", "") or "", coords=[tuple(c) for c in part],
                                 culvert="地下" in str(p.get("type", ""))))
    # Connectivity through drainage channels only: natural rivers meet the drains at
    # gates and culverts, not as open joins, and would chain unrelated systems together.
    lines = [LineString(np.column_stack(metres(*zip(*w["coords"])))) for w in ways]
    footprint = cyclic_footprint()
    # Lines drawn across the open lake (flow lines, not drains) cannot carry farm drainage.
    lake = gpd.GeoSeries(lakes).union_all() if lakes else None
    in_lake = lambda line: lake is not None and line.intersection(lake).length > 0.5 * line.length
    drainage = [i for i, w in enumerate(ways)
                if w["kind"] != "river" and lines[i].intersects(footprint) and not in_lake(lines[i])]
    index = gpd.GeoSeries([lines[i] for i in drainage]).sindex
    adjacent = {i: set() for i in drainage}
    for a, i in enumerate(drainage):
        for b in index.query(lines[i].buffer(SNAP_M)):
            j = drainage[int(b)]
            if j != i and lines[i].distance(lines[j]) <= SNAP_M:
                adjacent[i].add(j)
    seeds = [i for i in drainage if ways[i]["osm_id"] in MAIN_CANAL_WAYS]
    connected, stack = set(seeds), list(seeds)
    while stack:
        i = stack.pop()
        for j in adjacent[i] - connected:
            connected.add(j)
            stack.append(j)
    features = []

    def add(w, coords, role, length_m):
        props = dict(source=w["source"], osm_id=w["osm_id"], waterway=w["kind"], name=w["name"], role=role,
                     culvert=bool(w.get("culvert")), length_m=round(length_m, 1))
        if w["source"] == "gsi":
            props["gsi_id"] = w.get("gsi_id")
        features.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=[list(c) for c in coords]),
                             properties=props))

    for i, w in enumerate(ways):
        if w["osm_id"] in MAIN_CANAL_WAYS:
            add(w, w["coords"], "main_canal", lines[i].length)
        elif i in connected:
            # The pool's network stops at the cyclic area: beyond it channels run on to rivers
            # and other outlets, and for now drainage leaves the district only at the station.
            for part_line, role in ((lines[i].intersection(footprint), "connected_drain"),
                                    (lines[i].difference(footprint), "other_drain")):
                for piece in getattr(part_line, "geoms", [part_line]):
                    if piece.is_empty or piece.geom_type != "LineString" or piece.length < 1.0:
                        continue
                    lon, lat = lonlat(*np.asarray(piece.coords).T)
                    add(w, list(zip(np.round(lon, 7), np.round(lat, 7))), role, piece.length)
        else:
            add(w, w["coords"], "river" if w["kind"] == "river" else "other_drain", lines[i].length)
    features.append(dict(type="Feature", geometry=dict(type="Point", coordinates=list(STATION)),
                         properties=dict(role="station", name="白山甚兵衛機場 (Hakusan-Jimbee pumping station)",
                                         osm_id=1082034661, max_supply_m3_s=3.304, service_area_ha=983,
                                         in_service="April 2015 (FY2015)")))
    collection = dict(
        type="FeatureCollection",
        attribution="Channels: © OpenStreetMap contributors (ODbL), via the Overpass API; GSI river / channel "
                    "centrelines (国土地理院 ベクトルタイル 河川中心線, experimental_rvrcl). "
                    "Station: MAFF Kanto Regional Agricultural Administration Office; located on OSM + GSI aerial imagery.",
        station=list(STATION), main_canal_osm_ways=list(MAIN_CANAL_WAYS), snap_m=SNAP_M,
        pool_network_rule="joined (within snap_m) to the main canal through drainage channels that lie "
                          "within 250 m of cyclic parcels (Okura Fig. 2b: non-cyclic blocks drain out of the scheme) "
                          "and not mostly across the lake (OSM 印旛沼 water area); connected channels are clipped "
                          "to that area, so drainage leaves only at the station",
        features=features)
    with open(os.path.join(BUNDLED, "infrastructure.geojson"), "w", encoding="utf-8") as out:
        json.dump(collection, out, ensure_ascii=False)
    lengths = pd.DataFrame([f["properties"] for f in features if f["geometry"]["type"] == "LineString"])
    print(lengths.groupby("role").length_m.agg(["count", "sum"]).assign(km=lambda d: d["sum"] / 1000).round(1))


STRUCTURES = f"""[out:json][timeout:150];
(
  way["waterway"]["tunnel"="culvert"]({BOX[0]},{BOX[1]},{BOX[2]},{BOX[3]});
  nwr["waterway"~"^(sluice_gate|weir|lock_gate|dam)$"]({BOX[0]},{BOX[1]},{BOX[2]},{BOX[3]});
  nwr["man_made"~"^(pipeline|pumping_station|water_works)$"]({BOX[0]},{BOX[1]},{BOX[2]},{BOX[3]});
);
out tags geom;"""

# What is known about the district's pipes and underground collection but has no public
# route or geometry. Kept with the network so maps and code can cite it.
PIPELINES = dict(
    note="Pressurised irrigation pipelines from the station to a hydrant (給水栓) at each paddy; "
         "routes are not public.",
    pumps="3 × φ700 mm (白山甚兵衛揚水機場); max supply 3.304 m3/s",
    hakusan_main_m=940, hakusan_branch_m=2940,              # 白山幹線 / 支線用水路 (North area)
    jimbee_main_m=1480 + 300, jimbee_main_diameter_mm=900,  # 甚兵衛幹線 (300 m φ900 laid 2011)
    jimbee_branch_m=2390,                                   # 甚兵衛支線用水路 (South area)
    sources=["日本工業経済新聞 2012-08-09, https://www.nikoukei.co.jp/news/detail/214935",
             "MAFF Kanto, Inbanuma Phase 2 main facilities (capacity)"])
LATERAL_COLLECTION = dict(
    farm_drains="Farm drains have bare-soil beds: surface drainage seeps into the lateral layer and "
                "lateral flow enters the drains; both reach the pool (Okura et al. 2025 §4.2).",
    pool="The pool is lined with concrete piles; lateral flow and groundwater enter through the "
         "seams, and small drainage pipes in the piles relieve pressure (Okura et al. 2025 §4.2).",
    subsurface_drains="No public record of pipe underdrains (暗渠排水) in this district was found.",
    model="cyclic.physics: Hooghoudt lateral drain flow with 10 m drain spacing and 0.8 m "
          "equivalent depth ('dense drain network, hence short L'); a per-paddy share of it is "
          "recaptured by the pool, set by distance to the nearest channel reaching the pool.")


def add_structures():
    """Mark culverts on the channels and add structures, other pipelines and the documented
    pipe / lateral-collection notes to infrastructure.geojson, without changing the
    channel network (so everything derived from it stays the same)."""
    path = os.path.join(BUNDLED, "infrastructure.geojson")
    with open(path, encoding="utf-8") as src:
        collection = json.load(src)
    raw = overpass(STRUCTURES)
    culverts = {int(e["id"]) for e in raw["elements"]
                if e["type"] == "way" and e.get("tags", {}).get("tunnel") == "culvert"}
    kept = []
    for f in collection["features"]:
        p = f["properties"]
        if p.get("role") in ("structure", "other_pipeline"):
            continue                                    # rebuilt below
        if f["geometry"]["type"] == "LineString" and p.get("source", "osm") == "osm":
            p["culvert"] = p.get("osm_id") in culverts     # GSI marks its own (人工水路（地下）)
        kept.append(f)
    for e in raw["elements"]:
        tags = e.get("tags", {})
        kind = tags.get("man_made") or tags.get("waterway")
        if tags.get("tunnel") == "culvert" or kind is None:
            continue
        geom = e.get("geometry")
        if kind == "pipeline" and geom:
            kept.append(dict(type="Feature",
                             geometry=dict(type="LineString", coordinates=[[p["lon"], p["lat"]] for p in geom]),
                             properties=dict(role="other_pipeline", osm_id=int(e["id"]), kind=kind,
                                             name=tags.get("name", ""), substance=tags.get("substance", ""),
                                             note="mapped pipeline; not the district's irrigation pipes")))
            continue
        if geom:
            lon = float(np.mean([p["lon"] for p in geom]))
            lat = float(np.mean([p["lat"] for p in geom]))
        else:
            lon, lat = float(e["lon"]), float(e["lat"])
        kept.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[lon, lat]),
                         properties=dict(role="structure", osm_id=int(e["id"]), kind=kind,
                                         name=tags.get("name", ""), operator=tags.get("operator", ""))))
    collection["features"] = kept
    collection["irrigation_pipelines"] = PIPELINES
    collection["lateral_collection"] = LATERAL_COLLECTION
    with open(path, "w", encoding="utf-8") as out:
        json.dump(collection, out, ensure_ascii=False)
    n = lambda role: sum(f["properties"].get("role") == role for f in kept)
    print(f"culverts marked {sum(bool(f['properties'].get('culvert')) for f in kept)}, "
          f"structures {n('structure')}, other pipelines {n('other_pipeline')}")


# --- Fig. 1a membership ------------------------------------------------------------------------

def figure_image(pdf_path):
    import io

    import pypdf
    from PIL import Image

    page = pypdf.PdfReader(pdf_path).pages[2]           # Fig. 1 is on page 3
    image = max(page.images, key=lambda im: len(im.data))
    return np.asarray(Image.open(io.BytesIO(image.data)).convert("RGB")).astype(int)


def pixel_classes(im):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    white = (R > 240) & (G > 240) & (B > 240)
    lake = (B > 235) & (G > 225) & (R < 225)
    gold = (R > 150) & (G > 100) & (B < 150) & (R - B > 60)
    magenta = (R > 160) & (G < 140) & (B > 120)
    grey = (abs(R - G) < 25) & (abs(G - B) < 25) & ~white
    light_grey = grey & (R > 120) & (R < 236)
    cls = np.zeros(R.shape, dtype=np.int8)               # 0 nothing
    cls[light_grey] = 3
    cls[gold] = 1
    cls[magenta] = 2
    cls[lake] = 4
    return cls


# Parts of the map drawn over by the figure's own furniture (x0, x1, y0, y1) in pixels. The
# legend sits beyond the north block's drawn north-east edge (white margin between them), so
# parcels under it are outside; under the compass or scale bar the figure cannot say.
LEGEND = (903, 1190, 0, 243)
HIDDEN = ((648, 792, 893, 1036), (686, 992, 1026, 1090))
# Panel (a) map area: right of the Japan inset's border (x ~ 588), above the credit line.
MAP = (596, 1186, 8, 1088)


def register(cls, xm, ym):
    inside = np.zeros(cls.shape, dtype=bool)
    inside[MAP[2]:MAP[3], MAP[0]:MAP[1]] = np.isin(cls[MAP[2]:MAP[3], MAP[0]:MAP[1]], (1, 2, 3))
    for x0, x1, y0, y1 in (LEGEND, *HIDDEN):
        inside[y0:y1, x0:x1] = False
    H, W = cls.shape

    def score(p):
        u, v = project(p, xm, ym)
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = (ui >= 1) & (ui < W - 1) & (vi >= 1) & (vi < H - 1)
        hit = np.zeros(len(u), dtype=bool)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                hit[ok] |= inside[vi[ok] + dv, ui[ok] + du]
        return hit.mean()

    best = (0.0, None)
    for s in np.linspace(0.088, 0.114, 14):
        for th in np.radians([-4, -2, 0, 2, 4]):
            for tx in range(880, 960, 6):
                for ty in range(590, 680, 6):
                    sc = score((s, th, tx, ty))
                    if sc > best[0]:
                        best = (sc, (s, th, tx, ty))
    p, steps = list(best[1]), [0.001, math.radians(0.5), 1.5, 1.5]
    for _ in range(8):
        for k in range(4):
            for sign in (-1, 1):
                q = list(p)
                q[k] += sign * steps[k]
                sc = score(q)
                if sc > best[0]:
                    best, p = (sc, q), q
        steps = [x * 0.6 for x in steps]
    return best


def project(p, xm, ym):
    s, th, tx, ty = p
    c, sn = math.cos(th), math.sin(th)
    return tx + s * (c * xm - sn * ym), ty - s * (sn * xm + c * ym)


def classify(cls, u, v):
    H, W = cls.shape
    labels = []
    for a, b in zip(np.round(u).astype(int), np.round(v).astype(int)):
        if LEGEND[0] <= a < LEGEND[1] and b < LEGEND[3]:
            labels.append("outside")
            continue
        if any(x0 <= a < x1 and y0 <= b < y1 for x0, x1, y0, y1 in HIDDEN):
            labels.append("unknown")
            continue
        if not (MAP[0] + 3 <= a < MAP[1] - 3 and MAP[2] + 3 <= b < MAP[3] - 3):
            labels.append("outside")
            continue
        win = cls[b - 3:b + 4, a - 3:a + 4].ravel()
        gold, magenta = int((win == 1).sum()), int((win == 2).sum())
        # Non-cyclic fill is read from a tighter 5 x 5 window: light grey bleeds ~30 m past
        # the district edge in a 7 x 7 one.
        light = int((cls[b - 2:b + 3, a - 2:a + 3] == 3).sum())
        if gold + magenta >= 5:
            labels.append("north_cyclic" if gold >= magenta else "south_cyclic")
        elif light >= 8:
            labels.append("non_cyclic")
        else:
            labels.append("outside")
    return labels


def build_membership(pdf_path):
    import geopandas as gpd

    parcels = gpd.read_file(os.path.join(DATA, "fude_parcels.gpkg"))
    paddy = parcels[parcels["land_type"] == "田"].copy()
    centres = paddy.to_crs("EPSG:6677").geometry.centroid.to_crs("EPSG:4326")
    lon, lat = centres.x.to_numpy(), centres.y.to_numpy()
    xm, ym = metres(lon, lat)
    cls = pixel_classes(figure_image(pdf_path))
    # Register on the parcels the previous outline kept (all were meant to be the district).
    outline = gpd.read_file(os.path.join(DATA, "district_outline.geojson")).to_crs(4326).union_all()
    within = centres.within(outline).to_numpy()
    hit, params = register(cls, xm[within], ym[within])
    s, th, tx, ty = params
    print(f"registration: {100 * hit:.1f}% of outline parcels on drawn parcels; "
          f"{1000 * s:.1f} px/km (scale bar: 101.2), rotation {math.degrees(th):.2f} deg")
    u, v = project(params, xm, ym)
    frame = pd.DataFrame(dict(polygon_uuid=paddy["polygon_uuid"].to_numpy(), lon=lon, lat=lat,
                              fig_u=np.round(u, 1), fig_v=np.round(v, 1),
                              fig1_class=classify(cls, u, v), in_old_outline=within.astype(int)))
    area = paddy.to_crs("EPSG:6677").area.to_numpy() / 1e4
    frame["area_ha"] = np.round(area, 4)
    header = (f"# Okura et al. (2025) Fig. 1a registered by similarity transform about the station "
              f"{STATION}: {1000 * s:.2f} px/km, rotation {math.degrees(th):.3f} deg, station at px "
              f"({tx:.1f}, {ty:.1f}); {100 * hit:.1f}% of old-outline parcels land on drawn parcels.\n")
    path = os.path.join(BUNDLED, "district_membership.csv")
    with open(path, "w", encoding="utf-8", newline="") as out:
        out.write(header)
        frame.to_csv(out, index=False)
    print(frame.groupby("fig1_class").area_ha.agg(["count", "sum"]).round(1))
    print(pd.crosstab(frame.fig1_class, frame.in_old_outline, values=frame.area_ha, aggfunc="sum").round(1))


# --- ground elevation (GSI 5 m DEM) -----------------------------------------------------------

DEM_URL = "https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/15/{x}/{y}.png"   # GSI 基盤地図 DEM5A


def dem_sampler(zoom=15):
    """Elevation (m, T.P.) at lon/lat from GSI's 5 m DEM PNG tiles; NaN over water voids."""
    import io

    from PIL import Image

    tiles = {}

    def tile(tx, ty):
        if (tx, ty) not in tiles:
            req = urllib.request.Request(DEM_URL.format(x=tx, y=ty),
                                         headers={"User-Agent": "LearningWaterBalance-research/1.0"})
            try:
                raw = urllib.request.urlopen(req, timeout=30).read()
                rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).astype(np.int64)
                v = rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]
                tiles[(tx, ty)] = np.where(v < 2 ** 23, v * 0.01, np.where(v > 2 ** 23, (v - 2 ** 24) * 0.01, np.nan))
            except Exception:  # noqa: BLE001  (no tile = no data)
                tiles[(tx, ty)] = np.full((256, 256), np.nan)
        return tiles[(tx, ty)]

    def sample(lon, lat):
        n = 2 ** zoom
        x = (np.asarray(lon) + 180) / 360 * n
        y = (1 - np.arcsinh(np.tan(np.radians(np.asarray(lat)))) / np.pi) / 2 * n
        out = np.empty(x.shape)
        for k, (a, b) in enumerate(zip(x.ravel(), y.ravel())):
            out.ravel()[k] = tile(int(a), int(b))[int((b - int(b)) * 256), int((a - int(a)) * 256)]
        return out

    return sample


def build_elevation():
    """Median ground elevation of each member parcel (5 m DEM points inside the parcel)."""
    import geopandas as gpd
    import shapely

    sys.path.insert(0, ROOT)
    from cyclic import data

    frame = data.paddies()
    sample = dem_sampler()
    metric = frame.to_crs("EPSG:6677").geometry
    back = gpd.GeoSeries(crs="EPSG:6677")
    elev = np.full(len(frame), np.nan)
    for k, geom in enumerate(metric.values):
        minx, miny, maxx, maxy = geom.bounds
        gx, gy = np.meshgrid(np.arange(minx + 2.5, maxx, 5.0), np.arange(miny + 2.5, maxy, 5.0))
        inside = shapely.contains_xy(geom, gx, gy)
        px, py = (gx[inside], gy[inside]) if inside.any() else (np.array([geom.centroid.x]), np.array([geom.centroid.y]))
        lonlat = gpd.GeoSeries(gpd.points_from_xy(px, py), crs="EPSG:6677").to_crs(4326)
        values = sample(lonlat.x.to_numpy(), lonlat.y.to_numpy())
        if np.isfinite(values).any():
            elev[k] = float(np.nanmedian(values))
    missing = ~np.isfinite(elev)
    if missing.any():   # a parcel wholly in a DEM void takes its nearest neighbours' median
        from scipy.spatial import cKDTree
        xy = np.column_stack(metres(frame["lon"], frame["lat"]))
        tree = cKDTree(xy[~missing])
        _, near = tree.query(xy[missing], k=5)
        elev[missing] = np.nanmedian(elev[~missing][near], axis=1)
    out = pd.DataFrame(dict(polygon_uuid=frame["polygon_uuid"], ground_elev_m=np.round(elev, 3)))
    path = os.path.join(BUNDLED, "paddy_elevation.csv")
    with open(path, "w", encoding="utf-8", newline="") as dst:
        dst.write("# Median GSI 5 m DEM (dem5a, m above T.P.) over 5 m points inside each parcel; "
                  f"{int(missing.sum())} parcel(s) in DEM voids filled from 5 nearest neighbours.\n")
        out.to_csv(dst, index=False)
    print(out.ground_elev_m.describe().round(2))


# --- Okura Fig. 3: main drainage canal water depth ---------------------------------------------

# Axis calibration of Fig. 3 (page 5): pixel rows of the 0 m and 3 m gridlines and the x of
# 4/17 and 8/21 (= day 126) tick labels, for panels (a) 2021 and (b) 2022.
FIG3 = {2021: dict(y0=672, y3=110, x0=152, x126=1672, ylo=100, yhi=690),
        2022: dict(y0=1515, y3=955, x0=137, x126=1680, ylo=945, yhi=1525)}


def build_canal_depth(pdf_path):
    """Daily water depth of the main drainage canal at the North and South ADCP sites."""
    import io

    import pypdf
    from PIL import Image

    page = pypdf.PdfReader(pdf_path).pages[4]
    image = max(page.images, key=lambda im: len(im.data))
    im = np.asarray(Image.open(io.BytesIO(image.data)).convert("RGB")).astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    pink = (R > 190) & (G < 90) & (B > 80) & (B < 190)     # "The North block side"
    green = (G > 140) & (R < 120) & (B < 120)              # "The South block side"
    rows = []
    for year, p in FIG3.items():
        per_day = (p["x126"] - p["x0"]) / 126.0

        def trace(mask):
            out = np.full(131, np.nan)
            for d in range(131):
                x = int(round(p["x0"] + d * per_day))
                hit = np.nonzero(mask[p["ylo"]:p["yhi"], max(x - 2, 0):x + 3].any(axis=1))[0]
                if hit.size:
                    out[d] = 3.0 * (p["y0"] - (p["ylo"] + float(np.median(hit)))) / (p["y0"] - p["y3"])
            return out

        south = trace(green)
        north = trace(pink)
        north = np.where(np.isfinite(north), north, south)  # pink hidden under green: equal depth
        idx = np.arange(131)
        north, south = (np.interp(idx, idx[np.isfinite(s)], s[np.isfinite(s)]) for s in (north, south))
        for d, date in enumerate(pd.date_range(f"{year}-04-17", periods=131, freq="D")):
            rows.append(dict(date=str(date.date()), year=year, north_depth_m=round(float(north[d]), 3),
                             south_depth_m=round(float(south[d]), 3)))
    frame = pd.DataFrame(rows)
    path = os.path.join(BUNDLED, "okura_fig3_canal_depth.csv")
    with open(path, "w", encoding="utf-8", newline="") as dst:
        dst.write("# Water depth (m) of the main drainage canal at the North / South ADCP sites, digitised "
                  "from Okura et al. (2025) Fig. 3 (colour-traced; the last 3 days extend past the axis).\n")
        frame.to_csv(dst, index=False)
    print(frame.groupby("year")[["north_depth_m", "south_depth_m"]].mean().round(3))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    gsi_cache = sys.argv[2] if len(sys.argv) > 2 else None   # optional JSON of fetched GSI tiles
    build_membership(sys.argv[1])        # the channel network's footprint uses the membership
    build_network(gsi_cache)
    add_structures()
    build_canal_depth(sys.argv[1])
    # Elevation needs the rebuilt paddies: run cyclic.data.paddies(refresh=True) first.
    build_elevation()
