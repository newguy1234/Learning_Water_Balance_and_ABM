"""The district's physical infrastructure: pumping station, pool, drainage channels, and
which parcels belong to the Hakusan-Jimbee (白山甚兵衛) irrigation district.

Files (built by tools/build_infrastructure.py; sources in stem/bundled/SOURCES.md):
    bundled/infrastructure.geojson   OSM drainage channels with a role, plus the station
    bundled/district_membership.csv  every Fude paddy parcel classed from Okura Fig. 1a

Roles of the channels
    main_canal       the main drainage canal along the lake dyke, through the station
                     (Okura Fig. 1b / 2a); it collects the cyclic blocks' drainage
    connected_drain  farm and branch drains that reach the main canal
    other_drain      drains of other systems (they do not reach this station's pool)
    river            natural rivers, for context

How water gets to the pool (``drainage_routes``): a paddy's surface drainage runs to the
nearest connected channel (``entry_m``, through farm drains OSM does not map), then along
the channel network to the station (``path_m``, shortest path). Non-cyclic paddies drain
out of the scheme (Okura Fig. 2b), so their route is informational only.

Also here: each parcel's ground elevation (``elevation``, GSI 5 m DEM) and the main
drainage canal's daily water depth at the North / South sites (``canal_depth``, Okura
Fig. 3), which together decide when a high canal backs drainage up onto the fields.
"""

from __future__ import annotations

import functools
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED = os.path.join(HERE, "bundled")
NETWORK_FILE = os.path.join(BUNDLED, "infrastructure.geojson")
MEMBERSHIP_FILE = os.path.join(BUNDLED, "district_membership.csv")

# The station's round discharge tank (OSM way 1082034661; MAFF: D = 26.0 m, H = 26.4 m).
STATION_LONLAT = (140.26372, 35.80755)
# Okura Fig. 2b: the 18 x 68 m pool adjoins the station between the two canal arms; at
# district scale it sits at the station.
POOL_LONLAT = STATION_LONLAT
MEMBER_CLASSES = ("north_cyclic", "south_cyclic", "non_cyclic")
CYCLIC_CLASSES = ("north_cyclic", "south_cyclic")
DENSIFY_M = 20.0   # channel vertices at least this close, so nearest vertex ≈ nearest point


def metres(lon, lat, origin=STATION_LONLAT):
    """Local east/north metres about ``origin`` (the district spans < 15 km)."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    return ((lon - origin[0]) * 111_320.0 * math.cos(math.radians(origin[1])),
            (lat - origin[1]) * 110_540.0)


def available() -> bool:
    return os.path.exists(NETWORK_FILE) and os.path.exists(MEMBERSHIP_FILE)


@functools.lru_cache(maxsize=1)
def network() -> dict:
    """The bundled GeoJSON FeatureCollection (channels + station)."""
    with open(NETWORK_FILE, encoding="utf-8") as src:
        return json.load(src)


def channels(roles=None) -> list[dict]:
    """Channel features, optionally only these roles."""
    roles = None if roles is None else set([roles] if isinstance(roles, str) else roles)
    return [f for f in network()["features"]
            if f["geometry"]["type"] == "LineString"
            and (roles is None or f["properties"]["role"] in roles)]


def main_canal_vertices() -> list[tuple[float, float]]:
    """The main drainage canal as one (lon, lat) polyline, north end first."""
    coords = [tuple(c) for f in channels("main_canal") for c in f["geometry"]["coordinates"]]
    if coords and coords[0][1] < coords[-1][1]:
        coords = coords[::-1]
    return coords


@functools.lru_cache(maxsize=1)
def membership():
    """Okura Fig. 1a class per Fude parcel (pandas DataFrame indexed by polygon_uuid)."""
    import pandas as pd

    return pd.read_csv(MEMBERSHIP_FILE, comment="#").set_index("polygon_uuid")


def classify(polygon_uuid, lat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(keep, block, cyclic) for parcels. Cyclic parcels take their block from the figure's
    colour; non-cyclic ones are North or South of the station (Okura Fig. 2a)."""
    table = membership()
    cls = table["fig1_class"].reindex(np.asarray(polygon_uuid)).fillna("outside").to_numpy()
    keep = np.isin(cls, MEMBER_CLASSES)
    north = np.where(cls == "north_cyclic", True,
                     np.where(cls == "south_cyclic", False, np.asarray(lat) > STATION_LONLAT[1]))
    block = np.where(north, "North", "South")
    return keep, block, np.isin(cls, CYCLIC_CLASSES)


@functools.lru_cache(maxsize=1)
def _graph():
    """Connected channels as a vertex graph, with shortest path length to the station."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    snap = float(network().get("snap_m", 15.0))
    features = channels(("main_canal", "connected_drain"))
    xs, ys, rows, cols, weights, way_of = [], [], [], [], [], []
    for w, feature in enumerate(features):
        lon, lat = np.asarray(feature["geometry"]["coordinates"], dtype=float).T
        x, y = metres(lon, lat)
        # Densify so vertices sit every <= DENSIFY_M along the channel.
        px, py = [x[0]], [y[0]]
        for a in range(1, len(x)):
            steps = max(1, int(math.ceil(math.hypot(x[a] - x[a - 1], y[a] - y[a - 1]) / DENSIFY_M)))
            for k in range(1, steps + 1):
                px.append(x[a - 1] + (x[a] - x[a - 1]) * k / steps)
                py.append(y[a - 1] + (y[a] - y[a - 1]) * k / steps)
        start = len(xs)
        xs += px
        ys += py
        way_of += [w] * len(px)
        for k in range(len(px) - 1):
            rows.append(start + k)
            cols.append(start + k + 1)
            weights.append(math.hypot(px[k + 1] - px[k], py[k + 1] - py[k]))
    pts = np.column_stack([xs, ys])
    tree = cKDTree(pts)
    way_of = np.asarray(way_of)
    # Junctions: vertices of different channels within the snap distance.
    for i, j in tree.query_pairs(snap):
        if way_of[i] != way_of[j]:
            rows.append(i)
            cols.append(j)
            weights.append(max(float(np.hypot(*(pts[i] - pts[j]))), 1e-3))
    n = len(pts)
    matrix = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    station = int(tree.query([0.0, 0.0])[1])
    to_station = dijkstra(matrix, directed=False, indices=station)
    reachable = np.isfinite(to_station)
    return pts[reachable], to_station[reachable]


def drainage_routes(lon, lat) -> dict:
    """Per point: ``entry_m`` to the nearest connected channel, ``path_m`` along the
    channels from there to the station, ``total_m`` = entry + path."""
    from scipy.spatial import cKDTree

    pts, along = _graph()
    x, y = metres(lon, lat)
    entry, nearest = cKDTree(pts).query(np.column_stack([x, y]))
    path = along[nearest]
    return {"entry_m": entry, "path_m": path, "total_m": entry + path}


def pool_network_vertices() -> list[list[tuple[float, float]]]:
    """Every channel that reaches the pool (main canal + connected drains) as polylines."""
    return [[tuple(c) for c in f["geometry"]["coordinates"]]
            for f in channels(("main_canal", "connected_drain"))]


# --- ground elevation and the main drainage canal's water level ------------------------------

ELEVATION_FILE = os.path.join(BUNDLED, "paddy_elevation.csv")
CANAL_DEPTH_FILE = os.path.join(BUNDLED, "okura_fig3_canal_depth.csv")


@functools.lru_cache(maxsize=1)
def elevation():
    """Median ground elevation (m above T.P., GSI 5 m DEM) per parcel, by polygon_uuid."""
    import pandas as pd

    if not os.path.exists(ELEVATION_FILE):
        return pd.Series(dtype=float)
    return pd.read_csv(ELEVATION_FILE, comment="#").set_index("polygon_uuid")["ground_elev_m"]


@functools.lru_cache(maxsize=4)
def canal_depth(year: int):
    """Daily main drainage canal water depth (m) at the North and South ADCP sites, from
    Okura Fig. 3, as {"North": array, "South": array} over the 131-day season; None when
    the year was not observed."""
    import pandas as pd

    if not os.path.exists(CANAL_DEPTH_FILE):
        return None
    frame = pd.read_csv(CANAL_DEPTH_FILE, comment="#")
    rows = frame[frame["year"] == int(year)]
    if rows.empty:
        return None
    return {"North": rows["north_depth_m"].to_numpy(float), "South": rows["south_depth_m"].to_numpy(float)}


def normal_canal_depth_m() -> float:
    """The canal's normal (no-backwater) depth: the 2022 median at both sites. Okura §4.1:
    2022 was dry and the canal ran at its usual level; wet 2021 held it high (backwater)."""
    series = canal_depth(2022)
    return float(np.median(np.concatenate([series["North"], series["South"]]))) if series else 0.80


# --- map key ---------------------------------------------------------------------------------

# What each drawn thing is, for every map of the district (label, matplotlib style).
KEY = {
    "main_canal": ("Main drainage canal (collects cyclic drainage, feeds the pool)",
                   dict(color="#1f4e9c", lw=2.8)),
    "connected_drain": ("Drains reaching the pool (surface drainage + lateral flow)",
                        dict(color="#3a7bd5", lw=1.3)),
    "culvert": ("Culvert: that drain runs underground", dict(color="#3a7bd5", lw=1.6, ls=(0, (1, 1.5)))),
    "other_drain": ("Drains of other systems (do not reach this pool)", dict(color="#9a9486", lw=0.6)),
    "river": ("River", dict(color="#8ec5e8", lw=1.0)),
    "irrigation_pipeline": ("Irrigation pipelines to each paddy's hydrant (routes not public)",
                            dict(color="#d9822b", lw=1.6, ls=(0, (4, 2)))),
    "other_pipeline": ("Other mapped pipeline (not irrigation)", dict(color="#8e5bb5", lw=1.0)),
    "station": ("Pumping station + pool (lateral flow also seeps in at the pile seams)",
                dict(marker="*", color="#1a1a1a", markersize=15, markeredgecolor="white", ls="none")),
    "sluice_gate": ("Sluice gate", dict(marker="v", color="#444", markersize=7, ls="none")),
    "dam": ("Dam / weir", dict(marker="s", color="#444", markersize=6, ls="none")),
    "pumping_station": ("Other pumping station / water works", dict(marker="D", color="#777", markersize=6, ls="none")),
}


def draw(ax, roles=("river", "other_drain", "connected_drain", "main_canal", "other_pipeline"),
         structures=True, key=True, key_kwargs=None):
    """Draw the infrastructure on a lon/lat matplotlib axis, with a key saying what is what.
    Returns the key's handles so a caller can place them itself (``key=False``)."""
    from matplotlib.lines import Line2D

    if not available():
        return []
    # Keep the map framed on what the caller already drew (the district), not the
    # whole channel network around it.
    frame = (ax.get_xlim(), ax.get_ylim()) if ax.has_data() else None
    shown = []
    for role in roles:
        if role == "other_pipeline":
            drawn = [f for f in network()["features"] if f["properties"].get("role") == "other_pipeline"]
        else:
            drawn = channels(role)
        for feature in drawn:
            xs, ys = zip(*feature["geometry"]["coordinates"])
            style = dict(KEY[role][1])
            if role in ("main_canal", "connected_drain") and feature["properties"].get("culvert"):
                style, shown_role = dict(KEY["culvert"][1]), "culvert"
            else:
                shown_role = role
            ax.plot(xs, ys, zorder=5 if role in ("main_canal", "connected_drain") else 2,
                    solid_capstyle="round", **style)
            if shown_role not in shown:
                shown.append(shown_role)
    if structures:
        for feature in network()["features"]:
            p = feature["properties"]
            if p.get("role") != "structure":
                continue
            kind = {"weir": "dam", "lock_gate": "sluice_gate", "water_works": "pumping_station"}.get(p["kind"], p["kind"])
            if kind not in KEY:
                continue
            ax.plot(*feature["geometry"]["coordinates"], zorder=6, **KEY[kind][1])
            if kind not in shown:
                shown.append(kind)
        ax.plot(*STATION_LONLAT, zorder=7, **KEY["station"][1])
        shown.append("station")
    shown.append("irrigation_pipeline")   # key only: there is no public route to draw
    if frame is not None:
        ax.set_xlim(*frame[0])
        ax.set_ylim(*frame[1])
    order = [r for r in KEY if r in shown]
    handles = [Line2D([], [], label=KEY[r][0], **KEY[r][1]) for r in order]
    if key:
        ax.legend(handles=handles, **({"loc": "lower left", "fontsize": 7, "framealpha": 0.9} | (key_kwargs or {})))
    return handles


def pipeline_summary() -> str:
    """One line on the district's irrigation pipelines (bundled notes)."""
    p = network().get("irrigation_pipelines", {})
    if not p:
        return ""
    return (f"Irrigation pipelines (routes not public): Hakusan (North) main {p['hakusan_main_m']:,} m + branch "
            f"{p['hakusan_branch_m']:,} m; Jimbee (South) main {p['jimbee_main_m']:,} m (incl. 300 m of "
            f"{p['jimbee_main_diameter_mm']} mm) + branch {p['jimbee_branch_m']:,} m; station pumps 3 x 700 mm, "
            "max 3.304 m3/s.")


def outline(frame_geometry=None):
    """District outline from the member parcels (for maps)."""
    import geopandas as gpd

    if frame_geometry is None:
        from .district import load_district
        frame_geometry = gpd.GeoSeries.from_wkt(load_district(2022).geometry_wkt, crs="EPSG:4326")
    metric = gpd.GeoSeries(frame_geometry, crs="EPSG:4326").to_crs("EPSG:6677")
    return gpd.GeoSeries([metric.buffer(25).union_all().buffer(-20)], crs="EPSG:6677").to_crs(4326).iloc[0]
