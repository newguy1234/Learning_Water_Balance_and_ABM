"""Rebuild the Shiroyama-Jimbee North / South block boundaries from public data.

CURRENT METHOD (2026-09-29). When cyclic/bundled/district_membership.csv exists (built by
tools/build_infrastructure.py), membership, block and cyclic status come from Okura et al.
(2025) Fig. 1a, registered onto the Fude parcels -- that map is drawn from the Chiba
Prefecture Federation of Land Improvement Associations' infrastructure data. The traced
outline and straight DIVIDER below are then only the fallback. See
cyclic/infrastructure.py and cyclic/bundled/SOURCES.md.

    pip install duckdb geopandas shapely matplotlib
    python -m cyclic blocks

WHERE THE DATA COMES FROM
    MAFF's Fude polygons (農地の区画情報, 筆ポリゴン) -- every farm parcel in Japan,
    tagged 田 (paddy) or 畑 (upland). Two ways to get them:

    1. Source Coop GeoParquet mirror (what this script uses). One file per fiscal
       year, ~14 GB, but DuckDB reads only the rows inside your bounding box over
       HTTP, so a district this size downloads in a minute or two. No login.
           https://data.source.coop/pacificspatial/field-polygon-jp/parquet/
           jp_field_polygons_{2021..2024}.parquet

    2. MAFF directly, by prefecture or municipality, as GeoJSON in a zip. Needs a
       short usage survey before the download link appears.
           https://open.fude.maff.go.jp/
       If you use this route, unzip it and set LOCAL_PARCELS to the .geojson path.

    The first run caches the parcels to data/fude_parcels.gpkg, so later runs are
    instant and offline.

WHAT YOU HAVE TO SUPPLY
    Nothing public records where Shiroyama-Jimbee ends or where the North/South split
    runs. That lives with the Imbanuma Land Improvement District. So two things below
    are yours to set, from Okura et al. (2025) Fig. 1:

    DISTRICT   the district outline, traced as (lon, lat) corners. Left as None it
               keeps every paddy in the bounding box, which is what over-counts.
    DIVIDER    the main drainage canal, as two or more (lon, lat) points. Paddy north
               of it is North block, south of it is South block.

    Then read the printed table: when North is near 567.7 ha and South near 415.2 ha,
    the boundary matches the paper. Adjust and re-run until it does.
"""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon, box

# --- settings -------------------------------------------------------------------

# lon/lat bounding box around the district (min lon, min lat, max lon, max lat).
BBOX = (140.219444, 35.760833, 140.313889, 35.880)

# Fiscal year of the parcel survey. Source Coop has 2021 to 2024.
FISCAL_YEAR = 2021

# Okura et al. (2025), Table 1 -- what the rebuilt blocks should come out at.
TARGET_HA = {"North": 567.7, "South": 415.2}

# The irrigation pumping station, which Okura places between the two blocks.
POOL = (140.267, 35.807)

# The district outline, traced from Okura Fig. 1a as (lon, lat). None keeps the
# whole bounding box, which reproduces the over-count.
DISTRICT = None
# DISTRICT = [(140.23, 35.77), (140.30, 35.77), (140.30, 35.87), (140.23, 35.87)]

# Or a GeoJSON / GeoPackage outline, which wins over DISTRICT when set. Use this for
# anything traced in QGIS, or for district_outline.geojson, which already cuts the
# district down from 4,102 ha to about 1,121 ha against the paper's 982.9.
# DISTRICT_FILE = None
DISTRICT_FILE = "district_outline.geojson"

# The main drainage canal dividing North from South, as (lon, lat) points running
# west to east alongside Lake Kita Imbanuma (Okura §2.1: north of the canal is the
# North block, south is the South; leftover recapture is same-day LFV, not lining-catch).
# The default is a placeholder through the pumping station -- replace it with the
# canal traced from Okura Fig. 1b.
DIVIDER = [(BBOX[0], POOL[1]), (BBOX[2], POOL[1])]

# Set this to a MAFF GeoJSON path to skip Source Coop entirely.
LOCAL_PARCELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DATA = os.path.join(PROJECT, "data")
OUTPUTS = os.path.join(PROJECT, "outputs")
# GeoPackage rather than parquet, so the cache needs nothing beyond geopandas.
CACHE = os.path.join(DATA, "fude_parcels.gpkg")

# Japan Plane Rectangular CS zone IX (JGD2011), which covers Chiba. Areas are measured
# in metres here; measuring them in degrees would be wrong by a factor that changes
# with latitude.
AREA_CRS = "EPSG:6677"


# --- download -------------------------------------------------------------------

def download_parcels(bbox=BBOX, year=FISCAL_YEAR):
    """Fude parcels inside the box, from the Source Coop mirror, cached locally."""
    if os.path.exists(CACHE):
        print(f"using cached parcels: {CACHE}")
        return gpd.read_file(CACHE)

    import duckdb

    minx, miny, maxx, maxy = bbox
    url = ("https://data.source.coop/pacificspatial/field-polygon-jp/parquet/"
           f"jp_field_polygons_{year}.parquet")
    print(f"querying {url}\n  inside {bbox} ... (a minute or two the first time)")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # Filtering on the stored centroid first lets DuckDB skip most of the 14 GB file
    # using parquet statistics; ST_Intersects then decides membership exactly.
    margin = 0.01
    frame = con.execute(f"""
        SELECT polygon_uuid, land_type_jp AS land_type, ST_AsText(GEOM) AS wkt
        FROM read_parquet('{url}')
        WHERE point_lng BETWEEN {minx - margin} AND {maxx + margin}
          AND point_lat BETWEEN {miny - margin} AND {maxy + margin}
          AND ST_Intersects(GEOM, ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}))
    """).df()
    con.close()

    if frame.empty:
        raise SystemExit("No parcels found -- check the bbox and fiscal year.")

    parcels = gpd.GeoDataFrame(
        frame.drop(columns="wkt"),
        geometry=[wkt.loads(text) for text in frame["wkt"]],
        crs="EPSG:4326",
    )
    parcels.to_file(CACHE, driver="GPKG")
    print(f"  {len(parcels):,} parcels, cached to {CACHE}")
    return parcels


def load_parcels():
    if LOCAL_PARCELS:
        print(f"reading {LOCAL_PARCELS}")
        parcels = gpd.read_file(LOCAL_PARCELS).to_crs("EPSG:4326")
        # MAFF's own files call the column land_type; keep whatever is there.
        return parcels.cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]]
    return download_parcels()


# --- rebuild --------------------------------------------------------------------

def north_of(line, point):
    """True when a point lies north of a west-to-east polyline.

    Uses the line's latitude at the point's longitude, interpolated between vertices,
    so a canal that bends is followed rather than straightened.
    """
    xs = [x for x, _ in line.coords]
    ys = [y for _, y in line.coords]
    if point.x <= xs[0]:
        return point.y > ys[0]
    if point.x >= xs[-1]:
        return point.y > ys[-1]
    for (x0, y0), (x1, y1) in zip(line.coords, list(line.coords)[1:]):
        if x0 <= point.x <= x1:
            share = (point.x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return point.y > y0 + share * (y1 - y0)
    return point.y > ys[-1]


def rebuild(parcels):
    # Paddy only. Upland (畑) parcels are not irrigated by the cyclic system.
    paddy = parcels[parcels["land_type"] == "田"].copy()
    print(f"\n{len(parcels):,} parcels in the box, {len(paddy):,} of them paddy")

    from . import infrastructure
    if infrastructure.available():
        centres = paddy.to_crs(AREA_CRS).geometry.centroid.to_crs("EPSG:4326")
        keep, block, cyclic = infrastructure.classify(paddy["polygon_uuid"].to_numpy(),
                                                      centres.y.to_numpy())
        paddy["block"] = block
        paddy["fig1_cyclic"] = cyclic.astype(int)
        paddy = paddy[keep].copy()
        paddy["area_ha"] = paddy.to_crs(AREA_CRS).area / 10_000.0
        outline = infrastructure.outline(paddy.geometry)
        canal = LineString(infrastructure.main_canal_vertices())
        return paddy, outline, canal

    if DISTRICT_FILE:
        # union_all so an outline made of several pieces is treated as one district.
        # Relative paths are taken from data/, not wherever it is run from.
        path = DISTRICT_FILE if os.path.isabs(DISTRICT_FILE) else os.path.join(DATA, DISTRICT_FILE)
        outline = gpd.read_file(path).to_crs("EPSG:4326").union_all()
    elif DISTRICT:
        outline = Polygon(DISTRICT)
    else:
        outline = box(*BBOX)
    paddy = paddy[paddy.geometry.centroid.within(outline)].copy()

    canal = LineString(DIVIDER)
    paddy["block"] = [
        "North" if north_of(canal, geom.centroid) else "South"
        for geom in paddy.geometry
    ]
    paddy["area_ha"] = paddy.to_crs(AREA_CRS).area / 10_000.0
    return paddy, outline, canal


def report(paddy):
    print(f"\n  {'block':<7} {'parcels':>8} {'rebuilt ha':>11} {'Okura ha':>9} {'error':>8}")
    total_rebuilt = total_target = 0.0
    for block in ("North", "South"):
        rows = paddy[paddy["block"] == block]
        area = rows["area_ha"].sum()
        target = TARGET_HA[block]
        total_rebuilt += area
        total_target += target
        print(f"  {block:<7} {len(rows):>8,} {area:>11.1f} {target:>9.1f} "
              f"{100 * (area - target) / target:>+7.1f}%")
    print(f"  {'total':<7} {len(paddy):>8,} {total_rebuilt:>11.1f} {total_target:>9.1f} "
          f"{100 * (total_rebuilt - total_target) / total_target:>+7.1f}%")
    if "fig1_cyclic" in paddy:
        from .okura import TABLE1_HA
        print(f"\n  {'block':<7} {'part':<11} {'rebuilt ha':>11} {'Okura ha':>9}")
        for block in ("North", "South"):
            for flag, part in ((1, "cyclic"), (0, "non_cyclic")):
                rows = paddy[(paddy["block"] == block) & (paddy["fig1_cyclic"] == flag)]
                print(f"  {block:<7} {part:<11} {rows['area_ha'].sum():>11.1f} {TABLE1_HA[block][part]:>9.1f}")


def plot(paddy, outline, canal, save_to=None):
    fig, axis = plt.subplots(figsize=(12, 10))
    colours = {("North", 1): "#1f6fb4", ("North", 0): "#9cc3e6",
               ("South", 1): "#2e7d32", ("South", 0): "#a5d6a7"}
    cyclic = paddy["fig1_cyclic"] if "fig1_cyclic" in paddy else None

    for (block, flag), colour in colours.items():
        rows = paddy[(paddy["block"] == block) & ((cyclic == flag) if cyclic is not None else (flag == 1))]
        if rows.empty:
            continue
        part = ("cyclic" if flag else "non-cyclic") if cyclic is not None else ""
        rows.plot(ax=axis, color=colour, edgecolor="none", alpha=0.8,
                  label=f"{block} {part}  {rows['area_ha'].sum():,.0f} ha")

    gpd.GeoSeries([outline], crs="EPSG:4326").boundary.plot(
        ax=axis, color="black", linewidth=1.0, linestyle="--", label="district outline")
    from . import infrastructure
    key = []
    if infrastructure.available():
        key = infrastructure.draw(axis, key=False)
    else:
        gpd.GeoSeries([canal], crs="EPSG:4326").plot(
            ax=axis, color="#d94801", linewidth=2.2, label="North / South divider", zorder=4)
        axis.plot(*POOL, marker="*", markersize=16, color="#d94801", zorder=5,
                  markeredgecolor="white", label="irrigation pumping station")

    axis.set_title("Hakusan-Jimbee (Shiroyama-Jimbee) district: Okura Fig. 1a on MAFF Fude parcels")
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
    blocks_key = axis.get_legend_handles_labels()[0]
    axis.legend(handles=blocks_key + key, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    if infrastructure.available():
        fig.text(0.01, 0.01, infrastructure.pipeline_summary(), fontsize=6.5, wrap=True)
    axis.set_aspect(1 / 0.81)   # cos(35.8 deg): keeps the map from looking squashed
    axis.ticklabel_format(useOffset=False)
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=140, bbox_inches="tight")
        print(f"\nsaved {save_to}")
    plt.show()


def main():
    parcels = load_parcels()
    paddy, outline, canal = rebuild(parcels)
    report(paddy)
    os.makedirs(OUTPUTS, exist_ok=True)
    plot(paddy, outline, canal, save_to=os.path.join(OUTPUTS, "blocks.png"))


if __name__ == "__main__":
    main()
