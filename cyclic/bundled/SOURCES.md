# District infrastructure: sources

What `cyclic/infrastructure.py` (and through it `cyclic.data.paddies()` and the ABM) uses,
where each piece comes from, and what is still assumed. Built by
`tools/build_infrastructure.py`; retrieved 2026-09-29.

## The study area is the Hakusan-Jimbee pumping station's service area

| Fact | Source |
|---|---|
| Okura's study site ("Shiroyama-Jimbee") is 白山甚兵衛機場's command area: North 567.7 ha + South 415.2 ha = 982.9 ha (Table 1) | Okura et al. (2025), Table 1 |
| 白山甚兵衛機場 serves 983 ha, merged from 白山機場 (604 ha) and 甚兵衛機場 (391 ha); cyclic irrigation since April 2015; about half its irrigation water is taken from the lowland drainage canal (低地排水路) | MAFF Kanto, 2015 briefing (below), pp. 8, 11, 14 |
| Station: in service FY2015, max. supply 3.304 m³/s, discharge tank PC D = 26.0 m, H = 26.4 m | MAFF Kanto, main facilities page |
| Station location 140.26372 E, 35.80755 N: the round discharge tank (OSM way 1082034661), in 成田市松崎干拓地, beside the lake dyke where the main drainage canal meets a canal from the south-east; checked on GSI aerial imagery | OpenStreetMap; GSI seamless aerial photos |
| System layout: lake / lowland drainage canal → pumping station → trunk pipeline → branch pipelines → hydrant (給水栓) at each paddy → branch drains → lowland drainage canal → pumped back; surplus pumped to the lake to hold the canal level constant | MAFF Kanto, 2015 briefing, pp. 9–10 |
| Main drainage canal along the lake shore with arms north and south of the station, ~10 m wide; pool 18 × 68 m between them | Okura et al. (2025), Fig. 1b, Fig. 2a, Fig. 2b |
| Whole Inbanuma Phase 2 project: 4,958 ha; 3 pumping + 3 combined stations; supply pipelines 52.6 km (trunk 1.2 km, branch 51.7 km); main drains 1.1 km; detailed specifications not public (「一般に公表されていない諸元」) | MAFF project evaluation R7; MAFF 2015 briefing p. 7 |

## Parcel membership, blocks and cyclic status: `district_membership.csv`

Every MAFF Fude paddy (田) parcel in the box is classed from **Okura et al. (2025) Fig. 1a**,
which the authors drew from the Chiba Prefecture Federation of Land Improvement
Associations' agricultural infrastructure data. The figure is registered onto the parcels
by a similarity transform about the station (fitted scale 101.0 px/km against the figure's
own scale bar of 101.2 px/km, rotation 0.5°), then each parcel is read from the colours
around its centroid: gold = North cyclic, magenta = South cyclic, light grey = non-cyclic.

Result against Okura Table 1 (gross parcel area, before `cyclic.data` scales each part to
Table 1 exactly):

| Part | Fig. 1a on Fude (ha) | Table 1 (ha) |
|---|---|---|
| North cyclic | 250.7 | 260.4 |
| North non-cyclic | 313.8 | 307.3 |
| South cyclic | 265.4 | 263.5 |
| South non-cyclic | 150.1 | 151.7 |
| Total | 980.0 | 982.9 |

Discrepancies with the earlier traced outline (`data/district_outline.geojson`, 4,925 parcels,
~1,121 ha) that this corrects:

* **Dropped (~221 ha: 187.1 ha read as outside, 33.9 ha unknown).** The far southern valley
  paddies (beyond where Fig. 1a ends), a band at the north block's north-east corner past its
  drawn edge, a strip between the Tone-side block and the North cyclic block, and a few
  parcels just east of the station.
* **Added (~77 ha).** The non-cyclic strip along the south-west lake shore and the isolated
  valley to the north-east, both drawn in Fig. 1a but outside the old outline.
* **Cyclic status.** Previously the parcels nearest a straight placeholder line through the
  pump were marked cyclic until Table 1's area was filled; now it is what Fig. 1a shows.
* **Pool location.** Previously 140.267 E, 35.807 N (~270 m east, in a solar farm); now the
  station itself.
* **Canal.** Previously a straight east–west line through the pump; now the real main
  drainage canal (OSM way 60622667) along the lake dyke.

`unknown` parcels (hidden under the figure's compass or scale bar, 138.7 ha, of which
33.9 ha were inside the old outline) are left out: including them would overshoot Table 1's
total by 31 ha. Parcels under the legend are `outside`: the north block's drawn edge stops
short of the legend with white space between.

## Drainage channels: `infrastructure.geojson`

Two mapped sources, each feature tagged with its `source`:

* OpenStreetMap waterways (`river`, `stream`, `ditch`, `drain`, `canal`), via the Overpass API.
  © OpenStreetMap contributors, available under the Open Database License (ODbL).
* GSI river / channel centrelines (国土地理院 ベクトルタイル 河川中心線, `experimental_rvrcl`,
  z16), including narrow channels (細河川) and underground man-made channels
  (人工水路（地下）, marked as culverts). These map many farm and collector drains that OSM
  lacks. First-class rivers (一級河川) are kept as `river`.

Roles:

* `main_canal`: OSM way 60622667, the channel along the lake dyke through the station
  (2.4 km), matching Okura Fig. 1b's main drainage canal.
* `connected_drain`: drainage channels joined to the main canal, 40.9 km (25.8 km GSI,
  15.1 km OSM). Rules, all inferences on the mapped lines:
  * channel ends within 25 m are joined (drains pass under farm roads through short culverts
    the maps leave out; connected length rises from 29 km at 15 m to 45 km at 25 m and levels
    off, 46 km at 50 m);
  * only channels within 250 m of cyclic parcels count (Okura Fig. 2b: non-cyclic blocks drain
    out of the scheme), and connected channels are clipped to that area, so drainage leaves
    the district only at the station, never on to the rivers;
  * lines drawn mostly across the open lake (OSM 印旛沼 water area) are lake flow lines, not
    drains, and are left out.
* `other_drain`: channels of other systems (373 km), which do not reach this pool.
* `river`: natural rivers, for context; not used for connectivity (they meet the drains at
  gates and culverts).

Irrigation pipeline routes are **not** included: GSI road centrelines map the farm roads
along which pipelines usually run, but no source ties this district's pipes to them.

## Pipes, structures and underground collection (also in `infrastructure.geojson`)

**Irrigation pipelines** (key `irrigation_pipelines`; drawn in the key only, routes not public):
the station's three φ700 mm pumps feed pressurised pipelines to a hydrant (給水栓) at each
paddy. Built lengths: 白山 (Hakusan, North area) main 940 m + branch 2,940 m; 甚兵衛 (Jimbee,
South area) main 1,480 m + 300 m of φ900 mm laid in 2011 + branch 2,390 m
(日本工業経済新聞, 2012-08-09, https://www.nikoukei.co.jp/news/detail/214935). Before renewal,
about 90% of the area's pipes were concrete or asbestos cement (MAFF 2015 briefing, p. 6).

**Structures** (role `structure`, OSM): a sluice gate on the 長門川 at the district's
north-west edge, two dams / weirs, and two pumping stations / water works; the 成田給水場 is
Narita's drinking-water station, not irrigation. **Culverts**: channel sections OSM tags
`tunnel=culvert` (`culvert: true`), 180 of them, 12 on drains that reach the pool.
**Other pipelines** (role `other_pipeline`): three mapped east of the district; not the
irrigation pipes.

**Underground / lateral-flow collection** (key `lateral_collection`):

* Farm drains have bare-soil beds: surface drainage seeps into the lateral layer and lateral
  flow enters the drains, both reaching the pool (Okura et al. 2025, §4.2).
* The pool is lined with concrete piles; lateral flow and groundwater enter through the seams,
  and small drainage pipes in the piles relieve pressure (Okura et al. 2025, §4.2).
* No public record of pipe underdrains (暗渠排水) in this district was found.
* In the model: Hooghoudt lateral drain flow with 10 m drain spacing and 0.8 m equivalent
  depth (`cyclic.physics.Physics`), and a per-paddy recaptured share set by distance to the
  nearest channel that reaches the pool.

## Canal water level: `okura_fig3_canal_depth.csv`

Daily water depth of the main drainage canal at the North and South ADCP sites, 17 April –
25 August 2021 and 2022, colour-traced from **Okura et al. (2025) Fig. 3** (the last three days
run past the plotted axis and repeat the last value). 2021 North averaged 0.97 m, 2022 both
arms 0.82 m. The normal (no-backwater) depth is the 2022 median, 0.80 m; Okura §4.1 puts the
2021 North backwater down to the canal staying high in the wet year.

## Ground elevation: `paddy_elevation.csv`

Median of GSI's 5 m DEM (基盤地図情報 数値標高モデル 5 m, `dem5a` tiles, m above T.P.) over
5 m points inside each member parcel. North cyclic land sits at a median ~1.2 m, South cyclic
~3.2 m (Okura §4.1: the South block is higher, so less affected by backwater). The DEM has
voids over water, so the canal's own surface elevation is not taken from it.

## How these are used

* **Recapture of lateral flow** (`canal_distance_m`, `cyclic.canal.distance_to_canal_m`):
  distance to the nearest channel that reaches the pool (main canal or a connected drain),
  since lateral flow is recaptured where it enters those drains (Okura §4.2). It sets each
  cyclic paddy's share (`pixel_ratio_fields`; district mean still pinned to Table 5).
* **Drainage timing**: along the channels to the station (`drain_path_m`).
* **Backwater** (ABM, `SeasonConfig.canal_depth`): a cyclic paddy's farm drain is free while
  its ground stands at least 0.10 m above the drain reference plus the canal's rise over its
  normal depth. The reference is set by one rule, not fitted to 2021: at the normal depth the
  lowest 2% of cyclic land (by area) is just at that margin. Below it, the blocked share of
  its drainage rises linearly to all of it, and that water stays on the field.
* **Irrigation supply** (ABM, `SeasonConfig.pump_capacity_m3_s`): at most the station's
  3.304 m³/s (MAFF) is delivered in a day; nearer paddies are served first beyond that.

## What is still assumed

* **Farm drains are only partly mapped in OSM.** A paddy's `drain_entry_m` is the straight
  distance to the nearest connected channel; the real farm-drain path is longer.
* **Irrigation pipelines are not public.** The station serves paddies in order of straight
  distance from the pool (`pool_distance_m`), not along the pipes.
* **Drainage flow speed** (`SeasonConfig.drain_velocity_m_s`, default 0.2 m/s) is an
  assumption for slow flat-lowland drains, not a measurement.
* **No farm-drain bed leakage.** Okura §4.2 describes drainage leaking through bare-soil farm
  drain beds into the lateral layer; the plot-scale data to size it (Tables S3/S4) is not in
  this project.

Better data would come from the Inbanuma Land Improvement District (印旛沼土地改良区) facility
ledgers / the Chiba federation's GIS, the MAFF Inbanuma Phase 2 project office's design
drawings, or Chiba Prefecture's Inba Agricultural Office (field consolidation drawings).

## References

* Okura, F., Kato, T. et al. (2025). Water balance analysis of hydrological processes in cyclic
  irrigation: A case study of the Imbanuma irrigation area in Chiba, Japan. *Journal of
  Hydrology: Regional Studies* 57, 102111. https://doi.org/10.1016/j.ejrh.2024.102111 (CC BY-NC 4.0)
* MAFF Kanto Regional Agricultural Administration Office, Inbanuma Phase 2 project: main
  facilities. https://www.maff.go.jp/kanto/nouson/sekkei/kokuei/inba/gaiyo/02.html
* MAFF (2025). Project evaluation, 印旛沼二期地区 (R7).
  https://www.maff.go.jp/j//nousin/noukei/jigyouhyouka/R7-02/attach/pdf/r7kityuu-6.pdf
* MAFF Kanto, 印旛沼二期農業水利事業所 (2015-08-05). 循環かんがいによる印旛沼の水質保全の取組について.
  https://inba-numa.com/html/file/kaigi/iinkai23_5sankou_150805.pdf
* OpenStreetMap contributors. https://www.openstreetmap.org (ODbL), Overpass API https://overpass-api.de
* Geospatial Information Authority of Japan, seamless aerial photographs (visual check only)
  and 5 m DEM tiles (`dem5a_png`). https://maps.gsi.go.jp/development/ichiran.html
* MAFF 農地の区画情報（筆ポリゴン）, Source Coop mirror
  https://data.source.coop/pacificspatial/field-polygon-jp/
