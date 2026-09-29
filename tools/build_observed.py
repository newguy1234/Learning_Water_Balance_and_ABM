"""Build the observed series the environment is scored against, from Okura et al. (2025).

    python tools/digitize_fig8.py
    python tools/digitize_fig6.py
    python tools/build_observed.py

In the units the paper measured them in:

    irrigation   daily, per cyclic block, a VOLUME (m3/d) -- Fig. 8, the pump records,
                 digitized. Not rescaled: it is the measurement.
    drainage     daily, per cyclic block, a DEPTH (mm/d) -- Fig. 6's monthly discharge,
                 digitized, rescaled so the season equals Table 4 (the ADCP totals), and
                 spread within each month half evenly and half by rain. Where Fig. 6 does not
                 agree with Table 4 to within TOLERANCE it is not used and the drainage is
                 interpolated from the tables instead (okura.interpolate_daily), and the file
                 says so in its source column.
    withdrawal   daily, W (m3/d) -- Fig. 8, digitized
    rain, ET, O, LFV_pump -- okura.interpolate_daily (Tables 2 and 5)

Writes data/observed/okura_<year>_blocks.csv and okura_<year>_station.csv, which
cyclic.load_observed prefers over the interpolation.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from cyclic import data, okura  # noqa: E402

OBSERVED = os.path.join(PROJECT, "data", "observed")
RAIN_WEIGHT = 0.5
TOLERANCE = 0.30


def main():
    fig8 = pd.read_csv(os.path.join(OBSERVED, "okura_fig8_daily.csv"), parse_dates=["date"])
    fig6 = pd.read_csv(os.path.join(OBSERVED, "okura_fig6_monthly.csv"))

    for year in okura.YEARS:
        dates = okura.season_dates(year)
        month = dates.month.to_numpy()
        interpolated, station = okura.interpolate_daily(year, data.weather(year))
        daily = fig8[fig8["year"] == year].set_index("date").reindex(dates)
        frames = []
        print(f"\n{year}")
        for block in okura.BLOCKS:
            area = okura.cyclic_area_m2(block)
            t4 = okura.TABLE4_MM[(block, year)]
            base = interpolated[interpolated["block"] == block].reset_index(drop=True)
            rain = base["rain_mm"].to_numpy()
            irrigation_m3 = daily[f"irrigation_{block}_m3"].to_numpy()

            monthly = fig6[(fig6["block"] == block) & (fig6["year"] == year)].set_index("month")
            figure_total = float((monthly["discharge"] * monthly["days"]).sum())
            figure_mean, table_mean = figure_total / okura.SEASON_DAYS, t4["drainage"] / okura.SEASON_DAYS
            if figure_total <= 0 or abs(figure_mean - table_mean) / table_mean > TOLERANCE:
                drainage_mm = base["drainage_mm"].to_numpy()
                how = (f"drainage interpolated from Tables 2/4/7 (Fig. 6 discharge averages {figure_mean:.1f} "
                       f"mm/d against Table 4's {table_mean:.1f})")
            else:
                scale = t4["drainage"] / figure_total
                drainage_mm = np.zeros(len(dates))
                for m in okura.MONTHS:
                    sel = np.flatnonzero(month == m)
                    total = monthly.loc[m, "discharge"] * len(sel) * scale
                    r = rain[sel]
                    by_rain = r / r.sum() if r.sum() > 0 else np.full(len(sel), 1.0 / len(sel))
                    drainage_mm[sel] = total * ((1 - RAIN_WEIGHT) / len(sel) + RAIN_WEIGHT * by_rain)
                how = f"drainage from Fig. 6 monthly discharge x {scale:.3f} to match Table 4"

            source = f"Okura et al. (2025): irrigation Fig. 8 daily (digitized); {how}"
            frame = pd.DataFrame({
                "date": dates, "block": block, "area_m2": area,
                "irrigation_m3": irrigation_m3, "irrigation_mm": irrigation_m3 / area * 1000.0,
                "drainage_mm": drainage_mm, "drainage_m3": drainage_mm * area / 1000.0,
                "rain_mm": rain, "et_mm": base["et_mm"].to_numpy(), "source": source,
            })
            table_irrigation_m3 = t4["irrigation"] * area / 1000.0
            frame["irrigation_all_m3"] = irrigation_m3 * okura.TABLE5_1000M3[year][f"IR_all_{block}"] * 1000.0 / table_irrigation_m3
            frames.append(frame)

            got = irrigation_m3.sum() / 1000.0
            want = table_irrigation_m3 / 1000.0
            print(f"  {block}: irrigation {got:,.0f} x10^3 m3 (Fig. 8) vs {want:,.0f} (Table 4) {100 * (got - want) / want:+.1f}%"
                  f"  = {irrigation_m3.sum() / area * 1000.0 / okura.SEASON_DAYS:.1f} mm/d;  "
                  f"drainage {drainage_mm.sum():.0f} mm vs {t4['drainage']} -- {how}")

        blocks = pd.concat(frames, ignore_index=True)
        station["W_m3"] = daily["withdrawal_m3"].to_numpy()
        print(f"  withdrawal {station['W_m3'].sum() / 1000.0:,.0f} x10^3 m3 (Fig. 8) vs "
              f"{okura.TABLE5_1000M3[year]['W']:,} (Table 5)")
        blocks.to_csv(os.path.join(OBSERVED, f"okura_{year}_blocks.csv"), index=False)
        station.to_csv(os.path.join(OBSERVED, f"okura_{year}_station.csv"), index=False)
    print(f"\nwritten to {OBSERVED}")


if __name__ == "__main__":
    main()
