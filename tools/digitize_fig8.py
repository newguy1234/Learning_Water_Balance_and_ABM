"""Digitize Okura et al. (2025) Fig. 8 into a daily series.

    python tools/digitize_fig8.py [path to the paper's PDF]

Fig. 8 plots, per day of the 131-day irrigation period, the irrigation water pumped to the
North and South cyclic blocks and the withdrawal from Lake Kita-Imbanuma, in x10^3 m3/d --
panel (a) 2021, panel (b) 2022. These are the pump records: the observed irrigation the
environment should be scored against.

HOW
  * the figure image is taken straight out of the PDF (page 15) and cached in data/
  * the y scale comes from the light-grey gridlines (0 to 160 in steps of 20), fitted
  * each series is picked out by colour: pink North, green South, blue withdrawal
  * the x scale is the extent of the lines: the first point is April 17, the last August 25
  * at each day the line's row is read at that column; where the day is a peak or a trough
    (both neighbours on the same side) the line's top or bottom edge is used instead of its
    middle, so spikes are not flattened by the line's own thickness
  * a day hidden under another series is interpolated from its neighbours

CHECKS, printed: the season totals against Table 4 x Table 1 area (irrigation) and Table 5
(withdrawal); and outputs/digitized_fig8_check.png, the digitized points drawn over the image.

Writes data/observed/okura_fig8_daily.csv: date, year, irrigation_North_m3,
irrigation_South_m3, withdrawal_m3 (per day), and whether each value was interpolated.
"""

import io
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DATA = os.path.join(PROJECT, "data")
OUTPUTS = os.path.join(PROJECT, "outputs")
sys.path.insert(0, PROJECT)

from cyclic import okura  # noqa: E402

PDF = (r"C:\Users\psypc2\OneDrive - The University of Nottingham\Desktop\Water balance analysis of "
       r"hydrological processes in cyclic irrigation a case study of the Imbanuma irrigation area in Chiba Japan.pdf")
PAGE = 15
YEARS = (2021, 2022)
TICKS = np.arange(160, -1, -20)            # gridlines from the top down, x10^3 m3/d
SERIES = ("irrigation_North", "irrigation_South", "withdrawal")


def figure_image(pdf):
    cache = os.path.join(DATA, "okura_fig8.png")
    if not os.path.exists(cache):
        import pypdf

        image = pypdf.PdfReader(pdf).pages[PAGE - 1].images[0]
        Image.open(io.BytesIO(image.data)).convert("RGB").save(cache)
    return np.asarray(Image.open(cache).convert("RGB"))


def hsv(rgb):
    a = rgb.astype(float) / 255.0
    top, bottom = a.max(-1), a.min(-1)
    delta = top - bottom
    safe = np.where(delta > 0, delta, 1.0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    hue = np.select([top == r, top == g], [((g - b) / safe) % 6, (b - r) / safe + 2], (r - g) / safe + 4) * 60.0
    hue = np.where(delta > 0, hue, 0.0)
    sat = np.where(top > 0, delta / np.where(top > 0, top, 1.0), 0.0)
    return hue, sat, top


def colour_masks(rgb):
    hue, sat, val = hsv(rgb)
    coloured = (sat > 0.35) & (val > 0.25)
    series = {
        "irrigation_North": coloured & (hue > 300) & (hue < 350),
        "irrigation_South": coloured & (hue > 70) & (hue < 160),
        "withdrawal": coloured & (hue > 195) & (hue < 245),
    }
    grey = (sat < 0.08) & (val > 0.70) & (val < 0.95)
    return series, grey


def runs(indices):
    """Consecutive runs in a sorted index array, as (start, end) pairs."""
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return list(zip(starts, ends))


def gridline_rows(grey, width):
    left, right = int(0.12 * width), int(0.95 * width)
    rows = np.flatnonzero(grey[:, left:right].mean(axis=1) > 0.5)
    # The figure's frame is grey all the way to the left edge; gridlines start after the labels.
    rows = rows[~grey[rows, int(0.02 * width)]]
    # A gridline crossed by a steep series can come out as two runs a pixel or two apart.
    merged = []
    for centre in ((a + b) / 2.0 for a, b in runs(rows)):
        if merged and centre - merged[-1][-1] <= 4:
            merged[-1].append(centre)
        else:
            merged.append([centre])
    return np.array([np.mean(group) for group in merged])


def panels(centres):
    """Split gridline rows into the two panels by the large gap between them."""
    gaps = np.diff(centres)
    cut = np.argmax(gaps)
    groups = [centres[:cut + 1], centres[cut + 1:]]
    for group in groups:
        if len(group) != len(TICKS):
            raise SystemExit(f"expected {len(TICKS)} gridlines per panel, found {len(group)}: {group}")
    return groups


def trace(mask, x0, x1, days):
    """One value (a row) per day for one series' mask."""
    xs = x0 + (x1 - x0) * np.arange(days) / (days - 1)
    step = (x1 - x0) / (days - 1)

    def rows_at(x, spread=1):
        col = int(round(x))
        return np.flatnonzero(mask[:, max(col - spread, 0):col + spread + 1].any(axis=1))

    def middle(x):
        ys = rows_at(x, 0)
        return np.median(ys) if ys.size else np.nan

    # Line half-thickness, from columns where the line is flat.
    widths = []
    for x in np.linspace(x0, x1, 400):
        ys = rows_at(x, 0)
        if 0 < ys.size < 12 and ys.max() - ys.min() == ys.size - 1:
            widths.append(ys.size)
    half = (np.median(widths) if widths else 4.0) / 2.0

    rows, hidden = np.full(days, np.nan), np.zeros(days, dtype=bool)
    for day, x in enumerate(xs):
        ys = rows_at(x)
        if ys.size == 0:
            hidden[day] = True
            continue
        mid = np.median(ys)
        left = middle(x - step / 3) if day > 0 else np.nan
        right = middle(x + step / 3) if day < days - 1 else np.nan
        if np.isfinite(left) and np.isfinite(right) and left > mid and right > mid:
            rows[day] = ys.min() + half          # a peak: its top edge
        elif np.isfinite(left) and np.isfinite(right) and left < mid and right < mid:
            rows[day] = ys.max() - half          # a trough: its bottom edge
        else:
            rows[day] = mid
    if hidden.any():
        known = np.flatnonzero(~hidden)
        rows[hidden] = np.interp(np.flatnonzero(hidden), known, rows[known])
    return xs, rows, hidden


def main(pdf=PDF):
    rgb = figure_image(pdf)
    height, width, _ = rgb.shape
    series_masks, grey = colour_masks(rgb)
    groups = panels(gridline_rows(grey, width))

    frames, overlay = [], []
    for year, centres in zip(YEARS, groups):
        slope, intercept = np.polyfit(centres, TICKS, 1)          # value = slope * row + intercept
        residual = np.abs(slope * centres + intercept - TICKS).max()
        top, bottom = int(centres[0]) - 15, int(centres[-1]) + 15
        crop = {name: m[top:bottom] for name, m in series_masks.items()}
        columns = np.flatnonzero(np.any([m.any(axis=0) for m in crop.values()], axis=0))
        columns = columns[(columns > 0.05 * width) & (columns < 0.99 * width)]
        x0, x1 = columns.min(), columns.max()

        dates = okura.season_dates(year)
        frame = pd.DataFrame({"date": dates, "year": year})
        for name in SERIES:
            xs, rows, hidden = trace(crop[name], x0, x1, len(dates))
            values = np.clip(slope * (rows + top) + intercept, 0.0, None)
            frame[f"{name}_m3"] = values * 1000.0
            frame[f"{name}_interpolated"] = hidden
            overlay.append((year, name, xs, rows + top))
        frames.append(frame)
        print(f"{year}: gridlines at rows {np.round(centres).astype(int).tolist()}, fit residual "
              f"{residual:.2f} x10^3 m3/d; lines span columns {x0}-{x1}")

    daily = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.join(DATA, "observed"), exist_ok=True)
    path = os.path.join(DATA, "observed", "okura_fig8_daily.csv")
    daily.to_csv(path, index=False)

    print("\nseason totals, x10^3 m3            digitized    paper     error")
    for year in YEARS:
        rows = daily[daily["year"] == year]
        for block in okura.BLOCKS:
            got = rows[f"irrigation_{block}_m3"].sum() / 1000.0
            want = okura.TABLE4_MM[(block, year)]["irrigation"] * okura.cyclic_area_m2(block) / 1e6
            print(f"  {year} irrigation {block:<5}  (Table 4) {got:10.0f} {want:8.0f}  {100 * (got - want) / want:+6.1f}%"
                  f"   hidden days {int(rows[f'irrigation_{block}_interpolated'].sum())}")
        got = rows["withdrawal_m3"].sum() / 1000.0
        want = okura.TABLE5_1000M3[year]["W"]
        print(f"  {year} withdrawal W      (Table 5) {got:10.0f} {want:8.0f}  {100 * (got - want) / want:+6.1f}%"
              f"   hidden days {int(rows['withdrawal_interpolated'].sum())}")
    print(f"\nwritten {path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"irrigation_North": "black", "irrigation_South": "orange", "withdrawal": "red"}
    fig, ax = plt.subplots(figsize=(19, 20.5), dpi=100)
    ax.imshow(rgb)
    for year, name, xs, rows in overlay:
        ax.plot(xs, rows, "o", ms=3.5, mfc="none", mec=colours[name], mew=1.0)
    ax.set_axis_off()
    fig.tight_layout()
    os.makedirs(OUTPUTS, exist_ok=True)
    check = os.path.join(OUTPUTS, "digitized_fig8_check.png")
    fig.savefig(check)
    print(f"written {check}")
    return daily


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PDF)
