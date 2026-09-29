"""Digitize Okura et al. (2025) Fig. 6 into monthly block water balances.

    python tools/digitize_fig6.py [path to the paper's PDF]

Fig. 6 is a stacked bar per month (April-August) and year (2021, 2022), panel (a) the North
block and (b) the South block, in mm/d: irrigation and rainfall above zero; evapotranspiration,
discharge and lateral flow below it. Lateral flow drawn above zero is negative Lf (inflow).

It is the only place the paper gives surface drainage (discharge) by month, so it is what
sets the monthly shape of the observed drainage series.

HOW
  * the image is taken out of the PDF (page 13) and cached in data/
  * y scale from the gridlines (30 to -30 in steps of 10), fitted
  * each bar is found from the columns its colours occupy; each row inside it is labelled by
    the hatch colour that dominates it (blue irrigation, orange rainfall, dark grey ET, gold
    discharge, green lateral flow), short gaps from the white in the hatching are filled, and
    each labelled run is a segment whose height is its value

CHECKS, printed: rainfall and ET against Table 2 (the same station for both blocks) and the
monthly block balance, irrigation + rain - ET - discharge - lateral, which Eq. 3 makes zero.

Writes data/observed/okura_fig6_monthly.csv: block, year, month, days and each term in mm/d.
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
sys.path.insert(0, PROJECT)

from cyclic import okura  # noqa: E402

sys.path.insert(0, HERE)
from digitize_fig8 import PDF, hsv, runs  # noqa: E402

PAGE = 13
TICKS = np.arange(30, -31, -10)
TERMS = ("irrigation", "rainfall", "et", "discharge", "lateral")
BARS = [(year, month) for month in okura.MONTHS for year in (2021, 2022)]


def figure_image(pdf):
    cache = os.path.join(DATA, "okura_fig6.png")
    if not os.path.exists(cache):
        import pypdf

        image = pypdf.PdfReader(pdf).pages[PAGE - 1].images[0]
        Image.open(io.BytesIO(image.data)).convert("RGB").save(cache)
    return np.asarray(Image.open(cache).convert("RGB"))


def labels(rgb):
    hue, sat, val = hsv(rgb)
    classes = np.full(rgb.shape[:2], -1, dtype=int)
    classes[(sat > 0.3) & (hue > 195) & (hue < 240) & (val > 0.45)] = 0          # irrigation
    classes[(sat > 0.25) & (hue > 8) & (hue < 36) & (val > 0.70)] = 1            # rainfall
    classes[(sat < 0.2) & (val < 0.5)] = 2                                       # evapotranspiration
    classes[(sat > 0.45) & (hue >= 36) & (hue < 62) & (val > 0.35)] = 3          # discharge
    classes[(sat > 0.25) & (hue > 70) & (hue < 140) & (val > 0.35)] = 4          # lateral flow
    grey = (sat < 0.08) & (val > 0.70) & (val < 0.95)
    return classes, grey


def gridlines(grey, width):
    left, right = int(0.15 * width), int(0.95 * width)
    rows = np.flatnonzero(grey[:, left:right].mean(axis=1) > 0.25)
    rows = rows[~grey[rows, int(0.01 * width)]]
    centres = np.array([(a + b) / 2.0 for a, b in runs(rows)])
    # Two panels of seven lines; keep, per panel, the seven most evenly spaced.
    cut = np.argmax(np.diff(centres))
    groups = [centres[:cut + 1], centres[cut + 1:]]
    out = []
    for group in groups:
        if len(group) < len(TICKS):
            raise SystemExit(f"found only {len(group)} gridlines in a panel: {group}")
        best = min((group[i:i + len(TICKS)] for i in range(len(group) - len(TICKS) + 1)),
                   key=lambda g: np.std(np.diff(g)))
        out.append(best)
    return out


def segments(classes, top, bottom, left, right):
    inner = classes[top:bottom, left + (right - left) // 6: right - (right - left) // 6]
    width = inner.shape[1]
    row_label = np.full(inner.shape[0], -1)
    for k in range(len(TERMS)):
        pass
    counts = np.stack([(inner == k).sum(axis=1) for k in range(len(TERMS))], axis=1)
    best = counts.argmax(axis=1)
    row_label = np.where(counts.max(axis=1) >= 0.08 * width, best, -1)
    # Hatching leaves white rows inside a segment: fill short gaps between equal labels.
    labelled = np.flatnonzero(row_label >= 0)
    for a, b in zip(labelled[:-1], labelled[1:]):
        if 1 < b - a <= 9 and row_label[a] == row_label[b]:
            row_label[a + 1:b] = row_label[a]
    found = []
    for k in range(len(TERMS)):
        for start, end in runs(np.flatnonzero(row_label == k)):
            if end - start >= 3:
                found.append((k, start + top, end + top))
    return found


def main(pdf=PDF):
    rgb = figure_image(pdf)
    height, width, _ = rgb.shape
    classes, grey = labels(rgb)
    rows = []
    for block, centres in zip(okura.BLOCKS, gridlines(grey, width)):
        slope, intercept = np.polyfit(centres, TICKS, 1)
        mm_per_px = abs(slope)
        zero = -intercept / slope
        top, bottom = int(centres[0]) - 5, int(centres[-1]) + 5
        plot_left = int(0.08 * width)
        coverage = (classes[top:bottom] >= 0).mean(axis=0)
        coverage[:plot_left] = 0
        bars = [(a, b) for a, b in runs(np.flatnonzero(coverage > 0.12)) if b - a > 20]
        if len(bars) != len(BARS):
            raise SystemExit(f"{block}: expected {len(BARS)} bars, found {len(bars)}: {bars}")
        for (year, month), (left, right) in zip(BARS, bars):
            values = dict.fromkeys(TERMS, 0.0)
            for k, start, end in segments(classes, top, bottom, left, right):
                size = (end - start + 1) * mm_per_px
                above = (start + end) / 2.0 < zero
                if TERMS[k] == "lateral" and above:
                    size = -size                     # drawn above zero: lateral inflow, negative Lf
                values[TERMS[k]] += size
            rows.append({"block": block, "year": year, "month": month,
                         "days": int((okura.season_dates(year).month == month).sum()), **values})

    table = pd.DataFrame(rows)
    table["balance"] = table["irrigation"] + table["rainfall"] - table["et"] - table["discharge"] - table["lateral"]
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))

    print("\nrainfall and ET against Table 2 (mm/d)")
    for year in (2021, 2022):
        for month in okura.MONTHS:
            days = int((okura.season_dates(year).month == month).sum())
            got = table[(table.year == year) & (table.month == month)]
            print(f"  {year}-{month:02d}  rain {got.rainfall.mean():5.1f} vs {okura.TABLE2_MM[year]['rain'][month] / days:5.1f}"
                  f"   ET {got.et.mean():4.1f} vs {okura.TABLE2_MM[year]['et'][month] / days:4.1f}")

    print("\nseason means weighted by days, against Table 4 (mm/d)")
    for (block, year), got in table.groupby(["block", "year"]):
        t4 = okura.TABLE4_MM[(block, year)]
        weights = got["days"] / got["days"].sum()
        line = "  ".join(f"{term} {float((got[term] * weights).sum()):5.1f} vs {t4[key] / okura.SEASON_DAYS:5.1f}"
                         for term, key in (("irrigation", "irrigation"), ("discharge", "drainage"),
                                           ("lateral", "lateral"), ("et", "et")))
        print(f"  {block} {year}: {line}")

    path = os.path.join(DATA, "observed", "okura_fig6_monthly.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False)
    print(f"\nwritten {path}")
    return table


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PDF)
