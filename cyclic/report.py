"""Shiroyama-Jimbee cyclic irrigation: a farmer policy run against Okura et al. (2025).

    python -m cyclic report [agent]     or register one in example.py and pass its name

  1. checks the paper's tables against each other
  2. runs the season with every paddy farmed by AGENT
  3. prints the model beside the paper (Tables 4, 5, 6, 7)
  4. scores irrigation and drainage per cyclic block against the observed series --
     interpolated from the paper for now; put the measured CSVs in data/observed/ and the
     same score uses them (format in cyclic/imitation.py)
  5. shows what sets WUE', RUE and LUE: the identities, and each parameter's elasticity
  6. writes outputs/ and plots

YOUR OWN FARMER. Write it in example.py next to this package and run  python -m cyclic report NAME.
ExampleFarmer below is a worked example; decide() sees one paddy's Observation:

    paddy_id, block, cyclic, area_m2, day, date,
    sat_ndvi, sat_et_mm, sat_days_ago    last Sentinel-2 pass over this plot
    rain_mm, et0_mm, wind_*, tmean_c, rh_*, ...  yesterday's Open-Meteo daily weather
    depth_mm                             standing water this morning

and returns (irrigation_mm, discharge_mm).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cyclic
from cyclic import efficiency, imitation, okura

YEAR = okura.COMPARISON_YEAR
AGENT = "et_follower"     # any registered name, or {"North": ..., "South": ...}
SENSITIVITY = True
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


@cyclic.register_agent("example_farmer")
class ExampleFarmer(cyclic.FarmerPolicy):
    """Example: let the plot fall to 15 mm, then send enough for the next three days."""

    defaults = dict(refill_below_mm=15.0, ceiling_mm=60.0)

    def decide(self, obs):
        expected_et = obs.sat_et_mm if np.isfinite(obs.sat_et_mm) else 1.1 * obs.et0_mm
        irrigate = 3 * (expected_et + 11.0) if obs.depth_mm < self.params["refill_below_mm"] else 0.0
        return irrigate, max(obs.depth_mm - self.params["ceiling_mm"], 0.0)


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def main(agent=None, year=None, sensitivity=None):
    global AGENT, YEAR, SENSITIVITY
    AGENT = AGENT if agent is None else agent
    YEAR = YEAR if year is None else int(year)
    SENSITIVITY = SENSITIVITY if sensitivity is None else sensitivity
    os.makedirs(OUT, exist_ok=True)

    section("1. Do Okura et al.'s tables agree with each other?")
    print(okura.check_tables(YEAR).to_string(index=False))

    section(f"2. Running {YEAR} with agent {AGENT!r}")
    model = cyclic.CyclicModel(YEAR, agents=AGENT)
    print(f"{model.n:,} paddies ({model.inputs.cyclic.sum():,} cyclic), {model.n_days} days, "
          f"satellite passes {[d.strftime('%m-%d') for d in model.inputs.pass_dates]}")
    results = model.run()
    print(f"largest per-plot water balance error: {model.balance_error_mm:.2e} mm")

    observed_blocks, observed_station, source = imitation.load_observed(YEAR)

    section("3. Model beside the paper")
    print(efficiency.compare(results.blocks, results.station, YEAR).to_string(index=False))

    section(f"4. Irrigation and drainage per cyclic block vs observed ({source})")
    scored = imitation.score(results.blocks, observed_blocks)
    print(scored.table.to_string(index=False))
    print(f"fidelity {scored.fidelity:.3f}   loss {scored.loss:.4f}")

    section("5a. What sets the efficiencies: the identities, model and paper-interpolated")
    print("model:")
    print(efficiency.identities(results.blocks, results.station, model.physics).to_string())
    print("\nobserved series:")
    print(efficiency.identities(observed_blocks, observed_station).to_string())

    if SENSITIVITY and AGENT == "et_follower":
        section("5b. Elasticities: % change in each metric per % change in each parameter")
        table, _ = efficiency.sensitivity(
            YEAR, agent_names=("target_depth_mm", "seepage_allowance_mm", "ceiling_mm", "final_drain_day"),
            physics_names=("plow_pan_k_mm_d", "outlet_loss_per_day", "pool_capture_share"))
        print(table.to_string())

    section("6. Per paddy")
    paddies = results.paddy_season()
    print(paddies.groupby(["block", "cyclic"])[
        ["area_ha", "irrigation_mm", "farmer_discharge_mm", "passive_drainage_mm", "lateral_mm",
         "et_mm", "wue_prime_pct", "mean_depth_mm"]].agg(["mean", "std"]).T.to_string())

    folder = results.write_csv(os.path.join(OUT, f"run_{YEAR}_{AGENT}"))
    print(f"\nwritten to {folder}")

    plot(results, observed_blocks, paddies, source)


def plot(results, observed_blocks, paddies, source):
    sim, obs = results.blocks, observed_blocks
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for col, block in enumerate(okura.BLOCKS):
        s = sim[sim["block"] == block]
        o = obs[obs["block"] == block]
        for row, q in enumerate(("irrigation", "drainage")):
            ax = axes[row, col]
            ax.plot(o["date"], o[f"{q}_m3"] / o["area_m2"] * 1000, color="black", lw=1.2, label="observed")
            ax.plot(s["date"], s[f"{q}_mm"], color="tab:blue", lw=1.2, label="model")
            if q == "drainage":
                ax.plot(s["date"], s["farmer_discharge_mm"], color="tab:orange", lw=0.8,
                        label="of which farmers' discharge")
            ax.set_title(f"{block} cyclic block -- {q} (mm/d)")
            ax.legend(fontsize=8)
    fig.suptitle(f"Model vs observed ({source})")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"blocks_{results.year}.png"), dpi=130)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    for ax, column, title in zip(axes, ("irrigation_mm", "drainage_mm", "wue_prime_pct"),
                                 ("season irrigation (mm)", "season drainage (mm)", "WUE' per paddy (%)")):
        points = ax.scatter(paddies["lon"], paddies["lat"], c=paddies[column], s=3, cmap="viridis")
        cyclic_edge = paddies[paddies["cyclic"]]
        ax.scatter(cyclic_edge["lon"], cyclic_edge["lat"], s=0.2, c="red", alpha=0.3)
        ax.set_title(title + "  (red dots: cyclic)")
        ax.set_aspect(1 / 0.81)
        fig.colorbar(points, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"paddies_{results.year}.png"), dpi=130)

    months = efficiency.monthly_wue_prime(sim)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, block in zip(axes, okura.BLOCKS):
        x = np.arange(len(okura.MONTHS))
        paper = [okura.TABLE7_WUE_PCT[(block, results.year)][m] for m in okura.MONTHS]
        ax.bar(x - 0.2, paper, 0.4, label="paper (Table 7)", color="black")
        ax.bar(x + 0.2, [months.loc[m, block] for m in okura.MONTHS], 0.4, label="model", color="tab:blue")
        ax.set_xticks(x, list(okura.MONTHS.values()))
        ax.set_title(f"WUE' {block} block (%)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"wue_{results.year}.png"), dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
