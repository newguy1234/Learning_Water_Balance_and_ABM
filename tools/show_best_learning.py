"""Render a checkpoint's training selection and held-out water/ET fits."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclic.imitation import load_observed


def render(directory):
    directory = Path(directory)
    summary = json.loads((directory / "evaluation.json").read_text())
    history = pd.read_csv(directory / "history.csv")
    sim = pd.read_csv(directory / "evaluation_blocks.csv", parse_dates=["date"])
    obs = load_observed(summary["evaluation_year"])[0]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    scores = " · ".join(f"{label} {summary[key]:.1%}" for label, key in (
        ("Irrigation", "irrigation_fidelity"), ("Discharge", "discharge_fidelity"), ("ET", "et_fidelity")))
    selected = "baseline" if summary['best_episode'] < 0 else f"episode {summary['best_episode'] + 1}"
    fig.suptitle(f"Best frozen snapshot: {selected} — train {summary['training_year']}, evaluate {summary['evaluation_year']}\n"
                 f"Held-out fidelity: {scores}", fontsize=15)
    for col, block in enumerate(("North", "South")):
        s = sim[sim.block == block]
        o = obs[obs.block == block]
        for row, (key, label) in enumerate((("irrigation_m3", "Irrigation (1,000 m³/day)"),
                                          ("drainage_m3", "Discharge (1,000 m³/day)"),
                                          ("et_mm", "ET (mm/day)"))):
            ax = axes[row, col]
            divisor = 1 if row == 2 else 1000
            ax.plot(s.date, s[key] / divisor, label="Selected policy", color="#176a91", lw=1.5)
            if row < 2:
                ax.plot(o.date, o[key] / divisor, label="Observed block series", color="#222222", lw=1.4)
            else:
                ax.plot(s.date, s.et_target_mm, label="Satellite-derived target (interpolated)", color="#222222", lw=1.4)
            ax.set_title(f"{block} — {label}", fontsize=11)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=25)
    fig.text(.06, .025,
             "Selection uses 2021 irrigation/discharge loss, including daily errors. 2022 is evaluated only after selection.\n"
             "ET fit remains high while discharge is weak; ET fidelity alone does not establish realistic water management.", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .92))
    path = directory / "best_so_far.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(path)
    return path


if __name__ == "__main__":
    render(sys.argv[1])
