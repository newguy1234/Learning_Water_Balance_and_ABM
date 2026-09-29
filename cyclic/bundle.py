# -*- coding: utf-8 -*-
"""Ship and load a frozen physics bundle so ``pip install`` users get EMS baselines
without re-running discovery.

Bundle contents (JSON under ``cyclic/bundled/``):
  - district leftover ratios (joint 2021+2022 + per-year pins)
  - hop / leftover-fate equation summaries from ``outputs/paper/ems/equations.csv``
  - Okura efficiency pins (Table 6 WUE'/RUE/LUE context)
  - metadata (package version, years, claim)

Rebuild after experiments::

    python -m cyclic.bundle rebuild
"""

from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

from . import okura

_PACKAGE_DIR = Path(__file__).resolve().parent
BUNDLED_DIR = _PACKAGE_DIR / "bundled"
DEFAULT_NAME = "physics_bundle.json"
EMS_DIR = _PACKAGE_DIR.parent / "outputs" / "paper" / "ems"


def bundled_path(name: str = DEFAULT_NAME) -> Path:
    return BUNDLED_DIR / name


def has_bundled_physics(name: str = DEFAULT_NAME) -> bool:
    return bundled_path(name).is_file()


def load_physics_bundle(name: str = DEFAULT_NAME) -> dict:
    """Load the pip-shipped physics cache (or raise if missing)."""
    path = bundled_path(name)
    if not path.is_file():
        try:
            ref = resources.files("cyclic.bundled").joinpath(name)
            with resources.as_file(ref) as resolved:
                path = Path(resolved)
        except Exception as exc:  # pragma: no cover
            raise FileNotFoundError(
                f"No bundled physics at {bundled_path(name)}. "
                "Run `python -m cyclic.bundle rebuild` after ems_experiments."
            ) from exc
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def physics_as_namespace(bundle: dict | None = None) -> SimpleNamespace:
    """Convenience view: ``.capture``, ``.neighbour``, ``.hop_linear``, …"""
    blob = bundle or load_physics_bundle()
    ratios = blob.get("ratios", {})
    joint = ratios.get("joint", {})
    hops = blob.get("hop_laws", {})
    return SimpleNamespace(
        version=blob.get("version"),
        years=tuple(blob.get("years", (2021, 2022))),
        capture=float(joint.get("capture", 0.76)),
        neighbour=float(joint.get("neighbour", 0.12)),
        stay=float(joint.get("stay", 0.12)),
        capture_cv=float(joint.get("capture_cv", 0.0)),
        pins={int(k): v for k, v in ratios.get("pins", {}).items()},
        hop_linear=hops.get("linear_general"),
        hop_physics=hops.get("head_equalise"),
        efficiency_pins=blob.get("efficiency_pins", {}),
        claim=blob.get("claim", ""),
        raw=blob,
    )


def save_physics_bundle(bundle: dict, name: str = DEFAULT_NAME) -> Path:
    BUNDLED_DIR.mkdir(parents=True, exist_ok=True)
    path = bundled_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)
    return path


def _read_ems_csv(name: str):
    import pandas as pd

    path = EMS_DIR / name
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _is_finite(value) -> bool:
    try:
        import math

        return not (isinstance(value, float) and math.isnan(value))
    except Exception:
        return False


def build_physics_bundle() -> dict:
    """Assemble a bundle from EMS CSVs + Okura table pins (no heavy model run)."""
    from .identify import construct_ratios
    from .underground import CLAIM

    ratios_csv = _read_ems_csv("ratios.csv")
    equations_csv = _read_ems_csv("equations.csv")
    farmers_csv = _read_ems_csv("farmers.csv")
    gym_csv = _read_ems_csv("gym_rollouts.csv")

    joint = {"capture": 0.7629, "neighbour": 0.1186, "stay": 0.1186, "capture_cv": 0.085}
    pins = {}
    freeze = []
    if ratios_csv is not None:
        for _, row in ratios_csv.iterrows():
            proto = str(row.get("protocol", ""))
            if proto == "joint":
                joint = {
                    "capture": float(row["capture"]),
                    "neighbour": float(row.get("neighbour", 0.1186) or 0.1186),
                    "stay": float(row.get("stay", 0.1186) or 0.1186),
                    "capture_cv": float(row.get("capture_cv", 0.085) or 0.085),
                    "fidelity": float(row["fidelity"]),
                    "eval_year": int(row["eval"]),
                }
            elif proto == "closed_form":
                pins[str(int(row["eval"]))] = {
                    "capture": float(row["capture"]),
                    "pin": float(row["pin"]),
                    "fidelity": float(row["fidelity"]),
                }
            elif proto == "freeze":
                freeze.append({
                    "train": int(row["train"]),
                    "eval": int(row["eval"]),
                    "capture": float(row["capture"]),
                    "pin": float(row["pin"]),
                    "fidelity": float(row["fidelity"]),
                })

    table_shares = {str(y): construct_ratios(y) for y in (2021, 2022)}

    hop_laws = {}
    if equations_csv is not None:
        for _, row in equations_csv.iterrows():
            key = f"{row.get('equation', '')}::{row.get('form', '')}::{row.get('scope', '')}"
            n_params = row["n parameters"] if "n parameters" in row else None
            hop_laws[key] = {
                "equation": row.get("equation"),
                "form": row.get("form"),
                "scope": row.get("scope"),
                "n_parameters": int(n_params) if n_params is not None and _is_finite(n_params) else None,
                "rmse_2021": float(row["2021 RMSE mm"]) if "2021 RMSE mm" in row else None,
                "rmse_2022": float(row["2022 RMSE mm"]) if "2022 RMSE mm" in row else None,
                "law": row.get("law"),
            }
        for _, row in equations_csv.iterrows():
            full = f"{row.get('equation')}::{row.get('form')}::{row.get('scope')}"
            if row.get("form") == "linear" and row.get("scope") == "general":
                hop_laws["linear_general"] = hop_laws[full]
            if row.get("form") == "head_equalise":
                hop_laws["head_equalise"] = hop_laws[full]

    efficiency_pins = {
        str(y): {
            "LUE": okura.TABLE6[y]["LUE"],
            "RUE_all": okura.TABLE6[y]["RUE_all"],
            "RUE_North": okura.TABLE6[y]["RUE_North"],
            "RUE_South": okura.TABLE6[y]["RUE_South"],
            "SDR": okura.TABLE6[y]["SDR"],
            "lake_ratio": okura.TABLE6[y]["lake_ratio"],
        }
        for y in (2021, 2022)
    }

    baselines = {}
    if farmers_csv is not None:
        baselines["farmers"] = farmers_csv.to_dict(orient="records")
    if gym_csv is not None:
        baselines["gym_rollouts"] = gym_csv.to_dict(orient="records")

    if EMS_DIR.is_dir():
        dest = BUNDLED_DIR / "ems"
        dest.mkdir(parents=True, exist_ok=True)
        for name in (
            "ratios.csv",
            "equations.csv",
            "farmers.csv",
            "gym_rollouts.csv",
            "observation_roles.csv",
            "environment.json",
        ):
            src = EMS_DIR / name
            if src.is_file():
                shutil.copy2(src, dest / name)

    return {
        "version": "0.2.0",
        "years": [2021, 2022],
        "claim": CLAIM,
        "ratios": {
            "joint": joint,
            "pins": pins,
            "freeze": freeze,
            "table_shares": {
                y: {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in shares.items()
                }
                for y, shares in table_shares.items()
            },
        },
        "hop_laws": hop_laws,
        "efficiency_pins": efficiency_pins,
        "baselines": baselines,
        "notes": (
            "Frozen EMS paper baselines + Okura table pins. "
            "ABM environments load capture/neighbour/stay from ratios.joint; "
            "they do not need observed IR/D time series."
        ),
    }


def rebuild(name: str = DEFAULT_NAME) -> Path:
    bundle = build_physics_bundle()
    return save_physics_bundle(bundle, name=name)


def main(argv=None):
    import sys

    args = list(argv or sys.argv[1:])
    cmd = args[0] if args else "path"
    if cmd == "rebuild":
        path = rebuild()
        print(f"wrote {path}")
        return 0
    if cmd == "path":
        print(bundled_path())
        print("exists" if has_bundled_physics() else "missing")
        return 0
    if cmd == "show":
        phys = physics_as_namespace()
        print(
            f"capture={phys.capture:.4f} neighbour={phys.neighbour:.4f} stay={phys.stay:.4f}"
        )
        print(f"years={phys.years}")
        return 0
    print("usage: python -m cyclic.bundle [rebuild|path|show]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
