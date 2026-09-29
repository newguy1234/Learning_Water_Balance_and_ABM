# -*- coding: utf-8 -*-
"""CLI: python -m abm ui|run|compare|physics"""

from __future__ import annotations

import argparse
import json


def cmd_physics(_args):
    from cyclic.bundle import has_bundled_physics, physics_as_namespace, bundled_path

    if not has_bundled_physics():
        print(f"missing bundle at {bundled_path()}")
        print("run: python -m cyclic.bundle rebuild")
        return 1
    phys = physics_as_namespace()
    print(f"bundle: {bundled_path()}")
    print(f"capture={phys.capture:.4f} neighbour={phys.neighbour:.4f} stay={phys.stay:.4f}")
    print(f"years={phys.years}")
    pins = phys.efficiency_pins.get("2022", {})
    if pins:
        print(f"paper pins 2022: LUE={pins.get('LUE')} RUE_all={pins.get('RUE_all')}")
    return 0


def cmd_run(args):
    from .simulate import run_season

    world, eff = run_season(
        agent_name=args.agents,
        year=args.year,
        days=args.days,
        seed=args.seed,
        paddies_per_farmer=args.per_farmer or None,
    )
    print(json.dumps(eff, indent=2))
    if args.csv:
        from .metrics import summarise_run

        summarise_run(world, args.agents).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    return 0


def cmd_compare(args):
    from .simulate import compare_agents

    names = [n.strip() for n in args.agents.split(",") if n.strip()]
    table = compare_agents(
        names=names or None,
        year=args.year,
        days=args.days,
    )
    cols = [
        "agent",
        "WUE_prime_pct",
        "RUE_all",
        "LUE",
        "capture",
        "paper_LUE",
        "paper_RUE_all",
    ]
    print(table[[c for c in cols if c in table.columns]].to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    return 0


def cmd_ui(args):
    from .ui import serve

    return serve(port=args.port, open_browser=not args.no_browser) or 0


def cmd_mesa_ui(args):
    from .mesa_viz import serve

    return serve(port=args.port, open_browser=not args.no_browser) or 0


def cmd_agents(_args):
    from .agents import list_agents

    for name in list_agents():
        print(name)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="abm", description="ABM on bundled cyclic physics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("physics", help="show bundled capture / paper efficiency pins")
    p.set_defaults(func=cmd_physics)

    p = sub.add_parser("agents", help="list registered farmer agents")
    p.set_defaults(func=cmd_agents)

    p = sub.add_parser("ui", help="launch the field UI (Solara)")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("mesa-ui", help="launch Mesa's own SolaraViz shell for the same model")
    p.add_argument("--port", type=int, default=8775)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_mesa_ui)

    p = sub.add_parser("run", help="run one season with a simple agent")
    p.add_argument("--agents", default="steward")
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--days", type=int, default=131)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-farmer", type=int, default=0, help="multi-farmer ownership size")
    p.add_argument("--csv", default="")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="compare simple agents on WUE'/RUE/LUE")
    p.add_argument("--agents", default="maximiser,survivalist,steward,guardian,adaptive_experimenter,passive_operator,opportunist")
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--days", type=int, default=131)
    p.add_argument("--csv", default="")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
