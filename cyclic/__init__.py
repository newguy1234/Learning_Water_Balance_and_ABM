"""Cyclic irrigation: **inferred underground flow ratios** (Okura et al. 2025).

Contribution: with surface IR/D/W as **hard copies** of Fig. 8 / ADCP / tables
(not estimated schedules), infer per-point leftover fate underground
(capture Kx+Ky → LFV vs stay Kz+ground → deep). See ``cyclic.underground``.

    report = cyclic.infer_underground_ratios(2022)
    both = cyclic.underground_ratio_train_both()   # train K_i on 2021+2022
    hold = cyclic.underground_ratio_holdout()      # optional single-year freeze
    print(cyclic.contribution_text(hold))

Surface schedule and farmer discharge are supporting machinery, not the claim:

    @cyclic.register_forecaster("my_farmer")
    class MyFarmer(cyclic.FarmerPolicy):
        def decide_batch(self, obs):
            return irrigation_mm, discharge_mm   # IR ignored when given_irrigation=True

    @cyclic.register_ratio("my_ratios")
    class MyRatios(cyclic.RatioPolicy):
        def partition(self, leftover, obs, targets):
            return shares   # optional leftover Kx, Ky, Kz, ground

    @cyclic.register_lateral("my_hops")
    class MyHops(cyclic.LateralPolicy):
        def flux(self, stock, graph, area, physics, neighbour_et_mm=None):
            return after, delta   # sees neighbour water + neighbour FAO-56 ET; no soil K

    env = cyclic.CyclicEnv(given_irrigation=True)   # water-fate gym
    cyclic.rollout("persist", given_irrigation=True)
    cyclic.ratio_holdout("tables")                  # alias of underground freeze protocol
    cyclic.discover_lateral_equation()              # optional neighbour-head H

Copernicus Data Space (Sentinel-2 over Imbanuma)::

    pip install -e ".[copernicus]"
    python -m cyclic copernicus register   # save CDSE account locally
    python -m cyclic copernicus download 2022
"""

import os
import subprocess
import sys

from . import data, okura, physics
from .okura import COMPARISON_YEAR
from .agents import (AGENTS, ETFollower, ETMatch, FarmerPolicy, FieldAggregate, Flooded, FrozenStay, HeadEqualise, LATERALS, LateralPolicy, LinearFate, LinearHops, LinearPolicy, NonlinearHops, Observation,
                     ObservationBatch, PaddyAgent, PersistDischarge, PumpingStation, RATIOS, Rainfed, RatioFinder, RatioPolicy,
                     SatellitePhysics, TablesRatio, WaterFatePolicy, leftover_from_season, make_agent, make_lateral, make_ratio, point_targets,
                     register_agent, register_lateral, register_ratio)
from .agents import register_agent as register_forecaster
from .et_net import ETNet, SpillNet, train_et_net  # noqa: F401  -- registers et_net
from .model import CyclicModel, Results
from .identify import HydrologyProblem, construct_ratios, load_best_ratios, run_finder, spread_field_action, solve_classical, ratio_holdout, discover_pixel_shares, soil_flow_holdout, discover_lateral_equation, equation_from_cache, equation_cache_status, equation_score_table, equation_scope_summary, equation_figure, load_equation_report, pin_water_fate, pin_summary_text, simple_leftover_holdout, simple_leftover_table, relative_et_holdout, relative_et_table, load_paddy_ground_water_law, evaluate_ground_water_ratio
from .water_fate import configure_given_ir_d_water_fate, solve_penetration_k
from .underground import (CLAIM as UNDERGROUND_CLAIM, infer_underground_ratios,
                          infer_underground_ratios_joint, underground_ratio_train_both,
                          underground_ratio_holdout, contribution_text, write_contribution)
from .benchmark import farmer_track, ratio_track, two_way, write_benchmark
from . import efficiency, flyover, graph, imitation
from .imitation import LEARNERS, Cancelled, Learner, LearnResult, load_observed, register_learner, score
from .env import CyclicEnv, ImitationEnv, load_agent, rollout, save_agent, saved_agents, holdout, GYM_WEIGHTS
from .agents import (
    OBS_CHANNEL_PACKS, DEFAULT_ROLE_CHANNELS, resolve_observation_channels,
)
from .copernicus import (
    IMBANUMA_BBOX, access_token, clear_credentials, district_bbox,
    download_season as copernicus_download, load_credentials,
    register_interactive as copernicus_register,
    save_credentials, search_sentinel2 as copernicus_search,
    status as copernicus_status,
)

# Gymnasium registration (optional dependency).
try:
    from . import gym_env as _gym_env  # noqa: F401  -- registers on import
    from .gym_env import CyclicGym, ENV_ID, ENV_ID_OPEN, register_envs
except ImportError:  # pragma: no cover
    CyclicGym = None
    ENV_ID = "Cyclic-Imbanuma-v0"
    ENV_ID_OPEN = "Cyclic-Imbanuma-Open-v0"

    def register_envs(*_a, **_k):
        raise ImportError('Install gymnasium: pip install -e ".[gym]"')

FORECASTERS = AGENTS

PROJECT = data.PROJECT
SERVER_FLAG = "CYCLIC_SOLARA_SERVER"


def is_server():
    """True inside the process Solara serves from. Guard a script's __main__ with it, or
    serving that script would run it again inside the server."""
    return os.environ.get(SERVER_FLAG) == "1" or "solara" in sys.argv[0].replace("\\", "/").lower()


def free_port(port, tries=20):
    """The first port from `port` up that nothing is listening on."""
    import socket

    for candidate in range(port, port + tries):
        for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
            try:
                with socket.socket(family, socket.SOCK_STREAM) as probe:
                    probe.bind((host, candidate))
            except OSError as error:
                if family == socket.AF_INET6 and getattr(error, "errno", None) in (97, 10047):
                    continue                        # no IPv6 on this machine: IPv4 decides
                break
        else:
            return candidate
    raise OSError(f"no free port between {port} and {port + tries - 1}")


def page():
    """The interface's Page component, for `Page = cyclic.page()` in a file Solara serves."""
    from .app import Page

    return Page


def serve(script=None, port=8766, open_browser=True, reload=False):
    """Start the web interface and block until it is closed (Ctrl+C).

    `script` is a file of yours that imports your agents -- pass __file__ --
    which Solara imports in the server so they appear there. reload=False runs Solara with
    --production: its hot reloader treats the .venv inside this project as project code,
    re-imports numpy when any file changes, and crashes with "cannot load module more than
    once per process".
    """
    if is_server():
        return None
    wanted, port = port, free_port(port)
    if port != wanted:
        print(f"Port {wanted} is already in use -- probably an interface that is still running. "
              f"Using {port} instead.", flush=True)
    target = os.path.abspath(script) if script else os.path.join(os.path.dirname(__file__), "app.py")
    command = [sys.executable, "-m", "solara", "run", target, "--port", str(port)]
    if not open_browser:
        command.append("--no-open")
    if not reload:
        command.append("--production")
    print(f"Opening the interface at http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    try:
        return subprocess.call(command, cwd=PROJECT, env={**os.environ, SERVER_FLAG: "1"})
    except KeyboardInterrupt:
        return 0

from .paddy_export import export_paddy_dataset, load_paddy_dataset
