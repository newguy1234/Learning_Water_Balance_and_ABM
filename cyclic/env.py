"""The environment: the Shiroyama-Jimbee district as a Gymnasium loop.

The intended benchmark is one gym with two (or three) plugged-in pieces. Today's
pool irrigation is a known block total (`irrigation_sent_mm`). A **farmer model**
(`@register_forecaster`) outputs surface drainage (and may propose a per-paddy IR
split, renormalised to Okura). A separate **equation builder** (`@register_lateral`)
learns leftover hops from neighbour water and neighbour ET. An optional **ratio
agent** (`@register_ratio`) supplies leftover simplex shares. FAO-56 then says how
much the crop got; leftover shares say how much underground drains recaptured as
LFV. Register algorithms in example.py.

Per-role **observation channels** can hide or show variables (POMDP / ablation):
farmer vs equation learner need not see the same fields (e.g. hide satellite ET
from the hop learner to block shortcuts).

    import cyclic

    env = cyclic.CyclicEnv(
        given_irrigation=True,
        farmer_obs="farmer",          # weather + stock + forcing (+ satellite)
        equation_obs="equation",      # neighbour heads/ET; no satellite by default
    )
    obs_farmer = env.observation("farmer")
    obs_eq = env.observation("equation")

    import cyclic

    env = cyclic.CyclicEnv(given_irrigation=True)   # hydrology gym, 2022
    observation, info = env.reset()
    agent.reset(env.model, range(env.model.n))      # if the policy needs the model
    while True:
        action = agent.decide_batch(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.learn(observation, *cyclic.env.CyclicEnv.unpack(action)[:2], reward, info)
        if terminated or truncated:
            break
        observation = next_observation

`given_irrigation=False` is the older loop where IR is also an action (and a leaked
proxy target until Okura daily files are loaded).

OBSERVATION  ObservationBatch per paddy, including yesterday's Open-Meteo, last
             Sentinel-2 NDVI / Kc×ET0, pond and soil stores, irrigation_sent_mm
             (today's Okura IR at the point), last_discharge_mm / last_et_mm /
             last_lfv_mm (yesterday's fate), neighbour_depth_mm / neighbour_et_mm
             (mean neighbour ponding and FAO-56 ET), and that paddy's up / down /
             left / right neighbour water and ET. observation.as_dict() is plain arrays.

ACTION       discharge_mm, optionally irrigation_mm as a disaggregation of the known
             Okura block total, optionally leftover shares (kx_ratio, ...). A 2-tuple
             (irrigation_mm, discharge_mm) still works.

REWARD       exp(-miss) in (0, 1], weighted. Gym defaults (given_irrigation=True):
             irrigation 0, drainage / et / lateral / lfv 1. Crop-got is FAO-56 vs
             satellite Kc×ET0; sent-back is D vs Okura ADCP; lateral is leftover vs
             Table 4 Lf; lfv is same-day recapture vs Table 5 LFV_pump / n_days.

    cyclic.rollout(agent, given_irrigation=True)
    cyclic.holdout(agent)                 train year then the other year
    cyclic.save_agent(agent, name)
    cyclic.load_agent(name)
"""

from __future__ import annotations

import os
import pickle
from types import SimpleNamespace

import numpy as np

from . import data, flyover, imitation, okura
from .agents import (
    RATIO_SHARE_KEYS, DEFAULT_ROLE_CHANNELS, OBS_CHANNEL_PACKS,
    make_agent, resolve_observation_channels,
)
from .model import CyclicModel, Results

BLOCKS = okura.BLOCKS
AGENT_DIR = os.path.join(data.PROJECT, "outputs", "agents")
# Open loop: IR is also an action. Gym loop zeros irrigation and scores LFV.
DEFAULT_WEIGHTS = {"irrigation": 1.0, "drainage": 1.0, "et": 1.0, "lateral": 1.0,
                   "lfv": 0.0, "need": 0.0}
GYM_WEIGHTS = {"irrigation": 0.0, "drainage": 1.0, "et": 1.0, "lateral": 1.0,
               "lfv": 1.0, "need": 0.0}


class CyclicEnv:
    """One season of the district, acted in from outside, a day per step.

    Observation masking (hide/show) is per role so a farmer learner and an
    equation learner can be given different information sets::

        env = CyclicEnv(given_irrigation=True,
                        farmer_obs="farmer",
                        equation_obs="equation",
                        ratio_obs="ratio")
        env.observation("farmer")     # masked batch for the forecaster
        env.observation("equation")   # masked batch for the hop learner
        env.set_obs_channels("farmer", hide=("satellite",))
    """

    def __init__(self, year=okura.COMPARISON_YEAR, observed=None, physics=None, forcing="open_meteo",
                 record_paddies=False,
                 reward_weights=None, pass_weight=2.0, potential="pixel",
                 given_irrigation=False, leftover_shares=None,
                 farmer_obs="farmer", equation_obs="equation", ratio_obs="ratio",
                 obs_channels=None):
        self.year, self.forcing = year, forcing
        from .physics import Physics
        self.physics = physics if physics is not None else Physics()
        self.potential = potential
        self.record_paddies = record_paddies
        self.given_irrigation = bool(given_irrigation)
        if leftover_shares is None and self.given_irrigation:
            leftover_shares = "tables"
        self.leftover_shares = leftover_shares
        base = GYM_WEIGHTS if self.given_irrigation else DEFAULT_WEIGHTS
        self.weights = {**base, **(reward_weights or {})}
        self.pass_weight = float(pass_weight)
        self.observed = observed or imitation.load_observed(year)
        dates = okura.season_dates(year)
        self.season_days = len(dates)
        self.expert, self.scale = {}, {}
        lfv_season = okura.TABLE5_1000M3[year]["LFV_pump"] * 1000.0
        daily_lfv = lfv_season / max(len(dates), 1)
        self.scale[("district", "lfv")] = max(daily_lfv, 1.0)
        self._lfv_daily_m3 = daily_lfv
        for block in BLOCKS:
            series = imitation.block_series(self.observed[0], block, dates)
            self.expert[block] = series
            self.scale[(block, "irrigation")] = max(float(np.mean(series["irrigation_m3"])), 1.0)
            self.scale[(block, "drainage")] = max(float(np.mean(series["drainage_m3"])), 1.0)
            lat_season = okura.TABLE4_MM[(block, year)]["lateral"] * okura.cyclic_area_m2(block) / 1000.0
            daily_lat = lat_season / max(len(dates), 1)
            series["lateral_m3"] = np.full(len(dates), daily_lat)
            self.scale[(block, "lateral")] = max(daily_lat, 1.0)
        self.model = None
        # Per-role observation channel config (POMDP / ablation).
        self.obs_channels = {
            "farmer": farmer_obs,
            "equation": equation_obs,
            "ratio": ratio_obs,
            "full": "full",
        }
        if obs_channels:
            self.obs_channels.update(obs_channels)
        self._obs_hide = {role: () for role in self.obs_channels}
        self._obs_show = {role: () for role in self.obs_channels}

    @property
    def done(self):
        return self.model is not None and not self.model.running

    @property
    def n_paddies(self):
        return len(data.season_inputs(self.year, self.forcing, self.potential).paddy_id)

    def reset(self, seed=None):
        """Start a season. Returns (first observation, info)."""
        self.model = CyclicModel(self.year, agents="rainfed", physics=self.physics, forcing=self.forcing,
                                 record_paddies=self.record_paddies, seed=0 if seed is None else seed,
                                 potential=self.potential)
        inputs = self.model.inputs
        self._masks = {block: inputs.block == block for block in BLOCKS}
        self._cyclic_weights = {block: np.where((inputs.block == block) & inputs.cyclic, inputs.area_m2, 0.0)
                                for block in BLOCKS}
        self._area_share = inputs.area_m2 / inputs.area_m2.sum()
        self._flyover = []
        self._et = {"miss": np.zeros(self.model.n), "target": np.zeros(self.model.n),
                    "pass_miss": np.zeros(self.model.n), "pass_target": np.zeros(self.model.n)}
        info = {"day": 0, "date": inputs.dates[0], "n_paddies": self.model.n,
                "season_days": self.season_days, "demonstration": self.observed[2],
                "given_irrigation": self.given_irrigation,
                "obs_roles": sorted(self.obs_channels.keys()),
                "obs_packs": sorted(OBS_CHANNEL_PACKS.keys())}
        self._install_leftover_shares(self.leftover_shares)
        return self.observation("farmer"), info

    def set_obs_channels(self, role, channels=None, hide=None, show=None):
        """Configure which observation packs/fields role sees (hide/show overlays)."""
        role = str(role)
        if channels is not None:
            self.obs_channels[role] = channels
        if hide is not None:
            self._obs_hide[role] = tuple(hide)
        if show is not None:
            self._obs_show[role] = tuple(show)
        if role not in self._obs_hide:
            self._obs_hide[role] = ()
        if role not in self._obs_show:
            self._obs_show[role] = ()
        return resolve_observation_channels(
            self.obs_channels.get(role, "full"),
            hide=self._obs_hide.get(role),
            show=self._obs_show.get(role),
        )

    def visible_fields(self, role="farmer"):
        """Frozen set of ObservationBatch field names currently visible to `role`."""
        return resolve_observation_channels(
            self.obs_channels.get(role, "full"),
            hide=self._obs_hide.get(role),
            show=self._obs_show.get(role),
        )

    def observation(self, role="full"):
        """Full or role-masked observation batch for this morning.

        role='full' returns every channel. role='farmer' / 'equation' / 'ratio'
        (or any key in obs_channels) zeros hidden fields so learners cannot peek.
        """
        batch = self.model._observe()
        sent = self._spread_okura_irrigation(self.model.day)
        batch.irrigation_sent_mm = sent
        if role is None or role == "full":
            return batch
        return batch.masked(
            channels=self.obs_channels.get(role, role),
            hide=self._obs_hide.get(role),
            show=self._obs_show.get(role),
        )

    def _spread_okura_irrigation(self, day):
        """Okura block pump volume, uniform millimetres on that block's cyclic paddies."""
        m = self.model
        out = np.zeros(m.n)
        day = int(np.clip(day, 0, self.season_days - 1))
        for block in BLOCKS:
            mask = self._masks[block] & m.inputs.cyclic
            area = float(m.area[mask].sum()) or 1.0
            vol = float(self.expert[block]["irrigation_m3"][day])
            out[mask] = vol / area * 1000.0
        return out

    def _pin_irrigation_to_okura(self, day, proposed):
        """Keep Okura block totals; use `proposed` only as a spatial disaggregation."""
        uniform = self._spread_okura_irrigation(day)
        proposed = np.broadcast_to(np.asarray(proposed, dtype=float), (self.model.n,)).copy()
        if not np.any(np.nan_to_num(proposed) > 0):
            return uniform
        out = np.zeros(self.model.n)
        for block in BLOCKS:
            mask = self._masks[block] & self.model.inputs.cyclic
            want = float(self.expert[block]["irrigation_m3"][day])
            have = float((np.maximum(proposed[mask], 0.0) * self.model.area[mask]).sum() / 1000.0)
            if have <= 1e-12:
                out[mask] = uniform[mask]
            else:
                out[mask] = np.maximum(proposed[mask], 0.0) * (want / have)
        return out

    def _install_leftover_shares(self, shares):
        if shares is None or self.model is None:
            return
        from .canal import pixel_ratio_fields
        from .identify import construct_ratios, load_best_ratios
        if isinstance(shares, str):
            if shares in ("tables", "table"):
                shares = construct_ratios(self.year)
            elif shares in ("fitted", "best"):
                shares = load_best_ratios(self.year, "season") or construct_ratios(self.year)
            else:
                raise ValueError(f"unknown leftover_shares {shares!r}")
        if not isinstance(shares, dict):
            return
        capture = shares.get("pool_capture_share",
                             float(np.mean(np.asarray(shares.get("kx_ratio", 0.0)))
                                   + np.mean(np.asarray(shares.get("ky_ratio", 0.0)))))
        farm = shares.get("farm_to_main", 1.0)
        m, p = self.model, self.model.physics
        tew, taw = p.total_evaporable_mm, p.total_available_mm
        top = tew / max(tew + taw, 1e-9)
        m.pixel_shares = pixel_ratio_fields(
            m.n, m.inputs.canal_distance_m, getattr(m, "paddy_edges", None),
            m.inputs.cyclic, float(np.mean(np.asarray(capture))), float(np.mean(np.asarray(farm))),
            top, p.canal_length_m, area_m2=m.area)
        for key in RATIO_SHARE_KEYS:
            if key in shares and np.size(shares[key]) == m.n:
                m.pixel_shares[key] = np.asarray(shares[key], dtype=float)

    @staticmethod
    def unpack(action):
        shares = None
        if isinstance(action, dict):
            irrigation = action.get("irrigation_mm", 0.0)
            discharge = action.get("discharge_mm", 0.0)
            if any(key in action for key in RATIO_SHARE_KEYS):
                shares = {key: action[key] for key in RATIO_SHARE_KEYS if key in action}
            return irrigation, discharge, shares
        if isinstance(action, (tuple, list)) and len(action) >= 2:
            shares = action[2] if len(action) > 2 else None
            return action[0], action[1], shares
        array = np.asarray(action, dtype=float)
        if array.ndim == 2 and array.shape[1] == 2:
            return array[:, 0], array[:, 1], None
        raise ValueError("an action is (irrigation_mm, discharge_mm), an (n, 2) array, "
                         "or a dict with discharge_mm (and optional irrigation_mm / shares)")

    def step(self, action):
        """Apply today's action. Returns (observation, reward, terminated, truncated, info);
        the observation is None once the season has ended."""
        if self.model is None or self.done:
            raise RuntimeError("the season is over or has not started: call reset()")
        irrigation, discharge, shares = self.unpack(action)
        m, inputs = self.model, self.model.inputs
        day = m.day
        if self.given_irrigation:
            irrigation = self._pin_irrigation_to_okura(day, irrigation)
        if shares:
            self._install_leftover_shares(shares)
        m.advance(irrigation, discharge)
        w = self.weights

        blocks = {}
        paddy_irrigation = np.zeros(m.n)
        paddy_drainage = np.zeros(m.n)
        paddy_lateral = np.zeros(m.n)
        leftover = np.broadcast_to(
            np.asarray(m.last_flux.get("leftover_mm", m.last_flux.get("lateral_mm", 0.0)), dtype=float),
            (m.n,))
        for block in BLOCKS:
            prefix = f"{block}|cyclic|"
            area = m.today[prefix + "area_m2"]
            volume = m.today[prefix + "irrigation_m3"]
            returned = m.today[prefix + "drainage_m3"]
            observed_volume = float(self.expert[block]["irrigation_m3"][day])
            observed_returned = float(self.expert[block]["drainage_m3"][day])
            r_irrigation = float(np.exp(-abs(volume - observed_volume) / self.scale[(block, "irrigation")]))
            r_drainage = float(np.exp(-abs(returned - observed_returned) / self.scale[(block, "drainage")]))
            weights = self._cyclic_weights[block]
            sim_lateral = float(leftover @ weights) / 1000.0
            observed_lateral = float(self.expert[block]["lateral_m3"][day])
            r_lateral = float(np.exp(-abs(sim_lateral - observed_lateral) / self.scale[(block, "lateral")]))
            w_block = w["irrigation"] + w["drainage"] + w.get("lateral", 0.0)
            blocks[block] = {
                "area_m2": area,
                "irrigation_m3": volume, "observed_irrigation_m3": observed_volume,
                "irrigation_mm": volume / area * 1000.0, "observed_irrigation_mm": observed_volume / area * 1000.0,
                "drainage_m3": returned, "observed_drainage_m3": observed_returned,
                "lateral_m3": sim_lateral, "observed_lateral_m3": observed_lateral,
                "et_mm": m.today[prefix + "et_m3"] / area * 1000.0,
                "et_target_mm": float(inputs.et_target_mm[day] @ weights) / weights.sum(),
                "irrigation_reward": r_irrigation, "drainage_reward": r_drainage,
                "lateral_reward": r_lateral,
                "reward": (w["irrigation"] * r_irrigation + w["drainage"] * r_drainage
                           + w.get("lateral", 0.0) * r_lateral) / max(w_block, 1e-12),
            }
            paddy_irrigation[self._masks[block]] = r_irrigation
            paddy_drainage[self._masks[block]] = r_drainage
            paddy_lateral[self._masks[block]] = r_lateral

        sim_lfv = float(m.station.today.get("LFV_pump_m3", 0.0))
        r_lfv = float(np.exp(-abs(sim_lfv - self._lfv_daily_m3) / self.scale[("district", "lfv")]))
        paddy_lfv = np.full(m.n, r_lfv)

        simulated = m.last_flux["et_mm"]
        target = inputs.et_target_mm[day].astype(float)
        is_pass = inputs.is_pass[day]
        miss = np.abs(simulated - target)
        r_et = np.exp(-miss / np.maximum(target, 0.5))
        self._et["miss"] += miss
        self._et["target"] += target
        self._et["pass_miss"] += miss * is_pass
        self._et["pass_target"] += target * is_pass

        et_weight = w["et"] * np.where(is_pass, self.pass_weight, 1.0)
        need = np.asarray(m.last_flux.get("need_mm", 0.0), dtype=float)
        r_need = np.exp(-need / np.maximum(target, 0.5))
        need_weight = w.get("need", 0.0)
        lat_weight = w.get("lateral", 0.0)
        lfv_weight = w.get("lfv", 0.0)
        denom = np.maximum(
            w["irrigation"] + w["drainage"] + et_weight + need_weight + lat_weight + lfv_weight, 1e-12)
        paddy_reward = ((w["irrigation"] * paddy_irrigation + w["drainage"] * paddy_drainage
                         + et_weight * r_et + need_weight * r_need + lat_weight * paddy_lateral
                         + lfv_weight * paddy_lfv)
                        / denom)
        reward = float(paddy_reward @ self._area_share)

        lfv_mm = np.broadcast_to(
            np.asarray(m.last_flux.get("lfv_mm", 0.0), dtype=float), (m.n,))
        sent_back = np.broadcast_to(
            np.asarray(m.last_flux.get("discharge_mm", m.last_flux.get("drainage_mm", 0.0)), dtype=float),
            (m.n,))
        passes_today = flyover.score_day(inputs, day, simulated)
        self._flyover.extend(passes_today)
        info = {"day": day, "date": inputs.dates[day], "blocks": blocks, "paddy_reward": paddy_reward,
                "flyover": passes_today,
                "et": {"simulated_mm": simulated, "target_mm": target, "is_pass": is_pass, "reward": r_et,
                       "training_weight": w["et"], "independent_observation": False},
                "lateral": {"training_weight": w.get("lateral", 0.0)},
                "lfv": {"m3": sim_lfv, "observed_m3": self._lfv_daily_m3, "reward": r_lfv,
                        "training_weight": lfv_weight},
                "fate": {"irrigation_sent_mm": np.asarray(irrigation, dtype=float),
                         "got_mm": simulated, "sent_back_mm": sent_back,
                         "lateral_mm": leftover, "lfv_mm": lfv_mm},
                "need": {"mm": need, "reward": r_need, "stock_mm": m.state.need.copy()},
                "station": dict(m.station.today),
                "given_irrigation": self.given_irrigation}
        terminated = self.done
        if terminated:
            results = Results(m)
            score = imitation.score(results.blocks, self.observed[0])
            share = self._area_share
            per_paddy = np.clip(1.0 - self._et["miss"] / np.maximum(self._et["target"], 1e-9), 0.0, 1.0)
            has_pass = self._et["pass_target"] > 0
            per_paddy_pass = np.where(
                has_pass, np.clip(1.0 - self._et["pass_miss"] / np.maximum(self._et["pass_target"], 1e-9), 0.0, 1.0),
                np.nan)
            score.flyover = flyover.summary(self._flyover)
            score.et_flyover_fidelity = score.flyover.fidelity
            score.et_fidelity = float(per_paddy @ share)
            score.et_pass_fidelity = float(np.nansum(per_paddy_pass * share) / share[has_pass].sum()) \
                if has_pass.any() else float("nan")
            info["results"] = results
            info["score"] = score
            info["et_score"] = {"fidelity": per_paddy, "pass_fidelity": per_paddy_pass}
            if m.record_paddies and getattr(m, "save_csv", True):
                results.write_csv()
        if terminated:
            return None, reward, terminated, False, info
        # Default next-obs is farmer-masked; full + equation views in info for multi-agent.
        next_obs = self.observation("farmer")
        info["obs_farmer"] = next_obs
        info["obs_equation"] = self.observation("equation")
        info["obs_ratio"] = self.observation("ratio")
        info["obs_full"] = self.observation("full")
        return next_obs, reward, terminated, False, info


# Kept so older scripts keep working.
ImitationEnv = CyclicEnv


def rollout(agent, year=okura.COMPARISON_YEAR, observed=None, physics=None, record_paddies=True,
            learn=False, **env_options):
    """One season with the agent acting. Returns mean_reward, rewards, score, results.

    `given_irrigation=True` is the water-fate gym (Okura IR known; score D, ET, Lf, LFV).
    Set learn=True to call agent.learn after each day (historical coupling).
    """
    agent = make_agent(agent)
    env = CyclicEnv(year, observed, physics, record_paddies=record_paddies, **env_options)
    observation, info = env.reset()
    if hasattr(agent, "reset"):
        agent.reset(env.model, np.arange(env.model.n))
    rewards = []
    while True:
        morning = observation
        action = agent.decide_batch(morning)
        observation, reward, terminated, truncated, info = env.step(action)
        if learn and hasattr(agent, "learn"):
            irrigation, discharge, _shares = CyclicEnv.unpack(action)
            agent.learn(morning, irrigation, discharge, reward, info)
        rewards.append(reward)
        if terminated or truncated:
            break
    if learn and hasattr(agent, "end_episode"):
        agent.end_episode(info.get("score"), info)
    return SimpleNamespace(mean_reward=float(np.mean(rewards)), rewards=rewards,
                           score=info["score"], results=info["results"],
                           given_irrigation=env.given_irrigation)


def holdout(agent, train_year=2021, eval_year=None, learn=False, **env_options):
    """Run one year as history, then the other year with the same algorithm.

    Default eval year is the other of 2021/2022. Gym defaults on: given_irrigation=True.
    History never updates the evaluation snapshot unless learn=True on the training year.
    """
    from .validation import split
    import copy

    env_options = dict(env_options)
    env_options.setdefault("given_irrigation", True)
    train_year, eval_year = split(train_year, eval_year)
    policy = make_agent(agent)
    trained = rollout(policy, year=train_year, learn=learn, **env_options)
    frozen = copy.deepcopy(policy)
    if learn:
        frozen.training_years = tuple(sorted(set(getattr(frozen, "training_years", ()) or ()) | {train_year}))
    evaluated = rollout(frozen, year=eval_year, learn=False, **env_options)
    return SimpleNamespace(train=trained, eval=evaluated, train_year=train_year, eval_year=eval_year)


# --- keeping trained agents ---------------------------------------------------------------

def save_agent(agent, name):
    """Pickle a trained agent to outputs/agents/<name>.pkl. Its class has to be importable
    where it is loaded -- define agents in a module (example.py), not in a __main__ script."""
    os.makedirs(AGENT_DIR, exist_ok=True)
    path = os.path.join(AGENT_DIR, f"{name}.pkl")
    with open(path, "wb") as handle:
        pickle.dump(agent, handle)
    return path


def saved_agents():
    """{name: path} for every agent saved with save_agent."""
    if not os.path.isdir(AGENT_DIR):
        return {}
    return {os.path.splitext(f)[0]: os.path.join(AGENT_DIR, f)
            for f in sorted(os.listdir(AGENT_DIR)) if f.endswith(".pkl")}


def load_agent(name):
    path = saved_agents().get(name, name)
    with open(path, "rb") as handle:
        agent = pickle.load(handle)
    if not hasattr(agent, "training_years"):
        agent.training_years = None
    return agent
