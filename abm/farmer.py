# -*- coding: utf-8 -*-
"""BaseFarmer — the one class every ABM farmer inherits from, inside or outside ``abm``.

A farmer owns paddies. Each day it asks the canal for water (``request``), takes what the
station actually sent (``apply`` ≤ sent), may discharge (``drain``) and sets its outlet
weir board (``weir``: pond above the crest spills out through the outlet on its own, see
``cyclic.physics.advance(weir_mm=...)``). ``BaseFarmer.act`` returns all four as a
``Plan`` after enforcing the hard constraints, whatever a subclass decided.

Common personality variables (``TRAITS``) live on every farmer as ``self.traits``. Each is
drawn once per farmer: the class's ``traits_mean`` (or the trait's default) plus seeded
noise of ``trait_variability`` × the trait's range, so no two farmers of one kind are
identical. A trait given explicitly as a param (``make_agent(name, part_time=.8)`` or in
a class's ``defaults``) is used exactly, without noise.

Default day of a BaseFarmer (every step is a method a subclass may override):

  crop calendar   crop_day = day + calendar_shift_days; eagerness ±1 → ±max_shift_days, so
                  the stage target, drain windows and the crop's own growth (canopy/ET in
                  the physics) run early or late for this farmer's paddies
  target          crop.target_depth_mm(crop_day) × target_multiplier + a risk buffer,
                  drifted softly toward the neighbours (``neighbour_target``)
  bounds          lower = target − drawdown·lower_gap_mm, upper = target + overfill·upper_gap_mm;
                  in the mid-season / harvest drain windows lower = 0 and upper = target
  checking        a farmer only looks at its paddies on check days: its days off always, and
                  on working days when at home (part_time = share of them away) and its
                  monitor interval has passed; never more than 7 days without a look.
                  Days off vary a little: Sat+Sun for most, a day earlier or later for
                  some (schedule_jitter), and some weeks one moves (week_swap)
  fill and drop   on a check day it refills to the upper bound only if the pond would
                  otherwise fall below the lower bound before its next check; the refill
                  stops short by the rain it trusts in the forecast (foresight = how many
                  days ahead, trust decaying with lead time); if one fill cannot last
                  until the next check it leaves the inlet trickling (standing inlet)
  flow-through    on hot days (ET0 ≥ heat_et0_mm) and while away, a flow_through farmer
                  leaves extra water running in; it passes over the weir back to the drain
  weir board      crest = upper bound (target in drain windows, bund during holds)
  pricing         with a volumetric price, water above the essential fill (overfill,
                  flow-through, standing slack) is cut by 1 / (1 + sensitivity·price/ref)
  hard rules      apply ≤ sent, drain ≤ stock, then every ``Constraint`` (e.g. no drainage
                  for 7 days after herbicide) — enforced after the subclass's decision

Subclass freely: set ``traits_mean`` / ``defaults``, override any hook (``stage_target_mm``,
``bounds_mm``, ``checks_today``, ``expected_rain_mm``, ``plan`` …), or add constraints::

    class MyFarmer(abm.BaseFarmer):
        traits_mean = dict(part_time=.9, flow_through=.6)
        constraints = abm.BaseFarmer.constraints + (abm.NoDrainHold("fertiliser", 40, 3),)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import crop


@dataclass(frozen=True)
class Trait:
    name: str
    low: float
    high: float
    default: float
    meaning: str


# Common personality variables every farmer carries (self.traits). Order is fixed: it
# sets the seeded draw, so appending a trait never changes an existing farmer's others.
TRAITS: tuple[Trait, ...] = (
    Trait("wealth", 0, 1, .5, "capital; softens the response to a water price"),
    Trait("experience", 0, 1, .5, "damps the reaction to a single bad outcome"),
    Trait("knowledge", 0, 1, .5, "counts seepage and today's rain into the refill"),
    Trait("labour", 0, 1, .5, "hands available; with work_ethic sets the monitor interval"),
    Trait("cooperation", 0, 1, .5, "levels water with neighbours; holds back when the canal is short"),
    Trait("work_ethic", 0, 1, .5, "with labour sets how many days between looks"),
    Trait("risk_tolerance", 0, 1, .5, "0 keeps a risk buffer above the stage target"),
    Trait("neighbour_offset", -1, 1, 0., "prefers to sit wetter (+) or drier (-) than the neighbours"),
    Trait("eagerness", -1, 1, 0., "+1 runs the crop calendar max_shift_days early, -1 as many late"),
    Trait("drawdown", 0, 1, .5, "how far below target the pond may fall before a refill (lower bound)"),
    Trait("overfill", 0, 1, .5, "how far above target a refill goes (upper bound, weir crest)"),
    Trait("part_time", 0, 1, 0., "share of working days away from the farm; days off always home"),
    Trait("foresight", 0, 1, .5, "how many days of rain forecast it acts on (× max_forecast_days)"),
    Trait("flow_through", 0, 1, .2, "leaves the inlet running (kakenagashi) on hot days and while away"),
    Trait("price_sensitivity", 0, 1, .5, "how much a volumetric water price trims discretionary water"),
)
TRAIT_NAMES = tuple(t.name for t in TRAITS)
TRAIT_RANGE = {t.name: (t.low, t.high) for t in TRAITS}

# Soft neighbour-relative target (BaseFarmer.neighbour_target). cooperation and
# neighbour_offset are traits; these are the shared tuning knobs.
NEIGHBOUR_DEFAULTS = dict(
    cooperation=.5,              # 0..1: 1 levels with the neighbours, 0 follows its own offset
    neighbour_offset=0.,         # -1..1: drier (-) or wetter (+) than the neighbours' mean
    neighbour_offset_mm=15.,     # offset ±1 -> ±15 mm relative to the neighbours
    neighbour_weight=.3,         # share of the way the target moves toward that level
    neighbour_max_shift_mm=20.,  # tanh cap on the pull before weighting
    neighbour_memory=.3,         # daily update rate of the remembered neighbour level
)

# Shared params (scenario assumptions, not fitted values). Any farmer may override them.
FARMER_DEFAULTS = dict(
    seed=0,
    trait_variability=.08,  # personality spread: noise sd as a share of each trait's range
    target_multiplier=1.,   # scales the stage target (learned by PersonalityFarmer)
    max_shift_days=4,       # eagerness ±1 -> crop calendar ± this many days
    lower_gap_mm=25.,       # drawdown 1 -> refill trigger this far below target
    upper_gap_mm=20.,       # overfill 1 -> refill level this far above target
    risk_buffer_mm=8.,      # risk_tolerance 0 -> target raised by this much
    max_forecast_days=4,    # foresight 1 -> acts on rain up to this many days ahead
    forecast_decay=.8,      # trust in rain forecast k days ahead = decay ** k
    heat_et0_mm=5.,         # ET0 at or above this is a hot day
    heat_flow_mm=20.,       # flow_through 1 -> extra mm/day run through on hot days
    away_slack=.3,          # flow_through 1 -> standing inlet 30% above the expected loss
    price_ref_per_m3=20.,   # price that halves a fully price-sensitive farmer's extras
    freeboard_mm=20.,       # weir crest never set above bund − freeboard
    schedule_jitter=.3,     # share of farmers whose days off sit a day off Sat+Sun
    week_swap=.15,          # chance a week moves one of its days off to another day
    scarcity_cut=.25,       # cooperation 1 trims its request 25% when the canal is short
    **{k: v for k, v in NEIGHBOUR_DEFAULTS.items() if k not in TRAIT_NAMES},
)


@dataclass
class Plan:
    """One farmer's day, per owned paddy (``ids``). mm over each paddy's area."""
    ids: np.ndarray
    apply: np.ndarray    # irrigation taken from what the station sent (≤ sent)
    drain: np.ndarray    # water let out on purpose, beyond the weir
    weir: np.ndarray     # outlet weir crest (mm); inf = no board spill
    request: np.ndarray  # what the farmer asked of the canal today (demand)
    notes: dict = field(default_factory=dict)


class Constraint:
    """A hard rule ``BaseFarmer.act`` enforces after the farmer's own decision.

    Subclass and override ``active`` / ``enforce``; ``crop_day`` is the farmer's own
    (calendar-shifted) crop day, ``pond`` its owned ponding."""

    name = "constraint"

    def active(self, farmer: "BaseFarmer", obs: dict, crop_day: int) -> bool:
        return False

    def enforce(self, farmer: "BaseFarmer", obs: dict, plan: Plan, pond: np.ndarray) -> None:
        pass


@dataclass
class NoDrainHold(Constraint):
    """No drainage for ``days`` from crop day ``start_day`` (e.g. herbicide labels in
    Japan require holding 3–5 cm for about 7 days with no overflow or drainage; panicle
    top-dressing is held a few days). The weir board goes up to the bund, explicit
    discharge is forbidden, and the pond is topped up to ``min_pond_mm`` if the canal
    allows."""

    name: str = "hold"
    start_day: int = 0
    days: int = 7
    min_pond_mm: float = 0.

    def active(self, farmer, obs, crop_day):
        return self.start_day <= crop_day < self.start_day + self.days

    def enforce(self, farmer, obs, plan, pond):
        plan.drain[:] = 0.
        plan.weir[:] = float(obs.get("bund_mm", np.inf))
        if self.min_pond_mm > 0:
            top = np.maximum(self.min_pond_mm - pond + farmer.daily_loss_mm(obs), 0.)
            plan.request = np.maximum(plan.request, top)
            plan.apply = np.maximum(plan.apply, top)
        plan.notes.setdefault("holds", []).append(self.name)


# Scenario assumptions: herbicide a few days after transplanting (7-day hold at >= 3 cm),
# panicle top-dressing just after the mid-season drain (4-day hold).
DEFAULT_CONSTRAINTS: tuple[Constraint, ...] = (
    NoDrainHold("herbicide", start_day=5, days=7, min_pond_mm=30.),
    NoDrainHold("topdressing", start_day=crop.MIDSUMMER_START + crop.MIDSUMMER_DAYS, days=4),
)


def in_drain_window(crop_day: int) -> bool:
    """Mid-season drainage or the final harvest drain, on the farmer's own calendar."""
    return (crop.MIDSUMMER_START <= crop_day < crop.MIDSUMMER_START + crop.MIDSUMMER_DAYS
            or crop_day >= crop.HARVEST_DRAIN_DAY)


class BaseFarmer:
    """Plot-level farmer: asks for water, applies ≤ what was sent, sets its weir board.

    See the module docstring for the default day. Every ABM farmer must inherit from this
    class (``abm.register_farmer`` refuses anything else)."""

    name = "base"
    defaults: dict = {}
    bounds: dict = {}
    traits_mean: dict = {}
    constraints: tuple[Constraint, ...] = DEFAULT_CONSTRAINTS

    def __init__(self, **kwargs):
        merged = dict(FARMER_DEFAULTS)
        merged.update(getattr(self, "defaults", {}) or {})
        merged.update(kwargs)
        self.params = merged
        rng = np.random.default_rng(int(merged["seed"]))
        spread = float(merged["trait_variability"])
        self.traits = {}
        for t in TRAITS:
            noise = rng.normal(0., spread * (t.high - t.low))  # always drawn: stable order
            if t.name in merged:
                value = float(merged[t.name])
            else:
                value = float(self.traits_mean.get(t.name, t.default)) + noise
            self.traits[t.name] = float(np.clip(value, t.low, t.high))
        # Age is descriptive; it does not deterministically assign behaviour.
        self.traits["age"] = int(rng.integers(25, 76))
        # Usual weekly days off: Sat+Sun, or for some farmers the pair a day earlier/later.
        # Drawn from its own stream so the traits above keep their values.
        week = np.random.default_rng([int(merged["seed"]) & 0xFFFFFFFF, 23])
        shift = 0 if week.random() >= merged["schedule_jitter"] else (1 if week.random() < .5 else -1)
        self.days_off = tuple(sorted((d + shift) % 7 for d in (5, 6)))
        # The personality as drawn; self.traits may drift (learning) during a season.
        self.initial_traits = dict(self.traits)
        self.target_multiplier = float(merged["target_multiplier"])
        self.monitor_interval = max(1, int(round(1 + 4 * (1 - self.traits["labour"]) * (1 - self.traits["work_ethic"]))))
        self.neighbour_level = {}   # paddy id -> remembered neighbour ponding (mm)
        self.last_checked_day = -1
        self.checked_today = False
        self.standing_mm = None     # inlet left open between checks (per owned paddy)
        self.weir_setting = None    # weir crest left in place between checks
        self.last_target_mm = None
        self.last_lower_mm = None
        self.last_upper_mm = None
        self.last_request_mm = None
        self.last_demand_mm = 0.
        self.price_factor = 1.
        self.spend = 0.
        self.state = "wait"

    # --- personality ------------------------------------------------------------------------

    @property
    def cooperation(self) -> float:
        return float(self.traits["cooperation"])

    @cooperation.setter
    def cooperation(self, value: float) -> None:
        self.traits["cooperation"] = float(np.clip(value, 0, 1))

    @property
    def cooperation_level(self) -> float:
        """This farmer's collaboration (0..1), whatever kind of farmer it is."""
        return self.cooperation

    def set_cooperation(self, value: float) -> None:
        """Set this farmer's collaboration (0..1) — used by collaboration scenarios."""
        self.cooperation = value
        self.params["cooperation"] = self.cooperation

    @property
    def calendar_shift_days(self) -> int:
        """Days this farmer runs ahead (+) or behind (-) the district crop calendar."""
        return int(round(self.params["max_shift_days"] * self.traits["eagerness"]))

    def crop_day(self, day: int) -> int:
        return int(day) + self.calendar_shift_days

    # --- hooks a subclass may override -------------------------------------------------------

    def stage_target_mm(self, obs: dict, crop_day: int) -> float:
        """Ideal pond on this farmer's own calendar, scaled by its multiplier and risk buffer."""
        target = crop.target_depth_mm(crop_day) * self.target_multiplier
        if not in_drain_window(crop_day):
            target += self.params["risk_buffer_mm"] * (1 - self.traits["risk_tolerance"])
        return float(target)

    def bounds_mm(self, target: np.ndarray, crop_day: int, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) around the target: refill trigger and refill level."""
        target = np.asarray(target, dtype=float)
        ceiling = float(obs.get("bund_mm", np.inf)) - self.params["freeboard_mm"]
        if in_drain_window(crop_day):
            return np.zeros_like(target), np.minimum(target, ceiling)
        lower = np.maximum(target - self.traits["drawdown"] * self.params["lower_gap_mm"], 0.)
        upper = np.minimum(target + self.traits["overfill"] * self.params["upper_gap_mm"], ceiling)
        return lower, np.maximum(upper, lower)

    def _draw(self, *keys: int) -> float:
        return float(np.random.default_rng([int(self.params["seed"]) & 0xFFFFFFFF, *map(int, keys)]).random())

    def days_off_in_week(self, day: int) -> tuple[int, ...]:
        """This farmer's days off (0 Monday .. 6 Sunday) in the week holding ``day``.

        Its own usual pair (``self.days_off``: Sat+Sun for most, a day earlier or later
        for some) and, some weeks (``week_swap``), one of them moved to another day —
        overtime, a shift change, a market day."""
        off = list(self.days_off)
        week = int(day) // 7
        if self._draw(week, 29) < self.params["week_swap"]:
            which = int(self._draw(week, 31) * len(off)) % len(off)
            others = [d for d in range(7) if d not in off]
            off[which] = others[int(self._draw(week, 37) * len(others)) % len(others)]
        return tuple(off)

    def is_day_off(self, day: int, weekday: int | None) -> bool:
        return weekday is not None and int(weekday) in self.days_off_in_week(day)

    def is_home(self, day: int, weekday: int | None) -> bool:
        """Days off always; a working day unless this part-timer is away (seeded per day)."""
        if weekday is None or self.is_day_off(day, weekday) or self.traits["part_time"] <= 0:
            return True
        return self._draw(day, 17) >= self.traits["part_time"]

    def would_check(self, day: int, weekday: int | None, last: int) -> bool:
        if last < 0 or day - last >= 7:
            return True
        return self.is_home(day, weekday) and (day - last >= self.monitor_interval or self.is_day_off(day, weekday))

    def checks_today(self, obs: dict) -> bool:
        return self.would_check(int(obs["day"]), obs.get("weekday"), self.last_checked_day)

    def days_to_next_check(self, obs: dict) -> int:
        """The farmer knows its own work schedule: days until it next looks."""
        day, weekday = int(obs["day"]), obs.get("weekday")
        for k in range(1, 8):
            wd = None if weekday is None else (int(weekday) + k) % 7
            if self.would_check(day + k, wd, day):
                return k
        return 7

    def daily_loss_mm(self, obs: dict) -> np.ndarray:
        """Expected pond fall per day: ET (own calendar) + the seepage it reads off the pond."""
        n = len(obs["owned_ids"])
        et = np.asarray(obs.get("owned_et_demand_mm", np.full(n, float(obs["et_mm"]))), dtype=float)
        seepage = np.asarray(obs.get("owned_seepage_mm", np.zeros(n)), dtype=float)
        return et + self.traits["knowledge"] * seepage

    def expected_rain_mm(self, obs: dict) -> float:
        """Forecast rain after today that it trusts (foresight days, decaying with lead)."""
        forecast = np.asarray(obs.get("rain_forecast_mm", []), dtype=float)
        horizon = int(round(self.params["max_forecast_days"] * self.traits["foresight"]))
        decay = float(self.params["forecast_decay"])
        return float(sum(forecast[k] * decay ** k for k in range(1, min(horizon, forecast.size - 1) + 1)))

    def is_hot(self, obs: dict) -> bool:
        return float(obs.get("et0_mm", 0.)) >= self.params["heat_et0_mm"]

    def price_multiplier(self, obs: dict) -> float:
        """Pricing check: share of discretionary water kept at today's volumetric price."""
        price = float(obs.get("water_price_per_m3", 0.) or 0.)
        if price <= 0:
            return 1.
        pressure = self.traits["price_sensitivity"] * (1 - .5 * self.traits["wealth"])
        return 1. / (1. + pressure * price / max(self.params["price_ref_per_m3"], 1e-9))

    def learn(self, obs: dict) -> None:
        """Update beliefs on a check day before planning (PersonalityFarmer: social learning)."""

    def adjust(self, obs: dict, plan: Plan) -> None:
        """Last personal touch on a check-day plan, before the hard constraints."""

    # --- the default decision ----------------------------------------------------------------

    def neighbour_target(self, obs: dict, target, offset: float | None = None,
                         cooperation: float | None = None) -> np.ndarray:
        """Per owned paddy: ``target`` drifted softly by the farmer's stance on its neighbours.

            pull = cooperation · (neighbour level − target)        collaborative: level up
                 + (1 − cooperation) · offset · neighbour_offset_mm  uncollaborative: own way

        The neighbour level is a running memory of the mean ponding on each paddy's (up to
        4) cardinal neighbours. The pull is capped by a tanh and weighted, so the target
        moves only part of the way. ``offset``/``cooperation`` default to the traits.
        """
        p = {**NEIGHBOUR_DEFAULTS, **self.params}
        ids = np.asarray(obs.get("owned_ids", np.arange(len(obs["ponding_mm"]))), dtype=int)
        pond = np.asarray(obs.get("owned_ponding_mm", np.asarray(obs["ponding_mm"])[ids]), dtype=float)
        neighbour_mean = np.asarray(obs.get("neighbour_mean_mm", pond), dtype=float)
        offset = float(self.traits["neighbour_offset"] if offset is None else offset)
        cooperation = float(np.clip(self.cooperation if cooperation is None else cooperation, 0, 1))
        memory = float(np.clip(p["neighbour_memory"], 0, 1))
        level = np.array([
            (1 - memory) * self.neighbour_level.get(int(i), float(m)) + memory * float(m)
            for i, m in zip(ids, neighbour_mean)])
        self.neighbour_level.update(zip((int(i) for i in ids), level))
        target = np.broadcast_to(np.asarray(target, dtype=float), ids.shape)
        cap = max(float(p["neighbour_max_shift_mm"]), 1e-6)
        pull = cooperation * (level - target) + (1 - cooperation) * offset * p["neighbour_offset_mm"]
        shift = cap * np.tanh(pull / cap)
        return np.maximum(target + float(p["neighbour_weight"]) * shift, 0.)

    def plan(self, obs: dict) -> Plan:
        """The default day (module docstring); owned-paddy arrays."""
        ids = np.asarray(obs["owned_ids"], dtype=int)
        n = len(ids)
        pond = np.asarray(obs["owned_ponding_mm"], dtype=float)
        sent = np.asarray(obs["owned_irrigation_sent_mm"], dtype=float)
        if self.standing_mm is None or len(self.standing_mm) != n:
            self.standing_mm, self.weir_setting = np.zeros(n), np.full(n, np.inf)
        self.checked_today = self.checks_today(obs)
        if not self.checked_today:
            # Away or not due: whatever the inlet and board were left at keeps running.
            self.state = "routine"
            return Plan(ids, self.standing_mm.copy(), np.zeros(n), self.weir_setting.copy(),
                        self.standing_mm.copy(), dict(checked=False))
        day = int(obs["day"])
        self.last_checked_day = day
        self.learn(obs)
        cd = self.crop_day(day)
        target = self.neighbour_target(obs, self.stage_target_mm(obs, cd))
        lower, upper = self.bounds_mm(target, cd, obs)
        self.last_target_mm, self.last_lower_mm, self.last_upper_mm = target.copy(), lower.copy(), upper.copy()

        loss = self.daily_loss_mm(obs)
        gap = self.days_to_next_check(obs)
        rain_now = self.traits["knowledge"] * float(obs["rain_mm"])
        rain_ahead = self.expected_rain_mm(obs)
        # Fill and let drop: refill only if the pond would sink below the lower bound
        # before the next look; leave room behind the board for the rain it expects.
        projected = pond + rain_now + rain_ahead - loss * gap
        refill = projected < lower
        fill_to = np.maximum(upper - rain_ahead, lower)
        fill = np.where(refill, np.maximum(fill_to - pond - rain_now + loss, 0.), 0.)
        essential = np.where(refill, np.maximum(np.minimum(target, fill_to) - pond - rain_now + loss, 0.), 0.)
        # Cannot last until the next look on one fill: leave the inlet trickling.
        trickle = np.where(refill, np.maximum(lower + loss * gap - fill_to - rain_ahead, 0.) / gap, 0.)
        if gap > 1:
            slack = self.traits["flow_through"] * self.params["away_slack"] * loss
        else:
            slack = np.zeros(n)
        hot = self.is_hot(obs) and not in_drain_window(cd)
        flow = np.full(n, self.traits["flow_through"] * self.params["heat_flow_mm"] if hot else 0.)
        scarce = float(sent.sum()) < float((fill + flow).sum())
        fill = fill * (1 - self.params["scarcity_cut"] * self.cooperation * scarce)
        essential = np.minimum(essential, fill)
        # Pricing check: essential water stays, the extras shrink with the price.
        self.price_factor = keep = self.price_multiplier(obs)
        fill = essential + keep * (fill - essential)
        flow, slack = keep * flow, keep * slack
        standing = trickle + slack + (flow if gap > 1 else 0.)
        request = fill + flow
        plan = Plan(ids, np.minimum(request, sent), np.zeros(n), upper.copy(), request,
                    dict(checked=True, refill=refill, gap=gap, hot=hot, price_factor=keep))
        self.adjust(obs, plan)
        self.standing_mm, self.weir_setting = np.asarray(standing, dtype=float) * np.ones(n), plan.weir.copy()
        self.state = "irrigating" if plan.apply.any() else ("waiting_for_water" if plan.request.any() else "wait")
        return plan

    # --- hard rules --------------------------------------------------------------------------

    def all_constraints(self) -> tuple[Constraint, ...]:
        return tuple(self.constraints)

    def enforce(self, obs: dict, plan: Plan) -> Plan:
        """Apply ≤ sent, drain ≤ stock, and every active constraint — last word, always."""
        pond = np.asarray(obs["owned_ponding_mm"], dtype=float)
        sent = np.asarray(obs["owned_irrigation_sent_mm"], dtype=float)
        as_arr = lambda a: np.broadcast_to(np.asarray(a, dtype=float), pond.shape).copy()
        plan.apply, plan.drain = as_arr(plan.apply), as_arr(plan.drain)
        plan.weir, plan.request = as_arr(plan.weir), as_arr(plan.request)
        cd = self.crop_day(int(obs.get("day", 0)))
        for rule in self.all_constraints():
            if rule.active(self, obs, cd):
                rule.enforce(self, obs, plan, pond)
        plan.request = np.maximum(plan.request, 0.)
        plan.apply = np.clip(plan.apply, 0., sent)
        stock = np.maximum(pond + float(obs.get("rain_mm", 0.)), 0.)
        plan.drain = np.clip(plan.drain, 0., stock + plan.apply)
        plan.weir = np.maximum(plan.weir, 0.)
        return plan

    def act(self, obs: dict) -> Plan:
        """Today's plan for the owned paddies, hard rules enforced."""
        obs = _owned_view(obs)
        if type(self).decide_owned is not BaseFarmer.decide_owned:
            # A subclass that writes its own decide_owned (apply, drain) still gets the
            # hard rules; it has no weir board unless it sets one in plan().
            apply, drain = self.decide_owned(obs)
            n = len(obs["owned_ids"])
            plan = Plan(np.asarray(obs["owned_ids"], dtype=int), apply, drain, np.full(n, np.inf), apply)
        else:
            plan = self.plan(obs)
        plan = self.enforce(obs, plan)
        self.last_request_mm = plan.request.copy()
        self.last_demand_mm = float(np.mean(plan.request)) if len(plan.request) else 0.
        area = np.asarray(obs.get("area_m2", np.full(len(obs["ponding_mm"]), 1e4)), dtype=float)[plan.ids]
        self.spend += float(obs.get("water_price_per_m3", 0.) or 0.) * float((plan.apply * area).sum() / 1000.)
        return plan

    def decide_owned(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Legacy override point: return (apply, drain) for the owned paddies."""
        plan = self.plan(obs)
        return plan.apply, plan.drain

    def decide(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """(apply, drain) over every paddy in ``obs`` (zeros off the farm)."""
        plan = self.act(obs)
        n = len(obs["ponding_mm"])
        apply, drain = np.zeros(n), np.zeros(n)
        apply[plan.ids], drain[plan.ids] = plan.apply, plan.drain
        return apply, drain


def _owned_view(obs: dict) -> dict:
    """Fill the owned_* keys when a farmer is handed the whole field (no owned_ids)."""
    if "owned_ids" in obs and "owned_ponding_mm" in obs and "owned_irrigation_sent_mm" in obs:
        return obs
    ids = np.asarray(obs.get("owned_ids", np.arange(len(obs["ponding_mm"]))), dtype=int)
    out = dict(obs, owned_ids=ids)
    out.setdefault("owned_ponding_mm", np.asarray(obs["ponding_mm"], dtype=float)[ids])
    out.setdefault("owned_irrigation_sent_mm", np.asarray(obs.get("irrigation_sent_mm", np.zeros(len(obs["ponding_mm"]))), dtype=float)[ids])
    if "neighbour_mean_mm" not in obs:
        out["neighbour_mean_mm"] = out["owned_ponding_mm"]
    return out
