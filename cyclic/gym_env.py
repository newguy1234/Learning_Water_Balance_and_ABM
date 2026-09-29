"""Official Gymnasium API for CyclicEnv (Environmental Modelling & Software package).

Install::

    pip install -e ".[gym]"

Use::

    import gymnasium as gym
    import cyclic.gym_env  # registers Cyclic-Imbanuma-v0

    env = gym.make("Cyclic-Imbanuma-v0", year=2022, given_irrigation=True)
    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

Observation is a float32 feature matrix ``(n_paddies, n_features)`` for the
active role (default ``farmer``). Multi-role batches remain in ``info``
(``obs_equation``, ``obs_ratio``, ``obs_batch``).
"""

from __future__ import annotations

import numpy as np

from .env import CyclicEnv, GYM_WEIGHTS
from . import okura
from .agents import ObservationBatch, resolve_observation_channels

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Gymnasium is required for cyclic.gym_env. Install with: pip install -e \".[gym]\""
    ) from exc


ENV_ID = "Cyclic-Imbanuma-v0"
ENV_ID_OPEN = "Cyclic-Imbanuma-Open-v0"
_NON_FEATURE = frozenset({"paddy_id", "block", "cyclic"})


def _feature_dim(role):
    keep = resolve_observation_channels(role)
    return sum(1 for key in ObservationBatch.FIELDS if key in keep and key not in _NON_FEATURE)


class CyclicGym(gym.Env):
    """Gymnasium Env wrapping the Imbanuma cyclic-irrigation district.

    Parameters
    ----------
    year : int
        Season year (2021 or 2022).
    given_irrigation : bool
        If True (default gym), Okura IR is known forcing; action is discharge
        (and optional irrigation disaggregation still accepted).
    obs_role : str
        Which observation channel pack feeds ``observation_space`` / ``obs``
        (``farmer``, ``equation``, ``ratio``, ``full``).
    array_obs : bool
        If True (default), observations are ``(n, d)`` float32 arrays.
        If False, returns the rich ``ObservationBatch`` (not a standard Box).
    """

    metadata = {
        "render_modes": [],
        "name": "CyclicEnv",
        "district": "Shiroyama-Jinbee / Imbanuma, Chiba (Okura et al. 2025)",
        "journal_target": "Environmental Modelling & Software",
    }

    def __init__(self, year=okura.COMPARISON_YEAR, given_irrigation=True,
                 obs_role="farmer", array_obs=True, render_mode=None, **env_kwargs):
        super().__init__()
        self.year = int(year)
        self.given_irrigation = bool(given_irrigation)
        self.obs_role = str(obs_role)
        self.array_obs = bool(array_obs)
        self.render_mode = render_mode
        env_kwargs = dict(env_kwargs)
        env_kwargs.setdefault("given_irrigation", self.given_irrigation)
        env_kwargs.setdefault("record_paddies", False)
        self._env = CyclicEnv(year=self.year, **env_kwargs)
        self._feature_names = ()
        self._n = self._env.n_paddies
        self._d = max(_feature_dim(self.obs_role), 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n, self._d), dtype=np.float32)
        if self.given_irrigation:
            # discharge only (mm); irrigation may still be passed via dict actions
            self.action_space = spaces.Box(
                low=0.0, high=200.0, shape=(self._n,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(
                low=0.0, high=200.0, shape=(self._n, 2), dtype=np.float32)

    @property
    def core(self):
        """Underlying ``CyclicEnv`` (channel config, Copernicus-ready AOI, etc.)."""
        return self._env

    @property
    def feature_names(self):
        return self._feature_names

    def _encode_obs(self, batch):
        if batch is None:
            return None
        if not self.array_obs:
            return batch
        matrix, names = batch.feature_matrix(channels=self.obs_role)
        self._feature_names = names
        self._d = matrix.shape[1]
        self._n = matrix.shape[0]
        # NaN (e.g. missing satellite) → 0 so obs stays inside Box.
        matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=matrix.shape, dtype=np.float32)
        return matrix

    def _decode_action(self, action):
        action = np.asarray(action, dtype=float)
        if self.given_irrigation:
            if action.ndim == 1:
                return np.zeros(self._n), action
            if action.ndim == 2 and action.shape[-1] == 2:
                return action[:, 0], action[:, 1]
            return np.zeros(self._n), np.broadcast_to(action.ravel()[: self._n], (self._n,))
        if action.ndim == 1:
            half = action.size // 2
            return action[:half], action[half: half * 2]
        return action[:, 0], action[:, 1]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        role = options.get("obs_role", self.obs_role)
        batch, info = self._env.reset(seed=seed)
        self._n = self._env.model.n
        if self.given_irrigation:
            self.action_space = spaces.Box(
                low=0.0, high=200.0, shape=(self._n,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(
                low=0.0, high=200.0, shape=(self._n, 2), dtype=np.float32)
        role_batch = self._env.observation(role)
        obs = self._encode_obs(role_batch)
        info = dict(info)
        info["obs_batch"] = role_batch
        info["obs_farmer"] = self._env.observation("farmer")
        info["obs_equation"] = self._env.observation("equation")
        info["obs_ratio"] = self._env.observation("ratio")
        info["feature_names"] = self._feature_names
        info["obs_role"] = role
        return obs, info

    def step(self, action):
        if isinstance(action, dict):
            packed = action
        else:
            irrigation, discharge = self._decode_action(action)
            packed = (irrigation, discharge)
        batch, reward, terminated, truncated, info = self._env.step(packed)
        info = dict(info)
        info["obs_role"] = self.obs_role
        if batch is None:
            obs = None
        else:
            role = self.obs_role
            role_batch = info.get(f"obs_{role}") or self._env.observation(role)
            obs = self._encode_obs(role_batch)
            info["obs_batch"] = role_batch
            info["obs_farmer"] = info.get("obs_farmer") or self._env.observation("farmer")
            info["obs_equation"] = info.get("obs_equation") or self._env.observation("equation")
            info["obs_ratio"] = info.get("obs_ratio") or self._env.observation("ratio")
            info["feature_names"] = self._feature_names
        reward = float(reward)
        return obs, reward, bool(terminated), bool(truncated), info

    def set_obs_channels(self, role, channels=None, hide=None, show=None):
        return self._env.set_obs_channels(role, channels=channels, hide=hide, show=show)

    def close(self):
        self._env.model = None


def make_cyclic_env(**kwargs):
    """Factory used by Gymnasium registration."""
    return CyclicGym(**kwargs)


def register_envs(force=False):
    """Register Cyclic-Imbanuma-* with Gymnasium (idempotent)."""
    registry = gym.envs.registry
    specs = getattr(registry, "env_specs", None)
    if specs is None:
        # Gymnasium >= 0.29 uses dict-like registry
        known = ENV_ID in registry
    else:
        known = ENV_ID in specs
    if known and not force:
        return
    try:
        gym.register(
            id=ENV_ID,
            entry_point="cyclic.gym_env:CyclicGym",
            kwargs={"given_irrigation": True, "year": okura.COMPARISON_YEAR},
        )
    except Exception:
        if force:
            raise
    try:
        gym.register(
            id=ENV_ID_OPEN,
            entry_point="cyclic.gym_env:CyclicGym",
            kwargs={"given_irrigation": False, "year": okura.COMPARISON_YEAR},
        )
    except Exception:
        if force:
            raise


# Register on import so ``import cyclic.gym_env`` is enough before gym.make.
register_envs()
