"""environment.py

gymnasium.Env implementation of the spacecraft mode-scheduling problem.
Puts together orbital_dynamics.py  and config.py
into the actual MDP: attitude/slew mechanics, cone-based subsystem
activation, power/data bookkeeping, degradation and reward (the science return).
"""

from enum import IntEnum

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from orbital_dynamics import OrbitPropagator

class Action(IntEnum):
    IDLE = 0
    SUN_POINT = 1
    NADIR_POINT = 2
    GROUND_STATION_POINT = 3


# observation vector layout (14 dims total)
OBS_SOC = 0
OBS_BUFFER = 1
OBS_ATTITUDE = slice(2, 5)          # current pointing unit vector (3)
OBS_TARGET_ATTITUDE = slice(5, 8)   # commanded target unit vector (3)
OBS_SLEW_PROGRESS = 8
OBS_ECLIPSE = 9
OBS_GROUND_CONTACT = 10
OBS_BATTERY_HEALTH = 11
OBS_INSTRUMENT_EFFICIENCY = 12
OBS_TIME_FRAC = 13
OBS_DIM = 14


def slerp(v0: np.ndarray, v1: np.ndarray, frac: float) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors."""
    frac = np.clip(frac, 0.0, 1.0)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    if dot > 0.9995:  # nearly identical directions, linear interp is fine
        result = v0 + frac * (v1 - v0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1 - frac) * theta) / sin_theta
    b = np.sin(frac * theta) / sin_theta
    return a * v0 + b * v1


class SpacecraftSchedulerEnv(gym.Env):
    """Discrete-action, continuous-state MDP: the agent commands one of (Idle, Sun-point, Nadir-point,
    Ground-station-point) each step, subject to slew dynamics and eclipse-based action masking
    and is rewarded for data volume successfully downlinked."""

    metadata = {"render_modes": []}

    def __init__(self, orbit_params, gs_params, sc_params, reward_params, train_params):
        super().__init__()
        self.orbit_params = orbit_params
        self.gs_params = gs_params
        self.sc = sc_params
        self.reward = reward_params
        self.train = train_params

        self.propagator = OrbitPropagator(orbit_params, gs_params)
        self.dt = train_params.timestep_s
        self.episode_length_s = train_params.num_orbits_per_episode * self.propagator.period_s
        self.max_steps = int(self.episode_length_s / self.dt)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, 0.0] + [-1.0] * 6 + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0] + [1.0] * 6 + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # populated in reset()
        self._rng = None
        self.t = 0.0
        self.step_count = 0
        self.soc_wh = 0.0
        self.buffer_mb = 0.0
        self.battery_health = 1.0
        self.instrument_efficiency = 1.0
        self.current_attitude = np.array([1.0, 0.0, 0.0])
        self.target_attitude = np.array([1.0, 0.0, 0.0])
        self.slew_start_t = 0.0
        self.slew_duration = 0.0
        self.last_action = Action.IDLE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed if seed is not None else self.train.seed)

        # random start time within the orbit period so eclipse/contact
        # timing relative to the episode isn't identical every episode
        self.t = self._rng.uniform(0.0, self.propagator.period_s)
        self.step_count = 0

        # degradation: sampled once per episode, held fixed
        lo, hi = self.sc.battery_health_range
        self.battery_health = self._rng.uniform(lo, hi)
        lo, hi = self.sc.instrument_efficiency_range
        self.instrument_efficiency = self._rng.uniform(lo, hi)

        # start fully charged, empty buffer
        self.soc_wh = self.sc.battery_capacity_wh * self.battery_health
        self.buffer_mb = 0.0

        # start pointed at nadir, arbitrarily chosen. Spacecraft is idle and no slew in progress
        geom = self.propagator.get_geometry(self.t)
        self.current_attitude = geom["nadir_dir"].copy()
        self.target_attitude = self.current_attitude.copy()
        self.slew_start_t = self.t
        self.slew_duration = 0.0
        self.last_action = Action.IDLE

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        geom = self.propagator.get_geometry(self.t)

        soc_frac = self.soc_wh / (self.sc.battery_capacity_wh * self.battery_health)
        soc_noisy = soc_frac + self._rng.normal(0, self.sc.soc_sensor_noise_std)
        soc_noisy = np.clip(soc_noisy, 0.0, 1.0)

        buffer_frac = self.buffer_mb / self.sc.buffer_capacity_mb
        buffer_noisy = buffer_frac + self._rng.normal(0, self.sc.buffer_sensor_noise_std)
        buffer_noisy = np.clip(buffer_noisy, 0.0, 1.0)

        if self.slew_duration > 0.0:
            progress = (self.t - self.slew_start_t) / self.slew_duration
        else:
            progress = 1.0
        progress = float(np.clip(progress, 0.0, 1.0))

        time_frac = self.step_count / self.max_steps

        obs = np.concatenate([
            [soc_noisy],
            [buffer_noisy],
            self.current_attitude,
            self.target_attitude,
            [progress],
            [float(geom["is_eclipse"])],
            [float(geom["is_ground_contact"])],
            [self.battery_health],
            [self.instrument_efficiency],
            [time_frac],
        ]).astype(np.float32)
        return obs