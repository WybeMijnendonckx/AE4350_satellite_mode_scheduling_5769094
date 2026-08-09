"""environment.py

gymnasium.Env implementation of the spacecraft mode-scheduling problem.
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
OBS_POINTING_ERROR_FRAC = 8
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

        effective_capacity = self.sc.battery_capacity_wh * self.battery_health
        soc_frac = self.soc_wh / effective_capacity
        soc_noisy = soc_frac + self._rng.normal(0, self.sc.soc_sensor_noise_std)
        soc_noisy = np.clip(soc_noisy, 0.0, 1.0)

        buffer_frac = self.buffer_mb / self.sc.buffer_capacity_mb
        buffer_noisy = buffer_frac + self._rng.normal(0, self.sc.buffer_sensor_noise_std)
        buffer_noisy = np.clip(buffer_noisy, 0.0, 1.0)

        angle_to_target = np.rad2deg(np.arccos(np.clip(np.dot(self.current_attitude, self.target_attitude), -1, 1)))
        pointing_error_frac = angle_to_target / 180.0

        time_frac = self.step_count / self.max_steps

        obs = np.concatenate([
            [soc_noisy],
            [buffer_noisy],
            self.current_attitude,
            self.target_attitude,
            [pointing_error_frac],
            [float(geom["is_eclipse"])],
            [float(geom["is_ground_contact"])],
            [self.battery_health],
            [self.instrument_efficiency],
            [time_frac],
        ]).astype(np.float32)
        return obs
    
    def get_action_mask(self, is_eclipse: bool) -> np.ndarray:
        """Boolean mask over the 4 actions; False = unavailable this step.
        Only Sun-pointing is masked because it is unavailable during eclipse."""
        mask = np.ones(4, dtype=bool)
        mask[Action.SUN_POINT] = not is_eclipse
        return mask
    
    def step(self, action: int):
        geom = self.propagator.get_geometry(self.t)

        # safety net: agent.py should mask invalid actions itself, but
        # fall back to Idle here rather than silently doing something invalid
        mask = self.get_action_mask(geom["is_eclipse"])
        if not mask[action]:
            action = Action.IDLE

        # determine desired pointing target for the commanded action
        if action == Action.IDLE:
            desired_target = self.current_attitude
        elif action == Action.SUN_POINT:
            desired_target = geom["sun_dir"]
        elif action == Action.NADIR_POINT:
            desired_target = geom["nadir_dir"]
        else:  # GROUND_STATION_POINT
            desired_target = geom["gs_los_dir"]
        self.target_attitude = desired_target

        # rotate current attitude toward target, bounded by slew rate
        dot = np.clip(np.dot(self.current_attitude, desired_target), -1.0, 1.0)
        angle_deg = np.rad2deg(np.arccos(dot))
        max_step_deg = self.sc.slew_rate_deg_s * self.dt
        is_slewing = angle_deg > 1e-6

        if angle_deg <= max_step_deg:
            self.current_attitude = desired_target
        else:
            frac = max_step_deg / angle_deg
            self.current_attitude = slerp(self.current_attitude, desired_target, frac)

        # cone checks against the (updated) current attitude
        angle_to_nadir = np.rad2deg(np.arccos(np.clip(np.dot(self.current_attitude, geom["nadir_dir"]), -1, 1)))
        angle_to_gs = np.rad2deg(np.arccos(np.clip(np.dot(self.current_attitude, geom["gs_los_dir"]), -1, 1)))
        angle_to_sun = np.rad2deg(np.arccos(np.clip(np.dot(self.current_attitude, geom["sun_dir"]), -1, 1)))

        imaging_active = (angle_to_nadir <= self.sc.instrument_fov_deg) and (self.buffer_mb < self.sc.buffer_capacity_mb)
        downlink_active = (
            geom["is_ground_contact"]
            and angle_to_gs <= self.gs_params.antenna_beamwidth_deg
            and self.buffer_mb > 0.0
        )
        # charging: cosine law, zero beyond cutoff angle, zero in eclipse
        sun_factor = max(0.0, np.cos(np.deg2rad(angle_to_sun))) if angle_to_sun <= self.sc.solar_cutoff_angle_deg else 0.0
        charging_power_w = 0.0 if geom["is_eclipse"] else self.sc.power_charging_max_w * sun_factor
        charging_power_w *= 1.0 + self._rng.normal(0, self.sc.solar_input_noise_std)
        charging_power_w = max(0.0, charging_power_w)

        # power balance
        power_draw_w = self.sc.power_housekeeping_w
        if imaging_active:
            power_draw_w += self.sc.power_imaging_w
        if downlink_active:
            power_draw_w += self.sc.power_downlink_w
        if is_slewing:
            slew_frac = min(1.0, angle_deg / max_step_deg)  # partial step if slew completes mid-step
            power_draw_w += self.sc.power_slew_w * slew_frac
            
        delta_wh = (charging_power_w - power_draw_w) * self.dt / 3600.0
        effective_capacity = self.sc.battery_capacity_wh * self.battery_health
        self.soc_wh = np.clip(self.soc_wh + delta_wh, 0.0, effective_capacity)

        # data balance (mb throughout, consistent with *_mbps rates)
        data_generated_mb = self.sc.imaging_data_rate_mbps * self.instrument_efficiency * self.dt if imaging_active else 0.0
        self.buffer_mb = min(self.sc.buffer_capacity_mb, self.buffer_mb + data_generated_mb)

        data_downlinked_mb = self.gs_params.downlink_rate_mbps * self.dt if downlink_active else 0.0
        data_downlinked_mb = min(data_downlinked_mb, self.buffer_mb)
        self.buffer_mb -= data_downlinked_mb

        # degradation
        soc_frac = self.soc_wh / effective_capacity
        if soc_frac < self.sc.battery_safety_floor_frac:
            self.battery_health = max(0.0, self.battery_health - self.sc.battery_degradation_per_step)

        # reward
        reward = self.reward.reward_per_mb_downlinked * data_downlinked_mb
        reward += self.reward.imaging_shaping_bonus * data_generated_mb

        # termination / truncation
        terminated = self.battery_health < self.sc.battery_health_hard_floor
        if terminated:
            reward += self.reward.terminal_penalty_hard_floor

        self.t += self.dt
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "imaging_active": imaging_active,
            "downlink_active": downlink_active,
            "charging_power_w": charging_power_w,
            "is_slewing": is_slewing,
            "data_downlinked_mb": data_downlinked_mb,
            "action_mask": mask,
        }
        return obs, reward, terminated, truncated, info