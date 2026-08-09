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