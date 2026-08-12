"""evaluate.py

Loads a trained checkpoint and does greedy runs for evaluation, 
detailed episode trajectory logging and
a sensitivity-analysis sweep over interesting config parameters.
"""

import copy
import os

import numpy as np

from config import (
    OrbitParams,
    GroundStationParams,
    SpacecraftParams,
    RewardParams,
    RLParams,
    TrainingParams,
)
from environment import SpacecraftSchedulerEnv, OBS_DIM
from agent import DQNAgent
from train import train, CHECKPOINT_DIR, RESULTS_DIR


def load_agent(checkpoint_path: str, rl_params: RLParams = None) -> DQNAgent:
    """Loads a trained Q-network into a DQNAgent, ready for greedy evaluation"""
    rl_params = rl_params or RLParams()
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=4, rl_params=rl_params)
    agent.load(checkpoint_path)
    return agent


def run_episode(env: SpacecraftSchedulerEnv, agent: DQNAgent, seed: int, greedy: bool = True) -> dict:
    """Runs a single episode and returns a full step level trajectory"""
    obs, info = env.reset(seed=seed)

    log = {
        "t": [], "soc_frac": [], "buffer_frac": [], "attitude": [],
        "imaging_active": [], "downlink_active": [], "charging_power_w": [],
        "is_eclipse": [], "is_ground_contact": [], "reward": [], "action": [],
    }

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        geom = env.propagator.get_geometry(env.t)
        mask = env.get_action_mask(geom["is_eclipse"])
        action = agent.select_action(obs, mask, greedy=greedy)
        next_obs, reward, terminated, truncated, info = env.step(action)

        effective_capacity = env.sc.battery_capacity_wh * env.battery_health
        log["t"].append(env.t)
        log["soc_frac"].append(env.soc_wh / effective_capacity)
        log["buffer_frac"].append(env.buffer_mb / env.sc.buffer_capacity_mb)
        log["attitude"].append(env.current_attitude.copy())
        log["imaging_active"].append(info["imaging_active"])
        log["downlink_active"].append(info["downlink_active"])
        log["charging_power_w"].append(info["charging_power_w"])
        log["is_eclipse"].append(geom["is_eclipse"])
        log["is_ground_contact"].append(geom["is_ground_contact"])
        log["reward"].append(reward)
        log["action"].append(int(action))

        total_reward += reward
        obs = next_obs

    log = {k: np.array(v) for k, v in log.items()}
    log["total_reward"] = total_reward
    log["terminated"] = terminated
    log["final_battery_health"] = env.battery_health
    return log