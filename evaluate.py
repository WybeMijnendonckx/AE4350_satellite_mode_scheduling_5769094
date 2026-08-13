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

def evaluate_agent(agent: DQNAgent, env: SpacecraftSchedulerEnv, n_episodes: int = 20, seed_base: int = 100) -> dict:
    """Runs greedy episodes and returns summary per episode"""
    rewards = []
    terminated_flags = []
    lengths = []
    final_battery_healths = []

    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_base + i)
        terminated = False
        truncated = False
        ep_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            geom = env.propagator.get_geometry(env.t)
            mask = env.get_action_mask(geom["is_eclipse"])
            action = agent.select_action(obs, mask, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

        rewards.append(ep_reward)
        terminated_flags.append(terminated)
        lengths.append(steps)
        final_battery_healths.append(env.battery_health)

    return {
        "rewards": np.array(rewards),
        "terminated": np.array(terminated_flags),
        "lengths": np.array(lengths),
        "final_battery_health": np.array(final_battery_healths),
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
    }


def run_sensitivity_sweep(param_specs: list, n_seeds: int = 3, num_episodes: int = 500, n_eval_episodes: int = 20) -> dict:
    """General-purpose sensitivity sweep"""
    results = {}

    for spec in param_specs:
        name, group, values, setter = spec["name"], spec["group"], spec["values"], spec["setter"]

        for value in values:
            for seed in range(n_seeds):
                print(f"\n=== sweep: {name}={value}, seed={seed} ===")

                orbit_p, gs_p, sc_p = OrbitParams(), GroundStationParams(), SpacecraftParams()
                target = {"orbit": orbit_p, "gs": gs_p, "sc": sc_p}[group]
                setter(target, value)

                sweep_agent, sweep_env = train(
                    num_episodes=num_episodes, seed=seed,
                    orbit_params=orbit_p, gs_params=gs_p, sc_params=sc_p,
                    save_checkpoints=False,
                )
                eval_stats = evaluate_agent(sweep_agent, sweep_env, n_episodes=n_eval_episodes)

                results[f"{name}_{value}_seed{seed}"] = eval_stats

    return results

def save_sweep_results(results: dict, path: str) -> None:
    """Function to flatten the nested dict of dicts into one .npz file,
    so the whole sweep is in one file for plotting.py to load once and slice by name."""
    flat = {}
    for run_key, stats in results.items():
        for stat_name, arr in stats.items():
            flat[f"{run_key}__{stat_name}"] = arr
    np.savez(path, **flat)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                         help="also run the full sensitivity sweep (slow)") # about 15 hr runtime
    args = parser.parse_args()

    train_params = TrainingParams()
    env = SpacecraftSchedulerEnv(OrbitParams(), GroundStationParams(), SpacecraftParams(), RewardParams(), train_params)
    agent = load_agent(os.path.join(CHECKPOINT_DIR, "final_model.pt"))

    print("Running detailed single-episode trajectory")
    trajectory = run_episode(env, agent, seed=0, greedy=True)
    np.savez(os.path.join(RESULTS_DIR, "example_trajectory.npz"), **trajectory)
    print(f"Episode reward: {trajectory['total_reward']:.1f}, terminated: {bool(trajectory['terminated'])}")

    print("\nRunning bulk greedy evaluation (20 episodes)")
    eval_stats = evaluate_agent(agent, env, n_episodes=20)
    print(f"Mean reward: {eval_stats['mean_reward']:.1f} +/- {eval_stats['std_reward']:.1f}")
    np.savez(os.path.join(RESULTS_DIR, "eval_stats.npz"), **eval_stats)

    if args.sweep:
        print("\nRunning sensitivity sweep, run in the background.")
        param_specs = [
            {
                "name": "solar_declination_deg", "group": "orbit",
                "values": [-23.4, 0.0, 23.4],
                "setter": lambda target, v: setattr(target, "solar_declination_deg", v),
            },
            {
                "name": "elevation_mask_deg", "group": "gs",
                "values": [5.0, 10.0, 20.0],
                "setter": lambda target, v: setattr(target, "elevation_mask_deg", v),
            },
            {
                "name": "battery_health_lower", "group": "sc",
                "values": [0.5, 0.8, 0.95],
                "setter": lambda target, v: setattr(target, "battery_health_range", (v, 1.0)),
            },
        ]
        sweep_results = run_sensitivity_sweep(param_specs, n_seeds=3, num_episodes=500, n_eval_episodes=20)
        save_sweep_results(sweep_results, os.path.join(RESULTS_DIR, "sensitivity_sweep.npz"))
        print(f"\nSweep results saved to {RESULTS_DIR}/sensitivity_sweep.npz")
    else:
        print("\n(Skipping sensitivity sweep, pass --sweep to run it.)")