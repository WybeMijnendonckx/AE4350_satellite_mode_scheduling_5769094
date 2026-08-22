"""main.py

used for random tests and debugging experiments, not part of the 
main training/evaluation pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt

from orbital_dynamics import OrbitPropagator, orbital_period_s
from config import OrbitParams, GroundStationParams, SpacecraftParams, RewardParams, TrainingParams

from evaluate import load_agent, evaluate_agent
from environment import SpacecraftSchedulerEnv


orbit_params = OrbitParams()  
gs_params = GroundStationParams()

env = SpacecraftSchedulerEnv(OrbitParams(), GroundStationParams(), SpacecraftParams(), RewardParams(), TrainingParams())
agent = load_agent("checkpoints/final_model_unshaped.pt")

stats = evaluate_agent(agent, env, n_episodes=50)
print(f"\nmean reward: {stats['mean_reward']:.1f} +/- {stats['std_reward']:.1f}")
print(f"terminated (hard battery floor) fraction: {stats['terminated'].mean():.4%}")
print(f"reward distribution: min={stats['rewards'].min():.0f}, max={stats['rewards'].max():.0f}")

def true_downlinked_data(checkpoint_path, n_episodes=50):
    env = SpacecraftSchedulerEnv(OrbitParams(), GroundStationParams(), SpacecraftParams(), RewardParams(), TrainingParams())
    agent = load_agent(checkpoint_path)

    totals = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=100 + i)
        terminated = truncated = False
        total_mb = 0.0
        while not (terminated or truncated):
            geom = env.propagator.get_geometry(env.t)
            mask = env.get_action_mask(geom["is_eclipse"])
            action = agent.select_action(obs, mask, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_mb += info["data_downlinked_mb"]
        totals.append(total_mb)
    totals = np.array(totals)
    print(f"actual data downlinked: mean={totals.mean():.1f} +/- {totals.std():.1f}, "
          f"min={totals.min():.1f}, max={totals.max():.1f}")
    return totals

print("\nShaped (0.00005):")
true_downlinked_data("checkpoints/final_model_shaped.pt")  
print("Unshaped (0.0):")
true_downlinked_data("checkpoints/final_model_unshaped.pt")



from train import train

agent, _ = train(num_episodes=300, seed=42, run_name="calibration")

sc = SpacecraftParams()
env = SpacecraftSchedulerEnv(OrbitParams(), GroundStationParams(), sc,
                             RewardParams(), TrainingParams())

max_buffer, mean_buffer, imaging_duty, downlink_duty, full_frac, downlinked = [], [], [], [], [], []

for i in range(20):
    obs, info = env.reset(seed=500 + i)
    terminated = truncated = False
    buf, img, dwn, total_mb, terminated_flags = [], [], [], 0.0, []

    while not (terminated or truncated):
        geom = env.propagator.get_geometry(env.t)
        mask = env.get_action_mask(geom["is_eclipse"])
        action = agent.select_action(obs, mask, greedy=True)
        obs, reward, terminated, truncated, info = env.step(action)

        buf.append(env.buffer_mb / sc.buffer_capacity_mb)
        img.append(info["imaging_active"])
        dwn.append(info["downlink_active"])
        total_mb += info["data_downlinked_mb"]

    buf, img, dwn = np.array(buf), np.array(img), np.array(dwn)
    max_buffer.append(buf.max())
    mean_buffer.append(buf.mean())
    full_frac.append((buf > 0.99).mean())
    imaging_duty.append(img.mean())
    downlink_duty.append(dwn.mean())
    downlinked.append(total_mb)
    terminated_flags.append(terminated)

print(f"buffer peak (fraction):   mean={np.mean(max_buffer):.3f}, "
      f"min={np.min(max_buffer):.3f}, max={np.max(max_buffer):.3f}")
print(f"buffer mean (fraction):   {np.mean(mean_buffer):.3f}")
print(f"steps spent at >99% full: {np.mean(full_frac):.2%}")
print(f"imaging duty cycle:       {np.mean(imaging_duty):.2%}  (target ~15-20%)")
print(f"downlink duty cycle:      {np.mean(downlink_duty):.2%}  (contact windows are ~2% of the day)")
print(f"data downlinked per day:  {np.mean(downlinked):.0f} Mb  (theoretical max ~75000)")
print(f"termination rate (hard battery floor): {np.mean(terminated_flags):.1%}")