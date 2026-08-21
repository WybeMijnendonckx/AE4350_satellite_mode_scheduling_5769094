import numpy as np
import matplotlib.pyplot as plt

from orbital_dynamics import OrbitPropagator, orbital_period_s
from config import OrbitParams, GroundStationParams, SpacecraftParams, RewardParams, TrainingParams

from evaluate import load_agent, evaluate_agent
from environment import SpacecraftSchedulerEnv


orbit_params = OrbitParams()  
gs_params = GroundStationParams() 

prop = OrbitPropagator(orbit_params, gs_params)
rev = 15
T = orbital_period_s(orbit_params.altitude_km)
dt = 5.0  # seconds
ts = np.arange((rev-1)*T, rev*T, dt)
t_min = ts / 60.0

eclipse_flags = np.array([prop.is_eclipse(t) for t in ts])
contact_flags = np.array([prop.is_ground_contact(t) for t in ts])
elevations = np.array([prop.elevation_deg(t) for t in ts])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(t_min, elevations, color="black", linewidth=1)
ax1.axhline(gs_params.elevation_mask_deg, color="gray", linestyle="--", linewidth=0.8, label="elevation mask")
ax1.fill_between(t_min, -90, 90, where=eclipse_flags, color="navy", alpha=0.2, label="eclipse")
ax1.fill_between(t_min, -90, 90, where=contact_flags, color="orange", alpha=0.4, label="ground contact")
ax1.set_ylabel("Elevation to Redu [deg]")
ax1.set_ylim(-90, 90)
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title(f"Orbit geometry over one period (T = {T/60:.2f} min), {orbit_params.altitude_km:.0f} km SSO")

ax2.fill_between(t_min, 0, 1, where=eclipse_flags, step="mid", color="navy", alpha=0.5, label="eclipse")
ax2.fill_between(t_min, 1, 2, where=contact_flags, step="mid", color="orange", alpha=0.5, label="ground contact")
ax2.set_yticks([0.5, 1.5])
ax2.set_yticklabels(["eclipse", "contact"])
ax2.set_xlabel("Time [min]")
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("test_results/orbit_geometry_check.png", dpi=150)
plt.show()


# read episode_rewards training metrics and plot
data = np.load("results/training_metrics_unshaped.npz")
traj_data = np.load('results/example_trajectory_unshaped.npz')
episode_rewards = data["episode_rewards"]
episode_lengths = data["episode_lengths"]
imaging_active = traj_data['imaging_active']
print(f"Training metrics loaded: {len(episode_rewards)} episodes, mean reward: {episode_rewards.mean():.1f}, mean length: {episode_lengths.mean():.1f} steps")
print(f'Fraction while imaging: {imaging_active.mean():.6f}')


# Plot training metrics and rolling mean and linear regression line
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
roll = 50
con = np.convolve(episode_rewards, np.ones(roll)/roll, mode='valid')
plt.plot(np.arange(roll-1, len(con)+(roll-1)), con, color='red', linewidth=2, label=f'Rolling mean ({roll} episodes)')
plt.plot(np.arange(len(episode_rewards)), np.poly1d(np.polyfit(np.arange(len(episode_rewards)), episode_rewards, 1))(np.arange(len(episode_rewards))), color='green', linewidth=2, label='Linear regression')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.legend()
plt.show()




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

print("\nShaped (0.05):")
true_downlinked_data("checkpoints/final_model_shaped.pt")  
print("Unshaped (0.0):")
true_downlinked_data("checkpoints/final_model_unshaped.pt")