import matplotlib.pyplot as plt
import numpy as np
from orbital_dynamics import OrbitPropagator, orbital_period_s
from config import OrbitParams, GroundStationParams, SpacecraftParams

def plot_rewards(datafile, output_path, roll=50, show=False):
    data = np.load(datafile)
    episode_rewards = data["episode_rewards"]
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    con = np.convolve(episode_rewards, np.ones(roll)/roll, mode='valid')
    plt.plot(np.arange(roll-1, len(con)+(roll-1)), con, color='red', linewidth=2, label=f'Rolling mean ({roll} episodes)')
    plt.plot(np.arange(len(episode_rewards)), np.poly1d(np.polyfit(np.arange(len(episode_rewards)), episode_rewards, 1))(np.arange(len(episode_rewards))), color='green', linewidth=2, label='Linear regression')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()

def plot_orbit_geometry(output_path, orbit_params=None, gs_params=None, n_orbits=1, show=False):
    orbit_params = orbit_params or OrbitParams()
    gs_params = gs_params or GroundStationParams()
    prop = OrbitPropagator(orbit_params, gs_params)

    T = orbital_period_s(orbit_params.altitude_km)
    dt = 5.0
    ts = np.arange(0, n_orbits * T, dt)
    t_min = ts / 60.0

    eclipse_flags = np.array([prop.is_eclipse(t) for t in ts])
    contact_flags = np.array([prop.is_ground_contact(t) for t in ts])
    elevations = np.array([prop.elevation_deg(t) for t in ts])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t_min, elevations, color="black", linewidth=1)
    ax1.axhline(gs_params.elevation_mask_deg, color="gray", linestyle="--",
                linewidth=0.8, label="elevation mask")
    ax1.fill_between(t_min, -90, 90, where=eclipse_flags, color="navy",
                      alpha=0.2, label="eclipse")
    ax1.fill_between(t_min, -90, 90, where=contact_flags, color="orange",
                      alpha=0.4, label="ground contact")
    ax1.set_ylabel(f"Elevation to {gs_params.name} [deg]")
    ax1.set_ylim(-90, 90)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"Orbit geometry over {n_orbits} orbit(s) "
                  f"(T = {T/60:.2f} min), {orbit_params.altitude_km:.0f} km SSO")

    ax2.fill_between(t_min, 0, 1, where=eclipse_flags, step="mid",
                      color="navy", alpha=0.5, label="eclipse")
    ax2.fill_between(t_min, 1, 2, where=contact_flags, step="mid",
                      color="orange", alpha=0.5, label="ground contact")
    ax2.set_yticks([0.5, 1.5])
    ax2.set_yticklabels(["eclipse", "contact"])
    ax2.set_xlabel("Time [min]")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.legend()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()

def plot_episode_trajectory(trajectory_path, output_path, battery_safety_floor_frac=None, battery_health_hard_floor=None, show=False):
    data = np.load(trajectory_path)
    t_hours = data["t"] / 3600.0
    soc_frac = data["soc_frac"]
    buffer_frac = data["buffer_frac"]
    battery_health = data["battery_health"]
    imaging_active = data["imaging_active"]
    downlink_active = data["downlink_active"]
    charging_power_w = data["charging_power_w"]
    is_eclipse = data["is_eclipse"]
    is_ground_contact = data["is_ground_contact"]
    reward = data["reward"]
    total_reward = float(data["total_reward"])
    terminated = bool(data["terminated"])

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    ax = axes[0]
    ax.fill_between(t_hours, 0, 1, where=is_eclipse, color="navy", alpha=0.12,
                     transform=ax.get_xaxis_transform(), label="eclipse")
    ax.plot(t_hours, soc_frac, color="tab:red", label="SOC (fraction of capacity)")
    ax.plot(t_hours, buffer_frac, color="tab:orange", label="Buffer fill (fraction of capacity)")
    ax.plot(t_hours, battery_health, color="darkred", linewidth=1.5, label="Battery health (degradation)")
    if battery_safety_floor_frac is not None:
        ax.axhline(battery_safety_floor_frac, color="red", linestyle="--",
                   linewidth=0.8, label="battery safety floor")
    if battery_health_hard_floor is not None:
        ax.axhline(battery_health_hard_floor, color="darkred", linestyle=":",
                   linewidth=1.2, label="battery hard floor (mission loss)")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Episode trajectory (total reward={total_reward:.0f}, terminated={terminated})")

    ax = axes[1]
    ax.fill_between(t_hours, 2, 3, where=is_ground_contact, step="mid", alpha=0.6,
                     color="gold", label="ground contact")
    ax.fill_between(t_hours, 0, 1, where=imaging_active, step="mid", color="seagreen",
                     alpha=0.6, label="imaging active")
    ax.fill_between(t_hours, 1, 2, where=downlink_active, step="mid", color="purple",
                     alpha=0.6, label="downlink active")
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["imaging", "downlink", 'ground contact'])
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.plot(t_hours, charging_power_w, color="darkgreen")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Charging power [W]")

    ax = axes[3]
    ax.plot(t_hours, np.cumsum(reward), color="black")
    ax.set_ylabel("Cumulative reward")
    ax.set_xlabel("Time [hours]")

    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    plot_rewards("results/training_metrics_unshaped.npz", "plot_results/training_progress_unshaped.pdf", roll=50, show=False)
    plot_rewards("results/training_metrics_shaped.npz", "plot_results/training_progress_shaped.pdf", roll=50, show=False)
    plot_orbit_geometry("plot_results/orbit_geometry_check.pdf", n_orbits=30, show=False)
    plot_episode_trajectory("results/example_trajectory_unshaped.npz", "plot_results/example_trajectory_unshaped.pdf", battery_safety_floor_frac=SpacecraftParams().battery_safety_floor_frac, battery_health_hard_floor=SpacecraftParams().battery_health_hard_floor, show=True)
    plot_episode_trajectory("results/example_trajectory_shaped.npz", "plot_results/example_trajectory_shaped.pdf", battery_safety_floor_frac=SpacecraftParams().battery_safety_floor_frac, battery_health_hard_floor=SpacecraftParams().battery_health_hard_floor, show=True)