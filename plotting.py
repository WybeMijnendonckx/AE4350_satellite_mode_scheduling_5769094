import matplotlib.pyplot as plt
import numpy as np
from orbital_dynamics import OrbitPropagator, orbital_period_s
from config import OrbitParams, GroundStationParams, SpacecraftParams
import re

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

    plt.figure(figsize=(10, 6))

    plt.plot(t_min, elevations, color="black", linewidth=1)
    plt.axhline(gs_params.elevation_mask_deg, color="gray", linestyle="--",
                linewidth=0.8, label="elevation mask")
    plt.fill_between(t_min, -90, 90, where=eclipse_flags, color="navy",
                      alpha=0.2, label="eclipse")
    plt.fill_between(t_min, -90, 90, where=contact_flags, color="orange",
                      alpha=0.4, label="ground contact")
    plt.ylabel(f"Elevation to {gs_params.name} [deg]")
    plt.ylim(-90, 90)
    plt.legend(loc="upper right", fontsize=8)
    plt.title(f"Orbit geometry over {n_orbits} orbit(s) "
                  f"(T = {T/60:.2f} min), {orbit_params.altitude_km:.0f} km SSO")

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
    ax.axhline(1, color="gray", linewidth=0.5, linestyle="--")
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

def plot_shaping_comparison(comparison_path, output_path, show=False):
    data = np.load(comparison_path)
    unshaped_mb = data["unshaped_downlinked_mb"]
    shaped_mb = data["shaped_downlinked_mb"]
    unshaped_terminated = data["unshaped_terminated"]
    shaped_terminated = data["shaped_terminated"]

    labels = ["Unshaped\n(bonus=0.0)", "Shaped\n(bonus>0.0)"]
    means = [unshaped_mb.mean(), shaped_mb.mean()]
    stds = [unshaped_mb.std(), shaped_mb.std()]
    term_rates = [unshaped_terminated.mean() * 100, shaped_terminated.mean() * 100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.bar(labels, means, yerr=stds, capsize=6, color=["tab:gray", "tab:blue"])
    ax1.set_ylabel("Data downlinked per episode [Mb]")
    ax1.set_title("True science return (mean \u00b1 std)")

    ax2.bar(labels, term_rates, color=["tab:gray", "tab:blue"])
    ax2.set_ylabel("Termination rate [%]")
    ax2.set_title("Hard battery-floor failures")
    ax2.set_ylim(0, max(100, max(term_rates) * 1.2 + 1))

    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()

def plot_sensitivity_sweep(sensitivity_path, output_path, show=False):
    data = np.load(sensitivity_path)
    key_pattern = re.compile(r"^(.+)_(-?\d+(?:\.\d+)?)_seed(\d+)__(.+)$")

    grouped = {}
    grouped_term = {}

    for key in data.files:
        m = key_pattern.match(key)
        if m is None:
            continue
        name, value_str, seed_str, stat_name = m.groups()
        value = float(value_str)
        seed = int(seed_str)

        if stat_name == "mean_reward":
            grouped.setdefault(name, {}).setdefault(value, {})[seed] = float(data[key])
        elif stat_name == "terminated":
            grouped_term.setdefault(name, {}).setdefault(value, {})[seed] = float(np.mean(data[key]))

    param_names = sorted(grouped.keys())
    n_params = len(param_names)
    if n_params == 0:
        raise ValueError(f"No sweep keys matched the expected pattern in {sensitivity_path}")

    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4.5))
    if n_params == 1:
        axes = [axes]

    for ax, name in zip(axes, param_names):
        values = sorted(grouped[name].keys())
        means = []
        stds = []
        term_rates = []
        for v in values:
            seed_means = np.array(list(grouped[name][v].values()))
            means.append(seed_means.mean())
            stds.append(seed_means.std())

            if name in grouped_term and v in grouped_term[name]:
                seed_term_rates = np.array(list(grouped_term[name][v].values()))
                term_rates.append(seed_term_rates.mean())
            else:
                term_rates.append(None)

        ax.margins(y=0.25)
        ax.errorbar(values, means, yerr=stds, marker="o", capsize=4, color="tab:blue")

        for v, m, s, term_rate in zip(values, means, stds, term_rates):
            if term_rate is None:
                continue
            ax.annotate(
                f"{term_rate:.0%} term.",
                xy=(v, m + s), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8, color="dimgray",
            )

        ax.set_xlabel(name)
        ax.set_ylabel("Mean reward (across seeds)")
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    #plot_rewards("results/training_metrics_unshaped.npz", "plot_results/training_progress_unshaped.pdf", roll=50, show=False)
    #plot_rewards("results/training_metrics_shaped.npz", "plot_results/training_progress_shaped.pdf", roll=50, show=False)
    #plot_orbit_geometry("plot_results/orbit_geometry_check.pdf", n_orbits=15, show=False)
    #plot_episode_trajectory("results/example_trajectory_unshaped.npz", "plot_results/example_trajectory_unshaped.pdf", battery_safety_floor_frac=SpacecraftParams().battery_safety_floor_frac, battery_health_hard_floor=SpacecraftParams().battery_health_hard_floor, show=True)
    #plot_episode_trajectory("results/example_trajectory_shaped.npz", "plot_results/example_trajectory_shaped.pdf", battery_safety_floor_frac=SpacecraftParams().battery_safety_floor_frac, battery_health_hard_floor=SpacecraftParams().battery_health_hard_floor, show=True)
    #plot_shaping_comparison("results/shaping_ablation.npz", "plot_results/shaping_comparison.pdf", show=True)
    plot_sensitivity_sweep("results/sensitivity_sweep_6_seeds.npz", "plot_results/sensitivity_sweep.pdf", show=True)