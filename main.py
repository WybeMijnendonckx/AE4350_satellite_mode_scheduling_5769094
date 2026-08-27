"""main.py

used for random tests and debugging experiments, not part of the 
main training/evaluation pipeline. Needs to be cleaned up for proper use.
"""

import numpy as np
from config import OrbitParams, GroundStationParams, SpacecraftParams, RewardParams, TrainingParams
from evaluate import load_agent
from environment import SpacecraftSchedulerEnv
import re



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


def training_termination_rates(results_dir="results"):
    print("=== Training termination rate (epsilon-greedy, all 1000 episodes) ===")
    for run_name in ["shaped", "unshaped"]:
        d = np.load(f"{results_dir}/training_metrics_{run_name}.npz")
        term_rate = d["episode_terminated"].mean()
        first50 = d["episode_rewards"][:50].mean()
        last50 = d["episode_rewards"][-50:].mean()
        print(f"  {run_name}: termination_rate={term_rate:.1%}, "
              f"first-50-ep mean reward={first50:.1f}, last-50-ep mean reward={last50:.1f}")
    print()

def shaping_ablation(results_dir="results"):
    print("=== Reward-shaping ablation (greedy evaluation) ===")
    d = np.load(f"{results_dir}/shaping_ablation.npz")
    for label, mb_key, term_key in [
        ("unshaped", "unshaped_downlinked_mb", "unshaped_terminated"),
        ("shaped", "shaped_downlinked_mb", "shaped_terminated"),
    ]:
        mb = d[mb_key]
        term = d[term_key]
        print(f"  {label}: downlinked_mb={mb.mean():.0f} +/- {mb.std():.0f} (n={len(mb)} episodes), "
              f"termination_rate={term.mean():.1%}")

    print("  --- cross-check against eval_stats_{shaped,unshaped}.npz ---")
    for run_name in ["shaped", "unshaped"]:
        d2 = np.load(f"{results_dir}/eval_stats_{run_name}.npz")
        print(f"  {run_name}: mean_reward={float(d2['mean_reward']):.2f} +/- "
              f"{float(d2['std_reward']):.2f} (n={len(d2['rewards'])} episodes), "
              f"termination_rate={d2['terminated'].mean():.1%}")
    print()


def sensitivity_sweep(results_dir="results", sweep_filename="sensitivity_sweep_6_seeds.npz"):
    print("=== Environment/mission-parameter sensitivity sweep ===")
    d = np.load(f"{results_dir}/{sweep_filename}")
    key_pattern = re.compile(r"^(.+)_(-?\d+(?:\.\d+)?)_seed(\d+)__(.+)$")

    grouped = {}
    for key in d.files:
        m = key_pattern.match(key)
        if m is None:
            continue
        name, value_str, seed_str, stat_name = m.groups()
        value = float(value_str)
        seed = int(seed_str)
        grouped.setdefault(name, {}).setdefault(value, {}).setdefault(seed, {})[stat_name] = d[key]

    for name in sorted(grouped.keys()):
        print(f"\n  --- {name} ---")
        for value in sorted(grouped[name].keys()):
            seeds = grouped[name][value]
            seed_ids = sorted(seeds.keys())
            mean_rewards = [float(seeds[s]["mean_reward"]) for s in seed_ids]
            term_fracs = [float(seeds[s]["terminated"].mean()) for s in seed_ids]

            reward_across_seeds = np.mean(mean_rewards)
            reward_std_across_seeds = np.std(mean_rewards)
            term_across_seeds = np.mean(term_fracs)
            term_std_across_seeds = np.std(term_fracs)

            print(f"    value={value}: "
                  f"reward={reward_across_seeds:.1f} +/- {reward_std_across_seeds:.1f} (across {len(seed_ids)} seeds), "
                  f"termination_rate={term_across_seeds:.1%} +/- {term_std_across_seeds:.1%}")
            print(f"      per-seed reward:      {[round(x, 1) for x in mean_rewards]}")
            print(f"      per-seed termination: {[f'{x:.0%}' for x in term_fracs]}")

    print()

if __name__ == "__main__":
    print("\nShaped (0.00005):")
    true_downlinked_data("checkpoints/final_model_shaped.pt")  
    print("Unshaped (0.0):")
    true_downlinked_data("checkpoints/final_model_unshaped.pt")
    print('\n')
    training_termination_rates()
    shaping_ablation()
    sensitivity_sweep()