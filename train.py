"""train.py

Runs training loop and saves model checkpoints.
"""

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

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"


def make_env(train_params: TrainingParams) -> SpacecraftSchedulerEnv:
    return SpacecraftSchedulerEnv(
        OrbitParams(),
        GroundStationParams(),
        SpacecraftParams(),
        RewardParams(),
        train_params,
    )


def train(num_episodes: int = None, seed: int = None):
    train_params = TrainingParams()
    if num_episodes is not None:
        train_params.num_episodes = num_episodes
    if seed is not None:
        train_params.seed = seed

    np.random.seed(train_params.seed)

    env = make_env(train_params)
    rl_params = RLParams()
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=env.action_space.n, rl_params=rl_params)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # metric logs per episode, for plotting later on
    episode_rewards = []
    episode_lengths = []
    episode_terminated = []      # True = hard battery failure
    episode_final_battery_health = []
    episode_final_epsilon = []
    episode_losses = []          # mean training loss over the episode

    best_reward = -np.inf

    for episode in range(train_params.num_episodes):
        obs, info = env.reset(seed=train_params.seed + episode)
        episode_reward = 0.0
        losses = []
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            geom = env.propagator.get_geometry(env.t)
            mask = env.get_action_mask(geom["is_eclipse"])

            action = agent.select_action(obs, mask, greedy=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, terminated)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            episode_reward += reward
            obs = next_obs
            steps += 1
            episode_rewards.append(episode_reward)

        episode_lengths.append(steps)
        episode_terminated.append(terminated)
        episode_final_battery_health.append(env.battery_health)
        episode_final_epsilon.append(agent.epsilon)
        episode_losses.append(np.mean(losses) if losses else np.nan)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(CHECKPOINT_DIR, "best_model.pt"))

        if (episode + 1) % 50 == 0:
            recent = episode_rewards[-50:]
            print(
                f"episode {episode + 1}/{train_params.num_episodes} | "
                f"reward: {episode_reward:.1f} | "
                f"mean(last 50): {np.mean(recent):.1f} | "
                f"epsilon: {agent.epsilon:.3f} | "
                f"terminated: {terminated} | "
                f"battery_health: {env.battery_health:.3f}"
            )

    agent.save(os.path.join(CHECKPOINT_DIR, "final_model.pt"))

    np.savez(
        os.path.join(RESULTS_DIR, "training_metrics.npz"),
        episode_rewards=np.array(episode_rewards),
        episode_lengths=np.array(episode_lengths),
        episode_terminated=np.array(episode_terminated),
        episode_final_battery_health=np.array(episode_final_battery_health),
        episode_final_epsilon=np.array(episode_final_epsilon),
        episode_losses=np.array(episode_losses),
    )

    print(f"\nTraining complete. Best episode reward: {best_reward:.1f}")
    print(f"Final model saved to {CHECKPOINT_DIR}/final_model.pt")
    print(f"Best model saved to {CHECKPOINT_DIR}/best_model.pt")
    print(f"Metrics saved to {RESULTS_DIR}/training_metrics.npz")

    return agent, env


if __name__ == "__main__":
    train(num_episodes=50)