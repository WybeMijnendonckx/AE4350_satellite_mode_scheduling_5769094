"""agent.py

Double DQN agent
"""

from collections import deque
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from networks import QNetwork
from environment import Action, OBS_ECLIPSE # maybe have to change this when renaming

DEVICE = torch.device("cpu")


def build_action_mask(obs_batch: np.ndarray) -> np.ndarray:
    #Only Sun-pointing is masked during eclipse and eclipse is directly observable at OBS_ECLIPSE
    batch_size = obs_batch.shape[0]
    mask = np.ones((batch_size, 4), dtype=bool)
    is_eclipse = obs_batch[:, OBS_ECLIPSE] > 0.5
    mask[:, Action.SUN_POINT] = ~is_eclipse
    return mask


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, terminated):
        self.buffer.append((obs, action, reward, next_obs, terminated))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, terminated = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(terminated, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, rl_params):
        self.n_actions = n_actions
        self.gamma = rl_params.gamma
        self.batch_size = rl_params.batch_size
        self.target_update_freq = rl_params.target_update_freq

        self.epsilon_start = rl_params.epsilon_start
        self.epsilon_end = rl_params.epsilon_end
        self.epsilon_decay_steps = rl_params.epsilon_decay_steps
        self.train_step_count = 0

        self.online_net = QNetwork(obs_dim, n_actions, rl_params.hidden_layer_sizes).to(DEVICE)
        self.target_net = QNetwork(obs_dim, n_actions, rl_params.hidden_layer_sizes).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=rl_params.learning_rate)
        self.replay_buffer = ReplayBuffer(rl_params.replay_buffer_size)

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.train_step_count / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        valid_actions = np.flatnonzero(mask)

        if not greedy and random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q_values = self.online_net(obs_t).squeeze(0).cpu().numpy()

        q_values_masked = np.where(mask, q_values, -np.inf)
        return int(np.argmax(q_values_masked))

    def store_transition(self, obs, action, reward, next_obs, terminated):
        self.replay_buffer.push(obs, action, reward, next_obs, terminated)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None  # not enough data yet

        obs, action, reward, next_obs, terminated = self.replay_buffer.sample(self.batch_size)

        obs_t = torch.as_tensor(obs, device=DEVICE)
        action_t = torch.as_tensor(action, device=DEVICE)
        reward_t = torch.as_tensor(reward, device=DEVICE)
        next_obs_t = torch.as_tensor(next_obs, device=DEVICE)
        terminated_t = torch.as_tensor(terminated, device=DEVICE)

        # current Q-values for the actions actually taken
        q_values = self.online_net(obs_t)
        q_sa = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

        # Double DQN target
        next_mask = build_action_mask(next_obs)
        with torch.no_grad():
            next_q_online = self.online_net(next_obs_t).cpu().numpy()
            next_q_online_masked = np.where(next_mask, next_q_online, -np.inf)
            best_next_actions = torch.as_tensor(
                np.argmax(next_q_online_masked, axis=1), device=DEVICE
            )

            next_q_target = self.target_net(next_obs_t)
            next_q_sa = next_q_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            td_target = reward_t + self.gamma * (1.0 - terminated_t) * next_q_sa

        loss = nn.functional.mse_loss(q_sa, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location=DEVICE)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)