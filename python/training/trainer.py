"""DQN Trainer class."""

import random
from itertools import count
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ..config import BATCH_SIZE, GAMMA, TAU, LR, MEMORY_CAPACITY, GRAD_CLIP_VALUE
from ..models import DQN
from .replay import ReplayMemory, Transition
from .utils import calculate_epsilon, save_dqn_to_gguf, dump_sampled_states_and_rewards


class DQNTrainer:
    """Encapsulates DQN training logic."""

    def __init__(
        self,
        env,
        device: torch.device,
        learning_rate: float = LR,
        memory_capacity: int = MEMORY_CAPACITY,
    ):
        self.env = env
        self.device = device
        self.learning_rate = learning_rate

        # Get dimensions from environment
        self.n_actions = env.action_space.n
        state, _ = env.reset()
        self.n_observations = len(state)

        # Initialize networks
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Initialize optimizer and memory
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=learning_rate, amsgrad=True
        )
        self.memory = ReplayMemory(memory_capacity)

        # Training state
        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select action using epsilon-greedy policy."""
        sample = random.random()
        eps_threshold = calculate_epsilon(self.steps_done)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.env.action_space.sample()]], device=self.device, dtype=torch.long
            )

    def optimize_model(self) -> None:
        """Perform one step of optimization."""
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute mask of non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        # Compute expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), GRAD_CLIP_VALUE)
        self.optimizer.step()

    def soft_update_target(self) -> None:
        """Soft update of target network weights."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            )
        self.target_net.load_state_dict(target_net_state_dict)

    def train(
        self,
        num_episodes: int,
        save_on_500: bool = False,
        callback: Optional[callable] = None,
    ) -> None:
        """Run training loop."""
        saved_after_500 = False

        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                next_state = (
                    None
                    if terminated
                    else torch.tensor(
                        observation, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                )

                self.memory.push(state, action, next_state, reward)
                state = next_state

                self.optimize_model()
                self.soft_update_target()

                # Save GGUF when reaching 500 steps
                if save_on_500 and (not saved_after_500) and (t + 1 >= 500):
                    save_dqn_to_gguf(
                        self.policy_net,
                        f"policy_net_ep{i_episode + 1}_step{t + 1}.gguf",
                        name="policy_net",
                    )
                    save_dqn_to_gguf(
                        self.target_net,
                        f"target_net_ep{i_episode + 1}_step{t + 1}.gguf",
                        name="target_net",
                    )
                    dump_sampled_states_and_rewards(
                        self.memory,
                        100,
                        prefix=f"sample_ep{i_episode + 1}_step{t + 1}",
                    )
                    saved_after_500 = True
                    print(
                        f"Saved policy/target nets to GGUF + 100 sampled (state,reward) binaries "
                        f"at episode {i_episode + 1}, step {t + 1}"
                    )

                if done:
                    self.episode_durations.append(t + 1)
                    if callback:
                        callback(i_episode, t + 1)
                    break

        print("Training complete")
