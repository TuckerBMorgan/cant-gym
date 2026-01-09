#!/usr/bin/env python3
"""DQN CartPole training script."""

import argparse

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch

from ..config import LR
from ..training import DQNTrainer


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_plotting():
    """Set up matplotlib for training visualization."""
    is_ipython = "inline" in matplotlib.get_backend()
    plt.ion()
    return is_ipython


def plot_durations(episode_durations, is_ipython: bool, show_result: bool = False):
    """Plot episode durations."""
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")

    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        from IPython import display

        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def main():
    parser = argparse.ArgumentParser(description="Train DQN on CartPole")
    parser.add_argument(
        "--episodes", type=int, default=None, help="Number of episodes (default: 600 on GPU, 50 on CPU)"
    )
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument(
        "--save-gguf", action="store_true", help="Save GGUF when reaching 500 steps"
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Set number of episodes based on device if not specified
    if args.episodes is None:
        num_episodes = 600 if device.type in ("cuda", "mps") else 50
    else:
        num_episodes = args.episodes

    # Create environment and trainer
    env = gym.make("CartPole-v1")
    trainer = DQNTrainer(env, device, learning_rate=args.lr)

    # Setup plotting
    is_ipython = False
    if not args.no_plot:
        is_ipython = setup_plotting()

    # Create callback for plotting
    def on_episode_end(episode: int, duration: int):
        if not args.no_plot:
            plot_durations(trainer.episode_durations, is_ipython)

    # Train
    trainer.train(
        num_episodes=num_episodes, save_on_500=args.save_gguf, callback=on_episode_end
    )

    # Show final results
    if not args.no_plot:
        plot_durations(trainer.episode_durations, is_ipython, show_result=True)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
