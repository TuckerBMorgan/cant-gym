"""Training utility functions."""

import math
import numpy as np
import torch
import torch.nn as nn

from ..config import EPS_START, EPS_END, EPS_DECAY


def calculate_epsilon(steps_done: int) -> float:
    """Calculate epsilon for epsilon-greedy exploration."""
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)


def save_dqn_to_gguf(
    model: nn.Module, out_path: str, *, architecture: str = "dqn", name: str = "dqn"
) -> None:
    """Save a simple PyTorch DQN MLP to a GGUF file.

    Notes:
      - This writes float32 tensors (no quantization).
      - The resulting GGUF is meant as a portable tensor bundle; it's not
        guaranteed to be directly loadable by llama.cpp runtimes without
        additional, architecture-specific metadata.
    """
    try:
        import gguf
    except ImportError:
        raise RuntimeError(
            "gguf package is not installed. Add it to your env (pip install gguf) to enable GGUF export."
        )

    state = model.state_dict()
    writer = gguf.GGUFWriter(out_path, architecture)

    # Best-effort metadata (API differs across gguf versions)
    if hasattr(writer, "add_name"):
        try:
            writer.add_name(name)
        except Exception:
            pass
    if hasattr(writer, "add_architecture"):
        try:
            writer.add_architecture(architecture)
        except Exception:
            pass

    for k, v in state.items():
        arr = v.detach().cpu().numpy()
        # Ensure contiguous float32
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        arr = np.ascontiguousarray(arr)
        writer.add_tensor(k, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def dump_sampled_states_and_rewards(
    memory, n: int, *, prefix: str
) -> None:
    """Sample transitions and write (state, reward) to two raw float32 binary files.

    Output format (no header):
      - "{prefix}_states_f32.bin": concatenated float32 state vectors
      - "{prefix}_rewards_f32.bin": float32 rewards

    For CartPole, each state is length-4, so the states file will contain
    n*4 float32s.
    """
    if len(memory) == 0:
        raise RuntimeError("ReplayMemory is empty; cannot sample.")

    k = min(int(n), len(memory))
    batch = memory.sample(k)

    # Each stored state is a tensor shaped [1, obs_dim]. Rewards are [1].
    states_np = np.stack(
        [tr.state.detach().to("cpu").to(torch.float32).view(-1).numpy() for tr in batch],
        axis=0,
    ).astype(np.float32, copy=False)

    rewards_np = np.stack(
        [tr.reward.detach().to("cpu").to(torch.float32).view(-1).numpy() for tr in batch],
        axis=0,
    ).astype(np.float32, copy=False)

    # Write raw float32 (little-endian) with no header.
    states_np.tofile(f"{prefix}_states_f32.bin")
    rewards_np.tofile(f"{prefix}_rewards_f32.bin")
