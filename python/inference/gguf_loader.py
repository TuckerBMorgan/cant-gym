"""GGUF model loading utilities."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from ..config import HIDDEN_LAYER_SIZE


def load_states_bin(path: str, obs_dim: int) -> torch.Tensor:
    """Load states from a binary file."""
    with open(path, "rb") as f:
        data = f.read()
    floats = np.frombuffer(data, dtype=np.float32)
    if floats.size % obs_dim != 0:
        raise ValueError(
            f"State file has {floats.size} floats, not divisible by obs_dim={obs_dim}"
        )
    states = floats.reshape(-1, obs_dim)
    # Make writable (avoids torch warning)
    return torch.from_numpy(states.copy())


def build_policy_net(
    obs_dim: int,
    hidden1: int = HIDDEN_LAYER_SIZE,
    hidden2: int = HIDDEN_LAYER_SIZE,
    n_actions: int = 2,
) -> nn.Module:
    """Build a policy network matching the GGUF structure.

    MUST match the net you used when saving to gguf.
    Names are chosen to match GGUF keys: layer1.*, layer2.*, layer3.*
    """
    return nn.Sequential(
        OrderedDict(
            [
                ("layer1", nn.Linear(obs_dim, hidden1)),
                ("relu1", nn.ReLU()),
                ("layer2", nn.Linear(hidden1, hidden2)),
                ("relu2", nn.ReLU()),
                ("layer3", nn.Linear(hidden2, n_actions)),
            ]
        )
    )


def load_policy_from_gguf(path: str, model: nn.Module) -> None:
    """Load policy weights from a GGUF file into the model."""
    from gguf import GGUFReader

    reader = GGUFReader(path)

    # Collect gguf tensors into a dict
    gguf_tensors = {}
    for t in reader.tensors:
        gguf_tensors[t.name] = torch.from_numpy(t.data.copy())

    # Build a state_dict that matches the torch model keys exactly
    model_sd = model.state_dict()
    new_sd = {}

    missing = []
    for k in model_sd.keys():
        if k in gguf_tensors:
            w = gguf_tensors[k]
            # Ensure dtype matches model (usually float32)
            if w.dtype != model_sd[k].dtype:
                w = w.to(dtype=model_sd[k].dtype)
            new_sd[k] = w
        else:
            missing.append(k)

    if missing:
        available = sorted(list(gguf_tensors.keys()))
        raise RuntimeError(
            "GGUF did not contain required keys for this model.\n"
            f"Missing keys: {missing}\n"
            f"Available GGUF keys (first 50): {available[:50]}"
        )

    model.load_state_dict(new_sd, strict=True)
