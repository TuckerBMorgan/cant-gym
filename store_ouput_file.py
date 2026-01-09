import argparse
import numpy as np
import torch
from collections import OrderedDict

from gguf import GGUFReader


def load_states_bin(path: str, obs_dim: int) -> torch.Tensor:
    with open(path, "rb") as f:
        data = f.read()
    floats = np.frombuffer(data, dtype=np.float32)
    if floats.size % obs_dim != 0:
        raise ValueError(f"State file has {floats.size} floats, not divisible by obs_dim={obs_dim}")
    states = floats.reshape(-1, obs_dim)
    # Make writable (avoids torch warning)
    return torch.from_numpy(states.copy())


def build_policy_net(obs_dim: int, hidden1: int = 128, hidden2: int = 128, n_actions: int = 2) -> torch.nn.Module:
    """
    MUST match the net you used when saving to gguf.
    Names are chosen to match GGUF keys: layer1.*, layer2.*, layer3.*
    """
    return torch.nn.Sequential(OrderedDict([
        ("layer1", torch.nn.Linear(obs_dim, hidden1)),
        ("relu1", torch.nn.ReLU()),
        ("layer2", torch.nn.Linear(hidden1, hidden2)),
        ("relu2", torch.nn.ReLU()),
        ("layer3", torch.nn.Linear(hidden2, n_actions)),
    ]))


def load_policy_from_gguf(path: str, model: torch.nn.Module):
    reader = GGUFReader(path)

    # Collect gguf tensors into a dict
    gguf_tensors = {}
    for t in reader.tensors:
        gguf_tensors[t.name] = torch.from_numpy(t.data.copy())  # copy -> writable & safe

    # Build a state_dict that matches the torch model keys exactly
    model_sd = model.state_dict()
    new_sd = {}

    # Direct-name match is expected: layer1.weight, layer1.bias, etc.
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
        # Helpful debugging: show what's available
        available = sorted(list(gguf_tensors.keys()))
        raise RuntimeError(
            "GGUF did not contain required keys for this model.\n"
            f"Missing keys: {missing}\n"
            f"Available GGUF keys (first 50): {available[:50]}"
        )

    model.load_state_dict(new_sd, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="policy_net.gguf")
    ap.add_argument("--states", required=True, help="*_states_f32.bin")
    ap.add_argument("--out", default="policy_outputs.txt")
    ap.add_argument("--obs-dim", type=int, default=4)
    ap.add_argument("--hidden1", type=int, default=128)
    ap.add_argument("--hidden2", type=int, default=128)
    ap.add_argument("--actions", type=int, default=2)
    ap.add_argument("--argmax", action="store_true", help="Also write argmax action per state")
    args = ap.parse_args()

    policy = build_policy_net(args.obs_dim, args.hidden1, args.hidden2, args.actions)
    load_policy_from_gguf(args.policy, policy)
    policy.eval()

    states = load_states_bin(args.states, args.obs_dim)

    with torch.no_grad():
        out = policy(states)  # [N, actions]

    with open(args.out, "w") as f:
        for i in range(out.shape[0]):
            row = out[i].tolist()
            if args.argmax:
                action = int(torch.argmax(out[i]).item())
                f.write(f"{i}\t{row}\targmax={action}\n")
            else:
                f.write(f"{i}\t{row}\n")

    print(f"Wrote {out.shape[0]} outputs to {args.out}")


if __name__ == "__main__":
    main()
