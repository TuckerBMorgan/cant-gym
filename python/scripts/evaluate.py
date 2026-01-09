#!/usr/bin/env python3
"""Evaluate a trained DQN model from GGUF."""

import argparse

import torch

from ..inference import build_policy_net, load_policy_from_gguf, load_states_bin


def main():
    ap = argparse.ArgumentParser(description="Run inference with a GGUF policy")
    ap.add_argument("--policy", required=True, help="policy_net.gguf")
    ap.add_argument("--states", required=True, help="*_states_f32.bin")
    ap.add_argument("--out", default="policy_outputs.txt", help="Output file")
    ap.add_argument("--obs-dim", type=int, default=4)
    ap.add_argument("--hidden1", type=int, default=128)
    ap.add_argument("--hidden2", type=int, default=128)
    ap.add_argument("--actions", type=int, default=2)
    ap.add_argument(
        "--argmax", action="store_true", help="Also write argmax action per state"
    )
    args = ap.parse_args()

    policy = build_policy_net(args.obs_dim, args.hidden1, args.hidden2, args.actions)
    load_policy_from_gguf(args.policy, policy)
    policy.eval()

    states = load_states_bin(args.states, args.obs_dim)

    with torch.no_grad():
        out = policy(states)

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
