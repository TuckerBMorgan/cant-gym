# DQN CartPole

Deep Q-Network (DQN) implementation for the CartPole-v1 environment, with both Rust and Python implementations.

## Project Structure

```
.
├── src/                    # Rust implementation
│   ├── main.rs            # Entry point with TUI dashboard
│   ├── config.rs          # Training hyperparameters
│   ├── dqn/               # DQN core
│   │   ├── model.rs       # DQN network architecture
│   │   ├── replay.rs      # Replay memory
│   │   └── training.rs    # Training loop utilities
│   ├── tui/               # Terminal UI dashboard
│   │   ├── state.rs       # Dashboard state management
│   │   └── render.rs      # UI rendering
│   └── utils/             # Utilities
│       ├── io.rs          # File I/O
│       └── timing.rs      # Performance timing
├── python/                 # Python implementation
│   ├── config.py          # Training hyperparameters
│   ├── models/
│   │   └── dqn.py         # DQN network (PyTorch)
│   ├── training/
│   │   ├── replay.py      # Replay memory
│   │   ├── trainer.py     # Training orchestration
│   │   └── utils.py       # Helpers (epsilon calc, GGUF export)
│   ├── inference/
│   │   └── gguf_loader.py # Load models from GGUF format
│   └── scripts/
│       ├── train.py       # Training entry point
│       └── evaluate.py    # Evaluation entry point
├── Cargo.toml             # Rust dependencies
├── pyproject.toml         # Python dependencies
└── requirements.txt       # Python requirements (pip)
```

## Rust Setup

### Dependencies

The Rust implementation depends on two local crates:
- `cant` - Custom tensor/neural network library
- `gym-rs` - Gymnasium environments for Rust

Clone these to sibling directories:
```bash
git clone <your-cant-repo> ../can-t
git clone <your-gym-rs-repo> ../gym-rs
```

### Build & Run

```bash
cargo build --release
cargo run --release -- --lr 0.03
```

The TUI dashboard shows:
- Episode duration over time
- Epsilon decay curve
- Training statistics
- Performance timings

Press `q` to quit.

## Python Setup

### Install

```bash
# With pip
pip install -e .

# With uv
uv pip install -e .
```

### Train

```bash
# Using entry point
dqn-train --episodes 600

# Or directly
python -m python.scripts.train --episodes 600
```

### Evaluate

```bash
dqn-eval --policy policy_net.gguf --states sample_states_f32.bin --out outputs.txt
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| BATCH_SIZE | 128 | Transitions per optimization step |
| GAMMA | 0.99 | Discount factor |
| EPS_START | 0.9 | Initial exploration rate |
| EPS_END | 0.01 | Final exploration rate |
| EPS_DECAY | 2500 | Epsilon decay rate |
| TAU | 0.005 | Target network soft update rate |
| LR | 3e-2 (Rust) / 3e-4 (Python) | Learning rate |

## Network Architecture

```
Input (4) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(2) → Output
```

The 4 inputs are CartPole observations: cart position, cart velocity, pole angle, pole angular velocity.
The 2 outputs are Q-values for actions: push left, push right.
