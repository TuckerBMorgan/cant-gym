"""DQN Training Configuration.

All hyperparameters and constants consolidated here.
"""

# Number of transitions sampled from replay buffer per optimization step
BATCH_SIZE = 128

# Discount factor for future rewards
GAMMA = 0.99

# Epsilon-greedy exploration parameters
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500

# Target network soft update rate
TAU = 0.005

# Learning rate
LR = 3e-4

# Replay memory capacity
MEMORY_CAPACITY = 10_000

# Network architecture
HIDDEN_LAYER_SIZE = 128

# Gradient clipping threshold
GRAD_CLIP_VALUE = 100.0
