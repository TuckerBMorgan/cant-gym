//! DQN Training Configuration
//!
//! All hyperparameters and constants consolidated here.

/// Number of transitions sampled from replay buffer per optimization step
pub const BATCH_SIZE: usize = 128;

/// Discount factor for future rewards
pub const GAMMA: f32 = 0.99;

/// Epsilon-greedy exploration parameters
pub const EPS_START: f32 = 0.9;
pub const EPS_END: f32 = 0.01;
pub const EPS_DECAY: f32 = 2500.0;

/// Target network soft update rate
pub const TAU: f32 = 0.005;

/// Default learning rate
pub const LR: f32 = 3e-2;

/// Total training episodes
pub const TOTAL_EPISODES: usize = 600;

/// Replay memory capacity
pub const MEMORY_CAPACITY: usize = 10_000;

/// Network architecture
pub const HIDDEN_LAYER_SIZE: usize = 128;
pub const OBSERVATION_SIZE: usize = 4;

/// Gradient clipping threshold
pub const GRAD_CLIP_VALUE: f32 = 100.0;

/// Maximum steps per episode (CartPole-v1 limit)
pub const MAX_STEPS_PER_EPISODE: usize = 500;
