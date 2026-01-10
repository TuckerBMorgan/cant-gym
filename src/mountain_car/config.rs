//! MountainCar DQN Training Configuration
//!
//! Hyperparameters tuned for the sparse reward MountainCar environment.

/// Number of transitions sampled from replay buffer per optimization step
pub const BATCH_SIZE: usize = 128;

/// Discount factor for future rewards
pub const GAMMA: f32 = 0.99;

/// Epsilon-greedy exploration parameters
/// Higher initial exploration due to sparse rewards
pub const EPS_START: f32 = 1.0;
pub const EPS_END: f32 = 0.01;
/// Slower decay to ensure sufficient exploration
pub const EPS_DECAY: f32 = 5000.0;

/// Target network soft update rate
pub const TAU: f32 = 0.005;

/// Default learning rate (lower than CartPole)
pub const LR: f32 = 1e-3;

/// Total training episodes (more episodes needed for sparse rewards)
pub const TOTAL_EPISODES: usize = 1000;

/// Replay memory capacity
pub const MEMORY_CAPACITY: usize = 10_000;

/// Network architecture
pub const HIDDEN_LAYER_SIZE: usize = 128;
/// MountainCar has 2 observations: position and velocity
pub const OBSERVATION_SIZE: usize = 2;
/// MountainCar has 3 actions: push left (0), no push (1), push right (2)
pub const NUM_ACTIONS: usize = 3;

/// Gradient clipping threshold
pub const GRAD_CLIP_VALUE: f32 = 100.0;

/// Maximum steps per episode (MountainCar-v0 limit)
pub const MAX_STEPS_PER_EPISODE: usize = 200;
