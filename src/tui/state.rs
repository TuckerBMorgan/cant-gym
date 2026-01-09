use std::collections::HashMap;

use crate::config::EPS_START;

#[derive(Clone)]
pub struct TrainingUpdate {
    pub episode: usize,
    pub episode_duration: usize,
    pub steps_done: usize,
    pub epsilon: f32,
    pub memory_size: usize,
    pub training_complete: bool,
    pub timings: HashMap<String, u128>,
}

pub struct DashboardState {
    pub episode_durations: Vec<(f64, f64)>,
    pub epsilon_history: Vec<(f64, f64)>,
    pub current_episode: usize,
    pub current_epsilon: f32,
    pub total_steps: usize,
    pub memory_size: usize,
    pub best_duration: usize,
    pub avg_duration_100: f64,
    pub training_complete: bool,
    pub timing: HashMap<String, u128>,
}

impl DashboardState {
    pub fn new() -> Self {
        DashboardState {
            episode_durations: Vec::new(),
            epsilon_history: Vec::new(),
            current_episode: 0,
            current_epsilon: EPS_START,
            total_steps: 0,
            memory_size: 0,
            best_duration: 0,
            avg_duration_100: 0.0,
            training_complete: false,
            timing: HashMap::new(),
        }
    }

    pub fn update(&mut self, update: TrainingUpdate) {
        self.current_episode = update.episode;
        self.current_epsilon = update.epsilon;
        self.total_steps = update.steps_done;
        self.memory_size = update.memory_size;
        self.training_complete = update.training_complete;
        self.timing = update.timings;

        if update.episode_duration > 0 {
            self.episode_durations
                .push((update.episode as f64, update.episode_duration as f64));
            self.epsilon_history
                .push((update.episode as f64, update.epsilon as f64));

            if update.episode_duration > self.best_duration {
                self.best_duration = update.episode_duration;
            }

            // Calculate moving average of last 100 episodes
            let recent: Vec<f64> = self
                .episode_durations
                .iter()
                .rev()
                .take(100)
                .map(|(_, d)| *d)
                .collect();
            if !recent.is_empty() {
                self.avg_duration_100 = recent.iter().sum::<f64>() / recent.len() as f64;
            }
        }
    }
}

impl Default for DashboardState {
    fn default() -> Self {
        Self::new()
    }
}
