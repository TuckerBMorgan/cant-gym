use cant::{
    central::{Shape, Tensor, clip_gradients, get_equation},
    nn::{Layer, Linear, Model},
    optimizers::{AdamW, Optimizer}, utils::GGUFFile
};
use par_iter::prelude::*;
use gym_rs::{
    core::Env, envs::classical_control::cartpole::{CartPoleEnv, CartPoleObservation}, utils::renderer::RenderMode,
};
use rand::seq::SliceRandom;
use rand::Rng;
use std::{collections::{HashMap, VecDeque}, time::Instant};
use std::io::{self, stdout};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    prelude::*,
    widgets::*,
};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short='l', long="lr")]
    learning_rate: Option<f32>,
}


fn update_timings(timings: &mut HashMap<String, u128>, label: String, start: &Instant) {

    if timings.contains_key(&label) == false {
        timings.insert(label, start.elapsed().as_micros());
    }
    else {
        let update = timings[&label] + start.elapsed().as_millis();
        timings.insert(label, update);
    }
}

// ============================================================================
// DQN Components (unchanged from original)
// ============================================================================

#[derive(Clone, Copy)]
pub struct Transition {
    state: Tensor,
    action: Tensor,
    next_state: Option<Tensor>,
    reward: Tensor
}

impl Transition {
    pub fn new(state: Tensor, action: Tensor, next_state: Option<Tensor>, reward: Tensor) -> Transition {
        Transition {
            state,
            action,
            next_state,
            reward
        }
    }
}

pub struct ReplayMemory {
    memory_buffer: VecDeque<Transition>,
    max_capacity: usize
}

impl ReplayMemory {
    pub fn new(max_capacity: usize) -> ReplayMemory {
        ReplayMemory {
            memory_buffer: VecDeque::with_capacity(max_capacity),
            max_capacity
        }
    }

    pub fn push_transition(&mut self, transition: Transition) {
        if self.memory_buffer.len() == self.max_capacity {
            let old = self.memory_buffer.pop_front();
            if let Some(mut transition) = old {
                transition.action.set_keep_alive(false);
                transition.reward.set_keep_alive(false);
                transition.state.set_keep_alive(false);
                if let Some(mut next_state) = transition.next_state {
                    next_state.set_keep_alive(false);
                }
            }
        }
        self.memory_buffer.push_back(transition);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Transition> {
        assert!(
            batch_size <= self.memory_buffer.len(),
            "Cannot sample more elements than are in the replay buffer"
        );

        let mut rng = rand::thread_rng();
        self.memory_buffer
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }

    pub fn length(&self) -> usize {
        self.memory_buffer.len()
    }
}

pub struct DQN {
    layer_1: Linear,
    layer_2: Linear,
    layer_3: Linear
}

impl DQN {
    pub fn new(number_of_observations: usize, number_of_actions: usize) -> DQN {
        DQN {
            layer_1: Linear::new(number_of_observations, 128, true),
            layer_2: Linear::new(128, 128, true),
            layer_3: Linear::new(128, number_of_actions, true)
        }
    }

    pub fn from_layers(layer_1: Linear, layer_2: Linear, layer_3: Linear) -> DQN {
        DQN {
            layer_1,
            layer_2,
            layer_3
        }
    }
}

impl Model for DQN {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let x = self.layer_1.forward(input).relu();
        let x = self.layer_2.forward(x).relu();
        self.layer_3.forward(x)
    }

    fn get_parameters(&self) -> Vec<cant::central::TensorID> {
        vec![
            self.layer_1.weights.id, self.layer_1.bias.unwrap().id,
            self.layer_2.weights.id, self.layer_2.bias.unwrap().id,
            self.layer_3.weights.id, self.layer_3.bias.unwrap().id
        ]
    }
}

pub struct TransitionSOA {
    state: Vec<Tensor>,
    action: Vec<Tensor>,
    next_state: Vec<Option<Tensor>>,
    reward: Vec<Tensor>,
}

impl TransitionSOA {
    pub fn new(transitions: Vec<Transition>) -> TransitionSOA {
        let mut state = vec![];
        let mut action = vec![];
        let mut next_state = vec![];
        let mut reward = vec![];

        for transition in transitions {
            state.push(transition.state);
            action.push(transition.action);
            next_state.push(transition.next_state);
            reward.push(transition.reward);
        }

        TransitionSOA {
            state,
            action,
            next_state,
            reward
        }
    }
}

// ============================================================================
// Training Constants
// ============================================================================

const BATCH_SIZE: usize = 128;
const GAMMA: f32 = 0.99;
const EPS_START: f32 = 0.9;
const EPS_END: f32 = 0.01;
const EPS_DECAY: f32 = 2500.0;
const TAU: f32 = 0.005;
const LR: f32 = 3e-2;
const TOTAL_EPISODES: usize = 600;

// ============================================================================
// Dashboard State
// ============================================================================

#[derive(Clone)]
pub struct TrainingUpdate {
    pub episode: usize,
    pub episode_duration: usize,
    pub steps_done: usize,
    pub epsilon: f32,
    pub memory_size: usize,
    pub training_complete: bool,
    pub timings: HashMap<String, u128>
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
    pub timing: HashMap<String, u128>
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
            timing: HashMap::new()
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
            self.episode_durations.push((update.episode as f64, update.episode_duration as f64));
            self.epsilon_history.push((update.episode as f64, update.epsilon as f64));

            if update.episode_duration > self.best_duration {
                self.best_duration = update.episode_duration;
            }

            // Calculate moving average of last 100 episodes
            let recent: Vec<f64> = self.episode_durations
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

// ============================================================================
// Training Functions
// ============================================================================

fn calculate_epsilon(steps_done: usize) -> f32 {
    EPS_END + (EPS_START - EPS_END) * (-1.0 * steps_done as f32 / EPS_DECAY).exp()
}

fn select_action(state: Tensor, steps_done: usize, policy_net: &mut DQN) -> Tensor {
    let sample: f32 = rand::thread_rng().r#gen();
    let eps_threshold = calculate_epsilon(steps_done);

    if sample > eps_threshold {
        let action = policy_net.forward(state);
        let action_item = action.item();

        let mut return_index = 0usize;
        let mut best = action_item[0];

        for (i, &v) in action_item.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                return_index = i;
            }
        }
        let mut return_action = Tensor::from_vec(vec![return_index as f32], vec![1, 1]);
        return_action.set_keep_alive(true);
        return_action
    } else {
        let random_result = rand::thread_rng().gen_range(0..=1);
        let mut action = Tensor::element(Shape::new(vec![1, 1]), random_result as f32);
        action.set_keep_alive(true);
        action
    }
}

fn optimize_model(memory: &mut ReplayMemory, policy_net: &mut DQN, target_net: &mut DQN, optimizer: &mut AdamW, timings: &mut HashMap<String, u128>) {
    if memory.length() < BATCH_SIZE {
        return;
    }

    let start = Instant::now();

    let batch = memory.sample(BATCH_SIZE);
    let batch = TransitionSOA::new(batch);

    let non_final_mask: Vec<bool> = batch.next_state.iter().map(|x| x.is_some()).collect();

    let non_final_next_states: Vec<Tensor> = batch.next_state.iter().filter_map(|ns| *ns).collect();
    let mut count = 1;
    let mut non_final_next_states_batched = non_final_next_states[0];
    for nfns in non_final_next_states.iter().skip(1) {
        non_final_next_states_batched = non_final_next_states_batched.cat(*nfns, 0);
        count += 1;
    }

    non_final_next_states_batched = non_final_next_states_batched.reshape(Shape::new(vec![count, 4]));
    non_final_next_states_batched = non_final_next_states_batched.detach();

    let mut state_batch = batch.state[0].stack(batch.state[1..batch.state.len()].to_vec(), 0);
    state_batch = state_batch.detach();

    let mut action_batch = batch.action[0];
    for action in batch.action.iter().skip(1) {
        action_batch = action_batch.cat(*action, 0);
    }
    action_batch = action_batch.detach();

    let mut reward_batch = batch.reward[0];
    for reward in batch.reward.iter().skip(1) {
        reward_batch = reward_batch.cat(*reward, 0);
    }
    reward_batch = reward_batch.detach();

    let state_action_values = policy_net.forward(state_batch).gather(1, action_batch);
    let next_state_values = Tensor::zeros(Shape::new(vec![BATCH_SIZE]));

    let mut target_value = target_net.forward(non_final_next_states_batched).max(1, false);
    target_value = target_value.detach();
    let target_value = target_value.item();

    let mut j = 0;
    for i in 0..non_final_mask.len() {
        if non_final_mask[i] {
            next_state_values.set_index(cant::central::Indexable::Single(i), target_value[j]);
            j += 1;
        }
    }

    let expected_state_action_values = (next_state_values * GAMMA) + reward_batch;
    let loss = state_action_values.l1_loss(expected_state_action_values);

    update_timings(timings, String::from("LossCalculate"), &start);
    let start = Instant::now();
    optimizer.zero_grads();
    loss.backward();
    update_timings(timings, String::from("Baclwards"), &start);
    let start = Instant::now();
    clip_gradients(100.0);
    optimizer.update();
    update_timings(timings, String::from("optimizer"), &start);
    let start = Instant::now();
    get_equation().compact_tensor_store();
}

fn update_layer(policy_layer: &Linear, target_layer: &mut Linear, timings: &mut HashMap<String, u128>) {
    let mut start = Instant::now();

    let layer_weight_policy: Vec<f32> = policy_layer.weights.item().iter().map(|x| *x * TAU).collect();
    let layer_bias_policy: Vec<f32> = policy_layer.bias.unwrap().item().iter().map(|x| *x * TAU).collect();

    let layer_weight_target: Vec<f32> = target_layer.weights.item().iter().map(|x| *x * (1.0 - TAU)).collect();
    let layer_bias_target: Vec<f32> = target_layer.bias.unwrap().item().iter().map(|x| *x * (1.0 - TAU)).collect();

    let updated_target_layer_weight: Vec<f32> = layer_weight_policy.par_iter().zip(layer_weight_target).map(|(policy, target)| policy + target).collect();
    let updated_target_layer_bias: Vec<f32> = layer_bias_policy.par_iter().zip(layer_bias_target).map(|(policy, target)| policy + target).collect();
    
   // update_timings(timings, String::from("TauUpdate"), &start);

    let mut start = Instant::now();
    let mut target_layer_new_weight_tensor = Tensor::from_vec(updated_target_layer_weight, target_layer.weights.shape.dimensions());
    target_layer_new_weight_tensor.set_keep_alive(true);
    target_layer_new_weight_tensor.set_requires_grad(false);

    target_layer.weights.set_keep_alive(false);
    target_layer.weights = target_layer_new_weight_tensor;

    let mut target_layer_new_bias_tensor = Tensor::from_vec(updated_target_layer_bias, target_layer.bias.unwrap().shape.dimensions());
    target_layer_new_bias_tensor.set_keep_alive(true);
    target_layer_new_bias_tensor.set_requires_grad(false);

    target_layer.bias.unwrap().set_keep_alive(false);
    target_layer.bias = Some(target_layer_new_bias_tensor);

   // update_timings(timings, String::from("Cantwork"),&start);
}

fn update_model(policy_net: &mut DQN, target_net: &mut DQN, timings: &mut HashMap<String, u128>) {
    update_layer(&policy_net.layer_1, &mut target_net.layer_1, timings);
    update_layer(&policy_net.layer_2, &mut target_net.layer_2, timings);
    update_layer(&policy_net.layer_3, &mut target_net.layer_3, timings);
}

fn convert_observation_to_tensor(observation: CartPoleObservation) -> Tensor {
    let observation_as_vec: Vec<f64> = observation.into();
    let observation_as_vec: Vec<f32> = observation_as_vec.iter().map(|x| *x as f32).collect();
    let mut state = Tensor::from_vec(observation_as_vec, vec![4]);
    state.set_keep_alive(true);
    state
}

fn create_new_model_from_existing(policy_net: &mut DQN) -> DQN {
    let layer_1_weight = policy_net.layer_1.weights.item();
    let layer_1_bias = policy_net.layer_1.bias.unwrap().item();

    let layer_2_weight = policy_net.layer_2.weights.item();
    let layer_2_bias = policy_net.layer_2.bias.unwrap().item();

    let layer_3_weight = policy_net.layer_3.weights.item();
    let layer_3_bias = policy_net.layer_3.bias.unwrap().item();

    let layer_1_weight = Tensor::from_vec(layer_1_weight.into_raw_vec(), policy_net.layer_1.weights.shape.dimensions());
    let layer_1_bias = Tensor::from_vec(layer_1_bias.into_raw_vec(), policy_net.layer_1.bias.unwrap().shape.dimensions());

    let layer_2_weight = Tensor::from_vec(layer_2_weight.into_raw_vec(), policy_net.layer_2.weights.shape.dimensions());
    let layer_2_bias = Tensor::from_vec(layer_2_bias.into_raw_vec(), policy_net.layer_2.bias.unwrap().shape.dimensions());

    let layer_3_weight = Tensor::from_vec(layer_3_weight.into_raw_vec(), policy_net.layer_3.weights.shape.dimensions());
    let layer_3_bias = Tensor::from_vec(layer_3_bias.into_raw_vec(), policy_net.layer_3.bias.unwrap().shape.dimensions());

    let layer_1 = Linear::from_tensors(layer_1_weight, Some(layer_1_bias));
    let layer_2 = Linear::from_tensors(layer_2_weight, Some(layer_2_bias));
    let layer_3 = Linear::from_tensors(layer_3_weight, Some(layer_3_bias));

    DQN::from_layers(layer_1, layer_2, layer_3)
}

// ============================================================================
// Training Thread
// ============================================================================

fn run_training(tx: Sender<TrainingUpdate>, learning_rate: f32) {
    let mut env = CartPoleEnv::new(RenderMode::None);
    let number_of_actions = env.action_space.0;
    let number_of_observations = 4;

    let mut policy_net = DQN::new(number_of_observations, number_of_actions);
    let mut target_net = create_new_model_from_existing(&mut policy_net);

    let mut optimizers = AdamW::new(learning_rate);
    let mut memory = ReplayMemory::new(10000);

    let mut steps_done = 0;


    let mut total_timings = HashMap::new();

    for i_episode in 0..TOTAL_EPISODES {
        let (observation, _) = env.reset(None, false, None);
        let mut state = convert_observation_to_tensor(observation);
        let mut t = 0;

        loop {
            let start = Instant::now();
            let action = select_action(state, steps_done, &mut policy_net);
            steps_done+= 1;
            let action_reward = env.step(action.item()[[0, 0]] as usize);

            let start = Instant::now();
            let new_state = convert_observation_to_tensor(action_reward.observation);

            let mut reward = Tensor::from_vec(vec![action_reward.reward.into_inner() as f32], vec![1]);
            reward.set_keep_alive(true);

            let done = action_reward.done || action_reward.truncated || t == 500;
            let next_state = if action_reward.done {
                None
            } else {
                Some(new_state)
            };

            let transition = Transition::new(state, action, next_state, reward);
            memory.push_transition(transition);
            state = new_state;

            let start = Instant::now();
            optimize_model(&mut memory, &mut policy_net, &mut target_net, &mut optimizers, &mut total_timings);
            let start = Instant::now();
            update_model(&mut policy_net, &mut target_net, &mut total_timings);


            if done {
                let update = TrainingUpdate {
                    episode: i_episode + 1,
                    episode_duration: t,
                    steps_done,
                    epsilon: calculate_epsilon(steps_done),
                    memory_size: memory.length(),
                    training_complete: false,
                    timings: total_timings.clone()
                };
                let _ = tx.send(update);
                break;
            }
            t += 1;
        }
    }

    // Send final completion message
    let _ = tx.send(TrainingUpdate {
        episode: TOTAL_EPISODES,
        episode_duration: 0,
        steps_done,
        epsilon: calculate_epsilon(steps_done),
        memory_size: 0,
        training_complete: true,
        timings: total_timings
    });
}

// ============================================================================
// Dashboard UI
// ============================================================================

fn ui(frame: &mut Frame, state: &DashboardState) {
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Title
            Constraint::Min(10),    // Main content
            Constraint::Length(3),  // Footer
        ])
        .split(frame.area());

    // Title
    let title = Paragraph::new("🎮 DQN CartPole Training Dashboard")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)));
    frame.render_widget(title, main_layout[0]);

    // Main content area
    let content_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(70),  // Charts
            Constraint::Percentage(30),  // Stats
        ])
        .split(main_layout[1]);

    // Charts area
    let charts_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60),  // Episode duration chart
            Constraint::Percentage(40),  // Epsilon chart
        ])
        .split(content_layout[0]);

    // Episode Duration Chart
    render_duration_chart(frame, charts_layout[0], state);

    // Epsilon Chart
    render_epsilon_chart(frame, charts_layout[1], state);

    // Stats Panel
    render_stats_panel(frame, content_layout[1], state);

    // Footer
    let status = if state.training_complete {
        "✅ Training Complete! Press 'q' to exit"
    } else {
        "⏳ Training in progress... Press 'q' to quit"
    };
    let footer = Paragraph::new(status)
        .style(Style::default().fg(Color::Yellow))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    frame.render_widget(footer, main_layout[2]);
}

fn render_duration_chart(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let data: Vec<(f64, f64)> = state.episode_durations.clone();
    
    // Calculate moving average for smoother visualization
    let moving_avg: Vec<(f64, f64)> = state.episode_durations
        .iter()
        .enumerate()
        .map(|(i, (ep, _))| {
            let start = if i >= 20 { i - 20 } else { 0 };
            let window: Vec<f64> = state.episode_durations[start..=i]
                .iter()
                .map(|(_, d)| *d)
                .collect();
            let avg = window.iter().sum::<f64>() / window.len() as f64;
            (*ep, avg)
        })
        .collect();

    let datasets = vec![
        Dataset::default()
            .name("Episode Duration")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .graph_type(GraphType::Scatter)
            .data(&data),
        Dataset::default()
            .name("Moving Avg (20)")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Yellow))
            .graph_type(GraphType::Line)
            .data(&moving_avg),
    ];

    let max_duration = state.episode_durations
        .iter()
        .map(|(_, d)| *d)
        .fold(100.0, f64::max);

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(" 📊 Episode Duration Over Time ")
                .title_style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
        )
        .x_axis(
            Axis::default()
                .title("Episode")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, TOTAL_EPISODES as f64])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw(format!("{}", TOTAL_EPISODES / 2)),
                    Span::raw(format!("{}", TOTAL_EPISODES)),
                ])
        )
        .y_axis(
            Axis::default()
                .title("Steps")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, max_duration * 1.1])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw(format!("{:.0}", max_duration / 2.0)),
                    Span::raw(format!("{:.0}", max_duration)),
                ])
        );

    frame.render_widget(chart, area);
}

fn render_epsilon_chart(frame: &mut Frame, area: Rect, state: &DashboardState) {
    // Generate theoretical epsilon decay curve
    let theoretical_curve: Vec<(f64, f64)> = (0..=10000)
        .step_by(100)
        .map(|steps| {
            let eps = EPS_END + (EPS_START - EPS_END) * (-1.0 * steps as f32 / EPS_DECAY).exp();
            (steps as f64, eps as f64)
        })
        .collect();

    // Actual epsilon values based on steps
    let actual_epsilon: Vec<(f64, f64)> = state.epsilon_history
        .iter()
        .enumerate()
        .map(|(i, (_, eps))| {
            let approx_steps = if i < state.episode_durations.len() {
                state.episode_durations[..=i].iter().map(|(_, d)| *d).sum::<f64>()
            } else {
                0.0
            };
            (approx_steps, *eps)
        })
        .collect();

    let datasets = vec![
        Dataset::default()
            .name("Theoretical ε")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::DarkGray))
            .graph_type(GraphType::Line)
            .data(&theoretical_curve),
        Dataset::default()
            .name("Actual ε")
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Magenta))
            .graph_type(GraphType::Scatter)
            .data(&actual_epsilon),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(" 🎲 Epsilon (Exploration Rate) Decay ")
                .title_style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
        )
        .x_axis(
            Axis::default()
                .title("Total Steps")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 10000.0])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw("5000"),
                    Span::raw("10000"),
                ])
        )
        .y_axis(
            Axis::default()
                .title("ε")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 1.0])
                .labels(vec![
                    Span::raw("0.0"),
                    Span::raw("0.5"),
                    Span::raw("1.0"),
                ])
        );

    frame.render_widget(chart, area);
}

fn render_stats_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let stats_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),   // Current stats
            Constraint::Length(5),   // Epsilon gauge
            Constraint::Length(5),   // Progress gauge
            Constraint::Min(3),      // Hyperparameters
            Constraint::Min(3),      // Hyperparameters
        ])
        .split(area);

    // Current Statistics
    let stats_text = vec![
        Line::from(vec![
            Span::styled("Episode: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}/{}", state.current_episode, TOTAL_EPISODES),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            ),
        ]),
        Line::from(vec![
            Span::styled("Total Steps: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", state.total_steps),
                Style::default().fg(Color::White)
            ),
        ]),
        Line::from(vec![
            Span::styled("Memory Size: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}/10000", state.memory_size),
                Style::default().fg(Color::Yellow)
            ),
        ]),
        Line::from(vec![
            Span::styled("Best Duration: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{} steps", state.best_duration),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            ),
        ]),
        Line::from(vec![
            Span::styled("Avg (100 ep): ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1} steps", state.avg_duration_100),
                Style::default().fg(Color::Blue)
            ),
        ]),
    ];

    let stats = Paragraph::new(stats_text)
        .block(
            Block::default()
                .title(" 📈 Statistics ")
                .title_style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White))
        );
    frame.render_widget(stats, stats_layout[0]);

    // Epsilon Gauge
    let epsilon_percent = ((state.current_epsilon - EPS_END) / (EPS_START - EPS_END) * 100.0) as u16;
    let epsilon_gauge = Gauge::default()
        .block(
            Block::default()
                .title(format!(" 🎲 Random Action: {:.1}% ", state.current_epsilon * 100.0))
                .title_style(Style::default().fg(Color::Magenta))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
        )
        .gauge_style(Style::default().fg(Color::Magenta).bg(Color::DarkGray))
        .percent(epsilon_percent.min(100))
        .label(format!("ε = {:.3}", state.current_epsilon));
    frame.render_widget(epsilon_gauge, stats_layout[1]);

    // Progress Gauge
    let progress_percent = ((state.current_episode as f64 / TOTAL_EPISODES as f64) * 100.0) as u16;
    let progress_gauge = Gauge::default()
        .block(
            Block::default()
                .title(" 🏃 Training Progress ")
                .title_style(Style::default().fg(Color::Green))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
        )
        .gauge_style(Style::default().fg(Color::Green).bg(Color::DarkGray))
        .percent(progress_percent.min(100))
        .label(format!("{}%", progress_percent));
    frame.render_widget(progress_gauge, stats_layout[2]);

    // Hyperparameters
    let hyperparams = vec![
        Line::from(Span::styled("─── Hyperparameters ───", Style::default().fg(Color::DarkGray))),
        Line::from(format!("γ={:.2}  τ={:.3}  LR={:.0e}", GAMMA, TAU, LR)),
        Line::from(format!("Batch={}  ε∈[{:.2},{:.2}]", BATCH_SIZE, EPS_END, EPS_START)),
        Line::from(format!("ε_decay={:.0}", EPS_DECAY)),
    ];

    let mut total = 0;
    for (_, v) in &state.timing {
        total += v;
    }

    let mut timing_lines = vec![];
    for (k, v) in &state.timing {
        timing_lines.push(
            Line::from(format!("{:?} {:?}", k, *v as f32 / total as f32))
        );
    }



    let hyperparams_widget = Paragraph::new(hyperparams)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));

    let timings_block = Paragraph::new(timing_lines)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));

    frame.render_widget(hyperparams_widget, stats_layout[3]);
    frame.render_widget(timings_block, stats_layout[4]);
}

// ============================================================================
// Main Entry Point
// ============================================================================


use std::fs::File;
use std::io::{Read, Result};

pub fn load_state_samples_f32(
    path: &str,
    obs_dim: usize,
) -> Result<Vec<Vec<f32>>> {
    // Read entire file
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    // File must be a multiple of 4 bytes (f32)
    if bytes.len() % 4 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Binary state file size is not divisible by 4",
        ));
    }

    // Interpret bytes as f32 (little-endian)
    let float_count = bytes.len() / 4;
    if float_count % obs_dim != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Float count {} not divisible by obs_dim {}",
                float_count, obs_dim
            ),
        ));
    }

    let num_states = float_count / obs_dim;

    let mut states = Vec::with_capacity(num_states);

    for i in 0..num_states {
        let mut state = Vec::with_capacity(obs_dim);
        for j in 0..obs_dim {
            let idx = (i * obs_dim + j) * 4;
            let bytes_f32 = [
                bytes[idx],
                bytes[idx + 1],
                bytes[idx + 2],
                bytes[idx + 3],
            ];
            state.push(f32::from_le_bytes(bytes_f32));
        }
        states.push(state);
    }

    Ok(states)
}

/* 
fn test() {


    let mut gguf_file = GGUFFile::new("./policy_net.gguf");
    let linear_1 = Linear::from_gguf_file(String::from("layer1.weight"), Some(String::from("layer1.bias")), &mut gguf_file);
    let linear_2 = Linear::from_gguf_file(String::from("layer2.weight"), Some(String::from("layer2.bias")), &mut gguf_file);
    let linear_3 = Linear::from_gguf_file(String::from("layer3.weight"), Some(String::from("layer3.bias")), &mut gguf_file);

    let mut dqn_policy = DQN::from_layers(linear_1, linear_2, linear_3);

    let feed_data = load_state_samples_f32("./sample_ep247_step500_states_f32.bin", 4).unwrap();
    let states = Tensor::from_vec(feed_data.iter().flatten().map(|x|*x).collect(), vec![100, 4]);
    println!("{:?}", dqn_policy.forward(states).item());
    panic!()
}
*/
fn main() -> io::Result<()> {

    let args = Args::parse();
    let mut learning_rate = LR;
    if args.learning_rate.is_some() {
        learning_rate = args.learning_rate.unwrap();
    }
//    test();
    // Setup terminal
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    // Create channel for training updates
    let (tx, rx): (Sender<TrainingUpdate>, Receiver<TrainingUpdate>) = mpsc::channel();

    // Start training thread
    thread::spawn(move || {
        run_training(tx, learning_rate);
    });

    // Dashboard state
    let mut state = DashboardState::new();

    // Main event loop
    loop {
        // Process any pending training updates
        while let Ok(update) = rx.try_recv() {
            state.update(update);
        }

        // Draw UI
        terminal.draw(|frame| ui(frame, &state))?;

        // Handle input events
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}