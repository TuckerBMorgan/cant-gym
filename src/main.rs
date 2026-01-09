mod config;
mod dqn;
mod tui;
mod utils;

use std::collections::HashMap;
use std::io::{self, stdout};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;

use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use gym_rs::{core::Env, envs::classical_control::cartpole::CartPoleEnv, utils::renderer::RenderMode};
use ratatui::prelude::*;

use config::{LR, MEMORY_CAPACITY, TOTAL_EPISODES};
use dqn::{
    calculate_epsilon, convert_observation_to_tensor, create_new_model_from_existing,
    optimize_model, select_action, update_model, DQN, ReplayMemory, Transition,
};
use tui::{ui, DashboardState, TrainingUpdate};
use cant::optimizers::AdamW;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Learning rate for the optimizer
    #[arg(short = 'l', long = "lr")]
    learning_rate: Option<f32>,
}

fn run_training(tx: Sender<TrainingUpdate>, learning_rate: f32) {
    let mut env = CartPoleEnv::new(RenderMode::None);
    let number_of_actions = env.action_space.0;
    let number_of_observations = 4;

    let mut policy_net = DQN::new(number_of_observations, number_of_actions);
    let mut target_net = create_new_model_from_existing(&mut policy_net);

    let mut optimizers = AdamW::new(learning_rate);
    let mut memory = ReplayMemory::new(MEMORY_CAPACITY);

    let mut steps_done = 0;
    let mut total_timings = HashMap::new();

    for i_episode in 0..TOTAL_EPISODES {
        let (observation, _) = env.reset(None, false, None);
        let mut state = convert_observation_to_tensor(observation);
        let mut t = 0;

        loop {
            let action = select_action(state, steps_done, &mut policy_net);
            steps_done += 1;
            let action_reward = env.step(action.item()[[0, 0]] as usize);

            let new_state = convert_observation_to_tensor(action_reward.observation);

            let mut reward =
                cant::central::Tensor::from_vec(vec![action_reward.reward.into_inner() as f32], vec![1]);
            reward.set_keep_alive(true);

            let done = action_reward.done || action_reward.truncated || t == config::MAX_STEPS_PER_EPISODE;
            let next_state = if action_reward.done { None } else { Some(new_state) };

            let transition = Transition::new(state, action, next_state, reward);
            memory.push_transition(transition);
            state = new_state;

            optimize_model(
                &mut memory,
                &mut policy_net,
                &mut target_net,
                &mut optimizers,
                &mut total_timings,
            );
            update_model(&mut policy_net, &mut target_net, &mut total_timings);

            if done {
                let update = TrainingUpdate {
                    episode: i_episode + 1,
                    episode_duration: t,
                    steps_done,
                    epsilon: calculate_epsilon(steps_done),
                    memory_size: memory.length(),
                    training_complete: false,
                    timings: total_timings.clone(),
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
        timings: total_timings,
    });
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let learning_rate = args.learning_rate.unwrap_or(LR);

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
