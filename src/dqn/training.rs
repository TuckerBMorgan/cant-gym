use cant::{
    central::{clip_gradients, get_equation, Shape, Tensor},
    nn::{Layer, Linear, Model},
    optimizers::AdamW,
};
use gym_rs::envs::classical_control::cartpole::CartPoleObservation;
use par_iter::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

use crate::config::{
    BATCH_SIZE, EPS_DECAY, EPS_END, EPS_START, GAMMA, GRAD_CLIP_VALUE, OBSERVATION_SIZE, TAU,
};
use crate::dqn::{DQN, ReplayMemory, TransitionSOA};
use crate::utils::update_timings;

pub fn calculate_epsilon(steps_done: usize) -> f32 {
    EPS_END + (EPS_START - EPS_END) * (-1.0 * steps_done as f32 / EPS_DECAY).exp()
}

pub fn select_action(state: Tensor, steps_done: usize, policy_net: &mut DQN) -> Tensor {
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

pub fn optimize_model(
    memory: &mut ReplayMemory,
    policy_net: &mut DQN,
    target_net: &mut DQN,
    optimizer: &mut AdamW,
    timings: &mut HashMap<String, u128>,
) {
    if memory.length() < BATCH_SIZE {
        return;
    }

    let start = Instant::now();

    let batch = memory.sample(BATCH_SIZE);
    let batch = TransitionSOA::new(batch);

    let non_final_mask: Vec<bool> = batch.next_state.iter().map(|x| x.is_some()).collect();

    let non_final_next_states: Vec<Tensor> =
        batch.next_state.iter().filter_map(|ns| *ns).collect();
    let mut count = 1;
    let mut non_final_next_states_batched = non_final_next_states[0];
    for nfns in non_final_next_states.iter().skip(1) {
        non_final_next_states_batched = non_final_next_states_batched.cat(*nfns, 0);
        count += 1;
    }

    non_final_next_states_batched =
        non_final_next_states_batched.reshape(Shape::new(vec![count, OBSERVATION_SIZE]));
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
    update_timings(timings, String::from("Backwards"), &start);
    let start = Instant::now();
    clip_gradients(GRAD_CLIP_VALUE);
    optimizer.update();
    update_timings(timings, String::from("optimizer"), &start);
    get_equation().compact_tensor_store();
}

fn update_layer(policy_layer: &Linear, target_layer: &mut Linear, _timings: &mut HashMap<String, u128>) {
    let layer_weight_policy: Vec<f32> = policy_layer
        .weights
        .item()
        .iter()
        .map(|x| *x * TAU)
        .collect();
    let layer_bias_policy: Vec<f32> = policy_layer
        .bias
        .expect("policy layer has bias")
        .item()
        .iter()
        .map(|x| *x * TAU)
        .collect();

    let layer_weight_target: Vec<f32> = target_layer
        .weights
        .item()
        .iter()
        .map(|x| *x * (1.0 - TAU))
        .collect();
    let layer_bias_target: Vec<f32> = target_layer
        .bias
        .expect("target layer has bias")
        .item()
        .iter()
        .map(|x| *x * (1.0 - TAU))
        .collect();

    let updated_target_layer_weight: Vec<f32> = layer_weight_policy
        .par_iter()
        .zip(layer_weight_target)
        .map(|(policy, target)| policy + target)
        .collect();
    let updated_target_layer_bias: Vec<f32> = layer_bias_policy
        .par_iter()
        .zip(layer_bias_target)
        .map(|(policy, target)| policy + target)
        .collect();

    let mut target_layer_new_weight_tensor = Tensor::from_vec(
        updated_target_layer_weight,
        target_layer.weights.shape.dimensions(),
    );
    target_layer_new_weight_tensor.set_keep_alive(true);
    target_layer_new_weight_tensor.set_requires_grad(false);

    target_layer.weights.set_keep_alive(false);
    target_layer.weights = target_layer_new_weight_tensor;

    let mut target_layer_new_bias_tensor = Tensor::from_vec(
        updated_target_layer_bias,
        target_layer
            .bias
            .expect("target layer has bias")
            .shape
            .dimensions(),
    );
    target_layer_new_bias_tensor.set_keep_alive(true);
    target_layer_new_bias_tensor.set_requires_grad(false);

    target_layer
        .bias
        .expect("target layer has bias")
        .set_keep_alive(false);
    target_layer.bias = Some(target_layer_new_bias_tensor);
}

pub fn update_model(policy_net: &mut DQN, target_net: &mut DQN, timings: &mut HashMap<String, u128>) {
    update_layer(&policy_net.layer_1, &mut target_net.layer_1, timings);
    update_layer(&policy_net.layer_2, &mut target_net.layer_2, timings);
    update_layer(&policy_net.layer_3, &mut target_net.layer_3, timings);
}

pub fn convert_observation_to_tensor(observation: CartPoleObservation) -> Tensor {
    let observation_as_vec: Vec<f64> = observation.into();
    let observation_as_vec: Vec<f32> = observation_as_vec.iter().map(|x| *x as f32).collect();
    let mut state = Tensor::from_vec(observation_as_vec, vec![OBSERVATION_SIZE]);
    state.set_keep_alive(true);
    state
}
