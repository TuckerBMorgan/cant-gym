use cant::{
    central::{Shape, Tensor, clip_gradients, get_equation, update_parameters, zero_all_grads},
    nn::{Layer, Linear, Model},
};
use gym_rs::{
    core::Env, envs::classical_control::cartpole::CartPoleEnv, utils::renderer::RenderMode,
};
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::{Rng, thread_rng};
use std::collections::VecDeque;

// Holds a single transition, which is a single interaction
// between the agent and the enviroment
#[derive(Clone, Copy)]
pub struct Transition {
    state: [f32; 4],
    action: [f32; 1],
    reward: f32,
    next_state: Option<[f32; 4]>,
    terminated: bool,
}

impl Transition {
    pub fn new(state: [f32;4], action:[f32;1], reward: f32, next_state: Option<[f32;4]>, terminated: bool) -> Transition {
        Transition { state, action, reward, next_state, terminated }
    }
}
const GAMMA: f32 = 0.99;

// All of the past 50000 transitions that we have seen
// that will happen over multiple runs
pub struct ReplayBuffer {
    transitions: VecDeque<Transition>,
}

impl ReplayBuffer {
    pub fn new() -> ReplayBuffer {
        ReplayBuffer {
            transitions: VecDeque::new(),
        }
    }

    pub fn add_transition(&mut self, new_transtion: Transition) {
        if self.transitions.len() == 50_000 {
            let _ = self.transitions.pop_front();
        }
        self.transitions.push_back(new_transtion);
    }

    pub fn sample(&mut self, batch_size: usize) -> Vec<Transition> {
        let mut rng = thread_rng();

        // Otherwise, randomly sample batch_size transitions
        self.transitions
            .make_contiguous()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }
}

// Both the policy net and the traget net have the same
// layout.
pub struct AgentBrain {
    layer_1: Linear,
    layer_2: Linear,
    layer_3: Linear,
}

impl AgentBrain {
    pub fn new(observation_space: usize, action_space: usize) -> AgentBrain {
        // The hidden layers sizes just come from ones I have found online
        let layer_1 = Linear::new(observation_space, 128, true);
        let layer_2 = Linear::new(128, 128, true);
        let layer_3 = Linear::new(128, action_space, true);

        AgentBrain {
            layer_1,
            layer_2,
            layer_3,
        }
    }
}

impl Model for AgentBrain {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let x = self.layer_1.forward(input).relu();
        let x = self.layer_2.forward(x).relu();
        let x = self.layer_3.forward(x);
        x
    }
    fn get_parameters(&self) -> Vec<cant::central::TensorID> {
        return vec![self.layer_1.weights.id, self.layer_1.bias.unwrap().id, self.layer_2.weights.id, self.layer_2.bias.unwrap().id, self.layer_3.weights.id, self.layer_3.bias.unwrap().id];
    }
}

pub struct Agent {
    policy_net: AgentBrain,
    target_net: AgentBrain,
}

impl Agent {
    pub fn new(observation_size: usize, action_size: usize) -> Agent {
        let policy_net = AgentBrain::new(observation_size, action_size);
        let mut target_net = AgentBrain::new(observation_size, action_size);
        
        let policy_weight_id = policy_net.layer_1.weights.id;
        let policy_weights = get_equation().get_data_flat_buffer(policy_weight_id).to_vec();

        let policy_bias_id = policy_net.layer_1.bias.unwrap().id;
        let policy_bias = get_equation().get_data_flat_buffer(policy_bias_id).to_vec();

        target_net.layer_1.weights = Tensor::from_vec(policy_weights.to_vec(), policy_net.layer_1.weights.shape.dimensions());
        target_net.layer_1.bias = Some(Tensor::from_vec(policy_bias.to_vec(), policy_net.layer_1.bias.unwrap().shape.dimensions()));


        let policy_weight_id = policy_net.layer_2.weights.id;
        let policy_weights = get_equation().get_data_flat_buffer(policy_weight_id).to_vec();

        let policy_bias_id = policy_net.layer_2.bias.unwrap().id;
        let policy_bias = get_equation().get_data_flat_buffer(policy_bias_id).to_vec();

        target_net.layer_2.weights = Tensor::from_vec(policy_weights.to_vec(), policy_net.layer_2.weights.shape.dimensions());
        target_net.layer_2.bias = Some(Tensor::from_vec(policy_bias.to_vec(), policy_net.layer_2.bias.unwrap().shape.dimensions()));

        let policy_weight_id = policy_net.layer_3.weights.id;
        let policy_weights = get_equation().get_data_flat_buffer(policy_weight_id).to_vec();

        let policy_bias_id = policy_net.layer_3.bias.unwrap().id;
        let policy_bias = get_equation().get_data_flat_buffer(policy_bias_id).to_vec();

        target_net.layer_3.weights = Tensor::from_vec(policy_weights.to_vec(), policy_net.layer_3.weights.shape.dimensions());
        target_net.layer_3.bias = Some(Tensor::from_vec(policy_bias.to_vec(), policy_net.layer_3.bias.unwrap().shape.dimensions()));

        Agent {
            policy_net,
            target_net
        }
    }

    pub fn select_action(&mut self, input: Tensor, random_chance: f32) -> usize {
        let random_action = rand::thread_rng().gen_range(0.0..1.0);
        if random_action < random_chance {
            return rand::thread_rng().gen_range(0..2);
        }

        let outputs =  self
            .policy_net
            .forward(input);
        let as_item = outputs.item().into_raw_vec();

        let (max_index, _max_value) = as_item
            .iter()
            .enumerate()
            .fold((0, f32::MIN), |(best_idx, best_val), (i, &v)| {
                if v > best_val {
                    (i, v)
                } else {
                    (best_idx, best_val)
                }
            });

        max_index
    }
}

fn optimize_model(
    policy_net: &mut AgentBrain,
    target_net: &mut AgentBrain,
    replay_buffer: &mut ReplayBuffer,
) {
    if replay_buffer.transitions.len() < 32 {
        return;
    }
    let transitions = replay_buffer.sample(32);

    let non_final_mask: Vec<bool> = transitions
        .iter()
        .map(|x| return x.next_state.is_some())
        .collect();

    let non_final_next_states: Vec<[f32; 4]> = transitions
        .iter()
        .filter_map(|x| x.next_state) // might need .cloned() depending on type
        .collect();

    let mut states = vec![];
    let mut rewards = vec![];
    let mut actions = vec![];
    // let mut terminal = vec![];

    for t in &transitions {
        states.extend_from_slice(&t.state);
        rewards.extend_from_slice(&[t.reward]);
        actions.extend_from_slice(&t.action);
    }

    let states = Tensor::from_vec(states, vec![32, 4]);
    let rewards = Tensor::from_vec(rewards, vec![32, 1]);
    let actions = Tensor::from_vec(actions, vec![32, 1]);

    let non_final_next_states = non_final_next_states.as_flattened().to_vec();
    let len_vec = non_final_next_states.len() / 4;
    let non_final_next_states = Tensor::from_vec(non_final_next_states, vec![len_vec, 4]);


    let state_action_value = policy_net.forward(states).gather(1, actions);

    let mut next_state_values = vec![32;0.0];
    let target_next_best = target_net
        .forward(non_final_next_states)
        .max(1, false)
        .detach()
        .item(); // shape: [N_non_final, 1] or Vec<f32>

    // j is index into the compact non-terminal array
    let mut j = 0;

    for i in 0..non_final_mask.len() {
        if non_final_mask[i] {
            next_state_values[i] = target_next_best[j];
            j += 1;
        }
    }
    let expected_state_action_values = (Tensor::from_vec(vec![next_state_values], vec![32]) * GAMMA) + rewards;

    let loss = (state_action_value - expected_state_action_values).pow(2.0).mean(vec![0, 1]);
    zero_all_grads();
    loss.backward();
    clip_gradients(100.0);
    update_parameters(-1e-3);
    get_equation().compact_tensor_store();
}

fn update_target_model(agent: &mut Agent, tau: f32) {
    let layer_1_weight = agent.policy_net.layer_1.weights.item() * tau + agent.target_net.layer_1.weights.item() * (1.0 - tau);
    let layer_1_bias = agent.policy_net.layer_1.bias.unwrap().item() * tau + agent.target_net.layer_1.bias.unwrap().item() * (1.0 - tau);

    let layer_2_weight = agent.policy_net.layer_2.weights.item() * tau + agent.target_net.layer_2.weights.item() * (1.0 - tau);
    let layer_2_bias = agent.policy_net.layer_2.bias.unwrap().item() * tau + agent.target_net.layer_2.bias.unwrap().item() * (1.0 - tau);

    let layer_3_weight = agent.policy_net.layer_3.weights.item() * tau + agent.target_net.layer_3.weights.item() * (1.0 - tau);
    let layer_3_bias = agent.policy_net.layer_3.bias.unwrap().item() * tau + agent.target_net.layer_3.bias.unwrap().item() * (1.0 - tau);

    agent.target_net.layer_1.weights = Tensor::from_vec(layer_1_weight.into_raw_vec(), vec![4, 128]);
    agent.target_net.layer_1.weights.set_keep_alive(true);
    agent.target_net.layer_1.weights.set_requires_grad(true);

    agent.target_net.layer_1.bias = Some(Tensor::from_vec(layer_1_bias.into_raw_vec(), vec![128]));
    agent.target_net.layer_1.bias.as_mut().unwrap().set_keep_alive(true);
    agent.target_net.layer_1.bias.as_mut().unwrap().set_requires_grad(true);

    agent.target_net.layer_2.weights = Tensor::from_vec(layer_2_weight.into_raw_vec(), vec![128, 128]);
    agent.target_net.layer_2.weights.set_keep_alive(true);
    agent.target_net.layer_2.weights.set_requires_grad(true);

    agent.target_net.layer_2.bias = Some(Tensor::from_vec(layer_2_bias.into_raw_vec(), vec![128]));
    agent.target_net.layer_2.bias.as_mut().unwrap().set_keep_alive(true);
    agent.target_net.layer_2.bias.as_mut().unwrap().set_requires_grad(true);

    agent.target_net.layer_3.weights = Tensor::from_vec(layer_3_weight.into_raw_vec(), vec![128, 2]);
    agent.target_net.layer_3.weights.set_keep_alive(true);
    agent.target_net.layer_3.weights.set_requires_grad(true);

    agent.target_net.layer_3.bias = Some(Tensor::from_vec(layer_3_bias.into_raw_vec(), vec![2]));
    agent.target_net.layer_3.bias.as_mut().unwrap().set_keep_alive(true);
    agent.target_net.layer_3.bias.as_mut().unwrap().set_requires_grad(true);
}

fn main() {


    let mut env = CartPoleEnv::new(RenderMode::None);



    let mut agent = Agent::new(4, 2);
    const N: usize = 500;
    let mut rewards = Vec::with_capacity(N);
    let mut replay_buffer = ReplayBuffer::new();
    let mut epsilon_start = 0.9;

    for n in 0..N {
        let epsilon = (epsilon_start * 0.995_f32.powi(n as i32)).max(0.05);
        println!("starting run {:?}", n);
        let mut current_reward = OrderedFloat(0.);

        let(observation, _) = env.reset(None, false, None);
        let state : Vec<f64> = observation.into();
        let mut state = [state[0] as f32, state[1] as f32, state[2] as f32, state[3] as f32];

        for i in 0..475 {
            let state_tensor = Tensor::from_vec(state.to_vec(), vec![4]);
            let action = agent.select_action(state_tensor, epsilon);
            let state_reward = env.step(action);
            current_reward += state_reward.reward;

            let mut next_state = None;
            if state_reward.done == false {
                let next_state_flat : Vec<f64> = state_reward.observation.into();

                next_state = Some([next_state_flat[0] as f32, next_state_flat[1] as f32, next_state_flat[2] as f32, next_state_flat[3] as f32]);
            }

            let transition = Transition::new(state, [action as f32], state_reward.reward.0 as f32, next_state, state_reward.done);
            replay_buffer.add_transition(transition);
            optimize_model(&mut agent.policy_net, &mut agent.target_net, &mut replay_buffer);
            update_target_model(&mut agent, 0.005);

            if state_reward.done {
                println!("Got to {} length", i);
                break;
            }
            state = next_state.unwrap();
        }

        env.reset(None, false, None);
        rewards.push(current_reward);
    }

    println!("{:?}", rewards);
}
