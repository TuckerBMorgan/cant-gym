use cant::central::Tensor;
use rand::seq::SliceRandom;
use std::collections::VecDeque;

#[derive(Clone, Copy)]
pub struct Transition {
    pub state: Tensor,
    pub action: Tensor,
    pub next_state: Option<Tensor>,
    pub reward: Tensor,
}

impl Transition {
    pub fn new(
        state: Tensor,
        action: Tensor,
        next_state: Option<Tensor>,
        reward: Tensor,
    ) -> Transition {
        Transition {
            state,
            action,
            next_state,
            reward,
        }
    }
}

pub struct ReplayMemory {
    memory_buffer: VecDeque<Transition>,
    max_capacity: usize,
}

impl ReplayMemory {
    pub fn new(max_capacity: usize) -> ReplayMemory {
        ReplayMemory {
            memory_buffer: VecDeque::with_capacity(max_capacity),
            max_capacity,
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

pub struct TransitionSOA {
    pub state: Vec<Tensor>,
    pub action: Vec<Tensor>,
    pub next_state: Vec<Option<Tensor>>,
    pub reward: Vec<Tensor>,
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
            reward,
        }
    }
}
