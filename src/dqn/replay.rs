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

        // Use index-based sampling to avoid cloning the entire buffer
        let mut rng = rand::thread_rng();
        let indices = rand::seq::index::sample(&mut rng, self.memory_buffer.len(), batch_size);
        indices.into_iter().map(|i| self.memory_buffer[i]).collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use cant::central::Shape;

    fn create_test_tensor(value: f32) -> Tensor {
        Tensor::element(Shape::new(vec![1]), value)
    }

    fn create_test_transition(value: f32) -> Transition {
        Transition::new(
            create_test_tensor(value),
            create_test_tensor(value),
            Some(create_test_tensor(value + 1.0)),
            create_test_tensor(1.0),
        )
    }

    #[test]
    fn test_replay_memory_new() {
        let memory = ReplayMemory::new(100);
        assert_eq!(memory.length(), 0);
        assert_eq!(memory.max_capacity, 100);
    }

    #[test]
    fn test_replay_memory_push_and_length() {
        let mut memory = ReplayMemory::new(100);

        for i in 0..10 {
            memory.push_transition(create_test_transition(i as f32));
        }

        assert_eq!(memory.length(), 10);
    }

    #[test]
    fn test_replay_memory_capacity_overflow() {
        let mut memory = ReplayMemory::new(5);

        // Push 10 transitions into a buffer of capacity 5
        for i in 0..10 {
            memory.push_transition(create_test_transition(i as f32));
        }

        // Should still be at capacity
        assert_eq!(memory.length(), 5);
    }

    #[test]
    fn test_replay_memory_sample_size() {
        let mut memory = ReplayMemory::new(100);

        for i in 0..50 {
            memory.push_transition(create_test_transition(i as f32));
        }

        let sample = memory.sample(10);
        assert_eq!(sample.len(), 10);
    }

    #[test]
    #[should_panic(expected = "Cannot sample more elements than are in the replay buffer")]
    fn test_replay_memory_sample_exceeds_size() {
        let mut memory = ReplayMemory::new(100);

        for i in 0..5 {
            memory.push_transition(create_test_transition(i as f32));
        }

        // This should panic
        let _ = memory.sample(10);
    }

    #[test]
    fn test_transition_soa_conversion() {
        let transitions = vec![
            create_test_transition(1.0),
            create_test_transition(2.0),
            create_test_transition(3.0),
        ];

        let soa = TransitionSOA::new(transitions);

        assert_eq!(soa.state.len(), 3);
        assert_eq!(soa.action.len(), 3);
        assert_eq!(soa.next_state.len(), 3);
        assert_eq!(soa.reward.len(), 3);
    }

    #[test]
    fn test_transition_with_terminal_state() {
        let terminal_transition = Transition::new(
            create_test_tensor(1.0),
            create_test_tensor(0.0),
            None, // Terminal state
            create_test_tensor(10.0),
        );

        assert!(terminal_transition.next_state.is_none());

        let soa = TransitionSOA::new(vec![terminal_transition]);
        assert!(soa.next_state[0].is_none());
    }
}
