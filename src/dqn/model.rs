use cant::{
    central::{Tensor, TensorID},
    nn::{Layer, Linear, Model},
};

use crate::config::HIDDEN_LAYER_SIZE;

pub struct DQN {
    pub layer_1: Linear,
    pub layer_2: Linear,
    pub layer_3: Linear,
}

impl DQN {
    pub fn new(number_of_observations: usize, number_of_actions: usize) -> DQN {
        DQN {
            layer_1: Linear::new(number_of_observations, HIDDEN_LAYER_SIZE, true),
            layer_2: Linear::new(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, true),
            layer_3: Linear::new(HIDDEN_LAYER_SIZE, number_of_actions, true),
        }
    }

    pub fn from_layers(layer_1: Linear, layer_2: Linear, layer_3: Linear) -> DQN {
        DQN {
            layer_1,
            layer_2,
            layer_3,
        }
    }
}

impl Model for DQN {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let x = self.layer_1.forward(input).relu();
        let x = self.layer_2.forward(x).relu();
        self.layer_3.forward(x)
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![
            self.layer_1.weights.id,
            self.layer_1.bias.expect("layer_1 initialized with bias").id,
            self.layer_2.weights.id,
            self.layer_2.bias.expect("layer_2 initialized with bias").id,
            self.layer_3.weights.id,
            self.layer_3.bias.expect("layer_3 initialized with bias").id,
        ]
    }
}

pub fn create_new_model_from_existing(policy_net: &mut DQN) -> DQN {
    let layer_1_weight = policy_net.layer_1.weights.item();
    let layer_1_bias = policy_net.layer_1.bias.expect("layer_1 has bias").item();

    let layer_2_weight = policy_net.layer_2.weights.item();
    let layer_2_bias = policy_net.layer_2.bias.expect("layer_2 has bias").item();

    let layer_3_weight = policy_net.layer_3.weights.item();
    let layer_3_bias = policy_net.layer_3.bias.expect("layer_3 has bias").item();

    let layer_1_weight = Tensor::from_vec(
        layer_1_weight.into_raw_vec(),
        policy_net.layer_1.weights.shape.dimensions(),
    );
    let layer_1_bias = Tensor::from_vec(
        layer_1_bias.into_raw_vec(),
        policy_net.layer_1.bias.expect("layer_1 has bias").shape.dimensions(),
    );

    let layer_2_weight = Tensor::from_vec(
        layer_2_weight.into_raw_vec(),
        policy_net.layer_2.weights.shape.dimensions(),
    );
    let layer_2_bias = Tensor::from_vec(
        layer_2_bias.into_raw_vec(),
        policy_net.layer_2.bias.expect("layer_2 has bias").shape.dimensions(),
    );

    let layer_3_weight = Tensor::from_vec(
        layer_3_weight.into_raw_vec(),
        policy_net.layer_3.weights.shape.dimensions(),
    );
    let layer_3_bias = Tensor::from_vec(
        layer_3_bias.into_raw_vec(),
        policy_net.layer_3.bias.expect("layer_3 has bias").shape.dimensions(),
    );

    let layer_1 = Linear::from_tensors(layer_1_weight, Some(layer_1_bias));
    let layer_2 = Linear::from_tensors(layer_2_weight, Some(layer_2_bias));
    let layer_3 = Linear::from_tensors(layer_3_weight, Some(layer_3_bias));

    DQN::from_layers(layer_1, layer_2, layer_3)
}
