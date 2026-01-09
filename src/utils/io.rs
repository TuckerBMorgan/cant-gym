use std::fs::File;
use std::io::{Read, Result};

pub fn load_state_samples_f32(path: &str, obs_dim: usize) -> Result<Vec<Vec<f32>>> {
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
