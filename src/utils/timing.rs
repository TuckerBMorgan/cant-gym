use std::collections::HashMap;
use std::time::Instant;

pub fn update_timings(timings: &mut HashMap<String, u128>, label: String, start: &Instant) {
    if !timings.contains_key(&label) {
        timings.insert(label, start.elapsed().as_micros());
    } else {
        let update = timings[&label] + start.elapsed().as_millis();
        timings.insert(label, update);
    }
}
