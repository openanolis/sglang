//! Weighted Random Load Balancing Policy WASM Plugin
//!
//! This is an example WASM plugin that implements a weighted random selection
//! policy for load balancing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Input data for the policy
#[derive(Debug, Serialize, Deserialize)]
struct PolicyInput {
    workers: Vec<WorkerInfo>,
    request_text: Option<String>,
    config: HashMap<String, serde_json::Value>,
    state: HashMap<String, serde_json::Value>,
}

/// Output data from the policy
#[derive(Debug, Serialize, Deserialize)]
struct PolicyOutput {
    selected_worker: Option<usize>,
    state: Option<HashMap<String, serde_json::Value>>,
    error: Option<String>,
}

/// Worker information
#[derive(Debug, Serialize, Deserialize)]
struct WorkerInfo {
    url: String,
    healthy: bool,
    load: isize,
}

/// Weighted random policy implementation
struct WeightedRandomPolicy {
    weights: HashMap<String, f64>,
}

impl WeightedRandomPolicy {
    fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    fn select_worker(&self, workers: &[WorkerInfo]) -> Option<usize> {
        let healthy_workers: Vec<_> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.healthy)
            .collect();

        if healthy_workers.is_empty() {
            return None;
        }

        // Calculate total weight
        let total_weight: f64 = healthy_workers
            .iter()
            .map(|(_, w)| self.weights.get(&w.url).unwrap_or(&1.0))
            .sum();

        if total_weight <= 0.0 {
            // Fallback to uniform random if no valid weights
            return Some(healthy_workers[0].0);
        }

        // Generate random value
        let random_value = self.generate_random() * total_weight;

        // Select worker based on weights
        let mut cumulative_weight = 0.0;
        for (idx, worker) in healthy_workers {
            let weight = self.weights.get(&worker.url).unwrap_or(&1.0);
            cumulative_weight += weight;
            if random_value <= cumulative_weight {
                return Some(*idx);
            }
        }

        // Fallback to last worker
        Some(healthy_workers.last().unwrap().0)
    }

    fn generate_random(&self) -> f64 {
        // Simple pseudo-random number generation
        // In a real implementation, you'd use a proper RNG
        use std::time::{SystemTime, UNIX_EPOCH};
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        (time % 1000) as f64 / 1000.0
    }

    fn update_weights_from_config(&mut self, config: &HashMap<String, serde_json::Value>) {
        self.weights.clear();
        
        if let Some(weights) = config.get("weights") {
            if let Some(weights_obj) = weights.as_object() {
                for (url, weight) in weights_obj {
                    if let Some(weight_val) = weight.as_f64() {
                        self.weights.insert(url.clone(), weight_val);
                    }
                }
            }
        }
    }
}

/// Main policy instance
static mut POLICY: Option<WeightedRandomPolicy> = None;

/// Initialize the policy
#[no_mangle]
pub extern "C" fn init() {
    unsafe {
        POLICY = Some(WeightedRandomPolicy::new());
    }
}

/// Select a worker using the weighted random policy
#[no_mangle]
pub extern "C" fn select_worker(input_ptr: *const u8, input_len: usize) -> *mut u8 {
    let input_data = unsafe {
        std::slice::from_raw_parts(input_ptr, input_len)
    };

    let input: PolicyInput = match serde_json::from_slice(input_data) {
        Ok(input) => input,
        Err(e) => {
            let error_output = PolicyOutput {
                selected_worker: None,
                state: None,
                error: Some(format!("Failed to parse input: {}", e)),
            };
            return serialize_output(&error_output);
        }
    };

    let policy = unsafe {
        POLICY.as_mut().expect("Policy not initialized")
    };

    // Update weights from config
    policy.update_weights_from_config(&input.config);

    // Select worker
    let selected_worker = policy.select_worker(&input.workers);

    let output = PolicyOutput {
        selected_worker,
        state: None,
        error: None,
    };

    serialize_output(&output)
}

/// Serialize output to JSON and return pointer
fn serialize_output(output: &PolicyOutput) -> *mut u8 {
    let json = serde_json::to_vec(output).expect("Failed to serialize output");
    let boxed = json.into_boxed_slice();
    Box::into_raw(boxed) as *mut u8
}

/// Allocate memory for WASM
#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    let mut buf = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Deallocate memory for WASM
#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, size: usize) {
    unsafe {
        let _buf = Vec::from_raw_parts(ptr, size, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_random_policy() {
        let mut policy = WeightedRandomPolicy::new();
        
        let workers = vec![
            WorkerInfo {
                url: "http://w1:8000".to_string(),
                healthy: true,
                load: 0,
            },
            WorkerInfo {
                url: "http://w2:8000".to_string(),
                healthy: true,
                load: 0,
            },
        ];

        // Test with default weights
        let selected = policy.select_worker(&workers);
        assert!(selected.is_some());
        assert!(selected.unwrap() < workers.len());

        // Test with custom weights
        let mut config = HashMap::new();
        let mut weights = HashMap::new();
        weights.insert("http://w1:8000".to_string(), serde_json::Value::Number(serde_json::Number::from(2)));
        weights.insert("http://w2:8000".to_string(), serde_json::Value::Number(serde_json::Number::from(1)));
        config.insert("weights".to_string(), serde_json::Value::Object(weights));

        policy.update_weights_from_config(&config);
        let selected = policy.select_worker(&workers);
        assert!(selected.is_some());
    }

    #[test]
    fn test_no_healthy_workers() {
        let policy = WeightedRandomPolicy::new();
        
        let workers = vec![
            WorkerInfo {
                url: "http://w1:8000".to_string(),
                healthy: false,
                load: 0,
            },
        ];

        let selected = policy.select_worker(&workers);
        assert!(selected.is_none());
    }
}
