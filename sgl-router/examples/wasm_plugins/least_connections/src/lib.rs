//! Least Connections Load Balancing Policy WASM Plugin
//!
//! This is an example WASM plugin that implements a least connections
//! selection policy for load balancing.

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

/// Least connections policy implementation
struct LeastConnectionsPolicy {
    connection_counts: HashMap<String, usize>,
}

impl LeastConnectionsPolicy {
    fn new() -> Self {
        Self {
            connection_counts: HashMap::new(),
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

        // Find worker with least connections
        let mut min_connections = usize::MAX;
        let mut selected_idx = None;

        for (idx, worker) in healthy_workers {
            let connections = self.connection_counts.get(&worker.url).unwrap_or(&0);
            if *connections < min_connections {
                min_connections = *connections;
                selected_idx = Some(*idx);
            }
        }

        selected_idx
    }

    fn update_connection_counts(&mut self, workers: &[WorkerInfo]) {
        for worker in workers {
            // Use the load from worker info as connection count
            let connections = worker.load.max(0) as usize;
            self.connection_counts.insert(worker.url.clone(), connections);
        }
    }
}

/// Main policy instance
static mut POLICY: Option<LeastConnectionsPolicy> = None;

/// Initialize the policy
#[no_mangle]
pub extern "C" fn init() {
    unsafe {
        POLICY = Some(LeastConnectionsPolicy::new());
    }
}

/// Select a worker using the least connections policy
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

    // Update connection counts from worker info
    policy.update_connection_counts(&input.workers);

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
    fn test_least_connections_policy() {
        let mut policy = LeastConnectionsPolicy::new();
        
        let workers = vec![
            WorkerInfo {
                url: "http://w1:8000".to_string(),
                healthy: true,
                load: 5,
            },
            WorkerInfo {
                url: "http://w2:8000".to_string(),
                healthy: true,
                load: 2,
            },
            WorkerInfo {
                url: "http://w3:8000".to_string(),
                healthy: true,
                load: 8,
            },
        ];

        policy.update_connection_counts(&workers);
        let selected = policy.select_worker(&workers);
        
        assert!(selected.is_some());
        // Should select worker with least connections (w2 with load 2)
        assert_eq!(selected.unwrap(), 1);
    }

    #[test]
    fn test_no_healthy_workers() {
        let policy = LeastConnectionsPolicy::new();
        
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

    #[test]
    fn test_equal_connections() {
        let mut policy = LeastConnectionsPolicy::new();
        
        let workers = vec![
            WorkerInfo {
                url: "http://w1:8000".to_string(),
                healthy: true,
                load: 3,
            },
            WorkerInfo {
                url: "http://w2:8000".to_string(),
                healthy: true,
                load: 3,
            },
        ];

        policy.update_connection_counts(&workers);
        let selected = policy.select_worker(&workers);
        
        assert!(selected.is_some());
        // Should select the first worker when connections are equal
        assert_eq!(selected.unwrap(), 0);
    }
}
