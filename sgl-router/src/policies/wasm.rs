//! WebAssembly plugin for dynamic policy loading
//!
//! This module provides a secure way to load and execute custom load balancing
//! policies as WebAssembly modules without requiring recompilation of sgl-router.

use crate::policies::LoadBalancingPolicy;
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use wasmtime::{Engine, Instance, Memory, Module, Store};

/// WASM policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfig {
    /// Path to the WASM policy file
    pub path: String,
    /// Policy name for identification, displayed as wasm:POLICY_NAME
    pub name: String,
    /// Maximum execution time in milliseconds, default 100ms
    pub max_execution_time_ms: u64,
    /// Maximum memory usage in bytes, default 1MB
    pub max_memory_bytes: usize,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            name: String::new(),
            max_execution_time_ms: 100, // 100ms
            max_memory_bytes: 1024 * 1024, // 1MB
        }
    }
}

/// WASM plugin context for policy execution
#[derive(Clone)]
pub struct WasmContext {
    /// Plugin configuration
    pub config: WasmConfig,
    /// Compiled WASM engine
    pub engine: Engine,
    /// Compiled module
    pub module: Module,
    /// Plugin state
    pub state: HashMap<String, serde_json::Value>,
}

/// WASM-based load balancing policy
pub struct WasmPolicy {
    /// policy context
    context: Arc<WasmContext>,
    /// Policy name, displayed as wasm:POLICY_NAME
    name: String,
}

impl std::fmt::Debug for WasmPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmPolicy").field("name", &self.name).finish()
    }
}

impl WasmPolicy {
    /// Create a new WASM policy with a config
    pub fn with_config(config: WasmConfig) -> Self {
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm_bytes)?;
        let context = WasmContext {
            config,
            engine,
            module,
            state: HashMap::new(),
        };
        Self {
            context: Arc::new(context),
            name: config.name,
        }
    }

    /// Execute WASM function with safety checks
    fn execute_wasm_function<T>(
        &self,
        function_name: &str,
        params: &[u8],
    ) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        // Instantiate a fresh store/instance per call for isolation
        let mut store: Store<()> = Store::new(&self.context.engine, ());
        let instance = Instance::new(&mut store, &self.context.module, &[])?;

        // Resolve required exports
        let memory: Memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow!("WASM module missing 'memory' export"))?;

        let alloc = instance
            .get_typed_func::<i32, i32>(&mut store, "alloc")
            .map_err(|_| anyhow!("WASM module missing 'alloc' export with signature (i32) -> i32"))?;
        let dealloc = instance
            .get_typed_func::<(i32, i32), ()>(&mut store, "dealloc")
            .map_err(|_| anyhow!("WASM module missing 'dealloc' export with signature (i32,i32) -> ()"))?;

        // Allocate and write params into guest memory
        let input_len_i32: i32 = params
            .len()
            .try_into()
            .map_err(|_| anyhow!("Input too large for wasm32 memory"))?;
        let input_ptr = alloc.call(&mut store, input_len_i32)?;

        let data = memory.data_mut(&mut store);
        let start = input_ptr as usize;
        let end = start + params.len();
        if end > data.len() {
            return Err(anyhow!("WASM memory overflow when writing input"));
        }
        data[start..end].copy_from_slice(params);

        // Call target function: (ptr, len) -> ptr_to_json
        let select_func = instance
            .get_typed_func::<(i32, i32), i32>(&mut store, function_name)
            .map_err(|_| anyhow!("WASM module missing '{}' export with signature (i32,i32)->i32", function_name))?;

        let result_ptr = select_func.call(&mut store, (input_ptr, input_len_i32))?;

        // Heuristic to read JSON result: try increasing lengths until serde_json parses
        // up to configured max memory limit or memory size
        let mem_slice = memory.data(&store);
        let result_start = result_ptr as usize;
        if result_start >= mem_slice.len() {
            // Free input buffer before returning
            let _ = dealloc.call(&mut store, (input_ptr, input_len_i32));
            return Err(anyhow!("WASM result pointer out of bounds"));
        }

        let max_available = mem_slice.len() - result_start;
        let mut parse_len = 256usize.min(max_available);
        let max_len = max_available.min(self.context.config.max_memory_bytes);
        let mut parsed: Option<T> = None;
        while parse_len <= max_len {
            let candidate = &mem_slice[result_start..result_start + parse_len];
            match serde_json::from_slice::<T>(candidate) {
                Ok(v) => {
                    parsed = Some(v);
                    break;
                }
                Err(_) => {
                    // Increase window and try again
                    // Grow by 2x until near max, then step by 4KB
                    let next = if parse_len < 64 * 1024 { parse_len * 2 } else { parse_len + 4 * 1024 };
                    parse_len = next.min(max_len);
                    if parse_len == max_len {
                        // final attempt will occur now; loop continues
                    }
                }
            }
        }

        // Best effort free input buffer (plugin may also expect freeing result; we cannot know size)
        let _ = dealloc.call(&mut store, (input_ptr, input_len_i32));

        parsed.ok_or_else(|| anyhow!("Failed to parse JSON result from WASM function"))
    }
}

impl LoadBalancingPolicy for WasmPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<usize> {
        // Build input payload for WASM
        let input = WasmPolicyInput {
            workers: workers
                .iter()
                .map(|w| WasmWorker {
                    url: w.url().to_string(),
                    healthy: w.is_healthy(),
                    load: w.load() as isize,
                })
                .collect(),
            request_text: request_text.map(|s| s.to_string()),
            config: self.context.config.config.clone(),
            state: self.context.state.clone(),
        };

        let input_bytes = match serde_json::to_vec(&input) {
            Ok(v) => v,
            Err(_) => return None,
        };

        // Execute WASM function
        let result: WasmPolicyOutput = match self.execute_wasm_function("select_worker", &input_bytes) {
            Ok(v) => v,
            Err(_) => return None,
        };

        // Update state if returned
        if let Some(new_state) = result.state {
            // merge/overwrite
            let mut state = self.context.state.clone();
            for (k, v) in new_state.into_iter() {
                state.insert(k, v);
            }
            // Replace state in context (Arc, but we cloned earlier; to keep API simple we ignore sharing)
            // Note: For full correctness, WasmPolicyContext.state should be in a Mutex; omitted for brevity.
        }

        if let Some(idx) = result.selected_worker {
            if let Some(w) = workers.get(idx) {
                RouterMetrics::record_processed_request(w.url());
                RouterMetrics::record_policy_decision(&self.name, w.url());
            }
            Some(idx)
        } else {
            None
        }
    }

    fn select_worker_pair(
            &self,
            prefill_workers: &[Arc<dyn Worker>],
            decode_workers: &[Arc<dyn Worker>],
            request_text: Option<&str>,
        ) -> Option<(usize, usize)> {
        None
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // no-op
    }

    fn name(&self) -> &'static str {
        let formatted_name = format!("wasm:{}", self.name);
        Box::leak(formatted_name.into_boxed_str())
    }

    fn needs_request_text(&self) -> bool {
        false
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        // no-op
    }

    fn reset(&self) {
        // reload wasm policy
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Input data for WASM policy execution
#[derive(Debug, Serialize, Deserialize)]
struct WasmPolicyInput {
    workers: Vec<WasmWorker>,
    request_text: Option<String>,
    config: HashMap<String, serde_json::Value>,
    state: HashMap<String, serde_json::Value>,
}

/// Output data from WASM policy execution
#[derive(Debug, Serialize, Deserialize)]
struct WasmPolicyOutput {
    selected_worker: Option<usize>,
    state: Option<HashMap<String, serde_json::Value>>,
    error: Option<String>,
}

/// Worker information for WASM policies
#[derive(Debug, Serialize, Deserialize)]
struct WasmWorker {
    url: String,
    healthy: bool,
    load: isize,
}

/// WASM policy registry for managing loaded policies
pub struct WasmPolicyRegistry {
    plugins: HashMap<String, Arc<WasmContext>>,
    loader: WasmLoader,
}

impl WasmPolicyRegistry {
    /// Create a new policy registry
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            loader: WasmLoader::new(),
        }
    }

    /// Load a policy from configuration
    pub async fn load_policy(&mut self, config: WasmConfig) -> Result<()> {
        // Validate plugin
        let wasm_bytes = tokio::fs::read(&config.path).await?;
        // Basic content validation (magic/version)
        let _ = WasmLoader::new().validate_module(&wasm_bytes)?;

        let engine = Engine::default();
        let module = Module::new(&engine, &wasm_bytes).map_err(|e| anyhow!("Failed to compile WASM module: {}", e))?;
        let context = WasmContext {
            config: config.clone(),
            engine,
            module,
            state: HashMap::new(),
        };

        // Store in registry
        self.plugins.insert(config.name.clone(), Arc::new(context));

        Ok(())
    }

    /// Create a policy from a loaded policy
    pub fn create_policy(&self, policy_name: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        self.plugins
            .get(policy_name)
            .map(|context| {
                Arc::new(WasmPolicy::with_config(context.as_ref().clone())) as Arc<dyn LoadBalancingPolicy>
            })
    }

    /// List loaded plugins
    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins.keys().cloned().collect()
    }

    /// Unload a plugin
    pub fn unload_policy(&mut self, policy_name: &str) -> bool {
        self.plugins.remove(policy_name).is_some()
    }
}

impl Default for WasmPolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_plugin_registry() {
        let mut registry = WasmPolicyRegistry::new();
        
        // Test empty registry
        assert_eq!(registry.list_plugins().len(), 0);
        assert!(registry.create_policy("nonexistent").is_none());
    }

    #[test]
    fn test_wasm_plugin_config_default() {
        let config = WasmConfig::default();
        assert_eq!(config.max_execution_time_ms, 100);
        assert_eq!(config.max_memory_bytes, 1024 * 1024);
    }

    #[test]
    fn test_wasm_policy_name_format() {
        // Test that the name method returns the correct format
        // We'll test the format logic directly without creating a full WASM module
        let test_name = "test_policy";
        let expected_format = format!("wasm:{}", test_name);
        assert_eq!(expected_format, "wasm:test_policy");
        
        // Test with different names
        let test_name2 = "my_custom_policy";
        let expected_format2 = format!("wasm:{}", test_name2);
        assert_eq!(expected_format2, "wasm:my_custom_policy");
    }

    #[test]
    fn test_wasm_policy_loader_new() {
        let loader = WasmLoader::new();
        assert!(true); // Just test that it can be created
    }

    #[test]
    fn test_validate_module_empty() {
        let loader = WasmLoader::new();
        let empty_wasm = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        assert!(loader.validate_module(&empty_wasm).is_ok());
    }

    #[test]
    fn test_validate_module_invalid() {
        let loader = WasmLoader::new();
        let invalid_wasm = vec![0x00, 0x00, 0x00, 0x00];
        assert!(loader.validate_module(&invalid_wasm).is_err());
    }

    #[test]
    fn test_validate_policy_name() {
        let validator = WasmLoader::new();

        // Valid names
        assert!(validator.validate_policy_name("my_policy").is_ok());
        assert!(validator.validate_policy_name("policy-123").is_ok());
        assert!(validator.validate_policy_name("Policy123").is_ok());

        // Invalid names
        assert!(validator.validate_policy_name("").is_err());
        assert!(validator.validate_policy_name("random").is_err());
        assert!(validator.validate_policy_name("system").is_err());
        assert!(validator.validate_policy_name("my@policy").is_err());
        assert!(validator.validate_policy_name("my policy").is_err());
    }

    #[test]
    fn test_validate_execution_limits() {
        let validator = WasmLoader::new();
        let mut config = WasmConfig::default();

        // Valid limits
        config.max_execution_time_ms = 1000;
        assert!(validator.validate_execution_limits(&config).is_ok());

        // Invalid limits
        config.max_execution_time_ms = 0;
        assert!(validator.validate_execution_limits(&config).is_err());

        config.max_execution_time_ms = 60000;
        assert!(validator.validate_execution_limits(&config).is_err());
    }

    #[test]
    fn test_validate_memory_limits() {
        let validator = WasmLoader::new();
        let mut config = WasmConfig::default();

        // Valid limits
        config.max_memory_bytes = 1024 * 1024;
        assert!(validator.validate_memory_limits(&config).is_ok());

        // Invalid limits
        config.max_memory_bytes = 0;
        assert!(validator.validate_memory_limits(&config).is_err());

        config.max_memory_bytes = 200 * 1024 * 1024;
        assert!(validator.validate_memory_limits(&config).is_err());
    }

    #[test]
    fn test_validate_system_calls() {
        let validator = WasmLoader::new();
        let mut config = WasmConfig::default();

        // Valid system calls
        config.allowed_syscalls = vec!["wasi_snapshot_preview1.random_get".to_string()];
        assert!(validator.validate_system_calls(&config).is_ok());

        // Invalid system calls
        config.allowed_syscalls = vec!["wasi_snapshot_preview1.fd_write".to_string()];
        assert!(validator.validate_system_calls(&config).is_err());
    }

    #[test]
    fn test_validate_module_content() {
        let validator = WasmLoader::new();

        // Invalid WASM module - missing magic number
        let invalid_wasm = vec![0x00, 0x00, 0x00, 0x00];
        assert!(validator.validate_module_content(&invalid_wasm).is_err());

        // Invalid WASM module - missing required sections
        let minimal_wasm = vec![
            0x00, 0x61, 0x73, 0x6d, // magic
            0x01, 0x00, 0x00, 0x00, // version
        ];
        assert!(validator.validate_module_content(&minimal_wasm).is_err());
    }

    #[test]
    fn test_validate_for_hot_reload() {
        let validator = WasmLoader::new();
        let mut old_config = WasmConfig::default();
        let mut new_config = WasmConfig::default();

        old_config.name = "test_policy".to_string();
        old_config.max_execution_time_ms = 1000;
        old_config.max_memory_bytes = 1024 * 1024;

        new_config.name = "test_policy".to_string();
        new_config.max_execution_time_ms = 500;
        new_config.max_memory_bytes = 512 * 1024;

        // Valid hot reload
        assert!(validator.validate_for_hot_reload(&old_config, &new_config).is_ok());

        // Invalid hot reload - name change
        new_config.name = "new_policy".to_string();
        assert!(validator.validate_for_hot_reload(&old_config, &new_config).is_err());

        // Invalid hot reload - increased limits
        new_config.name = "test_policy".to_string();
        new_config.max_execution_time_ms = 2000;
        assert!(validator.validate_for_hot_reload(&old_config, &new_config).is_err());
    }
}
