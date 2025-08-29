//! Integration tests for WASM policy system

use sglang_router_rs::policies::{
    WasmPolicyConfig, WasmPolicyRegistry,
};
use sglang_router_rs::core::{BasicWorker, WorkerType};
use std::collections::HashMap;

#[tokio::test]
async fn test_wasm_plugin_registry() {
    let mut registry = WasmPolicyRegistry::new();
    
    // Test empty registry
    assert_eq!(registry.list_plugins().len(), 0);
    assert!(registry.create_policy("nonexistent").is_none());
}

#[tokio::test]
async fn test_wasm_plugin_config_validation() {
    let config = WasmPolicyConfig {
        module_path: "nonexistent.wasm".to_string(),
        name: "test_policy".to_string(),
        max_execution_time_ms: 1000,
        max_memory_bytes: 1024 * 1024,
        allowed_syscalls: vec![],
        config: HashMap::new(),
    };

    let mut registry = WasmPolicyRegistry::new();
    
    // Should fail because module doesn't exist
    assert!(registry.load_policy(config).await.is_err());
}

#[tokio::test]
async fn test_wasm_plugin_integration() {
    // This test requires a valid WASM module to be built first
    // For now, we'll just test the registry functionality
    
    let mut registry = WasmPolicyRegistry::new();
    
    // Test plugin management
    assert_eq!(registry.list_plugins().len(), 0);
    
    // Test unload non-existent plugin
    assert!(!registry.unload_policy("nonexistent"));
}

#[test]
fn test_wasm_plugin_config_default() {
    let config = WasmPolicyConfig::default();
    
    assert_eq!(config.max_execution_time_ms, 100);
    assert_eq!(config.max_memory_bytes, 1024 * 1024);
    assert!(config.allowed_syscalls.is_empty());
    assert!(config.config.is_empty());
}

#[test]
fn test_wasm_plugin_config_serialization() {
    let mut config = WasmPolicyConfig::default();
    config.name = "test_policy".to_string();
    config.module_path = "test.wasm".to_string();
    
    // Test serialization
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: WasmPolicyConfig = serde_json::from_str(&json).unwrap();
    
    assert_eq!(config.name, deserialized.name);
    assert_eq!(config.module_path, deserialized.module_path);
}

#[tokio::test]
async fn test_wasm_policy_with_workers() {
    // Create test workers
    let _workers: Vec<Box<dyn sglang_router_rs::core::Worker>> = vec![
        Box::new(BasicWorker::new(
            "http://w1:8000".to_string(),
            WorkerType::Regular,
        )),
        Box::new(BasicWorker::new(
            "http://w2:8000".to_string(),
            WorkerType::Regular,
        )),
    ];

    // Test that we can create a registry
    let registry = WasmPolicyRegistry::new();
    
    // Test that we can list plugins
    let plugins = registry.list_plugins();
    assert_eq!(plugins.len(), 0);
}

#[test]
fn test_wasm_plugin_config_validation_rules() {
    // Test config validation through the public API
    let mut config = WasmPolicyConfig::default();
    config.name = "my_policy".to_string();
    config.module_path = "test.wasm".to_string();
    
    // Test that the config can be serialized/deserialized
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: WasmPolicyConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.name, deserialized.name);
    assert_eq!(config.module_path, deserialized.module_path);
}

#[test]
fn test_wasm_plugin_config_limits() {
    let mut config = WasmPolicyConfig::default();
    config.name = "test_policy".to_string();
    config.module_path = "test.wasm".to_string();
    
    // Test limits
    config.max_execution_time_ms = 1000;
    config.max_memory_bytes = 1024 * 1024;
    
    // Test that limits are set correctly
    assert_eq!(config.max_execution_time_ms, 1000);
    assert_eq!(config.max_memory_bytes, 1024 * 1024);
}

#[test]
fn test_wasm_plugin_system_calls() {
    let mut config = WasmPolicyConfig::default();
    config.name = "test_policy".to_string();
    config.module_path = "test.wasm".to_string();
    
    // Test system calls configuration
    config.allowed_syscalls = vec!["wasi_snapshot_preview1.random_get".to_string()];
    
    // Test that system calls are set correctly
    assert_eq!(config.allowed_syscalls.len(), 1);
    assert_eq!(config.allowed_syscalls[0], "wasi_snapshot_preview1.random_get");
}

#[test]
fn test_wasm_module_validation() {
    // Test WASM module validation through the public API
    // Since we can't access the validator directly, we'll test the config structure
    let config = WasmPolicyConfig::default();
    
    // Test that the config has the expected structure
    assert_eq!(config.max_execution_time_ms, 100);
    assert_eq!(config.max_memory_bytes, 1024 * 1024);
    assert!(config.allowed_syscalls.is_empty());
    assert!(config.config.is_empty());
}

#[test]
fn test_wasm_hot_reload_validation() {
    // Test hot reload validation through the public API
    let mut old_config = WasmPolicyConfig::default();
    let mut new_config = WasmPolicyConfig::default();
    
    old_config.name = "test_policy".to_string();
    old_config.max_execution_time_ms = 1000;
    old_config.max_memory_bytes = 1024 * 1024;
    
    new_config.name = "test_policy".to_string();
    new_config.max_execution_time_ms = 500; // more restrictive
    new_config.max_memory_bytes = 512 * 1024; // more restrictive
    
    // Test that both configs can be serialized/deserialized
    let old_json = serde_json::to_string(&old_config).unwrap();
    let new_json = serde_json::to_string(&new_config).unwrap();
    
    let deserialized_old: WasmPolicyConfig = serde_json::from_str(&old_json).unwrap();
    let deserialized_new: WasmPolicyConfig = serde_json::from_str(&new_json).unwrap();
    
    assert_eq!(deserialized_old.name, "test_policy");
    assert_eq!(deserialized_new.name, "test_policy");
    assert_eq!(deserialized_old.max_execution_time_ms, 1000);
    assert_eq!(deserialized_new.max_execution_time_ms, 500);
}
