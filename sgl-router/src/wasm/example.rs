//! WASM Manager Usage Example
//!
//! This example demonstrates how to use the WASM management module
//! to load, execute, and manage WebAssembly modules with security
//! and performance monitoring.

use anyhow::Result;
use sgl_router::wasm::wasm_manager::*;
use std::collections::HashMap;
use tracing::{info, warn};

/// Example demonstrating basic WASM manager usage
pub async fn basic_usage_example() -> Result<()> {
    info!("Starting WASM manager basic usage example");

    // Create manager configuration
    let config = WasmManagerConfig {
        max_concurrent_modules: 10,
        default_module_timeout_secs: 30,
        cleanup_interval_secs: 60,
        enable_module_cache: true,
        cache_size_limit_bytes: 50 * 1024 * 1024, // 50MB
        enable_performance_monitoring: true,
        enable_sandbox_by_default: true,
        max_modules_in_memory: 100,
        validation_config: ValidationConfig {
            validate_bytecode: true,
            check_signatures: false,
            validate_dependencies: true,
            max_module_size_bytes: 10 * 1024 * 1024, // 10MB
            allowed_features: vec!["bulk-memory".to_string(), "reference-types".to_string()],
        },
        logging_config: LoggingConfig {
            log_level: "info".to_string(),
            log_execution_details: true,
            log_security_violations: true,
            log_performance_metrics: false,
            max_log_file_size_bytes: 5 * 1024 * 1024, // 5MB
        },
        resource_limits: ResourceLimits {
            max_total_memory_bytes: 512 * 1024 * 1024, // 512MB
            max_cpu_percent: 70.0,
            max_file_descriptors: 512,
            max_network_connections: 50,
        },
        security_policies: SecurityPolicies {
            default_security_context: SecurityContext::default(),
            enforce_strict_security: true,
            allow_unsigned_modules: true,
            quarantine_on_violation: true,
            max_violations_before_quarantine: 3,
        },
    };

    // Create and start the WASM manager
    let manager = WasmManager::new(config);
    manager.start().await?;

    // Create a sample module configuration
    let module_config = WasmModuleConfig {
        name: "example_module".to_string(),
        version: "1.0.0".to_string(),
        description: Some("Example WASM module for demonstration".to_string()),
        path: "/tmp/example.wasm".to_string(),
        author: Some("Example Author".to_string()),
        license: Some("MIT".to_string()),
        dependencies: vec![],
        security_context: SecurityContext {
            sandbox_enabled: true,
            allowed_syscalls: vec![],
            max_memory_bytes: Some(32 * 1024 * 1024), // 32MB
            max_execution_time_secs: 30,
            env_vars: HashMap::new(),
            network_access: NetworkAccess {
                enabled: false,
                allowed_domains: vec![],
                allowed_ports: vec![],
                max_connections: 0,
            },
            filesystem_access: FilesystemAccess {
                enabled: false,
                allowed_read_paths: vec![],
                allowed_write_paths: vec![],
                max_file_size_bytes: None,
            },
            cpu_limits: CpuLimits {
                max_cpu_percent: 50.0,
                max_threads: 2,
                cpu_time_limit_secs: Some(60),
            },
        },
        preload_modules: vec![],
        tags: vec!["example".to_string(), "demo".to_string()],
        priority: 1,
        cache_enabled: true,
        timeout_config: TimeoutConfig {
            default_timeout_secs: 30,
            max_timeout_secs: 120,
            allow_extension: true,
            max_extensions: 2,
        },
    };

    // Create a dummy WASM bytecode for demonstration
    let dummy_bytecode = create_dummy_wasm_bytecode();

    // Create module instance manually (in real usage, you would load from file)
    let module = WasmModule::new(module_config.clone(), dummy_bytecode)?;
    let module_arc = Arc::new(module);

    // Register the module
    {
        let mut modules = manager.modules.write().await;
        modules.insert(module_config.name.clone(), module_arc);
    }

    info!("Module '{}' loaded successfully", module_config.name);

    // Execute the module
    let input_data = serde_json::json!({
        "message": "Hello from WASM manager!",
        "value": 42
    });

    let result = manager.execute_module(
        &module_config.name,
        "process_data",
        Some(input_data),
    ).await?;

    info!("Execution result: success={}, time={}ms", 
          result.success, result.execution_time_ms);

    if let Some(data) = result.data {
        info!("Execution data: {}", data);
    }

    if let Some(error) = result.error {
        warn!("Execution error: {}", error);
    }

    // Get module information
    let module_info = manager.get_module_info(&module_config.name).await?;
    info!("Module info: name={}, status={}, executions={}", 
          module_info.name, module_info.status, module_info.stats.total_executions);

    // Get manager statistics
    let manager_stats = manager.get_manager_stats().await;
    info!("Manager stats: modules={}, active_executions={}, memory={} bytes",
          manager_stats.total_modules, 
          manager_stats.active_executions,
          manager_stats.total_memory_bytes);

    // List all modules
    let modules = manager.list_modules().await;
    info!("Loaded modules: {}", modules.len());
    for module in modules {
        info!("  - {} (v{}, status: {})", 
              module.name, module.version, module.status);
    }

    // Stop the manager
    manager.stop().await?;
    info!("WASM manager stopped successfully");

    Ok(())
}

/// Example demonstrating security features
pub async fn security_example() -> Result<()> {
    info!("Starting WASM manager security example");

    let config = WasmManagerConfig::default();
    let manager = WasmManager::new(config);
    manager.start().await?;

    // Create a module with restricted security context
    let restricted_config = WasmModuleConfig {
        name: "restricted_module".to_string(),
        version: "1.0.0".to_string(),
        description: Some("Module with restricted permissions".to_string()),
        path: "/tmp/restricted.wasm".to_string(),
        author: None,
        license: None,
        dependencies: vec![],
        security_context: SecurityContext {
            sandbox_enabled: true,
            allowed_syscalls: vec![],
            max_memory_bytes: Some(1024 * 1024), // 1MB
            max_execution_time_secs: 5,
            env_vars: HashMap::new(),
            network_access: NetworkAccess {
                enabled: false,
                allowed_domains: vec![],
                allowed_ports: vec![],
                max_connections: 0,
            },
            filesystem_access: FilesystemAccess {
                enabled: false,
                allowed_read_paths: vec![],
                allowed_write_paths: vec![],
                max_file_size_bytes: None,
            },
            cpu_limits: CpuLimits {
                max_cpu_percent: 10.0,
                max_threads: 1,
                cpu_time_limit_secs: Some(10),
            },
        },
        preload_modules: vec![],
        tags: vec!["restricted".to_string()],
        priority: 0,
        cache_enabled: false,
        timeout_config: TimeoutConfig {
            default_timeout_secs: 5,
            max_timeout_secs: 10,
            allow_extension: false,
            max_extensions: 0,
        },
    };

    let dummy_bytecode = create_dummy_wasm_bytecode();
    let module = WasmModule::new(restricted_config.clone(), dummy_bytecode)?;
    let module_arc = Arc::new(module);

    // Register the module
    {
        let mut modules = manager.modules.write().await;
        modules.insert(restricted_config.name.clone(), module_arc);
    }

    // Try to execute with a dangerous function name
    let result = manager.execute_module(
        &restricted_config.name,
        "system", // Dangerous function name
        None,
    ).await;

    match result {
        Ok(execution_result) => {
            if !execution_result.security_violations.is_empty() {
                info!("Security violations detected: {}", 
                      execution_result.security_violations.len());
                for violation in execution_result.security_violations {
                    warn!("Violation: {} - {}", 
                          violation.violation_type, violation.description);
                }
            }
        }
        Err(e) => {
            warn!("Execution failed due to security restrictions: {}", e);
        }
    }

    // Check module health
    let health = manager.get_module_health(&restricted_config.name).await?;
    info!("Module health status: {}", health);

    manager.stop().await?;
    Ok(())
}

/// Example demonstrating performance monitoring
pub async fn performance_monitoring_example() -> Result<()> {
    info!("Starting WASM manager performance monitoring example");

    let config = WasmManagerConfig {
        enable_performance_monitoring: true,
        ..Default::default()
    };

    let manager = WasmManager::new(config);
    manager.start().await?;

    // Create multiple modules for performance testing
    for i in 0..5 {
        let module_config = WasmModuleConfig {
            name: format!("perf_module_{}", i),
            version: "1.0.0".to_string(),
            description: Some(format!("Performance test module {}", i)),
            path: format!("/tmp/perf_{}.wasm", i),
            author: None,
            license: None,
            dependencies: vec![],
            security_context: SecurityContext::default(),
            preload_modules: vec![],
            tags: vec!["performance".to_string()],
            priority: i,
            cache_enabled: true,
            timeout_config: TimeoutConfig::default(),
        };

        let dummy_bytecode = create_dummy_wasm_bytecode();
        let module = WasmModule::new(module_config.clone(), dummy_bytecode)?;
        let module_arc = Arc::new(module);

        {
            let mut modules = manager.modules.write().await;
            modules.insert(module_config.name.clone(), module_arc);
        }

        // Execute the module multiple times
        for j in 0..10 {
            let input = serde_json::json!({
                "iteration": j,
                "module_id": i
            });

            let result = manager.execute_module(
                &module_config.name,
                "process_data",
                Some(input),
            ).await?;

            info!("Module {} iteration {}: {}ms", 
                  module_config.name, j, result.execution_time_ms);
        }
    }

    // Get performance statistics
    let manager_stats = manager.get_manager_stats().await;
    info!("Performance stats:");
    info!("  Total modules: {}", manager_stats.total_modules);
    info!("  Cache hits: {}", manager_stats.cache_hits);
    info!("  Cache misses: {}", manager_stats.cache_misses);
    info!("  Total memory: {} bytes", manager_stats.total_memory_bytes);

    // Get individual module statistics
    let modules = manager.list_modules().await;
    for module_info in modules {
        let stats = &module_info.stats;
        info!("Module {} stats:", module_info.name);
        info!("  Total executions: {}", stats.total_executions);
        info!("  Success rate: {:.2}%", 
              (stats.successful_executions as f64 / stats.total_executions as f64) * 100.0);
        info!("  Avg execution time: {:.2}ms", stats.avg_execution_time_ms);
        info!("  Peak memory: {} bytes", stats.peak_memory_usage_bytes);
    }

    manager.stop().await?;
    Ok(())
}

/// Create dummy WASM bytecode for testing
fn create_dummy_wasm_bytecode() -> Vec<u8> {
    // This is a minimal valid WASM module
    vec![
        0x00, 0x61, 0x73, 0x6d, // WASM magic number
        0x01, 0x00, 0x00, 0x00, // Version 1
        0x00, 0x00, 0x00, 0x00, // Empty module
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_usage() {
        // This test would require actual WASM files
        // For now, we'll just test the configuration creation
        let config = WasmManagerConfig::default();
        assert!(config.max_concurrent_modules > 0);
        assert!(config.enable_performance_monitoring);
    }

    #[tokio::test]
    async fn test_dummy_bytecode_creation() {
        let bytecode = create_dummy_wasm_bytecode();
        assert_eq!(bytecode[0..4], [0x00, 0x61, 0x73, 0x6d]); // WASM magic
    }
}
