# WASM Management Module

A comprehensive WebAssembly (WASM) management system for Rust applications that provides secure execution, performance monitoring, and resource management for WASM modules.

## Features

### 🔒 Security & Sandboxing
- **Sandboxed Execution**: Secure execution environment with configurable restrictions
- **Security Violation Detection**: Automatic detection and reporting of security violations
- **Module Quarantine**: Automatic quarantine of modules with security issues
- **Resource Limits**: Configurable memory, CPU, and execution time limits
- **Network & Filesystem Access Control**: Fine-grained permissions for external access

### 📊 Performance Monitoring
- **Real-time Metrics**: Execution time, memory usage, CPU utilization tracking
- **Performance Statistics**: Comprehensive statistics for each module
- **Resource Usage Tracking**: Global resource usage monitoring
- **Cache Management**: Intelligent module caching with hit/miss statistics

### 🚀 Module Management
- **Dynamic Loading**: Load and unload WASM modules at runtime
- **Module Registry**: Centralized registry for all loaded modules
- **Dependency Management**: Support for module dependencies
- **Lifecycle Management**: Complete module lifecycle from load to unload
- **Integrity Verification**: SHA256 hash verification for module integrity

### ⚡ Concurrency & Resource Management
- **Concurrent Execution**: Configurable limits on concurrent module executions
- **Resource Limits**: Global and per-module resource constraints
- **Automatic Cleanup**: Automatic cleanup of idle modules
- **Memory Management**: Efficient memory usage tracking and limits

## Quick Start

### Basic Usage

```rust
use sgl_router::wasm::{WasmManager, WasmManagerBuilder, WasmModuleConfig, SecurityContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create and configure the WASM manager
    let manager = WasmManagerBuilder::new()
        .max_concurrent_modules(10)
        .default_timeout(30)
        .enable_performance_monitoring(true)
        .max_memory(256 * 1024 * 1024) // 256MB
        .build();

    // Start the manager
    manager.start().await?;

    // Configure a WASM module
    let module_config = WasmModuleConfig {
        name: "my_module".to_string(),
        version: "1.0.0".to_string(),
        description: Some("My WASM module".to_string()),
        path: "/path/to/module.wasm".to_string(),
        author: Some("Author Name".to_string()),
        license: Some("MIT".to_string()),
        dependencies: vec![],
        security_context: SecurityContext::default(),
        preload_modules: vec![],
        tags: vec!["example".to_string()],
        priority: 1,
        cache_enabled: true,
        timeout_config: TimeoutConfig::default(),
    };

    // Load the module
    let module_id = manager.load_module(module_config).await?;
    println!("Module loaded with ID: {}", module_id);

    // Execute a function
    let input_data = serde_json::json!({
        "message": "Hello, WASM!",
        "value": 42
    });

    let result = manager.execute_module(
        "my_module",
        "process_data",
        Some(input_data),
    ).await?;

    println!("Execution result: {:?}", result);

    // Get module information
    let module_info = manager.get_module_info("my_module").await?;
    println!("Module stats: {:?}", module_info.stats);

    // Stop the manager
    manager.stop().await?;

    Ok(())
}
```

### Security Configuration

```rust
use sgl_router::wasm::{SecurityContext, NetworkAccess, FilesystemAccess, CpuLimits};

let security_context = SecurityContext {
    sandbox_enabled: true,
    allowed_syscalls: vec![],
    max_memory_bytes: Some(32 * 1024 * 1024), // 32MB
    max_execution_time_secs: 30,
    env_vars: std::collections::HashMap::new(),
    network_access: NetworkAccess {
        enabled: false, // No network access
        allowed_domains: vec![],
        allowed_ports: vec![],
        max_connections: 0,
    },
    filesystem_access: FilesystemAccess {
        enabled: false, // No filesystem access
        allowed_read_paths: vec![],
        allowed_write_paths: vec![],
        max_file_size_bytes: None,
    },
    cpu_limits: CpuLimits {
        max_cpu_percent: 50.0,
        max_threads: 2,
        cpu_time_limit_secs: Some(60),
    },
};
```

### Performance Monitoring

```rust
// Get manager statistics
let stats = manager.get_manager_stats().await;
println!("Total modules: {}", stats.total_modules);
println!("Active executions: {}", stats.active_executions);
println!("Memory usage: {} bytes", stats.total_memory_bytes);
println!("Cache hit rate: {:.2}%", 
    (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0);

// Get module health status
let health = manager.get_module_health("my_module").await?;
println!("Module health: {}", health);

// List all modules with their statistics
let modules = manager.list_modules().await;
for module in modules {
    println!("Module: {} - Executions: {}, Avg time: {:.2}ms", 
        module.name, 
        module.stats.total_executions,
        module.stats.avg_execution_time_ms);
}
```

## Architecture

### Core Components

1. **WasmManager**: Main manager class that orchestrates all operations
2. **WasmModule**: Individual module instance with metadata and runtime state
3. **WasmExecutor**: Execution engine for running module functions
4. **SecurityMonitor**: Monitors and enforces security policies
5. **ResourceUsageTracker**: Tracks global resource usage
6. **ModuleCache**: Manages module caching and persistence

### Security Model

The WASM management system implements a multi-layered security model:

1. **Sandboxing**: Each module runs in an isolated environment
2. **Resource Limits**: Strict limits on memory, CPU, and execution time
3. **Access Control**: Fine-grained control over network and filesystem access
4. **Violation Detection**: Automatic detection of security violations
5. **Quarantine System**: Automatic isolation of problematic modules

### Performance Features

- **Concurrent Execution**: Multiple modules can execute simultaneously
- **Resource Monitoring**: Real-time tracking of resource usage
- **Caching**: Intelligent caching of compiled modules
- **Statistics**: Comprehensive performance metrics
- **Cleanup**: Automatic cleanup of idle resources

## Configuration

### Manager Configuration

```rust
let config = WasmManagerConfig {
    max_concurrent_modules: 100,
    default_module_timeout_secs: 30,
    cleanup_interval_secs: 300,
    enable_module_cache: true,
    cache_size_limit_bytes: 100 * 1024 * 1024, // 100MB
    enable_performance_monitoring: true,
    enable_sandbox_by_default: true,
    max_modules_in_memory: 1000,
    validation_config: ValidationConfig {
        validate_bytecode: true,
        check_signatures: false,
        validate_dependencies: true,
        max_module_size_bytes: 50 * 1024 * 1024, // 50MB
        allowed_features: vec!["bulk-memory".to_string(), "reference-types".to_string()],
    },
    logging_config: LoggingConfig {
        log_level: "info".to_string(),
        log_execution_details: true,
        log_security_violations: true,
        log_performance_metrics: false,
        max_log_file_size_bytes: 10 * 1024 * 1024, // 10MB
    },
    resource_limits: ResourceLimits {
        max_total_memory_bytes: 1024 * 1024 * 1024, // 1GB
        max_cpu_percent: 80.0,
        max_file_descriptors: 1024,
        max_network_connections: 100,
    },
    security_policies: SecurityPolicies {
        default_security_context: SecurityContext::default(),
        enforce_strict_security: true,
        allow_unsigned_modules: true,
        quarantine_on_violation: true,
        max_violations_before_quarantine: 3,
    },
};
```

## Error Handling

The module provides comprehensive error handling with specific error types:

```rust
use sgl_router::wasm::{WasmError, WasmResult};

async fn handle_module_operation() -> WasmResult<()> {
    match manager.execute_module("module", "function", None).await {
        Ok(result) => {
            println!("Execution successful: {:?}", result);
            Ok(())
        }
        Err(WasmError::ModuleNotFound(name)) => {
            eprintln!("Module not found: {}", name);
            Ok(())
        }
        Err(WasmError::SecurityViolation(msg)) => {
            eprintln!("Security violation: {}", msg);
            Ok(())
        }
        Err(e) => Err(e),
    }
}
```

## Testing

The module includes comprehensive tests and examples:

```rust
// Run the basic usage example
cargo test test_basic_usage

// Run security example
cargo test test_security_example

// Run performance monitoring example
cargo test test_performance_monitoring
```

## Dependencies

The WASM management module requires the following dependencies:

```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"
uuid = { version = "1.0", features = ["v4"] }
sha2 = "0.10"
chrono = { version = "0.4", features = ["serde"] }
```

## Future Enhancements

- **WASM Runtime Integration**: Integration with actual WASM runtimes (wasmtime, wasmer)
- **Module Signing**: Digital signature verification for modules
- **Hot Reloading**: Dynamic module updates without restart
- **Distributed Execution**: Support for distributed WASM execution
- **Advanced Caching**: More sophisticated caching strategies
- **Metrics Export**: Integration with monitoring systems (Prometheus, etc.)

## License

This module is part of the sgl-router project and follows the same license terms.
