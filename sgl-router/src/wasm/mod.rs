//! WebAssembly (WASM) Management Module
//!
//! This module provides comprehensive WASM module management including:
//! - Module loading and unloading
//! - Secure execution with sandboxing
//! - Performance monitoring and metrics
//! - Resource management and limits
//! - Security violation detection and quarantine
//! - Module caching and persistence

pub mod wasm_manager;
pub mod config;
pub mod errors;
pub mod example;

// Re-export main types for easy access
pub use wasm_manager::{
    WasmManager,
    WasmManagerConfig,
    WasmModule,
    WasmModuleConfig,
    WasmModuleInfo,
    WasmModuleStatus,
    WasmModuleStats,
    WasmExecutionResult,
    SecurityContext,
    SecurityViolation,
    ViolationSeverity,
    PerformanceMetrics,
    ModuleHealthStatus,
    ManagerStats,
    NetworkAccess,
    FilesystemAccess,
    CpuLimits,
    TimeoutConfig,
    ValidationConfig,
    LoggingConfig,
    ResourceLimits,
    SecurityPolicies,
    WasmRuntimeInstance,
    ResourceUsageTracker,
    SecurityMonitor,
    ModuleCache,
    CachedModule,
};

// Re-export configuration types
pub use config::*;

// Re-export error types
pub use errors::*;

/// WASM module management result type
pub type WasmResult<T> = Result<T, WasmError>;

/// WASM module management error type
#[derive(Debug, thiserror::Error)]
pub enum WasmError {
    #[error("Module not found: {0}")]
    ModuleNotFound(String),
    
    #[error("Module already exists: {0}")]
    ModuleAlreadyExists(String),
    
    #[error("Module is quarantined: {0}")]
    ModuleQuarantined(String),
    
    #[error("Security violation: {0}")]
    SecurityViolation(String),
    
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
    
    #[error("Execution timeout: {0}")]
    ExecutionTimeout(String),
    
    #[error("Invalid WASM bytecode: {0}")]
    InvalidBytecode(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// WASM manager builder for fluent configuration
pub struct WasmManagerBuilder {
    config: WasmManagerConfig,
}

impl WasmManagerBuilder {
    /// Create a new WASM manager builder with default configuration
    pub fn new() -> Self {
        Self {
            config: WasmManagerConfig::default(),
        }
    }

    /// Set maximum concurrent modules
    pub fn max_concurrent_modules(mut self, count: usize) -> Self {
        self.config.max_concurrent_modules = count;
        self
    }

    /// Set default module timeout
    pub fn default_timeout(mut self, timeout_secs: u64) -> Self {
        self.config.default_module_timeout_secs = timeout_secs;
        self
    }

    /// Enable or disable performance monitoring
    pub fn enable_performance_monitoring(mut self, enable: bool) -> Self {
        self.config.enable_performance_monitoring = enable;
        self
    }

    /// Enable or disable security sandbox by default
    pub fn enable_sandbox_by_default(mut self, enable: bool) -> Self {
        self.config.enable_sandbox_by_default = enable;
        self
    }

    /// Set maximum memory limit
    pub fn max_memory(mut self, memory_bytes: u64) -> Self {
        self.config.resource_limits.max_total_memory_bytes = memory_bytes;
        self
    }

    /// Set maximum CPU usage percentage
    pub fn max_cpu_percent(mut self, cpu_percent: f32) -> Self {
        self.config.resource_limits.max_cpu_percent = cpu_percent;
        self
    }

    /// Build the WASM manager
    pub fn build(self) -> WasmManager {
        WasmManager::new(self.config)
    }
}

impl Default for WasmManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common WASM operations
pub mod utils {
    use super::*;
    use std::path::Path;

    /// Load a WASM module from file with default configuration
    pub async fn load_module_from_file<P: AsRef<Path>>(
        manager: &WasmManager,
        file_path: P,
        module_name: String,
    ) -> WasmResult<String> {
        let path = file_path.as_ref().to_string_lossy().to_string();
        
        let config = WasmModuleConfig {
            name: module_name,
            version: "1.0.0".to_string(),
            description: None,
            path,
            author: None,
            license: None,
            dependencies: vec![],
            security_context: SecurityContext::default(),
            preload_modules: vec![],
            tags: vec![],
            priority: 0,
            cache_enabled: true,
            timeout_config: TimeoutConfig::default(),
        };

        manager.load_module(config).await
            .map_err(|e| WasmError::Other(e))
    }

    /// Execute a WASM module function with simple input
    pub async fn execute_simple_function(
        manager: &WasmManager,
        module_name: &str,
        function_name: &str,
        input: Option<serde_json::Value>,
    ) -> WasmResult<WasmExecutionResult> {
        manager.execute_module(module_name, function_name, input).await
            .map_err(|e| WasmError::Other(e))
    }

    /// Get module health status as string
    pub async fn get_module_health_string(
        manager: &WasmManager,
        module_name: &str,
    ) -> WasmResult<String> {
        let health = manager.get_module_health(module_name).await
            .map_err(|e| WasmError::Other(e))?;
        Ok(health.to_string())
    }

    /// Check if module is healthy (not quarantined or in error state)
    pub async fn is_module_healthy(
        manager: &WasmManager,
        module_name: &str,
    ) -> WasmResult<bool> {
        let health = manager.get_module_health(module_name).await
            .map_err(|e| WasmError::Other(e))?;
        Ok(matches!(health, ModuleHealthStatus::Active | ModuleHealthStatus::Idle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_manager_builder() {
        let manager = WasmManagerBuilder::new()
            .max_concurrent_modules(50)
            .default_timeout(60)
            .enable_performance_monitoring(true)
            .max_memory(256 * 1024 * 1024) // 256MB
            .build();

        assert_eq!(manager.config.max_concurrent_modules, 50);
        assert_eq!(manager.config.default_module_timeout_secs, 60);
        assert!(manager.config.enable_performance_monitoring);
        assert_eq!(manager.config.resource_limits.max_total_memory_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn test_error_display() {
        let error = WasmError::ModuleNotFound("test_module".to_string());
        assert!(error.to_string().contains("Module not found"));
        assert!(error.to_string().contains("test_module"));
    }
}
