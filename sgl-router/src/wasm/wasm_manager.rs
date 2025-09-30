//! WebAssembly (WASM) Management Module
//!
//! Provides comprehensive WASM module management including loading, execution, 
//! lifecycle management, security sandboxing, and resource management.
//! Supports dynamic WASM module loading with secure execution environment
//! and advanced monitoring capabilities.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tracing::{debug, info, warn};
use uuid::Uuid;
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicU32, Ordering};
use std::fmt;
use sha2::{Sha256, Digest};

/// WASM module execution status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WasmModuleStatus {
    /// Module loaded but not initialized
    Loaded,
    /// Module initialized and ready for execution
    Initialized,
    /// Module currently executing
    Running,
    /// Module execution paused
    Paused,
    /// Module stopped
    Stopped,
    /// Module encountered an error
    Error(String),
    /// Module is being compiled/validated
    Compiling,
    /// Module is being unloaded
    Unloading,
}

impl fmt::Display for WasmModuleStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WasmModuleStatus::Loaded => write!(f, "Loaded"),
            WasmModuleStatus::Initialized => write!(f, "Initialized"),
            WasmModuleStatus::Running => write!(f, "Running"),
            WasmModuleStatus::Paused => write!(f, "Paused"),
            WasmModuleStatus::Stopped => write!(f, "Stopped"),
            WasmModuleStatus::Error(msg) => write!(f, "Error: {}", msg),
            WasmModuleStatus::Compiling => write!(f, "Compiling"),
            WasmModuleStatus::Unloading => write!(f, "Unloading"),
        }
    }
}

/// WASM module information and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleInfo {
    /// Unique module identifier
    pub id: String,
    /// Module name
    pub name: String,
    /// Module version
    pub version: String,
    /// Module description
    pub description: Option<String>,
    /// Module file path
    pub path: String,
    /// Current module status
    pub status: WasmModuleStatus,
    /// Module creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Module size in bytes
    pub size_bytes: u64,
    /// Module hash for integrity verification
    pub hash: Option<String>,
    /// Module author information
    pub author: Option<String>,
    /// Module license information
    pub license: Option<String>,
    /// Module dependencies
    pub dependencies: Vec<String>,
    /// Execution statistics
    pub stats: WasmModuleStats,
    /// Security context information
    pub security_context: SecurityContext,
}

/// WASM module information with interior mutability for runtime updates
#[derive(Debug)]
pub struct WasmModuleInfoInner {
    /// Unique module identifier
    pub id: String,
    /// Module name
    pub name: String,
    /// Module version
    pub version: String,
    /// Module description
    pub description: Option<String>,
    /// Module file path
    pub path: String,
    /// Current module status
    pub status: Arc<RwLock<WasmModuleStatus>>,
    /// Module creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub updated_at: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
    /// Module size in bytes
    pub size_bytes: u64,
    /// Module hash for integrity verification
    pub hash: Option<String>,
    /// Module author information
    pub author: Option<String>,
    /// Module license information
    pub license: Option<String>,
    /// Module dependencies
    pub dependencies: Vec<String>,
    /// Execution statistics
    pub stats: Arc<RwLock<WasmModuleStats>>,
    /// Security context information
    pub security_context: SecurityContext,
    /// Last access time for cleanup
    pub last_accessed: Arc<RwLock<Instant>>,
}

/// Security context for WASM module execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// Sandbox mode enabled
    pub sandbox_enabled: bool,
    /// Allowed system calls
    pub allowed_syscalls: Vec<String>,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<usize>,
    /// Maximum execution time in seconds
    pub max_execution_time_secs: u64,
    /// Environment variables available to module
    pub env_vars: HashMap<String, String>,
    /// Network access permissions
    pub network_access: NetworkAccess,
    /// File system access permissions
    pub filesystem_access: FilesystemAccess,
    /// CPU usage limits
    pub cpu_limits: CpuLimits,
}

/// Network access configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAccess {
    /// Allow network access
    pub enabled: bool,
    /// Allowed domains/URLs
    pub allowed_domains: Vec<String>,
    /// Allowed ports
    pub allowed_ports: Vec<u16>,
    /// Maximum concurrent connections
    pub max_connections: usize,
}

/// Filesystem access configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemAccess {
    /// Allow filesystem access
    pub enabled: bool,
    /// Allowed read paths
    pub allowed_read_paths: Vec<String>,
    /// Allowed write paths
    pub allowed_write_paths: Vec<String>,
    /// Maximum file size for operations
    pub max_file_size_bytes: Option<u64>,
}

/// CPU usage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuLimits {
    /// Maximum CPU usage percentage
    pub max_cpu_percent: f32,
    /// Maximum number of threads
    pub max_threads: usize,
    /// CPU time limit in seconds
    pub cpu_time_limit_secs: Option<u64>,
}

impl Default for SecurityContext {
    fn default() -> Self {
        Self {
            sandbox_enabled: true,
            allowed_syscalls: vec![],
            max_memory_bytes: Some(64 * 1024 * 1024), // 64MB
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
                max_threads: 4,
                cpu_time_limit_secs: Some(60),
            },
        }
    }
}

/// WASM module execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WasmModuleStats {
    /// Total number of executions
    pub total_executions: u64,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Number of failed executions
    pub failed_executions: u64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: u64,
    /// Minimum execution time in milliseconds
    pub min_execution_time_ms: u64,
    /// Last execution timestamp
    pub last_execution: Option<chrono::DateTime<chrono::Utc>>,
    /// Total memory usage in bytes
    pub total_memory_usage_bytes: u64,
    /// Peak memory usage in bytes
    pub peak_memory_usage_bytes: u64,
    /// Total CPU time used in milliseconds
    pub total_cpu_time_ms: u64,
    /// Number of times module was loaded
    pub load_count: u64,
    /// Number of times module was unloaded
    pub unload_count: u64,
    /// Current active executions
    pub active_executions: u32,
}

/// WASM execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmExecutionResult {
    /// Whether execution was successful
    pub success: bool,
    /// Return data from execution
    pub data: Option<serde_json::Value>,
    /// Error message if execution failed
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<usize>,
    /// CPU time used in milliseconds
    pub cpu_time_ms: Option<u64>,
    /// Execution ID for tracking
    pub execution_id: String,
    /// Module ID that was executed
    pub module_id: String,
    /// Function name that was executed
    pub function_name: String,
    /// Security violations detected
    pub security_violations: Vec<SecurityViolation>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Security violation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityViolation {
    /// Type of violation
    pub violation_type: String,
    /// Description of the violation
    pub description: String,
    /// Severity level
    pub severity: ViolationSeverity,
    /// Timestamp when violation occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Security violation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Low severity - warning
    Low,
    /// Medium severity - concerning
    Medium,
    /// High severity - dangerous
    High,
    /// Critical severity - immediate threat
    Critical,
}

/// Performance metrics for execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Instructions executed
    pub instructions_executed: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Page faults
    pub page_faults: u64,
    /// Context switches
    pub context_switches: u64,
}

/// WASM module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleConfig {
    /// Module name
    pub name: String,
    /// Module version
    pub version: String,
    /// Module description
    pub description: Option<String>,
    /// Module file path
    pub path: String,
    /// Module author
    pub author: Option<String>,
    /// Module license
    pub license: Option<String>,
    /// Module dependencies
    pub dependencies: Vec<String>,
    /// Security context configuration
    pub security_context: SecurityContext,
    /// Preload modules
    pub preload_modules: Vec<String>,
    /// Module tags for categorization
    pub tags: Vec<String>,
    /// Module priority (higher = more important)
    pub priority: i32,
    /// Whether module should be cached
    pub cache_enabled: bool,
    /// Module timeout configuration
    pub timeout_config: TimeoutConfig,
}

/// Timeout configuration for module execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Default execution timeout in seconds
    pub default_timeout_secs: u64,
    /// Maximum execution timeout in seconds
    pub max_timeout_secs: u64,
    /// Whether to allow timeout extension
    pub allow_extension: bool,
    /// Maximum number of timeout extensions
    pub max_extensions: u32,
}

/// WASM manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmManagerConfig {
    /// Maximum number of concurrent module executions
    pub max_concurrent_modules: usize,
    /// Default module timeout in seconds
    pub default_module_timeout_secs: u64,
    /// Module cleanup interval in seconds
    pub cleanup_interval_secs: u64,
    /// Whether to enable module caching
    pub enable_module_cache: bool,
    /// Cache size limit in bytes
    pub cache_size_limit_bytes: usize,
    /// Whether to enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Whether to enable security sandbox by default
    pub enable_sandbox_by_default: bool,
    /// Maximum number of modules to keep in memory
    pub max_modules_in_memory: usize,
    /// Module validation settings
    pub validation_config: ValidationConfig,
    /// Logging configuration
    pub logging_config: LoggingConfig,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Security policies
    pub security_policies: SecurityPolicies,
}

/// Module validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Whether to validate WASM bytecode on load
    pub validate_bytecode: bool,
    /// Whether to check module signatures
    pub check_signatures: bool,
    /// Whether to validate module dependencies
    pub validate_dependencies: bool,
    /// Maximum module size in bytes
    pub max_module_size_bytes: u64,
    /// Allowed WASM features
    pub allowed_features: Vec<String>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level for WASM operations
    pub log_level: String,
    /// Whether to log execution details
    pub log_execution_details: bool,
    /// Whether to log security violations
    pub log_security_violations: bool,
    /// Whether to log performance metrics
    pub log_performance_metrics: bool,
    /// Maximum log file size in bytes
    pub max_log_file_size_bytes: u64,
}

/// Resource limits for the WASM manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum total memory usage in bytes
    pub max_total_memory_bytes: u64,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: f32,
    /// Maximum number of file descriptors
    pub max_file_descriptors: usize,
    /// Maximum number of network connections
    pub max_network_connections: usize,
}

/// Security policies for WASM execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicies {
    /// Default security context for new modules
    pub default_security_context: SecurityContext,
    /// Whether to enforce strict security by default
    pub enforce_strict_security: bool,
    /// Whether to allow unsigned modules
    pub allow_unsigned_modules: bool,
    /// Whether to quarantine modules on security violations
    pub quarantine_on_violation: bool,
    /// Maximum number of security violations before quarantine
    pub max_violations_before_quarantine: u32,
}

impl Default for WasmManagerConfig {
    fn default() -> Self {
        Self {
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
        }
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout_secs: 30,
            max_timeout_secs: 300,
            allow_extension: true,
            max_extensions: 3,
        }
    }
}

/// WASM module instance
pub struct WasmModule {
    /// Module information and metadata with interior mutability
    pub info: WasmModuleInfoInner,
    /// Module configuration
    pub config: WasmModuleConfig,
    /// Module bytecode
    pub bytecode: Vec<u8>,
    /// Reference count for garbage collection
    pub ref_count: Arc<Mutex<u32>>,
    /// Module execution semaphore for concurrency control
    pub execution_semaphore: Arc<Semaphore>,
    /// Module runtime instance (placeholder for actual WASM runtime)
    pub runtime_instance: Arc<Mutex<Option<WasmRuntimeInstance>>>,
    /// Security violation count
    pub violation_count: Arc<AtomicU32>,
    /// Module quarantine status
    pub is_quarantined: Arc<Mutex<bool>>,
    /// Module performance metrics
    pub performance_metrics: Arc<Mutex<PerformanceMetrics>>,
}

/// WASM runtime instance (placeholder for actual runtime integration)
#[derive(Debug, Clone)]
pub struct WasmRuntimeInstance {
    /// Runtime instance ID
    pub instance_id: String,
    /// Runtime type (e.g., "wasmtime", "wasmer", "wasm3")
    pub runtime_type: String,
    /// Runtime version
    pub runtime_version: String,
    /// Instance memory usage
    pub memory_usage_bytes: usize,
    /// Instance CPU usage
    pub cpu_usage_percent: f32,
}

impl WasmModule {
    /// Create a new WASM module instance
    pub fn new(config: WasmModuleConfig, bytecode: Vec<u8>) -> Result<Self> {
        let id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        let size_bytes = bytecode.len() as u64;
        
        // Calculate module hash for integrity verification
        let hash = Some(calculate_module_hash(&bytecode));
        
        let info = WasmModuleInfoInner {
            id: id.clone(),
            name: config.name.clone(),
            version: config.version.clone(),
            description: config.description.clone(),
            path: config.path.clone(),
            status: Arc::new(RwLock::new(WasmModuleStatus::Loaded)),
            created_at: now,
            updated_at: Arc::new(RwLock::new(now)),
            size_bytes,
            hash,
            author: config.author.clone(),
            license: config.license.clone(),
            dependencies: config.dependencies.clone(),
            stats: Arc::new(RwLock::new(WasmModuleStats::default())),
            security_context: config.security_context.clone(),
            last_accessed: Arc::new(RwLock::new(Instant::now())),
        };

        // Create execution semaphore based on security context
        let max_concurrent = config.security_context.cpu_limits.max_threads;
        let execution_semaphore = Arc::new(Semaphore::new(max_concurrent));

        Ok(Self {
            info,
            config,
            bytecode,
            ref_count: Arc::new(Mutex::new(0)),
            execution_semaphore,
            runtime_instance: Arc::new(Mutex::new(None)),
            violation_count: Arc::new(AtomicU32::new(0)),
            is_quarantined: Arc::new(Mutex::new(false)),
            performance_metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
        })
    }

    /// Update module status
    pub async fn update_status(&self, status: WasmModuleStatus) {
        {
            let mut current_status = self.info.status.write().await;
            *current_status = status;
        }
        
        // Update timestamp
        {
            let mut updated_at = self.info.updated_at.write().await;
            *updated_at = chrono::Utc::now();
        }
        
        // Update last access time
        {
            let mut last_accessed = self.info.last_accessed.write().await;
            *last_accessed = Instant::now();
        }
    }

    /// Update execution statistics
    pub async fn update_stats(&self, execution_time_ms: u64, success: bool, memory_usage: Option<usize>, cpu_time_ms: Option<u64>) {
        let mut stats = self.info.stats.write().await;
        
        // Update execution counts
        stats.total_executions += 1;
        if success {
            stats.successful_executions += 1;
        } else {
            stats.failed_executions += 1;
        }
        
        // Update execution time statistics
        if stats.total_executions == 1 {
            stats.avg_execution_time_ms = execution_time_ms as f64;
            stats.max_execution_time_ms = execution_time_ms;
            stats.min_execution_time_ms = execution_time_ms;
        } else {
            // Calculate running average
            let total_time = stats.avg_execution_time_ms * (stats.total_executions - 1) as f64;
            stats.avg_execution_time_ms = (total_time + execution_time_ms as f64) / stats.total_executions as f64;
            
            // Update min/max
            stats.max_execution_time_ms = stats.max_execution_time_ms.max(execution_time_ms);
            stats.min_execution_time_ms = stats.min_execution_time_ms.min(execution_time_ms);
        }
        
        // Update memory usage
        if let Some(memory) = memory_usage {
            stats.total_memory_usage_bytes += memory as u64;
            stats.peak_memory_usage_bytes = stats.peak_memory_usage_bytes.max(memory as u64);
        }
        
        // Update CPU time
        if let Some(cpu_time) = cpu_time_ms {
            stats.total_cpu_time_ms += cpu_time;
        }
        
        // Update last execution time
        stats.last_execution = Some(chrono::Utc::now());
        
        // Update last access time
        {
            let mut last_accessed = self.info.last_accessed.write().await;
            *last_accessed = Instant::now();
        }
    }

    /// Increment reference count
    pub async fn increment_ref_count(&self) -> u32 {
        let mut count = self.ref_count.lock().await;
        *count += 1;
        *count
    }

    /// Decrement reference count
    pub async fn decrement_ref_count(&self) -> u32 {
        let mut count = self.ref_count.lock().await;
        if *count > 0 {
            *count -= 1;
        }
        *count
    }

    /// Get current reference count
    pub async fn get_ref_count(&self) -> u32 {
        *self.ref_count.lock().await
    }

    /// Check if module is quarantined
    pub async fn is_quarantined(&self) -> bool {
        *self.is_quarantined.lock().await
    }

    /// Quarantine module due to security violations
    pub async fn quarantine(&self, reason: String) {
        let mut quarantined = self.is_quarantined.lock().await;
        *quarantined = true;
        self.update_status(WasmModuleStatus::Error(format!("Quarantined: {}", reason))).await;
        warn!("Module {} quarantined: {}", self.info.name, reason);
    }

    /// Remove module from quarantine
    pub async fn unquarantine(&self) {
        let mut quarantined = self.is_quarantined.lock().await;
        *quarantined = false;
        self.violation_count.store(0, Ordering::Relaxed);
        self.update_status(WasmModuleStatus::Initialized).await;
        info!("Module {} removed from quarantine", self.info.name);
    }

    /// Record security violation
    pub async fn record_violation(&self, violation: SecurityViolation) {
        let count = self.violation_count.fetch_add(1, Ordering::Relaxed);
        warn!("Security violation in module {}: {} (count: {})", 
              self.info.name, violation.description, count + 1);
        
        // Check if module should be quarantined
        if count + 1 >= 3 { // Default threshold, should be configurable
            self.quarantine(format!("Too many security violations: {}", count + 1)).await;
        }
    }

    /// Get module health status
    pub async fn get_health_status(&self) -> ModuleHealthStatus {
        let ref_count = self.get_ref_count().await;
        let is_quarantined = self.is_quarantined().await;
        let violation_count = self.violation_count.load(Ordering::Relaxed);
        
        if is_quarantined {
            ModuleHealthStatus::Quarantined
        } else if violation_count > 0 {
            ModuleHealthStatus::Degraded
        } else if ref_count > 0 {
            ModuleHealthStatus::Active
        } else {
            ModuleHealthStatus::Idle
        }
    }

    /// Validate module integrity
    pub fn validate_integrity(&self) -> Result<()> {
        if let Some(expected_hash) = &self.info.hash {
            let current_hash = calculate_module_hash(&self.bytecode);
            if current_hash != *expected_hash {
                return Err(anyhow!("Module integrity check failed: hash mismatch"));
            }
        }
        Ok(())
    }

    /// Get a snapshot of module information for external access
    pub async fn get_info_snapshot(&self) -> WasmModuleInfo {
        WasmModuleInfo {
            id: self.info.id.clone(),
            name: self.info.name.clone(),
            version: self.info.version.clone(),
            description: self.info.description.clone(),
            path: self.info.path.clone(),
            status: self.info.status.read().await.clone(),
            created_at: self.info.created_at,
            updated_at: *self.info.updated_at.read().await,
            size_bytes: self.info.size_bytes,
            hash: self.info.hash.clone(),
            author: self.info.author.clone(),
            license: self.info.license.clone(),
            dependencies: self.info.dependencies.clone(),
            stats: self.info.stats.read().await.clone(),
            security_context: self.info.security_context.clone(),
        }
    }

    /// Get last access time for cleanup decisions
    pub async fn get_last_accessed(&self) -> Instant {
        *self.info.last_accessed.read().await
    }
}

/// Manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerStats {
    /// Total number of loaded modules
    pub total_modules: usize,
    /// Number of currently active executions
    pub active_executions: usize,
    /// Total memory usage in bytes
    pub total_memory_bytes: usize,
    /// Total security violations
    pub total_violations: u64,
    /// Number of quarantined modules
    pub quarantined_modules: usize,
    /// Cache hit count
    pub cache_hits: u64,
    /// Cache miss count
    pub cache_misses: u64,
}

/// Module health status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModuleHealthStatus {
    /// Module is healthy and active
    Active,
    /// Module is idle (no active executions)
    Idle,
    /// Module has some issues but still functional
    Degraded,
    /// Module is quarantined due to security issues
    Quarantined,
    /// Module is in error state
    Error,
}

impl fmt::Display for ModuleHealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModuleHealthStatus::Active => write!(f, "Active"),
            ModuleHealthStatus::Idle => write!(f, "Idle"),
            ModuleHealthStatus::Degraded => write!(f, "Degraded"),
            ModuleHealthStatus::Quarantined => write!(f, "Quarantined"),
            ModuleHealthStatus::Error => write!(f, "Error"),
        }
    }
}

/// WASM manager for handling module lifecycle and execution
pub struct WasmManager {
    /// Manager configuration
    config: WasmManagerConfig,
    /// Module registry
    modules: Arc<RwLock<HashMap<String, Arc<WasmModule>>>>,
    /// WASM executor
    executor: Arc<WasmExecutor>,
    /// Cleanup task handle
    cleanup_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Performance monitoring task handle
    monitoring_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Global execution semaphore for concurrency control
    global_execution_semaphore: Arc<Semaphore>,
    /// Resource usage tracking
    resource_usage: Arc<ResourceUsageTracker>,
    /// Security monitor
    security_monitor: Arc<SecurityMonitor>,
    /// Module cache
    module_cache: Arc<ModuleCache>,
}

/// Resource usage tracker
#[derive(Debug)]
pub struct ResourceUsageTracker {
    /// Total memory usage in bytes
    pub total_memory_bytes: AtomicUsize,
    /// Total CPU usage percentage
    pub total_cpu_percent: AtomicUsize,
    /// Number of active executions
    pub active_executions: AtomicUsize,
    /// Number of file descriptors in use
    pub file_descriptors_in_use: AtomicUsize,
    /// Number of network connections
    pub network_connections: AtomicUsize,
}

impl Default for ResourceUsageTracker {
    fn default() -> Self {
        Self {
            total_memory_bytes: AtomicUsize::new(0),
            total_cpu_percent: AtomicUsize::new(0),
            active_executions: AtomicUsize::new(0),
            file_descriptors_in_use: AtomicUsize::new(0),
            network_connections: AtomicUsize::new(0),
        }
    }
}

/// Security monitor for tracking violations and enforcing policies
#[derive(Debug)]
pub struct SecurityMonitor {
    /// Total security violations across all modules
    pub total_violations: AtomicU64,
    /// Quarantined modules count
    pub quarantined_modules: AtomicUsize,
    /// Security policy violations
    pub policy_violations: AtomicU64,
}

impl Default for SecurityMonitor {
    fn default() -> Self {
        Self {
            total_violations: AtomicU64::new(0),
            quarantined_modules: AtomicUsize::new(0),
            policy_violations: AtomicU64::new(0),
        }
    }
}

/// Module cache for storing compiled modules
#[derive(Debug)]
pub struct ModuleCache {
    /// Cache storage
    cache: Arc<RwLock<HashMap<String, CachedModule>>>,
    /// Cache size in bytes
    cache_size: AtomicUsize,
    /// Cache hit count
    cache_hits: AtomicU64,
    /// Cache miss count
    cache_misses: AtomicU64,
}

/// Cached module information
#[derive(Debug, Clone)]
pub struct CachedModule {
    /// Module bytecode
    pub bytecode: Vec<u8>,
    /// Module metadata
    pub metadata: WasmModuleInfo,
    /// Cache timestamp
    pub cached_at: chrono::DateTime<chrono::Utc>,
    /// Access count
    pub access_count: u64,
    /// Last access time
    pub last_accessed: chrono::DateTime<chrono::Utc>,
}

impl Default for ModuleCache {
    fn default() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_size: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }
}

impl WasmManager {
    /// Create a new WASM manager
    pub fn new(config: WasmManagerConfig) -> Self {
        let executor = Arc::new(WasmExecutor::new(config.clone()));
        let global_execution_semaphore = Arc::new(Semaphore::new(config.max_concurrent_modules));
        
        Self {
            config,
            modules: Arc::new(RwLock::new(HashMap::new())),
            executor,
            cleanup_handle: Arc::new(Mutex::new(None)),
            monitoring_handle: Arc::new(Mutex::new(None)),
            global_execution_semaphore,
            resource_usage: Arc::new(ResourceUsageTracker::default()),
            security_monitor: Arc::new(SecurityMonitor::default()),
            module_cache: Arc::new(ModuleCache::default()),
        }
    }

    /// Start the WASM manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting WASM manager");
        
        // Start cleanup task
        if self.config.cleanup_interval_secs > 0 {
            let cleanup_handle = self.start_cleanup_task().await?;
            let mut handle = self.cleanup_handle.lock().await;
            *handle = Some(cleanup_handle);
        }

        // Start performance monitoring task
        if self.config.enable_performance_monitoring {
            let monitoring_handle = self.start_monitoring_task().await?;
            let mut handle = self.monitoring_handle.lock().await;
            *handle = Some(monitoring_handle);
        }

        info!("WASM manager started successfully");
        Ok(())
    }

    /// Stop the WASM manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping WASM manager");
        
        // Stop cleanup task
        {
            let mut handle = self.cleanup_handle.lock().await;
            if let Some(handle) = handle.take() {
                handle.abort();
            }
        }

        // Stop monitoring task
        {
            let mut handle = self.monitoring_handle.lock().await;
            if let Some(handle) = handle.take() {
                handle.abort();
            }
        }

        // Stop all modules
        let mut modules = self.modules.write().await;
        for (_, module) in modules.iter() {
            module.update_status(WasmModuleStatus::Stopped).await;
        }
        modules.clear();

        // Clear module cache
        {
            let mut cache = self.module_cache.cache.write().await;
            cache.clear();
        }

        info!("WASM manager stopped successfully");
        Ok(())
    }

    /// Load a WASM module
    pub async fn load_module(&self, config: WasmModuleConfig) -> Result<String> {
        info!("Loading WASM module: {}", config.name);

        // Check if module already exists
        {
            let modules = self.modules.read().await;
            if modules.contains_key(&config.name) {
                return Err(anyhow!("Module {} already exists", config.name));
            }
        }

        // Check resource limits
        self.check_resource_limits(&config).await?;

        // Read module bytecode
        let bytecode = tokio::fs::read(&config.path).await
            .map_err(|e| anyhow!("Failed to read module file {}: {}", config.path, e))?;

        // Validate WASM bytecode
        self.validate_wasm_bytecode(&bytecode, &config).await?;

        // Create module instance
        let module = WasmModule::new(config.clone(), bytecode)?;
        let module_id = module.info.id.clone();
        let module_arc = Arc::new(module);

        // Register module
        {
            let mut modules = self.modules.write().await;
            modules.insert(config.name.clone(), module_arc);
        }

        // Update resource usage
        self.resource_usage.total_memory_bytes.fetch_add(
            config.path.len(), Ordering::Relaxed
        );

        info!("WASM module {} loaded successfully, ID: {}", config.name, module_id);
        Ok(module_id)
    }

    /// Check resource limits before loading module
    async fn check_resource_limits(&self, config: &WasmModuleConfig) -> Result<()> {
        // Check memory limits
        let current_memory = self.resource_usage.total_memory_bytes.load(Ordering::Relaxed);
        let max_memory = self.config.resource_limits.max_total_memory_bytes as usize;
        
        if current_memory + config.path.len() > max_memory {
            return Err(anyhow!("Memory limit exceeded: {} > {}", 
                current_memory + config.path.len(), max_memory));
        }

        // Check module count limits
        let module_count = self.modules.read().await.len();
        if module_count >= self.config.max_modules_in_memory {
            return Err(anyhow!("Module count limit exceeded: {} >= {}", 
                module_count, self.config.max_modules_in_memory));
        }

        Ok(())
    }

    /// Start performance monitoring task
    async fn start_monitoring_task(&self) -> Result<tokio::task::JoinHandle<()>> {
        let resource_usage = self.resource_usage.clone();
        let security_monitor = self.security_monitor.clone();
        let modules = self.modules.clone();
        let monitoring_interval = Duration::from_secs(10); // Monitor every 10 seconds

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(monitoring_interval);
            
            loop {
                interval.tick().await;
                
                // Update resource usage statistics
                let module_count = modules.read().await.len();
                let active_executions = resource_usage.active_executions.load(Ordering::Relaxed);
                
                debug!("Resource monitoring - Modules: {}, Active executions: {}, Memory: {} bytes", 
                    module_count, active_executions, 
                    resource_usage.total_memory_bytes.load(Ordering::Relaxed));
                
                // Check for resource violations
                if active_executions > 100 { // Configurable threshold
                    warn!("High number of active executions: {}", active_executions);
                }
                
                // Log security statistics
                let total_violations = security_monitor.total_violations.load(Ordering::Relaxed);
                let quarantined_modules = security_monitor.quarantined_modules.load(Ordering::Relaxed);
                
                if total_violations > 0 || quarantined_modules > 0 {
                    info!("Security status - Violations: {}, Quarantined modules: {}", 
                        total_violations, quarantined_modules);
                }
            }
        });

        Ok(handle)
    }

    /// Unload a WASM module
    pub async fn unload_module(&self, module_name: &str) -> Result<()> {
        info!("Unloading WASM module: {}", module_name);

        // First, get the module reference without holding the write lock
        let module = {
            let modules = self.modules.read().await;
            modules.get(module_name).cloned()
        };

        if let Some(module) = module {
            // Wait for all references to be released with a timeout
            let timeout = Duration::from_secs(30);
            let start = Instant::now();
            
            while module.get_ref_count().await > 0 {
                if start.elapsed() > timeout {
                    warn!("Module {} still has {} references after timeout, force unloading", 
                          module_name, module.get_ref_count().await);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }

            // Update module status before removing
            module.update_status(WasmModuleStatus::Unloading).await;
            
            // Now safely remove from registry
            let mut modules = self.modules.write().await;
            if let Some(removed_module) = modules.remove(module_name) {
                // Update resource usage
                self.resource_usage.total_memory_bytes.fetch_sub(
                    removed_module.bytecode.len(), Ordering::Relaxed
                );
                
                info!("WASM module {} unloaded successfully", module_name);
                Ok(())
            } else {
                Err(anyhow!("Module {} was removed by another operation", module_name))
            }
        } else {
            Err(anyhow!("Module {} does not exist", module_name))
        }
    }

    /// Execute a WASM module function
    pub async fn execute_module(
        &self,
        module_name: &str,
        function_name: &str,
        input_data: Option<serde_json::Value>,
    ) -> Result<WasmExecutionResult> {
        debug!("Executing WASM module: {} function: {}", module_name, function_name);

        // Acquire global execution permit
        let _global_permit = self.global_execution_semaphore.acquire().await
            .map_err(|e| anyhow!("Failed to acquire execution permit: {}", e))?;

        // Get module with proper error handling
        let module = {
            let modules = self.modules.read().await;
            modules.get(module_name)
                .ok_or_else(|| anyhow!("Module {} does not exist", module_name))?
                .clone()
        };

        // Check if module is quarantined
        if module.is_quarantined().await {
            return Err(anyhow!("Module {} is quarantined and cannot be executed", module_name));
        }

        // Create execution guard to ensure proper cleanup
        let _execution_guard = ExecutionGuard::new(module.clone(), self.resource_usage.clone());

        // Acquire module execution permit
        let _module_permit = module.execution_semaphore.acquire().await
            .map_err(|e| anyhow!("Failed to acquire module execution permit: {}", e))?;

        // Execute module with proper error handling
        let start_time = Instant::now();
        let result = self.executor.execute(&module, function_name, input_data).await;
        let execution_time = start_time.elapsed().as_millis() as u64;

        // Update statistics
        let success = result.success;
        module.update_stats(execution_time, success, result.memory_usage_bytes, result.cpu_time_ms).await;

        debug!("WASM module {} execution completed, time: {}ms", module_name, execution_time);
        Ok(result)
    }

    /// Get module information
    pub async fn get_module_info(&self, module_name: &str) -> Result<WasmModuleInfo> {
        let modules = self.modules.read().await;
        let module = modules.get(module_name)
            .ok_or_else(|| anyhow!("Module {} does not exist", module_name))?;
        
        Ok(module.get_info_snapshot().await)
    }

    /// List all modules
    pub async fn list_modules(&self) -> Vec<WasmModuleInfo> {
        let modules = self.modules.read().await;
        let mut module_infos = Vec::new();
        
        for module in modules.values() {
            module_infos.push(module.get_info_snapshot().await);
        }
        
        module_infos
    }

    /// Get module health status
    pub async fn get_module_health(&self, module_name: &str) -> Result<ModuleHealthStatus> {
        let modules = self.modules.read().await;
        let module = modules.get(module_name)
            .ok_or_else(|| anyhow!("Module {} does not exist", module_name))?;
        
        Ok(module.get_health_status().await)
    }

    /// Get manager statistics
    pub async fn get_manager_stats(&self) -> ManagerStats {
        let modules = self.modules.read().await;
        let module_count = modules.len();
        let active_executions = self.resource_usage.active_executions.load(Ordering::Relaxed);
        let total_memory = self.resource_usage.total_memory_bytes.load(Ordering::Relaxed);
        let total_violations = self.security_monitor.total_violations.load(Ordering::Relaxed);
        let quarantined_modules = self.security_monitor.quarantined_modules.load(Ordering::Relaxed);

        ManagerStats {
            total_modules: module_count,
            active_executions,
            total_memory_bytes: total_memory,
            total_violations,
            quarantined_modules,
            cache_hits: self.module_cache.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.module_cache.cache_misses.load(Ordering::Relaxed),
        }
    }

    /// Quarantine a module
    pub async fn quarantine_module(&self, module_name: &str, reason: String) -> Result<()> {
        let modules = self.modules.read().await;
        let module = modules.get(module_name)
            .ok_or_else(|| anyhow!("Module {} does not exist", module_name))?;
        
        module.quarantine(reason).await;
        self.security_monitor.quarantined_modules.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Unquarantine a module
    pub async fn unquarantine_module(&self, module_name: &str) -> Result<()> {
        let modules = self.modules.read().await;
        let module = modules.get(module_name)
            .ok_or_else(|| anyhow!("Module {} does not exist", module_name))?;
        
        module.unquarantine().await;
        self.security_monitor.quarantined_modules.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    /// Validate WASM bytecode
    async fn validate_wasm_bytecode(&self, bytecode: &[u8], config: &WasmModuleConfig) -> Result<()> {
        // Check WASM magic number
        if bytecode.len() < 4 {
            return Err(anyhow!("WASM bytecode too short"));
        }

        let magic = &bytecode[0..4];
        if magic != b"\x00asm" {
            return Err(anyhow!("Invalid WASM magic number"));
        }

        // Check module size limits
        if bytecode.len() > self.config.validation_config.max_module_size_bytes as usize {
            return Err(anyhow!("Module size exceeds limit: {} > {}", 
                bytecode.len(), self.config.validation_config.max_module_size_bytes));
        }

        // Additional validation based on configuration
        if self.config.validation_config.validate_bytecode {
            // Here you would add more detailed WASM bytecode validation
            // For example, using wasmparser library for comprehensive validation
            self.validate_wasm_structure(bytecode)?;
        }

        // Validate dependencies if configured
        if self.config.validation_config.validate_dependencies {
            self.validate_module_dependencies(config).await?;
        }

        Ok(())
    }

    /// Validate WASM module structure
    fn validate_wasm_structure(&self, bytecode: &[u8]) -> Result<()> {
        // Basic structure validation
        // In a real implementation, you would use wasmparser or similar
        // to validate the module structure, imports, exports, etc.
        
        // For now, just check that it's not empty after magic number
        if bytecode.len() <= 8 {
            return Err(anyhow!("WASM module appears to be incomplete"));
        }

        Ok(())
    }

    /// Validate module dependencies
    async fn validate_module_dependencies(&self, config: &WasmModuleConfig) -> Result<()> {
        // Check if all dependencies are available
        let modules = self.modules.read().await;
        for dependency in &config.dependencies {
            if !modules.contains_key(dependency) {
                return Err(anyhow!("Dependency {} not found", dependency));
            }
        }
        Ok(())
    }

    /// Start cleanup task for idle modules
    async fn start_cleanup_task(&self) -> Result<tokio::task::JoinHandle<()>> {
        let modules = self.modules.clone();
        let cleanup_interval = self.config.cleanup_interval_secs;
        let max_idle_time = Duration::from_secs(cleanup_interval * 2);
        let resource_usage = self.resource_usage.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(cleanup_interval));
            
            loop {
                interval.tick().await;
                
                let now = Instant::now();
                let mut modules_to_remove = Vec::new();
                
                {
                    let modules_guard = modules.read().await;
                    for (name, module) in modules_guard.iter() {
                        // Check if module has been idle for too long
                        let last_accessed = module.get_last_accessed().await;
                        if now.duration_since(last_accessed) > max_idle_time {
                            let ref_count = module.get_ref_count().await;
                            if ref_count == 0 {
                                modules_to_remove.push(name.clone());
                            }
                        }
                    }
                }

                // Remove idle modules
                if !modules_to_remove.is_empty() {
                    let mut modules_guard = modules.write().await;
                    for name in modules_to_remove {
                        if let Some(module) = modules_guard.remove(&name) {
                            module.update_status(WasmModuleStatus::Stopped).await;
                            
                            // Update resource usage
                            resource_usage.total_memory_bytes.fetch_sub(
                                module.bytecode.len(), Ordering::Relaxed
                            );
                            
                            info!("Cleaned up idle module: {}", name);
                        }
                    }
                }
            }
        });

        Ok(handle)
    }
}

/// WASM executor for running module functions
pub struct WasmExecutor {
    config: WasmManagerConfig,
}

impl WasmExecutor {
    /// Create a new WASM executor
    pub fn new(config: WasmManagerConfig) -> Self {
        Self { config }
    }

    /// Execute a WASM module function
    pub async fn execute(
        &self,
        module: &WasmModule,
        function_name: &str,
        input_data: Option<serde_json::Value>,
    ) -> WasmExecutionResult {
        let start_time = Instant::now();
        let execution_id = Uuid::new_v4().to_string();
        
        // Update module status to running
        module.update_status(WasmModuleStatus::Running).await;

        // Validate security context
        let security_violations = self.validate_security_context(module, function_name, &input_data).await;

        // Execute the module (simulated for now)
        let result = self.simulate_execution(module, function_name, input_data).await;
        
        let execution_time = start_time.elapsed().as_millis() as u64;

        // Update module status
        let status = if result.success {
            WasmModuleStatus::Initialized
        } else {
            WasmModuleStatus::Error(result.error.clone().unwrap_or_default())
        };
        module.update_status(status).await;

        WasmExecutionResult {
            success: result.success,
            data: result.data,
            error: result.error,
            execution_time_ms: execution_time,
            memory_usage_bytes: Some(1024 * 1024), // Simulated memory usage
            cpu_time_ms: Some(execution_time / 2), // Simulated CPU time
            execution_id,
            module_id: module.info.id.clone(),
            function_name: function_name.to_string(),
            security_violations,
            performance_metrics: PerformanceMetrics::default(),
        }
    }

    /// Validate security context before execution
    async fn validate_security_context(
        &self,
        module: &WasmModule,
        function_name: &str,
        _input_data: &Option<serde_json::Value>,
    ) -> Vec<SecurityViolation> {
        let mut violations = Vec::new();
        
        // Check execution timeout
        if module.config.timeout_config.default_timeout_secs > self.config.default_module_timeout_secs {
            violations.push(SecurityViolation {
                violation_type: "timeout_exceeded".to_string(),
                description: format!("Module timeout {} exceeds manager limit {}", 
                    module.config.timeout_config.default_timeout_secs, 
                    self.config.default_module_timeout_secs),
                severity: ViolationSeverity::Medium,
                timestamp: chrono::Utc::now(),
            });
        }

        // Check memory limits
        if let Some(max_memory) = module.config.security_context.max_memory_bytes {
            if max_memory > self.config.resource_limits.max_total_memory_bytes as usize {
                violations.push(SecurityViolation {
                    violation_type: "memory_limit_exceeded".to_string(),
                    description: format!("Module memory limit {} exceeds manager limit {}", 
                        max_memory, self.config.resource_limits.max_total_memory_bytes),
                    severity: ViolationSeverity::High,
                    timestamp: chrono::Utc::now(),
                });
            }
        }

        // Check for dangerous function names
        let dangerous_functions = ["system", "exec", "eval", "shell"];
        if dangerous_functions.contains(&function_name) {
            violations.push(SecurityViolation {
                violation_type: "dangerous_function".to_string(),
                description: format!("Function '{}' is considered dangerous", function_name),
                severity: ViolationSeverity::High,
                timestamp: chrono::Utc::now(),
            });
        }

        violations
    }

    /// Simulate WASM execution (in real implementation, use wasmtime or other WASM runtime)
    async fn simulate_execution(
        &self,
        module: &WasmModule,
        function_name: &str,
        input_data: Option<serde_json::Value>,
    ) -> WasmExecutionResult {
        // Simulate execution delay
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Simulate execution result
        let success = function_name != "error_function";
        let data = if success {
            Some(serde_json::json!({
                "module": module.info.name,
                "function": function_name,
                "input": input_data,
                "result": "execution_successful"
            }))
        } else {
            None
        };

        let error = if !success {
            Some("Simulated execution error".to_string())
        } else {
            None
        };

        WasmExecutionResult {
            success,
            data,
            error,
            execution_time_ms: 0, // Will be set by caller
            memory_usage_bytes: None,
            cpu_time_ms: None,
            execution_id: String::new(), // Will be set by caller
            module_id: String::new(), // Will be set by caller
            function_name: String::new(), // Will be set by caller
            security_violations: Vec::new(), // Will be set by caller
            performance_metrics: PerformanceMetrics::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wasm_manager_creation() {
        let config = WasmManagerConfig::default();
        let manager = WasmManager::new(config);
        
        // 测试管理器创建
        assert_eq!(manager.config.max_concurrent_modules, 100);
    }

    #[tokio::test]
    async fn test_wasm_module_creation() {
        let config = WasmModuleConfig {
            name: "test_module".to_string(),
            version: "1.0.0".to_string(),
            description: Some("Test module".to_string()),
            path: "/tmp/test.wasm".to_string(),
            author: Some("Test Author".to_string()),
            license: Some("MIT".to_string()),
            dependencies: vec![],
            security_context: SecurityContext::default(),
            preload_modules: vec![],
            tags: vec!["test".to_string()],
            priority: 0,
            cache_enabled: true,
            timeout_config: TimeoutConfig::default(),
        };

        // Create simulated WASM bytecode
        let bytecode = b"\x00asm\x01\x00\x00\x00".to_vec();
        
        let module = WasmModule::new(config, bytecode).unwrap();
        assert_eq!(module.info.name, "test_module");
        let status = module.info.status.read().await;
        assert_eq!(*status, WasmModuleStatus::Loaded);
    }

    #[tokio::test]
    async fn test_wasm_module_stats() {
        let config = WasmModuleConfig {
            name: "test_module".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            path: "/tmp/test.wasm".to_string(),
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

        let bytecode = b"\x00asm\x01\x00\x00\x00".to_vec();
        let module = WasmModule::new(config, bytecode).unwrap();

        // Test statistics update
        module.update_stats(100, true, Some(1024), Some(50)).await;
        let stats = module.info.stats.read().await;
        assert_eq!(stats.total_executions, 1);
        assert_eq!(stats.successful_executions, 1);
        assert_eq!(stats.avg_execution_time_ms, 100.0);
        assert_eq!(stats.total_memory_usage_bytes, 1024);
        assert_eq!(stats.total_cpu_time_ms, 50);
        drop(stats);

        module.update_stats(200, false, Some(2048), Some(100)).await;
        let stats = module.info.stats.read().await;
        assert_eq!(stats.total_executions, 2);
        assert_eq!(stats.successful_executions, 1);
        assert_eq!(stats.failed_executions, 1);
        assert_eq!(stats.avg_execution_time_ms, 150.0);
        assert_eq!(stats.total_memory_usage_bytes, 3072);
        assert_eq!(stats.total_cpu_time_ms, 150);
    }

    #[tokio::test]
    async fn test_wasm_module_status_update() {
        let config = WasmModuleConfig {
            name: "test_module".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            path: "/tmp/test.wasm".to_string(),
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

        let bytecode = b"\x00asm\x01\x00\x00\x00".to_vec();
        let module = WasmModule::new(config, bytecode).unwrap();

        // Test status update
        module.update_status(WasmModuleStatus::Running).await;
        let status = module.info.status.read().await;
        assert_eq!(*status, WasmModuleStatus::Running);
    }

    #[tokio::test]
    async fn test_wasm_module_info_snapshot() {
        let config = WasmModuleConfig {
            name: "test_module".to_string(),
            version: "1.0.0".to_string(),
            description: Some("Test module".to_string()),
            path: "/tmp/test.wasm".to_string(),
            author: Some("Test Author".to_string()),
            license: Some("MIT".to_string()),
            dependencies: vec![],
            security_context: SecurityContext::default(),
            preload_modules: vec![],
            tags: vec!["test".to_string()],
            priority: 1,
            cache_enabled: true,
            timeout_config: TimeoutConfig::default(),
        };

        let bytecode = b"\x00asm\x01\x00\x00\x00".to_vec();
        let module = WasmModule::new(config, bytecode).unwrap();

        // Test info snapshot
        let info_snapshot = module.get_info_snapshot().await;
        assert_eq!(info_snapshot.name, "test_module");
        assert_eq!(info_snapshot.version, "1.0.0");
        assert_eq!(info_snapshot.description, Some("Test module".to_string()));
        assert_eq!(info_snapshot.author, Some("Test Author".to_string()));
        assert_eq!(info_snapshot.license, Some("MIT".to_string()));
    }
}

/// Execution guard to ensure proper resource cleanup
pub struct ExecutionGuard {
    module: Arc<WasmModule>,
    resource_usage: Arc<ResourceUsageTracker>,
}

impl ExecutionGuard {
    fn new(module: Arc<WasmModule>, resource_usage: Arc<ResourceUsageTracker>) -> Self {
        // Increment reference count
        let module_clone = module.clone();
        tokio::spawn(async move {
            module_clone.increment_ref_count().await;
        });
        
        // Update resource usage
        resource_usage.active_executions.fetch_add(1, Ordering::Relaxed);
        
        Self { module, resource_usage }
    }
}

impl Drop for ExecutionGuard {
    fn drop(&mut self) {
        // Decrement reference count
        let module = self.module.clone();
        tokio::spawn(async move {
            module.decrement_ref_count().await;
        });
        
        // Update resource usage
        self.resource_usage.active_executions.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Calculate SHA256 hash of module bytecode for integrity verification
fn calculate_module_hash(bytecode: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytecode);
    format!("{:x}", hasher.finalize())
}
