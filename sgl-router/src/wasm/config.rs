//! WASM 配置模块
//!
//! 提供 WASM 模块的配置管理功能，包括配置验证、默认值设置和配置合并。

use crate::wasm::errors::{WasmError, ErrorSeverity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;

/// WASM 运行时配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRuntimeConfig {
    /// 运行时类型
    pub runtime_type: WasmRuntimeType,
    /// 最大内存页面数
    pub max_memory_pages: u32,
    /// 最大表大小
    pub max_table_size: u32,
    /// 是否启用多线程
    pub enable_multi_threading: bool,
    /// 是否启用 SIMD
    pub enable_simd: bool,
    /// 是否启用引用类型
    pub enable_reference_types: bool,
    /// 是否启用批量内存操作
    pub enable_bulk_memory: bool,
    /// 是否启用尾调用
    pub enable_tail_call: bool,
    /// 是否启用组件模型
    pub enable_component_model: bool,
}

/// WASM 运行时类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WasmRuntimeType {
    /// Wasmtime 运行时
    Wasmtime,
    /// Wasmer 运行时
    Wasmer,
    /// WasmEdge 运行时
    WasmEdge,
    /// 自定义运行时
    Custom(String),
}

impl Default for WasmRuntimeConfig {
    fn default() -> Self {
        Self {
            runtime_type: WasmRuntimeType::Wasmtime,
            max_memory_pages: 65536, // 4GB
            max_table_size: 10000000,
            enable_multi_threading: false,
            enable_simd: true,
            enable_reference_types: true,
            enable_bulk_memory: true,
            enable_tail_call: false,
            enable_component_model: false,
        }
    }
}

/// WASM 安全配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmSecurityConfig {
    /// 是否启用沙箱模式
    pub enable_sandbox: bool,
    /// 允许的系统调用列表
    pub allowed_syscalls: Vec<String>,
    /// 禁止的系统调用列表
    pub forbidden_syscalls: Vec<String>,
    /// 允许的文件系统路径
    pub allowed_filesystem_paths: Vec<String>,
    /// 禁止的文件系统路径
    pub forbidden_filesystem_paths: Vec<String>,
    /// 允许的网络地址
    pub allowed_network_addresses: Vec<String>,
    /// 禁止的网络地址
    pub forbidden_network_addresses: Vec<String>,
    /// 最大文件大小（字节）
    pub max_file_size: Option<usize>,
    /// 最大网络连接数
    pub max_network_connections: Option<usize>,
    /// 是否启用内存保护
    pub enable_memory_protection: bool,
    /// 是否启用栈保护
    pub enable_stack_protection: bool,
}

impl Default for WasmSecurityConfig {
    fn default() -> Self {
        Self {
            enable_sandbox: true,
            allowed_syscalls: vec![
                "exit".to_string(),
                "abort".to_string(),
                "clock_time_get".to_string(),
                "random_get".to_string(),
            ],
            forbidden_syscalls: vec![
                "proc_exit".to_string(),
                "fd_prestat_get".to_string(),
                "fd_prestat_dir_name".to_string(),
                "fd_close".to_string(),
                "fd_read".to_string(),
                "fd_write".to_string(),
                "fd_seek".to_string(),
                "fd_tell".to_string(),
                "fd_fdstat_get".to_string(),
                "fd_fdstat_set_flags".to_string(),
                "fd_fdstat_set_rights".to_string(),
                "fd_sync".to_string(),
                "fd_datasync".to_string(),
                "fd_readdir".to_string(),
                "fd_advise".to_string(),
                "fd_allocate".to_string(),
                "path_create_directory".to_string(),
                "path_link".to_string(),
                "path_open".to_string(),
                "path_readlink".to_string(),
                "path_remove_directory".to_string(),
                "path_rename".to_string(),
                "path_filestat_get".to_string(),
                "path_filestat_set_times".to_string(),
                "path_symlink".to_string(),
                "path_unlink_file".to_string(),
                "poll_oneoff".to_string(),
                "proc_raise".to_string(),
                "sched_yield".to_string(),
                "random_get".to_string(),
                "sock_accept".to_string(),
                "sock_recv".to_string(),
                "sock_send".to_string(),
                "sock_shutdown".to_string(),
            ],
            allowed_filesystem_paths: vec!["/tmp".to_string()],
            forbidden_filesystem_paths: vec!["/etc".to_string(), "/usr".to_string(), "/bin".to_string()],
            allowed_network_addresses: vec![],
            forbidden_network_addresses: vec!["0.0.0.0".to_string()],
            max_file_size: Some(10 * 1024 * 1024), // 10MB
            max_network_connections: Some(10),
            enable_memory_protection: true,
            enable_stack_protection: true,
        }
    }
}

/// WASM 性能配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmPerformanceConfig {
    /// 是否启用 JIT 编译
    pub enable_jit: bool,
    /// 是否启用优化
    pub enable_optimization: bool,
    /// 优化级别 (0-3)
    pub optimization_level: u8,
    /// 是否启用缓存
    pub enable_cache: bool,
    /// 缓存大小（字节）
    pub cache_size: usize,
    /// 是否启用预编译
    pub enable_precompilation: bool,
    /// 预编译线程数
    pub precompilation_threads: usize,
    /// 是否启用性能分析
    pub enable_profiling: bool,
    /// 性能分析采样率
    pub profiling_sample_rate: f64,
}

impl Default for WasmPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_jit: true,
            enable_optimization: true,
            optimization_level: 2,
            enable_cache: true,
            cache_size: 100 * 1024 * 1024, // 100MB
            enable_precompilation: false,
            precompilation_threads: 4,
            enable_profiling: false,
            profiling_sample_rate: 0.01, // 1%
        }
    }
}

/// WASM 日志配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmLoggingConfig {
    /// 日志级别
    pub log_level: WasmLogLevel,
    /// 是否启用模块日志
    pub enable_module_logging: bool,
    /// 是否启用执行日志
    pub enable_execution_logging: bool,
    /// 是否启用性能日志
    pub enable_performance_logging: bool,
    /// 日志文件路径
    pub log_file_path: Option<String>,
    /// 日志文件最大大小（字节）
    pub max_log_file_size: Option<usize>,
    /// 日志文件保留数量
    pub max_log_files: Option<usize>,
    /// 是否启用结构化日志
    pub enable_structured_logging: bool,
}

/// WASM 日志级别
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WasmLogLevel {
    /// 错误级别
    Error,
    /// 警告级别
    Warn,
    /// 信息级别
    Info,
    /// 调试级别
    Debug,
    /// 跟踪级别
    Trace,
}

impl Default for WasmLoggingConfig {
    fn default() -> Self {
        Self {
            log_level: WasmLogLevel::Info,
            enable_module_logging: true,
            enable_execution_logging: true,
            enable_performance_logging: false,
            log_file_path: None,
            max_log_file_size: Some(100 * 1024 * 1024), // 100MB
            max_log_files: Some(10),
            enable_structured_logging: true,
        }
    }
}

/// WASM 监控配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmMonitoringConfig {
    /// 是否启用监控
    pub enable_monitoring: bool,
    /// 监控端口
    pub monitoring_port: Option<u16>,
    /// 监控主机
    pub monitoring_host: Option<String>,
    /// 是否启用指标收集
    pub enable_metrics: bool,
    /// 是否启用健康检查
    pub enable_health_check: bool,
    /// 健康检查间隔（秒）
    pub health_check_interval: u64,
    /// 是否启用性能指标
    pub enable_performance_metrics: bool,
    /// 是否启用资源使用监控
    pub enable_resource_monitoring: bool,
}

impl Default for WasmMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            monitoring_port: Some(9090),
            monitoring_host: Some("127.0.0.1".to_string()),
            enable_metrics: true,
            enable_health_check: true,
            health_check_interval: 30,
            enable_performance_metrics: true,
            enable_resource_monitoring: true,
        }
    }
}

/// 完整的 WASM 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfig {
    /// 运行时配置
    pub runtime: WasmRuntimeConfig,
    /// 安全配置
    pub security: WasmSecurityConfig,
    /// 性能配置
    pub performance: WasmPerformanceConfig,
    /// 日志配置
    pub logging: WasmLoggingConfig,
    /// 监控配置
    pub monitoring: WasmMonitoringConfig,
    /// 全局设置
    pub global: WasmGlobalConfig,
}

/// WASM 全局配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmGlobalConfig {
    /// 最大并发模块数
    pub max_concurrent_modules: usize,
    /// 模块超时时间（秒）
    pub module_timeout_secs: u64,
    /// 模块清理间隔（秒）
    pub cleanup_interval_secs: u64,
    /// 是否启用模块缓存
    pub enable_module_cache: bool,
    /// 缓存大小限制（字节）
    pub cache_size_limit_bytes: usize,
    /// 环境变量
    pub env_vars: HashMap<String, String>,
    /// 预加载模块列表
    pub preload_modules: Vec<String>,
    /// 模块搜索路径
    pub module_search_paths: Vec<String>,
}

impl Default for WasmGlobalConfig {
    fn default() -> Self {
        Self {
            max_concurrent_modules: 100,
            module_timeout_secs: 30,
            cleanup_interval_secs: 300,
            enable_module_cache: true,
            cache_size_limit_bytes: 100 * 1024 * 1024, // 100MB
            env_vars: HashMap::new(),
            preload_modules: vec![],
            module_search_paths: vec!["./modules".to_string(), "/usr/local/lib/wasm".to_string()],
        }
    }
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            runtime: WasmRuntimeConfig::default(),
            security: WasmSecurityConfig::default(),
            performance: WasmPerformanceConfig::default(),
            logging: WasmLoggingConfig::default(),
            monitoring: WasmMonitoringConfig::default(),
            global: WasmGlobalConfig::default(),
        }
    }
}

impl WasmConfig {
    /// 从文件加载配置
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: WasmConfig = toml::from_str(&content)
            .map_err(|e| WasmError::config_error(
                format!("无法解析配置文件: {}", e),
                Some("toml_parse".to_string())
            ))?;
        
        config.validate()?;
        Ok(config)
    }

    /// 保存配置到文件
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| WasmError::serialization_error(
                format!("无法序列化配置: {}", e),
                Some("toml_serialize".to_string()),
                Some(Box::new(e))
            ))?;
        
        std::fs::write(path, content)
            .map_err(|e| WasmError::file_system_error(
                "无法写入配置文件".to_string(),
                None,
                Some(Box::new(e))
            ))?;
        
        Ok(())
    }

    /// 验证配置
    pub fn validate(&self) -> Result<()> {
        // 验证全局配置
        if self.global.max_concurrent_modules == 0 {
            return Err(WasmError::config_error(
                "max_concurrent_modules 必须大于 0".to_string(),
                Some("max_concurrent_modules".to_string())
            ).into());
        }

        if self.global.module_timeout_secs == 0 {
            return Err(WasmError::config_error(
                "module_timeout_secs 必须大于 0".to_string(),
                Some("module_timeout_secs".to_string())
            ).into());
        }

        // 验证运行时配置
        if self.runtime.max_memory_pages == 0 {
            return Err(WasmError::config_error(
                "max_memory_pages 必须大于 0".to_string(),
                Some("max_memory_pages".to_string())
            ).into());
        }

        // 验证性能配置
        if self.performance.optimization_level > 3 {
            return Err(WasmError::config_error(
                "optimization_level 必须在 0-3 之间".to_string(),
                Some("optimization_level".to_string())
            ).into());
        }

        if self.performance.profiling_sample_rate < 0.0 || self.performance.profiling_sample_rate > 1.0 {
            return Err(WasmError::config_error(
                "profiling_sample_rate 必须在 0.0-1.0 之间".to_string(),
                Some("profiling_sample_rate".to_string())
            ).into());
        }

        // 验证监控配置
        if let Some(port) = self.monitoring.monitoring_port {
            if port == 0 {
                return Err(WasmError::config_error(
                    "monitoring_port 不能为 0".to_string(),
                    Some("monitoring_port".to_string())
                ).into());
            }
        }

        Ok(())
    }

    /// 合并配置
    pub fn merge(&mut self, other: WasmConfig) {
        // 合并全局配置
        if other.global.max_concurrent_modules != 0 {
            self.global.max_concurrent_modules = other.global.max_concurrent_modules;
        }
        if other.global.module_timeout_secs != 0 {
            self.global.module_timeout_secs = other.global.module_timeout_secs;
        }
        if other.global.cleanup_interval_secs != 0 {
            self.global.cleanup_interval_secs = other.global.cleanup_interval_secs;
        }
        self.global.enable_module_cache = other.global.enable_module_cache;
        if other.global.cache_size_limit_bytes != 0 {
            self.global.cache_size_limit_bytes = other.global.cache_size_limit_bytes;
        }
        
        // 合并环境变量
        for (key, value) in other.global.env_vars {
            self.global.env_vars.insert(key, value);
        }
        
        // 合并预加载模块
        for module in other.global.preload_modules {
            if !self.global.preload_modules.contains(&module) {
                self.global.preload_modules.push(module);
            }
        }
        
        // 合并模块搜索路径
        for path in other.global.module_search_paths {
            if !self.global.module_search_paths.contains(&path) {
                self.global.module_search_paths.push(path);
            }
        }

        // 合并运行时配置
        self.runtime = other.runtime;
        
        // 合并安全配置
        self.security = other.security;
        
        // 合并性能配置
        self.performance = other.performance;
        
        // 合并日志配置
        self.logging = other.logging;
        
        // 合并监控配置
        self.monitoring = other.monitoring;
    }

    /// 获取配置摘要
    pub fn summary(&self) -> WasmConfigSummary {
        WasmConfigSummary {
            runtime_type: format!("{:?}", self.runtime.runtime_type),
            max_concurrent_modules: self.global.max_concurrent_modules,
            module_timeout_secs: self.global.module_timeout_secs,
            enable_sandbox: self.security.enable_sandbox,
            enable_jit: self.performance.enable_jit,
            enable_monitoring: self.monitoring.enable_monitoring,
            log_level: format!("{:?}", self.logging.log_level),
        }
    }
}

/// WASM 配置摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfigSummary {
    /// 运行时类型
    pub runtime_type: String,
    /// 最大并发模块数
    pub max_concurrent_modules: usize,
    /// 模块超时时间
    pub module_timeout_secs: u64,
    /// 是否启用沙箱
    pub enable_sandbox: bool,
    /// 是否启用 JIT
    pub enable_jit: bool,
    /// 是否启用监控
    pub enable_monitoring: bool,
    /// 日志级别
    pub log_level: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WasmConfig::default();
        assert_eq!(config.global.max_concurrent_modules, 100);
        assert_eq!(config.runtime.max_memory_pages, 65536);
        assert!(config.security.enable_sandbox);
        assert!(config.performance.enable_jit);
    }

    #[test]
    fn test_config_validation() {
        let mut config = WasmConfig::default();
        config.global.max_concurrent_modules = 0;
        
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_merge() {
        let mut config1 = WasmConfig::default();
        let mut config2 = WasmConfig::default();
        
        config2.global.max_concurrent_modules = 200;
        config2.global.env_vars.insert("TEST_VAR".to_string(), "test_value".to_string());
        
        config1.merge(config2);
        
        assert_eq!(config1.global.max_concurrent_modules, 200);
        assert_eq!(config1.global.env_vars.get("TEST_VAR"), Some(&"test_value".to_string()));
    }

    #[test]
    fn test_config_summary() {
        let config = WasmConfig::default();
        let summary = config.summary();
        
        assert_eq!(summary.max_concurrent_modules, 100);
        assert!(summary.enable_sandbox);
        assert!(summary.enable_jit);
    }
}
