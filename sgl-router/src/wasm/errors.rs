//! WASM 相关错误类型定义
//!
//! 提供 WASM 模块管理过程中可能出现的各种错误类型和处理逻辑。

use thiserror::Error;

/// WASM 模块管理错误类型
#[derive(Error, Debug)]
pub enum WasmError {
    /// 模块加载错误
    #[error("模块加载失败: {message}")]
    ModuleLoadError {
        /// 错误消息
        message: String,
        /// 模块名称
        module_name: Option<String>,
        /// 原始错误
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// 模块执行错误
    #[error("模块执行失败: {message}")]
    ModuleExecutionError {
        /// 错误消息
        message: String,
        /// 模块名称
        module_name: String,
        /// 函数名称
        function_name: String,
        /// 原始错误
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// 模块验证错误
    #[error("模块验证失败: {message}")]
    ModuleValidationError {
        /// 错误消息
        message: String,
        /// 模块路径
        module_path: Option<String>,
    },

    /// 模块未找到错误
    #[error("模块未找到: {module_name}")]
    ModuleNotFound {
        /// 模块名称
        module_name: String,
    },

    /// 模块已存在错误
    #[error("模块已存在: {module_name}")]
    ModuleAlreadyExists {
        /// 模块名称
        module_name: String,
    },

    /// 模块状态错误
    #[error("模块状态错误: {message}")]
    ModuleStateError {
        /// 错误消息
        message: String,
        /// 模块名称
        module_name: String,
        /// 当前状态
        current_state: String,
        /// 期望状态
        expected_state: Option<String>,
    },

    /// 配置错误
    #[error("配置错误: {message}")]
    ConfigError {
        /// 错误消息
        message: String,
        /// 配置字段
        field: Option<String>,
    },

    /// 资源限制错误
    #[error("资源限制错误: {message}")]
    ResourceLimitError {
        /// 错误消息
        message: String,
        /// 资源类型
        resource_type: String,
        /// 当前使用量
        current_usage: u64,
        /// 限制值
        limit: u64,
    },

    /// 超时错误
    #[error("操作超时: {message}")]
    TimeoutError {
        /// 错误消息
        message: String,
        /// 超时时间（秒）
        timeout_secs: u64,
    },

    /// 内存错误
    #[error("内存错误: {message}")]
    MemoryError {
        /// 错误消息
        message: String,
        /// 内存使用量（字节）
        memory_usage: Option<usize>,
        /// 内存限制（字节）
        memory_limit: Option<usize>,
    },

    /// 安全错误
    #[error("安全错误: {message}")]
    SecurityError {
        /// 错误消息
        message: String,
        /// 安全违规类型
        violation_type: String,
    },

    /// 文件系统错误
    #[error("文件系统错误: {message}")]
    FileSystemError {
        /// 错误消息
        message: String,
        /// 文件路径
        file_path: Option<String>,
        /// 原始错误
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// 网络错误
    #[error("网络错误: {message}")]
    NetworkError {
        /// 错误消息
        message: String,
        /// 网络地址
        address: Option<String>,
        /// 原始错误
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// 序列化/反序列化错误
    #[error("序列化错误: {message}")]
    SerializationError {
        /// 错误消息
        message: String,
        /// 数据类型
        data_type: Option<String>,
        /// 原始错误
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// 内部错误
    #[error("内部错误: {message}")]
    InternalError {
        /// 错误消息
        message: String,
        /// 原始错误
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

impl WasmError {
    /// 创建模块加载错误
    pub fn module_load_error(
        message: impl Into<String>,
        module_name: Option<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::ModuleLoadError {
            message: message.into(),
            module_name,
            source,
        }
    }

    /// 创建模块执行错误
    pub fn module_execution_error(
        message: impl Into<String>,
        module_name: impl Into<String>,
        function_name: impl Into<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::ModuleExecutionError {
            message: message.into(),
            module_name: module_name.into(),
            function_name: function_name.into(),
            source,
        }
    }

    /// 创建模块验证错误
    pub fn module_validation_error(
        message: impl Into<String>,
        module_path: Option<String>,
    ) -> Self {
        Self::ModuleValidationError {
            message: message.into(),
            module_path,
        }
    }

    /// 创建模块未找到错误
    pub fn module_not_found(module_name: impl Into<String>) -> Self {
        Self::ModuleNotFound {
            module_name: module_name.into(),
        }
    }

    /// 创建模块已存在错误
    pub fn module_already_exists(module_name: impl Into<String>) -> Self {
        Self::ModuleAlreadyExists {
            module_name: module_name.into(),
        }
    }

    /// 创建模块状态错误
    pub fn module_state_error(
        message: impl Into<String>,
        module_name: impl Into<String>,
        current_state: impl Into<String>,
        expected_state: Option<String>,
    ) -> Self {
        Self::ModuleStateError {
            message: message.into(),
            module_name: module_name.into(),
            current_state: current_state.into(),
            expected_state,
        }
    }

    /// 创建配置错误
    pub fn config_error(message: impl Into<String>, field: Option<String>) -> Self {
        Self::ConfigError {
            message: message.into(),
            field,
        }
    }

    /// 创建资源限制错误
    pub fn resource_limit_error(
        message: impl Into<String>,
        resource_type: impl Into<String>,
        current_usage: u64,
        limit: u64,
    ) -> Self {
        Self::ResourceLimitError {
            message: message.into(),
            resource_type: resource_type.into(),
            current_usage,
            limit,
        }
    }

    /// 创建超时错误
    pub fn timeout_error(message: impl Into<String>, timeout_secs: u64) -> Self {
        Self::TimeoutError {
            message: message.into(),
            timeout_secs,
        }
    }

    /// 创建内存错误
    pub fn memory_error(
        message: impl Into<String>,
        memory_usage: Option<usize>,
        memory_limit: Option<usize>,
    ) -> Self {
        Self::MemoryError {
            message: message.into(),
            memory_usage,
            memory_limit,
        }
    }

    /// 创建安全错误
    pub fn security_error(
        message: impl Into<String>,
        violation_type: impl Into<String>,
    ) -> Self {
        Self::SecurityError {
            message: message.into(),
            violation_type: violation_type.into(),
        }
    }

    /// 创建文件系统错误
    pub fn file_system_error(
        message: impl Into<String>,
        file_path: Option<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::FileSystemError {
            message: message.into(),
            file_path,
            source,
        }
    }

    /// 创建网络错误
    pub fn network_error(
        message: impl Into<String>,
        address: Option<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::NetworkError {
            message: message.into(),
            address,
            source,
        }
    }

    /// 创建序列化错误
    pub fn serialization_error(
        message: impl Into<String>,
        data_type: Option<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::SerializationError {
            message: message.into(),
            data_type,
            source,
        }
    }

    /// 创建内部错误
    pub fn internal_error(
        message: impl Into<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::InternalError {
            message: message.into(),
            source,
        }
    }

    /// 获取错误严重程度
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            WasmError::ModuleLoadError { .. } => ErrorSeverity::High,
            WasmError::ModuleExecutionError { .. } => ErrorSeverity::High,
            WasmError::ModuleValidationError { .. } => ErrorSeverity::High,
            WasmError::ModuleNotFound { .. } => ErrorSeverity::Medium,
            WasmError::ModuleAlreadyExists { .. } => ErrorSeverity::Low,
            WasmError::ModuleStateError { .. } => ErrorSeverity::Medium,
            WasmError::ConfigError { .. } => ErrorSeverity::Medium,
            WasmError::ResourceLimitError { .. } => ErrorSeverity::High,
            WasmError::TimeoutError { .. } => ErrorSeverity::Medium,
            WasmError::MemoryError { .. } => ErrorSeverity::High,
            WasmError::SecurityError { .. } => ErrorSeverity::Critical,
            WasmError::FileSystemError { .. } => ErrorSeverity::Medium,
            WasmError::NetworkError { .. } => ErrorSeverity::Medium,
            WasmError::SerializationError { .. } => ErrorSeverity::Low,
            WasmError::InternalError { .. } => ErrorSeverity::Critical,
        }
    }

    /// 检查错误是否可恢复
    pub fn is_recoverable(&self) -> bool {
        match self {
            WasmError::ModuleLoadError { .. } => true,
            WasmError::ModuleExecutionError { .. } => true,
            WasmError::ModuleValidationError { .. } => false,
            WasmError::ModuleNotFound { .. } => true,
            WasmError::ModuleAlreadyExists { .. } => true,
            WasmError::ModuleStateError { .. } => true,
            WasmError::ConfigError { .. } => true,
            WasmError::ResourceLimitError { .. } => true,
            WasmError::TimeoutError { .. } => true,
            WasmError::MemoryError { .. } => true,
            WasmError::SecurityError { .. } => false,
            WasmError::FileSystemError { .. } => true,
            WasmError::NetworkError { .. } => true,
            WasmError::SerializationError { .. } => true,
            WasmError::InternalError { .. } => false,
        }
    }

    /// 获取错误上下文信息
    pub fn context(&self) -> ErrorContext {
        match self {
            WasmError::ModuleLoadError { module_name, .. } => ErrorContext {
                module_name: module_name.clone(),
                function_name: None,
                operation: Some("load".to_string()),
            },
            WasmError::ModuleExecutionError { module_name, function_name, .. } => ErrorContext {
                module_name: Some(module_name.clone()),
                function_name: Some(function_name.clone()),
                operation: Some("execute".to_string()),
            },
            WasmError::ModuleValidationError { module_path, .. } => ErrorContext {
                module_name: module_path.as_ref().and_then(|p| {
                    std::path::Path::new(p).file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                }),
                function_name: None,
                operation: Some("validate".to_string()),
            },
            WasmError::ModuleNotFound { module_name } => ErrorContext {
                module_name: Some(module_name.clone()),
                function_name: None,
                operation: Some("find".to_string()),
            },
            WasmError::ModuleAlreadyExists { module_name } => ErrorContext {
                module_name: Some(module_name.clone()),
                function_name: None,
                operation: Some("create".to_string()),
            },
            WasmError::ModuleStateError { module_name, .. } => ErrorContext {
                module_name: Some(module_name.clone()),
                function_name: None,
                operation: Some("state_change".to_string()),
            },
            _ => ErrorContext {
                module_name: None,
                function_name: None,
                operation: None,
            },
        }
    }
}

/// 错误严重程度
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    /// 低严重程度 - 不影响核心功能
    Low,
    /// 中等严重程度 - 影响部分功能
    Medium,
    /// 高严重程度 - 影响核心功能
    High,
    /// 严重程度 - 系统无法继续运行
    Critical,
}

/// 错误上下文信息
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// 模块名称
    pub module_name: Option<String>,
    /// 函数名称
    pub function_name: Option<String>,
    /// 操作类型
    pub operation: Option<String>,
}

/// 错误处理结果
#[derive(Debug, Clone)]
pub struct ErrorHandlingResult {
    /// 是否应该重试
    pub should_retry: bool,
    /// 重试延迟（毫秒）
    pub retry_delay_ms: Option<u64>,
    /// 是否应该记录错误
    pub should_log: bool,
    /// 是否应该通知用户
    pub should_notify: bool,
    /// 建议的恢复操作
    pub recovery_action: Option<RecoveryAction>,
}

/// 恢复操作建议
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// 重新加载模块
    ReloadModule,
    /// 重启管理器
    RestartManager,
    /// 清理资源
    CleanupResources,
    /// 检查配置
    CheckConfiguration,
    /// 联系管理员
    ContactAdministrator,
    /// 无操作
    NoAction,
}

impl WasmError {
    /// 获取错误处理建议
    pub fn get_handling_result(&self) -> ErrorHandlingResult {
        match self {
            WasmError::ModuleLoadError { .. } => ErrorHandlingResult {
                should_retry: true,
                retry_delay_ms: Some(1000),
                should_log: true,
                should_notify: true,
                recovery_action: Some(RecoveryAction::ReloadModule),
            },
            WasmError::ModuleExecutionError { .. } => ErrorHandlingResult {
                should_retry: true,
                retry_delay_ms: Some(500),
                should_log: true,
                should_notify: false,
                recovery_action: Some(RecoveryAction::NoAction),
            },
            WasmError::ModuleValidationError { .. } => ErrorHandlingResult {
                should_retry: false,
                retry_delay_ms: None,
                should_log: true,
                should_notify: true,
                recovery_action: Some(RecoveryAction::CheckConfiguration),
            },
            WasmError::ModuleNotFound { .. } => ErrorHandlingResult {
                should_retry: false,
                retry_delay_ms: None,
                should_log: true,
                should_notify: false,
                recovery_action: Some(RecoveryAction::ReloadModule),
            },
            WasmError::ModuleAlreadyExists { .. } => ErrorHandlingResult {
                should_retry: false,
                retry_delay_ms: None,
                should_log: false,
                should_notify: false,
                recovery_action: Some(RecoveryAction::NoAction),
            },
            WasmError::ResourceLimitError { .. } => ErrorHandlingResult {
                should_retry: true,
                retry_delay_ms: Some(5000),
                should_log: true,
                should_notify: true,
                recovery_action: Some(RecoveryAction::CleanupResources),
            },
            WasmError::TimeoutError { .. } => ErrorHandlingResult {
                should_retry: true,
                retry_delay_ms: Some(2000),
                should_log: true,
                should_notify: false,
                recovery_action: Some(RecoveryAction::NoAction),
            },
            WasmError::SecurityError { .. } => ErrorHandlingResult {
                should_retry: false,
                retry_delay_ms: None,
                should_log: true,
                should_notify: true,
                recovery_action: Some(RecoveryAction::ContactAdministrator),
            },
            WasmError::InternalError { .. } => ErrorHandlingResult {
                should_retry: false,
                retry_delay_ms: None,
                should_log: true,
                should_notify: true,
                recovery_action: Some(RecoveryAction::RestartManager),
            },
            _ => ErrorHandlingResult {
                should_retry: false,
                retry_delay_ms: None,
                should_log: true,
                should_notify: false,
                recovery_action: Some(RecoveryAction::NoAction),
            },
        }
    }
}

/// 错误统计信息
#[derive(Debug, Clone, Default)]
pub struct ErrorStats {
    /// 总错误数
    pub total_errors: u64,
    /// 按类型分组的错误数
    pub errors_by_type: std::collections::HashMap<String, u64>,
    /// 按严重程度分组的错误数
    pub errors_by_severity: std::collections::HashMap<ErrorSeverity, u64>,
    /// 最后错误时间
    pub last_error_time: Option<chrono::DateTime<chrono::Utc>>,
}

impl ErrorStats {
    /// 记录错误
    pub fn record_error(&mut self, error: &WasmError) {
        self.total_errors += 1;
        
        let error_type = format!("{:?}", error);
        *self.errors_by_type.entry(error_type).or_insert(0) += 1;
        
        let severity = error.severity();
        // Note: This would need ErrorSeverity to implement Hash
        // For now, we'll just increment the total count
        *self.errors_by_severity.entry(severity).or_insert(0) += 1;
        
        self.last_error_time = Some(chrono::Utc::now());
    }

    /// 获取错误率（错误数/总操作数）
    pub fn error_rate(&self, total_operations: u64) -> f64 {
        if total_operations == 0 {
            0.0
        } else {
            self.total_errors as f64 / total_operations as f64
        }
    }

    /// 重置统计信息
    pub fn reset(&mut self) {
        self.total_errors = 0;
        self.errors_by_type.clear();
        self.errors_by_severity.clear();
        self.last_error_time = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = WasmError::module_not_found("test_module");
        assert!(matches!(error, WasmError::ModuleNotFound { .. }));
        
        if let WasmError::ModuleNotFound { module_name } = error {
            assert_eq!(module_name, "test_module");
        }
    }

    #[test]
    fn test_error_severity() {
        let security_error = WasmError::security_error("test", "unauthorized_access");
        assert_eq!(security_error.severity(), ErrorSeverity::Critical);
        
        let config_error = WasmError::config_error("invalid config".to_string(), Some("timeout".to_string()));
        assert_eq!(config_error.severity(), ErrorSeverity::Medium);
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable_error = WasmError::module_load_error("failed to load", None, None);
        assert!(recoverable_error.is_recoverable());
        
        let non_recoverable_error = WasmError::security_error("test", "unauthorized_access");
        assert!(!non_recoverable_error.is_recoverable());
    }

    #[test]
    fn test_error_context() {
        let error = WasmError::module_execution_error(
            "execution failed",
            "test_module",
            "test_function",
            None,
        );
        
        let context = error.context();
        assert_eq!(context.module_name, Some("test_module".to_string()));
        assert_eq!(context.function_name, Some("test_function".to_string()));
        assert_eq!(context.operation, Some("execute".to_string()));
    }

    #[test]
    fn test_error_handling_result() {
        let error = WasmError::module_load_error("failed to load", None, None);
        let result = error.get_handling_result();
        
        assert!(result.should_retry);
        assert_eq!(result.retry_delay_ms, Some(1000));
        assert!(result.should_log);
        assert!(result.should_notify);
        assert!(matches!(result.recovery_action, Some(RecoveryAction::ReloadModule)));
    }

    #[test]
    fn test_error_stats() {
        let mut stats = ErrorStats::default();
        
        let error1 = WasmError::module_not_found("module1");
        let error2 = WasmError::module_not_found("module2");
        let error3 = WasmError::config_error("invalid", None);
        
        stats.record_error(&error1);
        stats.record_error(&error2);
        stats.record_error(&error3);
        
        assert_eq!(stats.total_errors, 3);
        assert_eq!(stats.errors_by_type.len(), 2);
        assert_eq!(stats.errors_by_severity.len(), 2);
        assert!(stats.last_error_time.is_some());
    }
}
