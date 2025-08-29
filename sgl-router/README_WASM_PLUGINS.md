# SGL-Router WASM 插件系统

## 概述

本项目实现了 SGL-Router 的动态扩展 policies 功能，使用 WebAssembly (WASM) 来支持用户自定义的负载均衡策略，无需重新编译 sgl-router。

## 功能特性

- **动态加载**: 无需重启服务即可加载新的策略插件
- **安全沙箱**: 严格的资源限制和系统调用控制
- **热重载**: 支持运行时更新插件配置
- **类型安全**: 基于 Rust 的类型安全接口
- **性能优化**: 高效的 WASM 执行环境

## 架构设计

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Policy Config │    │  WASM Registry  │    │  WASM Sandbox   │
│                 │───▶│                 │───▶│                 │
│ - Module Path   │    │ - Plugin Mgmt   │    │ - Execution     │
│ - Security      │    │ - Validation    │    │ - Resource Ctrl │
│ - Limits        │    │ - Hot Reload    │    │ - Isolation     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 核心组件

### 1. WASM 插件系统 (`src/policies/wasm/`)

- **mod.rs**: 核心模块，定义插件接口和数据结构
- **loader.rs**: WASM 模块加载器
- **validator.rs**: 插件配置和安全性验证器

### 2. 示例插件 (`examples/wasm_plugins/`)

- **weighted_random**: 加权随机选择策略
- **least_connections**: 最少连接数策略

### 3. 构建脚本 (`scripts/build_wasm_plugins.sh`)

自动化构建、测试和清理 WASM 插件的脚本。

## 使用方法

### 1. 加载插件

```rust
use sglang_router_rs::policies::{PolicyFactory, WasmPolicyConfig};
use std::collections::HashMap;

let mut factory = PolicyFactory::new();

let config = WasmPolicyConfig {
    module_path: "plugins/my_policy.wasm".to_string(),
    name: "my_policy".to_string(),
    max_execution_time_ms: 1000,
    max_memory_bytes: 1024 * 1024,
    allowed_syscalls: vec![],
    config: HashMap::new(),
};

factory.load_wasm_plugin(config).await?;
let policy = factory.create_by_name("my_policy").unwrap();
```

### 2. 管理插件

```rust
// 列出已加载的插件
let plugins = factory.list_wasm_plugins();

// 卸载插件
factory.unload_wasm_plugin("my_policy");
```

### 3. 构建插件

```bash
# 构建所有插件
./scripts/build_wasm_plugins.sh build

# 测试插件
./scripts/build_wasm_plugins.sh test

# 清理构建产物
./scripts/build_wasm_plugins.sh clean
```

## 安全限制

- **执行时间**: 最大 30 秒
- **内存使用**: 最大 100MB
- **文件系统**: 仅限 `/tmp` 目录
- **系统调用**: 默认禁用，可选择性启用
- **网络访问**: 禁用
- **插件名称**: 不能使用保留名称

## 插件开发

### 1. 创建插件项目

```bash
mkdir my_policy_plugin
cd my_policy_plugin
cargo init --lib
```

### 2. 配置 Cargo.toml

```toml
[package]
name = "my-policy-plugin"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
panic = "abort"
```

### 3. 实现插件逻辑

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
struct PolicyInput {
    workers: Vec<WorkerInfo>,
    request_text: Option<String>,
    config: HashMap<String, serde_json::Value>,
    state: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PolicyOutput {
    selected_worker: Option<usize>,
    state: Option<HashMap<String, serde_json::Value>>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkerInfo {
    url: String,
    healthy: bool,
    load: isize,
}

// 必需的导出函数
#[no_mangle]
pub extern "C" fn init() {
    // 初始化策略
}

#[no_mangle]
pub extern "C" fn select_worker(input_ptr: *const u8, input_len: usize) -> *mut u8 {
    // 实现负载均衡逻辑
    // 返回选择的 worker 索引
}

#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    let mut buf = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, size: usize) {
    unsafe {
        let _buf = Vec::from_raw_parts(ptr, size, size);
    }
}
```

### 4. 构建插件

```bash
cargo install wasm-pack
wasm-pack build --target web --release
```

## 测试

运行集成测试：

```bash
cargo test --test wasm_policy_tests
```

## 依赖

- `wasmtime`: WASM 运行时
- `wasmtime-wasi`: WASI 支持
- `serde`: 序列化支持
- `anyhow`: 错误处理

## 文档

详细的使用文档请参考 `docs/wasm_plugins.md`。

## 示例

项目包含两个完整的示例插件：

1. **weighted_random**: 根据配置的权重随机选择 worker
2. **least_connections**: 选择连接数最少的 worker

这些示例展示了如何实现不同类型的负载均衡策略。

## 注意事项

1. 当前实现是简化版本，实际的 WASM 执行功能需要进一步完善
2. 生产环境中需要更严格的安全验证
3. 插件开发需要遵循特定的接口规范
4. 建议在测试环境中充分验证插件功能

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个功能。
