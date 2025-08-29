# WASM 插件系统使用指南

## 概述

WASM 插件系统允许用户在不重新编译 sgl-router 的情况下，通过 WebAssembly 模块实现自定义的负载均衡策略。

## 特性

- 动态加载插件
- 安全沙箱执行
- 热重载支持
- 资源限制控制

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

### 3. 实现插件

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

## 使用插件

### 加载插件

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

### 管理插件

```rust
// 列出插件
let plugins = factory.list_wasm_plugins();

// 卸载插件
factory.unload_wasm_plugin("my_policy");
```

## 构建脚本

```bash
# 构建所有插件
./scripts/build_wasm_plugins.sh build

# 测试插件
./scripts/build_wasm_plugins.sh test

# 清理
./scripts/build_wasm_plugins.sh clean
```

## 安全限制

- 执行时间限制：最大 30 秒
- 内存限制：最大 100MB
- 文件系统：仅限 `/tmp` 目录
- 系统调用：默认禁用
- 网络访问：禁用

## 示例插件

项目包含两个示例插件：

1. **weighted_random**: 加权随机选择策略
2. **least_connections**: 最少连接数策略

详见 `examples/wasm_plugins/` 目录。
