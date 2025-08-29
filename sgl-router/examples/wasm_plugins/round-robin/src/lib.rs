mod bindings;
use bindings::exports::sgl_router::policy::policy::{Guest, LoadsEntry, PolicyInput};
use bindings::exports::sgl_router::policy::policy;

struct WasmPolicy;

static mut NEEDS_TEXT: bool = false;

impl Guest for WasmPolicy {
    fn name() -> String {
        "wasm:round-robin".to_string()
    }

    fn needs_request_text() -> bool {
        unsafe { NEEDS_TEXT }
    }

    fn select_worker(input: PolicyInput) -> Option<u32> {
        // 最简单：返回第一个 healthy
        for (idx, w) in input.workers.iter().enumerate() {
            if w.healthy {
                return Some(idx as u32);
            }
        }
        None
    }

    fn update_loads(loads: Vec<LoadsEntry>) {
        // 可在此更新内部状态，这里仅作为示例
        let _ = loads;
    }

    fn reset() {
        // 重置内部状态
        unsafe { NEEDS_TEXT = false; }
    }
}

// 将实现注册给生成的导出模块
bindings::export!(WasmPolicy);