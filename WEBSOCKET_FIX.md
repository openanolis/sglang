# WebSocket Realtime API 修复说明

## 问题描述

执行时遇到以下错误：
1. `404 Not Found` - 访问 `/v1/realtime` 时返回 404
2. `Unsupported upgrade request` - WebSocket 升级请求不支持
3. `No supported WebSocket library detected` - 缺少 WebSocket 库

## 解决方案

### 1. 安装 WebSocket 支持库

已更新 `python/pyproject.toml`，添加了 `websockets` 依赖。

**安装方法：**

如果使用虚拟环境：
```bash
cd /home/test/Projects/sglang-openanolis_omni
pip install websockets
```

或者重新安装项目（推荐）：
```bash
cd /home/test/Projects/sglang-openanolis_omni
pip install -e "python"
```

如果使用系统 Python（需要虚拟环境）：
```bash
cd /home/test/Projects/sglang-openanolis_omni
python3 -m venv venv
source venv/bin/activate
pip install -e "python"
```

### 2. 使用 WebSocket 客户端连接

**重要：** `/v1/realtime` 是 WebSocket 端点，**不能**使用 HTTP GET 请求访问。

必须使用 WebSocket 客户端连接，例如：

**Python 示例：**
```python
import asyncio
import websockets
import json

async def connect_realtime():
    uri = "ws://127.0.0.1:30000/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    async with websockets.connect(uri) as websocket:
        # 发送 session.update 事件
        await websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad", "threshold": 0.5}
            }
        }))
        
        # 接收响应
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(connect_realtime())
```

**JavaScript/Node.js 示例：**
```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://127.0.0.1:30000/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01');

ws.on('open', function open() {
  console.log('Connected');
  
  // 发送 session.update 事件
  ws.send(JSON.stringify({
    type: 'session.update',
    session: {
      modalities: ['text', 'audio'],
      input_audio_format: 'pcm16',
      output_audio_format: 'pcm16'
    }
  }));
});

ws.on('message', function message(data) {
  console.log('Received:', data.toString());
});
```

**使用 curl 测试（需要支持 WebSocket）：**
```bash
# curl 不支持 WebSocket，请使用专门的 WebSocket 客户端工具
# 例如：wscat
npm install -g wscat
wscat -c "ws://127.0.0.1:30000/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
```

### 3. 验证修复

1. 确保已安装 `websockets` 库：
   ```bash
   python3 -c "import websockets; print('websockets installed')"
   ```

2. 重启服务器

3. 使用 WebSocket 客户端连接（不要使用 HTTP GET）

## 注意事项

- `/v1/realtime` 端点**只支持 WebSocket 协议**，不支持 HTTP GET/POST
- 确保服务器启动时使用的是支持 WebSocket 的 uvicorn（安装 `websockets` 后会自动支持）
- 如果仍然遇到问题，检查服务器日志中是否有 `OpenAIServingRealtime` 初始化错误

