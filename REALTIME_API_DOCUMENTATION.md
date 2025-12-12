# SGLang Realtime API Design, Implementation, and Demo

## Overview

This document describes the design, implementation, and usage of the OpenAI Realtime API compatible implementation in SGLang. The implementation follows the [OpenAI Realtime API specification](https://platform.openai.com/docs/api-reference/realtime) and is compatible with the [OpenAI Realtime Console](https://github.com/openai/openai-realtime-console/tree/websockets).

## Architecture Design

### Core Components

The Realtime API implementation consists of three main components:

1. **`RealtimeSession`** - Manages per-connection session state
2. **`OpenAIServingRealtime`** - Handles WebSocket connections and event routing
3. **Event Handlers** - Process specific client events and generate server responses

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    HTTP Server (FastAPI)                    │
│                    /v1/realtime (WebSocket)                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│            OpenAIServingRealtime                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  handle_websocket()                                  │   │
│  │  - Creates RealtimeSession                           │   │
│  │  - Routes events to handlers                         │   │
│  │  - Manages session lifecycle                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Event Handlers                                      │   │
│  │  - _handle_session_update()                          │   │
│  │  - _handle_audio_append()                            │   │
│  │  - _handle_audio_commit()                            │   │
│  │  - _handle_conversation_item_create()                │   │
│  │  - _handle_response_create()                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Response Generators                                 │   │
│  │  - _process_text_and_generate()                      │   │
│  │  - _process_audio_and_generate()                     │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    RealtimeSession                          │
│  - session_id: Unique session identifier                    │
│  - configuration: Session settings                          │
│  - audio_buffer: Accumulated audio data                     │
│  - conversation_history: Message history                    │
│  - is_generating: Generation state flag                     │
│  - vad_manager: VAD manager (optional)                      │
│  - last_speech_time: Last detected speech timestamp         │
│  - silence_start_time: Silence period start timestamp       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              TokenizerManager & ChatHandler                 │
│              (Shared with Chat API)                         │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Session Management

Each WebSocket connection creates a unique `RealtimeSession` instance that maintains:

- **Session Configuration**: Model, modalities, audio formats, instructions, etc.
- **Audio Buffer**: Accumulates audio chunks from `input_audio_buffer.append` events
- **Conversation History**: Maintains the conversation context for response generation
- **Generation State**: Prevents concurrent response generation

```python
class RealtimeSession:
    def __init__(self, session_id, tokenizer_manager, template_manager, model=None):
        self.session_id = session_id
        self.configuration = RealtimeSessionConfiguration(...)
        self.audio_buffer = AudioBuffer(sample_rate=16000, mono=True)
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_generating = False
```

### 2. Event-Driven Architecture

The implementation uses an event-driven architecture where:

- **Client Events**: Sent by the client to trigger actions
- **Server Events**: Sent by the server to notify the client of state changes

#### Supported Client Events

| Event Type | Handler | Description |
|------------|---------|-------------|
| `session.update` | `_handle_session_update()` | Update session configuration |
| `input_audio_buffer.append` | `_handle_audio_append()` | Append audio chunk to buffer |
| `input_audio_buffer.clear` | `_handle_audio_clear()` | Clear audio buffer |
| `input_audio_buffer.commit` | `_handle_audio_commit()` | Trigger audio processing |
| `conversation.item.create` | `_handle_conversation_item_create()` | Create conversation item |
| `response.create` | `_handle_response_create()` | Trigger response generation |

#### Supported Server Events

| Event Type | When Sent | Description |
|------------|-----------|-------------|
| `session.created` | On connection | Initial session configuration |
| `session.updated` | After `session.update` | Updated session configuration |
| `conversation.item.created` | After item creation | New conversation item |
| `response.create` | When response starts | Response creation notification |
| `response.content_part.added` | Before first delta | Content part initialization |
| `response.text.delta` | During generation | Incremental text updates |
| `response.output_item.done` | After generation | Item completion |
| `error` | On errors | Error notification |

### 3. Response Generation Flow

#### Text-Only Response Flow

```
1. Client sends conversation.item.create (user message)
   └─> Server creates item and adds to history
   └─> Auto-triggers response.create

2. _handle_response_create()
   └─> Validates conversation history
   └─> Calls _process_text_and_generate()

3. _process_text_and_generate()
   ├─> Generates response_id
   ├─> Sends response.create event
   ├─> Creates assistant item
   ├─> Sends conversation.item.created
   ├─> Sends response.content_part.added
   ├─> Streams text deltas via response.text.delta
   └─> Sends response.output_item.done
```

#### Audio Response Flow

```
1. Client sends input_audio_buffer.append (multiple times)
   └─> Audio chunks accumulated in buffer

2. Client sends input_audio_buffer.commit
   └─> Triggers _process_audio_and_generate()

3. _process_audio_and_generate()
   ├─> Converts audio to base64 data URL
   ├─> Creates ChatCompletionRequest with audio
   ├─> Streams through tokenizer_manager
   ├─> Sends text deltas
   └─> Updates conversation history
```

### 4. Key Implementation Features

#### Delta-Based Streaming

The implementation sends incremental text deltas rather than cumulative text:

```python
previous_text = ""
async for chunk in self.tokenizer_manager.generate_request(...):
    cumulative_text = chunk["text"]
    delta_text = cumulative_text[len(previous_text):]
    if delta_text:
        await self._send_event(websocket, RealtimeResponseTextDeltaEvent(
            type=EVENT_TYPE_RESPONSE_TEXT_DELTA,
            item_id=item_id,
            content_index=content_index,
            delta=delta_text  # Only incremental text
        ))
        previous_text = cumulative_text
```

#### Automatic Response Triggering

When a user creates a conversation item, the server automatically triggers response generation:

```python
if role == "user" and not session.is_generating:
    if has_text:
        await self._handle_response_create(session, {}, websocket)
    elif has_audio or audio_result:
        asyncio.create_task(
            self._process_audio_and_generate(session, audio_array, websocket)
        )
```

#### Session Configuration Updates

Session updates support partial updates using Pydantic's `model_copy`:

```python
partial_configuration = RealtimeSessionConfiguration.model_validate(
    session_update_dict, strict=False
)
session.configuration = session.configuration.model_copy(
    update=partial_configuration.model_dump(exclude_unset=True)
)
```

#### Voice Activity Detection (VAD)

SGLang Realtime API supports optional server-side Voice Activity Detection using [silero-vad](https://github.com/snakers4/silero-vad). When enabled, the server automatically detects speech activity in incoming audio and can automatically commit audio when silence is detected.

**How VAD Works**:

1. When VAD is enabled via `turn_detection` configuration, the server initializes a VAD manager for the session
2. As audio chunks are appended via `input_audio_buffer.append`, the server performs real-time VAD detection
3. The server tracks speech and silence periods
4. When silence duration exceeds the configured threshold (`silence_duration_ms`), the server automatically triggers `input_audio_buffer.commit`
5. This enables hands-free operation where users don't need to manually commit audio

**VAD Configuration**:

VAD is configured through the `turn_detection` field in session configuration:

```python
{
    "type": "session.update",
    "session": {
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "silence_duration_ms": 500,
            "idle_timeout_ms": 5000
        }
    }
}
```

**VAD Parameters**:

- `type` (required): Must be `"server_vad"` to enable VAD
- `threshold` (optional, default: 0.5): VAD detection threshold (0.0 to 1.0). Higher values require stronger speech signals
- `silence_duration_ms` (optional, default: 500): Duration of silence in milliseconds before auto-committing audio
- `idle_timeout_ms` (optional): Maximum idle time before committing audio even without speech detection
- `prefix_padding_ms` (optional): Padding time before speech start (for future use)
- `create_response` (optional): Whether to auto-create response (for future use)
- `interrupt_response` (optional): Whether to interrupt ongoing response (for future use)

**Installation**:

To use VAD functionality, install the optional dependency:

```bash
pip install silero-vad
```

Or install SGLang with VAD support:

```bash
pip install sglang[vad]
```

**VAD Flow**:

```
1. Client sends session.update with turn_detection configuration
   └─> Server initializes VAD manager for the session

2. Client sends input_audio_buffer.append (multiple times)
   └─> Audio chunks accumulated in buffer
   └─> Server performs VAD detection on each chunk
   └─> Server tracks speech/silence state

3. When silence duration >= silence_duration_ms:
   └─> Server automatically triggers input_audio_buffer.commit
   └─> Audio processing begins automatically
```

## Comparison with OpenAI Reference Implementation

### Similarities

1. **Event Types**: All core event types match the OpenAI specification
2. **Session Management**: Similar session lifecycle and configuration model
3. **WebSocket Protocol**: Compatible with OpenAI's WebSocket subprotocols
4. **Event Flow**: Follows the same event sequence patterns

### Differences

1. **Response Creation**: 
   - OpenAI: `response.created` event sent by server
   - SGLang: `response.create` event sent by server (current implementation)
   
2. **Item Management**:
   - OpenAI: Client manages conversation items explicitly
   - SGLang: Server auto-creates assistant items during response generation

3. **Audio Processing**:
   - OpenAI: Supports multiple audio formats and VAD
   - SGLang: Supports PCM16, basic audio processing, and server-side VAD (using silero-vad)

## Usage Examples

### 1. Basic Text Conversation

```python
import asyncio
import websockets
import json

async def text_conversation():
    uri = "ws://localhost:30000/v1/realtime?model=your-model"
    
    async with websockets.connect(uri) as websocket:
        # Receive session.created
        session_created = json.loads(await websocket.recv())
        print(f"Session created: {session_created['session']['id']}")
        
        # Update session (optional)
        await websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": "You are a helpful assistant.",
                "temperature": 0.7
            }
        }))
        
        # Receive session.updated
        await websocket.recv()
        
        # Create user message
        await websocket.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "Hello, how are you?"
                }]
            }
        }))
        
        # Receive conversation.item.created
        item_created = json.loads(await websocket.recv())
        print(f"Item created: {item_created['item']['id']}")
        
        # Receive response.create
        response_create = json.loads(await websocket.recv())
        print(f"Response created: {response_create['response']['id']}")
        
        # Receive conversation.item.created (assistant)
        assistant_item = json.loads(await websocket.recv())
        
        # Receive content_part.added
        await websocket.recv()
        
        # Receive text deltas
        while True:
            message = json.loads(await websocket.recv())
            if message["type"] == "response.text.delta":
                print(message["delta"], end="", flush=True)
            elif message["type"] == "response.output_item.done":
                print("\nResponse complete")
                break

asyncio.run(text_conversation())
```

### 2. Audio Conversation

```python
import asyncio
import websockets
import json
import base64

async def audio_conversation():
    uri = "ws://localhost:30000/v1/realtime?model=your-model"
    
    async with websockets.connect(uri) as websocket:
        # Receive session.created
        await websocket.recv()
        
        # Configure for audio
        await websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }))
        await websocket.recv()  # session.updated
        
        # Read audio file and encode
        with open("audio.wav", "rb") as f:
            audio_data = base64.b64encode(f.read()).decode()
        
        # Append audio chunks (simulated)
        chunk_size = 1024
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            await websocket.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": chunk
            }))
        
        # Commit audio
        await websocket.send(json.dumps({
            "type": "input_audio_buffer.commit"
        }))
        
        # Receive response events
        while True:
            message = json.loads(await websocket.recv())
            if message["type"] == "response.text.delta":
                print(message["delta"], end="", flush=True)
            elif message["type"] == "response.output_item.done":
                break

asyncio.run(audio_conversation())
```

### 3. Audio Conversation with VAD (Voice Activity Detection)

This example demonstrates using server-side VAD for automatic audio commit:

```python
import asyncio
import websockets
import json
import base64

async def audio_conversation_with_vad():
    uri = "ws://localhost:30000/v1/realtime?model=your-model"
    
    async with websockets.connect(uri) as websocket:
        # Receive session.created
        await websocket.recv()
        
        # Configure for audio with VAD
        await websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 500,
                    "idle_timeout_ms": 5000
                }
            }
        }))
        await websocket.recv()  # session.updated
        
        # Stream audio chunks continuously
        # VAD will automatically detect when speech ends
        # and commit the audio automatically
        with open("audio.wav", "rb") as f:
            audio_data = base64.b64encode(f.read()).decode()
        
        chunk_size = 1024
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            await websocket.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": chunk
            }))
            # No need to manually commit - VAD will do it automatically
            # when silence is detected
        
        # Receive response events
        while True:
            message = json.loads(await websocket.recv())
            if message["type"] == "response.text.delta":
                print(message["delta"], end="", flush=True)
            elif message["type"] == "response.output_item.done":
                break

asyncio.run(audio_conversation_with_vad())
```

**Key Differences with VAD**:

- No manual `input_audio_buffer.commit` needed
- Server automatically detects speech end and commits audio
- More natural conversation flow
- Requires `silero-vad` to be installed

### 4. Using OpenAI Realtime Console

The implementation is compatible with the [OpenAI Realtime Console](https://github.com/openai/openai-realtime-console/tree/websockets):

1. **Clone the console**:
   ```bash
   git clone https://github.com/openai/openai-realtime-console.git
   cd openai-realtime-console
   git checkout websockets
   ```

2. **Configure the console** to point to your SGLang server:
   - Edit `src/lib/realtime-api-beta/index.js` or use environment variables
   - Set the WebSocket URL to `ws://localhost:30000/v1/realtime`

3. **Start the console**:
   ```bash
   npm install
   npm start
   ```

4. **Connect to your SGLang server**:
   - The console will connect via WebSocket
   - You can send text messages, audio, and interact with the model in real-time

## API Endpoint

### WebSocket Endpoint

**URL**: `ws://host:port/v1/realtime?model=model-name`

**Subprotocols Supported**:
- `openai-beta.realtime-v1`
- `realtime`
- `openai-insecure-api-key.${apiKey}` (for API key authentication)

**Query Parameters**:
- `model` (optional): Model name to use for the session

## Event Reference

### Client → Server Events

#### session.update
```json
{
  "type": "session.update",
  "session": {
    "instructions": "You are a helpful assistant",
    "temperature": 0.7,
    "modalities": ["text", "audio"],
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5,
      "silence_duration_ms": 500
    }
  }
}
```

#### input_audio_buffer.append
```json
{
  "type": "input_audio_buffer.append",
  "audio": "base64-encoded-audio-chunk"
}
```

#### conversation.item.create
```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "message",
    "role": "user",
    "content": [{
      "type": "input_text",
      "text": "Hello!"
    }]
  }
}
```

#### response.create
```json
{
  "type": "response.create"
}
```

### Server → Client Events

#### session.created
```json
{
  "type": "session.created",
  "event_id": "evt_...",
  "session": {
    "id": "session_...",
    "model": "your-model",
    "modalities": ["text"],
    ...
  }
}
```

#### response.text.delta
```json
{
  "type": "response.text.delta",
  "event_id": "evt_...",
  "item_id": "item_...",
  "content_index": 0,
  "delta": "Hello"
}
```

#### response.output_item.done
```json
{
  "type": "response.output_item.done",
  "event_id": "evt_...",
  "item": {
    "id": "item_...",
    "status": "completed"
  }
}
```

## Voice Activity Detection (VAD)

SGLang Realtime API supports optional server-side Voice Activity Detection (VAD) using [silero-vad](https://github.com/snakers4/silero-vad), an enterprise-grade VAD model. VAD enables automatic detection of speech activity and can automatically commit audio when silence is detected, providing a more natural hands-free conversation experience.

### Overview

When VAD is enabled, the server:

1. **Monitors audio in real-time**: As audio chunks arrive via `input_audio_buffer.append`, the server performs continuous VAD detection
2. **Tracks speech and silence**: The server maintains state about when speech was last detected
3. **Auto-commits on silence**: When silence duration exceeds the configured threshold, the server automatically triggers `input_audio_buffer.commit`
4. **Supports idle timeout**: Optionally commits audio after a maximum idle period even without speech detection

### Benefits

- **Hands-free operation**: No need to manually send `input_audio_buffer.commit`
- **Natural conversation flow**: Automatic turn-taking based on speech detection
- **Configurable sensitivity**: Adjust threshold and silence duration for different use cases
- **Low latency**: Real-time detection with minimal overhead

### Installation

VAD functionality requires the optional `silero-vad` dependency:

```bash
# Install silero-vad directly
pip install silero-vad

# Or install SGLang with VAD support
pip install sglang[vad]
```

**System Requirements**:

- Python 3.8+
- 1GB+ RAM
- Modern CPU with AVX, AVX2, AVX-512 or AMX instruction sets
- `torch>=1.12.0` and `torchaudio>=0.12.0` (usually installed with silero-vad)

### Configuration

Enable VAD by setting `turn_detection` in the session configuration:

```json
{
  "type": "session.update",
  "session": {
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5,
      "silence_duration_ms": 500,
      "idle_timeout_ms": 5000
    }
  }
}
```

**Configuration Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | required | Must be `"server_vad"` to enable VAD |
| `threshold` | float | 0.5 | VAD detection threshold (0.0-1.0). Higher = requires stronger speech signal |
| `silence_duration_ms` | integer | 500 | Duration of silence (ms) before auto-committing audio |
| `idle_timeout_ms` | integer | optional | Maximum idle time (ms) before committing even without speech |
| `prefix_padding_ms` | integer | optional | Padding time before speech start (reserved for future use) |
| `create_response` | boolean | optional | Auto-create response flag (reserved for future use) |
| `interrupt_response` | boolean | optional | Interrupt ongoing response flag (reserved for future use) |

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  Client sends input_audio_buffer.append                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Server: Append audio to buffer                             │
│  Server: Perform VAD detection on audio chunk               │
└────────────────────────────┬────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        Speech Detected?          Silence Detected?
                │                         │
                │                         │
        Update last_speech_time    Check silence duration
                │                         │
                │                         │
                │              ┌──────────┴──────────┐
                │              │                     │
                │              ▼                     ▼
                │      Duration >= threshold?   Duration < threshold?
                │              │                     │
                │              │                     │
                │              ▼                     │
                │      Auto-commit audio             │
                │              │                     │
                │              │                     │
                └──────────────┴─────────────────────┘
                             │
                             ▼
                    Process audio and generate response
```

### Best Practices

1. **Threshold Tuning**:
   - Lower threshold (0.3-0.4): More sensitive, detects quieter speech but may have false positives
   - Medium threshold (0.5): Balanced for most use cases
   - Higher threshold (0.6-0.7): Less sensitive, requires clearer speech but fewer false positives

2. **Silence Duration**:
   - Short duration (200-300ms): Faster response but may cut off slow speakers
   - Medium duration (500ms): Good balance for most conversations
   - Long duration (800-1000ms): Slower but allows for natural pauses

3. **Idle Timeout**:
   - Set `idle_timeout_ms` to handle cases where no speech is detected initially
   - Useful for handling background noise or very quiet speech

4. **Sample Rate**:
   - VAD supports 8000Hz and 16000Hz sample rates
   - 16000Hz is recommended for better accuracy

### Troubleshooting

**VAD not working**:
- Check that `silero-vad` is installed: `pip list | grep silero-vad`
- Verify `turn_detection.type` is set to `"server_vad"`
- Check server logs for VAD initialization errors

**Too many false positives**:
- Increase `threshold` value
- Increase `silence_duration_ms` to require longer silence

**Missing speech detection**:
- Decrease `threshold` value
- Check audio quality and sample rate
- Verify audio format is supported (PCM16 recommended)

**Auto-commit not triggering**:
- Check `silence_duration_ms` is not too high
- Verify audio contains actual speech (not just silence)
- Check server logs for VAD errors

## Configuration Options

### Session Configuration Fields

- `model`: Model identifier
- `modalities`: List of supported modalities (`["text"]`, `["text", "audio"]`)
- `instructions`: System instructions for the model
- `temperature`: Sampling temperature (0.0-2.0)
- `max_output_tokens`: Maximum tokens in response
- `input_audio_format`: Audio format (`"pcm16"`, `"g711_ulaw"`, `"g711_alaw"`)
- `output_audio_format`: Output audio format
- `voice`: Voice selection (if supported)
- `turn_detection`: Voice activity detection (VAD) settings
  - `type`: VAD type, use `"server_vad"` for server-side VAD
  - `threshold`: VAD detection threshold (0.0-1.0, default: 0.5)
  - `silence_duration_ms`: Silence duration in ms before auto-commit (default: 500)
  - `idle_timeout_ms`: Maximum idle time before auto-commit (optional)
  - `prefix_padding_ms`: Padding before speech start (optional, for future use)
  - `create_response`: Auto-create response flag (optional, for future use)
  - `interrupt_response`: Interrupt ongoing response flag (optional, for future use)
- `tools`: Function calling tools
- `tool_choice`: Tool usage strategy

## Error Handling

Errors are sent as `error` events:

```json
{
  "type": "error",
  "event_id": "evt_...",
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

Common error types:
- `invalid_request_error`: Invalid client request
- `internal_error`: Server-side error

## Limitations and Future Work

### Current Limitations

1. **Audio Support**: Basic PCM16 support, limited audio processing
2. **Voice Activity Detection**: Server-side VAD is implemented using silero-vad, but requires optional dependency installation
3. **Audio Output**: Text-only responses (audio output generation not implemented)
4. **Tool Calling**: Framework exists but needs integration
5. **Interruption**: Response cancellation not fully implemented

### Future Enhancements

1. Full audio input/output support
2. Enhanced VAD features (prefix padding, response interruption)
3. Response interruption/cancellation
4. Tool calling integration
5. Multi-modal support (images, etc.)
6. Streaming audio output

## Testing

### Manual Testing

Use the OpenAI Realtime Console or a custom WebSocket client to test:

```bash
# Start SGLang server
python -m sglang.launch_server --model-path your-model

# Connect with WebSocket client
wscat -c "ws://localhost:30000/v1/realtime?model=your-model"
```

### Example Test Sequence

1. Connect and receive `session.created`
2. Send `session.update` and receive `session.updated`
3. Send `conversation.item.create` with user message
4. Receive `conversation.item.created`
5. Receive `response.create`
6. Receive `conversation.item.created` (assistant)
7. Receive `response.content_part.added`
8. Receive multiple `response.text.delta` events
9. Receive `response.output_item.done`

## References

- [OpenAI Realtime API Documentation](https://platform.openai.com/docs/api-reference/realtime)
- [OpenAI Realtime Console](https://github.com/openai/openai-realtime-console/tree/websockets)
- [SGLang Documentation](https://github.com/sgl-project/sglang)

## Code Locations

- **Main Handler**: `python/sglang/srt/entrypoints/openai/serving_realtime.py`
- **Protocol Definitions**: `python/sglang/srt/entrypoints/openai/protocol.py`
- **HTTP Server Integration**: `python/sglang/srt/entrypoints/http_server.py` (lines 1333-1411)
- **VAD Utilities**: `python/sglang/srt/utils/vad_utils.py`
- **Audio Utilities**: `python/sglang/srt/utils/audio_utils.py`
