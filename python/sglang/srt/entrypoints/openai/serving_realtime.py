# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""OpenAI Realtime API serving handler."""

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import orjson
import torch
from fastapi import WebSocket, WebSocketDisconnect

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    RealtimeClientEvent,
    RealtimeErrorEvent,
    RealtimeInputAudioBufferAppendEvent,
    RealtimeInputAudioBufferClearEvent,
    RealtimeInputAudioBufferCommitEvent,
    RealtimeResponseAudioDeltaEvent,
    RealtimeResponseAudioTranscriptDeltaEvent,
    RealtimeResponseAudioTranscriptDoneEvent,
    RealtimeResponseCreateEvent,
    RealtimeResponseDoneEvent,
    RealtimeResponseTextDeltaEvent,
    RealtimeResponseTextDoneEvent,
    RealtimeServerEvent,
    RealtimeSessionSettings,
    RealtimeSessionUpdateEvent,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils.audio_utils import AudioBuffer, decode_audio_data

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class RealtimeSession:
    """Manages a single Realtime API session."""

    def __init__(
        self,
        session_id: str,
        tokenizer_manager: "TokenizerManager",
        template_manager: "TemplateManager",
    ):
        self.session_id = session_id
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.settings = RealtimeSessionSettings()
        self.audio_buffer = AudioBuffer(sample_rate=16000, mono=True)
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_response_id: Optional[str] = None
        self.is_generating = False
        self.created_at = time.time()

    def update_settings(self, settings: RealtimeSessionSettings):
        """Update session settings."""
        self.settings = settings

    def append_audio(self, audio_data: str, format: Optional[str] = None):
        """Append audio data to buffer."""
        try:
            # Decode base64 audio data
            audio_bytes = orjson.loads(f'"{audio_data}"') if isinstance(
                audio_data, str
            ) else audio_data
            if isinstance(audio_bytes, str):
                import base64

                audio_bytes = base64.b64decode(audio_bytes)
            self.audio_buffer.append(audio_bytes, format)
        except Exception as e:
            logger.error(f"Failed to append audio: {e}")
            raise

    def clear_audio_buffer(self):
        """Clear audio buffer."""
        self.audio_buffer.clear()

    def get_audio_for_processing(self) -> Optional[tuple]:
        """Get accumulated audio for processing."""
        return self.audio_buffer.get_audio(min_duration_ms=100)


class OpenAIServingRealtime(OpenAIServingBase):
    """Handler for OpenAI Realtime API WebSocket connections."""

    def __init__(
        self,
        tokenizer_manager: "TokenizerManager",
        template_manager: "TemplateManager",
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.sessions: Dict[str, RealtimeSession] = {}
        self.chat_handler = OpenAIServingChat(
            tokenizer_manager, template_manager
        )

    def _request_id_prefix(self) -> str:
        return "realtime-"

    def _validate_request(self, request: Any) -> Optional[str]:
        """Validate request (not used for WebSocket)."""
        return None

    def _convert_to_internal_request(
        self, request: Any, raw_request: Any = None
    ) -> tuple:
        """Convert request (not used for WebSocket)."""
        raise NotImplementedError("Not used for WebSocket")

    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection."""
        await websocket.accept()
        session_id = f"session_{uuid.uuid4().hex[:16]}"
        session = RealtimeSession(
            session_id, self.tokenizer_manager, self.template_manager
        )
        self.sessions[session_id] = session

        logger.info(f"Realtime API WebSocket connected: {session_id}")

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                try:
                    event_data = orjson.loads(data)
                    event_type = event_data.get("type")

                    if event_type == "session.update":
                        await self._handle_session_update(
                            session, event_data, websocket
                        )
                    elif event_type == "input_audio_buffer.append":
                        await self._handle_audio_append(
                            session, event_data, websocket
                        )
                    elif event_type == "input_audio_buffer.clear":
                        await self._handle_audio_clear(
                            session, event_data, websocket
                        )
                    elif event_type == "input_audio_buffer.commit":
                        await self._handle_audio_commit(
                            session, event_data, websocket
                        )
                    elif event_type == "response.create":
                        await self._handle_response_create(
                            session, event_data, websocket
                        )
                    else:
                        await self._send_error(
                            websocket,
                            f"Unknown event type: {event_type}",
                            "invalid_request_error",
                        )

                except json.JSONDecodeError:
                    await self._send_error(
                        websocket,
                        "Invalid JSON format",
                        "invalid_request_error",
                    )
                except Exception as e:
                    logger.exception(f"Error processing event: {e}")
                    await self._send_error(
                        websocket,
                        f"Internal error: {str(e)}",
                        "internal_error",
                    )

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {session_id}")
        finally:
            # Cleanup
            if session_id in self.sessions:
                del self.sessions[session_id]

    async def _handle_session_update(
        self,
        session: RealtimeSession,
        event_data: Dict[str, Any],
        websocket: WebSocket,
    ):
        """Handle session.update event."""
        try:
            session_settings = RealtimeSessionSettings(
                **event_data.get("session", {})
            )
            session.update_settings(session_settings)
            # Send confirmation (optional)
        except Exception as e:
            await self._send_error(
                websocket, f"Failed to update session: {e}", "invalid_request_error"
            )

    async def _handle_audio_append(
        self,
        session: RealtimeSession,
        event_data: Dict[str, Any],
        websocket: WebSocket,
    ):
        """Handle input_audio_buffer.append event."""
        try:
            audio_data = event_data.get("audio", "")
            format = session.settings.input_audio_format
            session.append_audio(audio_data, format)
        except Exception as e:
            await self._send_error(
                websocket, f"Failed to append audio: {e}", "invalid_request_error"
            )

    async def _handle_audio_clear(
        self,
        session: RealtimeSession,
        event_data: Dict[str, Any],
        websocket: WebSocket,
    ):
        """Handle input_audio_buffer.clear event."""
        session.clear_audio_buffer()

    async def _handle_audio_commit(
        self,
        session: RealtimeSession,
        event_data: Dict[str, Any],
        websocket: WebSocket,
    ):
        """Handle input_audio_buffer.commit event - trigger processing."""
        if session.is_generating:
            await self._send_error(
                websocket,
                "Generation already in progress",
                "invalid_request_error",
            )
            return

        # Get accumulated audio
        audio_result = session.get_audio_for_processing()
        if audio_result is None:
            await self._send_error(
                websocket, "No audio data to process", "invalid_request_error"
            )
            return

        audio_array, sample_rate = audio_result

        # Process audio and generate response
        asyncio.create_task(
            self._process_audio_and_generate(session, audio_array, websocket)
        )

    async def _handle_response_create(
        self,
        session: RealtimeSession,
        event_data: Dict[str, Any],
        websocket: WebSocket,
    ):
        """Handle response.create event."""
        # This can be used to create a response manually
        # For now, we'll trigger generation from audio commit
        pass

    async def _process_audio_and_generate(
        self,
        session: RealtimeSession,
        audio_array: np.ndarray,
        websocket: WebSocket,
    ):
        """Process audio input and generate response."""
        session.is_generating = True
        response_id = f"resp_{uuid.uuid4().hex[:16]}"
        session.current_response_id = response_id

        try:
            # Send response.create event
            await self._send_event(
                websocket,
                RealtimeResponseCreateEvent(
                    type="response.create", response={"id": response_id}
                ),
            )

            # Convert audio to base64 data URL format expected by qwen3-omni
            import base64
            import soundfile as sf
            from io import BytesIO

            # Convert numpy array to bytes
            audio_bytes_io = BytesIO()
            sf.write(audio_bytes_io, audio_array, 16000, format="WAV")
            audio_bytes = audio_bytes_io.getvalue()
            
            # Encode to base64
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            audio_data_url = f"data:audio/wav;base64,{audio_base64}"

            try:
                # Create chat completion request with audio
                chat_request = ChatCompletionRequest(
                    model=self.tokenizer_manager.served_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio_url",
                                    "audio_url": {"url": audio_data_url},
                                }
                            ],
                        }
                    ],
                    stream=True,
                    temperature=session.settings.temperature or 0.7,
                    max_tokens=session.settings.max_response_output_tokens or 4096,
                )

                # Process through chat handler
                adapted_request, _ = self.chat_handler._convert_to_internal_request(
                    chat_request
                )

                # Generate response
                text_output = ""
                async for chunk in self.tokenizer_manager.generate_request(
                    adapted_request, None
                ):
                    if "text" in chunk:
                        delta_text = chunk["text"]
                        if delta_text:
                            text_output += delta_text
                            # Send text delta event
                            await self._send_event(
                                websocket,
                                RealtimeResponseTextDeltaEvent(
                                    type="response.text.delta", delta=delta_text
                                ),
                            )

                # Send text done event
                if text_output:
                    await self._send_event(
                        websocket,
                        RealtimeResponseTextDoneEvent(
                            type="response.text.done", text=text_output
                        ),
                    )

                # Send response done event
                await self._send_event(
                    websocket,
                    RealtimeResponseDoneEvent(
                        type="response.done", response={"id": response_id}
                    ),
                )

                # Update conversation history
                session.conversation_history.append(
                    {"role": "user", "content": "audio_input"}
                )
                session.conversation_history.append(
                    {"role": "assistant", "content": text_output}
                )

        except Exception as e:
            logger.exception(f"Error processing audio: {e}")
            await self._send_error(
                websocket, f"Failed to process audio: {e}", "internal_error"
            )
        finally:
            session.is_generating = False
            session.current_response_id = None

    async def _send_event(
        self, websocket: WebSocket, event: RealtimeServerEvent
    ):
        """Send event to client."""
        try:
            event_dict = event.model_dump(exclude_none=True)
            await websocket.send_text(orjson.dumps(event_dict).decode())
        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    async def _send_error(
        self,
        websocket: WebSocket,
        message: str,
        error_type: str = "internal_error",
    ):
        """Send error event to client."""
        error_event = RealtimeErrorEvent(
            type="error",
            error={"message": message, "type": error_type, "code": 500},
        )
        await self._send_event(websocket, error_event)

