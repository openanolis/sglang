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
import base64
import json
import logging
import time
import uuid
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import orjson
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    RealtimeAudioConfig,
    RealtimeAudioFormat,
    RealtimeAudioInput,
    RealtimeAudioOutput,
    RealtimeConversationItemCreatedEvent,
    RealtimeErrorEvent,
    RealtimeResponseContentPartAddedEvent,
    RealtimeResponseCreateEvent,
    RealtimeResponseOutputItemDoneEvent,
    RealtimeResponseTextDeltaEvent,
    RealtimeServerEvent,
    RealtimeSessionConfiguration,
    RealtimeSessionCreatedEvent,
    RealtimeSessionUpdatedEvent,
    RealtimeTurnDetection,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.utils.audio_utils import AudioBuffer

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


# Realtime API Event Type Constants
# Client events (sent by client to server)
EVENT_TYPE_SESSION_UPDATE = "session.update"
EVENT_TYPE_INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
EVENT_TYPE_INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
EVENT_TYPE_INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
EVENT_TYPE_RESPONSE_CREATE = "response.create"
EVENT_TYPE_CONVERSATION_ITEM_CREATE = "conversation.item.create"

# Server events (sent by server to client)
EVENT_TYPE_SESSION_CREATED = "session.created"
EVENT_TYPE_SESSION_UPDATED = "session.updated"
EVENT_TYPE_RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
EVENT_TYPE_RESPONSE_TEXT_DELTA = "response.text.delta"
EVENT_TYPE_RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
EVENT_TYPE_CONVERSATION_ITEM_CREATED = "conversation.item.created"
EVENT_TYPE_ERROR = "error"


class RealtimeSession:
    """Manages a single Realtime API session."""

    def __init__(
        self,
        session_id: str,
        tokenizer_manager: "TokenizerManager",
        template_manager: "TemplateManager",
        model: Optional[str] = None,
    ):
        self.session_id = session_id
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.model = model or tokenizer_manager.served_model_name
        self.configuration = RealtimeSessionConfiguration(
            id=session_id,
            model=model,
        )
        self.audio_buffer = AudioBuffer(sample_rate=16000, mono=True)
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_response_id: Optional[str] = None
        self.is_generating = False
        self.created_at = time.time()

    def update_configuration(
        self, configuration: RealtimeSessionConfiguration
    ):
        """Update session settings with partial update.

        Only updates fields that are explicitly provided in the configuration
        object. According to OpenAI Realtime API spec, only provided fields
        are updated. Uses Pydantic's model_copy with update to merge
        configuration.
        """
        self.configuration = self.configuration.model_copy(
            update=configuration.model_dump(exclude_unset=True)
        )

    def append_audio(self, audio_data: str, format: Optional[str] = None):
        """Append audio data to buffer."""
        try:
            # Decode base64 audio data
            audio_bytes = orjson.loads(f'"{audio_data}"') if isinstance(
                audio_data, str
            ) else audio_data
            if isinstance(audio_bytes, str):
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

    def _get_max_tokens(self, configuration: RealtimeSessionConfiguration) -> int:
        """Get max_tokens value from configuration.

        Handles both field names and 'inf' value.
        """
        max_tokens = (
            configuration.max_output_tokens
            or configuration.max_response_output_tokens
        )
        if max_tokens and max_tokens != "inf":
            return int(max_tokens)
        return 4096  # Default value

    def _build_session_configuration(
        self, session: RealtimeSession
    ) -> RealtimeSessionConfiguration:
        """Build RealtimeSessionConfiguration from session configuration.

        This helper method constructs a complete session configuration
        from the session's configuration and metadata.
        """
        # Build audio configuration from configuration
        audio_config = None
        if (
            session.configuration.input_audio_format
            or session.configuration.output_audio_format
            or session.configuration.voice
        ):
            # Build input audio format
            input_format = None
            if session.configuration.input_audio_format:
                # Map format string to audio format object
                # Default rate is 24000 for PCM, adjust as needed
                input_format = RealtimeAudioFormat(
                    type="audio/pcm", rate=24000
                )

            # Build turn detection if provided
            turn_detection = None
            if session.configuration.turn_detection:
                turn_detection = RealtimeTurnDetection(
                    **session.configuration.turn_detection
                )

            # Build audio input
            # Note: RealtimeAudioInput doesn't have transcription and
            # turn_detection fields directly. These are handled at the
            # session configuration level.
            audio_input = RealtimeAudioInput(
                format=input_format or RealtimeAudioFormat(
                    type="audio/pcm", rate=24000
                ),
                noise_reduction=None,
            )

            # Build output audio format
            output_format = None
            if session.configuration.output_audio_format:
                output_format = RealtimeAudioFormat(
                    type="audio/pcm", rate=24000
                )

            # Build audio output
            audio_output = RealtimeAudioOutput(
                format=output_format or RealtimeAudioFormat(
                    type="audio/pcm", rate=24000
                ),
                voice=session.configuration.voice,
                speed=1.0,  # Default speed
            )

            audio_config = RealtimeAudioConfig(
                input=audio_input, output=audio_output
            )

        # Build complete session configuration according to API spec
        # Copy all configuration fields from session, preserving existing values
        session_config = RealtimeSessionConfiguration(
            type="realtime",
            object="realtime.session",
            id=session.session_id,
            model=session.model,
            modalities=session.configuration.modalities,
            instructions=session.configuration.instructions,
            tools=session.configuration.tools,
            tool_choice=session.configuration.tool_choice,
            max_output_tokens=session.configuration.max_output_tokens,
            max_response_output_tokens=session.configuration.max_response_output_tokens,
            temperature=session.configuration.temperature,
            voice=session.configuration.voice,
            input_audio_format=session.configuration.input_audio_format,
            output_audio_format=session.configuration.output_audio_format,
            input_audio_transcription=session.configuration.input_audio_transcription,
            turn_detection=session.configuration.turn_detection,
            audio=audio_config,
        )
        return session_config

    async def _send_session_created(
        self, websocket: WebSocket, session: RealtimeSession
    ) -> bool:
        """Send session.created event to client.

        According to OpenAI Realtime API spec:
        https://platform.openai.com/docs/api-reference/realtime-server-events/session/created

        Returns:
            True if event was sent successfully, False otherwise.
        """
        try:
            session_config = self._build_session_configuration(session)

            # Generate event ID for tracking
            event_id = f"evt_{uuid.uuid4().hex[:16]}"

            session_created_event = RealtimeSessionCreatedEvent(
                type=EVENT_TYPE_SESSION_CREATED,
                event_id=event_id,
                session=session_config,
            )
            await self._send_event(websocket, session_created_event)
            logger.debug(
                f"Sent session.created event for session {session.session_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send session.created event: {e}")
            await self._send_error(
                websocket,
                f"Failed to initialize session: {e}",
                "internal_error",
            )
            return False

    async def handle_websocket(
        self, websocket: WebSocket, model: Optional[str] = None
    ):
        """Handle WebSocket connection."""
        await websocket.accept()
        session_id = f"session_{uuid.uuid4().hex[:16]}"
        session = RealtimeSession(
            session_id, self.tokenizer_manager, self.template_manager, model=model
        )
        self.sessions[session_id] = session

        logger.info(
            f"Realtime API WebSocket connected: {session_id}, model: {model}"
        )

        # Send session.created event immediately after connection
        # According to OpenAI Realtime API spec:
        # https://platform.openai.com/docs/api-reference/realtime-server-events/session/created
        if not await self._send_session_created(websocket, session):
            return

        logger.debug("Starting message receive loop")

        try:
            while True:
                try:
                    logger.debug(f"Waiting for message from {session_id}...")

                    # Receive message from client
                    data = await websocket.receive_text()
                    logger.debug(
                        f"Received message from {session_id}: "
                        f"{data[:100] if len(data) > 100 else data}"
                    )

                    try:
                        event_data = orjson.loads(data)
                        event_type = event_data.get("type")
                        logger.debug(
                            f"Processing event type: {event_type} "
                            f"for session {session_id}"
                        )

                        if event_type == EVENT_TYPE_SESSION_UPDATE:
                            await self._handle_session_update(
                                session, event_data, websocket
                            )
                        elif event_type == EVENT_TYPE_INPUT_AUDIO_BUFFER_APPEND:
                            await self._handle_audio_append(
                                session, event_data, websocket
                            )
                        elif event_type == EVENT_TYPE_INPUT_AUDIO_BUFFER_CLEAR:
                            await self._handle_audio_clear(
                                session, event_data, websocket
                            )
                        elif event_type == EVENT_TYPE_INPUT_AUDIO_BUFFER_COMMIT:
                            await self._handle_audio_commit(
                                session, event_data, websocket
                            )
                        elif event_type == EVENT_TYPE_RESPONSE_CREATE:
                            await self._handle_response_create(
                                session, event_data, websocket
                            )
                        elif event_type == EVENT_TYPE_CONVERSATION_ITEM_CREATE:
                            await self._handle_conversation_item_create(
                                session, event_data, websocket
                            )
                        else:
                            await self._send_error(
                                websocket,
                                f"Unknown event type: {event_type}",
                                "invalid_request_error",
                            )

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON from {session_id}: {e}, "
                            f"data: {data[:200]}"
                        )
                        await self._send_error(
                            websocket,
                            "Invalid JSON format",
                            "invalid_request_error",
                        )
                    except Exception as e:
                        logger.exception(
                            f"Error processing event for {session_id}: {e}"
                        )
                        await self._send_error(
                            websocket,
                            f"Internal error: {str(e)}",
                            "internal_error",
                        )
                except WebSocketDisconnect:
                    # Client disconnected normally
                    logger.info(f"WebSocket disconnected normally: {session_id}")
                    break
                except Exception as e:
                    # Handle other exceptions (connection errors, etc.)
                    logger.warning(
                        f"Error receiving message from {session_id}: "
                        f"{type(e).__name__}: {e}"
                    )
                    # Check if connection is still open before breaking
                    try:
                        # Try to check connection state
                        # This will raise if disconnected
                        await websocket.receive_text()
                    except Exception:
                        logger.info(f"Connection closed for {session_id}")
                        break
                    raise  # Re-raise if connection is still open

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.exception(
                f"Unexpected error in WebSocket handler "
                f"for session {session_id}: {e}"
            )
        finally:
            # Cleanup
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.debug(f"Cleaned up session: {session_id}")

    async def _handle_session_update(
        self,
        session: RealtimeSession,
        event_data: Dict[str, Any],
        websocket: WebSocket,
    ):
        """Handle session.update event from client.

        According to OpenAI Realtime API spec:
        - Only provided fields are updated (partial update)
        - Server must respond with session.updated event
        - Model cannot be changed after session creation
        """
        try:
            # Extract session settings from client event
            session_update_dict = event_data.get("session", {})

            # Validate: model cannot be changed (if provided, ignore it)
            if "model" in session_update_dict:
                logger.warning(
                    "Client attempted to change model in session.update, "
                    "ignoring"
                )
                session_update_dict = {
                    k: v
                    for k, v in session_update_dict.items()
                    if k != "model"
                }

            # Create settings object from client-provided fields
            # Use model_validate to parse the partial update
            # This will only include fields that are explicitly provided
            partial_configuration = RealtimeSessionConfiguration.model_validate(
                session_update_dict, strict=False
            )

            # Update session settings (partial update)
            # Only fields provided in partial_settings will be updated
            session.update_configuration(partial_configuration)

            # Build complete session configuration from updated settings
            session_config = self._build_session_configuration(session)

            # Send session.updated event as response
            # According to spec:
            # https://platform.openai.com/docs/api-reference/realtime-server-events/session/updated
            event_id = f"evt_{uuid.uuid4().hex[:16]}"
            session_updated_event = RealtimeSessionUpdatedEvent(
                type=EVENT_TYPE_SESSION_UPDATED,
                event_id=event_id,
                session=session_config,  # Send complete updated session config
            )
            await self._send_event(websocket, session_updated_event)
            logger.debug(
                f"Session updated and session.updated event sent "
                f"for {session.session_id}"
            )

        except Exception as e:
            logger.exception(f"Failed to update session: {e}")
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
            format = session.configuration.input_audio_format
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

        # Clear accumulated audio buffer after getting the audio
        # This ensures we only process the audio that was committed
        session.clear_audio_buffer()
        logger.debug(
            f"Cleared audio buffer after commit "
            f"for session {session.session_id}"
        )

        # Process audio and generate response
        asyncio.create_task(
            self._process_audio_and_generate(session, audio_array, websocket)
        )

    async def _handle_conversation_item_create(
        self,
        session: RealtimeSession,
        event_data: Dict[str, Any],
        websocket: WebSocket,
    ):
        """Handle conversation.item.create event.

        According to OpenAI Realtime API spec:
        https://platform.openai.com/docs/api-reference/realtime-client-events/conversation/item/create

        Client sends an item to be created in the conversation.
        Server should respond with conversation.item.created event.
        """
        try:
            item_data = event_data.get("item", {})
            if not item_data:
                await self._send_error(
                    websocket,
                    "Missing 'item' field in conversation.item.create event",
                    "invalid_request_error",
                )
                return

            # Generate item ID
            item_id = f"item_{uuid.uuid4().hex[:16]}"

            # Extract item fields
            item_type = item_data.get("type", "message")
            role = item_data.get("role")
            content = item_data.get("content", [])
            previous_item_id = item_data.get("previous_item_id")
            status = item_data.get("status", "in_progress")

            # Build the created item with server-assigned fields
            created_item = {
                "id": item_id,
                "object": "conversation.item",
                "type": item_type,
                "status": status,
            }

            if role:
                created_item["role"] = role
            if content:
                created_item["content"] = content
            if previous_item_id:
                created_item["previous_item_id"] = previous_item_id

            # Add to conversation history
            if role == "user":
                # Extract text content from user messages
                text_content = ""
                audio_content = None
                for content_item in content:
                    if isinstance(content_item, dict):
                        if content_item.get("type") == "input_text":
                            text_content += content_item.get("text", "")
                        elif content_item.get("type") == "input_audio":
                            # Handle audio content
                            audio_data = content_item.get("audio")
                            if audio_data:
                                # Decode base64 audio if needed
                                session.append_audio(
                                    audio_data,
                                    session.configuration.input_audio_format,
                                )
                                audio_content = "audio_input"

                # Add to conversation history
                if text_content:
                    session.conversation_history.append(
                        {"role": "user", "content": text_content}
                    )
                elif audio_content:
                    session.conversation_history.append(
                        {"role": "user", "content": audio_content}
                    )

            # Send conversation.item.created event
            event_id = f"evt_{uuid.uuid4().hex[:16]}"
            item_created_event = RealtimeConversationItemCreatedEvent(
                event_id=event_id,
                conversation_id=session.session_id,
                type=EVENT_TYPE_CONVERSATION_ITEM_CREATED,
                item=created_item,
            )
            await self._send_event(websocket, item_created_event)
            logger.debug(
                f"Created conversation item {item_id} "
                f"for session {session.session_id}"
            )

            # According to client.js, creating a user item automatically
            # triggers response creation. If it's a user message,
            # trigger response generation.
            if role == "user" and not session.is_generating:
                has_audio = False
                has_text = False

                # Check content types
                for content_item in content:
                    if isinstance(content_item, dict):
                        content_type = content_item.get("type")
                        if content_type == "input_audio":
                            has_audio = True
                        elif content_type == "input_text":
                            has_text = True

                # Check if there's audio in buffer
                audio_result = session.get_audio_for_processing()

                if has_audio or audio_result:
                    # Commit audio buffer if turn detection is not active
                    if (
                        not session.configuration.turn_detection
                        and audio_result
                    ):
                        audio_array, sample_rate = audio_result
                        # Clear accumulated audio buffer after getting audio
                        session.clear_audio_buffer()
                        logger.debug(
                            f"Cleared audio buffer after conversation "
                            f"item create for session {session.session_id}"
                        )
                        asyncio.create_task(
                            self._process_audio_and_generate(
                                session, audio_array, websocket
                            )
                        )
                    # If turn detection is active, audio will be
                    # committed separately
                elif has_text:
                    # Trigger response creation for text-only messages
                    await self._handle_response_create(session, {}, websocket)

        except Exception as e:
            logger.exception(f"Failed to create conversation item: {e}")
            await self._send_error(
                websocket,
                f"Failed to create conversation item: {e}",
                "invalid_request_error",
            )

    async def _handle_response_create(
        self,
        session: RealtimeSession,
        event_data: Dict[str, Any],
        websocket: WebSocket,
    ):
        """Handle response.create event.

        According to OpenAI Realtime API spec:
        https://platform.openai.com/docs/api-reference/realtime-client-events/response/create

        Triggers the model to generate a response based on the conversation
        history.
        """
        if session.is_generating:
            await self._send_error(
                websocket,
                "Generation already in progress",
                "invalid_request_error",
            )
            return

        try:
            # Build messages from conversation history
            messages = []
            for hist_item in session.conversation_history:
                if isinstance(hist_item, dict) and "role" in hist_item:
                    messages.append(
                        {
                            "role": hist_item["role"],
                            "content": hist_item.get("content", ""),
                        }
                    )

            if not messages:
                await self._send_error(
                    websocket,
                    "No conversation history to generate response from",
                    "invalid_request_error",
                )
                return
        except Exception as e:
            logger.exception(f"Failed to create response: {e}")
            await self._send_error(
                websocket,
                f"Failed to create response: {e}",
                "internal_error",
            )

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
                    type=EVENT_TYPE_RESPONSE_CREATE, response={"id": response_id}
                ),
            )

            # Create assistant item for this response
            # The item_id from this item will be used in subsequent delta events
            item_id = f"item_{uuid.uuid4().hex[:16]}"
            assistant_item = {
                "id": item_id,
                "object": "conversation.item",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }

            # Send conversation.item.created event for the assistant item
            event_id = f"evt_{uuid.uuid4().hex[:16]}"
            item_created_event = RealtimeConversationItemCreatedEvent(
                event_id=event_id,
                conversation_id=session.session_id,
                type=EVENT_TYPE_CONVERSATION_ITEM_CREATED,
                item=assistant_item,
            )
            await self._send_event(websocket, item_created_event)
            logger.debug(
                f"Created assistant item {item_id} for response {response_id}"
            )

            content_index = 0  # Single content part for text responses

            # Send response.content_part.added event before first delta
            # According to conversation.js, this adds a content part to the item
            content_part_added_event = RealtimeResponseContentPartAddedEvent(
                type=EVENT_TYPE_RESPONSE_CONTENT_PART_ADDED,
                item_id=item_id,
                part={"type": "text", "text": ""},
            )
            await self._send_event(websocket, content_part_added_event)
            logger.debug(f"Added content part to item {item_id}")

            # Convert audio to base64 data URL format expected by qwen3-omni
            # Convert numpy array to bytes
            audio_bytes_io = BytesIO()
            sf.write(audio_bytes_io, audio_array, 16000, format="WAV")
            audio_bytes = audio_bytes_io.getvalue()

            # Encode to base64
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            audio_data_url = f"data:audio/wav;base64,{audio_base64}"

            # Create chat completion request with audio
            if not session.model:
                await self._send_error(
                    websocket, "Model not specified", "invalid_request_error"
                )
                return

            chat_request = ChatCompletionRequest(
                model=session.model,
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
                temperature=session.configuration.temperature or 0.7,
                max_tokens=self._get_max_tokens(session.configuration),
            )

            # Process through chat handler
            adapted_request, _ = self.chat_handler._convert_to_internal_request(
                chat_request
            )

            # Generate response
            text_output = ""
            previous_text = ""  # Track previous cumulative text to extract delta
            async for chunk in self.tokenizer_manager.generate_request(
                adapted_request, None
            ):
                if "text" in chunk:
                    cumulative_text = chunk["text"]
                    # Extract only the new delta text (incremental part)
                    delta_text = cumulative_text[len(previous_text):]
                    if delta_text:
                        text_output += delta_text
                        previous_text = cumulative_text
                        # Send text delta event with item_id and content_index
                        # delta field contains only the incremental text,
                        # not cumulative
                        await self._send_event(
                            websocket,
                            RealtimeResponseTextDeltaEvent(
                                type=EVENT_TYPE_RESPONSE_TEXT_DELTA,
                                item_id=item_id,
                                content_index=content_index,
                                delta=delta_text,
                            ),
                        )

            # Send response.output_item.done event to mark item as completed
            # According to conversation.js, this marks the item as done
            await self._send_event(
                websocket,
                RealtimeResponseOutputItemDoneEvent(
                    type=EVENT_TYPE_RESPONSE_OUTPUT_ITEM_DONE,
                    item={"id": item_id, "status": "completed"},
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

    async def _process_text_and_generate(
        self,
        session: RealtimeSession,
        messages: List[Dict[str, Any]],
        websocket: WebSocket,
    ):
        """Process text conversation and generate response."""
        session.is_generating = True
        response_id = f"resp_{uuid.uuid4().hex[:16]}"
        session.current_response_id = response_id

        try:
            # Send response.create event
            await self._send_event(
                websocket,
                RealtimeResponseCreateEvent(
                    type=EVENT_TYPE_RESPONSE_CREATE, response={"id": response_id}
                ),
            )

            # Create assistant item for this response
            # The item_id from this item will be used in subsequent delta events
            item_id = f"item_{uuid.uuid4().hex[:16]}"
            assistant_item = {
                "id": item_id,
                "object": "conversation.item",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }

            # Send conversation.item.created event for the assistant item
            event_id = f"evt_{uuid.uuid4().hex[:16]}"
            item_created_event = RealtimeConversationItemCreatedEvent(
                event_id=event_id,
                conversation_id=session.session_id,
                type=EVENT_TYPE_CONVERSATION_ITEM_CREATED,
                item=assistant_item,
            )
            await self._send_event(websocket, item_created_event)
            logger.debug(
                f"Created assistant item {item_id} for response {response_id}"
            )

            content_index = 0  # Single content part for text responses

            # Send response.content_part.added event before first delta
            # According to conversation.js, this adds a content part to the item
            content_part_added_event = RealtimeResponseContentPartAddedEvent(
                type=EVENT_TYPE_RESPONSE_CONTENT_PART_ADDED,
                item_id=item_id,
                part={"type": "text", "text": ""},
            )
            await self._send_event(websocket, content_part_added_event)
            logger.debug(f"Added content part to item {item_id}")

            # Create chat completion request
            if not session.model:
                await self._send_error(
                    websocket, "Model not specified", "invalid_request_error"
                )
                return

            chat_request = ChatCompletionRequest(
                model=session.model,
                messages=messages,  # type: ignore
                stream=True,
                temperature=session.configuration.temperature or 0.7,
                max_tokens=self._get_max_tokens(session.configuration),
            )

            # Process through chat handler
            adapted_request, _ = self.chat_handler._convert_to_internal_request(
                chat_request
            )

            # Generate response
            text_output = ""
            previous_text = ""  # Track previous cumulative text to extract delta
            async for chunk in self.tokenizer_manager.generate_request(
                adapted_request, None
            ):
                if "text" in chunk:
                    cumulative_text = chunk["text"]
                    # Extract only the new delta text (incremental part)
                    delta_text = cumulative_text[len(previous_text):]
                    if delta_text:
                        text_output += delta_text
                        previous_text = cumulative_text
                        # Send text delta event with item_id and content_index
                        # delta field contains only the incremental text,
                        # not cumulative
                        await self._send_event(
                            websocket,
                            RealtimeResponseTextDeltaEvent(
                                type=EVENT_TYPE_RESPONSE_TEXT_DELTA,
                                item_id=item_id,
                                content_index=content_index,
                                delta=delta_text,
                            ),
                        )

            # Send response.output_item.done event to mark item as completed
            # According to conversation.js, this marks the item as done
            await self._send_event(
                websocket,
                RealtimeResponseOutputItemDoneEvent(
                    type=EVENT_TYPE_RESPONSE_OUTPUT_ITEM_DONE,
                    item={"id": item_id, "status": "completed"},
                ),
            )

            # Update conversation history
            session.conversation_history.append(
                {"role": "assistant", "content": text_output}
            )

        except Exception as e:
            logger.exception(f"Error processing text: {e}")
            await self._send_error(
                websocket, f"Failed to process text: {e}", "internal_error"
            )
        finally:
            session.is_generating = False
            session.current_response_id = None

    async def _send_event(
        self, websocket: WebSocket, event: RealtimeServerEvent
    ):
        """Send event to client.

        Automatically generates event_id if not already set.
        """
        try:
            # Generate event_id if not already set
            event_id = None
            if hasattr(event, "event_id"):
                event_id = getattr(event, "event_id", None)

            if event_id is None:
                event_id = f"evt_{uuid.uuid4().hex[:16]}"
                # Update event with event_id using model_copy
                event = event.model_copy(update={"event_id": event_id})

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
            type=EVENT_TYPE_ERROR,
            error={"message": message, "type": error_type, "code": 500},
        )
        await self._send_event(websocket, error_event)

