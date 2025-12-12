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
"""Voice Activity Detection (VAD) utilities for Realtime API using silero-vad."""

import logging
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import silero-vad, but make it optional
try:
    from silero_vad import load_silero_vad, get_speech_timestamps

    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False
    logger.warning(
        "silero-vad is not installed. VAD functionality will be disabled. "
        "Install it with: pip install silero-vad"
    )


class VADManager:
    """Manager for Voice Activity Detection using silero-vad."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 400,
    ):
        """Initialize VAD manager.

        Args:
            sample_rate: Audio sample rate (8000 or 16000 Hz)
            threshold: VAD threshold (0.0 to 1.0)
            min_speech_duration_ms: Minimum speech duration in milliseconds
            max_speech_duration_s: Maximum speech duration in seconds
            min_silence_duration_ms: Minimum silence duration in milliseconds
            speech_pad_ms: Speech padding in milliseconds
        """
        if not SILERO_VAD_AVAILABLE:
            raise ImportError(
                "silero-vad is not installed. Install it with: pip install silero-vad"
            )

        if sample_rate not in [8000, 16000]:
            raise ValueError(
                f"Unsupported sample rate: {sample_rate}. "
                "Must be 8000 or 16000 Hz"
            )

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        # Load VAD model (lazy loading)
        self._model = None
        self._utils = None

    def _load_model(self):
        """Load silero-vad model (lazy loading)."""
        if self._model is None:
            try:
                logger.info(f"Loading silero-vad model for {self.sample_rate}Hz...")
                self._model, self._utils = load_silero_vad()
                logger.info("silero-vad model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load silero-vad model: {e}")
                raise

    def detect_speech(
        self, audio: np.ndarray, return_seconds: bool = False
    ) -> list:
        """Detect speech segments in audio.

        Args:
            audio: Audio array (float32, normalized to [-1, 1])
            return_seconds: If True, return timestamps in seconds; otherwise in samples

        Returns:
            List of speech segments, each as dict with 'start' and 'end' keys
        """
        if not SILERO_VAD_AVAILABLE:
            raise ImportError("silero-vad is not available")

        self._load_model()

        if len(audio) == 0:
            return []

        # Ensure audio is float32 and in the correct range
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize to [-1, 1] if needed
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = np.clip(audio, -1.0, 1.0)

        # Get speech timestamps
        try:
            get_speech_timestamps_fn = self._utils[0]
            speech_timestamps = get_speech_timestamps_fn(
                audio,
                self._model,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                max_speech_duration_s=self.max_speech_duration_s,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=return_seconds,
            )
            return speech_timestamps
        except Exception as e:
            logger.error(f"Error in VAD detection: {e}", exc_info=True)
            return []

    def is_speech(
        self, audio: np.ndarray, window_size_ms: Optional[int] = None
    ) -> Tuple[bool, float]:
        """Check if audio contains speech.

        Args:
            audio: Audio array (float32, normalized to [-1, 1])
            window_size_ms: Optional window size in milliseconds for
                chunked processing (currently unused)

        Returns:
            Tuple of (is_speech: bool, confidence: float)
        """
        if not SILERO_VAD_AVAILABLE:
            return False, 0.0

        if len(audio) == 0:
            return False, 0.0

        self._load_model()

        # Ensure audio is float32 and in the correct range
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize to [-1, 1] if needed
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = np.clip(audio, -1.0, 1.0)

        try:
            # Use get_speech_timestamps to detect speech
            get_speech_timestamps_fn = self._utils[0]
            speech_timestamps = get_speech_timestamps_fn(
                audio,
                self._model,
                threshold=self.threshold,
                min_speech_duration_ms=50,  # Lower for real-time detection
                min_silence_duration_ms=50,
                return_seconds=True,
            )

            # If we have any speech segments, return True
            if speech_timestamps and len(speech_timestamps) > 0:
                # Calculate average confidence (simplified - silero-vad doesn't
                # directly provide confidence). We use the presence of speech
                # segments as confidence indicator.
                return True, 1.0
            return False, 0.0
        except Exception as e:
            logger.error(f"Error in VAD speech detection: {e}", exc_info=True)
            return False, 0.0

    def check_silence_duration(
        self,
        audio: np.ndarray,
        silence_duration_ms: float,
        last_speech_time: Optional[float] = None,
    ) -> Tuple[bool, Optional[float]]:
        """Check if silence duration exceeds threshold.

        Args:
            audio: Audio array (float32, normalized to [-1, 1])
            silence_duration_ms: Required silence duration in milliseconds
            last_speech_time: Timestamp of last detected speech (optional)

        Returns:
            Tuple of (should_commit: bool, updated_last_speech_time: Optional[float])
        """
        if not SILERO_VAD_AVAILABLE:
            return False, None

        is_speech_now, _ = self.is_speech(audio)
        current_time = time.time()

        if is_speech_now:
            # Speech detected, update last speech time
            return False, current_time
        else:
            # No speech detected
            if last_speech_time is None:
                # No previous speech, don't commit
                return False, None

            # Calculate silence duration
            silence_duration = (current_time - last_speech_time) * 1000.0

            if silence_duration >= silence_duration_ms:
                # Silence duration exceeded, should commit
                return True, None
            else:
                # Still within silence threshold
                return False, last_speech_time

