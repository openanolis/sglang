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
"""Audio utilities for Realtime API."""

import base64
import logging
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pybase64
from scipy.signal import resample

try:
    import soundfile as sf
except ImportError:
    sf = None

logger = logging.getLogger(__name__)


def detect_audio_format(audio_data: bytes) -> str:
    """Detect audio format from raw bytes.
    
    Args:
        audio_data: Raw audio bytes
        
    Returns:
        Format string: 'pcm16', 'opus', 'base64', or 'unknown'
    """
    if len(audio_data) < 4:
        return "unknown"
    
    # Check for Opus magic bytes
    if audio_data[:4] == b"Opus":
        return "opus"
    
    # Check for base64 encoding (if it's a string-like representation)
    try:
        # Try to decode as base64
        decoded = base64.b64decode(audio_data[:100], validate=True)
        if len(decoded) > 0:
            return "base64"
    except Exception:
        pass
    
    # Default to PCM16 for raw binary data
    return "pcm16"


def decode_audio_data(
    audio_data: bytes, 
    format: Optional[str] = None,
    sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """Decode audio data from various formats.
    
    Args:
        audio_data: Audio data bytes
        format: Audio format ('pcm16', 'opus', 'base64', or None for auto-detect)
        sample_rate: Target sample rate (default: 16000)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_array, actual_sample_rate)
    """
    if sf is None:
        raise ImportError("soundfile is required for audio processing")
    
    if format is None:
        format = detect_audio_format(audio_data)
    
    try:
        if format == "base64":
            # Decode base64 first
            decoded = pybase64.b64decode(audio_data, validate=True)
            audio, original_sr = sf.read(BytesIO(decoded))
        elif format == "opus":
            # Opus decoding would require additional libraries
            # For now, try to read as if it's a valid audio file
            audio, original_sr = sf.read(BytesIO(audio_data))
        elif format == "pcm16":
            # PCM16: 16-bit signed integers, little-endian
            # Assume mono for now
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(
                np.float32
            ) / 32768.0
            original_sr = sample_rate  # Assume target sample rate for raw PCM
        else:
            # Try to read as a standard audio file
            audio, original_sr = sf.read(BytesIO(audio_data))
        
        # Resample if needed
        if original_sr != sample_rate:
            num_samples = int(len(audio) * float(sample_rate) / original_sr)
            audio = resample(audio, num_samples)
            original_sr = sample_rate
        
        # Convert to mono if requested
        if mono and len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        return audio, original_sr
    
    except Exception as e:
        logger.error(f"Failed to decode audio data: {e}")
        raise ValueError(f"Invalid audio format or corrupted data: {e}")


def encode_audio_data(
    audio: np.ndarray,
    format: str = "pcm16",
    sample_rate: int = 16000
) -> bytes:
    """Encode audio array to bytes.
    
    Args:
        audio: Audio array (mono, float32, normalized to [-1, 1])
        format: Output format ('pcm16', 'base64')
        sample_rate: Sample rate of the audio
        
    Returns:
        Encoded audio bytes
    """
    if format == "pcm16":
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32767.0).astype(np.int16)
        return audio_int16.tobytes()
    elif format == "base64":
        # Convert to PCM16 first, then base64
        audio_int16 = (audio * 32767.0).astype(np.int16)
        pcm_data = audio_int16.tobytes()
        return base64.b64encode(pcm_data)
    else:
        raise ValueError(f"Unsupported output format: {format}")


class AudioBuffer:
    """Audio buffer for accumulating audio chunks."""
    
    def __init__(self, sample_rate: int = 16000, mono: bool = True):
        self.sample_rate = sample_rate
        self.mono = mono
        self.chunks: list[bytes] = []
        self.total_samples = 0
    
    def append(self, audio_data: bytes, format: Optional[str] = None):
        """Append audio chunk to buffer.
        
        Args:
            audio_data: Audio data bytes
            format: Audio format (None for auto-detect)
        """
        self.chunks.append(audio_data)
        # Estimate sample count (rough estimate for PCM16)
        if format == "pcm16" or format is None:
            # PCM16: 2 bytes per sample
            self.total_samples += len(audio_data) // 2
        else:
            # For other formats, we'll decode to get accurate count
            try:
                audio, _ = decode_audio_data(audio_data, format, self.sample_rate, self.mono)
                self.total_samples += len(audio)
            except Exception:
                # Fallback: rough estimate
                self.total_samples += len(audio_data) // 2
    
    def clear(self):
        """Clear the buffer."""
        self.chunks = []
        self.total_samples = 0
    
    def get_audio(
        self, 
        format: Optional[str] = None,
        min_duration_ms: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Get accumulated audio as numpy array.
        
        Args:
            format: Audio format for decoding (None for auto-detect)
            min_duration_ms: Minimum duration in milliseconds (None to return all)
            
        Returns:
            Tuple of (audio_array, sample_rate) or None if buffer is empty
        """
        if not self.chunks:
            return None
        
        # Concatenate all chunks
        combined_data = b"".join(self.chunks)
        
        if len(combined_data) == 0:
            return None
        
        # Decode combined audio
        try:
            audio, sr = decode_audio_data(combined_data, format, self.sample_rate, self.mono)
            
            # Check minimum duration
            if min_duration_ms is not None:
                min_samples = int(min_duration_ms * sr / 1000.0)
                if len(audio) < min_samples:
                    return None
            
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to decode accumulated audio: {e}")
            return None
    
    def get_duration_ms(self) -> float:
        """Get total duration in milliseconds."""
        if self.total_samples == 0:
            return 0.0
        return (self.total_samples / self.sample_rate) * 1000.0

