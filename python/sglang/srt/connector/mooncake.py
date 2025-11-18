# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pickle
import tempfile
from typing import Generator, List, Optional, Tuple
from urllib.parse import urlparse

import torch

from sglang.srt.connector import BaseKVConnector
from sglang.srt.connector.serde import create_serde
from sglang.srt.connector.utils import pull_files_from_db

logger = logging.getLogger(__name__)


class MooncakeConnector(BaseKVConnector):
    """
    Connector for Mooncake distributed storage to load/save weights using put_tensor/get_tensor API.
    This connector communicates with Mooncake storage to store and retrieve PyTorch tensors.
    """

    def __init__(self, url: str):
        super().__init__(url)
        parsed_url = urlparse(url)
        self.host = parsed_url.hostname
        self.port = parsed_url.port
        self.model_name = parsed_url.path.lstrip("/")
        
        # Initialize Mooncake store client
        self._store = None
        self._connected = False
        
        # Serde for serialization/deserialization
        self.s, self.d = create_serde("safe")
        
        # Initialize connection
        self._init_connection()

    def _init_connection(self):
        """Initialize Mooncake store connection."""
        try:
            from mooncake.store import MooncakeDistributedStore
            
            # Create Mooncake store client
            self._store = MooncakeDistributedStore()
            # Setup connection - assuming metadata server is at host:port
            metadata_server = f"{self.host}:{self.port}"
            success = self._store.setup(
                local_hostname="0.0.0.0",  # Will be determined by Mooncake
                metadata_server=metadata_server,
                global_segment_size=1024 * 1024 * 16,  # 16MB default
                local_buffer_size=1024 * 1024 * 16,    # 16MB default
                protocol="tcp",
                rdma_devices="",
                master_server_addr="127.0.0.1:50051"  # Default master server
            )
            
            if success == 0:  # Success code
                self._connected = True
                logger.info(f"Connected to Mooncake store at {metadata_server}")
            else:
                raise RuntimeError(f"Failed to setup Mooncake store, return code: {success}")
                
        except ImportError as e:
            logger.error("Mooncake store module not available. Please install mooncake-store package.")
            raise ImportError("Mooncake store module is required but not available") from e
        except Exception as e:
            logger.error(f"Failed to connect to Mooncake store: {e}")
            raise

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor by key from Mooncake store."""
        if not self._connected:
            logger.error("Mooncake store is not connected")
            return None
            
        try:
            tensor = self._store.get_tensor(key)
            if tensor is None:
                logger.warning(f"Tensor not found for key: {key}")
                return None
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error getting tensor for key {key}: {e}")
            return None

    def getstr(self, key: str) -> Optional[str]:
        """Get string value by key from Mooncake store."""
        if not self._connected:
            logger.error("Mooncake store is not connected")
            return None
            
        try:
            # Use regular get for string data
            data = self._store.get(key)
            if data is None or len(data) == 0:
                return None
            
            # Convert bytes to string
            return data.decode('utf-8') if isinstance(data, bytes) else str(data)
            
        except Exception as e:
            logger.error(f"Error getting string for key {key}: {e}")
            return None

    def set(self, key: str, tensor: torch.Tensor) -> None:
        """Set tensor by key in Mooncake store."""
        if not self._connected:
            logger.error("Mooncake store is not connected")
            raise RuntimeError("Mooncake store is not connected")
            
        try:
            result = self._store.put_tensor(key, tensor)
            if result != 0:  # Non-zero indicates error
                logger.error(f"Failed to put tensor for key {key}, return code: {result}")
                raise RuntimeError(f"Failed to put tensor for key {key}")
                
        except Exception as e:
            logger.error(f"Error setting tensor for key {key}: {e}")
            raise

    def setstr(self, key: str, obj: str) -> None:
        """Set string by key in Mooncake store."""
        if not self._connected:
            logger.error("Mooncake store is not connected")
            raise RuntimeError("Mooncake store is not connected")
            
        try:
            # Use regular put for string data
            from sglang.srt.connector.serde import create_serde
            s, _ = create_serde("safe")
            data = obj.encode('utf-8') if isinstance(obj, str) else obj
            
            # For Mooncake, we need to handle this as raw bytes
            # Mooncake's put method accepts python buffer protocol
            result = self._store.put(key, data)
            if result != 0:  # Non-zero indicates error
                logger.error(f"Failed to put string for key {key}, return code: {result}")
                raise RuntimeError(f"Failed to put string for key {key}")
                
        except Exception as e:
            logger.error(f"Error setting string for key {key}: {e}")
            raise

    def list(self, prefix: str) -> List[str]:
        """List all keys with the given prefix from Mooncake store."""
        if not self._connected:
            logger.error("Mooncake store is not connected")
            return []
            
        try:
            # Mooncake doesn't have a direct list method, but we can use remove_by_regex
            # to find keys. For now, return empty list as placeholder.
            # In a real implementation, you might want to maintain a key registry
            logger.warning("Mooncake list operation not fully implemented yet")
            return []
            
        except Exception as e:
            logger.error(f"Error listing keys with prefix {prefix}: {e}")
            return []

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Iterate over model weights for the given rank."""
        # For Mooncake, we need to know the key patterns beforehand
        # This is a simplified implementation that assumes standard tensor key patterns
        keys = [
            f"{self.model_name}/keys/rank_{rank}/weight_{i}"
            for i in range(100)  # Arbitrary limit for now
        ]
        
        for key in keys:
            val = self.get(key)
            if val is not None:
                # Remove the prefix to get the actual weight name
                weight_name = key.removeprefix(f"{self.model_name}/keys/rank_{rank}/")
                yield weight_name, val

    def pull_files(
        self,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
    ) -> None:
        """Pull files from Mooncake store to local directory."""
        # For Mooncake, this would involve storing files as tensors or binary data
        # For now, use the existing pull_files_from_db pattern
        pull_files_from_db(self, self.model_name, allow_pattern, ignore_pattern)

    def close(self):
        """Close Mooncake store connection and cleanup resources."""
        try:
            if self._store and self._connected:
                self._store.close()
                self._connected = False
                logger.info("Closed Mooncake store connection")
        except Exception as e:
            logger.error(f"Error closing Mooncake store connection: {e}")
        finally:
            super().close()

    def batch_get_tensor(self, keys: List[str]) -> List[Optional[torch.Tensor]]:
        """Get multiple tensors by keys from Mooncake store using batch API."""
        if not self._connected:
            logger.error("Mooncake store is not connected")
            return [None] * len(keys)
            
        try:
            tensors = self._store.batch_get_tensor(keys)
            result_list = []
            for tensor in tensors:
                result_list.append(tensor if tensor is not None else None)
            return result_list
            
        except Exception as e:
            logger.error(f"Error during batch tensor get: {e}")
            return [None] * len(keys)

    def batch_put_tensor(self, keys: List[str], tensors: List[torch.Tensor]) -> List[int]:
        """Put multiple tensors by keys to Mooncake store using batch API."""
        if not self._connected:
            logger.error("Mooncake store is not connected")
            return [-1] * len(keys)  # Error codes
            
        try:
            import torch
            tensor_objs = []
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor):
                    tensor_objs.append(tensor)
                else:
                    logger.warning(f"Skipping non-tensor object for key: {keys[tensor_objs.__len__()]}")
                    continue
            
            results = self._store.batch_put_tensor(keys, tensor_objs)
            return results
            
        except Exception as e:
            logger.error(f"Error during batch tensor put: {e}")
            return [-1] * len(keys)  # Return error codes for all items
