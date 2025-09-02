# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Generator, List, Optional, Tuple
from urllib.parse import urlparse

import torch
import torch.distributed as dist

from sglang.srt.connector import BaseConnector
from sglang.srt.utils import init_custom_process_group

from mooncake import ep

logger = logging.getLogger(__name__)

class MoonCakeConnector(BaseConnector):
    def __init__(self, mooncake_session_id: str, device: torch.device = "cpu"):
        super().__init__(mooncake_sesssion_id, device)
        self.client = .client("mooncake")

    # Does mooncake need this?
    def build_group(self, gpu_id: int = -1, tp_rank: int = -1, mooncake_session_id: str = None, group_rank: int = 1, world_size: int =2):
        return

    def load_weights_from_mooncake(self, name, dtype, shape):
         target_dtype = (
             dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
         )
 
         try:
             weights = torch.empty(shape, dtype=target_dtype, device=self.device)
             dist.broadcast(weights, src=0, group=self._model_update_group)
             return weights
 
         except Exception as e:
             error_msg = (
                 f"Failed to load weights from mooncake: {e}. "
                 f"The full weights of the ModelRunner are partially updated. "
                 f"Please discard the whole weights."
             )
             logger.error(error_msg)
             raise RuntimeError(error_msg) from e

    # Implemented as a no-op to make BaseConnector interface consistent.
    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:
        """
        Pull files from mooncake storage into the temporary directory.

        Args:
            mooncake_model_path: The S3 path of the model.
            allow_pattern: A list of patterns of which files to pull.
            ignore_pattern: A list of patterns of which files not to pull.

        """
        bucket_name, base_dir, files = list_files(
            self.client, self.url, allow_pattern, ignore_pattern
        )
        if len(files) == 0:
            return

        for file in files:
            destination_file = os.path.join(self.local_dir, file.removeprefix(base_dir))
            local_dir = Path(destination_file).parent
            os.makedirs(local_dir, exist_ok=True)
            self.client.download_file(bucket_name, file, destination_file)

    # Implemented as a no-op to make BaseConnector interface consistent.
    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        return

    def close(self):
        self.connection.close()
        super().close()
