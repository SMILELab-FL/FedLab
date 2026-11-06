# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .serialization import SerializationTool


class Int4BlockCompressor(object):
    """Block-wise INT4 tensor compressor with dynamic scaling and residual error feedback.

    Pack two signed 4-bit values into a single uint8 byte, yielding an 8x reduction
    in raw tensor transmission size for serialized model parameters and gradients.

    Args:
        block_size (int, optional): Size of each quantization block. Must be positive and even. Defaults to 64.
        use_error_feedback (bool, optional): Whether to track local quantization residuals across rounds. Defaults to False.
    """

    def __init__(self, block_size: int = 64, use_error_feedback: bool = False):
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError(f"block_size must be a positive even integer, got {block_size}")
        self.block_size = block_size
        self.use_error_feedback = use_error_feedback
        self.residual: Optional[torch.Tensor] = None

    def reset_error(self) -> None:
        """Clear cached residual buffer."""
        self.residual = None

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
        """Quantize and pack a tensor into INT4 nibbles with per-block scale factors.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
                - packed: uint8 tensor holding two 4-bit elements per byte.
                - scale: float scale factors per block.
                - orig_shape: original tensor shape.
                - orig_dtype: original tensor dtype.
        """
        orig_shape = tensor.shape
        orig_dtype = tensor.dtype
        x = tensor.detach().float()

        if self.use_error_feedback:
            if self.residual is None or self.residual.shape != orig_shape or self.residual.device != tensor.device:
                self.residual = torch.zeros_like(x)
            x = x + self.residual

        flat = x.flatten()
        n = flat.numel()
        pad_len = (self.block_size - (n % self.block_size)) % self.block_size
        if pad_len > 0:
            padded = torch.nn.functional.pad(flat, (0, pad_len), value=0.0)
        else:
            padded = flat

        m = padded.numel() // self.block_size
        blocks = padded.view(m, self.block_size)

        max_abs = torch.max(torch.abs(blocks), dim=1, keepdim=True).values
        scale = torch.clamp(max_abs / 7.0, min=1e-8)

        quantized = torch.clamp(torch.round(blocks / scale), -8, 7).to(torch.int32)
        nibbles = (quantized + 8).to(torch.uint8)

        paired = nibbles.view(m, self.block_size // 2, 2)
        packed = ((paired[:, :, 0] << 4) | (paired[:, :, 1] & 0x0F)).flatten()
        scale_vec = scale.squeeze(-1)

        if self.use_error_feedback:
            decompressed = self._decompress_core(packed, scale_vec, orig_shape, orig_dtype)
            self.residual = (x - decompressed).detach()

        return packed, scale_vec, orig_shape, orig_dtype

    def _decompress_core(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        orig_shape: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        m = scale.numel()
        block_bytes = self.block_size // 2
        packed_blocks = packed.view(m, block_bytes)

        high = (packed_blocks >> 4) & 0x0F
        low = packed_blocks & 0x0F

        unpacked = torch.stack([high, low], dim=2).view(m, self.block_size)
        values = unpacked.to(torch.float32) - 8.0

        scaled = values * scale.view(m, 1)
        total_elements = orig_shape.numel()
        return scaled.flatten()[:total_elements].view(orig_shape).to(dtype)

    def decompress(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        orig_shape: torch.Size,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Unpack INT4 nibbles and rescale back to floating-point tensor."""
        return self._decompress_core(packed, scale, orig_shape, dtype)

    @classmethod
    def compress_tensor(
        cls,
        tensor: torch.Tensor,
        block_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
        """Stateless compression helper for individual tensors."""
        compressor = cls(block_size=block_size, use_error_feedback=False)
        return compressor.compress(tensor)

    @classmethod
    def decompress_tensor(
        cls,
        packed: torch.Tensor,
        scale: torch.Tensor,
        orig_shape: torch.Size,
        dtype: torch.dtype = torch.float32,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Stateless decompression helper for individual tensors."""
        compressor = cls(block_size=block_size, use_error_feedback=False)
        return compressor.decompress(packed, scale, orig_shape, dtype=dtype)

    @classmethod
    def compress_model(
        cls,
        model: nn.Module,
        block_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
        """Serialize and compress full model parameter vector."""
        flat_params = SerializationTool.serialize_model(model)
        return cls.compress_tensor(flat_params, block_size=block_size)

    @classmethod
    def deserialize_model(
        cls,
        model: nn.Module,
        packed: torch.Tensor,
        scale: torch.Tensor,
        orig_shape: torch.Size,
        dtype: torch.dtype = torch.float32,
        block_size: int = 64,
        mode: str = "copy",
    ) -> None:
        """Decompress parameter vector and deserialize into model."""
        decompressed = cls.decompress_tensor(
            packed, scale, orig_shape, dtype=dtype, block_size=block_size
        )
        SerializationTool.deserialize_model(model, decompressed, mode=mode)

    @classmethod
    def compress_model_gradients(
        cls,
        model: nn.Module,
        block_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
        """Serialize and compress model gradients."""
        flat_grads = SerializationTool.serialize_model_gradients(model)
        return cls.compress_tensor(flat_grads, block_size=block_size)

    @classmethod
    def deserialize_model_gradients(
        cls,
        model: nn.Module,
        packed: torch.Tensor,
        scale: torch.Tensor,
        orig_shape: torch.Size,
        dtype: torch.dtype = torch.float32,
        block_size: int = 64,
    ) -> None:
        """Decompress gradient vector and deserialize into model grad attributes."""
        decompressed = cls.decompress_tensor(
            packed, scale, orig_shape, dtype=dtype, block_size=block_size
        )
        SerializationTool.deserialize_model_gradients(model, decompressed)
