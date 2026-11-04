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

import unittest
import torch
import torch.nn as nn
from fedlab.utils.compressor import Int4BlockCompressor


class MLP(nn.Module):
    def __init__(self, in_features=64, hidden=32, out_features=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class CompressorTestCase(unittest.TestCase):
    def test_invalid_block_size(self):
        with self.assertRaises(ValueError):
            Int4BlockCompressor(block_size=63)
        with self.assertRaises(ValueError):
            Int4BlockCompressor(block_size=0)
        with self.assertRaises(ValueError):
            Int4BlockCompressor(block_size=-16)

    def test_shape_and_dtype_preservation(self):
        compressor = Int4BlockCompressor(block_size=64)

        for dtype in [torch.float32, torch.float16]:
            # Exact block multiple
            t1 = torch.randn(256, dtype=dtype)
            packed, scale, shape, orig_dtype = compressor.compress(t1)
            recon1 = compressor.decompress(packed, scale, shape, dtype=orig_dtype)
            self.assertEqual(recon1.shape, t1.shape)
            self.assertEqual(recon1.dtype, dtype)

            # Requires padding
            t2 = torch.randn(173, dtype=dtype)
            packed, scale, shape, orig_dtype = compressor.compress(t2)
            recon2 = compressor.decompress(packed, scale, shape, dtype=orig_dtype)
            self.assertEqual(recon2.shape, t2.shape)
            self.assertEqual(recon2.dtype, dtype)

            # Multi-dimensional
            t3 = torch.randn(2, 5, 17, dtype=dtype)
            packed, scale, shape, orig_dtype = compressor.compress(t3)
            recon3 = compressor.decompress(packed, scale, shape, dtype=orig_dtype)
            self.assertEqual(recon3.shape, t3.shape)
            self.assertEqual(recon3.dtype, dtype)

    def test_compression_ratio(self):
        n = 4096
        tensor = torch.randn(n, dtype=torch.float32)
        raw_bytes = tensor.numel() * tensor.element_size()

        packed, scale, _, _ = Int4BlockCompressor.compress_tensor(tensor, block_size=64)
        packed_bytes = packed.numel() * packed.element_size()
        scale_bytes = scale.numel() * scale.element_size()

        # 4 bits per element gives an exact 8x payload reduction
        self.assertEqual(packed_bytes, raw_bytes // 8)
        # With scales, total footprint is still >7x smaller than float32
        self.assertGreater(raw_bytes / (packed_bytes + scale_bytes), 7.0)

    def test_edge_tensors(self):
        compressor = Int4BlockCompressor(block_size=64)

        # All zeros
        zeros = torch.zeros(128)
        packed, scale, shape, dtype = compressor.compress(zeros)
        recon_zeros = compressor.decompress(packed, scale, shape, dtype)
        self.assertTrue(torch.allclose(recon_zeros, zeros, atol=1e-5))

        # Constant non-zero
        const = torch.full((128,), 3.5)
        packed, scale, shape, dtype = compressor.compress(const)
        recon_const = compressor.decompress(packed, scale, shape, dtype)
        self.assertTrue(torch.allclose(recon_const, const, atol=0.6))

    def test_reconstruction_fidelity(self):
        tensor = torch.randn(1024)
        packed, scale, shape, dtype = Int4BlockCompressor.compress_tensor(tensor, block_size=64)
        recon = Int4BlockCompressor.decompress_tensor(packed, scale, shape, dtype, block_size=64)

        mse = torch.mean((tensor - recon) ** 2).item()
        self.assertLess(mse, 0.03)

    def test_error_feedback_drift(self):
        compressor_ef = Int4BlockCompressor(block_size=64, use_error_feedback=True)
        compressor_raw = Int4BlockCompressor(block_size=64, use_error_feedback=False)

        torch.manual_seed(0)
        rounds = 25
        dim = 256

        target = torch.zeros(dim)
        accum_ef = torch.zeros(dim)
        accum_raw = torch.zeros(dim)

        for _ in range(rounds):
            update = torch.randn(dim) * 0.05
            target += update

            p_ef, s_ef, sh_ef, dt_ef = compressor_ef.compress(update)
            accum_ef += compressor_ef.decompress(p_ef, s_ef, sh_ef, dt_ef)

            p_raw, s_raw, sh_raw, dt_raw = compressor_raw.compress(update)
            accum_raw += compressor_raw.decompress(p_raw, s_raw, sh_raw, dt_raw)

        drift_ef = torch.norm(target - accum_ef).item()
        drift_raw = torch.norm(target - accum_raw).item()

        self.assertLess(drift_ef, drift_raw)

        compressor_ef.reset_error()
        self.assertIsNone(compressor_ef.residual)

    def test_model_gradients_pipeline(self):
        model = MLP(in_features=32, hidden=16, out_features=4)
        x = torch.randn(4, 32)
        model(x).sum().backward()

        packed, scale, shape, dtype = Int4BlockCompressor.compress_model_gradients(
            model, block_size=32
        )

        target = MLP(in_features=32, hidden=16, out_features=4)
        for p in target.parameters():
            p.grad = torch.zeros_like(p.data)

        Int4BlockCompressor.deserialize_model_gradients(
            target, packed, scale, shape, dtype=dtype, block_size=32
        )

        for p_src, p_dst in zip(model.parameters(), target.parameters()):
            self.assertIsNotNone(p_dst.grad)
            cos_sim = torch.cosine_similarity(
                p_src.grad.flatten().unsqueeze(0),
                p_dst.grad.flatten().unsqueeze(0),
            ).item()
            self.assertGreater(cos_sim, 0.95)

    def test_model_parameters_pipeline(self):
        model = MLP(in_features=32, hidden=16, out_features=4)
        packed, scale, shape, dtype = Int4BlockCompressor.compress_model(model, block_size=32)

        target = MLP(in_features=32, hidden=16, out_features=4)
        Int4BlockCompressor.deserialize_model(
            target, packed, scale, shape, dtype=dtype, block_size=32, mode="copy"
        )

        for p_src, p_dst in zip(model.parameters(), target.parameters()):
            diff = torch.norm(p_src.data - p_dst.data).item()
            norm = torch.norm(p_src.data).item()
            self.assertLess(diff / norm, 0.15)


if __name__ == "__main__":
    unittest.main()
