# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
from pathlib import Path

import torch

from soma import SomaLayer

ASSETS_DIR = Path("assets")


class TestNovaDevice(unittest.TestCase):
    def setUp(self):
        self.data_root = "assets"
        if not ASSETS_DIR.is_dir():
            self.skipTest("Assets not found, skipping device test")
        if not (ASSETS_DIR / "SOMA_neutral.npz").is_file():
            self.skipTest("Core asset SOMA_neutral.npz not found")

        # Dummy inputs (identity/scale shapes are set in _make_inputs after layer creation)
        self.batch_size = 1
        self.num_joints = 77  # Nova skeleton
        self.pose = torch.zeros(self.batch_size, self.num_joints, 3)
        self.transl = torch.zeros(self.batch_size, 3)

    def _make_inputs(self, layer):
        """Build identity_coeffs and scale_params from the layer's identity model."""
        im = layer.identity_model
        identity_coeffs = torch.zeros(self.batch_size, im.num_identity_coeffs)
        scale_params = None
        if im.num_scale_params is not None:
            scale_params = torch.zeros(self.batch_size, im.num_scale_params)
        return identity_coeffs, scale_params

    def test_cpu_initialization(self):
        """Test initializing on CPU."""
        device = "cpu"
        model = SomaLayer(data_root=self.data_root, device=device, identity_model_type="mhr")
        identity_coeffs, scale_params = self._make_inputs(model)

        # Check if parameters are on CPU
        for param in model.parameters():
            self.assertEqual(param.device.type, "cpu")

        # Check if buffers are on CPU
        for buffer in model.buffers():
            self.assertEqual(buffer.device.type, "cpu")

        # Run forward pass
        out = model(
            self.pose.to(device),
            identity_coeffs.to(device),
            scale_params.to(device),
            self.transl.to(device),
        )
        self.assertTrue("vertices" in out)

    def test_cuda0_only(self):
        target_device = "cuda:0"
        print(f"Initializing on {target_device}...")
        model = SomaLayer(data_root=self.data_root, device=target_device, identity_model_type="mhr")
        identity_coeffs, scale_params = self._make_inputs(model)

        out = model(
            self.pose.to(target_device),
            identity_coeffs.to(target_device),
            scale_params.to(target_device),
            self.transl.to(target_device),
        )
        self.assertTrue("vertices" in out)

    def test_cuda1_only(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("Need 2 GPUs")

        target_device = "cuda:1"
        print(f"Initializing on {target_device}...")
        model = SomaLayer(data_root=self.data_root, device=target_device, identity_model_type="mhr")
        identity_coeffs, scale_params = self._make_inputs(model)

        out = model(
            self.pose.to(target_device),
            identity_coeffs.to(target_device),
            scale_params.to(target_device),
            self.transl.to(target_device),
        )
        self.assertTrue("vertices" in out)

    def test_move_to_gpu(self):
        """Test moving model from CPU to GPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = "cpu"
        if torch.cuda.is_available():
            target_device = "cuda:0"

        if torch.cuda.device_count() > 1:
            target_device2 = "cuda:1"

        print(f"\nInitializing on {device}...")
        model = SomaLayer(data_root=self.data_root, device=device, identity_model_type="mhr")
        identity_coeffs, scale_params = self._make_inputs(model)

        print(f"Moving to {target_device}...")
        model.to(target_device)

        # Check if parameters moved
        print("Checking parameters...")
        for name, param in model.named_parameters():
            self.assertEqual(param.device.type, "cuda", f"Parameter {name} is not on CUDA")

        # Check if buffers moved
        print("Checking buffers...")
        for name, buffer in model.named_buffers():
            self.assertEqual(buffer.device.type, "cuda", f"Buffer {name} is not on CUDA")

        # CRITICAL: Check if the internal device attribute matches
        # This is where we expect bad design to fail or be inconsistent
        print(f"Model internal device attribute: {model.device}")

        # Run forward pass on GPU
        print("Running forward pass on GPU...")
        try:
            out = model(
                self.pose.to(target_device),
                identity_coeffs.to(target_device),
                scale_params.to(target_device),
                self.transl.to(target_device),
            )
            self.assertTrue("vertices" in out)
            self.assertEqual(out["vertices"].device.type, "cuda")
        except RuntimeError as e:
            self.fail(f"Forward pass failed after moving to GPU: {e}")

        if torch.cuda.device_count() > 1:
            print(f"Moving to {target_device2}...")
            model.to(target_device2)
            out = model(
                self.pose.to(target_device2),
                identity_coeffs.to(target_device2),
                scale_params.to(target_device2),
                self.transl.to(target_device2),
            )

    def test_gpu_to_cpu_roundtrip(self):
        """Simulate DDP teardown: model moved from GPU back to CPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        model = SomaLayer(data_root=self.data_root, device="cpu", identity_model_type="mhr")
        identity_coeffs, scale_params = self._make_inputs(model)
        model.to("cuda:0")
        try:
            model.cpu()
        except RuntimeError as e:
            self.fail(f"Moving SomaLayer from GPU to CPU failed (DDP teardown): {e}")
        # Verify forward pass on CPU still works
        out = model(self.pose, identity_coeffs, scale_params, self.transl)
        self.assertTrue("vertices" in out)


class TestSkeletonTransferDevice(unittest.TestCase):
    """Unit tests for SkeletonTransfer device transfer — no assets required."""

    def _make_skeleton_transfer(self, device="cpu"):
        from soma.geometry.skeleton_transfer import SkeletonTransfer

        J, V = 5, 20
        # Pass joint_parent_ids as a tensor to exercise the device-mismatch bug.
        joint_parent_ids = torch.tensor([0, 0, 1, 2, 3])
        bind_world_transforms = torch.eye(4).unsqueeze(0).repeat(J, 1, 1)
        bind_shape = torch.randn(V, 3)
        skinning_weights = torch.rand(V, J)
        skinning_weights /= skinning_weights.sum(dim=1, keepdim=True)
        return SkeletonTransfer(
            joint_parent_ids.to(device),
            bind_world_transforms.to(device),
            bind_shape.to(device),
            skinning_weights.to(device),
            use_warp_for_rotations=False,
            use_sparse_rbf_matrix=False,
        )

    def test_init_with_tensor_joint_parent_ids(self):
        """joint_parent_ids passed as a CPU tensor must not cause device errors."""
        st = self._make_skeleton_transfer("cpu")
        self.assertIsNotNone(st.regressor_mask)

    def test_gpu_to_cpu_roundtrip(self):
        """Simulates DDP teardown: SkeletonTransfer on GPU moved back to CPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        st = self._make_skeleton_transfer("cpu")
        st.to("cuda")
        try:
            st.cpu()
        except RuntimeError as e:
            self.fail(f"Moving SkeletonTransfer from GPU to CPU failed: {e}")

    def test_cpu_to_gpu_roundtrip(self):
        """Moving from CPU to GPU and back must leave all buffers on CPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        st = self._make_skeleton_transfer("cpu")
        st.cuda()
        st.cpu()
        for name, buf in st.named_buffers():
            if buf is not None:
                self.assertEqual(
                    buf.device.type, "cpu", f"Buffer {name} not on CPU after round-trip"
                )


def _ddp_worker(rank, world_size, data_root, broadcast_buffers):
    """Worker run in each DDP subprocess via mp.spawn."""
    # Must be set before CUDA / Warp initialize in the child process
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    import warp as wp

    wp.config.enable_mempools_at_init = False

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from soma import SomaLayer

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device(f"cuda:{rank}")
        soma = SomaLayer(data_root=data_root, device=device, identity_model_type="mhr")

        # Wrap in a model that has learnable params (more realistic DDP use-case)
        class _HeadModel(torch.nn.Module):
            def __init__(self, soma_layer):
                super().__init__()
                self.soma = soma_layer
                # Tiny learnable head so DDP has gradients to synchronize
                self.head = torch.nn.Linear(3, 1)

            def forward(self, pose, id_coeffs, scale_params, transl):
                out = self.soma(pose, id_coeffs, scale_params=scale_params, transl=transl)
                return self.head(out["vertices"].mean(dim=1))  # (B, 1)

        model = _HeadModel(soma).to(device)
        ddp_model = DDP(model, device_ids=[rank], broadcast_buffers=broadcast_buffers)

        batch_size = 1
        num_joints = 77
        im = soma.identity_model
        pose = torch.zeros(batch_size, num_joints, 3, device=device)
        id_coeffs = torch.zeros(batch_size, im.num_identity_coeffs, device=device)
        scale_params = torch.zeros(batch_size, im.num_scale_params, device=device)
        transl = torch.zeros(batch_size, 3, device=device)

        out = ddp_model(pose, id_coeffs, scale_params, transl)
        loss = out.sum()
        loss.backward()  # syncs head gradients via DDP
    finally:
        dist.destroy_process_group()


class TestDDPCompatibility(unittest.TestCase):
    """Verify SomaLayer wrapped in DDP does not crash due to sparse tensor buffers."""

    @classmethod
    def setUpClass(cls):
        if sys.platform != "linux":
            raise unittest.SkipTest("DDP/NCCL tests require Linux")
        if not ASSETS_DIR.is_dir():
            raise unittest.SkipTest("Assets not found")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        cls.data_root = "assets"
        cls.world_size = min(2, torch.cuda.device_count())

    def _run_ddp(self, broadcast_buffers):
        import torch.multiprocessing as mp

        mp.spawn(
            _ddp_worker,
            args=(self.world_size, self.data_root, broadcast_buffers),
            nprocs=self.world_size,
            join=True,
        )

    def test_ddp_broadcast_buffers_false(self):
        """DDP with broadcast_buffers=False: sparse tensor buffers not broadcast → should pass."""
        self._run_ddp(broadcast_buffers=False)

    def test_ddp_broadcast_buffers_true(self):
        """DDP with broadcast_buffers=True (default): sparse buffers are plain attrs → should pass."""
        self._run_ddp(broadcast_buffers=True)


if __name__ == "__main__":
    unittest.main()
