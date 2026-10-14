# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests verifying SOMALayer works correctly with PyTorch DataLoader under multiprocessing.

The specific concern is that Warp-based operations (LBS, rotation fitting) and Warp's own
initialization could fail in forked worker processes. Three patterns are exercised:
  1. Warp called only in main process (workers just load tensors)
  2. Lazy SOMALayer initialization inside worker __getitem__
  3. Per-worker SOMALayer init via worker_init_fn
  4. spawn multiprocessing context (fresh processes, no fork state)
"""

import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"

NUM_JOINTS = 77


def _assets_available():
    return ASSETS_DIR.is_dir() and (ASSETS_DIR / "SOMA_neutral.npz").is_file()


class SomaPoseDataset(Dataset):
    """Minimal dataset that returns pre-generated pose tensors.

    Workers only load CPU tensors — no Warp calls inside workers.
    The SOMALayer forward pass is run in the main process collation step.
    """

    def __init__(self, id_coeffs_dim, scale_dim, size=4):
        self.poses = torch.zeros(size, NUM_JOINTS, 3)
        self.identity_coeffs = torch.zeros(size, id_coeffs_dim)
        self.scale_params = torch.zeros(size, scale_dim)
        self.transl = torch.zeros(size, 3)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return (
            self.poses[idx],
            self.identity_coeffs[idx],
            self.scale_params[idx],
            self.transl[idx],
        )


class SomaDataset(Dataset):
    """Dataset that initializes SOMALayer once at construction."""

    def __init__(self, data_root, size=4):
        from soma import SOMALayer

        self.data_root = data_root
        self._layer = SOMALayer(
            data_root=self.data_root,
            device="cpu",
            identity_model_type="mhr",
            mode="warp",
        )
        im = self._layer.identity_model
        self.poses = torch.zeros(size, NUM_JOINTS, 3)
        self.identity_coeffs = torch.zeros(size, im.num_identity_coeffs)
        self.scale_params = torch.zeros(size, im.num_scale_params)
        self.transl = torch.zeros(size, 3)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx].unsqueeze(0)
        id_coeffs = self.identity_coeffs[idx].unsqueeze(0)
        scale = self.scale_params[idx].unsqueeze(0)
        transl = self.transl[idx].unsqueeze(0)
        with torch.no_grad():
            out = self._layer(pose, id_coeffs, scale, transl)
        return {
            "vertices": out["vertices"].squeeze(0),
            "joints": out["joints"].squeeze(0),
            "pose": pose.squeeze(0),
            "id_coeffs": id_coeffs.squeeze(0),
            "scale": scale.squeeze(0),
            "transl": transl.squeeze(0),
        }


class _LazySomaDataset(Dataset):
    """Dataset that initializes SOMALayer lazily inside __getitem__.

    This ensures Warp is initialized fresh in each worker process rather than
    inheriting state from a fork of the main process.
    """

    def __init__(self, data_root, id_coeffs_dim, scale_dim, size=4):
        self.data_root = data_root
        self.poses = torch.zeros(size, NUM_JOINTS, 3)
        self.identity_coeffs = torch.zeros(size, id_coeffs_dim)
        self.scale_params = torch.zeros(size, scale_dim)
        self.transl = torch.zeros(size, 3)
        self._layer = None

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        if self._layer is None:
            from soma import SOMALayer

            self._layer = SOMALayer(
                data_root=self.data_root,
                device="cpu",
                identity_model_type="mhr",
                mode="warp",
            )
        pose = self.poses[idx].unsqueeze(0)
        id_coeffs = self.identity_coeffs[idx].unsqueeze(0)
        scale = self.scale_params[idx].unsqueeze(0)
        transl = self.transl[idx].unsqueeze(0)
        with torch.no_grad():
            out = self._layer(pose, id_coeffs, scale, transl)
        return out["vertices"].squeeze(0), out["joints"].squeeze(0)


class _WorkerInitDataset(Dataset):
    """Dataset where SOMALayer is injected by worker_init_fn."""

    def __init__(self, id_coeffs_dim, scale_dim, size=4):
        self.poses = torch.zeros(size, NUM_JOINTS, 3)
        self.identity_coeffs = torch.zeros(size, id_coeffs_dim)
        self.scale_params = torch.zeros(size, scale_dim)
        self.transl = torch.zeros(size, 3)
        self._layer = None  # set by worker_init_fn

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx].unsqueeze(0)
        id_coeffs = self.identity_coeffs[idx].unsqueeze(0)
        scale = self.scale_params[idx].unsqueeze(0)
        transl = self.transl[idx].unsqueeze(0)
        with torch.no_grad():
            out = self._layer(pose, id_coeffs, scale, transl)
        return {
            "vertices": out["vertices"].squeeze(0),
            "joints": out["joints"].squeeze(0),
            "pose": pose.squeeze(0),
            "id_coeffs": id_coeffs.squeeze(0),
            "scale": scale.squeeze(0),
            "transl": transl.squeeze(0),
        }


def _soma_worker_init(worker_id):
    """worker_init_fn: initialize SOMALayer once per worker process."""
    info = torch.utils.data.get_worker_info()
    from soma import SOMALayer

    info.dataset._layer = SOMALayer(
        data_root=str(ASSETS_DIR),
        device="cpu",
        identity_model_type="mhr",
        mode="warp",
    )


class TestSomaLayerDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _assets_available():
            raise unittest.SkipTest(
                "Assets not found. Run `git lfs pull` to fetch SOMA_neutral.npz."
            )
        cls.data_root = str(ASSETS_DIR)
        # Query dims once so datasets don't hardcode them.
        layer = cls._make_layer_static(cls.data_root, "cpu")
        im = layer.identity_model
        cls.id_coeffs_dim = im.num_identity_coeffs
        cls.scale_dim = im.num_scale_params

    @staticmethod
    def _make_layer_static(data_root, device):
        from soma import SOMALayer

        return SOMALayer(
            data_root=data_root,
            device=device,
            identity_model_type="mhr",
            mode="warp",
        )

    def _make_layer(self, device="cpu"):
        return self._make_layer_static(self.data_root, device)

    def _assert_output_shapes(self, vertices, joints, batch_size, num_verts):
        self.assertEqual(vertices.dim(), 3)
        self.assertEqual(vertices.shape[0], batch_size)
        self.assertEqual(vertices.shape[2], 3)
        self.assertEqual(vertices.shape[1], num_verts)
        self.assertEqual(joints.shape, (batch_size, NUM_JOINTS, 3))

    def test_no_workers_baseline(self):
        """Sanity check: single-process DataLoader, Warp ops work correctly."""
        layer = self._make_layer("cpu")
        num_verts = layer.bind_shape.shape[0]
        dataset = SomaPoseDataset(self.id_coeffs_dim, self.scale_dim, size=4)
        loader = DataLoader(dataset, batch_size=2, num_workers=0)

        for poses, id_coeffs, scale_params, transl in loader:
            with torch.no_grad():
                out = layer(poses, id_coeffs, scale_params, transl)
            self.assertIn("vertices", out)
            self.assertIn("joints", out)
            self._assert_output_shapes(out["vertices"], out["joints"], 2, num_verts)

    def test_multi_worker_warp_in_main_process(self):
        """Safe pattern: workers only load tensors; Warp called only in the main process."""
        layer = self._make_layer("cpu")
        num_verts = layer.bind_shape.shape[0]
        dataset = SomaPoseDataset(self.id_coeffs_dim, self.scale_dim, size=4)
        loader = DataLoader(dataset, batch_size=2, num_workers=2)

        for poses, id_coeffs, scale_params, transl in loader:
            with torch.no_grad():
                out = layer(poses, id_coeffs, scale_params, transl)
            self.assertIn("vertices", out)
            self.assertIn("joints", out)
            self._assert_output_shapes(out["vertices"], out["joints"], 2, num_verts)

    def test_multi_worker_init_at_construction(self):
        """Warp is initialized fresh inside each forked worker via lazy SOMALayer init."""
        import multiprocessing

        if multiprocessing.get_start_method() != "fork":
            self.skipTest(
                "test requires fork-based DataLoader workers; "
                f"current start method is {multiprocessing.get_start_method()!r}"
            )
        dataset = SomaDataset(self.data_root, size=4)
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            device = "cuda:1"
        soma_layer = self._make_layer(device)

        with tempfile.TemporaryFile() as _tmp:
            _saved = os.dup(2)
            os.dup2(_tmp.fileno(), 2)
            try:
                for data in loader:
                    # move to device
                    for key, value in data.items():
                        data[key] = value.to(device)
                    vertices = data["vertices"]
                    joints = data["joints"]
                    batch_size = vertices.shape[0]
                    with torch.no_grad():
                        out = soma_layer(
                            data["pose"], data["id_coeffs"], data["scale"], data["transl"]
                        )
                        pred_joints = out["joints"]
                    diff_joints = (joints - pred_joints).abs().max()
                    self.assertLess(diff_joints, 1e-3)
                    self.assertEqual(vertices.dim(), 3)
                    self.assertEqual(vertices.shape[2], 3)
                    self.assertEqual(joints.shape, (batch_size, NUM_JOINTS, 3))
            finally:
                os.dup2(_saved, 2)
                os.close(_saved)
            _tmp.seek(0)
            _stderr = _tmp.read().decode("utf-8", errors="replace")

        self.assertNotIn(
            "Warp CUDA error 3",
            _stderr,
            "CUDA error 3 appeared in worker stderr — fork hook may not be working",
        )

    def test_multi_worker_lazy_init_in_worker(self):
        """Warp is initialized fresh inside each forked worker via lazy SOMALayer init."""
        dataset = _LazySomaDataset(self.data_root, self.id_coeffs_dim, self.scale_dim, size=4)
        # num_verts is unknown without a layer; just check dims
        loader = DataLoader(dataset, batch_size=2, num_workers=2)

        for vertices, joints in loader:
            batch_size = vertices.shape[0]
            self.assertEqual(vertices.dim(), 3)
            self.assertEqual(vertices.shape[2], 3)
            self.assertEqual(joints.shape[0], batch_size)
            self.assertEqual(joints.shape[1], NUM_JOINTS)
            self.assertEqual(joints.shape[2], 3)

    def test_multi_worker_worker_init_fn(self):
        """Recommended pattern: SOMALayer initialized once per worker via worker_init_fn."""
        dataset = _WorkerInitDataset(self.id_coeffs_dim, self.scale_dim, size=4)
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            worker_init_fn=_soma_worker_init,
        )
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            device = "cuda:1"
        soma_layer = self._make_layer(device)

        with tempfile.TemporaryFile() as _tmp:
            _saved = os.dup(2)
            os.dup2(_tmp.fileno(), 2)
            try:
                for data in loader:
                    # move to device
                    for key, value in data.items():
                        data[key] = value.to(device)
                    vertices = data["vertices"]
                    joints = data["joints"]
                    batch_size = vertices.shape[0]
                    with torch.no_grad():
                        out = soma_layer(
                            data["pose"], data["id_coeffs"], data["scale"], data["transl"]
                        )
                        pred_joints = out["joints"]
                    diff_joints = (joints - pred_joints).abs().max()
                    self.assertLess(diff_joints, 1e-3)
                    self.assertEqual(vertices.dim(), 3)
                    self.assertEqual(vertices.shape[2], 3)
                    self.assertEqual(joints.shape, (batch_size, NUM_JOINTS, 3))
            finally:
                os.dup2(_saved, 2)
                os.close(_saved)
            _tmp.seek(0)
            _stderr = _tmp.read().decode("utf-8", errors="replace")

        self.assertNotIn(
            "Warp CUDA error 3",
            _stderr,
            "CUDA error 3 appeared in worker stderr — fork hook may not be working",
        )

    def test_spawn_context(self):
        """spawn multiprocessing context: fresh processes, no fork state inheritance."""
        layer = self._make_layer("cpu")
        num_verts = layer.bind_shape.shape[0]
        dataset = SomaPoseDataset(self.id_coeffs_dim, self.scale_dim, size=4)
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            multiprocessing_context="spawn",
        )

        for poses, id_coeffs, scale_params, transl in loader:
            with torch.no_grad():
                out = layer(poses, id_coeffs, scale_params, transl)
            self.assertIn("vertices", out)
            self.assertIn("joints", out)
            self._assert_output_shapes(out["vertices"], out["joints"], 2, num_verts)

    def test_cuda_spawn_context(self):
        """CUDA-safe pattern: spawn context avoids CUDA fork issues. Skipped if no GPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        layer = self._make_layer("cuda")
        num_verts = layer.bind_shape.shape[0]
        dataset = SomaPoseDataset(self.id_coeffs_dim, self.scale_dim, size=4)
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            multiprocessing_context="spawn",
        )

        for poses, id_coeffs, scale_params, transl in loader:
            poses = poses.cuda()
            id_coeffs = id_coeffs.cuda()
            scale_params = scale_params.cuda()
            transl = transl.cuda()
            with torch.no_grad():
                out = layer(poses, id_coeffs, scale_params, transl)
            self.assertIn("vertices", out)
            self.assertIn("joints", out)
            self._assert_output_shapes(out["vertices"], out["joints"], 2, num_verts)


if __name__ == "__main__":
    unittest.main()
