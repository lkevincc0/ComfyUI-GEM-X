# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests: SOMALayer forward pass for each identity model (soma, mhr, anny, smpl, garment)
as in tools/demo_soma_vis.py. Parametrized over CPU and CUDA; CUDA is skipped when unavailable.
Fails if assets/SOMA_neutral.npz is not present (e.g. run `git lfs pull` after clone).
Optional models (smpl, anny) are skipped when their assets or dependencies are missing.
"""

from pathlib import Path

import pytest
import torch

# Repo root = parent of tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
CORE_ASSET = ASSETS_DIR / "SOMA_neutral.npz"


@pytest.fixture(scope="module")
def data_root():
    if not ASSETS_DIR.is_dir():
        pytest.fail(
            f"Assets directory not found: {ASSETS_DIR}. "
            "Clone the repo and run `git lfs pull` to fetch assets."
        )
    if not CORE_ASSET.is_file():
        pytest.fail(
            f"Required asset not found: {CORE_ASSET}. "
            "Run `git lfs pull` (or `git lfs pull assets/`) to fetch LFS-tracked files."
        )
    return str(ASSETS_DIR)


def _make_layer(data_root, identity_model_type, device, low_lod=False):
    """Create SOMALayer; return (layer, skip_reason). skip_reason is non-None to skip the test."""
    from soma import SOMALayer

    try:
        layer = SOMALayer(
            data_root=data_root,
            low_lod=low_lod,
            device=device,
            identity_model_type=identity_model_type,
            mode="warp",
        ).to(device)
        return layer, None
    except (FileNotFoundError, ImportError) as e:
        return None, f"Missing asset or dependency: {e}"


def _make_inputs(layer, identity_model_type, device, batch_size=1):
    """Build identity_coeffs and scale_params for the given identity model type."""
    if identity_model_type == "anny":
        ann = layer.identity_model.identity_model
        identity_coeffs = {
            k: torch.ones(batch_size, device=device) * 0.5 for k in ann.phenotype_labels
        }
        scale_params = {k: torch.zeros(batch_size, device=device) for k in ann.local_change_labels}
    elif identity_model_type == "mhr":
        n_id = layer.identity_model.num_identity_coeffs
        n_scale = layer.identity_model.num_scale_params
        identity_coeffs = torch.zeros(batch_size, n_id, device=device)
        scale_params = torch.zeros(batch_size, n_scale, device=device)
    else:
        n_id = layer.identity_model.num_identity_coeffs
        identity_coeffs = torch.zeros(batch_size, n_id, device=device)
        scale_params = None
    return identity_coeffs, scale_params


@pytest.mark.parametrize(
    "apply_correctives",
    [
        pytest.param(False, id="no_correctives"),
        pytest.param(True, id="correctives"),
    ],
)
@pytest.mark.parametrize(
    "low_lod",
    [
        pytest.param(False, id="high_lod"),
        pytest.param(True, id="low_lod"),
    ],
)
@pytest.mark.parametrize("identity_model_type", ["soma", "mhr", "anny", "smpl", "smplx", "garment"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_soma_layer_forward(data_root, identity_model_type, device, low_lod, apply_correctives):
    """SOMALayer forward pass for each identity model, LOD, and corrective mode."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    layer, skip_reason = _make_layer(data_root, identity_model_type, device, low_lod=low_lod)
    if skip_reason is not None:
        pytest.skip(skip_reason)
    if apply_correctives and layer.correctives_model is None:
        pytest.skip("Corrective model not available")

    if low_lod:
        assert layer.nv_lod_mid_to_low is not None
        expected_num_verts = layer.nv_lod_mid_to_low.shape[0]
        assert layer.bind_shape.shape[0] == expected_num_verts

    batch_size = 1
    num_pose_joints = 77
    pose = torch.zeros(batch_size, num_pose_joints, 3, 3, device=device)
    transl = torch.zeros(batch_size, 3, device=device)
    identity_coeffs, scale_params = _make_inputs(layer, identity_model_type, device, batch_size)

    with torch.no_grad():
        out = layer(
            pose,
            identity_coeffs,
            scale_params=scale_params,
            transl=transl,
            pose2rot=False,
            apply_correctives=apply_correctives,
        )

    assert "vertices" in out
    assert "joints" in out
    verts = out["vertices"]
    joints = out["joints"]
    assert verts.dim() == 3 and verts.shape[0] == batch_size and verts.shape[2] == 3
    if low_lod:
        assert verts.shape[1] == expected_num_verts, (
            f"Expected {expected_num_verts} low-LOD vertices, got {verts.shape[1]}"
        )
    assert joints.dim() == 3 and joints.shape[0] == batch_size and joints.shape[2] == 3
    assert joints.shape[1] == num_pose_joints
