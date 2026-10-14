# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PoseInversion.

Tests both ``fit()`` (analytical Kabsch) and ``fit(autograd_iters=...)``
(FK-based gradient optimization) against ground-truth posed vertices
from example_animation.npy.

Pose conventions
~~~~~~~~~~~~~~~~
- example_animation.npy stores local rotations *relative to T-pose*
  (joint orient not applied).  demo_soma_vis.py applies a t-pose
  correction before passing to ``soma.pose(absolute_pose=False)``.

- Both ``fit()`` and ``fit(autograd_iters=...)`` return *absolute*
  local rotations (joint orient already baked in), suitable for
  ``soma.pose(absolute_pose=True)`` or direct LBS via
  ``BatchedSkinning.pose(absolute_pose=True)``.

Requires CUDA and assets/.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
MOTION_FILE = ASSETS_DIR / "example_animation.npy"

# 94-joint skeleton to 77-joint mapping (from demo_soma_vis.py)
# fmt: off
_NVSKEL93_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "LeftForeArmTwist1", "LeftForeArmTwist2", "LeftArmTwist1", "LeftArmTwist2",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "RightForeArmTwist1", "RightForeArmTwist2", "RightArmTwist1", "RightArmTwist2",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "LeftShinTwist1", "LeftShinTwist2", "LeftLegTwist1", "LeftLegTwist2",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
    "RightShinTwist1", "RightShinTwist2", "RightLegTwist1", "RightLegTwist2",
]
_NVSKEL77_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]
# fmt: on
_93TO77_IDX = [_NVSKEL93_NAMES.index(n) for n in _NVSKEL77_NAMES]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _load_motion(soma, frames):
    """Load example_animation.npy frames, return ground-truth posed vertices.

    Follows the same pipeline as tools/demo_soma_vis.py:
    1. Remap 94-joint → 78-joint (root + 77)
    2. Apply t-pose correction
    3. Forward pass through soma.pose()

    Returns (posed_vertices, root_translation).
    """
    from soma.geometry.rig_utils import joint_local_to_world, joint_world_to_local

    device = soma.device
    motion_full = torch.from_numpy(np.load(MOTION_FILE)).float().to(device)
    rot_local = motion_full[..., :3, :3]
    root_trans = motion_full[:, 1, :3, 3]

    # Remap 94 → 78 joints (root + 77)
    if rot_local.shape[1] == 94:
        subset_idx = [0] + [i + 1 for i in _93TO77_IDX]
        rot_local = rot_local[:, subset_idx]

    # T-pose correction: animation data is in a different skeleton
    # convention; rotate world transforms to match SOMA's joint orient.
    correction = soma.t_pose_world[:, :3, :3].transpose(-2, -1)
    rot_world = joint_local_to_world(rot_local, soma.joint_parent_ids)
    rot_world = rot_world @ correction
    rot_local = joint_world_to_local(rot_world, soma.joint_parent_ids)

    # Build pose: global_orient (Hips=joint 1) + body (joints 2:)
    global_orient = rot_local[:, 1]
    body_pose = rot_local[:, 2:]
    pose = torch.cat([global_orient.unsqueeze(1), body_pose], dim=1)
    transl = root_trans

    # Select frames
    pose = pose[frames]
    transl = transl[frames]

    # Forward pass — these rotations are relative to T-pose
    with torch.no_grad():
        out = soma.pose(pose, transl=transl, pose2rot=False, absolute_pose=False)

    return out["vertices"], transl


@pytest.fixture(scope="module")
def soma_and_inv():
    """Create SOMALayer + PoseInversion, prepare mean-shape identity."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not ASSETS_DIR.is_dir():
        pytest.fail(f"Assets directory not found: {ASSETS_DIR}")
    if not MOTION_FILE.is_file():
        pytest.fail(f"Motion file not found: {MOTION_FILE}")

    from soma.pose_inversion import PoseInversion
    from soma.soma import SOMALayer

    device = "cuda"
    soma = SOMALayer(
        data_root=str(ASSETS_DIR),
        identity_model_type="soma",
        device=device,
        mode="warp",
        low_lod=True,
    )

    # Prepare mean shape
    n_id = soma.identity_model.num_identity_coeffs
    identity_coeffs = torch.zeros(1, n_id, device=device)
    soma.prepare_identity(identity_coeffs)

    inv = PoseInversion(soma, low_lod=True)
    inv.prepare_identity(identity_coeffs)

    return soma, inv


@requires_cuda
class TestInvert:
    """Tests for PoseInversion.fit() (analytical Kabsch)."""

    def test_single_frame_roundtrip(self, soma_and_inv):
        """Single frame: fit recovers pose with low error."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0])

        result = inv.fit(verts, body_iters=10, finger_iters=2, full_iters=1)

        J = result["rotations"].shape[1]  # 78 (root + 77 joints)
        assert result["rotations"].shape == (1, J, 3, 3)
        assert result["root_translation"].shape == (1, 3)
        assert result["per_vertex_error"].shape[0] == 1

        mean_err = result["per_vertex_error"].mean().item()
        max_err = result["per_vertex_error"].max().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"
        assert max_err < 0.05, f"Max vertex error too high: {max_err:.6f} m"

    def test_batch_roundtrip(self, soma_and_inv):
        """Multiple diverse frames: consistent low error across batch."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0, 100, 300, 600])

        result = inv.fit(verts, body_iters=10, finger_iters=2, full_iters=1)

        J = result["rotations"].shape[1]
        assert result["rotations"].shape == (4, J, 3, 3)
        assert result["per_vertex_error"].shape[0] == 4

        mean_err = result["per_vertex_error"].mean().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"

    def test_roundtrip_forward_pass(self, soma_and_inv):
        """Verify inverted rotations reproduce vertices via soma.pose().

        fit returns absolute local rotations for 78 joints
        (root + 77).  Strip the root (index 0) and pass to
        soma.pose(absolute_pose=True) to reconstruct.
        """
        soma, inv = soma_and_inv
        verts_gt, _ = _load_motion(soma, frames=[50, 200])

        result = inv.fit(verts_gt, body_iters=10, finger_iters=2, full_iters=1)

        # Strip root joint (index 0) — soma.pose() expects 77 joints
        rotations_no_root = result["rotations"][:, 1:]
        # fit uses raw LBS without correctives, so disable
        # correctives in the forward pass for a fair comparison.
        with torch.no_grad():
            out = soma.pose(
                rotations_no_root,
                transl=result["root_translation"],
                pose2rot=False,
                absolute_pose=True,
                apply_correctives=False,
            )
        verts_recon = out["vertices"]

        err = torch.norm(verts_recon - verts_gt, dim=-1)
        mean_err = err.mean().item()
        # Slightly higher threshold than internal per_vertex_error because
        # soma.pose() uses full skinning weights while fit
        # uses sparse top-K weights internally.
        assert mean_err < 0.02, f"Forward-pass roundtrip error too high: {mean_err:.6f} m"

    def test_batch_size_chunking(self, soma_and_inv):
        """batch_size parameter produces comparable results to all-at-once."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0, 50, 100, 150])

        result_all = inv.fit(verts, body_iters=5, finger_iters=2)
        result_chunked = inv.fit(verts, body_iters=5, finger_iters=2, batch_size=2)

        assert result_chunked["rotations"].shape == result_all["rotations"].shape

        # Analytical is deterministic, so results should be very close
        err_all = result_all["per_vertex_error"].mean().item()
        err_chunked = result_chunked["per_vertex_error"].mean().item()
        assert abs(err_all - err_chunked) < 0.005, (
            f"Chunked vs all-at-once error mismatch: {err_all:.6f} vs {err_chunked:.6f}"
        )

    def test_identity_pose_near_zero_error(self, soma_and_inv):
        """Rest pose (identity rotations) should fit with near-zero error."""
        soma, inv = soma_and_inv
        device = soma.device

        J = 77
        rot_mats = torch.eye(3, device=device).expand(1, J, 3, 3).clone()

        transl = torch.zeros(1, 3, device=device)
        with torch.no_grad():
            out = soma.pose(rot_mats, transl=transl, pose2rot=False)
        verts = out["vertices"]

        result = inv.fit(verts, body_iters=5, finger_iters=2, full_iters=1)

        mean_err = result["per_vertex_error"].mean().item()
        assert mean_err < 0.02, f"Identity pose error too high: {mean_err:.6f} m"


@requires_cuda
class TestInvertAutogradFK:
    """Tests for PoseInversion.fit(autograd_iters=...)."""

    def test_single_frame_roundtrip(self, soma_and_inv):
        """Single frame: fit(autograd_iters) recovers pose with low error."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0])

        result = inv.fit(verts, body_iters=0, full_iters=0, autograd_iters=20, autograd_lr=5e-3)

        J = result["rotations"].shape[1]
        assert result["rotations"].shape == (1, J, 3, 3)
        assert result["root_translation"].shape == (1, 3)
        assert result["per_vertex_error"].shape[0] == 1

        mean_err = result["per_vertex_error"].mean().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"

    def test_batch_roundtrip(self, soma_and_inv):
        """Multiple diverse frames: consistent low error across batch."""
        soma, inv = soma_and_inv
        verts, _ = _load_motion(soma, frames=[0, 100, 300, 600])

        result = inv.fit(verts, body_iters=0, full_iters=0, autograd_iters=20, autograd_lr=5e-3)

        assert result["rotations"].shape[0] == 4
        mean_err = result["per_vertex_error"].mean().item()
        assert mean_err < 0.01, f"Mean vertex error too high: {mean_err:.6f} m"
