# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mesh-level validation of PoseMirror_SOMA and PoseMirror_MHR.

For each mirror class, compares the parameter-mirrored mesh against a geometric
mesh mirror (flip X + mirror_vert_indices) in Nova topology, excluding facial
inner geometry.

Pose data paths are resolved exclusively from environment variables.
Tests skip gracefully when the variables are not set or the data is unavailable,
so CI and external users are unaffected.

Environment variables:
    SOMA_POSE_NPZ  Path to a single Nova pose .npz file (key: pose_local or transforms).
    SOMA_POSE_DIR  Directory of Nova pose .npz files (first file is used).
    MHR_POSE_NPZ   Path to a single MHR pose .npz file (key: pose_params).
    MHR_POSE_DIR    Directory of MHR pose .npz files (first file is used).

Usage:
    SOMA_POSE_NPZ="path/to/soma_pose.npz" MHR_POSE_NPZ="path/to/mhr_pose.npz" \
        pytest tests/test_pose_mirror.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
CORE_ASSET = ASSETS_DIR / "SOMA_neutral.npz"

MAX_FRAMES = 100
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Data-path resolution helpers
# ---------------------------------------------------------------------------


def _resolve_npz_path(env_single: str, env_dir: str) -> Path | None:
    """Return the first .npz found via the given environment variables.

    Checks *env_single* first (should point to a single file), then
    *env_dir* (should point to a directory; the first .npz is used).
    Returns ``None`` when neither variable is set or no file is found.
    """
    if val := os.environ.get(env_single):
        p = Path(val)
        return p if p.is_file() else None

    if val := os.environ.get(env_dir):
        search_dir = Path(val)
        if search_dir.is_dir():
            npzs = sorted(search_dir.glob("*.npz"))
            if npzs:
                return npzs[0]
    return None


def _load_mhr_param_names(npz_path: Path):
    """Extract the first 204 parameter names from parameter_transform.npz."""
    with np.load(npz_path, allow_pickle=False) as data:
        return list(data["parameter_names"][:204])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def soma_rig(device):
    """Load Nova rig data needed for mesh-level mirror comparison."""
    if not CORE_ASSET.is_file():
        pytest.skip(
            f"Core asset not found: {CORE_ASSET}. Run `git lfs pull` to fetch LFS-tracked files."
        )

    rig_data = np.load(CORE_ASSET, allow_pickle=False)

    if "mirror_vert_indices" not in rig_data:
        pytest.skip(
            "SOMA_neutral.npz does not contain 'mirror_vert_indices'. "
            "Either update the NPZ or point the test at an NPZ that includes it."
        )

    sw_sp = csc_matrix(
        (
            rig_data["skinning_weights_data"],
            rig_data["skinning_weights_indices"],
            rig_data["skinning_weights_indptr"],
        ),
        shape=rig_data["skinning_weights_shape"],
    ).todense()

    facial_inner = np.concatenate(
        [
            rig_data["segment_eye_bags"],
            rig_data["segment_mouth_bag"],
        ]
    )
    body_mask = np.ones(len(rig_data["bind_shape"]), dtype=bool)
    body_mask[facial_inner] = False

    return dict(
        joint_names=rig_data["joint_names"].tolist(),
        joint_parent_ids=torch.from_numpy(rig_data["joint_parent_ids"].astype(np.int32).copy()).to(
            device
        ),
        bind_pose_world=torch.from_numpy(rig_data["bind_pose_world"]).to(device),
        bind_shape=torch.from_numpy(rig_data["bind_shape"]).to(device),
        skinning_weights=torch.from_numpy(np.array(sw_sp)).to(device),
        mirror_vert_indices=rig_data["mirror_vert_indices"],
        body_mask=body_mask,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_soma_pose_mirror(device, soma_rig):
    """PoseMirror_SOMA: param-mirrored mesh should match geometric mesh mirror."""
    from soma.geometry.batched_skinning import BatchedSkinning
    from soma.geometry.rig_utils import (
        PoseMirror_SOMA,
        joint_local_to_world,
        joint_world_to_local,
    )

    npz_path = _resolve_npz_path("SOMA_POSE_NPZ", "SOMA_POSE_DIR")
    if npz_path is None:
        pytest.skip("Nova pose data not available (set SOMA_POSE_NPZ or SOMA_POSE_DIR)")

    soma_npz = np.load(npz_path)
    key = "pose_local" if "pose_local" in soma_npz else "transforms"
    pose_local_np = soma_npz[key].astype(np.float32)
    n_frames = min(pose_local_np.shape[0], MAX_FRAMES)
    pose_local = torch.from_numpy(pose_local_np[:n_frames]).to(device)

    joint_parent_ids = soma_rig["joint_parent_ids"]
    pose_world = joint_local_to_world(pose_local, joint_parent_ids)

    mirror = PoseMirror_SOMA(soma_rig["joint_names"])
    pose_mirror_world = mirror(pose_world)
    pose_mirror_local = joint_world_to_local(pose_mirror_world, joint_parent_ids)

    skinning = BatchedSkinning(
        joint_parent_ids,
        soma_rig["skinning_weights"],
        soma_rig["bind_pose_world"],
        soma_rig["bind_shape"],
        mode="warp",
    )

    verts_orig_list, verts_pm_list = [], []
    for s in range(0, n_frames, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n_frames)
        verts_orig_list.append(
            skinning.pose(pose_local[s:e, :, :3, :3], pose_local[s:e, 1, :3, 3]).cpu()
        )
        verts_pm_list.append(
            skinning.pose(pose_mirror_local[s:e, :, :3, :3], pose_mirror_local[s:e, 1, :3, 3]).cpu()
        )

    verts_orig = torch.cat(verts_orig_list)
    verts_param_mirror = torch.cat(verts_pm_list)

    mvi = soma_rig["mirror_vert_indices"]
    verts_mesh_mirror = verts_orig.clone()
    verts_mesh_mirror[..., 0] *= -1
    verts_mesh_mirror = verts_mesh_mirror[:, mvi]

    body_mask = soma_rig["body_mask"]
    err = (verts_param_mirror - verts_mesh_mirror).norm(dim=-1)[:, body_mask].numpy()
    mean_err = float(err.mean())
    p99_err = float(np.percentile(err, 99))

    print(f"\n  [SOMA mirror] frames={n_frames}  mean={mean_err:.6f}  p99={p99_err:.6f}")

    assert mean_err < 0.01, f"SOMA mirror mean error {mean_err:.6f} exceeds 0.01 threshold"
    assert p99_err < 0.02, f"SOMA mirror P99 error {p99_err:.6f} exceeds 0.02 threshold"


@torch.no_grad()
def test_mhr_pose_mirror(device, soma_rig):
    """PoseMirror_MHR: param-mirrored mesh should match geometric mesh mirror."""
    import trimesh

    from soma.geometry.barycentric_interp import BarycentricInterpolator
    from soma.geometry.rig_utils import PoseMirror_MHR

    npz_path = _resolve_npz_path("MHR_POSE_NPZ", "MHR_POSE_DIR")
    if npz_path is None:
        pytest.skip("MHR pose data not available (set MHR_POSE_NPZ or MHR_POSE_DIR)")

    pt_path = ASSETS_DIR / "MHR" / "parameter_transform.npz"
    model_path = ASSETS_DIR / "MHR" / "mhr_model_lod1.pt"
    mesh_mhr_path = ASSETS_DIR / "MHR" / "base_body_lod1.obj"
    mesh_soma_path = ASSETS_DIR / "MHR" / "SOMA_wrap_lod1.obj"
    for p in (pt_path, model_path, mesh_mhr_path, mesh_soma_path):
        if not p.is_file():
            pytest.skip(f"Required MHR asset not found: {p}")

    param_names = _load_mhr_param_names(pt_path)

    mhr_npz = np.load(npz_path)
    pp = torch.from_numpy(mhr_npz["pose_params"]).float().to(device)
    n_total = pp.shape[0]
    if pp.shape[1] < 204:
        pp = torch.cat([pp, torch.zeros(n_total, 204 - pp.shape[1], device=device)], 1)
    else:
        pp = pp[:, :204]
    n_frames = min(n_total, MAX_FRAMES)
    pp = pp[:n_frames]

    mirror = PoseMirror_MHR(param_names)
    pp_mirrored = mirror(pp)

    mhr_model = torch.jit.load(str(model_path), map_location=device)

    mesh_mhr = trimesh.load(str(mesh_mhr_path), maintain_order=True, process=False)
    mesh_soma = trimesh.load(str(mesh_soma_path), maintain_order=True, process=False)
    V_mhr = torch.from_numpy(mesh_mhr.vertices).float().to(device)
    F_mhr = torch.from_numpy(mesh_mhr.faces).to(device)
    V_soma = torch.from_numpy(mesh_soma.vertices).float().to(device)
    mhr_to_soma = BarycentricInterpolator(V_mhr, F_mhr, V_soma)

    id_c = torch.zeros(1, 45, device=device)
    fe = torch.zeros(1, 72, device=device)

    def mhr_forward(params_204):
        results = []
        for s in range(0, params_204.shape[0], BATCH_SIZE):
            e = min(s + BATCH_SIZE, params_204.shape[0])
            v, _ = mhr_model(
                id_c.expand(e - s, -1),
                params_204[s:e],
                fe.expand(e - s, -1),
            )
            results.append(mhr_to_soma(v).cpu())
            del v
        return torch.cat(results)

    verts_orig = mhr_forward(pp)
    verts_param_mirror = mhr_forward(pp_mirrored)

    mvi = soma_rig["mirror_vert_indices"]
    verts_mesh_mirror = verts_orig.clone()
    verts_mesh_mirror[..., 0] *= -1
    verts_mesh_mirror = verts_mesh_mirror[:, mvi]

    body_mask = soma_rig["body_mask"]
    err = (verts_param_mirror - verts_mesh_mirror).norm(dim=-1)[:, body_mask].numpy()
    mean_err = float(err.mean())
    p99_err = float(np.percentile(err, 99))

    print(f"\n  [MHR mirror] frames={n_frames}  mean={mean_err:.6f}  p99={p99_err:.6f}")

    assert mean_err < 0.005, f"MHR mirror mean error {mean_err:.6f} exceeds 0.005 threshold"
    assert p99_err < 0.01, f"MHR mirror P99 error {p99_err:.6f} exceeds 0.01 threshold"
