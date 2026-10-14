# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SMPL to SOMA pose converter.

Converts SMPL posed meshes to SOMA skeleton parameters using
PoseInversion.fit() — analytical iterative inverse-LBS Newton-Schulz
refinement, optionally followed by autograd FK optimization.

Usage:
    python -m tools.smpl2soma
    python -m tools.smpl2soma --body-iters 3 --full-iters 1 --batch-size 64
    python -m tools.smpl2soma --autograd-iters 10  # analytical + autograd
    python -m tools.smpl2soma --no-render
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import smplx
import torch

from soma.geometry.rig_utils import remove_joint_orient_local
from soma.geometry.transforms import matrix_to_rotvec, rotation_6d_to_matrix
from soma.io import add_npz_args, save_soma_npz
from soma.pose_inversion import PoseInversion
from soma.soma import SOMALayer
from soma.units import Unit
from tools.vis_pyrender import (
    default_pyopengl_platform,
    render_comparison_video,
    set_pyopengl_platform,
)

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

set_pyopengl_platform(default_pyopengl_platform())


def main():
    parser = argparse.ArgumentParser(description="SMPL to SOMA pose converter.")
    parser.add_argument(
        "--body-iters", type=int, default=2, help="Analytical body iterations (default: 2)."
    )
    parser.add_argument(
        "--finger-iters", type=int, default=0, help="Analytical finger iterations (default: 0)."
    )
    parser.add_argument(
        "--full-iters", type=int, default=1, help="Analytical full iterations (default: 1)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process frames in chunks of this size (default: all at once).",
    )
    parser.add_argument(
        "--subsample", type=int, default=4, help="Frame subsampling factor (default: 4)."
    )
    parser.add_argument(
        "--autograd-iters",
        type=int,
        default=0,
        help="Autograd FK refinement iterations after analytical solve (default: 0 = off).",
    )
    parser.add_argument(
        "--autograd-lr",
        type=float,
        default=5e-3,
        help="Learning rate for autograd FK (default: 5e-3).",
    )
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")
    add_npz_args(parser)
    args = parser.parse_args()

    device = "cuda"
    data_root = repo_root / "assets"

    # --- Load SMPL animation ---
    data_path = data_root / "SMPL" / "smpl_anim.npy"
    smpl_rot_mats = np.load(data_path, allow_pickle=True).item()

    body_pose_6d = torch.from_numpy(smpl_rot_mats["body_pose_6d"]).float().to(device)
    transl = torch.from_numpy(smpl_rot_mats["transl"]).float().to(device)
    global_orient_6d = torch.from_numpy(smpl_rot_mats["global_orient_6d"]).float().to(device)
    betas = torch.from_numpy(smpl_rot_mats["betas"]).float().to(device)

    body_pose = matrix_to_rotvec(rotation_6d_to_matrix(body_pose_6d))
    global_orient = matrix_to_rotvec(rotation_6d_to_matrix(global_orient_6d))

    seq_len = body_pose.shape[0]
    idx = np.arange(0, seq_len, args.subsample)
    body_pose = body_pose[idx]
    global_orient = global_orient[idx]
    betas = betas[idx]
    transl = transl[idx]
    num_frames = len(idx)
    print(f"Loaded {num_frames} frames (subsampled {args.subsample}x from {seq_len})")

    # --- Set up SMPL model ---
    smpl_model = smplx.create(
        model_type="smpl",
        model_path=data_root / "SMPL" / "SMPL_NEUTRAL.pkl",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)
    smpl_faces = smpl_model.faces

    # --- Set up SOMA + PoseInversion ---
    soma = SOMALayer(
        data_root,
        identity_model_type="smpl",
        device=device,
        mode="warp",
    )
    inv = PoseInversion(soma, low_lod=True)
    inv.prepare_identity(betas[:1])

    # --- Fused SMPL forward + inversion (chunked to bound memory) ---
    batch_size = args.batch_size or num_frames
    parts = [
        f"body={args.body_iters}, finger={args.finger_iters}, full={args.full_iters}",
    ]
    if args.autograd_iters > 0:
        parts.append(f"autograd={args.autograd_iters}, lr={args.autograd_lr}")
    if args.batch_size:
        parts.append(f"batch_size={batch_size}")
    print(f"\nInverting ({', '.join(parts)})...")

    # Warmup
    with torch.no_grad():
        warmup_out = smpl_model(
            body_pose=body_pose[:1],
            global_orient=global_orient[:1],
            betas=betas[:1],
            transl=transl[:1],
        )
    inv.fit(
        warmup_out.vertices,
        body_iters=args.body_iters,
        finger_iters=args.finger_iters,
        full_iters=args.full_iters,
        autograd_iters=args.autograd_iters,
        autograd_lr=args.autograd_lr,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    all_rotations = []
    all_root_transl = []
    all_errors = []

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        with torch.no_grad():
            smpl_out = smpl_model(
                body_pose=body_pose[start:end],
                global_orient=global_orient[start:end],
                betas=betas[start:end],
                transl=transl[start:end],
            )
        result = inv.fit(
            smpl_out.vertices,
            body_iters=args.body_iters,
            finger_iters=args.finger_iters,
            full_iters=args.full_iters,
            autograd_iters=args.autograd_iters,
            autograd_lr=args.autograd_lr,
        )
        all_rotations.append(result["rotations"].cpu())
        all_root_transl.append(result["root_translation"].cpu())
        all_errors.append(result["per_vertex_error"].cpu())

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    rotations = torch.cat(all_rotations, dim=0)
    root_transl = torch.cat(all_root_transl, dim=0)
    err = torch.cat(all_errors, dim=0)

    print(f"  Time: {dt:.3f}s ({num_frames / dt:.0f} fps)")
    print(f"  Mean vertex error: {err.mean():.6f} m")
    print(f"  Max vertex error:  {err.max():.6f} m")

    # --- Save NPZ if requested ---
    if args.output_npz:
        _soma = inv.soma
        # Convert absolute → relative, matrix → rotvec
        rel_rotations = remove_joint_orient_local(
            rotations, _soma._t_pose_orient, _soma._t_pose_orient_parent_T
        )
        poses_rotvec = matrix_to_rotvec(rel_rotations.reshape(-1, 3, 3)).reshape(
            rotations.shape[0], rotations.shape[1], 3
        )

        save_transl = root_transl.clone()
        target_unit = Unit.from_name(args.output_unit)
        unit_scale = _soma.output_unit.meters_per_unit / target_unit.meters_per_unit
        if unit_scale != 1.0:
            save_transl = save_transl * unit_scale

        save_soma_npz(
            args.output_npz,
            poses_rotvec,
            save_transl,
            joint_names=list(_soma.rig_data["joint_names"]),
            identity_model_type=_soma.identity_model_type,
            identity_coeffs=betas[:1],
            joint_orient=_soma._t_pose_orient,
            unit=args.output_unit,
            keep_root=args.keep_root,
        )

    if args.no_render:
        return

    # --- Render: re-run SMPL forward + SOMA reconstruct in chunks ---
    _soma = inv.soma
    bs = _soma.batched_skinning
    bind_transforms = _soma._cached_bind_transforms_world
    rest_shape = _soma._cached_rest_shape

    smpl_verts_all = []
    soma_verts_all = []

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        with torch.no_grad():
            smpl_out = smpl_model(
                body_pose=body_pose[start:end],
                global_orient=global_orient[start:end],
                betas=betas[start:end],
                transl=transl[start:end],
            )
        smpl_verts_all.append(smpl_out.vertices.cpu().numpy())

        chunk_bind = bind_transforms.expand(end - start, -1, -1, -1)
        chunk_rest = rest_shape.expand(end - start, -1, -1)
        bs.rebind(chunk_bind, chunk_rest)
        with torch.no_grad():
            sv, _ = bs.pose(
                rotations[start:end].to(device),
                root_transl[start:end].to(device),
                absolute_pose=True,
                return_transforms=True,
            )
        soma_verts_all.append(sv.detach().cpu().numpy())

    print("\nRendering comparison video...")
    render_comparison_video(
        "out/smpl2soma_test.mp4",
        np.concatenate(smpl_verts_all, axis=0),
        smpl_faces,
        np.concatenate(soma_verts_all, axis=0),
        _soma.faces.cpu().numpy(),
        label_source="SMPL",
        cam_dist_scale=3.0,
    )


if __name__ == "__main__":
    main()
