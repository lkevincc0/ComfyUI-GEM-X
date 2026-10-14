# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MHR to SOMA pose converter.

Reads SAM 3D Body parquet files containing MHR parameters
(shape_params, model_params) and converts them to SOMA skeleton
parameters using PoseInversion.

SAM 3D Body model_params layout (204 floats):
  [0:3]   = global translation (cm)
  [3:136] = pose parameters (axis-angle)
  [136:204] = body-part scale parameters (68)

Usage:
    python -m tools.mhr2soma --input ../nvhuman/data/sam_3d_body/data/coco_train
    python -m tools.mhr2soma --input ../nvhuman/data/sam_3d_body/data/coco_train --output-npz out/coco_soma.npz
    python -m tools.mhr2soma --input ../nvhuman/data/sam_3d_body/data/coco_train/000000.parquet --max-samples 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from soma.geometry.rig_utils import remove_joint_orient_local  # noqa: E402
from soma.geometry.transforms import matrix_to_rotvec  # noqa: E402
from soma.io import add_npz_args, save_soma_npz  # noqa: E402
from soma.pose_inversion import PoseInversion  # noqa: E402
from soma.soma import SOMALayer  # noqa: E402
from soma.units import Unit  # noqa: E402


def load_sam_parquet(path, max_samples=None):
    """Load MHR parameters from SAM 3D Body parquet file(s).

    Args:
        path: Path to a single .parquet file or a directory containing them.
        max_samples: Maximum number of samples to load (None = all).

    Returns:
        dict with shape_params (N, 45), model_params (N, 204), and metadata.
    """
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {path}")
        dfs = []
        n = 0
        for f in files:
            df = pd.read_parquet(f)
            df = df[df["mhr_valid"]]
            dfs.append(df)
            n += len(df)
            if max_samples is not None and n >= max_samples:
                break
        df = pd.concat(dfs, ignore_index=True)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df = df[df["mhr_valid"]]
    else:
        raise ValueError(f"Expected .parquet file or directory, got: {path}")

    if max_samples is not None:
        df = df.iloc[:max_samples]

    shape_params = np.stack(df["shape_params"].values).astype(np.float32)
    model_params = np.stack(df["model_params"].values).astype(np.float32)

    print(f"Loaded {len(df)} MHR samples from {path}")
    return {
        "shape_params": shape_params,
        "model_params": model_params,
        "datasets": df["dataset"].values if "dataset" in df.columns else None,
        "images": df["image"].values if "image" in df.columns else None,
    }


def parse_mhr_model_params(model_params):
    """Parse model_params (N, 204) into translation, pose, and scale.

    Returns:
        translation: (N, 3) in centimeters
        pose_params: (N, 133) axis-angle body pose (excluding global translation)
        scale_params: (N, 68) body-part scales
    """
    translation = model_params[:, :3]
    pose_params = model_params[:, 3:136]
    scale_params = model_params[:, 136:]
    return translation, pose_params, scale_params


def get_mhr_posed_vertices(mhr_jit, identity_coeffs, model_params, device, batch_size=64):
    """Run MHR forward pass to get posed vertices.

    Args:
        mhr_jit: TorchScript MHR model.
        identity_coeffs: (N, 45) shape parameters.
        model_params: (N, 204) full model parameters.
        device: torch device.
        batch_size: process in chunks.

    Returns:
        (N, V, 3) posed vertices in centimeters.
    """
    N = identity_coeffs.shape[0]
    face_expr = torch.zeros(1, 72, device=device)
    all_verts = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ic = identity_coeffs[start:end].to(device)
        mp = model_params[start:end].to(device)
        fe = face_expr.expand(end - start, -1)
        with torch.no_grad():
            verts, _ = mhr_jit(ic, mp, fe)
        all_verts.append(verts.cpu())

    return torch.cat(all_verts, dim=0)


def convert_mhr_to_soma(
    posed_vertices,
    inv,
    body_iters=2,
    finger_iters=0,
    full_iters=0,
    autograd_iters=0,
    autograd_lr=5e-3,
    leaf_weight=1.0,
    batch_size=64,
):
    """Invert MHR posed vertices to SOMA rotations.

    Args:
        posed_vertices: (N, V, 3) MHR vertices in the same unit as
            the PoseInversion's SOMALayer output_unit.
        inv: PoseInversion instance (already prepared).
        body_iters: analytical body chain iterations.
        finger_iters: analytical finger chain iterations.
        full_iters: analytical full chain iterations.
        autograd_iters: Adam optimization steps through FK + LBS.
        autograd_lr: learning rate for autograd Adam.
        leaf_weight: extremity vertex weight.  Float or dict, e.g.
            ``{"head": 2, "hands": 2, "feet": 5}``.
        batch_size: process in chunks.

    Returns:
        dict with rotations (N, J, 3, 3), root_translation (N, 3),
        per_vertex_error (N, V).
    """
    return inv.fit(
        posed_vertices.to(inv.soma.device),
        body_iters=body_iters,
        finger_iters=finger_iters,
        full_iters=full_iters,
        autograd_iters=autograd_iters,
        autograd_lr=autograd_lr,
        leaf_weight=leaf_weight,
        batch_size=batch_size,
    )


def main():
    parser = argparse.ArgumentParser(description="MHR to SOMA pose converter.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to SAM 3D Body .parquet file or directory of parquet files. Download them from https://huggingface.co/datasets/facebook/sam-3d-body-dataset",
    )
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")
    add_npz_args(parser)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to SOMA assets (default: <repo>/assets).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )
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
        "--autograd-iters",
        type=int,
        default=0,
        help="Autograd FK optimization steps after analytical solve (default: 0 = analytical only).",
    )
    parser.add_argument(
        "--autograd-lr", type=float, default=5e-3, help="Autograd learning rate (default: 5e-3)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing (default: 64).",
    )
    parser.add_argument(
        "--leaf-weight",
        type=float,
        default=1.0,
        help="Uniform extremity vertex weight (default: 1.0 = no upweight).",
    )
    parser.add_argument(
        "--foot-weight",
        type=float,
        default=None,
        help="Override foot vertex weight (default: same as --leaf-weight).",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda).")
    parser.add_argument("--fps", type=int, default=4, help="Video frame rate (default: 4).")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else repo_root / "assets"
    device = args.device

    # --- Load data ---
    sam_data = load_sam_parquet(args.input, max_samples=args.max_samples)
    shape_params = sam_data["shape_params"]
    model_params_raw = sam_data["model_params"]
    N = shape_params.shape[0]

    # Zero out MHR's 6 flexible bone-length parameters (last 6 of 133 pose params).
    # These modify bone lengths and are not representable in the SOMA skeleton.
    model_params_raw[:, 130:136] = 0.0

    translation_cm, pose_params, scale_params = parse_mhr_model_params(model_params_raw)
    print(f"  Samples: {N}")
    print(f"  Translation range (cm): [{translation_cm.min():.1f}, {translation_cm.max():.1f}]")
    print(f"  Scale params range: [{scale_params.min():.3f}, {scale_params.max():.3f}]")

    # --- Set up MHR model ---
    import trimesh

    mhr_faces = trimesh.load(
        data_root / "MHR" / "base_body_lod1.obj", maintain_order=True, process=False
    ).faces
    mhr_jit = torch.jit.load(data_root / "MHR" / "mhr_model_lod1.pt", map_location=device)
    identity_coeffs_t = torch.from_numpy(shape_params).float()
    model_params_t = torch.from_numpy(model_params_raw).float()
    face_expr = torch.zeros(1, 72, device=device)

    # --- Set up SOMA + PoseInversion ---
    # Use output_unit=CENTIMETERS so the internal rest shape matches MHR's
    # native unit — no ad-hoc unit conversion needed.
    print("\nInitializing SOMA layer...")
    soma = SOMALayer(
        data_root,
        identity_model_type="mhr",
        device=device,
        mode="warp",
        output_unit=Unit.CENTIMETERS,
    )

    all_ic = torch.from_numpy(shape_params).float().to(device)
    all_sp = torch.from_numpy(scale_params).float().to(device)

    # Use low LOD for inversion (faster, negligible accuracy loss),
    # high LOD (soma) for rendering/evaluation.
    inv = PoseInversion(soma, low_lod=True)

    # Build leaf_weight: uniform or per-group if --foot-weight is set
    if args.foot_weight is not None:
        leaf_weight = {
            "head": args.leaf_weight,
            "hands": args.leaf_weight,
            "feet": args.foot_weight,
        }
    else:
        leaf_weight = args.leaf_weight

    # --- Fused MHR forward + inversion (chunked to bound memory) ---
    import time

    parts = []
    if args.body_iters > 0 or args.finger_iters > 0 or args.full_iters > 0:
        parts.append(
            f"analytical (body={args.body_iters}, finger={args.finger_iters}, full={args.full_iters})"
        )
    if args.autograd_iters > 0:
        parts.append(f"autograd FK ({args.autograd_iters} iters, lr={args.autograd_lr})")
    method_desc = " + ".join(parts) if parts else "none"
    if leaf_weight != 1.0:
        method_desc += f", leaf_weight={leaf_weight}"

    batch_size = args.batch_size
    print(f"\nInverting {N} samples with {method_desc}...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    all_rotations = []
    all_root_transl = []
    all_errors = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # Prepare identity for this chunk
        inv.prepare_identity(all_ic[start:end], all_sp[start:end])

        # MHR forward pass for this chunk
        ic = identity_coeffs_t[start:end].to(device)
        mp = model_params_t[start:end].to(device)
        fe = face_expr.expand(end - start, -1)
        with torch.no_grad():
            verts_cm, _ = mhr_jit(ic, mp, fe)

        # Invert this chunk (no further chunking — already chunk-sized)
        result = convert_mhr_to_soma(
            verts_cm,
            inv,
            body_iters=args.body_iters,
            finger_iters=args.finger_iters,
            full_iters=args.full_iters,
            autograd_iters=args.autograd_iters,
            autograd_lr=args.autograd_lr,
            leaf_weight=leaf_weight,
            batch_size=None,
        )

        all_rotations.append(result["rotations"].cpu())
        all_root_transl.append(result["root_translation"].cpu())
        all_errors.append(result["per_vertex_error"].cpu())

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    rotations = torch.cat(all_rotations, dim=0)
    root_transl = torch.cat(all_root_transl, dim=0)
    err = torch.cat(all_errors, dim=0)

    unit_label = "cm" if soma.output_unit == Unit.CENTIMETERS else "m"
    print(f"  Inversion time: {dt:.2f}s ({N / dt:.0f} FPS)")
    print(f"  Mean vertex error: {err.mean():.6f} {unit_label}")
    print(f"  Max vertex error:  {err.max():.6f} {unit_label}")
    print(f"  Median vertex error: {err.median():.6f} {unit_label}")

    # --- Save output ---
    if args.output_npz:
        # Convert absolute → relative, matrix → rotvec
        rel_rotations = remove_joint_orient_local(
            rotations, soma._t_pose_orient, soma._t_pose_orient_parent_T
        )
        poses_rotvec = matrix_to_rotvec(rel_rotations.reshape(-1, 3, 3)).reshape(
            rotations.shape[0], rotations.shape[1], 3
        )

        save_transl = root_transl.clone()
        target_unit = Unit.from_name(args.output_unit)
        unit_scale = soma.output_unit.meters_per_unit / target_unit.meters_per_unit
        if unit_scale != 1.0:
            save_transl = save_transl * unit_scale

        save_soma_npz(
            args.output_npz,
            poses_rotvec,
            save_transl,
            joint_names=list(soma.rig_data["joint_names"]),
            identity_model_type=soma.identity_model_type,
            identity_coeffs=shape_params,
            scale_params=scale_params,
            joint_orient=soma._t_pose_orient,
            unit=args.output_unit,
            keep_root=args.keep_root,
        )

    if args.no_render:
        return

    # --- Render ---
    from tools.vis_pyrender import (  # noqa: E402
        default_pyopengl_platform,
        render_comparison_video,
        set_pyopengl_platform,
    )

    set_pyopengl_platform(default_pyopengl_platform())

    cm_to_m = Unit.CENTIMETERS.meters_per_unit
    soma_faces = soma.faces.cpu().numpy()

    # Re-run MHR forward + SOMA reconstruct in chunks for rendering
    # (avoids materializing all vertices at once).
    mhr_verts_all = []
    eval_verts_all = []
    bs = soma.batched_skinning

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # Prepare identity for this chunk (high LOD)
        soma.prepare_identity(all_ic[start:end], all_sp[start:end])
        chunk_bind = soma._cached_bind_transforms_world
        chunk_rest = soma._cached_rest_shape
        bs.rebind(chunk_bind, chunk_rest)

        # Re-run MHR forward for this chunk
        ic = identity_coeffs_t[start:end].to(device)
        mp = model_params_t[start:end].to(device)
        fe = face_expr.expand(end - start, -1)
        with torch.no_grad():
            verts_cm, _ = mhr_jit(ic, mp, fe)
        mhr_verts_all.append((verts_cm * cm_to_m).cpu().numpy())

        # Reconstruct via batched_skinning.pose() at high LOD
        chunk_rot = rotations[start:end].to(device)
        chunk_transl = root_transl[start:end].to(device)
        with torch.no_grad():
            eval_v, _ = bs.pose(chunk_rot, chunk_transl, absolute_pose=True, return_transforms=True)
        eval_verts_all.append((eval_v.detach() * cm_to_m).cpu().numpy())

    mhr_verts_m = np.concatenate(mhr_verts_all, axis=0)
    eval_verts = np.concatenate(eval_verts_all, axis=0)

    method_tag = "analytical"
    if args.autograd_iters > 0:
        method_tag += f"_autograd{args.autograd_iters}"
    out_name = f"out/mhr2soma_eval_{method_tag}.mp4"
    print(f"Rendering MHR (with correctives) vs SOMA (no correctives) -> {out_name}")
    render_comparison_video(
        out_name,
        mhr_verts_m,
        mhr_faces,
        eval_verts,
        soma_faces,
        center=True,
        cam_dist_scale=5.0,
        fps=args.fps,
        label_source="MHR",
        label_soma=f"SOMA {method_tag}",
    )


if __name__ == "__main__":
    main()
