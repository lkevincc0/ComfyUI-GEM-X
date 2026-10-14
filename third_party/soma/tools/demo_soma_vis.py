# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from soma import SOMALayer
from soma.geometry.rig_utils import joint_local_to_world, joint_world_to_local
from tools.vis_pyrender import (
    MeshRenderer,
    default_pyopengl_platform,
    look_at,
    set_pyopengl_platform,
)

# --------------------------------------------------------------------------------
# Joint Names & Mapping (from nvhuman_layer/joint_names.py)
# --------------------------------------------------------------------------------
# fmt: off
nvskel93_name = [
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

nvskel77_name = [
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
nvskel93to77_idx = [nvskel93_name.index(name) for name in nvskel77_name]

color_map = {
    "soma": (0.4, 0.8, 0.4, 1.0),  # light green
    "mhr": (0.98, 0.65, 0.15, 1.0),  # blue
    "anny": (0.25, 0.75, 1.0, 1.0),  # yellow
    "smpl": (0.55, 0.15, 0.85, 1.0),  # pink
    "smplx": (0.55, 0.15, 0.85, 1.0),  # pink
    "garment": (0.15, 0.15, 1.0, 1.0),  # orange
}


def get_smooth_noise(T, dim, device, num_keyframes=None, mode="normal"):
    if num_keyframes is None:
        num_keyframes = max(3, T // 30)

    if mode == "normal":
        keyframes = torch.randn(1, dim, num_keyframes, device=device)
    elif mode == "uniform":
        keyframes = torch.rand(1, dim, num_keyframes, device=device)

    res = F.interpolate(keyframes, size=T, mode="linear", align_corners=True)[0].T
    return res


def save_video(frames, path, fps=30):
    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="SOMA pyrender demo")
    parser.add_argument("--data-root", default="assets", help="Path to SOMA assets")
    parser.add_argument(
        "--motion-file",
        default="assets/example_animation.npy",
        help="Path to motion file (.npy). If None, uses a dummy motion.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="out/vis_identity_model")
    parser.add_argument("--image-size", type=int, default=1920)
    parser.add_argument("--pyopengl-platform", default=default_pyopengl_platform())
    parser.add_argument("--random-shape", action="store_true", default=False)
    parser.add_argument(
        "--identity-model-type",
        default="soma,mhr,anny,smpl,smplx,garment",
        help="Comma-separated list of identity models to use. Options: soma, mhr, anny, smpl, smplx garment (default: soma,mhr,anny,smpl,smplx,garment)",
    )
    parser.add_argument(
        "--pose-batch-size",
        type=int,
        default=0,
        help="Run forward pass in batches of this many poses to reduce GPU memory. 0 = process all frames at once (default). Try 32 or 64 if OOM.",
    )
    parser.add_argument(
        "--low-lod",
        action="store_true",
        default=False,
        help="Use low level-of-detail mesh (fewer vertices/faces)",
    )
    parser.add_argument(
        "--apply-correctives",
        action="store_true",
        default=False,
        help="Apply pose corrective offsets (default: False)",
    )
    parser.add_argument(
        "--gender",
        default="neutral",
        help="Gender of the model (default: neutral). Only used for smpl and smplx models.",
    )
    args = parser.parse_args()

    identity_models = [m.strip().lower() for m in args.identity_model_type.split(",")]
    valid_models = {"soma", "mhr", "anny", "smpl", "smplx", "garment"}
    invalid_models = set(identity_models) - valid_models
    if invalid_models:
        raise ValueError(
            f"Invalid identity model type(s): {invalid_models}. Valid options: {valid_models}"
        )
    args.identity_models = identity_models

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    set_pyopengl_platform(args.pyopengl_platform)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Initializing models: {', '.join(args.identity_models)}...")
    models = {}
    for identity_model_type in args.identity_models:
        if identity_model_type == "smpl":
            identity_model_kwargs = {
                "gender": args.gender,
            }
        else:
            identity_model_kwargs = {}
        models[identity_model_type] = SOMALayer(
            data_root=args.data_root,
            low_lod=args.low_lod,
            device=str(device),
            identity_model_type=identity_model_type,
            mode="warp",
            identity_model_kwargs=identity_model_kwargs,
        ).to(device)

    reference_model = models[args.identity_models[0]]

    if args.motion_file and os.path.exists(args.motion_file):
        print(f"Loading motion from {args.motion_file}...")
        motion_full = torch.from_numpy(np.load(args.motion_file)).float().to(device)
        joint_rot_mats_local = motion_full[..., :3, :3]
        root_trans = motion_full[..., 1, :3, 3]
    else:
        print("No motion file provided or file not found. Using dummy motion (T-pose rotation).")
        T = 30
        joint_rot_mats_local = (
            torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(T, 78, 1, 1)
        )
        angle = torch.linspace(0, 2 * np.pi, T, device=device)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        zeros = torch.zeros_like(angle)
        ones = torch.ones_like(angle)
        rot_y = torch.stack(
            [
                torch.stack([cos, zeros, sin], dim=-1),
                torch.stack([zeros, ones, zeros], dim=-1),
                torch.stack([-sin, zeros, cos], dim=-1),
            ],
            dim=-2,
        )  # (T, 3, 3)
        joint_rot_mats_local[:, 1] = rot_y  # Rotate Hips
        root_trans = torch.zeros(T, 3, device=device)

    if joint_rot_mats_local.shape[1] == 94:
        subset_idx = [0] + [i + 1 for i in nvskel93to77_idx]
        joint_rot_mats_local = joint_rot_mats_local[:, subset_idx]

    correction = reference_model.t_pose_world[:, :3, :3].transpose(-2, -1)
    joint_rot_mats_world = joint_local_to_world(
        joint_rot_mats_local, reference_model.joint_parent_ids
    )
    joint_rot_mats_world = joint_rot_mats_world @ correction
    joint_rot_mats_local = joint_world_to_local(
        joint_rot_mats_world, reference_model.joint_parent_ids
    )

    T = joint_rot_mats_local.shape[0]
    global_orient = joint_rot_mats_local[:T, 1]  # (T, 3, 3) - Hips is index 1
    body_pose = joint_rot_mats_local[:T, 2:]  # (T, 77, 3, 3)
    pose = torch.cat([global_orient.unsqueeze(1), body_pose], dim=1)

    # Prepare Identity Parameters
    identity_coeffs_map = {}
    for model_type, model in models.items():
        n = model.identity_model.num_identity_coeffs
        if model_type == "anny":
            anny_im = model.identity_model.identity_model
            if args.random_shape:
                phenotypes = {
                    k: get_smooth_noise(T, 1, device, mode="uniform").squeeze(-1)
                    for k in anny_im.phenotype_labels
                }
            else:
                phenotypes = {
                    k: torch.ones(T, device=device) * 0.5 for k in anny_im.phenotype_labels
                }
            local_changes = {k: torch.zeros(T, device=device) for k in anny_im.local_change_labels}
            identity_coeffs_map["anny"] = (phenotypes, local_changes)
        elif model_type == "mhr":
            n_scale = model.identity_model.num_scale_params
            if args.random_shape:
                coeffs = get_smooth_noise(T, n, device)
                scale = get_smooth_noise(T, n_scale, device, mode="normal") * 0.2
            else:
                coeffs = torch.zeros(T, n, device=device)
                scale = torch.zeros(T, n_scale, device=device)
            identity_coeffs_map[model_type] = (coeffs, scale)
        else:
            if args.random_shape:
                coeffs = get_smooth_noise(T, n, device)
            else:
                coeffs = torch.zeros(T, n, device=device)
            identity_coeffs_map[model_type] = (coeffs, None)

    transl = root_trans[:T]

    # 4. Forward Pass using prepare_identity() + pose() API.
    #    When identity is constant (not random_shape), prepare_identity is called
    #    once per model and only pose() runs per batch -- skipping the expensive
    #    identity model + skeleton transfer on every frame.
    pose_batch_size = args.pose_batch_size if args.pose_batch_size > 0 else T
    print(f"Running forward pass (pose_batch_size={pose_batch_size})...")

    outputs = {}
    with torch.no_grad():
        if not args.random_shape:
            for model_type, model in models.items():
                coeffs, scale = identity_coeffs_map[model_type]
                if isinstance(coeffs, dict):
                    coeffs_single = {k: v[:1] for k, v in coeffs.items()}
                    scale_single = {k: v[:1] for k, v in scale.items()} if scale else None
                else:
                    coeffs_single = coeffs[:1]
                    scale_single = scale[:1] if scale is not None else None
                model.prepare_identity(coeffs_single, scale_single)

        for start in range(0, T, pose_batch_size):
            end = min(start + pose_batch_size, T)
            pose_b = pose[start:end]
            transl_b = transl[start:end]

            for model_type, model in models.items():
                if args.random_shape:
                    coeffs, scale = identity_coeffs_map[model_type]
                    if isinstance(coeffs, dict):
                        coeffs_b = {k: v[start:end] for k, v in coeffs.items()}
                        scale_b = {k: v[start:end] for k, v in scale.items()} if scale else None
                    else:
                        coeffs_b = coeffs[start:end]
                        scale_b = scale[start:end] if scale is not None else None
                    model.prepare_identity(coeffs_b, scale_b)

                out_b = model.pose(
                    pose_b,
                    transl=transl_b,
                    pose2rot=False,
                    apply_correctives=args.apply_correctives,
                )
                if model_type not in outputs:
                    outputs[model_type] = {"vertices": [], "joints": []}
                outputs[model_type]["vertices"].append(out_b["vertices"])
                outputs[model_type]["joints"].append(out_b["joints"])

        for model_type in list(outputs.keys()):
            outputs[model_type]["vertices"] = torch.cat(outputs[model_type]["vertices"], dim=0)
            outputs[model_type]["joints"] = torch.cat(outputs[model_type]["joints"], dim=0)

    # 5. Render (model-first loop with streaming video writer)
    print("Rendering videos...")

    suffix = "rand_shape" if args.random_shape else "fixed_shape"
    faces = {
        model_type: models[model_type].faces.detach().cpu().numpy()
        for model_type in args.identity_models
    }
    cam_pose = look_at(
        eye=np.array([0.0, 1.0, 6.0]),
        target=np.array([0.0, 1.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    light_dir = np.array([0.0, -0.5, -1.0])
    renderer = MeshRenderer(image_size=args.image_size, light_intensity=5)

    for model_type in args.identity_models:
        out_path = f"{args.output_dir}/{model_type}_{suffix}.mp4"
        renderer.setup_mesh(
            faces=faces[model_type],
            mesh_color=color_map[model_type],
            cam_pose=cam_pose,
            light_dir=light_dir,
            metallic=0.0,
            roughness=0.5,
            base_color_factor=[0.9, 0.9, 0.9, 1.0],
        )
        writer = imageio.get_writer(out_path, fps=30)
        for t in tqdm(range(T), desc=model_type):
            verts = outputs[model_type]["vertices"][t].detach().cpu().numpy()
            img = renderer.render_frame(verts)
            writer.append_data(img[..., ::-1])
        writer.close()
        print(f"Saved {out_path}")

    renderer.delete()


if __name__ == "__main__":
    main()
