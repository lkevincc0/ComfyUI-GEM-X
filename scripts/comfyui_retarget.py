"""Retarget GEM-X SOMA motion (motion.npz) onto a rigged FBX character; run from the GEM-X root.

Per mapped bone the target world rotation is

    W_t(f) = C @ D_s(f) @ C.T @ A @ W_t_rest

where D_s is the SOMA joint's world rotation relative to the SOMA T-pose, C maps
SOMA world axes to the FBX scene axes (derived from both skeletons, so Y-up and Z-up
rigs both work) and A aligns the target rest pose (e.g. A-pose) to the T-pose.
Unmapped bones (twist, IK, extra spine) keep their rest local transform.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# Canonical joint -> SOMA joint, direction child (None: inherit parent alignment).
CANONICAL = [
    ("hips", "Hips", None),
    ("spine", "Spine1", "spine2"),
    ("spine2", "Spine2", "chest"),
    ("chest", "Chest", "neck"),
    ("neck", "Neck1", "head"),
    ("head", "Head", None),
]
for _side, _s in (("l", "Left"), ("r", "Right")):
    CANONICAL += [
        (f"{_side}_shoulder", f"{_s}Shoulder", f"{_side}_arm"),
        (f"{_side}_arm", f"{_s}Arm", f"{_side}_forearm"),
        (f"{_side}_forearm", f"{_s}ForeArm", f"{_side}_hand"),
        (f"{_side}_hand", f"{_s}Hand", None),
        (f"{_side}_upleg", f"{_s}Leg", f"{_side}_leg"),
        (f"{_side}_leg", f"{_s}Shin", f"{_side}_foot"),
        (f"{_side}_foot", f"{_s}Foot", f"{_side}_toe"),
        (f"{_side}_toe", f"{_s}ToeBase", None),
    ]


def _sided(left, right):
    out = {}
    for side, names in (("l", left), ("r", right)):
        for key, name in zip(("shoulder", "arm", "forearm", "hand", "upleg", "leg", "foot", "toe"), names):
            out[f"{side}_{key}"] = name
    return out


RIGS = {
    "mixamo": {
        "hips": "Hips", "spine": "Spine", "spine2": "Spine1", "chest": "Spine2",
        "neck": "Neck", "head": "Head",
        **_sided(["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                  "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"],
                 ["RightShoulder", "RightArm", "RightForeArm", "RightHand",
                  "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]),
    },
    "unreal": {
        "hips": "pelvis", "spine": "spine_01", "spine2": "spine_03", "chest": "spine_05",
        "neck": "neck_01", "head": "head",
        **_sided(["clavicle_l", "upperarm_l", "lowerarm_l", "hand_l", "thigh_l", "calf_l", "foot_l", "ball_l"],
                 ["clavicle_r", "upperarm_r", "lowerarm_r", "hand_r", "thigh_r", "calf_r", "foot_r", "ball_r"]),
    },
    "unity_kyle": {
        "hips": "Root", "chest": "Ribs", "neck": "Neck", "head": "Head",
        **_sided(["Left_Shoulder_Joint_01", "Left_Upper_Arm_Joint_01", "Left_Forearm_Joint_01",
                  "Left_Wrist_Joint_01", "Left_Thigh_Joint_01", "Left_Knee_Joint_01",
                  "Left_Ankle_Joint_01", "Left_Toe_Joint_01"],
                 ["Right_Shoulder_Joint_01", "Right_Upper_Arm_Joint_01", "Right_Forearm_Joint_01",
                  "Right_Wrist_Joint_01", "Right_Thigh_Joint_01", "Right_Knee_Joint_01",
                  "Right_Ankle_Joint_01", "Right_Toe_Joint_01"]),
    },
}
# Generic Unity humanoid naming falls back to the Mixamo table without the prefix.
REQUIRED = ("hips", "l_arm", "r_arm", "l_upleg", "r_upleg", "l_leg", "r_leg")


def _log(msg):
    print(f"[GEM-X retarget] {msg}", flush=True)


def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Degenerate skeleton direction")
    return v / n


def _basis(left, up):
    up = _unit(up)
    forward = _unit(np.cross(_unit(left), up))
    left = np.cross(up, forward)
    return np.stack([left, up, forward], axis=1)  # columns


def _align(a, b):
    """Shortest-arc rotation matrix taking direction a onto b."""
    a, b = _unit(a), _unit(b)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        return Rotation.from_rotvec(_unit(perp) * math.pi).as_matrix()
    return Rotation.from_rotvec(axis / s * math.atan2(s, c)).as_matrix()


# ----------------------------------------------------------------------------- SOMA side


def soma_world(npz_path, models_dir):
    """Return SOMA joint names, world rotations/positions per frame and in T-pose (meters)."""
    import torch

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "third_party" / "soma")]
    from gem.utils.soma_utils.soma_layer import SomaLayer
    from soma.correctives_model import CorrectivesMLP
    from soma.geometry.lbs import batch_rodrigues

    # Only skeleton FK is needed. The correctives loader also breaks on newer PyTorch
    # (its patched sparse-tensor validator rejects the weights_only keyword).
    CorrectivesMLP.load_checkpoint = staticmethod(lambda *args, **kwargs: None)

    data = np.load(npz_path)
    g = {k.split("/", 1)[1]: data[k] for k in data.files if k.startswith("body_params_global/")}
    fps = float(data["fps"])
    T = int(g["global_orient"].shape[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = SomaLayer(data_root=str(Path(models_dir) / "soma_assets"), low_lod=True, device=device,
                      identity_model_type="mhr", mode="warp" if device == "cuda" else "torch")
    soma = layer.soma
    names = [str(n) for n in soma.rig_data["joint_names"]]
    f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    ident = f32(g["identity_coeffs"]).reshape(T, -1).mean(0, keepdim=True)
    scale = f32(g["scale_params"]).reshape(T, -1).mean(0, keepdim=True)
    soma.prepare_identity(ident, scale[:, 1:], repose_to_bind_pose=False, global_scale=scale[:, :1])
    bind, rest = soma._cached_bind_transforms_world, soma._cached_rest_shape

    def fk(poses, transl):
        n = poses.shape[0]
        rot = soma._pad_poses(batch_rodrigues(poses.reshape(-1, 3)).reshape(n, 77, 3, 3))
        soma.batched_skinning.rebind(bind.expand(n, -1, -1, -1), rest.expand(n, -1, -1))
        _, world = soma.batched_skinning.pose(local_rotations=rot, hips_translations=transl,
                                              return_transforms=True)
        return world.detach().double().cpu().numpy()

    poses = torch.cat([f32(g["global_orient"]).reshape(T, 1, 3), f32(g["body_pose"]).reshape(T, 76, 3)], 1)
    transl = f32(g["transl"]).reshape(T, 3)
    with torch.no_grad():
        world = np.concatenate([fk(poses[i:i + 256], transl[i:i + 256]) for i in range(0, T, 256)])
        tpose = fk(torch.zeros((1, 77, 3), device=device), torch.zeros((1, 3), device=device))[0]
    return names, world, tpose, fps


# ----------------------------------------------------------------------------- FBX side


def _np(m):
    return np.array([[m.Get(i, j) for j in range(4)] for i in range(4)])


def _rot(m44):
    """FBX row-vector matrix -> column-convention orthonormal rotation."""
    r = m44[:3, :3] / np.linalg.norm(m44[:3, :3], axis=1, keepdims=True)
    return Rotation.from_matrix(r.T).as_matrix()


def load_scene(path):
    import fbx

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, "IOSRoot")
    manager.SetIOSettings(ios)
    scene = fbx.FbxScene.Create(manager, "")
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(str(path), -1, ios) or not importer.Import(scene):
        raise RuntimeError(f"Cannot import FBX {path}: {importer.GetStatus().GetErrorString()}")
    importer.Destroy()
    # Drop existing takes so evaluation returns the rest (bind) pose.
    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)
    for i in range(scene.GetSrcObjectCount(criteria) - 1, -1, -1):
        stack = scene.GetSrcObject(criteria, i)
        scene.DisconnectSrcObject(stack)
        stack.Destroy()
    return manager, scene


def save_fbx(manager, scene, path):
    import fbx

    ios = manager.GetIOSettings()
    ios.SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)  # "Export|AdvOptGrp|Fbx|EMBEDDED"
    exporter = fbx.FbxExporter.Create(manager, "")
    fmt = manager.GetIOPluginRegistry().GetNativeWriterFormat()
    if not exporter.Initialize(str(path), fmt, ios) or not exporter.Export(scene):
        raise RuntimeError(f"FBX export failed: {exporter.GetStatus().GetErrorString()}")
    exporter.Destroy()


def scene_nodes(scene):
    nodes = {}

    def walk(node):
        nodes[node.GetName()] = node
        for i in range(node.GetChildCount()):
            walk(node.GetChild(i))

    walk(scene.GetRootNode())
    return nodes


def detect_rig(nodes):
    def strip(name):
        return name.split(":")[-1]

    by_short = {}
    for name in nodes:
        by_short.setdefault(strip(name).lower(), name)
    best = None
    for rig, table in RIGS.items():
        found = {k: by_short.get(v.lower()) for k, v in table.items()}
        found = {k: v for k, v in found.items() if v}
        if all(k in found for k in REQUIRED) and (best is None or len(found) > len(best[1])):
            best = (rig, found)
    if best is None:
        raise ValueError("Unsupported skeleton: expected Mixamo, Unreal (UE4/UE5/UEFN) or Unity Robot Kyle bone names")
    return best


def _euler_order(node):
    import fbx

    return {
        fbx.EFbxRotationOrder.eEulerXYZ: "xyz", fbx.EFbxRotationOrder.eEulerXZY: "xzy",
        fbx.EFbxRotationOrder.eEulerYZX: "yzx", fbx.EFbxRotationOrder.eEulerYXZ: "yxz",
        fbx.EFbxRotationOrder.eEulerZXY: "zxy", fbx.EFbxRotationOrder.eEulerZYX: "zyx",
    }.get(node.RotationOrder.Get(), "xyz")


def _vec(prop):
    v = prop.Get()
    return np.array([v[0], v[1], v[2]], dtype=float)


MATERIAL_COLORS = {"red": (0.8, 0.08, 0.06), "orange": (0.94, 0.58, 0.33), "gray": (0.6, 0.6, 0.6),
                   "blue": (0.1, 0.3, 0.8)}


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp")


def resolve_textures(scene, search_dir, out_dir):
    """Point textures at files found by name under search_dir; return (missing, found) textures.

    FBX files usually store absolute paths from the author's machine. Files are matched
    by name (any image extension) and non-PNG/JPEG images are converted to PNG, since the
    ComfyUI viewer has no TGA/TIFF loader. Found textures are embedded on export.
    """
    textures = [scene.GetTexture(i) for i in range(scene.GetTextureCount())]
    textures = [t for t in textures if hasattr(t, "GetFileName")]
    index = {}
    for path in sorted(Path(search_dir).rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem.lower(), []).append(path)
    missing, resolved, found_textures = [], {}, []
    for tex in textures:
        original = str(tex.GetFileName()).replace("\\", "/")
        found = Path(original) if Path(original).is_file() else None
        if found is None:
            candidates = index.get(Path(original).stem.lower(), [])
            # Prefer browser-friendly formats when several copies exist.
            candidates.sort(key=lambda p: p.suffix.lower() not in (".png", ".jpg", ".jpeg"))
            found = candidates[0] if candidates else None
        if found is None:
            missing.append(tex)
            continue
        found_textures.append(tex)
        if found.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            from PIL import Image

            out_dir.mkdir(parents=True, exist_ok=True)
            png = out_dir / f"{found.stem}.png"
            with Image.open(found) as image:
                image.save(png)
            found = png
        # Only the absolute path is set: SetRelativeFileName stores garbage in fbxsdkpy,
        # and the SDK derives the relative path on export.
        tex.SetFileName(str(found.resolve()))
        resolved[Path(original).stem.lower()] = found.resolve()
        if str(found) != original:
            _log(f"texture {Path(original).name} -> {found}")
    # Embedding reads the separate Video (media) objects, so retarget those too.
    import fbx

    criteria = fbx.FbxCriteria.ObjectType(fbx.FbxVideo.ClassId)
    for i in range(scene.GetSrcObjectCount(criteria)):
        video = scene.GetSrcObject(criteria, i)
        found = resolved.get(Path(str(video.GetFileName()).replace("\\", "/")).stem.lower())
        if found is not None:
            video.SetFileName(str(found))
    return missing, found_textures


def recolor(scene, material, search_dir, texture_dir):
    """Use original textures when found; 'auto' falls back to solid orange if none are found."""
    import fbx

    missing, found = [], []
    if material in ("auto", "original"):
        missing, found = resolve_textures(scene, search_dir, texture_dir)
    if material == "original" or (material == "auto" and not missing):
        return "original"
    if material == "auto" and found:
        # Keep the textures we have (e.g. color map) and drop unresolved ones (e.g. normal).
        for tex in missing:
            _log(f"texture not found, unlinked: {tex.GetFileName()}")
            tex.DisconnectAllDstObject()
        return "original"
    if material == "auto":
        _log(f"{len(missing)} texture file(s) not found (e.g. {missing[0].GetFileName()}); using solid orange")
        material = "orange"
    color = fbx.FbxDouble3(*MATERIAL_COLORS[material])
    for i in range(scene.GetMaterialCount()):
        mat = scene.GetMaterial(i)
        prop = mat.GetFirstProperty()
        while prop.IsValid():  # unlink every texture (diffuse, normal, specular, ...)
            prop.DisconnectAllSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxTexture.ClassId))
            prop = mat.GetNextProperty(prop)
        if hasattr(mat, "Diffuse"):
            mat.Diffuse.Set(color)
            mat.DiffuseFactor.Set(1.0)
        if hasattr(mat, "Specular"):
            mat.Specular.Set(fbx.FbxDouble3(0.2, 0.2, 0.2))
            mat.Shininess.Set(20.0)
    return material


# ----------------------------------------------------------------------------- retarget


def retarget(npz_path, target_fbx, output_fbx, models_dir, yaw_offset=0.0, root_motion=True,
             material="auto"):
    import fbx

    names, s_world, s_tpose, fps = soma_world(npz_path, models_dir)
    T = s_world.shape[0]
    s_index = {n: i for i, n in enumerate(names)}
    manager, scene = load_scene(target_fbx)
    nodes = scene_nodes(scene)
    rig, mapping = detect_rig(nodes)
    _log(f"{Path(target_fbx).name}: rig={rig}, {len(mapping)} mapped bones, {T} frames @ {fps:g} fps")

    canon = {c: (soma_name, child) for c, soma_name, child in CANONICAL}
    s_rest_R = {c: s_tpose[s_index[canon[c][0]], :3, :3] for c in mapping}
    s_rest_p = {c: s_tpose[s_index[canon[c][0]], :3, 3] for c in mapping}
    t_rest_M = {n: _np(node.EvaluateGlobalTransform()) for n, node in nodes.items()}
    t_rest_R = {n: _rot(m) for n, m in t_rest_M.items()}
    t_rest_p = {n: m[3, :3].copy() for n, m in t_rest_M.items()}

    # Axis change SOMA -> scene, from each skeleton's own left/up directions.
    up_sign = scene.GetGlobalSettings().GetAxisSystem().GetUpVector()
    t_up = np.zeros(3)
    t_up[up_sign[0].value - 1] = float(up_sign[1])
    C = _basis(t_rest_p[mapping["l_upleg"]] - t_rest_p[mapping["r_upleg"]], t_up) @ \
        _basis(s_rest_p["l_upleg"] - s_rest_p["r_upleg"], np.array([0.0, 1.0, 0.0])).T

    # Rest alignment: rotate each target bone so its rest direction matches the SOMA T-pose.
    align = {}
    for c, (_, child) in canon.items():
        if c not in mapping:
            continue
        if child in mapping and c != "hips":
            align[c] = _align(t_rest_p[mapping[child]] - t_rest_p[mapping[c]],
                              C @ (s_rest_p[child] - s_rest_p[c]))
    for c, _, _ in CANONICAL:  # leaves inherit the nearest aligned ancestor in the chain
        if c in mapping and c not in align:
            parent = {"head": "neck", "l_hand": "l_forearm", "r_hand": "r_forearm",
                      "l_toe": "l_foot", "r_toe": "r_foot"}.get(c)
            align[c] = align.get(parent, np.eye(3))

    # Remove the initial heading so the character starts facing its own forward axis.
    hips = s_index["Hips"]
    s_forward = _basis(s_rest_p["l_upleg"] - s_rest_p["r_upleg"], np.array([0.0, 1.0, 0.0]))[:, 2]
    fwd = s_world[0, hips, :3, :3] @ s_rest_R["hips"].T @ s_forward
    heading = math.atan2(fwd[0], fwd[2]) - math.atan2(s_forward[0], s_forward[2])
    Y = Rotation.from_euler("y", -heading + math.radians(yaw_offset)).as_matrix()

    # World rotations of mapped target bones.
    t_world = {}
    for c, tname in mapping.items():
        j = s_index[canon[c][0]]
        D = np.einsum("ij,fjk,kl->fil", Y, s_world[:, j, :3, :3], s_rest_R[c].T)
        t_world[tname] = np.einsum("ij,fjk,kl->fil", C, D, C.T) @ (align[c] @ t_rest_R[tname])

    # Root translation: SOMA hips displacement, scaled by leg length, floor-relative.
    feet = [s_index[n] for n in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase")]
    s_floor_rest = s_tpose[1:, 1, 3].min()
    s_hips_h = s_rest_p["hips"][1] - s_floor_rest
    feet_h_rest = s_tpose[feet, 1, 3].min() - s_floor_rest
    p = np.einsum("ij,fj->fi", Y, s_world[:, hips, :3, 3])
    floor = np.percentile(np.einsum("ij,ftj->fti", Y, s_world[:, feet, :3, 3])[..., 1].min(1), 5) - feet_h_rest
    disp = p - np.array([p[0, 0], floor + s_hips_h, p[0, 2]])
    if not root_motion:
        disp[:, [0, 2]] = 0.0
    t_hips = mapping["hips"]
    t_floor = min(float(np.dot(t_rest_p[n], t_up)) for n in mapping.values())
    k = (float(np.dot(t_rest_p[t_hips], t_up)) - t_floor) / s_hips_h
    hips_world_pos = t_rest_p[t_hips] + k * np.einsum("ij,fj->fi", C, disp)
    _log(f"axis map det={np.linalg.det(C):+.2f}, scale={k:.2f}, heading={math.degrees(heading):.1f} deg")

    # FK over the full target hierarchy to get parent world rotations for local keys.
    world_cache = {}

    def world_rot(node):
        name = node.GetName()
        if name in t_world:
            return t_world[name]
        if name not in world_cache:
            parent = node.GetParent()
            if parent is None:
                world_cache[name] = np.broadcast_to(t_rest_R[name], (T, 3, 3))
            else:
                local = t_rest_R[parent.GetName()].T @ t_rest_R[name]
                world_cache[name] = world_rot(parent) @ local
        return world_cache[name]

    settings = scene.GetGlobalSettings()
    settings.SetTimeMode(fbx.FbxTime.EMode.eCustom)
    settings.SetCustomFrameRate(fps)
    stack = fbx.FbxAnimStack.Create(scene, "GEM-X")
    layer = fbx.FbxAnimLayer.Create(scene, "Base")
    stack.AddMember(layer)
    scene.SetCurrentAnimationStack(stack)
    times = []
    for f in range(T):
        t = fbx.FbxTime()
        t.SetSecondDouble(f / fps)
        times.append(t)
    span = fbx.FbxTimeSpan(times[0], times[-1])
    stack.SetLocalTimeSpan(span)
    settings.SetTimelineDefaultTimeSpan(span)
    linear = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear

    def write(prop, values):
        prop.ModifyFlag(fbx.FbxPropertyFlags.EFlags.eAnimatable, True)
        for axis, column in zip("XYZ", values.T):
            curve = prop.GetCurve(layer, axis, True)
            curve.KeyModifyBegin()
            for t, v in zip(times, column):
                idx = curve.KeyAdd(t)[0]
                curve.KeySetValue(idx, float(v))
                curve.KeySetInterpolation(idx, linear)
            curve.KeyModifyEnd()

    for tname in t_world:
        node = nodes[tname]
        parent = node.GetParent()
        local = np.swapaxes(world_rot(parent), 1, 2) @ t_world[tname]
        pre = post = np.eye(3)
        if node.RotationActive.Get():
            pre = Rotation.from_euler("xyz", _vec(node.PreRotation), degrees=True).as_matrix()
            post = Rotation.from_euler("xyz", _vec(node.PostRotation), degrees=True).as_matrix()
        lcl = pre.T @ local @ post
        order = _euler_order(node)
        euler = Rotation.from_matrix(lcl).as_euler(order, degrees=True)
        euler = np.degrees(np.unwrap(np.radians(euler), axis=0))
        # as_euler returns angles in the order given; FBX curves are X/Y/Z.
        write(node.LclRotation, euler[:, [order.index(a) for a in "xyz"]])

    parent = nodes[t_hips].GetParent()
    inv_parent = np.linalg.inv(t_rest_M[parent.GetName()])
    local_pos = np.c_[hips_world_pos, np.ones(T)] @ inv_parent
    write(nodes[t_hips].LclTranslation, local_pos[:, :3])

    # Self-check: evaluate the written animation and compare against the targets.
    scene.GetAnimationEvaluator().Reset()
    worst = 0.0
    for f in sorted({0, T // 2, T - 1}):
        for tname, target in t_world.items():
            got = _rot(_np(nodes[tname].EvaluateGlobalTransform(times[f])))
            worst = max(worst, np.degrees(np.linalg.norm(Rotation.from_matrix(got @ target[f].T).as_rotvec())))
    pos_err = np.linalg.norm(_np(nodes[t_hips].EvaluateGlobalTransform(times[-1]))[3, :3] - hips_world_pos[-1])
    _log(f"self-check: max rotation error {worst:.3f} deg, root position error {pos_err:.4f}")
    if worst > 1.0:
        raise RuntimeError(f"Retarget self-check failed: rotation error {worst:.2f} deg")

    output_fbx = Path(output_fbx)
    material = recolor(scene, material, Path(target_fbx).parent,
                       output_fbx.parent / f"{output_fbx.stem}_textures")
    save_fbx(manager, scene, output_fbx)
    manager.Destroy()
    # Preview 3D shows the file in its own units (cm rigs look 100x too big for the
    # default camera), Y-up, characters facing +Z. Frame the whole body from the front.
    height = 1.15 * (max(float(np.dot(t_rest_p[b], t_up)) for b in mapping.values()) - t_floor)
    camera = {"position": {"x": 0.9 * height, "y": 0.8 * height, "z": 2.6 * height},
              "target": {"x": 0.0, "y": 0.5 * height, "z": 0.0},
              "zoom": 1, "cameraType": "perspective"}
    return {"rig": rig, "frames": T, "fps": fps, "mapped_bones": len(mapping), "camera": camera,
            "max_rotation_error_deg": round(worst, 4), "material": material, "fbx": str(output_fbx)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, help="motion.npz from comfyui_infer.py")
    parser.add_argument("--target", required=True, help="rigged FBX character")
    parser.add_argument("--output", required=True, help="output animated FBX path")
    parser.add_argument("--models-dir", required=True, help="ComfyUI models/gem_x directory")
    parser.add_argument("--yaw", type=float, default=0.0, help="extra heading rotation in degrees")
    parser.add_argument("--in-place", action="store_true", help="drop horizontal root motion")
    parser.add_argument("--material", choices=("auto", "original", *MATERIAL_COLORS), default="auto",
                        help="auto: original textures (searched by name beside the FBX), else solid orange")
    args = parser.parse_args()
    info = retarget(args.motion, args.target, args.output, args.models_dir,
                    yaw_offset=args.yaw, root_motion=not args.in_place,
                    material=args.material)
    print(json.dumps(info), flush=True)
    sys.stderr.flush()
    # fbxsdkpy wrappers segfault when finalized after FbxManager.Destroy(); the FBX
    # is fully written at this point, so skip interpreter teardown.
    os._exit(0)


if __name__ == "__main__":
    main()
