"""Export metric BVH from the same SOMA forward kinematics used for FBX retargeting."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.spatial.transform import Rotation


def export_bvh(path, names, parents, world, rest, fps, dummy_root=False):
    first = 0 if dummy_root else 1
    children = {j: [] for j in range(first, len(names))}
    for j in range(first + 1, len(names)):
        children[int(parents[j])].append(j)
    delta = world[:, :, :3, :3] @ np.linalg.inv(rest[:, :3, :3])
    delta = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_matrix().reshape(delta.shape)
    if dummy_root:
        delta[:, 0] = np.eye(3)
    local = delta.copy()
    for j in range(first + 1, len(names)):
        local[:, j] = delta[:, parents[j]].transpose(0, 2, 1) @ delta[:, j]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected')
        angles = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_euler('ZXY').reshape(len(world), len(names), 3)
    angles = np.rad2deg(np.unwrap(angles, axis=0))
    offsets = rest[:, :3, 3].copy()
    for j in range(first + 1, len(names)):
        offsets[j] -= rest[parents[j], :3, 3]
    offsets[first] = 0
    translations = world[:, 1, :3, 3].copy()
    if dummy_root:
        translations -= rest[1, :3, 3]

    # Verify that BVH's fixed offsets and local rotation channels reproduce the source FK.
    positions = np.zeros_like(world[:, :, :3, 3])
    positions[:, first] = translations
    for j in range(first + 1, len(names)):
        parent = int(parents[j])
        positions[:, j] = positions[:, parent] + np.einsum('fij,j->fi', delta[:, parent], offsets[j])
    error = np.max(np.linalg.norm(positions[:, 1:] - world[:, 1:, :3, 3], axis=-1))
    if error > 0.001:
        raise ValueError(f'BVH skeleton reconstruction differs from SOMA FK by {error:.6f} meters')
    lines = ['HIERARCHY']
    order = []

    def joint(j, depth):
        indent = '  ' * depth
        order.append(j)
        lines.extend([f'{indent}{"ROOT" if j == first else "JOINT"} {names[j]}', indent + '{',
                      indent + '  OFFSET ' + ' '.join(f'{v:.9f}' for v in offsets[j]),
                      indent + ('  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation'
                                if j == first else '  CHANNELS 3 Zrotation Xrotation Yrotation')])
        for child in children[j]:
            joint(child, depth + 1)
        if not children[j]:
            lines.extend([indent + '  End Site', indent + '  {', indent + '    OFFSET 0 0 0', indent + '  }'])
        lines.append(indent + '}')
    joint(first, 0)
    channels = np.concatenate([translations, angles[:, order].reshape(len(world), -1)], axis=1)
    lines.extend(['MOTION', f'Frames: {len(world)}', f'Frame Time: {1 / fps:.12f}'])
    lines.extend(' '.join(f'{value:.8f}' for value in frame) for frame in channels)
    path.write_text('\n'.join(lines) + '\n')
    return {'file': path.name, 'joints': len(order), 'frames': len(world), 'fps': fps,
            'units': 'meters', 'rotation_order': 'ZXY', 'max_fk_error_meters': float(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, required=True)
    args = parser.parse_args()
    from comfyui_retarget import soma_world
    names, world, rest, fps = soma_world(args.motion, args.models_dir)
    with np.load(args.models_dir / 'soma_assets' / 'SOMA_neutral.npz', allow_pickle=True) as rig:
        parents = rig['joint_parent_ids'].astype(int)
    outputs = []
    for dummy, filename in [(False, 'motion_77j.bvh'), (True, 'motion_78j.bvh')]:
        target = args.motion.parent / filename
        temporary = target.with_suffix('.bvh.tmp')
        outputs.append(export_bvh(temporary, names, parents, world, rest, fps, dummy))
        temporary.replace(target)
        outputs[-1]['file'] = filename
    (args.motion.parent / 'bvh_info.json').write_text(json.dumps(outputs, indent=2))
    print(json.dumps(outputs), flush=True)


if __name__ == '__main__':
    main()
