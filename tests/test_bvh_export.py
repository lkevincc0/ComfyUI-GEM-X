import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('bvh_export', Path(__file__).resolve().parents[1] / 'scripts/comfyui_export_bvh.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BVHExportTests(unittest.TestCase):
    def test_oriented_rest_bones_roundtrip_through_exported_rotation_channels(self):
        names = ['Root', 'Hips', 'Arm', 'Hand']
        parents = np.array([0, 0, 1, 2])
        rest = np.repeat(np.eye(4)[None], 4, axis=0)
        rest[:, :3, :3] = Rotation.from_euler('XYZ', [[0, 0, 0], [10, 20, 30], [30, -20, 10], [0, 30, 50]], degrees=True).as_matrix()
        rest[:, :3, 3] = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]]
        delta = np.repeat(np.eye(3)[None, None], 3 * 4, axis=0).reshape(3, 4, 3, 3)
        delta[:, 1] = Rotation.from_euler('y', [[0], [30], [60]], degrees=True).as_matrix()
        delta[:, 2] = delta[:, 1] @ Rotation.from_euler('z', [[0], [20], [40]], degrees=True).as_matrix()
        delta[:, 3] = delta[:, 2]
        world = np.repeat(rest[None], 3, axis=0)
        world[:, :, :3, :3] = delta @ rest[:, :3, :3]
        world[:, 1, :3, 3] = [[0, 1, 0], [.2, 1.1, .1], [.4, 1.2, .2]]
        for j in [2, 3]:
            world[:, j, :3, 3] = world[:, parents[j], :3, 3] + np.einsum('fij,j->fi', delta[:, parents[j]], rest[j, :3, 3] - rest[parents[j], :3, 3])
        with tempfile.TemporaryDirectory() as folder:
            for dummy in [False, True]:
                path = Path(folder) / 'motion.bvh'
                info = module.export_bvh(path, names, parents, world, rest, 24, dummy)
                text = path.read_text()
                self.assertIn('Frames: 3', text)
                self.assertIn('Frame Time: 0.041666666667', text)
                self.assertEqual(info['joints'], 4 if dummy else 3)
                self.assertLess(info['max_fk_error_meters'], 1e-10)
                rows = np.array([[float(v) for v in line.split()] for line in text.split('Frame Time: ')[1].splitlines()[1:]])
                self.assertEqual(rows.shape, (3, (4 if dummy else 3) * 3 + 3))
                # Read Euler channels as a BVH consumer and reconstruct global delta rotations.
                rotations = Rotation.from_euler('ZXY', rows[:, 3:].reshape(-1, 3), degrees=True).as_matrix().reshape(3, -1, 3, 3)
                if dummy:
                    rotations = rotations[:, 1:]
                for j in range(1, 3):
                    rotations[:, j] = rotations[:, j - 1] @ rotations[:, j]
                np.testing.assert_allclose(rotations, delta[:, 1:], atol=1e-9)


if __name__ == '__main__':
    unittest.main()
