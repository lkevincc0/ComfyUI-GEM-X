"""Exercise the process boundary without downloading GEM weights or importing ComfyUI."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from comfyui_gem_x import NODE_CLASS_MAPPINGS
from comfyui_gem_x import nodes


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gem bridge ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "input, with spaces.mp4"
        self.source.write_bytes(b"test video")
        self.folders = types.SimpleNamespace(
            models_dir=str(self.root / "ComfyUI" / "models"),
            get_input_directory=lambda: str(self.root),
            get_output_directory=lambda: str(self.root / "outputs"),
        )
        self.management = types.ModuleType("comfy.model_management")
        self.management.unload_all_models = lambda: None
        self.management.soft_empty_cache = lambda: None
        self.management.throw_exception_if_processing_interrupted = lambda: None
        modules = {"folder_paths": self.folders, "comfy": types.ModuleType("comfy"),
                   "comfy.model_management": self.management}
        self.patcher = patch.dict(sys.modules, modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        # Never download real weights in tests.
        download = patch.object(nodes.models, "ensure_models", lambda *args, **kwargs: None)
        download.start()
        self.addCleanup(download.stop)

    def test_registration_has_no_model_imports(self):
        self.assertEqual(len(NODE_CLASS_MAPPINGS), 4)
        for cls in NODE_CLASS_MAPPINGS.values():
            self.assertTrue(callable(getattr(cls, cls.FUNCTION)))
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))

    def test_resolve_input_relative_path(self):
        result = nodes.GEMVideoPath().resolve(self.source.name)
        self.assertEqual(result, (str(self.source),))
        with self.assertRaisesRegex(ValueError, "does not exist"):
            nodes.GEMVideoPath().resolve("missing.mp4")

    def test_worker_failure_retains_diagnostics(self):
        log = self.root / "failure.log"
        with self.assertRaisesRegex(RuntimeError, "model asset missing"):
            nodes._run_worker(
                [sys.executable, "-c", "raise RuntimeError('model asset missing')"],
                self.root, log, lambda: None,
            )
        self.assertIn("RuntimeError", log.read_text())

    def test_cancel_terminates_worker(self):
        class Cancelled(Exception):
            pass

        def cancel():
            raise Cancelled()

        children = []
        original = subprocess.Popen

        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            children.append(process)
            return process

        with patch.object(nodes.subprocess, "Popen", side_effect=launch):
            with self.assertRaises(Cancelled):
                nodes._run_worker(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    self.root, self.root / "cancel.log", cancel,
                )
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())

    def test_infer_process_roundtrip_and_unique_outputs(self):
        worker = self.root / "fake_worker.py"
        worker.write_text('''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--video'); p.add_argument('--output'); p.add_argument('--preview')
p.add_argument('--seed'); p.add_argument('--checkpoint'); p.add_argument('--no-postprocess', action='store_true')
p.add_argument('--backend'); p.add_argument('--no-imgfeat', action='store_true')
p.add_argument('--models-dir', required=True)
a = p.parse_args()
out = Path(a.output)
assert Path(a.video).is_file()
assert a.no_postprocess and a.seed == '37'
assert a.backend == 'onnx'
assert Path(a.models_dir) == Path(a.video).parent / 'ComfyUI' / 'models' / 'gem_x'
pt = out / 'hpe_results.pt'; pt.write_bytes(b'prediction')
npz = out / 'motion.npz'; npz.write_bytes(b'arrays')
(out / 'manifest.json').write_text(json.dumps({'prediction_pt':str(pt), 'motion_npz':str(npz), 'preview_video':''}))
''')
        with patch.object(nodes, "_environment", return_value=(self.root, sys.executable, worker)):
            first = nodes.GEMInfer().infer(str(self.source), "none", False, 37)["result"]
            second = nodes.GEMInfer().infer(str(self.source), "none", False, 37)["result"]
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(first[2], "")
        self.assertTrue(Path(first[1]).is_file())
        self.assertEqual(json.loads(Path(first[3]).read_text()), first[0])

    def test_success_without_manifest_is_rejected(self):
        worker = self.root / "empty.py"
        worker.write_text("pass")
        with patch.object(nodes, "_environment", return_value=(self.root, sys.executable, worker)):
            with self.assertRaisesRegex(RuntimeError, "did not finish"):
                nodes.GEMInfer().infer(str(self.source), "none", False, 0)

    def test_explicit_missing_checkpoint_does_not_start_worker(self):
        with patch.object(nodes, "_environment", return_value=(self.root, sys.executable, self.source)):
            with self.assertRaisesRegex(ValueError, "Checkpoint does not exist"):
                nodes.GEMInfer().infer(str(self.source), "none", False, 0, "missing.ckpt")

    def test_onnx_rejects_unsupported_postprocessing(self):
        with patch.object(nodes, "_environment", return_value=(self.root, sys.executable, self.source)):
            with self.assertRaisesRegex(ValueError, "disable postprocess"):
                nodes.GEMInfer().infer(str(self.source), "none", True, 0)

    def test_checkpoint_cannot_load_from_repo_or_other_directories(self):
        external = self.root / "gem_soma.ckpt"
        external.write_bytes(b"outside models")
        with patch.object(nodes, "_environment", return_value=(self.root, sys.executable, self.source)):
            for value in (str(external), "../../../gem_soma.ckpt"):
                with self.assertRaisesRegex(ValueError, "inside ComfyUI/models/gem_x"):
                    nodes.GEMInfer().infer(str(self.source), "none", False, 0, value)

    def test_worker_model_paths_use_comfy_directory_in_fresh_process(self):
        import os

        model_dir = self.root / "ComfyUI" / "models" / "gem_x"
        script = '''from pathlib import Path
from gem.utils import hf_utils as hf
from gem.utils.model_paths import model_path
import os
root = Path(os.environ['GEM_X_MODEL_ROOT'])
for directory in [hf.DEFAULT_CKPT_DIR, hf.VITPOSE_CKPT_DIR, hf.SAM3D_CKPT_DIR,
                  hf.MHR_MODEL_DIR, hf.SOMA_DATA_DIR, hf.ONNX_DIR, model_path('soma_assets')]:
    assert Path(directory).is_relative_to(root), directory
onnx = root / 'onnx'
onnx.mkdir(parents=True)
for name in hf.ONNX_MODELS['vitpose']:
    (onnx / name).write_bytes(b'local model')
assert hf.download_onnx_model('vitpose') == str(onnx / 'vitpose.onnx')
'''
        result = subprocess.run([sys.executable, "-c", script],
                                env={**os.environ, "GEM_X_MODEL_ROOT": str(model_dir)},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
