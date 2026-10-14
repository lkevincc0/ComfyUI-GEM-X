"""Lightweight ComfyUI adapter. GEM imports only occur in the worker process."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from . import models


def _comfy_paths():
    import folder_paths

    return folder_paths


def _video_path(value):
    path = Path(value.strip().strip('"')).expanduser()
    if not path.is_absolute():
        path = Path(_comfy_paths().get_input_directory()) / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Video does not exist: {path}")
    return path


def _environment():
    root = Path(os.environ.get("GEM_X_ROOT", Path(__file__).resolve().parents[1])).expanduser()
    root = root.resolve()
    worker = root / "scripts" / "comfyui_infer.py"
    if not worker.is_file():
        raise RuntimeError("Set GEM_X_ROOT to the GEM-X checkout containing scripts/comfyui_infer.py")
    executable = os.environ.get("GEM_X_PYTHON", "")
    if not executable:
        executable = str(root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python"))
    python = shutil.which(executable)
    if not python:
        raise RuntimeError("Set GEM_X_PYTHON to the Python executable in your installed GEM-X environment")
    return root, python, worker


def _run_worker(command, root, log_path, check_interrupt):
    # A file avoids pipe deadlocks and retains the complete log on failure.
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                check_interrupt()
                time.sleep(0.2)
            check_interrupt()
            if process.returncode:
                with log_path.open("rb") as stream:
                    stream.seek(max(0, log_path.stat().st_size - 8000))
                    tail = stream.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"GEM-X failed. Full log: {log_path}\n{tail}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


class GEMVideoPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_path": ("STRING", {"default": "video.mp4"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "resolve"
    CATEGORY = "GEM-X"

    @classmethod
    def IS_CHANGED(cls, video_path):
        stat = _video_path(video_path).stat()
        return (stat.st_mtime_ns, stat.st_size)

    def resolve(self, video_path):
        return (str(_video_path(video_path)),)


class GEMFramesToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "encode"
    CATEGORY = "GEM-X"

    def encode(self, images, fps):
        import cv2
        import numpy as np
        from comfy.model_management import throw_exception_if_processing_interrupted

        if images.ndim != 4 or images.shape[-1] != 3 or len(images) < 2:
            raise ValueError("Provide at least two RGB frames in ComfyUI IMAGE format [N,H,W,3]")
        if not math.isfinite(fps) or not 1 <= fps <= 240:
            raise ValueError("fps must be between 1 and 240")
        height, width = images.shape[1:3]
        if width % 2 or height % 2:
            raise ValueError("Video width and height must be even; resize the frames before encoding")
        path = Path(_comfy_paths().get_temp_directory()) / f"gem_input_{uuid.uuid4().hex}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the temporary MP4")
        try:
            # Move one frame at a time so large GPU batches do not get duplicated on CPU.
            for frame in images:
                throw_exception_if_processing_interrupted()
                rgb = (frame.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except BaseException:
            writer.release()
            path.unlink(missing_ok=True)
            raise
        finally:
            writer.release()
        return (str(path),)


class GEMInfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_path": ("STRING", {"default": "video.mp4", "forceInput": True}),
            "preview": (["none", "keypoints", "mesh"], {"default": "keypoints"}),
            "postprocess": ("BOOLEAN", {"default": False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
        }, "optional": {
            "checkpoint": ("STRING", {"default": ""}),
            "backend": (["onnx", "pytorch"], {"default": "onnx"}),
            "image_features": (["sam3d", "keypoints_only"], {"default": "sam3d"}),
        }}

    RETURN_TYPES = ("GEM_MOTION", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("motion", "motion_npz", "preview_video", "manifest_json")
    FUNCTION = "infer"
    CATEGORY = "GEM-X"
    OUTPUT_NODE = True

    # Re-run on queue: input file and external weights may have changed in place.
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def infer(self, video_path, preview, postprocess, seed, checkpoint="", backend="onnx",
              image_features="sam3d"):
        from comfy.model_management import (
            soft_empty_cache,
            throw_exception_if_processing_interrupted,
            unload_all_models,
        )

        source = _video_path(video_path)
        root, python, worker = _environment()
        models_dir = (Path(_comfy_paths().models_dir) / "gem_x").resolve()
        if preview not in ("none", "keypoints", "mesh"):
            raise ValueError("Unknown preview mode")
        if backend not in ("onnx", "pytorch") or image_features not in ("sam3d", "keypoints_only"):
            raise ValueError("Unknown GEM backend or image feature mode")
        if backend == "onnx" and postprocess:
            raise ValueError("The upstream ONNX path does not run contact postprocessing; disable postprocess")
        if backend == "pytorch" and image_features == "keypoints_only":
            raise ValueError("Use the ONNX backend for keypoints_only mode")
        ckpt = None
        if checkpoint.strip():
            ckpt = Path(checkpoint.strip()).expanduser()
            if not ckpt.is_absolute():
                ckpt = models_dir / ckpt
            ckpt = ckpt.resolve()
            if not ckpt.is_relative_to(models_dir):
                raise ValueError("Checkpoint must be inside ComfyUI/models/gem_x")
            if not ckpt.is_file():
                raise ValueError(f"Checkpoint does not exist: {ckpt}")
        job = Path(_comfy_paths().get_output_directory()) / "gem_x" / uuid.uuid4().hex
        job.mkdir(parents=True)
        command = [python, "-u", str(worker), "--video", str(source), "--output", str(job),
                   "--preview", preview, "--seed", str(seed), "--backend", backend,
                   "--models-dir", str(models_dir)]
        if image_features == "keypoints_only":
            command.append("--no-imgfeat")
        if not postprocess:
            command.append("--no-postprocess")
        if ckpt is not None:
            command.extend(["--checkpoint", str(ckpt)])
        models.ensure_models(models_dir, models.group_for(backend, image_features),
                      throw_exception_if_processing_interrupted)
        # The worker has its own CUDA context; release Comfy-managed models first.
        unload_all_models()
        soft_empty_cache()
        _run_worker(command, root, job / "inference.log", throw_exception_if_processing_interrupted)
        manifest_path = job / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"GEM worker did not finish exporting results. See {job / 'inference.log'}")
        motion = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("prediction_pt", "motion_npz"):
            if not Path(motion[key]).is_file():
                raise RuntimeError(f"GEM worker reported a missing {key}: {motion[key]}")
        result = (motion, motion["motion_npz"], motion["preview_video"], str(manifest_path))
        output_dir = Path(_comfy_paths().get_output_directory()).resolve()
        files = {}
        for name, value in (("preview", motion["preview_video"]), ("motion", motion["motion_npz"]),
                            ("manifest", str(manifest_path))):
            if value:
                relative = Path(value).resolve().relative_to(output_dir)
                files[name] = {"filename": relative.name, "subfolder": str(relative.parent), "type": "output"}
        return {"ui": {"gem_result": [{"files": files, "fps": motion.get("fps"),
                                      "frames": motion.get("frame_count")}]}, "result": result}


def _character_dir():
    return (Path(_comfy_paths().models_dir) / "gem_x" / "fbx").resolve()


def _characters():
    folder = _character_dir()
    models.extract_characters(folder.parent)
    # Subfolders let a character keep its textures next to it, e.g. kyle/Robot_Color.png.
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                  if p.suffix.lower() == ".fbx" and p.is_file()) if folder.is_dir() else []


class GEMRetargetFBX:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "motion": ("GEM_MOTION",),
            "character": (_characters() or ["(put .fbx files in models/gem_x/fbx)"],),
            "root_motion": ("BOOLEAN", {"default": True,
                                        "tooltip": "Off keeps the character in place (vertical motion is kept)"}),
            "yaw_offset": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                                     "tooltip": "Extra heading in degrees; 0 starts facing the rig's forward axis"}),
            "material": (["auto", "original", "red", "orange", "gray", "blue"], {
                "default": "auto",
                "tooltip": "auto: original textures, found by file name next to the FBX "
                           "(subfolders included); solid orange if none are found, which would "
                           "otherwise render black. original never recolors"}),
        }}

    RETURN_TYPES = ("FILE_3D_FBX", "STRING", "LOAD3D_CAMERA")
    RETURN_NAMES = ("model_3d", "fbx_path", "camera_info")
    FUNCTION = "retarget"
    CATEGORY = "GEM-X"

    def retarget(self, motion, character, root_motion, yaw_offset, material="auto"):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy_api.latest import Types

        folder = _character_dir()
        target = (folder / character).resolve()
        if not target.is_relative_to(folder) or not target.is_file():
            raise ValueError(f"Character FBX not found in {folder}: {character}")
        motion_npz = Path(motion["motion_npz"])
        if not motion_npz.is_file():
            raise ValueError(f"Motion file is missing: {motion_npz}")
        models.ensure_models(folder.parent, "common", throw_exception_if_processing_interrupted)
        root, python, _ = _environment()
        worker = root / "scripts" / "comfyui_retarget.py"
        output = motion_npz.parent / f"{target.stem}_retarget_{uuid.uuid4().hex[:8]}.fbx"
        command = [python, "-u", str(worker), "--motion", str(motion_npz), "--target", str(target),
                   "--output", str(output), "--models-dir", str(folder.parent), "--yaw", str(yaw_offset),
                   "--material", material]
        if not root_motion:
            command.append("--in-place")
        _run_worker(command, root, output.with_suffix(".log"), throw_exception_if_processing_interrupted)
        if not output.is_file():
            raise RuntimeError(f"Retarget did not produce {output}")
        # The worker prints its result (incl. a camera framing the character) as JSON last.
        lines = output.with_suffix(".log").read_text(encoding="utf-8", errors="replace").splitlines()
        info = json.loads(next(line for line in reversed(lines) if line.startswith("{")))
        return (Types.File3D(str(output)), str(output), info["camera"])


NODE_CLASS_MAPPINGS = {
    "GEMXVideoPath": GEMVideoPath,
    "GEMXFramesToVideo": GEMFramesToVideo,
    "GEMXInfer": GEMInfer,
    "GEMXRetargetFBX": GEMRetargetFBX,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GEMXVideoPath": "GEM-X Video Path",
    "GEMXFramesToVideo": "GEM-X Frames to Video",
    "GEMXInfer": "GEM-X Video to SOMA Motion",
    "GEMXRetargetFBX": "GEM-X Retarget to FBX Character",
}
