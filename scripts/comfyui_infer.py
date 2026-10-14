"""Isolated CUDA inference worker for comfyui_gem_x; run from the GEM-X root."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--models-dir", required=True, help="ComfyUI models/gem_x directory")
    parser.add_argument("--preview", choices=("none", "keypoints", "mesh"), default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--backend", choices=("onnx", "pytorch"), default="onnx")
    parser.add_argument("--no-imgfeat", action="store_true")
    args = parser.parse_args()
    if args.backend == "onnx" and not args.no_postprocess:
        parser.error("The upstream ONNX path has no contact postprocessing; pass --no-postprocess")
    if args.backend == "pytorch" and args.no_imgfeat:
        parser.error("--no-imgfeat requires --backend onnx")

    root = Path(__file__).resolve().parents[1]
    models_dir = Path(args.models_dir).expanduser().resolve()
    os.environ["GEM_X_MODEL_ROOT"] = str(models_dir)
    # Vendored SOMA package; GEM itself is imported from the plugin root.
    sys.path[:0] = [str(root), str(root / "third_party" / "soma")]
    os.chdir(root)  # Asset paths in the upstream pipeline are relative to the checkout.
    source = Path(args.video).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (output / "manifest.json").exists() or (output / "hpe_results.pt").exists():
        raise ValueError("Use a fresh output directory to avoid stale preprocessing and predictions")
    output.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The migrated full GEM-X pipeline requires CUDA. "
            "The upstream offline scripts still contain hard-coded .cuda() calls; "
            "this adapter does not claim MPS/CPU support."
        )
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate; re-encode it with an explicit FPS")
    finally:
        cap.release()

    # Import upstream only inside this disposable process: demo_soma patches torch.load.
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    if args.backend == "onnx":
        from scripts.demo import demo_soma_onnx as demo
    else:
        from scripts.demo import demo_soma as demo

    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(config_name="demo_soma")
    # Assign paths as data rather than Hydra override expressions (commas, spaces, etc.).
    cfg.video_name = "motion"
    cfg.video_path = str(source)
    cfg.output_root = str(output.parent)
    cfg.output_dir = str(output)
    cfg.preprocess_dir = str(output / "preprocess")
    cfg.static_cam = True
    cfg.ckpt_path = str(checkpoint) if checkpoint else None
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Preserve source frame rate: the original demo copy helper writes everything at 30 fps.
    if args.backend == "onnx":
        demo.run_preprocess_fast(cfg, no_imgfeat=args.no_imgfeat)
        data = demo.load_data_dict(cfg, no_imgfeat=args.no_imgfeat)
    else:
        demo.run_preprocess(cfg)
        data = demo.load_data_dict(cfg)
    frames = int(data["length"].item())
    if frames < 2:
        raise ValueError("GEM-X needs a video containing at least two frames")
    model = None
    if args.backend == "onnx":
        # Reuse upstream preprocessing, ONNX runner, normalization and motion decoder.
        pred = demo.run_inference_fast(cfg, data, no_imgfeat=args.no_imgfeat)
    else:
        model = instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(demo.resolve_ckpt_path(cfg))
        model = model.eval().cuda()
        pred = demo.detach_to_cpu(model.predict(
            data, static_cam=True, postproc=not args.no_postprocess,
        ))
    torch.save(pred, cfg.paths.hpe_results)

    # A pickle-free exchange format; original .pt retains net_outputs for GEM consumers.
    arrays = {"fps": np.array(fps), "frame_count": np.array(frames)}
    for group in ("body_params_incam", "body_params_global"):
        for name, value in pred[group].items():
            arrays[f"{group}/{name}"] = value.numpy() if torch.is_tensor(value) else np.asarray(value)
    arrays["K_fullimg"] = pred["K_fullimg"].numpy()
    npz_path = output / "motion.npz"
    np.savez_compressed(npz_path, **arrays)

    preview = ""
    if args.preview != "none":
        preview = str(output / "0_kp2d77_overlay.mp4")
        from gem.utils.kp2d_utils import render_2d_keypoints

        render_2d_keypoints(
            video_path=cfg.video_path, vitpose_path=cfg.paths.vitpose,
            bbx_path=cfg.paths.bbx, output_path=preview, fps=fps,
        )
    if args.preview == "mesh":
        # Release inference network memory before constructing renderer/body models.
        del model
        torch.cuda.empty_cache()
        demo.render_incam(cfg, fps=fps)
        demo.render_global_o3d(cfg, fps=fps)
        demo.merge_videos_horizontal(
            [cfg.paths.incam_video, cfg.paths.global_video], cfg.paths.incam_global_horiz_video,
        )
        preview = str(cfg.paths.incam_global_horiz_video)

    if preview and (not Path(preview).is_file() or Path(preview).stat().st_size == 0):
        raise RuntimeError(f"Renderer did not produce a preview: {preview}")
    if preview:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg = get_ffmpeg_exe()
        browser_preview = output / "preview.mp4"
        subprocess.run([
            ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", preview,
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(browser_preview),
        ], check=True)
        preview = str(browser_preview)
    manifest = {
        "schema_version": 1, "body_model": "SOMA", "backend_requested": args.backend,
        "backend": "onnx_auto_with_pytorch_decoder" if args.backend == "onnx" else "pytorch_cuda",
        "image_features": not args.no_imgfeat,
        "models_dir": str(models_dir),
        "source_video": str(source), "fps": fps, "frame_count": frames,
        "camera_trajectory": "static_assumption", "seed": args.seed,
        "postprocess": not args.no_postprocess,
        "prediction_pt": str(Path(cfg.paths.hpe_results).resolve()),
        "motion_npz": str(npz_path), "preview_video": preview,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
    }
    # Manifest is the completion marker, published only when every requested output exists.
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output / "manifest.json")
    print(f"GEM-X complete: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
