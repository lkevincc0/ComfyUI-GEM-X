"""Download GEM-X weights into ComfyUI/models/gem_x on first use.

Run standalone to prefetch:  python -m comfyui_gem_x.models <ComfyUI/models/gem_x> [--all]
Each file has mirrors tried in order (HuggingFace, then ModelScope); downloads resume
from a .part file and are checked by size (and SHA256 where pinned).
"""

import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

HF_GEM = "https://huggingface.co/nvidia/GEM-X/resolve/main/"
MS_GEM = "https://modelscope.cn/models/nv-community/GEM-X/resolve/master/"
HF_SOMA = "https://huggingface.co/nvidia/SOMA-X/resolve/main/"
MS_SOMA = "https://modelscope.cn/models/nv-community/SOMA-X/resolve/master/"
# The locked SOMA code needs these two files from this commit; newer releases changed
# the SOMA_neutral.npz layout, so they come from the pinned Git LFS objects.
SOMA_PINNED = ("https://media.githubusercontent.com/media/NVlabs/SOMA-X/"
               "e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5/assets/")
YOLOX_ZIP = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
             "yolox_x_8xb8-300e_humanart-a39d44ed.zip")


def _gem(name, dest, size):
    return dest, [HF_GEM + name, MS_GEM + name], size, None


def _soma(name, size):
    return f"soma_assets/{name}", [HF_SOMA + name, MS_SOMA + name], size, None


# dest (relative to models/gem_x), mirrors, size in bytes, sha256
COMMON = [
    _gem("scale_mean.pth", "soma_data/scale_mean.pth", 1451),
    _gem("scale_comps.pth", "soma_data/scale_comps.pth", 8816),
    ("soma_assets/SOMA_neutral.npz", [SOMA_PINNED + "SOMA_neutral.npz"], 27488638,
     "515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f"),
    ("soma_assets/correctives_model.pt", [SOMA_PINNED + "correctives_model.pt"], 95437256,
     "3b0f82928dc25c952e0a23af59dbc9c02854ccb4c4c1c988f70bbffd7c284636"),
    _soma("MHR/mhr_model_lod6.pt", 2080751),
    _soma("MHR/base_body_lod6.obj", 39006),
    _soma("MHR/SOMA_wrap_lod1.obj", 1816801),
]
VITPOSE_ONNX = [
    _gem("onnx/vitpose.onnx", "onnx/vitpose.onnx", 4279801),
    _gem("onnx/vitpose.onnx.data", "onnx/vitpose.onnx.data", 3389390848),
]
SAM3D = [
    _gem("model_config.yaml", "checkpoints/sam-3d-body-dinov3/model_config.yaml", 1488),
    _gem("sam3d_body.ckpt", "checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt", 2109129346),
    _gem("mhr_model.pt", "mhr_data/mhr_model.pt", 696110248),
]
GROUPS = {
    # backend=onnx, image_features=sam3d (default)
    "onnx_sam3d": COMMON + VITPOSE_ONNX + SAM3D + [
        _gem("onnx/gem_denoiser.onnx", "onnx/gem_denoiser.onnx", 1508810),
        _gem("onnx/gem_denoiser.onnx.data", "onnx/gem_denoiser.onnx.data", 199114752),
        _gem("onnx/sam3db_backbone.onnx", "onnx/sam3db_backbone.onnx", 878918),
        _gem("onnx/sam3db_backbone.onnx.data", "onnx/sam3db_backbone.onnx.data", 3362053120),
    ],
    # backend=onnx, image_features=keypoints_only: no SAM3D downloads
    "onnx_keypoints": COMMON + VITPOSE_ONNX + [
        _gem("onnx/gem_denoiser_no_imgfeat.onnx", "onnx/gem_denoiser_no_imgfeat.onnx", 187936462),
        _gem("onnx/gem_denoiser_no_imgfeat.onnx.data", "onnx/gem_denoiser_no_imgfeat.onnx.data",
             1156259576),
    ],
    # backend=pytorch
    "pytorch": COMMON + SAM3D + [
        _gem("gem_soma.ckpt", "pretrained/gem_soma.ckpt", 541758499),
        _gem("vitpose.pth", "checkpoints/vitpose/vitpose.pth", 3388483384),
    ],
}
YOLOX_ONNX = "yolox/yolox_x_8xb8-300e_humanart-a39d44ed.onnx"


def _log(msg):
    print(f"[GEM-X models] {msg}", flush=True)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete(path, size, sha256):
    return path.is_file() and path.stat().st_size == size and (sha256 is None or _sha256(path) == sha256)


def _fetch(url, path, size, check_interrupt=None):
    """Download url to path, resuming a partial .part file."""
    part = path.with_name(path.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-GEM-X"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if have and response.status != 206:  # server ignored Range: restart
            have = 0
        with part.open("ab" if have else "wb") as out:
            done, step = have, max(size // 20, 1 << 20)
            next_report = done + step
            while block := response.read(1 << 20):
                if check_interrupt:
                    check_interrupt()
                out.write(block)
                done += len(block)
                if done >= next_report:
                    _log(f"  {path.name}: {done / 1e6:.0f}/{size / 1e6:.0f} MB")
                    next_report += step
    return part


def _download(entry, root, check_interrupt=None):
    dest, mirrors, size, sha256 = entry
    path = root / dest
    if _complete(path, size, sha256):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in mirrors:
        _log(f"downloading {dest} ({size / 1e6:.1f} MB) from {url.split('/')[2]}")
        try:
            part = _fetch(url, path, size, check_interrupt)
        except Exception as exc:  # try the next mirror
            if check_interrupt and type(exc).__name__ == "InterruptProcessingException":
                raise
            errors.append(f"{url}: {exc}")
            continue
        if _complete(part, size, sha256):
            part.replace(path)
            return
        errors.append(f"{url}: size/checksum mismatch")
        part.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {dest}; place it manually in {path.parent}.\n"
                       + "\n".join(errors))


def _download_yolox(root, check_interrupt=None):
    path = root / YOLOX_ONNX
    if path.is_file() and path.stat().st_size > 0:
        return
    archive = path.with_suffix(".zip")
    _download((str(archive.relative_to(root)), [YOLOX_ZIP], 368200375, None), root, check_interrupt)
    with zipfile.ZipFile(archive) as bundle:
        member = next(n for n in bundle.namelist() if n.endswith("end2end.onnx"))
        with bundle.open(member) as src, path.with_suffix(".part").open("wb") as dst:
            shutil.copyfileobj(src, dst)
    path.with_suffix(".part").replace(path)
    archive.unlink()


def group_for(backend, image_features):
    if backend == "pytorch":
        return "pytorch"
    return "onnx_keypoints" if image_features == "keypoints_only" else "onnx_sam3d"


def ensure_models(models_dir, group, check_interrupt=None):
    """Download whatever the group still needs; 'common' is SOMA body data only."""
    root = Path(models_dir)
    for entry in COMMON if group == "common" else GROUPS[group]:
        _download(entry, root, check_interrupt)
    if group != "common":
        _download_yolox(root, check_interrupt)


CHARACTERS_ZIP = Path(__file__).resolve().parents[1] / "assets" / "characters.zip"


def extract_characters(models_dir):
    """Unpack the bundled rigged characters (fbx/...) into models/gem_x; keeps existing files."""
    root = Path(models_dir)
    if not CHARACTERS_ZIP.is_file():
        return
    with zipfile.ZipFile(CHARACTERS_ZIP) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if member.is_dir() or target.exists() or not target.is_relative_to(root.resolve()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            _log(f"installed {member.filename}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    groups = list(GROUPS) if "--all" in sys.argv else ["onnx_sam3d"]
    seen = set()
    for group in groups:
        for entry in GROUPS[group]:
            if entry[0] not in seen:
                seen.add(entry[0])
                _download(entry, root)
    _download_yolox(root)
    _log(f"all models present in {root}")


if __name__ == "__main__":
    main()
