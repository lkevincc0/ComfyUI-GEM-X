"""ComfyUI-GEM-X installer, run by ComfyUI-Manager after cloning (or manually: python install.py).

1. Unpacks assets/characters.zip (rigged FBX characters) into ComfyUI/models/gem_x/.
2. Creates the inference environment .venv (reusing ComfyUI's PyTorch) and installs
   requirements-worker.txt. Set GEM_X_SKIP_VENV=1 to skip, e.g. when GEM_X_PYTHON points
   to an existing environment.

Model weights are not downloaded here; nodes fetch what the chosen mode needs on first run.
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def models_dir():
    if os.environ.get("COMFYUI_MODELS_DIR"):
        return Path(os.environ["COMFYUI_MODELS_DIR"]) / "gem_x"
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "gem_x"
    except ImportError:  # ComfyUI/custom_nodes/<this plugin>
        return ROOT.parents[1] / "models" / "gem_x"


def main():
    sys.path.insert(0, str(ROOT))
    from comfyui_gem_x.models import extract_characters

    target = models_dir()
    print(f"[GEM-X install] characters -> {target / 'fbx'}", flush=True)
    extract_characters(target)

    if os.environ.get("GEM_X_SKIP_VENV"):
        return
    venv = ROOT / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        print(f"[GEM-X install] creating {venv}", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
    subprocess.check_call([str(python), "-m", "pip", "install", "-r",
                           str(ROOT / "requirements-worker.txt")])
    print("[GEM-X install] done", flush=True)


if __name__ == "__main__":
    main()
