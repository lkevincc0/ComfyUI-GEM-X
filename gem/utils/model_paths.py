"""Model storage selected by the embedding application, or inputs for standalone demos."""

import os
from pathlib import Path


def model_path(*parts):
    return str(Path(os.environ.get("GEM_X_MODEL_ROOT", "inputs")).joinpath(*parts))
