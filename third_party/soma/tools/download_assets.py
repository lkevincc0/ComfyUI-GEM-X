# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse


def download_assets(target_dir=None, revision="main"):
    """Download SOMA assets from HuggingFace.

    Args:
        target_dir: If provided, used as the HuggingFace cache directory.
            The actual assets will be stored in a subdirectory managed by
            huggingface_hub.  If None, uses the default HF cache
            (``~/.cache/huggingface/hub/``).
        revision: Git revision (branch, tag, or commit hash) to download.

    Returns:
        Path to the downloaded assets directory.
    """
    from soma.assets import get_assets_dir

    path = get_assets_dir(revision=revision, cache_dir=target_dir)
    print(f"Assets downloaded to: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SOMA assets from HuggingFace")
    parser.add_argument(
        "--target-dir",
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface/hub/)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision to download (default: main)",
    )
    args = parser.parse_args()
    download_assets(target_dir=args.target_dir, revision=args.revision)
