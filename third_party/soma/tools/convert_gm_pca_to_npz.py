# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal script to convert GarmentMeasurements point.pca to point.npz format.

Usage:
    python convert_pca_to_npz.py <input.pca> <output.npz>

Example:
    python convert_pca_to_npz.py ../assets/GarmentMeasurements/point.pca ../assets/GarmentMeasurements/point.npz
"""

import struct
import sys

import numpy as np


def convert_pca_to_npz(input_file: str, output_file: str):
    """Convert binary PCA file to NPZ format."""
    with open(input_file, "rb") as f:
        # Read dimensions
        m = struct.unpack("I", f.read(4))[0]
        n = struct.unpack("I", f.read(4))[0]

        # Read PCA matrix (column-major order)
        pca_matrix = np.frombuffer(f.read(m * n * 8), dtype=np.float64).reshape(n, m).T

        # Read mean vector
        pca_mean = np.frombuffer(f.read(m * 8), dtype=np.float64)

        # Read eigenvalues
        eigenvalues = np.frombuffer(f.read(n * 8), dtype=np.float64)

    # Save as NPZ
    np.savez_compressed(
        output_file,
        pca_matrix=pca_matrix,
        pca_mean=pca_mean,
        eigenvalues=eigenvalues,
        dimensions=np.array([m, n], dtype=np.int32),
    )

    print(f"Converted {input_file} to {output_file}")
    print(f"Dimensions: m={m}, n={n}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    convert_pca_to_npz(sys.argv[1], sys.argv[2])
