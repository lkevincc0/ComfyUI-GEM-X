# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types
from typing import Optional, Tuple

import numpy as np


def default_pyopengl_platform() -> str:
    """Default PyOpenGL platform: empty string on Windows (use system OpenGL), 'egl' on Linux (headless)."""
    return "" if sys.platform == "win32" else "egl"


def _ensure_headless_pyrender() -> None:
    """Mock pyrender.viewer so pyrender can be imported without pyglet/X11 (headless/EGL only)."""
    if "pyrender.viewer" in sys.modules:
        return
    viewer_mod = types.ModuleType("pyrender.viewer")
    viewer_mod.Viewer = type("Viewer", (), {})  # dummy so "from .viewer import Viewer" succeeds
    sys.modules["pyrender.viewer"] = viewer_mod


def set_pyopengl_platform(platform: Optional[str]) -> None:
    if platform:
        os.environ.setdefault("PYOPENGL_PLATFORM", platform)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    z_axis = eye - target
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        z_norm = 1.0
    z_axis /= z_norm

    up = up / (np.linalg.norm(up) + 1e-8)
    x_axis = np.cross(up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        # Fallback if up is parallel to view direction.
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(up, z_axis)) > 0.9:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x_axis = np.cross(up, z_axis)
        x_norm = np.linalg.norm(x_axis)
    x_axis /= x_norm + 1e-8
    y_axis = np.cross(z_axis, x_axis)

    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = eye
    return T


def compute_camera_pose(vertices: np.ndarray, cam_dist_scale: float = 2.5) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    if not np.all(np.isfinite(verts)):
        raise ValueError("Vertices contain NaN or inf; cannot compute camera pose.")
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = (vmin + vmax) * 0.5
    extent = np.linalg.norm(vmax - vmin) + 1e-6
    eye = center + np.array([0.0, 0.0, max(extent * cam_dist_scale, 1.0)], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return look_at(eye, center, up)


def _compute_vertex_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals (pure numpy, no trimesh dependency)."""
    v0 = positions[faces[:, 0]]
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vertex_normals = np.zeros_like(positions)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals /= np.maximum(norms, 1e-8)
    return vertex_normals


class MeshRenderer:
    def __init__(
        self,
        image_size: int = 512,
        bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        focal_length: float = 4000.0,
        light_intensity: float = 3.0,
    ):
        _ensure_headless_pyrender()
        import pyrender

        self.pyrender = pyrender
        self.image_size = image_size
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=image_size, viewport_height=image_size
        )
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.05, 0.05, 0.05))

        self.camera = pyrender.IntrinsicsCamera(
            fx=focal_length, fy=focal_length, cx=image_size / 2, cy=image_size / 2
        )
        self.camera_node = self.scene.add(self.camera, pose=np.eye(4))

        self.light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
        self.light_node = self.scene.add(self.light, pose=np.eye(4))

        self.mesh_node = None
        self._cached_faces = None
        self._cached_material = None
        self._cached_vertex_colors = None

    def setup_mesh(
        self,
        faces: np.ndarray,
        mesh_color: Tuple[float, float, float, float] = (0.69, 0.39, 0.96, 1.0),
        cam_pose: Optional[np.ndarray] = None,
        light_dir: Optional[np.ndarray] = None,
        metallic: float = 0.0,
        roughness: float = 0.5,
        base_color_factor: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ):
        """Pre-compute scene state that stays constant across a frame sequence.

        Call once before a render_frame() loop.  Camera pose, light pose,
        face topology, material, and vertex colours are set here and reused
        for every subsequent render_frame() call.
        """
        if cam_pose is not None:
            self.scene.set_pose(self.camera_node, pose=cam_pose)
        if light_dir is not None:
            light_pose = look_at(np.zeros(3), light_dir, np.array([0.0, 1.0, 0.0]))
            self.scene.set_pose(self.light_node, pose=light_pose)
        elif cam_pose is not None:
            self.scene.set_pose(self.light_node, pose=cam_pose)

        self._cached_faces = np.asarray(faces, dtype=np.int32)
        if len(mesh_color) == 3:
            mesh_color = (*mesh_color, 1.0)
        self._cached_mesh_color = np.array(mesh_color, dtype=np.float32)
        self._cached_material = self.pyrender.MetallicRoughnessMaterial(
            metallicFactor=metallic,
            roughnessFactor=roughness,
            baseColorFactor=base_color_factor,
            doubleSided=True,
        )
        self._cached_vertex_colors = None

    def render_frame(self, vertices: np.ndarray) -> np.ndarray:
        """Fast render path — only vertex positions change.

        Reuses the faces / material / colours / camera set by setup_mesh().
        Builds a pyrender Primitive directly (skips trimesh object creation
        and Mesh.from_trimesh overhead).
        """
        verts = np.asarray(vertices, dtype=np.float32)

        if self._cached_vertex_colors is None or self._cached_vertex_colors.shape[0] != len(verts):
            self._cached_vertex_colors = np.tile(self._cached_mesh_color, (len(verts), 1))

        normals = _compute_vertex_normals(verts, self._cached_faces)

        primitive = self.pyrender.Primitive(
            positions=verts,
            normals=normals,
            indices=self._cached_faces,
            color_0=self._cached_vertex_colors,
            material=self._cached_material,
            mode=4,  # TRIANGLES
        )
        mesh = self.pyrender.Mesh(primitives=[primitive])

        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(mesh)

        color, _ = self.renderer.render(self.scene)
        return color

    def render(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh_color: Tuple[float, float, float, float] = (0.69, 0.39, 0.96, 1.0),
        cam_pose: Optional[np.ndarray] = None,
        light_dir: Optional[np.ndarray] = None,
        metallic: float = 0.0,
        roughness: float = 0.5,
        base_color_factor: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> np.ndarray:
        import trimesh

        # Update Camera Pose
        if cam_pose is None:
            cam_pose = compute_camera_pose(vertices)
        self.scene.set_pose(self.camera_node, pose=cam_pose)

        # Update Light Pose
        if light_dir is not None:
            light_pose = look_at(np.zeros(3), light_dir, np.array([0.0, 1.0, 0.0]))
            self.scene.set_pose(self.light_node, pose=light_pose)
        else:
            self.scene.set_pose(self.light_node, pose=cam_pose)

        # Update Mesh
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)

        if len(mesh_color) == 3:
            mesh_color = (*mesh_color, 1.0)

        # Create Trimesh
        verts = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        vertex_colors = np.tile(mesh_color, (len(verts), 1))

        tmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        tmesh.visual.vertex_colors = vertex_colors

        # Create Pyrender Mesh
        mesh = self.pyrender.Mesh.from_trimesh(tmesh, smooth=True)

        # Apply custom material properties
        material = self.pyrender.MetallicRoughnessMaterial(
            metallicFactor=metallic,
            roughnessFactor=roughness,
            baseColorFactor=base_color_factor,
            doubleSided=True,
        )
        for prim in mesh.primitives:
            prim.material = material

        self.mesh_node = self.scene.add(mesh)

        # Render
        color, _ = self.renderer.render(self.scene)
        return color

    def delete(self):
        self.renderer.delete()


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    image_size: int = 512,
    bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    mesh_color: Tuple[float, float, float, float] = (0.69, 0.39, 0.96, 1.0),
    focal_length: float = 4000.0,
    light_intensity: float = 3.0,
    cam_pose=None,
    light_dir: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    WARNING: This is inefficient for loops as it creates/destroys the renderer every time.
    Use MeshRenderer class instead.
    """
    renderer = MeshRenderer(
        image_size=image_size,
        bg_color=bg_color,
        focal_length=focal_length,
        light_intensity=light_intensity,
    )
    color = renderer.render(
        vertices, faces, mesh_color=mesh_color, cam_pose=cam_pose, light_dir=light_dir
    )
    renderer.delete()
    return color


def render_comparison_video(
    out_path: str,
    verts_source: np.ndarray,
    faces_source: np.ndarray,
    verts_soma: np.ndarray,
    faces_soma: np.ndarray,
    *,
    color_source: Tuple[float, float, float, float] = (0.98, 0.65, 0.15, 1.0),
    color_soma: Tuple[float, float, float, float] = (0.55, 0.15, 0.85, 1.0),
    cam_pose: Optional[np.ndarray] = None,
    light_dir: Optional[np.ndarray] = None,
    image_size: int = 1024,
    fps: int = 30,
    center: bool = False,
    cam_dist_scale: float = 2.5,
    label_source: str = "Source",
    label_soma: str = "SOMA",
) -> None:
    """Render a blended comparison video of source vs SOMA meshes.

    Uses the fast setup_mesh() + render_frame() path with a streaming
    video writer (no frame accumulation in memory).

    Args:
        out_path: Output video file path (e.g. 'out/comparison.mp4').
        verts_source: (N, V_src, 3) source model vertices.
        verts_soma: (N, V_soma, 3) SOMA reconstruction vertices.
        faces_source: (F_src, 3) source face indices.
        faces_soma: (F_soma, 3) SOMA face indices.
        color_source: RGBA colour for source mesh.
        color_soma: RGBA colour for SOMA mesh.
        cam_pose: 4x4 camera pose. Auto-computed from first frame if None.
        light_dir: Light direction vector. Defaults to (0, -0.5, -1).
        image_size: Render resolution (square).
        fps: Video frame rate.
        center: Subtract per-frame centroid from both meshes so the
            character stays centered.  Useful for motions with large
            global translations (e.g. SAM 3D Body).
        label_source: Name for source mesh (for print message).
        label_soma: Name for SOMA mesh (for print message).
    """
    import imageio.v2 as imageio
    from tqdm import tqdm

    N = len(verts_source)
    faces_source = np.asarray(faces_source, dtype=np.int32)
    faces_soma = np.asarray(faces_soma, dtype=np.int32)

    if center:
        # Per-frame centroid from source mesh; apply to both so overlay stays aligned
        centroids = verts_source.mean(axis=1, keepdims=True)  # (N, 1, 3)
        verts_source = verts_source - centroids
        verts_soma = verts_soma - centroids

    if cam_pose is None:
        cam_pose = compute_camera_pose(verts_source[0], cam_dist_scale=cam_dist_scale)
    if light_dir is None:
        light_dir = np.array([0.0, -0.5, -1.0])

    render_kwargs = dict(
        cam_pose=cam_pose,
        light_dir=light_dir,
        metallic=0.0,
        roughness=0.5,
        base_color_factor=[0.9, 0.9, 0.9, 1.0],
    )
    renderer = MeshRenderer(image_size=image_size, light_intensity=5)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps)

    for t in tqdm(range(N), desc="Rendering"):
        renderer.setup_mesh(faces=faces_source, mesh_color=color_source, **render_kwargs)
        img_source = renderer.render_frame(verts_source[t])

        renderer.setup_mesh(faces=faces_soma, mesh_color=color_soma, **render_kwargs)
        img_soma = renderer.render_frame(verts_soma[t])

        img_combined = (0.5 * img_source + 0.5 * img_soma).astype(np.uint8)
        writer.append_data(img_combined[..., ::-1])

    writer.close()
    renderer.delete()
    print(f"\nSaved: {out_path}  ({label_source}=orange, {label_soma}=purple)")


def save_image(path: str, image: np.ndarray) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError(
            "imageio is required to save images. Install with `pip install imageio`."
        ) from exc

    imageio.imwrite(path, image)
