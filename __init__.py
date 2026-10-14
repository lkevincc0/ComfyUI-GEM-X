"""ComfyUI custom-node entry point when the complete GEM-X checkout is installed."""

from .comfyui_gem_x import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./comfyui_gem_x/web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
