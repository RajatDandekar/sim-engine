"""Offscreen camera rendering using MuJoCo's built-in renderer."""

import mujoco
import numpy as np


class OffscreenRenderer:
    """Renders camera images from MuJoCo scenes."""

    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480):
        self.model = model
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height, width)

    def render(self, data: mujoco.MjData, camera_name: str) -> np.ndarray:
        """Render an RGB image from the named camera.

        Returns:
            np.ndarray of shape (height, width, 3), dtype uint8
        """
        self._renderer.update_scene(data, camera=camera_name)
        return self._renderer.render().copy()

    def close(self):
        """Clean up renderer resources."""
        if hasattr(self, '_renderer'):
            self._renderer.close()
