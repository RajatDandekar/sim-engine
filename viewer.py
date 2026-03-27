"""Interactive scene viewer — launch with: python -m sim_engine view

Uses OpenCV for display (works on all platforms without mjpython).
Press SPACE to start next episode, 'q' to quit.
Shows main webcam view + wrist camera picture-in-picture.
"""

import sys
import numpy as np
import mujoco

from . import config
from .renderer import OffscreenRenderer


def main():
    print("Loading SO-101 scene...")

    try:
        import cv2
    except ImportError:
        print("Viewer requires opencv: pip install opencv-python")
        sys.exit(1)

    from .env import SO101PickPlaceEnv
    from .policies.scripted_policy import ScriptedPolicy

    env = SO101PickPlaceEnv(render_images=False)
    policy = ScriptedPolicy(env)

    # Two renderers: main view + wrist camera
    main_renderer = OffscreenRenderer(env.model, 960, 720)
    wrist_renderer = OffscreenRenderer(env.model, 320, 240)

    episode = 0
    print("Viewer running")
    print("  SPACE = start next episode")
    print("  q     = quit")

    obs, info = env.reset(seed=42)
    policy.reset()
    state = "waiting"  # "running", "done", "waiting"
    step = 0

    # PiP settings
    pip_w, pip_h = 240, 180
    pip_margin = 10
    pip_border = 2

    while True:
        # Render main webcam view
        frame = main_renderer.render(env.data, "webcam")
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Render wrist camera and overlay as picture-in-picture
        wrist_frame = wrist_renderer.render(env.data, "arm_cam")
        wrist_bgr = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
        wrist_resized = cv2.resize(wrist_bgr, (pip_w, pip_h))

        # Position: bottom-right corner
        y1 = frame_bgr.shape[0] - pip_h - pip_margin
        y2 = y1 + pip_h
        x1 = frame_bgr.shape[1] - pip_w - pip_margin
        x2 = x1 + pip_w

        # Draw border
        cv2.rectangle(frame_bgr, (x1 - pip_border, y1 - pip_border),
                       (x2 + pip_border, y2 + pip_border), (200, 200, 200), pip_border)
        # Overlay wrist camera
        frame_bgr[y1:y2, x1:x2] = wrist_resized
        # Label
        cv2.putText(frame_bgr, "Wrist Cam", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Overlay status text
        if state == "waiting":
            msg = "Press SPACE to start"
            color = (100, 200, 255)
        elif state == "running":
            msg = f"Episode {episode + 1} | Step {step}"
            color = (0, 255, 0)
        elif state == "done":
            result = "SUCCESS" if info.get("success") else "DONE"
            msg = f"Episode {episode + 1} | {result} — press SPACE for next"
            color = (0, 255, 255) if info.get("success") else (0, 150, 255)

        cv2.putText(frame_bgr, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("SimEngine Viewer", frame_bgr)
        key = cv2.waitKey(33)

        if key == ord("q") or key == 27:
            break

        if state == "waiting":
            if key == ord(" "):
                state = "running"
                episode += 1
                step = 0

        elif state == "running":
            action = policy.select_action({
                "state": obs["state"],
                "images": {},
                "task": config.DEFAULT_TASK,
            })
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if policy._phase > 10 or truncated:
                state = "done"

        elif state == "done":
            if key == ord(" "):
                obs, info = env.reset(seed=np.random.randint(10000))
                policy.reset()
                state = "running"
                episode += 1
                step = 0

    cv2.destroyAllWindows()
    wrist_renderer.close()
    main_renderer.close()
    env.close()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
