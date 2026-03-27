"""SimEngine App -full UI for record, train, evaluate.

Launch with: python -m sim_engine app
"""

import sys
import os
import time
import threading
import numpy as np

from . import config


def main():
    try:
        import cv2
    except ImportError:
        print("Requires opencv: pip install opencv-python")
        sys.exit(1)

    from .env import SO101PickPlaceEnv
    from .renderer import OffscreenRenderer

    # ──────────────────────────────────────────────
    # Policy scanner
    # ──────────────────────────────────────────────

    def _scan_policies():
        """Scan for available policies and their checkpoints.
        Returns list of dicts: {name, key, file, implemented, checkpoint}
        """
        policies_dir = os.path.join(os.path.dirname(__file__), "policies")
        output_dir = config.DEFAULT_OUTPUT_DIR
        results = []
        key_idx = 0
        key_chars = "abcdefghij"

        for f in sorted(os.listdir(policies_dir)):
            if not f.endswith("_policy.py") or f == "base.py":
                continue
            name = f.replace("_policy.py", "")
            if name in ("random", "scripted"):
                continue  # handled separately

            fpath = os.path.join(policies_dir, f)
            with open(fpath) as fh:
                content = fh.read()
            implemented = "NotImplementedError" not in content

            # Look for checkpoint
            ckpt_dir = os.path.join(output_dir, name)
            checkpoint = None
            if os.path.exists(ckpt_dir):
                for ck in ("best.pt", "final.pt"):
                    p = os.path.join(ckpt_dir, ck)
                    if os.path.exists(p):
                        checkpoint = p
                        break

            results.append({
                "name": name,
                "key": key_chars[key_idx] if key_idx < len(key_chars) else "",
                "file": f,
                "implemented": implemented,
                "checkpoint": checkpoint,
            })
            key_idx += 1

        return results

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    WINDOW = "SimEngine"
    W, H = 960, 720
    BG = (30, 30, 35)
    WHITE = (220, 220, 220)
    GREEN = (0, 220, 100)
    YELLOW = (0, 220, 255)
    CYAN = (200, 200, 0)
    RED = (60, 60, 255)
    GRAY = (120, 120, 120)
    ORANGE = (0, 140, 255)

    def blank():
        return np.full((H, W, 3), BG, dtype=np.uint8)

    def put(img, text, pos, color=WHITE, scale=0.6, thick=1):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    def put_center(img, text, y, color=WHITE, scale=0.7, thick=2):
        sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
        x = (W - sz[0]) // 2
        put(img, text, (x, y), color, scale, thick)

    def draw_menu(title, items, footer=None):
        """Render a menu screen. items = [(key, label, color), ...]"""
        img = blank()
        put_center(img, title, 60, CYAN, 1.0, 2)
        cv2.line(img, (100, 80), (W - 100, 80), GRAY, 1)

        y = 140
        for key, label, color in items:
            put(img, f"[{key}]", (120, y), YELLOW, 0.7, 2)
            put(img, label, (200, y), color, 0.65, 1)
            y += 50

        if footer:
            put_center(img, footer, H - 30, GRAY, 0.5, 1)
        return img

    # ──────────────────────────────────────────────
    # Screens
    # ──────────────────────────────────────────────

    def main_menu():
        """Main menu loop. Returns the chosen action key."""
        while True:
            img = draw_menu("SimEngine  - SO-101 Pick & Place Simulator", [
                ("1", "Record Episodes        -collect demonstration data", WHITE),
                ("2", "Manage Policies         -view / add policy architectures", WHITE),
                ("3", "Train                   -train a policy on recorded data", WHITE),
                ("4", "Evaluate                -run a policy in simulation", WHITE),
                ("5", "Watch Demo              -watch the scripted policy", WHITE),
                ("q", "Quit", GRAY),
            ], "SimEngine v1.0  |  MuJoCo physics  |  Gymnasium env")

            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(100)
            if key == ord("1"): return "record"
            if key == ord("2"): return "policies"
            if key == ord("3"): return "train"
            if key == ord("4"): return "evaluate"
            if key == ord("5"): return "demo"
            if key == ord("q") or key == 27: return "quit"

    # ──────────────────────────────────────────────
    # 1. RECORD
    # ──────────────────────────────────────────────

    def record_screen():
        from .policies.scripted_policy import ScriptedPolicy
        from .dataset import DatasetRecorder

        while True:
            save_dir = config.DEFAULT_DATASET_DIR
            n_existing = len([f for f in os.listdir(save_dir) if f.endswith(".hdf5")]) if os.path.exists(save_dir) else 0

            img = draw_menu("Record Episodes", [
                ("s", f"Start Recording  (scripted policy, 10 episodes)", GREEN),
                ("n", f"Dataset: {save_dir}", WHITE),
                ("", f"Existing episodes: {n_existing}", GRAY),
                ("", "", WHITE),
                ("", "The scripted policy will automatically perform", GRAY),
                ("", "pick-and-place. Each episode is saved as HDF5.", GRAY),
                ("", "Images from both cameras are recorded.", GRAY),
                ("", "", WHITE),
                ("b", "Back to menu", GRAY),
            ])
            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(100)

            if key == ord("b") or key == 27: return
            if key == ord("s"):
                _run_recording(cv2, 10, save_dir)

    def _run_recording(cv2, num_episodes, save_dir):
        from .policies.scripted_policy import ScriptedPolicy
        from .dataset import DatasetRecorder

        env = SO101PickPlaceEnv(render_images=True)
        policy = ScriptedPolicy(env)
        recorder = DatasetRecorder(save_dir=save_dir)
        renderer = OffscreenRenderer(env.model, W, H)
        wrist_r = OffscreenRenderer(env.model, 320, 240)

        pip_w, pip_h = 200, 150
        successes = 0

        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep * 7 + int(time.time()) % 100)
            policy.reset()
            recorder.start_episode(task=config.DEFAULT_TASK)
            ep_done = False

            for step in range(config.MAX_EPISODE_STEPS):
                action = policy.select_action({
                    "state": obs["state"], "images": obs.get("images", {}),
                    "task": config.DEFAULT_TASK,
                })
                recorder.record_step(obs, action)
                obs, reward, terminated, truncated, info = env.step(action)

                # Render
                frame = renderer.render(env.data, "webcam")
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Wrist PiP
                wrist = wrist_r.render(env.data, "arm_cam")
                wrist_bgr = cv2.resize(cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR), (pip_w, pip_h))
                y1 = H - pip_h - 10
                x1 = W - pip_w - 10
                frame_bgr[y1:y1+pip_h, x1:x1+pip_w] = wrist_bgr
                put(frame_bgr, "Wrist", (x1, y1 - 5), GRAY, 0.4, 1)

                # Status bar
                status = f"RECORDING  |  Episode {ep+1}/{num_episodes}  |  Step {step}  |  Success: {successes}/{ep}"
                put(frame_bgr, status, (10, 30), RED, 0.65, 2)

                # Recording indicator (blinking red dot)
                if (step // 10) % 2 == 0:
                    cv2.circle(frame_bgr, (W - 30, 30), 10, (0, 0, 255), -1)

                cv2.imshow(WINDOW, frame_bgr)
                k = cv2.waitKey(20)
                if k == 27:  # ESC to abort
                    recorder.end_episode(success=False)
                    renderer.close()
                    wrist_r.close()
                    env.close()
                    return

                if policy._phase > 10 or truncated:
                    ep_done = True
                    break

            success = info.get("success", False)
            if success:
                successes += 1
            recorder.end_episode(success=success)

        renderer.close()
        wrist_r.close()
        env.close()

        # Show summary
        _show_message(cv2, f"Recording Complete!  {successes}/{num_episodes} successful",
                      f"Saved to: {save_dir}", 3000)

    # ──────────────────────────────────────────────
    # 2. POLICIES
    # ──────────────────────────────────────────────

    def policies_screen():
        policies_dir = os.path.join(os.path.dirname(__file__), "policies")
        while True:
            policies = _scan_policies()

            items = []
            key_map = {}

            for p in policies:
                fpath = os.path.join(policies_dir, p["file"])
                if p["implemented"]:
                    status = "[implemented]"
                    color = GREEN
                else:
                    status = "[skeleton -- press key to open in editor]"
                    color = YELLOW
                key_map[p["key"]] = fpath
                items.append((p["key"], f"{p['name'].upper():12s}  {status}", color))

            items += [
                ("", "", WHITE),
                ("", "Press a key above to open that policy in your editor.", CYAN),
                ("", "Implement select_action(obs) -> action (7,)", WHITE),
                ("", "Then go to Train to train it on recorded demos.", WHITE),
                ("", "", WHITE),
                ("", "Observation keys:", CYAN),
                ("", "  state:  (7,)  joint angles (deg) + gripper (0-100)", WHITE),
                ("", "  images: webcam (480x640x3) + arm_cam (480x640x3)", WHITE),
                ("", "  task:   str   language instruction", WHITE),
                ("", "Action:   (7,) in [-1, 1]  5 joints + gripper", WHITE),
                ("", "", WHITE),
                ("n", "Create new policy file", ORANGE),
                ("o", "Open policies folder", GRAY),
                ("b", "Back to menu", GRAY),
            ]

            img = draw_menu("Policies", items)
            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(100)

            if key == ord("b") or key == 27: return
            if key == ord("o"):
                os.system(f"open '{policies_dir}' 2>/dev/null || xdg-open '{policies_dir}' 2>/dev/null")
            if key == ord("n"):
                _create_new_policy(cv2, policies_dir)

            pressed = chr(key) if 0 <= key < 128 else ""
            if pressed in key_map:
                fpath = key_map[pressed]
                # Open file in VS Code, or fallback to system editor
                os.system(f"code '{fpath}' 2>/dev/null || open '{fpath}' 2>/dev/null || xdg-open '{fpath}' 2>/dev/null")
                _show_message(cv2, f"Opened {os.path.basename(fpath)} in editor",
                              "Edit, save, then come back to Train", 2000)

    def _create_new_policy(cv2, policies_dir):
        """Generate a new policy file from template."""
        # Simple name input via a fixed list
        names = ["custom", "transformer", "cnn", "rnn", "hybrid"]
        idx = [0]

        while True:
            img = blank()
            put_center(img, "Create New Policy", 60, CYAN, 0.9, 2)
            cv2.line(img, (100, 80), (W - 100, 80), GRAY, 1)

            put(img, "Select a name (UP/DOWN arrows, ENTER to create):", (120, 140), WHITE, 0.55, 1)

            for i, name in enumerate(names):
                y = 190 + i * 40
                color = GREEN if i == idx[0] else GRAY
                prefix = " >> " if i == idx[0] else "    "
                put(img, f"{prefix}{name}_policy.py", (160, y), color, 0.6, 2 if i == idx[0] else 1)

            put_center(img, "ESC to cancel", H - 30, GRAY, 0.45, 1)
            cv2.imshow(WINDOW, img)

            k = cv2.waitKey(100)
            if k == 27: return
            if k == 0:  # up arrow (platform dependent)
                idx[0] = max(0, idx[0] - 1)
            if k == 1:  # down arrow
                idx[0] = min(len(names) - 1, idx[0] + 1)
            # Arrow keys on macOS/OpenCV
            if k == 82 or k == 0xFF52:  # up
                idx[0] = max(0, idx[0] - 1)
            if k == 84 or k == 0xFF54:  # down
                idx[0] = min(len(names) - 1, idx[0] + 1)
            if k == 13 or k == 10:  # enter
                chosen = names[idx[0]]
                _write_policy_template(policies_dir, chosen)
                fpath = os.path.join(policies_dir, f"{chosen}_policy.py")
                os.system(f"code '{fpath}' 2>/dev/null || open '{fpath}' 2>/dev/null")
                _show_message(cv2, f"Created {chosen}_policy.py!", "Opened in editor. Implement select_action().", 3000)
                return

    def _write_policy_template(policies_dir, name):
        """Write a new policy file from template."""
        class_name = "".join(w.capitalize() for w in name.split("_")) + "Policy"
        fpath = os.path.join(policies_dir, f"{name}_policy.py")
        if os.path.exists(fpath):
            return  # don't overwrite

        content = f'''"""Custom policy: {name.upper()}

Implement your own policy architecture here.
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BasePolicy


class {class_name}Network(nn.Module):
    """Your neural network architecture."""

    def __init__(self, state_dim=7, action_dim=7, hidden_dim=256):
        super().__init__()
        # TODO: Define your model layers
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class {class_name}(BasePolicy):
    """Your custom policy.

    Modify __init__ and select_action to implement your approach.
    The model attribute must exist for training to work.
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = {class_name}Network().to(self.device)
        self.state_mean = np.zeros(7, dtype=np.float32)
        self.state_std = np.ones(7, dtype=np.float32)
        self.action_mean = np.zeros(7, dtype=np.float32)
        self.action_std = np.ones(7, dtype=np.float32)

    def select_action(self, observation: dict) -> np.ndarray:
        state = observation["state"]
        state_norm = (state - self.state_mean) / self.state_std
        state_t = torch.from_numpy(state_norm).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_norm = self.model(state_t).cpu().numpy()[0]

        action = action_norm * self.action_std + self.action_mean
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def set_normalization(self, stats: dict):
        self.state_mean = stats["state_mean"]
        self.state_std = stats["state_std"]
        self.action_mean = stats["action_mean"]
        self.action_std = stats["action_std"]

    def reset(self):
        pass

    def save(self, path: str):
        torch.save({{
            "model": self.model.state_dict(),
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
        }}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.state_mean = ckpt["state_mean"]
        self.state_std = ckpt["state_std"]
        self.action_mean = ckpt["action_mean"]
        self.action_std = ckpt["action_std"]
'''
        with open(fpath, "w") as f:
            f.write(content)

    # ──────────────────────────────────────────────
    # 3. TRAIN
    # ──────────────────────────────────────────────

    def train_screen():
        data_dir = config.DEFAULT_DATASET_DIR
        n_eps = len([f for f in os.listdir(data_dir) if f.endswith(".hdf5")]) if os.path.exists(data_dir) else 0

        if n_eps == 0:
            _show_message(cv2, "No episodes recorded yet!", "Record episodes first (option 1)", 2000)
            return

        while True:
            policies = _scan_policies()

            items = [
                ("", f"Dataset: {n_eps} episodes  |  {data_dir}", GREEN),
                ("", "", WHITE),
            ]

            key_map = {}
            for p in policies:
                if p["implemented"]:
                    label = f"{p['name'].upper():12s}  [ready to train]"
                    color = GREEN
                    key_map[p["key"]] = p["name"]
                else:
                    label = f"{p['name'].upper():12s}  [not implemented yet]"
                    color = GRAY
                items.append((p["key"], label, color))

            items += [
                ("", "", WHITE),
                ("", "How to add your own policy:", CYAN),
                ("", "  1. Edit sim_engine/policies/<name>_policy.py", WHITE),
                ("", "  2. Implement select_action() and the model", WHITE),
                ("", "  3. Come back here -- it will show [ready]", WHITE),
                ("", "", WHITE),
                ("", "100 epochs | batch=64 | lr=3e-4 | ESC to stop", GRAY),
                ("", "", WHITE),
                ("b", "Back to menu", GRAY),
            ]

            img = draw_menu("Train Policy", items)
            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(100)

            if key == ord("b") or key == 27: return

            pressed = chr(key) if 0 <= key < 128 else ""
            if pressed in key_map:
                _run_training(cv2, data_dir, key_map[pressed])

    def _run_training(cv2, data_dir, policy_name="mlp"):
        from .dataset import SimDataset
        import torch
        import importlib

        output_dir = os.path.join(config.DEFAULT_OUTPUT_DIR, policy_name)
        os.makedirs(output_dir, exist_ok=True)

        # Show loading
        img = blank()
        put_center(img, f"Loading dataset for {policy_name.upper()}...", H // 2, YELLOW, 0.8, 2)
        cv2.imshow(WINDOW, img)
        cv2.waitKey(1)

        dataset = SimDataset(data_dir=data_dir, load_images=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        stats = dataset.get_normalization_stats()

        # Dynamically load the policy module
        try:
            mod = importlib.import_module(f".policies.{policy_name}_policy", package="sim_engine")
            # Find the policy class (first class that has 'set_normalization' or ends with 'Policy')
            policy_cls = None
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if isinstance(obj, type) and attr_name.endswith("Policy") and attr_name != "BasePolicy":
                    policy_cls = obj
                    break
            if policy_cls is None:
                _show_message(cv2, f"No Policy class found in {policy_name}_policy.py", "", 3000)
                return
            policy = policy_cls(device="cpu")
        except Exception as e:
            _show_message(cv2, f"Error loading {policy_name}: {str(e)[:60]}", "Check your implementation", 4000)
            return

        if hasattr(policy, "set_normalization"):
            policy.set_normalization(stats)

        optimizer = torch.optim.Adam(policy.model.parameters(), lr=3e-4)
        loss_fn = torch.nn.MSELoss()

        epochs = 100
        best_loss = float("inf")
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            for batch in dataloader:
                states = batch["state"].float()
                actions = batch["action"].float()
                pred = policy.model(states)
                loss = loss_fn(pred, actions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1

            avg_loss = epoch_loss / max(n, 1)
            loss_history.append(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                policy.save(os.path.join(output_dir, "best.pt"))

            # Render training progress
            img = blank()
            put_center(img, f"Training {policy_name.upper()} Policy", 50, CYAN, 0.9, 2)

            # Progress bar
            progress = (epoch + 1) / epochs
            bar_x, bar_y, bar_w, bar_h = 100, 100, W - 200, 30
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GRAY, 1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), GREEN, -1)
            put(img, f"Epoch {epoch+1}/{epochs}  ({100*progress:.0f}%)", (bar_x, bar_y - 10), WHITE, 0.5, 1)

            # Stats
            put(img, f"Loss: {avg_loss:.6f}", (120, 180), WHITE, 0.65, 1)
            put(img, f"Best: {best_loss:.6f}", (120, 215), GREEN, 0.65, 1)
            put(img, f"Samples: {len(dataset)}", (120, 250), GRAY, 0.55, 1)

            # Loss curve
            if len(loss_history) > 1:
                chart_x, chart_y, chart_w, chart_h = 100, 290, W - 200, 300
                cv2.rectangle(img, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), GRAY, 1)
                put(img, "Loss Curve", (chart_x, chart_y - 5), GRAY, 0.45, 1)

                max_loss = max(loss_history)
                min_loss = min(loss_history)
                loss_range = max(max_loss - min_loss, 1e-6)

                pts = []
                for i, l in enumerate(loss_history):
                    px = chart_x + int(i / max(len(loss_history) - 1, 1) * chart_w)
                    py = chart_y + chart_h - int((l - min_loss) / loss_range * chart_h)
                    pts.append((px, py))

                for i in range(1, len(pts)):
                    cv2.line(img, pts[i-1], pts[i], GREEN, 2)

            put_center(img, "Press ESC to stop early", H - 30, GRAY, 0.45, 1)

            cv2.imshow(WINDOW, img)
            k = cv2.waitKey(1)
            if k == 27:
                break

        policy.save(os.path.join(output_dir, "final.pt"))
        _show_message(cv2, f"Training Complete!  Best loss: {best_loss:.6f}",
                      f"Saved to: {output_dir}/best.pt", 3000)

    # ──────────────────────────────────────────────
    # 4. EVALUATE
    # ──────────────────────────────────────────────

    def evaluate_screen():
        while True:
            policies = _scan_policies()

            items = [
                ("s", "Scripted         [baseline, ~90% success]", GREEN),
                ("r", "Random           [sanity check]", WHITE),
            ]

            key_map = {"s": ("scripted", None), "r": ("random", None)}

            for p in policies:
                if p["checkpoint"]:
                    label = f"{p['name'].upper():12s}  [trained: {os.path.basename(p['checkpoint'])}]"
                    color = YELLOW
                    key_map[p["key"]] = (p["name"], p["checkpoint"])
                elif p["implemented"]:
                    label = f"{p['name'].upper():12s}  [no checkpoint -- train first]"
                    color = GRAY
                else:
                    label = f"{p['name'].upper():12s}  [not implemented]"
                    color = GRAY
                items.append((p["key"] if p["checkpoint"] else "", label, color))

            items += [
                ("", "", WHITE),
                ("", "SPACE starts each episode. 10 episodes.", GRAY),
                ("", "", WHITE),
                ("b", "Back to menu", GRAY),
            ]

            img = draw_menu("Evaluate Policy", items)
            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(100)

            if key == ord("b") or key == 27: return

            pressed = chr(key) if 0 <= key < 128 else ""
            if pressed in key_map:
                pname, ckpt = key_map[pressed]
                _run_eval(cv2, pname, ckpt)

    def _run_eval(cv2, policy_type, checkpoint, num_episodes=10):
        import importlib

        env = SO101PickPlaceEnv(render_images=False)
        renderer = OffscreenRenderer(env.model, W, H)
        wrist_r = OffscreenRenderer(env.model, 320, 240)

        # Load policy
        if policy_type == "scripted":
            from .policies.scripted_policy import ScriptedPolicy
            policy = ScriptedPolicy(env)
        elif policy_type == "random":
            from .policies.random_policy import RandomPolicy
            policy = RandomPolicy()
        else:
            try:
                mod = importlib.import_module(f".policies.{policy_type}_policy", package="sim_engine")
                policy_cls = None
                for attr_name in dir(mod):
                    obj = getattr(mod, attr_name)
                    if isinstance(obj, type) and attr_name.endswith("Policy") and attr_name != "BasePolicy":
                        policy_cls = obj
                        break
                if policy_cls is None:
                    _show_message(cv2, f"No Policy class found in {policy_type}_policy.py", "", 3000)
                    renderer.close(); wrist_r.close(); env.close()
                    return
                policy = policy_cls()
                if checkpoint:
                    policy.load(checkpoint)
            except Exception as e:
                _show_message(cv2, f"Error: {str(e)[:70]}", "Check your implementation", 4000)
                renderer.close(); wrist_r.close(); env.close()
                return

        pip_w, pip_h = 200, 150
        successes = 0
        total_reward = 0

        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep * 3)
            policy.reset()
            ep_reward = 0
            waiting = True

            # Show "press SPACE" before each episode
            while waiting:
                frame = renderer.render(env.data, "webcam")
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Wrist PiP
                wrist = wrist_r.render(env.data, "arm_cam")
                wrist_bgr = cv2.resize(cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR), (pip_w, pip_h))
                y1 = H - pip_h - 10
                x1 = W - pip_w - 10
                frame_bgr[y1:y1+pip_h, x1:x1+pip_w] = wrist_bgr
                put(frame_bgr, "Wrist", (x1, y1 - 5), GRAY, 0.4, 1)

                msg = f"Episode {ep+1}/{num_episodes}  |  {policy_type.upper()}  |  Success: {successes}/{ep}"
                put(frame_bgr, msg, (10, 30), CYAN, 0.65, 2)
                put_center(frame_bgr, "Press SPACE to run", H // 2, YELLOW, 0.8, 2)

                cv2.imshow(WINDOW, frame_bgr)
                k = cv2.waitKey(50)
                if k == ord(" "): waiting = False
                if k == 27:
                    renderer.close(); wrist_r.close(); env.close()
                    return

            # Run episode
            for step in range(config.MAX_EPISODE_STEPS):
                action = policy.select_action({
                    "state": obs["state"], "images": {},
                    "task": config.DEFAULT_TASK,
                })
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                frame = renderer.render(env.data, "webcam")
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                wrist = wrist_r.render(env.data, "arm_cam")
                wrist_bgr = cv2.resize(cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR), (pip_w, pip_h))
                y1 = H - pip_h - 10
                x1 = W - pip_w - 10
                frame_bgr[y1:y1+pip_h, x1:x1+pip_w] = wrist_bgr
                put(frame_bgr, "Wrist", (x1, y1 - 5), GRAY, 0.4, 1)

                msg = f"Episode {ep+1}/{num_episodes}  |  {policy_type.upper()}  |  Step {step}  |  Reward: {ep_reward:.1f}"
                put(frame_bgr, msg, (10, 30), GREEN, 0.65, 2)

                cv2.imshow(WINDOW, frame_bgr)
                k = cv2.waitKey(20)
                if k == 27:
                    renderer.close(); wrist_r.close(); env.close()
                    return

                done = False
                if policy_type == "scripted":
                    done = getattr(policy, "_phase", 0) > 10 or truncated
                else:
                    done = terminated or truncated

                if done:
                    break

            if info.get("success"):
                successes += 1
            total_reward += ep_reward

        renderer.close()
        wrist_r.close()
        env.close()

        # Summary
        _show_message(cv2,
            f"Evaluation Complete  - {policy_type.upper()}",
            f"Success: {successes}/{num_episodes} ({100*successes/num_episodes:.0f}%)  |  Avg reward: {total_reward/num_episodes:.1f}",
            5000)

    # ──────────────────────────────────────────────
    # 5. DEMO (same as old viewer)
    # ──────────────────────────────────────────────

    def demo_screen():
        from .policies.scripted_policy import ScriptedPolicy

        env = SO101PickPlaceEnv(render_images=False)
        policy = ScriptedPolicy(env)
        renderer = OffscreenRenderer(env.model, W, H)
        wrist_r = OffscreenRenderer(env.model, 320, 240)

        obs, info = env.reset(seed=42)
        policy.reset()
        state = "waiting"
        episode = 0
        step = 0
        pip_w, pip_h = 200, 150

        while True:
            frame = renderer.render(env.data, "webcam")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Wrist PiP
            wrist = wrist_r.render(env.data, "arm_cam")
            wrist_bgr = cv2.resize(cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR), (pip_w, pip_h))
            y1 = H - pip_h - 10
            x1 = W - pip_w - 10
            frame_bgr[y1:y1+pip_h, x1:x1+pip_w] = wrist_bgr
            put(frame_bgr, "Wrist", (x1, y1 - 5), GRAY, 0.4, 1)

            if state == "waiting":
                msg = "Press SPACE to start  |  ESC = back to menu"
                put(frame_bgr, msg, (10, 30), YELLOW, 0.6, 2)
            elif state == "running":
                put(frame_bgr, f"Episode {episode}  |  Step {step}", (10, 30), GREEN, 0.65, 2)
            elif state == "done":
                result = "SUCCESS!" if info.get("success") else "DONE"
                put(frame_bgr, f"Episode {episode}  |  {result}  -SPACE for next", (10, 30), CYAN, 0.6, 2)

            cv2.imshow(WINDOW, frame_bgr)
            key = cv2.waitKey(33)

            if key == 27: break

            if state == "waiting":
                if key == ord(" "):
                    state = "running"
                    episode += 1
                    step = 0
            elif state == "running":
                action = policy.select_action({
                    "state": obs["state"], "images": {}, "task": config.DEFAULT_TASK,
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

        renderer.close()
        wrist_r.close()
        env.close()

    # ──────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────

    def _show_message(cv2, line1, line2, duration_ms=3000):
        img = blank()
        put_center(img, line1, H // 2 - 20, GREEN, 0.8, 2)
        put_center(img, line2, H // 2 + 30, WHITE, 0.55, 1)
        put_center(img, "Press any key to continue", H - 40, GRAY, 0.45, 1)
        cv2.imshow(WINDOW, img)
        cv2.waitKey(duration_ms)

    # ──────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, W, H)

    while True:
        choice = main_menu()
        if choice == "quit":
            break
        elif choice == "record":
            record_screen()
        elif choice == "policies":
            policies_screen()
        elif choice == "train":
            train_screen()
        elif choice == "evaluate":
            evaluate_screen()
        elif choice == "demo":
            demo_screen()

    cv2.destroyAllWindows()
    print("SimEngine closed.")


if __name__ == "__main__":
    main()
