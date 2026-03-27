"""Record demonstration episodes — run with: python -m sim_engine record"""

import argparse
import numpy as np

from . import config
from .env import SO101PickPlaceEnv
from .dataset import DatasetRecorder
from .policies.scripted_policy import ScriptedPolicy
from .policies.random_policy import RandomPolicy


def record_scripted(num_episodes=10, save_dir=None, task=None, render=False):
    """Record episodes using the scripted pick-and-place policy."""
    env = SO101PickPlaceEnv(render_mode="human" if render else None)
    recorder = DatasetRecorder(save_dir=save_dir)
    policy = ScriptedPolicy(env)

    successes = 0
    for ep in range(num_episodes):
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        obs, info = env.reset()
        policy.reset()
        recorder.start_episode(task=task)

        for step in range(config.MAX_EPISODE_STEPS):
            action = policy.select_action({"state": obs["state"], "images": obs.get("images", {}), "task": task or config.DEFAULT_TASK})
            recorder.record_step(obs, action)
            obs, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()

            if terminated:
                print(f"  Success at step {step}!")
                successes += 1
                break

        recorder.end_episode(success=info.get("success", False))

    env.close()
    print(f"\nRecording complete: {successes}/{num_episodes} successful ({100*successes/num_episodes:.0f}%)")
    print(f"Episodes saved to: {recorder.save_dir}")


def record_keyboard(save_dir=None, task=None):
    """Record episodes via keyboard teleoperation.

    Controls:
        Q/A: shoulder_pan +/-
        W/S: shoulder_lift +/-
        E/D: elbow_flex +/-
        R/F: wrist_flex +/-
        T/G: wrist_roll +/-
        Space: toggle gripper
        Enter: end episode
        Escape: quit
    """
    try:
        import pynput
    except ImportError:
        print("Keyboard mode requires pynput: pip install pynput")
        return

    from pynput import keyboard

    env = SO101PickPlaceEnv(render_mode="human")
    recorder = DatasetRecorder(save_dir=save_dir)

    # Key state
    action_delta = np.zeros(7, dtype=np.float32)
    gripper_open = True
    done_episode = False
    quit_all = False
    step_size = 0.15

    key_map = {
        'q': (0, +step_size), 'a': (0, -step_size),
        'w': (1, +step_size), 's': (1, -step_size),
        'e': (2, +step_size), 'd': (2, -step_size),
        'r': (3, +step_size), 'f': (3, -step_size),
        't': (4, +step_size), 'g': (4, -step_size),
    }

    def on_press(key):
        nonlocal gripper_open, done_episode, quit_all, action_delta
        try:
            k = key.char.lower()
            if k in key_map:
                idx, delta = key_map[k]
                action_delta[idx] = delta
        except AttributeError:
            if key == keyboard.Key.space:
                gripper_open = not gripper_open
            elif key == keyboard.Key.enter:
                done_episode = True
            elif key == keyboard.Key.esc:
                quit_all = True

    def on_release(key):
        nonlocal action_delta
        try:
            k = key.char.lower()
            if k in key_map:
                idx, _ = key_map[k]
                action_delta[idx] = 0.0
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\nKeyboard Teleoperation Mode")
    print("Controls: Q/A W/S E/D R/F T/G for joints, Space=gripper, Enter=end episode, Esc=quit")

    episode_num = 0
    while not quit_all:
        episode_num += 1
        print(f"\nEpisode {episode_num} — press Enter to end, Esc to quit")
        obs, info = env.reset()
        done_episode = False

        # Build a persistent action that accumulates
        current_action = np.zeros(7, dtype=np.float32)
        recorder.start_episode(task=task)

        for step in range(config.MAX_EPISODE_STEPS):
            if done_episode or quit_all:
                break

            # Accumulate deltas
            current_action[:5] = np.clip(current_action[:5] + action_delta[:5], -1.0, 1.0)
            current_action[5] = 1.0 if gripper_open else -1.0
            current_action[6] = current_action[5]

            recorder.record_step(obs, current_action)
            obs, reward, terminated, truncated, info = env.step(current_action)
            env.render()

        recorder.end_episode(success=info.get("success", False))

    listener.stop()
    env.close()
    print("\nKeyboard recording finished.")


def main():
    parser = argparse.ArgumentParser(description="Record demonstration episodes")
    parser.add_argument("--mode", choices=["scripted", "keyboard"], default="scripted")
    parser.add_argument("-n", "--num_episodes", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--task", type=str, default=config.DEFAULT_TASK)
    parser.add_argument("--render", action="store_true", help="Show viewer during scripted recording")
    args = parser.parse_args()

    if args.mode == "scripted":
        record_scripted(args.num_episodes, args.save_dir, args.task, args.render)
    else:
        record_keyboard(args.save_dir, args.task)


if __name__ == "__main__":
    main()
