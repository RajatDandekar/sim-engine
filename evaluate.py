"""Evaluation / inference runner — run with: python -m sim_engine eval"""

import argparse
import os
import numpy as np

from . import config
from .env import SO101PickPlaceEnv


def evaluate(policy_type, checkpoint=None, num_episodes=10, render=False,
             record_video=False, task=None, device="cpu"):
    """Run a policy in simulation and report success metrics."""
    # Load policy
    if policy_type == "random":
        from .policies.random_policy import RandomPolicy
        policy = RandomPolicy()
    elif policy_type == "scripted":
        env_tmp = SO101PickPlaceEnv(render_images=False)
        from .policies.scripted_policy import ScriptedPolicy
        policy = ScriptedPolicy(env_tmp)
        # Will re-assign env below
    elif policy_type == "mlp":
        from .policies.mlp_policy import MLPPolicy
        policy = MLPPolicy(device=device)
        if checkpoint:
            policy.load(checkpoint)
            print(f"Loaded checkpoint: {checkpoint}")
        else:
            print("Warning: no checkpoint specified, using random MLP weights")
    else:
        print(f"Policy '{policy_type}' not yet implemented.")
        print("Available: random, scripted, mlp")
        return

    render_mode = "human" if render else None
    env = SO101PickPlaceEnv(render_mode=render_mode, render_images=(record_video or render))
    task = task or config.DEFAULT_TASK

    # Re-assign env for scripted policy
    if policy_type == "scripted":
        from .policies.scripted_policy import ScriptedPolicy
        policy = ScriptedPolicy(env)

    # Video recording setup
    video_frames = []

    successes = 0
    total_reward = 0.0
    total_steps = 0

    print(f"\nEvaluating '{policy_type}' for {num_episodes} episodes...")
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        policy.reset()
        ep_reward = 0.0

        for step in range(config.MAX_EPISODE_STEPS):
            action = policy.select_action({
                "state": obs["state"],
                "images": obs.get("images", {}),
                "task": task,
            })
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if render:
                env.render()

            if record_video and "images" in obs:
                video_frames.append(obs["images"]["webcam"])

            if terminated or truncated:
                break

        total_steps += step + 1
        total_reward += ep_reward
        if info.get("success", False):
            successes += 1
            status = "SUCCESS"
        else:
            status = "FAIL"
        print(f"  Ep {ep+1}: {status} | steps={step+1} | reward={ep_reward:.2f}")

    env.close()

    # Print summary
    print(f"\n{'='*40}")
    print(f"Policy: {policy_type}")
    print(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.0f}%)")
    print(f"Avg reward: {total_reward/num_episodes:.2f}")
    print(f"Avg steps: {total_steps/num_episodes:.0f}")

    # Save video
    if record_video and video_frames:
        try:
            import cv2
            video_path = os.path.join(config.DEFAULT_OUTPUT_DIR, f"eval_{policy_type}.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            h, w = video_frames[0].shape[:2]
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            for frame in video_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"Video saved: {video_path}")
        except ImportError:
            print("Install opencv-python to save videos")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy in simulation")
    parser.add_argument("--policy", choices=["random", "scripted", "mlp", "act", "diffusion", "smolvla"],
                        default="scripted")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("-n", "--num_episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--task", type=str, default=config.DEFAULT_TASK)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate(args.policy, args.checkpoint, args.num_episodes, args.render,
             args.record_video, args.task, args.device)


if __name__ == "__main__":
    main()
