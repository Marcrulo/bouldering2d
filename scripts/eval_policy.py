from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3 import PPO
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bouldering2d.environment.climbing_env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", choices=["human", "rgb_array", "none"], default="human")
    parser.add_argument("--video", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = None if args.render == "none" else args.render
    env = make_env(render_mode=render_mode)
    model = PPO.load(args.model)

    frames: list = []
    rewards = []
    ascents = []
    for _ in tqdm(range(args.episodes), desc="Policy Eval", unit="ep"):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        best_ascent = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            frame = env.render() if render_mode else None
            if frame is not None and args.video:
                frames.append(frame)
            ep_reward += reward
            best_ascent = max(best_ascent, float(info.get("ascent", 0.0)))

        rewards.append(ep_reward)
        ascents.append(best_ascent)

    if args.video and frames:
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=30)

    env.close()
    print(f"episodes={args.episodes}")
    print(f"mean_reward={sum(rewards)/len(rewards):.3f}")
    print(f"mean_ascent={sum(ascents)/len(ascents):.3f}")


if __name__ == "__main__":
    main()
