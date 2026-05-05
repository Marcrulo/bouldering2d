from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
    effective_render_mode = "rgb_array" if args.video else render_mode

    vec_env = DummyVecEnv([lambda: make_env(render_mode=effective_render_mode)])
    normalizer_path = Path(args.model).parent / "vec_normalize.pkl"
    if normalizer_path.exists():
        vec_env = VecNormalize.load(str(normalizer_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model)

    frames: list = []
    rewards = []
    ascents = []
    for _ in tqdm(range(args.episodes), desc="Policy Eval", unit="ep"):
        obs = vec_env.reset()
        done = np.array([False])
        ep_reward = 0.0
        best_ascent = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += float(reward[0])
            best_ascent = max(best_ascent, float(info[0].get("ascent", 0.0)))

            if effective_render_mode is not None:
                frame = vec_env.render()
                if args.video:
                    if isinstance(frame, np.ndarray):
                        frames.append(frame)
                    elif isinstance(frame, list) and frame:
                        frames.append(frame[0])

        rewards.append(ep_reward)
        ascents.append(best_ascent)

    if args.video and frames:
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=30)

    vec_env.close()
    print(f"episodes={args.episodes}")
    print(f"mean_reward={sum(rewards)/len(rewards):.3f}")
    print(f"mean_ascent={sum(ascents)/len(ascents):.3f}")


if __name__ == "__main__":
    main()
