from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bouldering2d.environment.climbing_env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random policy on Bouldering2D")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true", help="Disable human rendering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_env(render_mode=None if args.no_render else "human")
    rng = np.random.default_rng(42)

    rewards = []
    ascents = []
    spans = []
    for _ in tqdm(range(args.episodes), desc="Random Agent", unit="ep"):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        best_ascent = 0.0
        y_min = float(env.player.pelvis[1])
        y_max = float(env.player.pelvis[1])

        # Use correlated random torques so joints keep moving instead of canceling each frame.
        joint_action = np.zeros(env.joint_action_dim, dtype=np.float32)
        grip_action = np.full(4, 0.8, dtype=np.float32)

        while not (done or truncated):
            noise = rng.normal(0.0, 0.35, size=env.joint_action_dim).astype(np.float32)
            joint_action = np.clip(0.85 * joint_action + noise, -1.0, 1.0)

            # Mostly keep attachments on; occasionally release/re-grab one limb.
            if rng.random() < 0.08:
                idx = int(rng.integers(0, 4))
                grip_action[idx] = -0.9 if rng.random() < 0.4 else 0.9
            else:
                grip_action = np.clip(0.9 * grip_action + 0.1 * 0.8, -1.0, 1.0)

            action = np.concatenate([joint_action, grip_action]).astype(np.float32)
            obs, reward, done, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            ep_reward += reward
            best_ascent = max(best_ascent, float(info.get("ascent", 0.0)))
            y = float(env.player.pelvis[1])
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        rewards.append(ep_reward)
        ascents.append(best_ascent)
        spans.append(y_max - y_min)

    env.close()
    print(f"episodes={args.episodes}")
    print(f"mean_reward={sum(rewards)/len(rewards):.3f}")
    print(f"mean_ascent={sum(ascents)/len(ascents):.3f}")
    print(f"mean_y_span={sum(spans)/len(spans):.3f}")


if __name__ == "__main__":
    main()
