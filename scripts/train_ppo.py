from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bouldering2d.config import TrainingConfig
from bouldering2d.environment.climbing_env import make_env


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_bar: tqdm | None = None
        self._last = 0

    def _on_training_start(self) -> None:
        self._last = self.num_timesteps
        self.progress_bar = tqdm(
            total=self.total_timesteps,
            initial=self.num_timesteps,
            desc="PPO Training",
            unit="steps",
        )

    def _on_step(self) -> bool:
        if self.progress_bar is None:
            return True
        current = self.num_timesteps
        delta = max(0, current - self._last)
        if delta:
            self.progress_bar.update(delta)
            self._last = current
        return True

    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_interval: int,
        eval_episodes: int,
        eval_render_mode: str,
        save_eval_video: bool,
        checkpoint_dir: Path,
        video_dir: Path,
    ):
        super().__init__()
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_render_mode = eval_render_mode
        self.save_eval_video = save_eval_video
        self.checkpoint_dir = checkpoint_dir
        self.video_dir = video_dir
        self.best_mean_reward = float("-inf")
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_interval:
            return True
        self._last_eval_step = self.num_timesteps

        metrics = run_eval(
            model=self.model,
            episodes=self.eval_episodes,
            render_mode=self.eval_render_mode,
            save_video=self.save_eval_video,
            video_path=self.video_dir / f"eval_{self.num_timesteps}.mp4",
        )
        mean_reward = metrics["mean_reward"]
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ascent", metrics["mean_ascent"])

        checkpoint_path = self.checkpoint_dir / f"ppo_step_{self.num_timesteps}.zip"
        self.model.save(checkpoint_path)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(self.checkpoint_dir / "ppo_best.zip")

        return True

    def _on_training_start(self) -> None:
        # Align eval scheduling when resuming so we keep evaluating every eval_interval steps.
        self._last_eval_step = (self.num_timesteps // self.eval_interval) * self.eval_interval


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    step_pattern = re.compile(r"^ppo_step_\d+\.zip$")
    candidates = [path for path in checkpoint_dir.glob("*.zip") if step_pattern.match(path.name)]

    final_checkpoint = checkpoint_dir / "ppo_final.zip"
    if final_checkpoint.exists():
        candidates.append(final_checkpoint)

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_eval(model: PPO, episodes: int, render_mode: str, save_video: bool, video_path: Path) -> dict[str, float]:
    env = make_env(render_mode=render_mode)
    rewards = []
    ascents = []
    frames: list[np.ndarray] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        ascent = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if save_video and isinstance(frame, np.ndarray):
                frames.append(frame)
            total_reward += reward
            ascent = max(ascent, float(info.get("ascent", 0.0)))
        rewards.append(total_reward)
        ascents.append(ascent)

    env.close()
    if save_video and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=30)

    return {
        "mean_reward": float(sum(rewards) / max(1, len(rewards))),
        "mean_ascent": float(sum(ascents) / max(1, len(ascents))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Bouldering2D")
    parser.add_argument("--total-timesteps", type=int, default=TrainingConfig().total_timesteps)
    parser.add_argument("--eval-interval", type=int, default=TrainingConfig().eval_interval)
    parser.add_argument("--eval-episodes", type=int, default=TrainingConfig().eval_episodes)
    parser.add_argument("--eval-render-mode", choices=["human", "rgb_array"], default="human")
    parser.add_argument("--save-eval-video", action="store_true")
    parser.add_argument("--run-name", type=str, default="ppo_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(
        total_timesteps=args.total_timesteps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )

    checkpoint_dir = ROOT / cfg.checkpoint_dir / args.run_name
    log_dir = ROOT / cfg.tensorboard_log_dir / args.run_name
    video_dir = ROOT / cfg.video_dir / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    with (checkpoint_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(asdict(cfg), fp, indent=2)

    env = DummyVecEnv([lambda: make_env(render_mode=None, seed=cfg.seed)])
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    is_resumed_run = latest_checkpoint is not None

    if latest_checkpoint is not None:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        model = PPO.load(str(latest_checkpoint), env=env, device="cpu")
        model.tensorboard_log = str(log_dir)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            tensorboard_log=str(log_dir),
            policy_kwargs=cfg.policy_kwargs,
            verbose=0,
            seed=cfg.seed,
            device="cpu",
        )

    timesteps_to_train = cfg.total_timesteps
    if is_resumed_run:
        current_steps = int(model.num_timesteps)
        timesteps_to_train = max(0, cfg.total_timesteps - current_steps)
        print(
            f"Run progress: current={current_steps}, target={cfg.total_timesteps}, remaining={timesteps_to_train}"
        )

    callbacks = CallbackList(
        [
            TqdmCallback(total_timesteps=cfg.total_timesteps),
            PeriodicEvalCallback(
                eval_interval=cfg.eval_interval,
                eval_episodes=cfg.eval_episodes,
                eval_render_mode=args.eval_render_mode,
                save_eval_video=args.save_eval_video,
                checkpoint_dir=checkpoint_dir,
                video_dir=video_dir,
            ),
        ]
    )

    if timesteps_to_train > 0:
        model.learn(
            total_timesteps=timesteps_to_train,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=not is_resumed_run,
        )
    else:
        print("Target timesteps already reached; skipping training.")

    model.save(checkpoint_dir / "ppo_final.zip")
    env.close()


if __name__ == "__main__":
    main()
