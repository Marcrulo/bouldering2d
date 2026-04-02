# Bouldering2D

A 2D bouldering reinforcement-learning environment built with Gymnasium and Pygame, trained with PPO (Stable-Baselines3).

## Features

- Stick-figure climber with articulated joints and anatomical joint limits
- Persistent gravity and stamina depletion
- Attach/release control for hands and feet on generated holds
- Camera illusion: player stays centered while wall features scroll
- PPO training script with `tqdm` and periodic human-render validation
- Random-agent baseline script for dynamics sanity checks

## Setup (uv)

```bash
uv sync
```

## Run Random Agent

```bash
uv run python scripts/random_agent.py --episodes 10
```

Render mode:

```bash
uv run python scripts/random_agent.py --episodes 3 --render
```

## Train PPO

```bash
uv run python scripts/train_ppo.py --total-timesteps 300000 --eval-interval 10000 --eval-episodes 2 --run-name baseline
```

Headless validation with saved eval videos:

```bash
uv run python scripts/train_ppo.py --total-timesteps 300000 --eval-interval 10000 --eval-episodes 2 --eval-render-mode rgb_array --save-eval-video --run-name baseline
```

Artifacts are written under:

- `data/checkpoints/<run-name>/`
- `data/logs/<run-name>/`
- `data/videos/<run-name>/`

## Evaluate a Trained Model

```bash
uv run python scripts/eval_policy.py --model data/checkpoints/baseline/ppo_best.zip --episodes 5 --render human
```

Save RGB-array video:

```bash
uv run python scripts/eval_policy.py --model data/checkpoints/baseline/ppo_best.zip --episodes 3 --render rgb_array --video data/videos/baseline/eval.mp4
```

## Tests

```bash
uv run pytest -q
```
