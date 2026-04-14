"""Range-of-motion inspector.

Cycles through every joint one at a time, sweeping from its minimum limit to
its maximum limit while all other joints stay at their rest pose.  The player
levitates at a fixed height with no holds in the background.

Controls
--------
  SPACE / right-arrow  – skip to next joint immediately
  left-arrow           – go back to previous joint
  Q / Escape           – quit
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bouldering2d.config import EnvConfig, PlayerConfig
from bouldering2d.environment.contacts import ContactManager
from bouldering2d.environment.muscle import MuscleState
from bouldering2d.environment.player_model import JOINT_LIMITS, JOINT_ORDER, PlayerModel
from bouldering2d.environment.renderer import RenderContext, Renderer

_FRESH_MUSCLES = MuscleState(fatigue={j: 0.0 for j in JOINT_ORDER})

# How many frames to spend sweeping each joint (min → max).
SWEEP_FRAMES = 120
FPS = 30


def _make_rest_player() -> PlayerModel:
    player = PlayerModel(PlayerConfig())
    player.reset(x=4.0, y=0.8)
    return player


def _draw_overlay(surface: pygame.Surface, joint_name: str, angle: float, lo: float, hi: float, idx: int, total: int) -> None:
    font_large = pygame.font.Font(None, 42)
    font_small = pygame.font.Font(None, 28)
    w = surface.get_width()

    # Joint name
    label = font_large.render(f"Joint: {joint_name}", True, (20, 20, 20))
    surface.blit(label, (w - label.get_width() - 16, 14))

    # Index
    idx_label = font_small.render(f"({idx + 1}/{total})", True, (80, 80, 80))
    surface.blit(idx_label, (w - idx_label.get_width() - 16, 56))

    # Angle value and range
    angle_label = font_small.render(f"angle: {angle:+.3f} rad   [{lo:+.2f} … {hi:+.2f}]", True, (40, 40, 40))
    surface.blit(angle_label, (w - angle_label.get_width() - 16, 84))

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = w - 220, 112, 200, 10
    pygame.draw.rect(surface, (180, 180, 180), (bar_x, bar_y, bar_w, bar_h))
    progress = (angle - lo) / (hi - lo) if hi > lo else 0.0
    pygame.draw.rect(surface, (39, 105, 179), (bar_x, bar_y, int(bar_w * progress), bar_h))

    # Controls hint
    hint = font_small.render("SPACE/→ next   ← prev   Q quit", True, (120, 120, 120))
    surface.blit(hint, (w - hint.get_width() - 16, surface.get_height() - 30))


def main() -> None:
    env_config = EnvConfig()
    player_config = PlayerConfig()

    pygame.init()
    screen = pygame.display.set_mode((env_config.screen_width, env_config.screen_height))
    pygame.display.set_caption("Range-of-Motion Inspector")
    clock = pygame.time.Clock()

    renderer = Renderer(env_config)
    contacts = ContactManager(player_config)  # no holds attached

    joint_idx = 0
    frame = 0
    running = True

    while running:
        # --- events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                    joint_idx = (joint_idx + 1) % len(JOINT_ORDER)
                    frame = 0
                elif event.key == pygame.K_LEFT:
                    joint_idx = (joint_idx - 1) % len(JOINT_ORDER)
                    frame = 0

        # --- update ---
        name = JOINT_ORDER[joint_idx]
        lo, hi = JOINT_LIMITS[name]

        # Ping-pong: sweep min→max then max→min, repeat.
        cycle = frame % (SWEEP_FRAMES * 2)
        t = cycle / SWEEP_FRAMES  # 0→2
        if t <= 1.0:
            angle = lo + (hi - lo) * t
        else:
            angle = hi - (hi - lo) * (t - 1.0)

        player = _make_rest_player()
        player.joint_angles[name] = angle

        endpoints = player.limb_endpoints()
        ctx = RenderContext(
            player=player,
            endpoints=endpoints,
            holds=[],
            observed_holds=[],
            contacts=contacts,
            stamina=100.0,
            step_count=frame,
            ascent=0.0,
            muscle_fatigue=_FRESH_MUSCLES,
        )

        # Draw scene directly (bypasses renderer's flip/tick so we can add overlay first)
        renderer._draw_scene(screen, ctx)
        _draw_overlay(screen, name, angle, lo, hi, joint_idx, len(JOINT_ORDER))
        pygame.display.flip()

        frame += 1
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
