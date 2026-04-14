"""Muscle-strength inspector.

For each joint, sweeps through its full range of motion while displaying a
torque-angle curve panel that shows:

  - The raw angle-dependent strength profile (Gaussian curve, no fatigue)
  - The fatigue-reduced effective strength (dashed line)
  - A vertical marker at the current joint angle
  - Numeric readout of current effective strength vs. peak

The stick figure's limbs are colour-coded by current effective strength
(blue = full strength → red = very weak).

Controls
--------
  SPACE / right-arrow  – next joint
  left-arrow           – previous joint
  F                    – add fatigue to current muscle (+0.2)
  R                    – reset all fatigue to zero
  Q / Escape           – quit
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bouldering2d.config import EnvConfig, MuscleConfig, PlayerConfig
from bouldering2d.environment.contacts import ContactManager
from bouldering2d.environment.muscle import MuscleModel, MuscleState
from bouldering2d.environment.player_model import JOINT_LIMITS, JOINT_ORDER, PlayerModel
from bouldering2d.environment.renderer import RenderContext, Renderer

SWEEP_FRAMES = 150
FPS = 30

# Panel dimensions (drawn bottom-right of screen)
PANEL_W = 340
PANEL_H = 220
PANEL_MARGIN = 16
CURVE_PAD = 30   # padding inside panel for axis labels


def _make_rest_player() -> PlayerModel:
    player = PlayerModel(PlayerConfig())
    player.reset(x=4.0, y=0.8)
    return player


def _lerp_color(
    c0: tuple[int, int, int],
    c1: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(c0[0] + (c1[0] - c0[0]) * t),
        int(c0[1] + (c1[1] - c0[1]) * t),
        int(c0[2] + (c1[2] - c0[2]) * t),
    )


def _strength_color(strength_fraction: float) -> tuple[int, int, int]:
    """Blue (1.0) → amber (0.5) → red (0.0)."""
    if strength_fraction >= 0.5:
        t = (1.0 - strength_fraction) * 2.0
        return _lerp_color((60, 120, 200), (220, 160, 30), t)
    else:
        t = (0.5 - strength_fraction) * 2.0
        return _lerp_color((220, 160, 30), (210, 35, 35), t)


def _draw_curve_panel(
    surface: pygame.Surface,
    joint: str,
    current_angle: float,
    muscle_model: MuscleModel,
    screen_w: int,
    screen_h: int,
) -> None:
    """Draw the torque-angle curve panel for the given joint."""
    lo, hi = JOINT_LIMITS[joint]
    cfg = muscle_model.config
    fatigue = muscle_model.state.fatigue[joint]

    # Panel background
    px = screen_w - PANEL_W - PANEL_MARGIN
    py = screen_h - PANEL_H - PANEL_MARGIN
    pygame.draw.rect(surface, (240, 232, 218), pygame.Rect(px, py, PANEL_W, PANEL_H), border_radius=6)
    pygame.draw.rect(surface, (140, 125, 108), pygame.Rect(px, py, PANEL_W, PANEL_H), 1, border_radius=6)

    # Chart area within the panel
    cx = px + CURVE_PAD
    cy = py + 12
    cw = PANEL_W - CURVE_PAD - 10
    ch = PANEL_H - 55  # leave room for legend below

    # Axis
    pygame.draw.line(surface, (100, 90, 80), (cx, cy + ch), (cx + cw, cy + ch), 1)  # x
    pygame.draw.line(surface, (100, 90, 80), (cx, cy), (cx, cy + ch), 1)            # y

    font_sm = pygame.font.Font(None, 20)

    # Y-axis labels
    for val, label in [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]:
        ys = cy + ch - int(val * ch)
        pygame.draw.line(surface, (180, 170, 155), (cx - 3, ys), (cx + cw, ys), 1)
        txt = font_sm.render(label, True, (80, 70, 60))
        surface.blit(txt, (px + 2, ys - 7))

    # X-axis labels (min, optimal, max)
    optimal = cfg.optimal_angle[joint]
    for angle_val, label in [(lo, f"{lo:.1f}"), (optimal, f"{optimal:.1f}*"), (hi, f"{hi:.1f}")]:
        t = (angle_val - lo) / (hi - lo) if hi > lo else 0.5
        xs = cx + int(t * cw)
        pygame.draw.line(surface, (180, 170, 155), (xs, cy), (xs, cy + ch + 3), 1)
        txt = font_sm.render(label, True, (80, 70, 60))
        surface.blit(txt, (xs - txt.get_width() // 2, cy + ch + 5))

    # --- Draw raw strength curve (no fatigue) ---
    n_points = cw
    prev_pt = None
    for i in range(n_points + 1):
        t = i / n_points
        a = lo + (hi - lo) * t
        mult = muscle_model.angle_multiplier(joint, a)
        xs = cx + int(t * cw)
        ys = cy + ch - int(mult * ch)
        color = _strength_color(mult)
        if prev_pt is not None:
            pygame.draw.line(surface, color, prev_pt, (xs, ys), 3)
        prev_pt = (xs, ys)

    # --- Draw fatigue-reduced effective strength curve (dashed) ---
    dash_len = 6
    prev_pt = None
    for i in range(n_points + 1):
        t = i / n_points
        a = lo + (hi - lo) * t
        raw_mult = muscle_model.angle_multiplier(joint, a)
        eff = raw_mult * (1.0 - cfg.fatigue_penalty_scale * fatigue)
        xs = cx + int(t * cw)
        ys = cy + ch - int(eff * ch)
        if prev_pt is not None:
            # Dash: draw every other segment
            seg_idx = i // dash_len
            if seg_idx % 2 == 0:
                pygame.draw.line(surface, (80, 80, 80), prev_pt, (xs, ys), 1)
        prev_pt = (xs, ys)

    # --- Current angle vertical marker ---
    t_cur = (current_angle - lo) / (hi - lo) if hi > lo else 0.5
    t_cur = max(0.0, min(1.0, t_cur))
    cur_x = cx + int(t_cur * cw)
    pygame.draw.line(surface, (20, 20, 20), (cur_x, cy), (cur_x, cy + ch), 2)

    # Dot on raw curve at current angle
    raw_at_cur = muscle_model.angle_multiplier(joint, current_angle)
    eff_at_cur = muscle_model.effective_torque_scale(joint, current_angle)
    peak = cfg.torque_scales[joint]
    dot_y = cy + ch - int(raw_at_cur * ch)
    pygame.draw.circle(surface, (20, 20, 20), (cur_x, dot_y), 5)
    pygame.draw.circle(surface, _strength_color(raw_at_cur), (cur_x, dot_y), 3)

    # --- Legend text ---
    font_leg = pygame.font.Font(None, 22)
    eff_pct = int(eff_at_cur / peak * 100) if peak > 0 else 0
    raw_pct = int(raw_at_cur * 100)
    fatigue_pct = int(fatigue * 100)

    lines = [
        f"Angle: {current_angle:+.2f} rad   (opt {optimal:+.2f})",
        f"Angle strength: {raw_pct}%     Fatigue: {fatigue_pct}%",
        f"Effective output: {eff_pct}% of peak ({peak:.1f} Nm)",
    ]
    for i, line in enumerate(lines):
        txt = font_leg.render(line, True, (30, 20, 10))
        surface.blit(txt, (px + 8, py + PANEL_H - 48 + i * 16))


def _draw_joint_highlight(
    surface: pygame.Surface,
    joint: str,
    current_angle: float,
    muscle_model: MuscleModel,
    endpoints,
    wp_to_px,
) -> None:
    """Draw a glowing ring on the joint(s) associated with the current muscle."""
    eff = muscle_model.effective_torque_scale(joint, current_angle)
    peak = muscle_model.config.torque_scales[joint]
    strength_frac = eff / peak if peak > 0 else 0.0
    color = _strength_color(strength_frac)

    from bouldering2d.environment.player_model import JOINT_LIMITS
    ep = endpoints

    joint_positions = {
        "neck":       [wp_to_px(float(ep.upper_torso[0]), float(ep.upper_torso[1]))],
        "spine":      [wp_to_px(float(ep.mid_torso[0]),   float(ep.mid_torso[1]))],
        "l_shoulder": [wp_to_px(float(ep.upper_torso[0]), float(ep.upper_torso[1]))],
        "r_shoulder": [wp_to_px(float(ep.upper_torso[0]), float(ep.upper_torso[1]))],
        "l_elbow":    [wp_to_px(float(ep.left_elbow[0]),  float(ep.left_elbow[1]))],
        "r_elbow":    [wp_to_px(float(ep.right_elbow[0]), float(ep.right_elbow[1]))],
        "l_hip":      [wp_to_px(float(ep.lower_torso[0]), float(ep.lower_torso[1]))],
        "r_hip":      [wp_to_px(float(ep.lower_torso[0]), float(ep.lower_torso[1]))],
        "l_knee":     [wp_to_px(float(ep.left_knee[0]),   float(ep.left_knee[1]))],
        "r_knee":     [wp_to_px(float(ep.right_knee[0]),  float(ep.right_knee[1]))],
    }

    for pt in joint_positions.get(joint, []):
        pygame.draw.circle(surface, color, pt, 14, 3)
        pygame.draw.circle(surface, (255, 255, 255), pt, 10, 2)


def _draw_top_overlay(
    surface: pygame.Surface,
    joint: str,
    joint_idx: int,
    current_angle: float,
    muscle_model: MuscleModel,
    screen_w: int,
) -> None:
    """Title bar: joint name, index, controls hint."""
    lo, hi = JOINT_LIMITS[joint]
    font_lg = pygame.font.Font(None, 42)
    font_sm = pygame.font.Font(None, 26)

    label = font_lg.render(f"Muscle: {joint}", True, (20, 20, 20))
    surface.blit(label, (screen_w - label.get_width() - 16, 14))

    idx_lbl = font_sm.render(f"({joint_idx + 1}/{len(JOINT_ORDER)})", True, (80, 80, 80))
    surface.blit(idx_lbl, (screen_w - idx_lbl.get_width() - 16, 56))

    hint = font_sm.render("SPACE/→ next   ← prev   F fatigue   R reset   Q quit", True, (100, 100, 100))
    surface.blit(hint, (screen_w - hint.get_width() - 16, surface.get_height() - 30))

    # Strength bar across the top
    bar_x = screen_w - 220
    bar_y = 80
    bar_w = 200
    bar_h = 10
    eff = muscle_model.effective_torque_scale(joint, current_angle)
    peak = muscle_model.config.torque_scales[joint]
    frac = eff / peak if peak > 0 else 0.0
    pygame.draw.rect(surface, (180, 170, 155), (bar_x, bar_y, bar_w, bar_h))
    fill_color = _strength_color(frac)
    pygame.draw.rect(surface, fill_color, (bar_x, bar_y, int(bar_w * frac), bar_h))

    strength_lbl = font_sm.render(f"Output: {int(frac * 100)}%", True, (40, 40, 40))
    surface.blit(strength_lbl, (bar_x, bar_y + 14))


def main() -> None:
    env_config = EnvConfig()
    player_config = PlayerConfig()
    muscle_config = MuscleConfig()

    pygame.init()
    screen = pygame.display.set_mode((env_config.screen_width, env_config.screen_height))
    pygame.display.set_caption("Muscle Strength Inspector")
    clock = pygame.time.Clock()

    renderer = Renderer(env_config)
    contacts = ContactManager(player_config)
    muscle_model = MuscleModel(muscle_config)

    x_scale = env_config.screen_width / env_config.wall_width
    y_scale = 120
    center_py = env_config.screen_height // 2

    def wp_to_px(wx: float, wy: float) -> tuple[int, int]:
        return int(wx * x_scale), int(center_py - (wy - 0.8) * y_scale)

    joint_idx = 0
    frame = 0
    running = True

    while running:
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
                elif event.key == pygame.K_f:
                    name = JOINT_ORDER[joint_idx]
                    muscle_model.state.fatigue[name] = min(
                        1.0, muscle_model.state.fatigue[name] + 0.2
                    )
                elif event.key == pygame.K_r:
                    muscle_model.reset()

        name = JOINT_ORDER[joint_idx]
        lo, hi = JOINT_LIMITS[name]

        # Ping-pong sweep
        cycle = frame % (SWEEP_FRAMES * 2)
        t = cycle / SWEEP_FRAMES
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
            muscle_fatigue=muscle_model.state,
        )

        renderer._draw_scene(screen, ctx)

        _draw_joint_highlight(screen, name, angle, muscle_model, endpoints, wp_to_px)
        _draw_curve_panel(
            screen, name, angle, muscle_model,
            env_config.screen_width, env_config.screen_height,
        )
        _draw_top_overlay(screen, name, joint_idx, angle, muscle_model, env_config.screen_width)

        pygame.display.flip()
        frame += 1
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
