from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from bouldering2d.config import EnvConfig
from bouldering2d.environment.contacts import ContactManager
from bouldering2d.environment.holds import Hold
from bouldering2d.environment.player_model import LimbEndpoints, PlayerModel


@dataclass(slots=True)
class RenderContext:
    player: PlayerModel
    endpoints: LimbEndpoints
    holds: list[Hold]
    observed_holds: list[Hold]
    contacts: ContactManager
    stamina: float
    step_count: int
    ascent: float


class Renderer:
    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.initialized = False

    def _init(self) -> None:
        if self.initialized:
            return
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.screen_width, self.config.screen_height))
        pygame.display.set_caption("Bouldering2D")
        self.clock = pygame.time.Clock()
        self.initialized = True

    def close(self) -> None:
        if self.initialized:
            pygame.quit()
            self.initialized = False

    def render(self, ctx: RenderContext, mode: str) -> np.ndarray | None:
        if mode not in {"human", "rgb_array"}:
            return None

        if mode == "human":
            self._init()
            assert self.screen is not None
            surface = self.screen
        else:
            if not pygame.get_init():
                pygame.init()
            if not pygame.font.get_init():
                pygame.font.init()
            surface = pygame.Surface((self.config.screen_width, self.config.screen_height))

        self._draw_scene(surface, ctx)

        if mode == "human":
            assert self.clock is not None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass
            pygame.display.flip()
            self.clock.tick(30)
            return None

        arr = pygame.surfarray.array3d(surface)
        return np.transpose(arr, (1, 0, 2))

    def _draw_scene(self, surface: pygame.Surface, ctx: RenderContext) -> None:
        surface.fill((243, 236, 220))
        camera_y = float(ctx.player.pelvis[1])
        center_px = self.config.screen_width // 2
        center_py = self.config.screen_height // 2
        x_scale = self.config.screen_width / self.config.wall_width
        y_scale = 120

        def wp_to_px(wx: float, wy: float) -> tuple[int, int]:
            px = int(wx * x_scale)
            py = int(center_py - (wy - camera_y) * y_scale)
            return px, py

        pygame.draw.rect(surface, (219, 206, 187), pygame.Rect(40, 0, self.config.screen_width - 80, self.config.screen_height))

        observed_hold_keys = {(h.x, h.y) for h in ctx.observed_holds}

        for hold in ctx.holds:
            px, py = wp_to_px(hold.x, hold.y)
            pygame.draw.circle(surface, (193, 80, 44), (px, py), 8)
            if (hold.x, hold.y) in observed_hold_keys:
                pygame.draw.circle(surface, (0, 255, 0), (px, py), 11, 2)

        ep = ctx.endpoints
        player_cfg = ctx.player.config
        pelvis_world = ep.lower_torso
        neck_world = ep.upper_torso

        l_elbow_world = ep.left_elbow.copy()
        r_elbow_world = ep.right_elbow.copy()
        l_hand_world = ep.left_hand.copy()
        r_hand_world = ep.right_hand.copy()
        l_knee_world = ep.left_knee.copy()
        r_knee_world = ep.right_knee.copy()
        l_foot_world = ep.left_foot.copy()
        r_foot_world = ep.right_foot.copy()

        arm_l2 = player_cfg.lower_arm_length + player_cfg.hand_length
        leg_l2 = player_cfg.lower_leg_length

        if ctx.contacts.state.left_hand is not None:
            target = np.array([ctx.contacts.state.left_hand.x, ctx.contacts.state.left_hand.y], dtype=np.float32)
            solved = self._solve_two_link(neck_world, l_elbow_world, target, player_cfg.upper_arm_length, arm_l2)
            if solved is not None:
                l_elbow_world, l_hand_world = solved

        if ctx.contacts.state.right_hand is not None:
            target = np.array([ctx.contacts.state.right_hand.x, ctx.contacts.state.right_hand.y], dtype=np.float32)
            solved = self._solve_two_link(neck_world, r_elbow_world, target, player_cfg.upper_arm_length, arm_l2)
            if solved is not None:
                r_elbow_world, r_hand_world = solved

        if ctx.contacts.state.left_foot is not None:
            target = np.array([ctx.contacts.state.left_foot.x, ctx.contacts.state.left_foot.y], dtype=np.float32)
            solved = self._solve_two_link(pelvis_world, l_knee_world, target, player_cfg.upper_leg_length, player_cfg.lower_leg_length)
            if solved is not None:
                l_knee_world, l_foot_world = solved

        if ctx.contacts.state.right_foot is not None:
            target = np.array([ctx.contacts.state.right_foot.x, ctx.contacts.state.right_foot.y], dtype=np.float32)
            solved = self._solve_two_link(pelvis_world, r_knee_world, target, player_cfg.upper_leg_length, player_cfg.lower_leg_length)
            if solved is not None:
                r_knee_world, r_foot_world = solved

        head = wp_to_px(float(ep.head[0]), float(ep.head[1]))
        neck_base = wp_to_px(float(ep.upper_torso[0]), float(ep.upper_torso[1]))
        chest = wp_to_px(float(ep.mid_torso[0]), float(ep.mid_torso[1]))
        pelvis = wp_to_px(float(ep.lower_torso[0]), float(ep.lower_torso[1]))
        le = wp_to_px(float(l_elbow_world[0]), float(l_elbow_world[1]))
        re = wp_to_px(float(r_elbow_world[0]), float(r_elbow_world[1]))
        lh = wp_to_px(float(l_hand_world[0]), float(l_hand_world[1]))
        rh = wp_to_px(float(r_hand_world[0]), float(r_hand_world[1]))
        lk = wp_to_px(float(l_knee_world[0]), float(l_knee_world[1]))
        rk = wp_to_px(float(r_knee_world[0]), float(r_knee_world[1]))
        lf = wp_to_px(float(l_foot_world[0]), float(l_foot_world[1]))
        rf = wp_to_px(float(r_foot_world[0]), float(r_foot_world[1]))

        pygame.draw.line(surface, (24, 26, 32), pelvis, chest, 5)
        pygame.draw.line(surface, (24, 26, 32), chest, neck_base, 5)
        pygame.draw.line(surface, (24, 26, 32), neck_base, head, 4)

        pygame.draw.line(surface, (24, 26, 32), neck_base, le, 4)
        pygame.draw.line(surface, (24, 26, 32), le, lh, 4)
        pygame.draw.line(surface, (24, 26, 32), neck_base, re, 4)
        pygame.draw.line(surface, (24, 26, 32), re, rh, 4)

        pygame.draw.line(surface, (24, 26, 32), pelvis, lk, 4)
        pygame.draw.line(surface, (24, 26, 32), lk, lf, 4)
        pygame.draw.line(surface, (24, 26, 32), pelvis, rk, 4)
        pygame.draw.line(surface, (24, 26, 32), rk, rf, 4)

        for joint in [chest, le, re, lk, rk]:
            pygame.draw.circle(surface, (36, 38, 45), joint, 4)
        pygame.draw.circle(surface, (39, 105, 179), head, 10)

        for hold in [
            ctx.contacts.state.left_hand,
            ctx.contacts.state.right_hand,
            ctx.contacts.state.left_foot,
            ctx.contacts.state.right_foot,
        ]:
            if hold is not None:
                hx, hy = wp_to_px(hold.x, hold.y)
                pygame.draw.circle(surface, (30, 160, 80), (hx, hy), 10, 2)

        font = pygame.font.Font(None, 30)
        hud = [
            f"Stamina: {ctx.stamina:5.1f}",
            f"Height: {ctx.ascent:5.2f}",
            f"Step: {ctx.step_count}",
            f"Contacts: {ctx.contacts.state.attached_count()}",
        ]
        for idx, line in enumerate(hud):
            text = font.render(line, True, (20, 20, 20))
            surface.blit(text, (14, 12 + idx * 25))

    @staticmethod
    def _solve_two_link(
        root: np.ndarray,
        preferred_joint: np.ndarray,
        target: np.ndarray,
        l1: float,
        l2: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        vec = target - root
        dist = float(np.linalg.norm(vec))

        # Never stretch limbs: if the hold is out of reach, keep original pose.
        if dist > (l1 + l2) or dist < abs(l1 - l2):
            return None

        if dist < 1e-6:
            return None

        direction = vec / dist
        a = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist)
        h_sq = max(0.0, l1 * l1 - a * a)
        h = float(np.sqrt(h_sq))

        base = root + direction * a
        perp = np.array([-direction[1], direction[0]], dtype=np.float32)
        c1 = base + perp * h
        c2 = base - perp * h

        if np.linalg.norm(c1 - preferred_joint) <= np.linalg.norm(c2 - preferred_joint):
            joint = c1.astype(np.float32)
        else:
            joint = c2.astype(np.float32)

        return joint, target.astype(np.float32)
