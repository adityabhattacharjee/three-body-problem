"""
Scene C: How Long Do Three Random Bodies Stay Together?
Loads precomputed example trajectories (examples_traj.npz) and the real
200-trial ensemble statistics (data/escape_statistics.npz) — no physics is
computed here.
"""
import os
import numpy as np
from manim import *

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/Users/abhijit/Library/CloudStorage/OneDrive-Personal/Documents/Aditya/SBC/Physics/three_body"

BG = "#0D1117"
COLORS = ["#F59E0B", "#3B82F6", "#10B981"]
REF_WIDTH = 14.222222


class MainScene(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        ex = np.load(os.path.join(HERE, "examples_traj.npz"))
        stats = np.load(os.path.join(BASE, "data", "escape_statistics.npz"), allow_pickle=True)

        title = Text("How Long Do Three Random Bodies Stay Together?", font_size=30, color=WHITE)
        title.to_edge(UP, buff=0.5)
        TITLE_BASE_H = title.height

        def upd_title(m):
            factor = self.camera.frame.width / REF_WIDTH
            m.scale_to_fit_height(TITLE_BASE_H * factor)
            top = self.camera.frame.get_top()
            m.move_to(top + DOWN * (m.height / 2 + 0.5 * factor))
        title.add_updater(upd_title)
        self.play(FadeIn(title), run_time=1.0)
        self.wait(0.8)

        # A fixed scale is computed from the ACTUAL trajectory extent (with a
        # safety margin) rather than guessed, so it is guaranteed to keep the
        # whole clip inside the default frame.
        def safe_fixed_scale(pos, aspect, margin=0.85):
            maxabsx = np.abs(pos[:, :, 0]).max()
            maxabsy = np.abs(pos[:, :, 1]).max()
            return margin * min(
                REF_WIDTH / (2.0 * maxabsx),
                (REF_WIDTH / aspect) / (2.0 * maxabsy),
            )

        # =========================================================
        #  Example 1: COLLISION
        # =========================================================
        collision_scale = safe_fixed_scale(ex["pos_collision"], REF_WIDTH / 8.0)
        self.play_example(
            pos=ex["pos_collision"], t_event=float(ex["t_event_collision"]),
            outcome_label="Example outcome: COLLISION", label_color="#EF4444",
            duration=8.0, dynamic_zoom=False, fixed_scale=collision_scale, title=title,
        )

        # =========================================================
        #  Example 2: ESCAPE
        # =========================================================
        self.play_example(
            pos=ex["pos_escape"], t_event=float(ex["t_event_escape"]),
            outcome_label="Example outcome: ESCAPE", label_color="#F59E0B",
            duration=9.0, dynamic_zoom=True, fixed_scale=1.0, title=title,
        )

        title.clear_updaters()
        self.play(FadeOut(title), run_time=0.6)

        # =========================================================
        #  Survival curve panel
        # =========================================================
        t_grid = stats["t_grid"].astype(float)
        survival_curve = stats["survival_curve"].astype(float)
        n_trials = int(stats["n_trials"])
        t_max = float(stats["t_max"])

        panel_title = Text("Survival statistics: 200 random three-body scattering trials",
                            font_size=28, color=WHITE)
        panel_title.to_edge(UP, buff=0.5)
        n_note = Text(f"n_trials = {n_trials}, fixed energy E = -1, t_max = {t_max:.0f}",
                       font_size=20, color=GRAY_B)
        n_note.next_to(panel_title, DOWN, buff=0.25)
        self.play(FadeIn(panel_title), FadeIn(n_note), run_time=1.0)

        axes = Axes(
            x_range=[0, 200, 50],
            y_range=[0, 1.0, 0.2],
            x_length=9.5,
            y_length=5.0,
            axis_config={"color": GRAY_B, "include_tip": True, "font_size": 20},
        )
        axes.move_to(ORIGIN + DOWN * 0.3)
        x_label = Text("T (time units)", font_size=22, color=WHITE).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Text("S(T) = survival probability", font_size=22, color=WHITE)
        y_label.rotate(90 * DEGREES).next_to(axes.y_axis, LEFT, buff=0.35)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=1.5)

        points = [axes.c2p(T, S) for T, S in zip(t_grid, survival_curve)]
        curve_line = VMobject(color="#3B82F6", stroke_width=3)
        curve_line.set_points_as_corners([points[0]])
        dots = VGroup(*[Dot(point=points[0], radius=0.06, color="#F59E0B")])

        self.add(curve_line, dots)
        for i in range(1, len(points)):
            new_line = VMobject(color="#3B82F6", stroke_width=3)
            new_line.set_points_as_corners(points[: i + 1])
            new_dot = Dot(point=points[i], radius=0.06, color="#F59E0B")
            self.play(
                Transform(curve_line, new_line),
                FadeIn(new_dot),
                run_time=0.7,
                rate_func=linear,
            )
            dots.add(new_dot)

        self.wait(1.5)

        readout = Text(
            f"S(50) = {survival_curve[np.argmin(np.abs(t_grid-50))]:.2f}   "
            f"S(200) = {survival_curve[-1]:.2f}",
            font_size=22, color=GRAY_B,
        )
        readout.next_to(axes, DOWN, buff=0.7)
        self.play(FadeIn(readout), run_time=0.8)
        self.wait(2.0)

        self.play(
            FadeOut(axes), FadeOut(x_label), FadeOut(y_label), FadeOut(curve_line),
            FadeOut(dots), FadeOut(panel_title), FadeOut(n_note), FadeOut(readout),
            run_time=1.0,
        )

        # ---------------- Closing caption ----------------
        caption1 = Text(
            "Generic three-body systems are chaotic: most either collide or eject a body.",
            font_size=26, color=WHITE,
        )
        caption2 = Text(
            "Survival probability S(T) measured from 200 independent random trials at fixed energy.",
            font_size=22, color=GRAY_B,
        )
        caption_group = VGroup(caption1, caption2).arrange(DOWN, buff=0.3)
        caption_group.move_to(ORIGIN)
        self.play(FadeIn(caption_group), run_time=1.0)
        self.wait(3.0)
        self.play(FadeOut(caption_group), run_time=1.0)
        self.wait(0.3)

    def play_example(self, pos, t_event, outcome_label, label_color, duration,
                      dynamic_zoom, fixed_scale, title=None):
        n = pos.shape[0]

        label = Text(outcome_label, font_size=26, color=label_color)
        label.to_edge(UP, buff=1.3)
        LABEL_BASE_H = label.height

        def upd_label(m):
            factor = self.camera.frame.width / REF_WIDTH
            m.scale_to_fit_height(LABEL_BASE_H * factor)
            if title is not None:
                m.next_to(title, DOWN, buff=0.5 * factor)
            else:
                top = self.camera.frame.get_top()
                m.move_to(top + DOWN * (m.height / 2 + 1.3 * factor))
        label.add_updater(upd_label)
        self.play(FadeIn(label), run_time=0.5)

        if dynamic_zoom:
            ASPECT = REF_WIDTH / 8.0
            # track the bodies' bounding-box center/extent each frame (not a
            # fixed origin) so the frame stays well-filled even when the
            # trajectory drifts off-center
            minx = pos[:, :, 0].min(axis=1)
            maxx = pos[:, :, 0].max(axis=1)
            miny = pos[:, :, 1].min(axis=1)
            maxy = pos[:, :, 1].max(axis=1)
            cx = (minx + maxx) / 2.0
            cy = (miny + maxy) / 2.0
            half_w = (maxx - minx) / 2.0
            half_h = (maxy - miny) / 2.0
            needed_width = np.maximum(2.0 * half_w, 2.0 * ASPECT * half_h) * 1.45 + 1.5
            # small forward look-ahead to avoid one-frame lag during fast motion
            lookahead = max(1, len(needed_width) // 40)
            dilated = np.array([
                needed_width[i:i + lookahead].max() for i in range(len(needed_width))
            ])
            running_max = np.maximum.accumulate(dilated)
            # Data-driven ceiling (see scene_b_hierarchical for why a guessed
            # constant is unsafe here).
            cam_width_series = np.clip(running_max, 9.0, running_max.max() * 1.02)
            self.add(self.camera.frame)

            def upd_cam(mob):
                f = tracker.get_value()
                w = float(np.interp(f, np.arange(n), cam_width_series))
                ccx = float(np.interp(f, np.arange(n), cx))
                ccy = float(np.interp(f, np.arange(n), cy))
                mob.set(width=w)
                mob.move_to(np.array([ccx, ccy, 0.0]))
            self.camera.frame.add_updater(upd_cam)

        def scaled_point(body_idx, f):
            x = np.interp(f, np.arange(n), pos[:, body_idx, 0]) * fixed_scale
            y = np.interp(f, np.arange(n), pos[:, body_idx, 1]) * fixed_scale
            return np.array([x, y, 0.0])

        tracker = ValueTracker(0.0)
        dots = VGroup(*[
            Dot(point=scaled_point(b, 0.0), radius=0.09, color=COLORS[b]) for b in range(3)
        ])

        def make_upd(b):
            def upd(mob):
                mob.move_to(scaled_point(b, tracker.get_value()))
            return upd
        for b in range(3):
            dots[b].add_updater(make_upd(b))

        trails = VGroup(*[
            TracedPath(dots[b].get_center, stroke_color=COLORS[b], stroke_width=3,
                       dissipating_time=1.2)
            for b in range(3)
        ])

        REF_H = 14.222222
        t_label = Text(f"t = 0.00", font_size=22, color=WHITE)
        t_label_base_h = t_label.height

        def upd_time(mob):
            f = tracker.get_value()
            val = float(np.interp(f, np.arange(n), np.linspace(0, t_event, n)))
            new_text = Text(f"t = {val:.2f}", font_size=22, color=WHITE)
            factor = (self.camera.frame.width / REF_H) if dynamic_zoom else 1.0
            new_text.scale_to_fit_height(t_label_base_h * factor)
            corner = self.camera.frame.get_corner(DR)
            new_text.move_to(corner + LEFT * (new_text.width / 2 + 0.4 * factor)
                              + UP * (new_text.height / 2 + 0.3 * factor))
            mob.become(new_text)
        t_label.add_updater(upd_time)

        self.add(trails, dots, t_label)
        self.play(*[FadeIn(d, scale=0.5) for d in dots], run_time=0.4)

        self.play(tracker.animate.set_value(n - 1), run_time=duration, rate_func=linear)

        for d in dots:
            d.clear_updaters()
        t_label.clear_updaters()
        if dynamic_zoom:
            self.camera.frame.clear_updaters()

        self.wait(0.6)
        label.clear_updaters()
        fade_group = [FadeOut(d) for d in dots] + [FadeOut(trails), FadeOut(t_label), FadeOut(label)]
        self.play(*fade_group, run_time=0.7)

        if dynamic_zoom:
            self.camera.frame.set(width=REF_WIDTH)
            self.camera.frame.move_to(ORIGIN)
