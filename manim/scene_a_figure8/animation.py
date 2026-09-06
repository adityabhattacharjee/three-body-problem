"""
Scene A: The Figure-Eight Orbit
Loads precomputed trajectory (figure8_traj.npz) — physics is NOT computed here.
"""
import os
import numpy as np
from manim import *

HERE = os.path.dirname(os.path.abspath(__file__))

BG = "#0D1117"
COLORS = ["#F59E0B", "#3B82F6", "#10B981"]  # body1 orange, body2 blue, body3 green

SCALE = 4.3  # data units -> screen units


class MainScene(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG

        data = np.load(os.path.join(HERE, "figure8_traj.npz"))
        t = data["t"]
        pos = data["pos"]  # (N, 3, 2)
        T = float(data["T"])
        max_cross_diff = float(data["max_cross_diff"])

        n = pos.shape[0]
        t_in_periods = t / T  # (N,)

        def scaled_point(body_idx, frame_f):
            # linear interpolation over the sample index (float)
            x = np.interp(frame_f, np.arange(n), pos[:, body_idx, 0]) * SCALE
            y = np.interp(frame_f, np.arange(n), pos[:, body_idx, 1]) * SCALE
            return np.array([x, y, 0.0])

        # ---------- Title ----------
        title = Text("The Figure-Eight Three-Body Orbit", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.5)
        subtitle = Text(
            "Equal masses, zero angular momentum, periodic solution",
            font_size=22, color=GRAY_B,
        )
        subtitle.next_to(title, DOWN, buff=0.25)

        self.play(FadeIn(title, shift=UP * 0.2), FadeIn(subtitle), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(subtitle), run_time=0.6)

        # ---------- Static faint full path (ghost of entire trajectory) ----------
        ghost_paths = VGroup()
        for b in range(3):
            pts = [np.array([pos[i, b, 0] * SCALE, pos[i, b, 1] * SCALE, 0.0])
                   for i in range(0, n, 2)]
            path = VMobject()
            path.set_points_as_corners(pts)
            path.set_stroke(color=COLORS[b], width=1.5, opacity=0.22)
            ghost_paths.add(path)
        self.play(*[Create(p) for p in ghost_paths], run_time=1.8)

        # ---------- Bodies + trails ----------
        frame_tracker = ValueTracker(0.0)

        dots = VGroup()
        labels = VGroup()
        for b in range(3):
            dot = Dot(point=scaled_point(b, 0.0), radius=0.09, color=COLORS[b])
            dots.add(dot)
        body_names = ["body 1", "body 2", "body 3"]
        for b in range(3):
            lbl = Text(body_names[b], font_size=16, color=COLORS[b])
            lbl.add_updater(lambda m, b=b: m.next_to(dots[b], UP, buff=0.12))
            labels.add(lbl)

        def make_updater(b):
            def upd(mob):
                mob.move_to(scaled_point(b, frame_tracker.get_value()))
            return upd

        for b in range(3):
            dots[b].add_updater(make_updater(b))

        trails = VGroup()
        for b in range(3):
            trail = TracedPath(
                dots[b].get_center,
                stroke_color=COLORS[b],
                stroke_width=3.5,
                dissipating_time=1.5,
            )
            trails.add(trail)

        # ---------- Elapsed-time counter ----------
        counter_label = Text("t = ", font_size=26, color=WHITE)
        counter_num = DecimalNumber(0.0, num_decimal_places=2, font_size=26, color=WHITE)
        counter_unit = Text(" T", font_size=26, color=WHITE)
        counter_group = VGroup(counter_label, counter_num, counter_unit).arrange(RIGHT, buff=0.06)
        counter_group.to_corner(DR, buff=0.4)

        def update_counter(mob):
            idx = frame_tracker.get_value()
            val = float(np.interp(idx, np.arange(n), t_in_periods))
            counter_num.set_value(val)
            mob.arrange(RIGHT, buff=0.06)
            mob.to_corner(DR, buff=0.4)

        counter_group.add_updater(update_counter)

        self.add(trails, dots, labels, counter_group)
        self.play(*[FadeIn(d, scale=0.5) for d in dots], run_time=0.6)

        # Animate through the full precomputed range (2.5 periods) at a
        # constant real-time rate over ~28s of screen time.
        self.play(
            frame_tracker.animate.set_value(n - 1),
            run_time=28.0,
            rate_func=linear,
        )

        for d in dots:
            d.clear_updaters()
        counter_group.clear_updaters()
        for lbl in labels:
            lbl.clear_updaters()

        self.wait(0.3)
        self.play(
            FadeOut(dots), FadeOut(labels), FadeOut(trails),
            FadeOut(ghost_paths), FadeOut(counter_group),
            FadeOut(title),
            run_time=1.0,
        )

        # ---------- Closing caption card ----------
        caption1 = Text(
            "Figure-eight three-body orbit — validated to "
            f"{max_cross_diff:.1e} agreement",
            font_size=28, color=WHITE,
        )
        caption2 = Text(
            "cross-checked between two independent integrators (scipy DOP853 & REBOUND IAS15)",
            font_size=22, color=GRAY_B,
        )
        caption_group = VGroup(caption1, caption2).arrange(DOWN, buff=0.3)
        caption_group.move_to(ORIGIN)

        self.play(FadeIn(caption_group), run_time=1.0)
        self.wait(3.0)
        self.play(FadeOut(caption_group), run_time=1.0)
        self.wait(0.3)
