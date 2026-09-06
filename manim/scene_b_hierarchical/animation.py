"""
Scene B: Hierarchical Triple - Stable vs Disrupted
Loads precomputed trajectories (hierarchical_traj.npz) — physics is NOT
computed here.
"""
import os
import numpy as np
from manim import *

HERE = os.path.dirname(os.path.abspath(__file__))

BG = "#0D1117"
C_INNER1 = "#F59E0B"  # orange
C_INNER2 = "#3B82F6"  # blue
C_OUTER = "#10B981"   # green

TU_PER_SEC = 12.0  # shared real-time pace for both clips (honest contrast)


class MainScene(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG
        data = np.load(os.path.join(HERE, "hierarchical_traj.npz"))

        pos_s = data["pos_stable"]          # (Ns, 3, 2)
        t_s = data["t_stable"]
        T_in_s = float(data["T_in_stable"])
        n_periods_stable_shown = int(data["n_periods_stable_shown"])

        pos_u = data["pos_unstable"]        # (Nu, 3, 2)
        t_u = data["t_unstable"]
        T_in_u = float(data["T_in_unstable"])
        t_disrupt_u = float(data["t_disruption_unstable"])

        crit_ratio = float(data["crit_ratio"])

        REF_WIDTH = 14.222222  # default MovingCameraScene frame width
        ASPECT = REF_WIDTH / 8.0

        # ---------------- Title + criterion ----------------
        # Title is camera-anchored (constant on-screen size/position relative
        # to the current camera frame) from the very start, so it stays
        # correctly placed through both the static stable clip AND the
        # dynamically-zoomed unstable clip without special-casing.
        title = Text("Hierarchical Triple: Stable vs Disrupted", font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.5)
        TITLE_BASE_H = title.height

        def upd_title(m):
            factor = self.camera.frame.width / REF_WIDTH
            m.scale_to_fit_height(TITLE_BASE_H * factor)
            top = self.camera.frame.get_top()
            m.move_to(top + DOWN * (m.height / 2 + 0.5 * factor))
        title.add_updater(upd_title)

        crit_text = Text(
            f"Mardling-Aarseth stability criterion:  a_out / a_in >~ {crit_ratio:.2f}",
            font_size=24, color=GRAY_B,
        )
        crit_text.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(title), FadeIn(crit_text), run_time=1.2)
        self.wait(2.0)
        self.play(FadeOut(crit_text), run_time=0.6)

        # =========================================================
        #  CLIP 1: STABLE (ratio = 6.0)
        # =========================================================
        label_s = Text("STABLE  —  a_out / a_in = 6.0  (above critical ratio)",
                        font_size=26, color="#10B981")
        label_s.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(label_s), run_time=0.8)

        # Compute a fixed scale that is GUARANTEED to keep the whole stable
        # trajectory (all 50 periods shown) within the default camera frame,
        # rather than an arbitrary hardcoded factor. The stable case is
        # bounded (that's the whole point), so a fixed scale -- no dynamic
        # zoom needed -- is correct here, as long as it is computed from the
        # actual data instead of guessed.
        maxabsx_s_raw = np.abs(pos_s[:, :, 0]).max()
        maxabsy_s_raw = np.abs(pos_s[:, :, 1]).max()
        SAFETY_MARGIN = 0.85  # leave 15% headroom
        SCALE_S = SAFETY_MARGIN * min(
            REF_WIDTH / (2.0 * maxabsx_s_raw),
            (REF_WIDTH / ASPECT) / (2.0 * maxabsy_s_raw),
        )
        n_s = pos_s.shape[0]

        def scaled_point_s(body_idx, f):
            x = np.interp(f, np.arange(n_s), pos_s[:, body_idx, 0]) * SCALE_S
            y = np.interp(f, np.arange(n_s), pos_s[:, body_idx, 1]) * SCALE_S
            return np.array([x, y, 0.0])

        dots_s = VGroup(
            Dot(point=scaled_point_s(0, 0), radius=0.09, color=C_INNER1),
            Dot(point=scaled_point_s(1, 0), radius=0.09, color=C_INNER2),
            Dot(point=scaled_point_s(2, 0), radius=0.11, color=C_OUTER),
        )
        legend_s = VGroup(
            Text("inner binary", font_size=18, color=WHITE),
            Dot(radius=0.06, color=C_INNER1), Dot(radius=0.06, color=C_INNER2),
            Text("outer body", font_size=18, color=WHITE),
            Dot(radius=0.06, color=C_OUTER),
        ).arrange(RIGHT, buff=0.15).to_corner(DL, buff=0.4)

        tracker_s = ValueTracker(0.0)

        def make_upd_s(b):
            def upd(mob):
                mob.move_to(scaled_point_s(b, tracker_s.get_value()))
            return upd
        for b in range(3):
            dots_s[b].add_updater(make_upd_s(b))

        trails_s = VGroup(*[
            TracedPath(dots_s[b].get_center, stroke_color=[C_INNER1, C_INNER2, C_OUTER][b],
                       stroke_width=2.5, dissipating_time=2.5)
            for b in range(3)
        ])

        counter_s = VGroup(
            Text("periods elapsed: ", font_size=22, color=WHITE),
            DecimalNumber(0.0, num_decimal_places=1, font_size=28, color=WHITE),
        ).arrange(RIGHT, buff=0.1).to_corner(DR, buff=0.4)

        t_periods_s = t_s / T_in_s

        def upd_counter_s(mob):
            f = tracker_s.get_value()
            val = float(np.interp(f, np.arange(n_s), t_periods_s))
            mob[1].set_value(val)
            mob.arrange(RIGHT, buff=0.1)
            mob.to_corner(DR, buff=0.4)
        counter_s.add_updater(upd_counter_s)

        self.add(trails_s, dots_s, legend_s, counter_s)
        self.play(*[FadeIn(d, scale=0.5) for d in dots_s], FadeIn(legend_s), run_time=0.6)

        # Show real motion for n_periods_stable_shown inner periods, at the
        # shared real-time pace TU_PER_SEC.
        shown_tu = n_periods_stable_shown * T_in_s
        run_time_s = shown_tu / TU_PER_SEC
        self.play(tracker_s.animate.set_value(n_s - 1), run_time=run_time_s, rate_func=linear)

        for d in dots_s:
            d.clear_updaters()
        counter_s.clear_updaters()

        # Fast-forward jump: this system was separately verified (in
        # hierarchical_stability.py) to remain bound through 500 inner
        # periods -- we do not re-integrate that here, we report it.
        ff_text = Text("<< fast-forwarding ... still bound at 500 inner periods >>",
                        font_size=24, color=YELLOW)
        ff_text.move_to(ORIGIN + DOWN * 2.6)
        self.play(FadeIn(ff_text), run_time=0.5)
        final_count = DecimalNumber(500.0, num_decimal_places=1, font_size=28, color=WHITE)
        final_label = Text("periods elapsed: ", font_size=22, color=WHITE)
        final_group = VGroup(final_label, final_count).arrange(RIGHT, buff=0.1).to_corner(DR, buff=0.4)
        self.remove(counter_s)
        self.add(final_group)
        self.wait(1.5)

        self.play(
            FadeOut(dots_s), FadeOut(trails_s), FadeOut(legend_s),
            FadeOut(final_group), FadeOut(ff_text), FadeOut(label_s),
            run_time=0.8,
        )

        # =========================================================
        #  CLIP 2: UNSTABLE (ratio = 1.8)
        # =========================================================
        label_u = Text("DISRUPTED  —  a_out / a_in = 1.8  (below critical ratio)",
                        font_size=26, color="#EF4444")
        label_u.next_to(title, DOWN, buff=0.3)
        LABEL_U_BASE_H = label_u.height

        def upd_label_u(m):
            factor = self.camera.frame.width / REF_WIDTH
            m.scale_to_fit_height(LABEL_U_BASE_H * factor)
            m.next_to(title, DOWN, buff=0.3 * factor)
        label_u.add_updater(upd_label_u)
        self.play(FadeIn(label_u), run_time=0.8)

        n_u = pos_u.shape[0]
        maxabsx_u = np.abs(pos_u[:, :, 0]).max(axis=1)   # (Nu,)
        maxabsy_u = np.abs(pos_u[:, :, 1]).max(axis=1)   # (Nu,)
        # width needed so BOTH |x|<=w/2 and |y|<=h/2=w/(2*ASPECT) hold, plus margin
        needed_width_u = np.maximum(2.0 * maxabsx_u, 2.0 * ASPECT * maxabsy_u) * 1.6 + 2.5
        # Look ahead a window so the camera starts widening well BEFORE the
        # body reaches an extreme point, instead of reacting late (which is
        # what let the fast ejection near disruption briefly poke outside
        # the frame -- the dissipating trail behind the ejected body needs
        # extra room too, not just the dot's instantaneous position).
        lookahead = max(1, len(needed_width_u) // 20)
        dilated_u = np.array([
            needed_width_u[i:i + lookahead].max()
            for i in range(len(needed_width_u))
        ])
        running_max_u = np.maximum.accumulate(dilated_u)
        # Upper bound must be driven by the actual data, not a guessed
        # constant -- a hardcoded ceiling silently saturates and defeats any
        # margin tuning if the real trajectory needs more room than guessed
        # (exactly what happened here: needed a ceiling of ~68, a hardcoded
        # 40-45 was silently capping it well short).
        cam_width_series = np.clip(running_max_u, 9.0, running_max_u.max() * 1.02)

        def pos_point_u(body_idx, f):
            x = np.interp(f, np.arange(n_u), pos_u[:, body_idx, 0])
            y = np.interp(f, np.arange(n_u), pos_u[:, body_idx, 1])
            return np.array([x, y, 0.0])

        dots_u = VGroup(
            Dot(point=pos_point_u(0, 0), radius=0.09, color=C_INNER1),
            Dot(point=pos_point_u(1, 0), radius=0.09, color=C_INNER2),
            Dot(point=pos_point_u(2, 0), radius=0.11, color=C_OUTER),
        )
        legend_u = VGroup(
            Text("inner binary", font_size=18, color=WHITE),
            Dot(radius=0.06, color=C_INNER1), Dot(radius=0.06, color=C_INNER2),
            Text("outer body", font_size=18, color=WHITE),
            Dot(radius=0.06, color=C_OUTER),
        ).arrange(RIGHT, buff=0.15)

        legend_u_base_height = legend_u.height

        def anchor_dl(m):
            # keep constant on-screen (pixel) size regardless of camera zoom,
            # anchored to the actual visible frame corner
            factor = self.camera.frame.width / REF_WIDTH
            m.scale_to_fit_height(legend_u_base_height * factor)
            corner = self.camera.frame.get_corner(DL)
            m.move_to(corner + RIGHT * (m.width / 2 + 0.4 * factor) + UP * (m.height / 2 + 0.3 * factor))
        legend_u.add_updater(anchor_dl)

        tracker_u = ValueTracker(0.0)

        def make_upd_u(b):
            def upd(mob):
                mob.move_to(pos_point_u(b, tracker_u.get_value()))
            return upd
        for b in range(3):
            dots_u[b].add_updater(make_upd_u(b))

        trails_u = VGroup(*[
            TracedPath(dots_u[b].get_center, stroke_color=[C_INNER1, C_INNER2, C_OUTER][b],
                       stroke_width=2.5, dissipating_time=2.0)
            for b in range(3)
        ])

        counter_u = VGroup(
            Text("periods elapsed: ", font_size=22, color=WHITE),
            DecimalNumber(0.0, num_decimal_places=1, font_size=28, color=WHITE),
        ).arrange(RIGHT, buff=0.1)
        counter_u_base_height = counter_u.height

        t_periods_u = t_u / T_in_u

        def upd_counter_u(mob):
            f = tracker_u.get_value()
            val = float(np.interp(f, np.arange(n_u), t_periods_u))
            mob[1].set_value(val)
            mob.arrange(RIGHT, buff=0.1)
            factor = self.camera.frame.width / REF_WIDTH
            mob.scale_to_fit_height(counter_u_base_height * factor)
            corner = self.camera.frame.get_corner(DR)
            mob.move_to(corner + LEFT * (mob.width / 2 + 0.4 * factor) + UP * (mob.height / 2 + 0.3 * factor))
        counter_u.add_updater(upd_counter_u)

        # camera zoom-out updater: ratchets outward with the growing orbit,
        # tracking the real trajectory (never re-scaled arbitrarily).
        def upd_cam(mob):
            f = tracker_u.get_value()
            w = float(np.interp(f, np.arange(n_u), cam_width_series))
            mob.set(width=w)
            mob.move_to(ORIGIN)
        self.add(self.camera.frame)  # required so the frame's updater fires during play()
        self.camera.frame.add_updater(upd_cam)

        self.add(trails_u, dots_u, legend_u, counter_u)
        self.play(*[FadeIn(d, scale=0.5) for d in dots_u], FadeIn(legend_u), run_time=0.6)

        run_time_u = (t_u[-1] - t_u[0]) / TU_PER_SEC
        self.play(tracker_u.animate.set_value(n_u - 1), run_time=run_time_u, rate_func=linear)

        for d in dots_u:
            d.clear_updaters()
        counter_u.clear_updaters()
        self.camera.frame.clear_updaters()

        outcome_text = Text(
            f"disrupted at t = {t_disrupt_u:.1f}  (~{t_disrupt_u/T_in_u:.1f} inner periods)",
            font_size=26, color="#EF4444",
        )
        OUTCOME_BASE_H = outcome_text.height

        def upd_outcome(m):
            factor = self.camera.frame.width / REF_WIDTH
            m.scale_to_fit_height(OUTCOME_BASE_H * factor)
            m.next_to(label_u, DOWN, buff=0.3 * factor)
        outcome_text.add_updater(upd_outcome)
        self.play(FadeIn(outcome_text), run_time=0.6)
        self.wait(2.0)

        label_u.clear_updaters()
        outcome_text.clear_updaters()
        self.play(
            FadeOut(dots_u), FadeOut(trails_u), FadeOut(legend_u),
            FadeOut(counter_u), FadeOut(outcome_text), FadeOut(label_u), FadeOut(title),
            run_time=0.8,
        )
        title.clear_updaters()
        self.camera.frame.set(width=REF_WIDTH)
        self.camera.frame.move_to(ORIGIN)

        # ---------------- Closing caption card ----------------
        caption1 = Text(
            "Same inner binary, same masses — only the outer distance differs.",
            font_size=26, color=WHITE,
        )
        caption2 = Text(
            "a_out/a_in above ~3.29: bound for >=500 inner periods.  "
            "Below it: disrupted within ~18 inner periods.",
            font_size=22, color=GRAY_B,
        )
        caption_group = VGroup(caption1, caption2).arrange(DOWN, buff=0.3)
        caption_group.move_to(ORIGIN)
        self.play(FadeIn(caption_group), run_time=1.0)
        self.wait(3.0)
        self.play(FadeOut(caption_group), run_time=1.0)
        self.wait(0.3)
