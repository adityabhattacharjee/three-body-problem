"""
Precompute two illustrative example trajectories (one 'collision', one
'escape') for the survival-statistics scene, using the SAME random_ic /
integration machinery as research/escape_statistics.py (fixed seed for
reproducibility). The saved survival-curve panel of the scene instead loads
data/escape_statistics.npz directly (the real 200-trial ensemble) -- these
two example trajectories are for illustration only.
Run standalone (NOT inside Manim). Saves examples_traj.npz.
"""
import sys
import os
import numpy as np

BASE = "/Users/abhijit/Library/CloudStorage/OneDrive-Personal/Documents/Aditya/SBC/Physics/three_body"
sys.path.insert(0, os.path.join(BASE, "research"))

from integrator import integrate_scipy  # noqa: E402
from escape_statistics import random_ic, MASSES  # noqa: E402

T_MAX = 60.0
COLLISION_RADIUS = 1e-3
ESCAPE_RADIUS = 30.0


def classify_and_trim(ic, seed_label):
    res = integrate_scipy(ic, MASSES, (0.0, T_MAX), n_eval=3000, rtol=1e-9, atol=1e-9,
                           collision_radius=COLLISION_RADIUS)
    y = res["y"]
    t = res["t"]
    if res["collided"]:
        outcome = "collision"
        t_event = res["t_collision"]
        # densely sample right up to the collision instant using dense_output
        t_dense = np.linspace(0.0, t_event, 400)
        y_dense = res["sol"].sol(t_dense).T
        return outcome, t_event, t_dense, y_dense

    pos = y[:, 0:6].reshape(-1, 3, 2)
    com_dist = np.linalg.norm(pos, axis=2)
    max_dist = com_dist.max(axis=1)
    esc_idx = np.where(max_dist > ESCAPE_RADIUS)[0]
    if len(esc_idx) > 0:
        outcome = "escape"
        t_event = t[esc_idx[0]]
        cutoff = esc_idx[0] + 1
        return outcome, t_event, t[:cutoff], y[:cutoff]

    return "survived", t[-1], t, y


rng = np.random.default_rng(1)
found = {"collision": None, "escape": None}

attempts = 0
while (found["collision"] is None or found["escape"] is None) and attempts < 200:
    attempts += 1
    ic = random_ic(rng, E_target=-1.0)
    if ic is None:
        continue
    outcome, t_event, t_arr, y_arr = classify_and_trim(ic, attempts)
    print(f"attempt {attempts}: outcome={outcome}, t_event={t_event:.3f}")
    if outcome in found and found[outcome] is None:
        found[outcome] = (ic, t_event, t_arr, y_arr)

if found["collision"] is None or found["escape"] is None:
    raise RuntimeError(f"Could not find both outcomes within {attempts} draws: {found.keys()}")

out_path = os.path.join(BASE, "manim", "scene_c_survival", "examples_traj.npz")

ic_c, t_event_c, t_c, y_c = found["collision"]
ic_e, t_event_e, t_e, y_e = found["escape"]

pos_c = y_c[:, 0:6].reshape(-1, 3, 2)
pos_e = y_e[:, 0:6].reshape(-1, 3, 2)

print(f"\nCOLLISION example: t_event={t_event_c:.3f}, n_samples={len(t_c)}")
print(f"ESCAPE example: t_event={t_event_e:.3f}, n_samples={len(t_e)}")

np.savez(
    out_path,
    ic_collision=ic_c, t_collision=t_c, pos_collision=pos_c, t_event_collision=t_event_c,
    ic_escape=ic_e, t_escape=t_e, pos_escape=pos_e, t_event_escape=t_event_e,
)
print(f"Saved to {out_path}")
