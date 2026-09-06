"""
Precompute two real trajectories for the hierarchical-triple scene:
  - STABLE case (a_out/a_in = 6.0): integrate ~50 real inner periods.
  - UNSTABLE case (a_out/a_in = 1.8): integrate (0, 90) to capture the real
    disruption at t ~ 80.55 (~18.1 inner periods).
Run standalone (NOT inside Manim). Saves hierarchical_traj.npz.
"""
import sys
import os
import numpy as np

BASE = "/Users/abhijit/Library/CloudStorage/OneDrive-Personal/Documents/Aditya/SBC/Physics/three_body"
sys.path.insert(0, os.path.join(BASE, "research"))

from integrator import integrate_scipy, energy, angular_momentum  # noqa: E402
from hierarchical_stability import hierarchical_ic, mardling_aarseth_ratio  # noqa: E402

MASSES = np.array([1.0, 1.0, 1.0])
CRIT_RATIO = mardling_aarseth_ratio(0.5)
print(f"Mardling-Aarseth critical ratio: {CRIT_RATIO:.3f}")

# ---------------- STABLE (ratio = 6.0) ----------------
ic_s, T_in_s = hierarchical_ic(1.0, 6.0)
n_periods_s = 50
t_span_s = (0.0, n_periods_s * T_in_s)
E0_s = energy(ic_s, MASSES)
L0_s = angular_momentum(ic_s, MASSES)
res_s = integrate_scipy(ic_s, MASSES, t_span_s, n_eval=2400, rtol=1e-11, atol=1e-11,
                         collision_radius=1e-4)
y_s = res_s["y"]
t_s = res_s["t"]
E1_s = energy(y_s[-1], MASSES)
L1_s = angular_momentum(y_s[-1], MASSES)
print(f"STABLE: T_in={T_in_s:.4f}, integrated {n_periods_s} periods "
      f"({t_span_s[1]:.2f} tu), samples={len(t_s)}, collided={res_s['collided']}")
print(f"  |dE|={abs(E1_s-E0_s):.3e}, |dL|={abs(L1_s-L0_s):.3e}")

pos_s = y_s[:, 0:6].reshape(-1, 3, 2)
d01_s = np.linalg.norm(pos_s[:, 0] - pos_s[:, 1], axis=1)
print(f"  inner separation range: [{d01_s.min():.3f}, {d01_s.max():.3f}] "
      f"(a_in=1.0 -> should stay of this order)")

# ---------------- UNSTABLE (ratio = 1.8) ----------------
ic_u, T_in_u = hierarchical_ic(1.0, 1.8)
t_span_u = (0.0, 90.0)
E0_u = energy(ic_u, MASSES)
L0_u = angular_momentum(ic_u, MASSES)
res_u = integrate_scipy(ic_u, MASSES, t_span_u, n_eval=3000, rtol=1e-11, atol=1e-11,
                         collision_radius=1e-4)
y_u = res_u["y"]
t_u = res_u["t"]
print(f"UNSTABLE: T_in={T_in_u:.4f}, requested span {t_span_u}, "
      f"samples={len(t_u)}, collided={res_u['collided']}")
if res_u["collided"]:
    print(f"  collision at t={res_u['t_collision']:.3f} "
          f"({res_u['t_collision']/T_in_u:.2f} inner periods)")

pos_u = y_u[:, 0:6].reshape(-1, 3, 2)
d01_u = np.linalg.norm(pos_u[:, 0] - pos_u[:, 1], axis=1)
d02_u = np.linalg.norm(pos_u[:, 0] - pos_u[:, 2], axis=1)
d12_u = np.linalg.norm(pos_u[:, 1] - pos_u[:, 2], axis=1)
max_any_u = np.maximum(np.maximum(d01_u, d02_u), d12_u)
esc_idx = np.where(max_any_u > 20.0 * 1.0)[0]
if len(esc_idx) > 0:
    print(f"  escape criterion crossed at t={t_u[esc_idx[0]]:.3f} "
          f"({t_u[esc_idx[0]]/T_in_u:.2f} inner periods)")

out_path = os.path.join(BASE, "manim", "scene_b_hierarchical", "hierarchical_traj.npz")
np.savez(
    out_path,
    pos_stable=pos_s, t_stable=t_s, T_in_stable=T_in_s, n_periods_stable_shown=n_periods_s,
    pos_unstable=pos_u, t_unstable=t_u, T_in_unstable=T_in_u,
    unstable_collided=res_u["collided"],
    t_disruption_unstable=(res_u["t_collision"] if res_u["collided"] else t_u[esc_idx[0]] if len(esc_idx) else t_u[-1]),
    crit_ratio=CRIT_RATIO,
)
print(f"Saved to {out_path}")
