"""
Precompute a dense, physically exact figure-eight trajectory for Manim playback.
Run standalone (NOT inside the Manim scene) and save to figure8_traj.npz.
"""
import sys
import os
import numpy as np

BASE = "/Users/abhijit/Library/CloudStorage/OneDrive-Personal/Documents/Aditya/SBC/Physics/three_body"
sys.path.insert(0, os.path.join(BASE, "research"))

from integrator import integrate_scipy  # noqa: E402

d = np.load(os.path.join(BASE, "data", "figure8_validated.npz"))
ic = d["ic"]
T = float(d["T"])
masses = d["masses"]
max_cross_diff = float(d["max_cross_integrator_diff"])

print(f"Loaded figure8_validated.npz: T={T:.6f}, masses={masses}, "
      f"max_cross_integrator_diff={max_cross_diff:.3e}")

t_span = (0.0, 2.5 * T)
res = integrate_scipy(ic, masses, t_span, n_eval=1500, rtol=1e-11, atol=1e-11)
t = res["t"]
y = res["y"]  # (1500, 12)

pos = y[:, 0:6].reshape(-1, 3, 2)  # (1500, 3, 2)

print(f"Integrated {t_span} -> t shape {t.shape}, pos shape {pos.shape}")
print(f"pos range x: [{pos[:,:,0].min():.3f}, {pos[:,:,0].max():.3f}], "
      f"y: [{pos[:,:,1].min():.3f}, {pos[:,:,1].max():.3f}]")

out_path = os.path.join(BASE, "manim", "scene_a_figure8", "figure8_traj.npz")
np.savez(out_path, t=t, pos=pos, T=T, max_cross_diff=max_cross_diff)
print(f"Saved to {out_path}")
