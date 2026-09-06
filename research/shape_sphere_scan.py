"""
Research question 1, corrected construction. seedless_scan.py, free_fall_scan.py,
and broucke_henon_scan.py all started EXACTLY COLLINEAR with zero angular
momentum -- and all three, empirically (see notes.md), turned out to sample the
triple-collision manifold almost everywhere: with L=0 there is no centrifugal
barrier, and a collinear start is already on/near the singular set. The
validated figure-eight is the counterexample that pinpoints the fix: its IC
r1=(0.970, -0.243) is NOT collinear (an off-axis, scalene-ish triangle) despite
also having L=0.

Fix: scan the genuine 2-D shape parameter space (alpha, h) -- the same
zero-momentum, energy-normalized velocity ansatz as seedless_scan.py, but with
body 3 moved OFF the r1-r2 axis: r3=(0, h). This is exactly the 2-D
"shape sphere" sweep Suvakov & Dmitrasinovic actually used (a 1-D slice
through it, as tried first, is a measure-zero cut that this project's own
data showed lands on the collapse basin).

Cost control (bounded to a single grid sweep, per plan):
  - Loose integrator tolerance (1e-8) plus a hard collision-radius cutoff for
    the coarse grid pass.
  - A post-hoc "min separation > 0.05 throughout" filter applied to survivors
    before they are even considered proximity-functional candidates -- this
    is the cheap test that discards the collapse basin before paying for
    refinement.
  - Two-stage refinement (loose Nelder-Mead polish -> tight least_squares)
    applied only to the surviving local minima, and only to the best few.
"""

import numpy as np
from scipy.optimize import least_squares, minimize

from integrator import integrate_scipy

MASSES = np.array([1.0, 1.0, 1.0])
G = 1.0
COLLISION_RADIUS = 1e-3
SEPARATION_FLOOR = 0.05  # candidate filter: trajectory must never get closer than this


def shape_ic(alpha, h, E_target=-1.0):
    r1 = np.array([-1.0, 0.0])
    r2 = np.array([1.0, 0.0])
    r3 = np.array([0.0, h])

    d12 = np.linalg.norm(r1 - r2)
    d13 = np.linalg.norm(r1 - r3)
    d23 = np.linalg.norm(r2 - r3)
    pe = -(G / d12 + G / d13 + G / d23)

    ke_target = E_target - pe
    if ke_target <= 0:
        return None
    speed = np.sqrt(ke_target / 3.0)

    v1 = speed * np.array([np.cos(alpha), np.sin(alpha)])
    v2 = v1.copy()
    v3 = -2 * v1

    return np.concatenate([r1, r2, r3, v1, v2, v3])


def evaluate(alpha, h, t_max=10.0, n_samples=800, E_target=-1.0):
    """
    Returns None if invalid/collided/ever-too-close; else (t_of_best_return,
    residual_at_best_return, min_separation_seen).
    """
    ic = shape_ic(alpha, h, E_target)
    if ic is None:
        return None
    try:
        res = integrate_scipy(ic, MASSES, (0.0, t_max), n_eval=n_samples,
                               rtol=1e-8, atol=1e-8, collision_radius=COLLISION_RADIUS)
    except Exception:
        return None
    if res["collided"]:
        return None
    y, t = res["y"], res["t"]

    pos = y[:, 0:6].reshape(-1, 3, 2)
    d01 = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
    d02 = np.linalg.norm(pos[:, 0] - pos[:, 2], axis=1)
    d12 = np.linalg.norm(pos[:, 1] - pos[:, 2], axis=1)
    min_sep = min(d01.min(), d02.min(), d12.min())
    if min_sep < SEPARATION_FLOOR:
        return None

    # Exclude a physically-motivated departure window (not just a few samples):
    # the trajectory must actually leave the neighborhood of X(0) before a
    # "return" is meaningful, else argmin trivially picks t~0.
    x0 = y[0, 0:6]
    diffs_all = np.linalg.norm(y[:, 0:6] - x0[None, :], axis=1)
    departed = np.where(diffs_all > 0.75)[0]
    if len(departed) == 0:
        return None  # never even left the start -- not a useful candidate
    floor = departed[0]
    diffs = diffs_all[floor:]
    idx = np.argmin(diffs) + floor
    return t[idx], diffs[idx - floor], min_sep


def grid_scan(n_alpha=20, n_h=20, alpha_range=(0.02, np.pi - 0.02),
              h_range=(0.05, 1.5), t_max=10.0):
    alphas = np.linspace(*alpha_range, n_alpha)
    hs = np.linspace(*h_range, n_h)
    records = []  # (alpha, h, t_best, score, min_sep)
    n_survived = 0
    for a in alphas:
        for h in hs:
            r = evaluate(a, h, t_max=t_max)
            if r is not None:
                t_best, score, min_sep = r
                records.append((a, h, t_best, score, min_sep))
                n_survived += 1
    return records, n_survived, n_alpha * n_h


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _residual_vector(p, E_target=-1.0, tol=1e-13):
    alpha, h, T, theta = p
    ic = shape_ic(alpha, h, E_target)
    if ic is None or T <= 0.1:
        return np.full(12, 1e2)
    try:
        res = integrate_scipy(ic, MASSES, (0.0, T), n_eval=2, rtol=tol, atol=tol,
                               collision_radius=COLLISION_RADIUS)
    except Exception:
        return np.full(12, 1e2)
    if res["collided"]:
        return np.full(12, 1e2)
    y0, y1 = res["y"][0], res["y"][-1]
    R = _rot(theta)
    pos1 = y1[0:6].reshape(3, 2)
    vel1 = y1[6:12].reshape(3, 2)
    pos1_derot = (R.T @ pos1.T).T
    vel1_derot = (R.T @ vel1.T).T
    dpos = (pos1_derot - y0[0:6].reshape(3, 2)).reshape(-1)
    dvel = (vel1_derot - y0[6:12].reshape(3, 2)).reshape(-1)
    return np.concatenate([dpos, dvel])


def refine(alpha0, h0, T0):
    p0 = np.array([alpha0, h0, T0, 0.0])

    def scalar_obj(p):
        return float(np.sum(_residual_vector(p, tol=1e-9) ** 2))

    polish = minimize(scalar_obj, p0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 400})
    p1 = polish.x
    if polish.fun > 1e-3:
        return p1, float(np.sqrt(polish.fun * 12)), False

    sol = least_squares(_residual_vector, p1, xtol=1e-13, ftol=1e-13, gtol=1e-13,
                         max_nfev=300)
    resid_norm = np.linalg.norm(sol.fun)
    return sol.x, resid_norm, sol.success


def main():
    print("2-D shape-sphere grid scan: alpha in [0,pi], h in [0.05, 1.5], E=-1...")
    records, n_survived, n_total = grid_scan(n_alpha=24, n_h=24, t_max=12.0)
    print(f"{n_survived}/{n_total} grid points survived the collision + "
          f"separation-floor ({SEPARATION_FLOOR}) filter.")

    if not records:
        print("No survivors at all -- even off-axis, this energy/time window is "
              "collision-dominated. Reporting negative result; no refinement attempted.")
        return

    records.sort(key=lambda r: r[3])  # ascending score
    top = records[:12]
    print("\nTop 12 lowest-residual survivors (candidates for refinement):")
    for a, h, tb, score, msep in top:
        print(f"  alpha={a:.4f} h={h:.4f} t_best={tb:.3f} score={score:.4e} min_sep={msep:.4f}")

    validated = []
    for a, h, tb, score, msep in top:
        params, resid, ok = refine(a, h, tb)
        tag = "PERIODIC" if (ok and resid < 1e-7) else "no-converge"
        print(f"  refine(alpha={a:.4f}, h={h:.4f}) -> "
              f"alpha*={params[0]:.5f} h*={params[1]:.5f} T*={params[2]:.5f} "
              f"theta*={params[3]:.5f} resid={resid:.2e} [{tag}]")
        if ok and resid < 1e-7:
            validated.append(tuple(params) + (resid,))

    uniq = []
    for a, h, T, th, r in validated:
        is_dup = any(abs(a - ua) < 1e-3 and abs(h - uh) < 1e-3 and abs(T - uT) < 1e-2
                     for ua, uh, uT, _, _ in uniq)
        if not is_dup:
            uniq.append((a, h, T, th, r))

    print(f"\n{len(uniq)} distinct validated periodic orbits from the 2-D "
          f"shape-sphere scan (seedless):")
    for a, h, T, th, r in uniq:
        print(f"  alpha*={a:.6f} h*={h:.6f} T*={T:.6f} theta*={th:.6f} residual={r:.2e}")

    np.savez("../data/shape_sphere_scan.npz",
             records=np.array(records),
             validated=np.array(uniq) if uniq else np.zeros((0, 5)))
    print("\nSaved scan + validated orbit table to ../data/shape_sphere_scan.npz")


if __name__ == "__main__":
    main()
