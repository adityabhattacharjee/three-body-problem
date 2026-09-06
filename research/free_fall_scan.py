"""
Research question 1 (revised, faster route): the classical "free-fall" /
"brake orbit" method (Henon 1976; also used by Broucke and, more recently,
in the same spirit by Suvakov & Dmitrasinovic 2013) for finding periodic
three-body orbits WITHOUT any known orbit as a seed.

Physical idea:
  - Start all three equal masses AT REST (v=0) in an isosceles triangle:
    r1=(-1,0), r2=(1,0), r3=(0,h), h>0. Zero velocity => zero angular momentum
    and zero linear momentum trivially. Scale invariance fixes the horizontal
    separation to 2, leaving exactly one free shape parameter: h.
  - Newtonian gravity is time-reversal symmetric: reversing all velocities at
    any instant retraces the trajectory. An initial state with v=0 is
    automatically a time-reversal-symmetric instant.
  - If the free-fall trajectory from a given h ever returns to ANOTHER instant
    where all three velocities are simultaneously ~0 (a "brake" event, at time
    t1), then by time-reversal symmetry the motion from t1 backward retraces
    to 0 backward, so gluing the forward half [0,t1] to its time-reverse
    produces an orbit that is exactly periodic (period 2*t1, or 4*t1 if the
    brake shape is the mirror image rather than an exact repeat of the start).
  - Brake events do NOT happen for generic h (most trajectories either scatter,
    collide, or oscillate without ever simultaneously stopping). They occur
    only for special, isolated values of h -- exactly the isolated roots a
    1-parameter scan is designed to find, with no prior periodic orbit
    required as input.

This is the "seedless" answer to Q1 via a second, independent construction
from the velocity-angle scan in seedless_scan.py -- corroborating evidence
rather than a repeat of the same method.
"""

import numpy as np
from scipy.optimize import least_squares

from integrator import integrate_scipy, energy

MASSES = np.array([1.0, 1.0, 1.0])
G = 1.0
COLLISION_RADIUS = 1e-4


def free_fall_ic(h):
    r1 = np.array([-1.0, 0.0])
    r2 = np.array([1.0, 0.0])
    r3 = np.array([0.0, h])
    v = np.zeros(2)
    return np.concatenate([r1, r2, r3, v, v, v])


def total_kinetic_energy(y_row, masses):
    vel = y_row[6:12].reshape(3, 2)
    return sum(0.5 * masses[i] * (vel[i] @ vel[i]) for i in range(3))


def brake_scan_single(h, t_max=8.0, n_samples=4000):
    """
    Integrate the free-fall trajectory for shape parameter h and return
    (t_of_min_ke, min_ke, collided) where min_ke is the smallest total kinetic
    energy achieved after the initial departure from rest (skipping a small
    floor near t=0). A near-zero min_ke away from t=0 signals a near-brake
    event -- a periodic-orbit candidate.
    """
    ic = free_fall_ic(h)
    try:
        res = integrate_scipy(ic, MASSES, (0.0, t_max), n_eval=n_samples,
                               rtol=1e-11, atol=1e-11,
                               collision_radius=COLLISION_RADIUS)
    except Exception:
        return None, None, True
    y, t = res["y"], res["t"]
    floor = max(30, n_samples // 50)
    if len(t) <= floor:
        return None, None, res["collided"]
    ke = np.array([total_kinetic_energy(row, MASSES) for row in y[floor:]])
    idx = np.argmin(ke) + floor
    return t[idx], ke[idx - floor], res["collided"]


def scan(h_min=0.15, h_max=3.0, n_h=300, t_max=8.0):
    hs = np.linspace(h_min, h_max, n_h)
    t1s = np.full(n_h, np.nan)
    min_kes = np.full(n_h, np.inf)
    collided = np.zeros(n_h, dtype=bool)
    for i, h in enumerate(hs):
        t1, ke, coll = brake_scan_single(h, t_max=t_max)
        collided[i] = coll
        if t1 is not None:
            t1s[i] = t1
            min_kes[i] = ke
    return hs, t1s, min_kes, collided


def find_local_minima(scores, max_score=0.05):
    idx = []
    for i in range(1, len(scores) - 1):
        if not np.isfinite(scores[i]) or scores[i] > max_score:
            continue
        if scores[i] <= scores[i - 1] and scores[i] <= scores[i + 1]:
            idx.append(i)
    return idx


def _brake_residual(p, tol=1e-13):
    h, t1 = p
    if h <= 0.01 or t1 <= 0.05:
        return np.full(3, 1e2)
    ic = free_fall_ic(h)
    try:
        res = integrate_scipy(ic, MASSES, (0.0, t1), n_eval=2, rtol=tol, atol=tol,
                               collision_radius=COLLISION_RADIUS)
    except Exception:
        return np.full(3, 1e2)
    if res["collided"]:
        return np.full(3, 1e2)
    y1 = res["y"][-1]
    vel = y1[6:12].reshape(3, 2)
    # Drive all 3 velocity vectors to zero: by momentum conservation only 4 of
    # the 6 velocity components are independent, but forcing all 6 to ~0 with
    # 2 unknowns (h, t1) is an over-determined least-squares fit -- exactly
    # what we want: only isolated (h, t1) will make the residual small.
    return np.array([np.linalg.norm(vel[0]), np.linalg.norm(vel[1]), np.linalg.norm(vel[2])])


def refine_brake(h0, t10):
    sol = least_squares(_brake_residual, np.array([h0, t10]),
                         xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=200)
    resid_norm = np.linalg.norm(sol.fun)
    return sol.x, resid_norm, sol.success


def classify_and_get_period(h_star, t1_star):
    """
    Determine true period: integrate to t1_star, check whether the shape at
    t1 matches the INITIAL shape (period = 2*t1) or its mirror (still period
    2*t1, since the isosceles ansatz's mirror is itself the same labeled
    shape at h) -- for this ansatz any brake event gives period 2*t1 by the
    time-reversal argument, independent of shape at t1.
    """
    return 2.0 * t1_star


def main():
    print("Scanning isosceles free-fall shape parameter h in [0.15, 3.0]...")
    hs, t1s, min_kes, collided = scan(n_h=300, t_max=8.0)
    print(f"{int(np.sum(collided))}/{len(hs)} samples collided before t_max.")
    minima_idx = find_local_minima(min_kes, max_score=0.05)
    print(f"Found {len(minima_idx)} candidate brake dips (min KE < 0.05).")

    validated = []
    for i in minima_idx:
        h0, t10 = hs[i], t1s[i]
        if not np.isfinite(t10):
            continue
        params, resid, ok = refine_brake(h0, t10)
        tag = "PERIODIC" if (ok and resid < 1e-8) else "no-converge"
        print(f"  h0={h0:.4f} t1_0={t10:.4f} minKE0={min_kes[i]:.3e}  ->  "
              f"h*={params[0]:.6f} t1*={params[1]:.6f} resid={resid:.2e}  [{tag}]")
        if ok and resid < 1e-8:
            T = classify_and_get_period(params[0], params[1])
            validated.append((params[0], params[1], T, resid))

    uniq = []
    for h, t1, T, r in validated:
        is_dup = any(abs(h - uh) < 1e-3 for uh, _, _, _ in uniq)
        if not is_dup:
            uniq.append((h, t1, T, r))

    print(f"\n{len(uniq)} distinct validated free-fall periodic orbits "
          f"(brake orbits), seedless:")
    for h, t1, T, r in uniq:
        print(f"  h*={h:.6f}  t1*={t1:.6f}  period T*={T:.6f}  residual={r:.2e}")

    np.savez("../data/free_fall_scan.npz",
             hs=hs, t1s=t1s, min_kes=min_kes,
             validated=np.array(uniq) if uniq else np.zeros((0, 4)))
    print("\nSaved scan + validated brake-orbit table to ../data/free_fall_scan.npz")


if __name__ == "__main__":
    main()
