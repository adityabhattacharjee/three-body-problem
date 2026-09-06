"""
Research question 1: without a known periodic orbit as a starting seed, can we
find MULTIPLE families of exactly-periodic ("stable ad infinitum" in the paper's
loose sense -- see taxonomy.md for the precise distinction from linear stability)
equal-mass three-body orbits?

Method: the free-fall / collinear scan of Suvakov & Dmitrasinovic (Phys. Rev. Lett.
110, 114301, 2013), reduced to its symmetry-and-conservation-law skeleton -- no
prior orbit family is assumed or hand-fed in as an initial guess.

Setup (uses only symmetry + Newton's laws, not any known solution):
  - Equal masses m1=m2=m3=1, G=1.
  - Collinear initial positions r1=(-1,0), r2=(1,0), r3=(0,0): the least-biased
    starting shape (an isosceles collinear configuration), fixed by the
    reflection symmetry of the equal-mass problem.
  - Zero total linear momentum: v1=v2=(vx,vy), v3=-2(vx,vy).
  - Zero total angular momentum is automatic for collinear r + parallel v1=v2.
  - Newtonian gravity is scale invariant, so WLOG fix the total energy E=-1;
    this pins |v1| for a given direction, leaving exactly one free parameter:
    the angle alpha of v1.
  - Scan alpha in [0, pi] (symmetry makes [pi, 2pi] redundant), integrate each
    trajectory, and score it with a return-proximity functional. Local minima
    of that functional are periodic-orbit CANDIDATES -- purely a byproduct of
    the scan, not seeded from literature.
  - Trajectories that suffer a close encounter (physically cannot be part of a
    smooth periodic orbit) are flagged and excluded from candidacy, rather than
    left to stall the integrator.
  - Each candidate is refined in two stages: first a derivative-free polish
    (Nelder-Mead on the scalar residual norm) to walk into the true orbit's
    basin of attraction, then Newton/least-squares to drive the residual to
    integrator precision, allowing the orbit to close up to a rigid rotation
    (many members of these families are periodic only in a uniformly rotating
    frame).

This directly answers Q1: yes, multiple distinct periodic families are
discoverable from a single symmetry ansatz with no prior seed, because the
scan variable (alpha) requires no knowledge of any existing periodic solution.
Whether any GIVEN scan run actually lands a converged orbit is an empirical,
reported result below -- not assumed.
"""

import numpy as np
from scipy.optimize import least_squares, minimize

from integrator import integrate_scipy

MASSES = np.array([1.0, 1.0, 1.0])
G = 1.0
COLLISION_RADIUS = 1e-3


def collinear_ic(alpha, E_target=-1.0):
    r1 = np.array([-1.0, 0.0])
    r2 = np.array([1.0, 0.0])
    r3 = np.array([0.0, 0.0])

    pe = -(G * 1 * 1 / np.linalg.norm(r1 - r2)
           + G * 1 * 1 / np.linalg.norm(r1 - r3)
           + G * 1 * 1 / np.linalg.norm(r2 - r3))

    # KE_target = E_target - PE ; v1=v2=(vx,vy), v3=-2(vx,vy)
    # KE = 0.5*(|v1|^2 + |v2|^2 + |v3|^2) = 0.5*(2|v|^2 + 4|v|^2) = 3|v|^2
    ke_target = E_target - pe
    if ke_target <= 0:
        return None
    speed = np.sqrt(ke_target / 3.0)

    v1 = speed * np.array([np.cos(alpha), np.sin(alpha)])
    v2 = v1.copy()
    v3 = -2 * v1

    return np.concatenate([r1, r2, r3, v1, v2, v3])


def proximity_functional(alpha, t_max=15.0, n_samples=1500, E_target=-1.0):
    """
    Integrate (bailing out early on close encounters) and return
    (best_return_time, min_residual), where residual is the minimum over
    sampled t of |X(t) - X(0)| restricted to the position sub-vector. A small
    residual means the SHAPE nearly recurs; exact periodicity (incl. any net
    rotation) is enforced later by refine_candidate.
    """
    ic = collinear_ic(alpha, E_target)
    if ic is None:
        return None
    try:
        res = integrate_scipy(ic, MASSES, (0.0, t_max), n_eval=n_samples,
                               rtol=1e-10, atol=1e-10,
                               collision_radius=COLLISION_RADIUS)
    except Exception:
        return None
    if res["collided"]:
        return None
    y = res["y"]
    t = res["t"]
    x0 = y[0, 0:6]
    floor = max(20, n_samples // 20)  # skip near t=0
    diffs = np.linalg.norm(y[floor:, 0:6] - x0[None, :], axis=1)
    if len(diffs) == 0:
        return None
    idx = np.argmin(diffs) + floor
    return t[idx], diffs[idx - floor]


def scan(n_alpha=400, t_max=15.0, E_target=-1.0):
    alphas = np.linspace(0.001, np.pi - 0.001, n_alpha)
    scores = np.full(n_alpha, np.inf)
    times = np.full(n_alpha, np.nan)
    for i, a in enumerate(alphas):
        r = proximity_functional(a, t_max=t_max, E_target=E_target)
        if r is not None:
            times[i], scores[i] = r
    return alphas, times, scores


def find_local_minima(scores, max_score=1.5):
    idx = []
    for i in range(1, len(scores) - 1):
        if not np.isfinite(scores[i]) or scores[i] > max_score:
            continue
        if scores[i] < scores[i - 1] and scores[i] < scores[i + 1]:
            idx.append(i)
    return idx


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _residual_vector(p, E_target, tol=1e-13):
    alpha, T, theta = p
    ic = collinear_ic(alpha, E_target)
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


def refine_candidate(alpha0, T0, E_target=-1.0):
    """
    Two-stage refinement of [alpha, T, theta]:
      1. Nelder-Mead polish on ||residual||^2 at loose integrator tolerance
         (derivative-free, robust to the rough landscape near a close
         encounter, cheap per-eval) to find the true basin.
      2. Newton/least-squares on the 12-d residual vector at tight integrator
         tolerance for a precise fix.
    """
    p0 = np.array([alpha0, T0, 0.0])

    def scalar_obj(p):
        return float(np.sum(_residual_vector(p, E_target, tol=1e-9) ** 2))

    polish = minimize(scalar_obj, p0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 300})
    p1 = polish.x
    if polish.fun > 1e-4:
        # polish didn't find a plausible basin; don't waste time on tight refinement
        return p1, float(np.sqrt(polish.fun)) * np.sqrt(12), False

    sol = least_squares(_residual_vector, p1, args=(E_target, 1e-13),
                         xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=200)
    resid_norm = np.linalg.norm(sol.fun)
    return sol.x, resid_norm, sol.success


def main():
    print("Scanning collinear free-fall-momentum ansatz, alpha in [0, pi]...")
    alphas, times, scores = scan(n_alpha=500, t_max=15.0)
    n_collided = int(np.sum(~np.isfinite(scores)))
    print(f"{n_collided}/{len(alphas)} samples hit a close encounter "
          f"(< {COLLISION_RADIUS}) before t_max and were excluded.")
    minima_idx = find_local_minima(scores)
    print(f"Found {len(minima_idx)} local minima of the return-proximity functional "
          f"(score < 1.5).")

    validated = []
    for i in minima_idx:
        a0, T0 = alphas[i], times[i]
        if not np.isfinite(T0):
            continue
        params, resid, ok = refine_candidate(a0, T0)
        tag = "PERIODIC" if (ok and resid < 1e-7) else "no-converge"
        print(f"  alpha0={a0:.4f} T0={T0:.3f} score0={scores[i]:.3e}  ->  "
              f"alpha*={params[0]:.5f} T*={params[1]:.5f} theta*={params[2]:.5f} "
              f"resid={resid:.2e}  [{tag}]")
        if ok and resid < 1e-7:
            validated.append((params[0], params[1], params[2], resid))

    # de-duplicate near-identical (alpha*, T*) pairs
    uniq = []
    for a, T, th, r in validated:
        is_dup = any(abs(a - ua) < 1e-3 and abs(T - uT) < 1e-2 for ua, uT, _, _ in uniq)
        if not is_dup:
            uniq.append((a, T, th, r))

    print(f"\n{len(uniq)} distinct validated periodic (up to rotation) orbits found, "
          f"from a single seedless alpha-scan:")
    for a, T, th, r in uniq:
        print(f"  alpha*={a:.6f}  T*={T:.6f}  theta*={th:.6f}  residual={r:.2e}")

    np.savez("../data/seedless_scan.npz",
             alphas=alphas, times=times, scores=scores,
             validated=np.array(uniq) if uniq else np.zeros((0, 4)))
    print("\nSaved scan + validated orbit table to ../data/seedless_scan.npz")


if __name__ == "__main__":
    main()
