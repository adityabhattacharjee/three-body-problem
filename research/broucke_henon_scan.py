"""
Research question 1, third and most tractable seedless construction: the
classical "Broucke-Henon" collinear symmetric scan (Broucke 1975, "Periodic
Orbits in the Three-Body Problem with Equal Masses", Cel. Mech. 12; Henon
1976). This is the historically first demonstration that a 1-parameter,
no-prior-orbit scan reliably yields many distinct periodic three-body orbits,
and it is why that whole family carries these two names.

Symmetric-syzygy method (why it needs no seed, and why it is more tractable
than the free-velocity-angle or free-fall scans tried first in this project):

  - Start collinear: r1=(-1,0), r2=(1,0), r3=(0,0) -- an Euler syzygy.
  - Start with velocities purely perpendicular to that line:
    v1=v2=(0,v), v3=(0,-2v)  (zero net momentum; L=0 automatically, since
    L = m[(-1)(v) - 0] + m[(1)(v) - 0] + 0 = 0 for any v).
  - This configuration is a fixed point of the symmetry
        S: (x,y) -> (-x,y), bodies 1<->2 swapped, combined with time reversal.
    Newtonian gravity commutes with S. A classical theorem for symmetric
    Hamiltonian systems (see e.g. Roberts 2007; Broucke 1975) then says: if
    the trajectory returns to ANOTHER instant with the SAME symmetry type
    (collinear again, velocities again perpendicular to the instantaneous
    line -- a "perpendicular syzygy") at some time t1, the full orbit is
    automatically exactly periodic, with period 2*t1 or 4*t1 depending on
    whether the bodies have returned to their original cyclic order or the
    swapped one. No proximity-functional guessing is needed to locate t1: a
    perpendicular syzygy is a codimension-1 event (a real zero-crossing of a
    smooth function of time) that scipy's event detection finds directly and
    precisely, for every trial v -- this is what makes the search tractable
    (contrast with seedless_scan.py's alpha-scan and free_fall_scan.py's
    brake-scan, both of which struggled empirically in this project's own
    runs; see notes.md).
  - The single free (scan) parameter is v = |v1|. No known periodic orbit is
    used to choose v -- v is swept over an interval and EVERY perpendicular
    syzygy the scan detects is a candidate, independent of prior literature.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from integrator import three_body_rhs, energy, angular_momentum

MASSES = np.array([1.0, 1.0, 1.0])
G = 1.0
COLLISION_RADIUS = 1e-4


def bh_ic(v):
    r1 = np.array([-1.0, 0.0])
    r2 = np.array([1.0, 0.0])
    r3 = np.array([0.0, 0.0])
    v1 = np.array([0.0, v])
    v2 = np.array([0.0, v])
    v3 = np.array([0.0, -2.0 * v])
    return np.concatenate([r1, r2, r3, v1, v2, v3])


def _collinearity(y):
    pos = y[0:6].reshape(3, 2)
    d1 = pos[1] - pos[0]
    d2 = pos[2] - pos[0]
    return d1[0] * d2[1] - d1[1] * d2[0]  # 2x signed triangle area


def _min_pair_distance(y):
    pos = y[0:6].reshape(3, 2)
    return min(np.linalg.norm(pos[0] - pos[1]),
               np.linalg.norm(pos[0] - pos[2]),
               np.linalg.norm(pos[1] - pos[2]))


def find_syzygies(v, t_max=20.0, n_events_needed=3):
    """
    Integrate forward from bh_ic(v) and return the times of the first
    n_events_needed perpendicular-syzygy candidates: zero-crossings of the
    collinearity function (any direction), excluding t~0, with a collision
    guard so a genuine finite-time collapse terminates the search early
    rather than crashing the integrator.
    """
    y0 = bh_ic(v)

    def collinear_event(t, y, masses):
        return _collinearity(y)
    collinear_event.direction = 0

    def collision_event(t, y, masses):
        return _min_pair_distance(y) - COLLISION_RADIUS
    collision_event.terminal = True
    collision_event.direction = -1

    sol = solve_ivp(three_body_rhs, (0.0, t_max), y0, args=(MASSES,),
                     method="DOP853", rtol=1e-11, atol=1e-11,
                     events=[collinear_event, collision_event],
                     dense_output=True, max_step=t_max / 200)
    syzygy_times = sol.t_events[0]
    collided = len(sol.t_events[1]) > 0
    syzygy_times = syzygy_times[syzygy_times > 1e-3]
    return syzygy_times[:n_events_needed], collided


def scan(v_min=0.05, v_max=1.2, n_v=250, t_max=20.0):
    vs = np.linspace(v_min, v_max, n_v)
    results = []  # list of (v, [syzygy times], collided)
    for v in vs:
        try:
            times, collided = find_syzygies(v, t_max=t_max)
        except Exception:
            times, collided = np.array([]), True
        results.append((v, times, collided))
    return results


def _residual_vector(p, period_multiple, tol=1e-13):
    v, t1 = p
    if v <= 1e-6 or t1 <= 0.02:
        return np.full(12, 1e2)
    T = period_multiple * t1
    y0 = bh_ic(v)
    try:
        sol = solve_ivp(three_body_rhs, (0.0, T), y0, args=(MASSES,),
                         method="DOP853", rtol=tol, atol=tol)
    except Exception:
        return np.full(12, 1e2)
    if not sol.success:
        return np.full(12, 1e2)
    y1 = sol.y[:, -1]
    return y1 - y0


def refine_bh(v0, t1_0, period_multiple):
    sol = least_squares(_residual_vector, np.array([v0, t1_0]), args=(period_multiple,),
                         xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=200)
    resid_norm = np.linalg.norm(sol.fun)
    return sol.x, resid_norm, sol.success


def main():
    print("Scanning Broucke-Henon perpendicular-syzygy ansatz, v in [0.05, 1.2]...")
    results = scan(n_v=250, t_max=20.0)
    n_collided = sum(1 for _, _, c in results if c)
    n_with_syzygy = sum(1 for _, t, _ in results if len(t) > 0)
    print(f"{n_collided}/{len(results)} collided before finding 3 syzygies; "
          f"{n_with_syzygy}/{len(results)} produced at least one syzygy candidate.")

    validated = []
    tried = 0
    for v, times, collided in results:
        if len(times) == 0:
            continue
        for k, t1 in enumerate(times):
            for mult in (2.0, 4.0):
                tried += 1
                params, resid, ok = refine_bh(v, t1, mult)
                if ok and resid < 1e-7:
                    v_star, t1_star = params
                    T_star = mult * t1_star
                    E0 = energy(bh_ic(v_star), MASSES)
                    L0 = angular_momentum(bh_ic(v_star), MASSES)
                    validated.append((v_star, T_star, resid, E0, L0))
        if len(validated) >= 40:
            break

    print(f"\nRefinement attempted on {tried} (v, syzygy, period-multiple) triples.")

    uniq = []
    for v, T, r, E0, L0 in validated:
        is_dup = any(abs(v - uv) < 1e-3 and abs(T - uT) < 1e-2 for uv, uT, _, _, _ in uniq)
        if not is_dup:
            uniq.append((v, T, r, E0, L0))
    uniq.sort(key=lambda row: row[0])

    print(f"\n{len(uniq)} distinct validated Broucke-Henon-type periodic orbits "
          f"found, seedless:")
    for v, T, r, E0, L0 in uniq:
        print(f"  v*={v:.6f}  T*={T:.6f}  E={E0:.6f}  L={L0:.2e}  residual={r:.2e}")

    np.savez("../data/broucke_henon_scan.npz",
             validated=np.array(uniq) if uniq else np.zeros((0, 5)))
    print("\nSaved validated orbit table to ../data/broucke_henon_scan.npz")


if __name__ == "__main__":
    main()
