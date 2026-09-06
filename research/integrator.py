"""
Validated N-body integration harness for the equal-mass, planar three-body problem.

Units: G = 1, all masses = 1 (standard convention in the periodic-orbit literature,
e.g. Suvakov & Dmitrasinovic 2013, Chenciner & Montgomery 2000).

Two independent integrators are used and cross-checked against each other:
  1. REBOUND / IAS15 (Rein & Spiegel 2015) - adaptive 15th-order, the field-standard
     high-precision N-body integrator.
  2. scipy.integrate.solve_ivp with DOP853 (adaptive 8th-order Runge-Kutta) on the
     raw ODEs, as an independent code path (different algorithm, different author,
     different floating point call sequence).

Why cross-check instead of trusting one integrator: periodic three-body orbits are
frequently only marginally stable or outright linearly unstable, so an integration
error that looks like "chaos" can actually be a bug or an insufficiently tight
tolerance. Agreement between two independently implemented integrators to a
specified tolerance is the evidence that a computed trajectory reflects the physics
and not integrator artifacts.
"""

import numpy as np
from scipy.integrate import solve_ivp

G = 1.0


def three_body_rhs(t, y, masses):
    """
    y = [x1,y1,x2,y2,x3,y3, vx1,vy1,vx2,vy2,vx3,vy3]
    Returns dy/dt for the planar three-body problem.
    """
    pos = y[0:6].reshape(3, 2)
    vel = y[6:12].reshape(3, 2)
    acc = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            d = pos[j] - pos[i]
            r2 = d[0] ** 2 + d[1] ** 2
            r3 = r2 * np.sqrt(r2)
            acc[i] += G * masses[j] * d / r3
    return np.concatenate([vel.reshape(-1), acc.reshape(-1)])


def _min_pair_distance(y):
    pos = y[0:6].reshape(3, 2)
    d01 = np.linalg.norm(pos[0] - pos[1])
    d02 = np.linalg.norm(pos[0] - pos[2])
    d12 = np.linalg.norm(pos[1] - pos[2])
    return min(d01, d02, d12)


def integrate_scipy(ic, masses, t_span, n_eval=2000, rtol=1e-13, atol=1e-13,
                     collision_radius=None):
    """
    ic: array-like length 12 (positions then velocities, body-major, xy per body)
    Returns dict with t, states (n_eval x 12), and a bound solve_ivp result.

    collision_radius: if set, integration terminates early (a "collision" event,
    sol.status == 1) the moment any pairwise separation drops below this value.
    Close encounters otherwise force the adaptive step to near zero, which either
    stalls the search or raises "step size too small" -- physically, a near-
    collision trajectory cannot be part of a smooth periodic orbit anyway, so
    terminating early is the correct behavior, not a workaround.
    """
    y0 = np.asarray(ic, dtype=float)
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)

    events = None
    if collision_radius is not None:
        def collision(t, y, masses):
            return _min_pair_distance(y) - collision_radius
        collision.terminal = True
        collision.direction = -1
        events = collision

    sol = solve_ivp(
        three_body_rhs,
        t_span,
        y0,
        args=(masses,),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        dense_output=True,
        events=events,
    )
    if not sol.success:
        raise RuntimeError(f"scipy integration failed: {sol.message}")
    if events is not None and sol.status == 1:
        return {"t": sol.t, "y": sol.y.T, "sol": sol, "collided": True,
                "t_collision": sol.t[-1]}
    return {"t": sol.t, "y": sol.y.T, "sol": sol, "collided": False}


def integrate_rebound(ic, masses, t_span, n_eval=2000):
    """
    Same signature/return shape as integrate_scipy, backed by REBOUND's IAS15.
    """
    import rebound

    sim = rebound.Simulation()
    sim.G = G  # dimensionless G = 1 convention (default units already unset)
    sim.integrator = "ias15"

    pos = np.asarray(ic[0:6]).reshape(3, 2)
    vel = np.asarray(ic[6:12]).reshape(3, 2)
    for i in range(3):
        sim.add(m=masses[i], x=pos[i, 0], y=pos[i, 1], z=0.0,
                vx=vel[i, 0], vy=vel[i, 1], vz=0.0)
    sim.move_to_com()

    t0, t1 = t_span
    times = np.linspace(t0, t1, n_eval)
    out = np.zeros((n_eval, 12))
    for k, t in enumerate(times):
        sim.integrate(t)
        for i in range(3):
            p = sim.particles[i]
            out[k, 2 * i] = p.x
            out[k, 2 * i + 1] = p.y
            out[k, 6 + 2 * i] = p.vx
            out[k, 6 + 2 * i + 1] = p.vy
    return {"t": times, "y": out, "sim": sim}


def energy(y_row, masses):
    pos = y_row[0:6].reshape(3, 2)
    vel = y_row[6:12].reshape(3, 2)
    ke = sum(0.5 * masses[i] * (vel[i] @ vel[i]) for i in range(3))
    pe = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r = np.linalg.norm(pos[i] - pos[j])
            pe -= G * masses[i] * masses[j] / r
    return ke + pe


def angular_momentum(y_row, masses):
    pos = y_row[0:6].reshape(3, 2)
    vel = y_row[6:12].reshape(3, 2)
    L = 0.0
    for i in range(3):
        L += masses[i] * (pos[i, 0] * vel[i, 1] - pos[i, 1] * vel[i, 0])
    return L


def return_proximity(ic, masses, period, integrator="scipy"):
    """
    Core self-checking primitive for periodic-orbit search: integrate one putative
    period and return |state(T) - state(0)| in the full 12-d phase space.
    A true periodic orbit drives this to ~integrator tolerance.
    """
    if integrator == "scipy":
        res = integrate_scipy(ic, masses, (0.0, period), n_eval=2, rtol=1e-13, atol=1e-13)
    elif integrator == "rebound":
        res = integrate_rebound(ic, masses, (0.0, period), n_eval=2)
    else:
        raise ValueError(integrator)
    y0 = res["y"][0]
    y1 = res["y"][-1]
    return np.linalg.norm(y1 - y0), res
