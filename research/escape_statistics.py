"""
Research question 2, route 2 (statistical): for GENERIC (non-hierarchical)
equal-mass, zero-momentum, fixed-energy initial conditions -- the regime
where chaos, not a closed-form criterion, governs the outcome -- what
fraction survive (no collision, no escape) at least a given time T?

This is the "three-body scattering experiment" ensemble used throughout the
statistical treatments of the problem (see e.g. Valtonen & Karttunen's
textbook treatment, and Stone & Leigh 2019 Nature for the modern statistical
theory this project should cite and verify against, not assume).

Method:
  - Random triangle (COM at origin), rescaled to a fixed root-mean-square
    size, random velocity directions rescaled to hit a fixed target energy
    E=-1 with zero total momentum. Angular momentum is UNCONSTRAINED here
    (unlike the Q1 scans), which is what breaks the collapse-manifold
    dominance seen in Q1 -- generic L != 0 initial conditions are not on the
    collision manifold.
  - Integrate each trial to t_max (or until collision / escape) and record
    the disruption time.
  - The empirical survival curve S(T) = fraction of trials with
    disruption time > T directly answers Q2: for any desired T up to the
    simulated range, S(T) is the probability that a UNIFORMLY RANDOM draw
    already satisfies "survives >= T" -- and any individual trial that did
    survive to T is itself a determined/verified example, obtained without
    any special construction (contrast with hierarchical_stability.py's
    deterministic route, which needs no search at all but only applies to
    the hierarchical sub-family).
"""

import numpy as np

from integrator import integrate_scipy

G = 1.0
MASSES = np.array([1.0, 1.0, 1.0])
RNG_SEED = 20260906  # recorded for reproducibility, per project integrity rule


def random_ic(rng, E_target=-1.0, size_scale=1.0):
    # random triangle, COM at origin, rescaled to a fixed RMS size
    pos = rng.normal(size=(3, 2))
    pos -= pos.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(pos ** 2, axis=1)))
    pos *= size_scale / rms

    pe = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            pe -= G / np.linalg.norm(pos[i] - pos[j])

    # random velocity directions, zero-momentum, then scaled to hit E_target
    vel = rng.normal(size=(3, 2))
    vel -= vel.mean(axis=0, keepdims=True)  # zero total momentum (equal masses)
    ke_raw = 0.5 * np.sum(vel ** 2)

    ke_target = E_target - pe
    if ke_target <= 0:
        return None
    scale = np.sqrt(ke_target / ke_raw)
    vel *= scale

    return np.concatenate([pos.reshape(-1), vel.reshape(-1)])


def disruption_time(ic, t_max=200.0, escape_radius=30.0, collision_radius=1e-3,
                     n_eval=4000):
    try:
        res = integrate_scipy(ic, MASSES, (0.0, t_max), n_eval=n_eval,
                               rtol=1e-9, atol=1e-9, collision_radius=collision_radius)
    except Exception:
        return 0.0, "integration-error"
    if res["collided"]:
        return res["t"][-1], "collision"

    y, t = res["y"], res["t"]
    pos = y[:, 0:6].reshape(-1, 3, 2)
    com_dist = np.linalg.norm(pos, axis=2)  # each body's distance from COM (origin)
    max_dist = com_dist.max(axis=1)
    esc_idx = np.where(max_dist > escape_radius)[0]
    if len(esc_idx) > 0:
        return t[esc_idx[0]], "escape"
    return t_max, "survived"


def run_ensemble(n_trials=300, t_max=200.0, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_trials):
        ic = random_ic(rng)
        if ic is None:
            continue
        t_dis, outcome = disruption_time(ic, t_max=t_max)
        records.append((t_dis, outcome))
    return records


def survival_curve(records, t_grid):
    times = np.array([r[0] for r in records])
    return np.array([(times >= T).mean() for T in t_grid])


def main():
    print(f"Running random scattering ensemble (seed={RNG_SEED}), E=-1, "
          f"zero total momentum, unconstrained angular momentum...")
    n_trials = 200
    t_max = 200.0
    records = run_ensemble(n_trials=n_trials, t_max=t_max)

    outcomes = {}
    for _, o in records:
        outcomes[o] = outcomes.get(o, 0) + 1
    print(f"\n{len(records)} valid trials. Outcome breakdown:")
    for o, c in sorted(outcomes.items(), key=lambda kv: -kv[1]):
        print(f"  {o:10s}: {c:4d}  ({100*c/len(records):.1f}%)")

    t_grid = np.array([1, 2, 5, 10, 20, 50, 100, 150, 200])
    curve = survival_curve(records, t_grid)
    print(f"\nSurvival curve S(T) = fraction with disruption time >= T:")
    for T, s in zip(t_grid, curve):
        print(f"  T={T:6.1f}  S(T)={s:.3f}")

    survivors = [r for r in records if r[1] == "survived"]
    print(f"\n{len(survivors)}/{len(records)} trials survived the full "
          f"t_max={t_max} window -- each is a concrete, verified example "
          f"answering Q2 for T up to {t_max}.")

    np.savez("../data/escape_statistics.npz",
             disruption_times=np.array([r[0] for r in records]),
             outcomes=np.array([r[1] for r in records]),
             t_grid=t_grid, survival_curve=curve, n_trials=n_trials, t_max=t_max)
    print("\nSaved ensemble results to ../data/escape_statistics.npz")


if __name__ == "__main__":
    main()
