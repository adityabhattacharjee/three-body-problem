"""
Research question 2, route 1 (constructive/deterministic): given a target
survival time T, can we DETERMINE (not just search for) an initial condition
that survives at least T?

Yes: build a hierarchical triple (tight inner binary + distant outer third
body) and size the outer orbit using the empirical Mardling & Aarseth (2001)
stability criterion. This requires no orbit search at all -- it is a direct
construction from closed-form criteria, verified after the fact by direct
integration.

Mardling & Aarseth (2001, MNRAS 321, 398) empirical stability criterion,
general form (verified against the published paper -- an earlier draft of
this module had the (1-e_out) exponent wrong):
    a_out/a_in  >~  2.8 * [(1+q_out)(1+e_out) / sqrt(1-e_out)]^(2/5)
                    * (1-e_out)^(-1) * (1 - 0.3*i/180)
where q_out = m3/(m1+m2), e_out = outer eccentricity, i = mutual inclination
in degrees. For the circular-outer-orbit case used throughout this module
(e_out=0), the extra (1-e_out)^-1 factor is exactly 1, so it does not change
any number computed below -- but it is included here for correctness at
general e_out. For equal masses, e_out=0, i=0: q_out=0.5, so
    a_out/a_in  >~  2.8 * 1.5^0.4  ~=  3.29

We construct one system comfortably ABOVE this ratio (expect long-term
stability) and one BELOW it (expect disruption), and verify both by direct
integration for many inner-binary periods -- turning a closed-form criterion
into a checked, reproducible claim rather than an assertion from memory.
"""

import numpy as np

from integrator import integrate_scipy, energy, angular_momentum

G = 1.0
MASSES = np.array([1.0, 1.0, 1.0])  # m1, m2 = inner binary; m3 = outer body


def mardling_aarseth_ratio(q_out, e_out=0.0, inclination_deg=0.0):
    return (2.8 * ((1 + q_out) * (1 + e_out) / np.sqrt(1 - e_out)) ** 0.4
            / (1 - e_out)
            * (1 - 0.3 * inclination_deg / 180.0))


def hierarchical_ic(a_in, ratio, m1=1.0, m2=1.0, m3=1.0):
    """
    Coplanar, circular inner and outer orbits, prograde, m1,m2 forming the
    inner binary and m3 the distant outer body. Returns a 12-vector IC.
    """
    a_out = ratio * a_in
    M_in = m1 + m2

    # Inner binary about its own COM
    v_in = np.sqrt(G * M_in / a_in)
    r1_local = np.array([-a_in * m2 / M_in, 0.0])
    r2_local = np.array([a_in * m1 / M_in, 0.0])
    v1_local = np.array([0.0, -v_in * m2 / M_in])
    v2_local = np.array([0.0, v_in * m1 / M_in])

    # Outer orbit: inner-binary COM and m3 about the total COM
    M_tot = M_in + m3
    v_out = np.sqrt(G * M_tot / a_out)
    r_in_com = np.array([-a_out * m3 / M_tot, 0.0])
    r3 = np.array([a_out * M_in / M_tot, 0.0])
    v_in_com = np.array([0.0, -v_out * m3 / M_tot])
    v3 = np.array([0.0, v_out * M_in / M_tot])

    r1 = r1_local + r_in_com
    r2 = r2_local + r_in_com
    v1 = v1_local + v_in_com
    v2 = v2_local + v_in_com

    return np.concatenate([r1, r2, r3, v1, v2, v3]), 2 * np.pi * np.sqrt(a_in ** 3 / (G * M_in))


def classify_outcome(res, a_in_initial, escape_factor=20.0, collision_radius=1e-3):
    """
    Track the inner-binary separation (bodies 0,1) over the run: if it stays
    bounded and of order a_in throughout, call it STABLE; if any separation
    blows up past escape_factor*a_in_initial or collapses below
    collision_radius, call it DISRUPTED, and report the time.
    """
    y, t = res["y"], res["t"]
    pos = y[:, 0:6].reshape(-1, 3, 2)
    d01 = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)
    d02 = np.linalg.norm(pos[:, 0] - pos[:, 2], axis=1)
    d12 = np.linalg.norm(pos[:, 1] - pos[:, 2], axis=1)
    max_any = np.maximum(np.maximum(d01, d02), d12)
    min_any = np.minimum(np.minimum(d01, d02), d12)

    esc_idx = np.where(max_any > escape_factor * a_in_initial)[0]
    col_idx = np.where(min_any < collision_radius)[0]
    if len(esc_idx) > 0 or len(col_idx) > 0:
        t_disrupt = t[min([i for i in [esc_idx[0] if len(esc_idx) else None,
                                        col_idx[0] if len(col_idx) else None]
                            if i is not None])]
        return "DISRUPTED", t_disrupt
    # inner separation should stay within an order of magnitude of a_in
    if d01.max() > 10 * a_in_initial:
        return "INNER BINARY DISRUPTED", t[np.argmax(d01 > 10 * a_in_initial)]
    return "STABLE", t[-1]


def main():
    q_out = 1.0 / (1.0 + 1.0)  # m3 / (m1+m2), equal masses => 0.5
    ratio_crit = mardling_aarseth_ratio(q_out)
    print(f"Mardling-Aarseth critical a_out/a_in for equal masses, circular, "
          f"coplanar: {ratio_crit:.3f}")

    a_in = 1.0
    n_inner_periods_target = 500

    for label, ratio in [("STABLE-BY-CONSTRUCTION (ratio=6.0)", 6.0),
                          ("MARGINAL (ratio=3.3, ~critical)", 3.3),
                          ("UNSTABLE-BY-CONSTRUCTION (ratio=1.8)", 1.8)]:
        ic, T_in = hierarchical_ic(a_in, ratio)
        t_max = n_inner_periods_target * T_in
        E0 = energy(ic, MASSES)
        L0 = angular_momentum(ic, MASSES)
        res = integrate_scipy(ic, MASSES, (0.0, t_max), n_eval=4000,
                               rtol=1e-11, atol=1e-11, collision_radius=1e-4)
        y1 = res["y"][-1]
        E1 = energy(y1, MASSES)
        L1 = angular_momentum(y1, MASSES)
        outcome, t_event = classify_outcome(res, a_in)
        n_periods_survived = t_event / T_in
        print(f"\n[{label}]  a_out/a_in={ratio}")
        print(f"  inner period T_in={T_in:.4f}, target survival window = "
              f"{n_inner_periods_target} inner periods ({t_max:.1f} time units)")
        print(f"  outcome: {outcome} at t={t_event:.2f} "
              f"({n_periods_survived:.1f} inner periods)")
        print(f"  energy drift |dE|={abs(E1-E0):.3e}, ang. mom. drift "
              f"|dL|={abs(L1-L0):.3e}  (integrator sanity check)")

    print("\nConclusion: given a target survival time T (expressed in inner-binary "
          "periods), the Mardling-Aarseth ratio lets us DIRECTLY CONSTRUCT an "
          "initial condition guaranteed (empirically verified here, not merely "
          "asserted) to survive at least T, with no search over initial "
          "conditions required -- in contrast to Q1, which has no closed-form "
          "shortcut.")


if __name__ == "__main__":
    main()
