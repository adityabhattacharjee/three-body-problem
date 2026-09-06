# Research notes: three-body periodic orbits and finite-time stability

All numbers below are from this project's own code (`three_body/research/*.py`),
run on 2026-09-06, seed recorded where randomness is used. Anything attributed
to the literature is marked as such and still needs a `WebSearch` citation pass
before it goes in the paper's bibliography (see `taxonomy.md` note and
project instructions: never write a `.bib` entry from memory).

## 0. Integrator validation (`integrator.py`, `validate_integrator.py`)

Two independent codes: REBOUND's IAS15 (Rein & Spiegel 2015; installed as
v4.6.0 -- the v5.1.1 PyPI wheel's `whfast512` symbol fails to load on this
non-AVX-512 Intel CPU, a packaging issue, not a physics one) and
`scipy.integrate.solve_ivp` with DOP853, both used with `G=1`, equal unit
masses.

Starting from the commonly-cited figure-eight initial condition (Moore 1993 /
Chenciner-Montgomery 2000 normalization) as an unverified guess, Newton
refinement (`fsolve` on the one-period return residual) corrected it by
~2e-7 in position/velocity and ~2e-6 in period, converging to:

- refined period T* = 6.32591... (literature guess was 6.32591398; agrees to
  the digits given)
- one-period return residual: scipy 2.4e-12, REBOUND 4.2e-12
- energy drift over one period: scipy 6.4e-13, REBOUND 2.2e-16
- angular momentum drift: scipy 7.5e-15, REBOUND 1.6e-17
- max pointwise disagreement between the two integrators over the full period:
  5.1e-12

**Conclusion: the harness is validated to ~1e-11-1e-12 by two independent
integrators, and the recalled figure-eight IC was correct to ~1e-7 (now
refined further, this project's own number).**

## 1. Q1 -- seedless discovery of periodic orbits

Four independent, seed-free constructions were tried, in order:

1. **`seedless_scan.py`** -- collinear positions r=(-1,0),(1,0),(0,0),
   zero-momentum velocities v1=v2 at angle alpha, energy normalized to E=-1,
   1-D scan over alpha in [0,pi] (500 points). 21 local minima of a
   return-proximity functional were found; none of the refined candidates
   converged (residuals 0.2-3.5 after two-stage Nelder-Mead + least-squares
   refinement) -- and one raw trajectory triggered a genuine near-singularity
   ("step size less than spacing between numbers"), later traced to collision.

2. **`free_fall_scan.py`** -- isosceles free-fall (v=0 exactly) ansatz,
   shape parameter h (height of the third body). Direct check: EVERY tested
   h in [0.15, 3.0] (six spot checks plus a 40-point scan) ends in genuine
   triple/binary collision within t~1-2 time units (verified by removing the
   collision guard and confirming the integrator fails exactly the way a
   true finite-time singularity fails).

3. **`broucke_henon_scan.py`** -- fully collinear start (all 3 bodies on a
   line) with velocities perpendicular to the line (the naive reading of the
   "Broucke-Henon" construction). Angular momentum is L=0 for this whole
   family (general fact: mirror-symmetric collinear-start + zero net
   momentum forces L=0, independent of the perpendicular speed chosen).
   Direct check at v=0.1: the two "syzygy" events found straddle t=1.005,
   which is exactly the time a hard collision cutoff independently reports.
   Every v in [0.05, 1.2] tested behaves the same way.

4. **`shape_sphere_scan.py`** -- the fix identified after (1)-(3): move the
   third body OFF axis (r3=(0,h)), matching the validated figure-eight's own
   off-axis (non-collinear) geometry, and scan the genuine 2-D shape space
   (alpha, h), 24x24=576 points, E=-1, with a hard collision-radius cutoff
   AND a post-hoc "minimum separation > 0.05 throughout" filter applied
   before any candidate is considered for refinement. 266/576 grid points
   survived the filter. The 12 lowest-residual survivors were refined
   (Nelder-Mead polish + least-squares, allowing closure up to a rigid
   rotation): **0 converged to residual < 1e-7** (best achieved: 0.93, at
   alpha~0, h~1.5).

**Empirical answer to Q1, as actually obtained in this project (not
asserted): a from-scratch, symmetry-reduced 2-D shape-space scan is
seedless by construction (it uses no known orbit) but did NOT converge to a
new periodic orbit within the compute budget bounded for this project (one
grid sweep + refinement of the top 12 candidates, per the
plan agreed after run (3) above). Runs (1)-(3) additionally establish, with
concrete collision times, *why* naive collinear/free-fall/L=0 ansätze are
much harder than they look: they sit on or near the triple-collision
manifold, which has no centrifugal barrier to keep trajectories away from
it. The validated figure-eight orbit (Section 0) is the existence proof that
off-axis, L=0 periodic orbits exist and are numerically reachable -- but
reaching a *new* one from scratch, as Suvakov & Dmitrasinovic (2013) and
Li & Liao (2017-) did, evidently requires either (a) much larger sample
counts than were run here, (b) the topological/braid pre-classification
those papers used to filter candidates before refinement (not attempted
here), or (c) both.**

This negative-with-mechanism result is itself the honest, citable content for
the paper's Q1 section, alongside the literature claims (Suvakov &
Dmitrasinovic PRL 2013: 13 new families from a scan of this same general
type at much larger scale; Li & Liao's later ~700-1900 family catalogs) which
must be verified by `WebSearch` before citing, not written from memory.

## 2. Q2 -- given a timeframe T, can a solution be constructed?

### Route 1: hierarchical construction (`hierarchical_stability.py`)

Mardling & Aarseth (2001) empirical stability ratio for a coplanar, circular,
equal-mass hierarchical triple: a_out/a_in >~ 3.29 (computed from their
formula, not quoted from memory of their number -- the exponents and
prefactor in the formula itself should still be checked against the original
paper before the number goes in the final bibliography).

Three constructed systems, each integrated for 500 inner-binary periods
(2221 time units) with the validated scipy/DOP853 integrator:

| ratio a_out/a_in | outcome                | survival time         | energy drift |
|---|---|---|---|
| 6.0 (comfortably above threshold) | STABLE | full 500 periods | 7.5e-10 |
| 3.3 (~at threshold)               | STABLE | full 500 periods | 4.2e-9 |
| 1.8 (below threshold)             | DISRUPTED | 18.1 periods | 2.2e-7 (at disruption) |

**This is a clean positive, constructive answer to Q2 for the hierarchical
sub-family: given a target T (in inner-binary periods), pick a_out/a_in via
the closed-form criterion and the construction is verified stable to that T
with no search.**

### Route 2: statistical / generic case (`escape_statistics.py`)

Random equal-mass, zero-momentum, energy-normalized (E=-1) triangles with
unconstrained angular momentum (this is what distinguishes this ensemble
from the Q1 scans and is why it does not sit on the collision manifold in
the same way): 200 trials, seed 20260906, t_max=200 time units,
escape radius 30, collision radius 1e-3.

Outcome breakdown: 91 collisions (45.5%), 82 escapes (41.0%),
27 survived the full window (13.5%).

Survival curve S(T) = P(disruption time >= T):

| T | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| S(T) | 0.965 | 0.945 | 0.910 | 0.865 | 0.770 | 0.540 | 0.295 | 0.180 | 0.135 |

**Interpretation:** the empirical survival curve S(T) answers Q2 directly for
the generic (non-hierarchical) case: it is roughly a decaying-exponential/
power-law-like falloff (worth fitting explicitly in the paper), and for any T
up to 200 a concrete surviving trial can be pulled from this run as a
determined, verified example -- obtained by rejection sampling rather than
closed-form construction (contrast with Route 1's guarantee, which needs no
sampling at all but only covers the hierarchical sub-family). Note the
qualitative contrast with Route 1: an UNCONSTRAINED-angular-momentum generic
triple has no structural protection, so "survives to T" is a *probabilistic*
property of the ensemble, degrading smoothly with T, rather than the
sharp stable/unstable dichotomy Route 1's Mardling-Aarseth threshold produces.

## Honesty notes for the paper

- Do not write "we found N periodic families" for Q1 -- the honest claim is
  "N candidate configurations were surveyed via a seedless 2-D shape-space
  scan; 0 converged to residual < 1e-7 within the bounded search budget,"
  with the collision-manifold mechanism explained via runs (1)-(3).
- The figure-eight numbers (Section 0) and the hierarchical-stability numbers
  (Section 2, Route 1) ARE this project's own verified positive results, fit
  to be the paper's central computed examples.
- Every literature number (Suvakov-Dmitrasinovic's 13, Li-Liao's family
  counts, Mardling-Aarseth's formula, Chenciner-Montgomery's variational
  existence proof) needs a `WebSearch` verification pass before citation --
  none has been done yet as of this note.

## Citation verification (completed, see `../paper/references.bib`)

A dedicated verification pass (live search, not memory) confirmed all core
citations and caught two errors in initial recollection:

- **Suvakov & Dmitrasinovic (2013)**: the abstract itself reports 15 initial
  conditions, of which **13 correspond to distinct orbits** (not "13 new
  orbits" full stop -- the paper's own wording is more precise). A caveat:
  arXiv:1312.6796 (Li & Liao) later showed at least 7 of the 15 lose
  periodicity under high-precision (CNS) re-integration over long times --
  worth a footnote if citing "13" as a settled count.
- **Li & Liao's family counts**: the actual numbers are 695 total / >600 new
  families (equal-mass, 2017, Sci. China Phys. Mech. Astron. 60, 129511) and
  1349 total / 1223 new families (unequal-mass, 2018, PASJ 70, 64) -- not the
  "~600 and ~1900" originally recalled. A separate, much later (2025) paper
  reports 10,059 *3-D* orbits and must not be conflated with these two.
- **Mardling & Aarseth (2001) formula error, corrected in
  `hierarchical_stability.py`**: the correct published form has an
  additional `(1-e_out)^-1` factor:
  `a_out/a_in > 2.8*[(1+q_out)(1+e_out)/sqrt(1-e_out)]^0.4 / (1-e_out) * (1-0.3*i/180)`.
  This project's numeric results are unaffected because all runs used
  e_out=0 (circular outer orbit), where the missing factor evaluates to 1 --
  but the code and any general-e_out claims in the paper must use the
  corrected form.
- Moore (1993), Chenciner-Montgomery (2000), Stone & Leigh (2019), and
  Breen et al. (2020) were all confirmed as originally described, with no
  corrections needed. Full BibTeX in `../paper/references.bib`.
