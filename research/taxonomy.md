# Taxonomy: what "stable" means for a three-body orbit

The project brief poses two categories — orbits stable "ad infinitum" and orbits
stable over "a finite time frame." These are good pedagogical anchors, but as
posed they conflate two physically distinct properties: **periodicity** (does
the trajectory exactly retrace itself?) and **stability** (does a small
perturbation stay small?). The paper should keep the two-bucket narrative as
its spine but define these four classes underneath it:

1. **Periodic.** The trajectory in the full 12-dimensional phase space (or
   6-dimensional shape space, up to rotation) exactly closes after some period
   `T`: `X(T) = X(0)` (or `X(T) = R(θ)·X(0)` for some fixed rotation, if the
   orbit only closes in a frame that has itself rotated by θ — common in this
   literature). Infinitely many periodic three-body families are known
   (figure-eight, Lagrange, Euler, Broucke–Hénon, the ~2000+ families of
   Li & Liao). Periodicity is a property of the *orbit*, checkable to
   arbitrary numerical precision by direct integration.

2. **Linearly stable periodic.** Periodic, *and* the monodromy matrix
   (the linearization of the return map around the periodic orbit) has all
   eigenvalues on the unit circle. This is the honest referent for "stable ad
   infinitum" in the strict dynamical-systems sense: a small perturbation to
   the initial condition stays bounded (does not grow exponentially) for all
   time, at least at linear order. It is *rare* — most catalogued periodic
   three-body orbits are periodic but linearly **unstable**: exact solutions
   of the exact initial condition, but any perturbation (including
   floating-point roundoff) grows and eventually destroys the periodicity.
   The figure-eight orbit is well known partly *because* it is one of the few
   that is (weakly) linearly stable.

3. **Hierarchically / KAM-stable.** Not periodic at all, but bounded for all
   time by a *structural* argument rather than an exact closure condition: a
   tight inner binary orbited by a distant third body, with the separation
   ratio large enough (Hill stability; the empirical Mardling–Aarseth 2001
   criterion used in `hierarchical_stability.py`) that the outer body cannot
   perturb the inner binary into disruption. This is the astronomically
   realistic sense of "stable forever" — most real triple star systems that
   persist do so this way, not by sitting on an exactly periodic orbit.

4. **Metastable / long-lived chaotic.** No collision or escape for at least
   `N` dynamical (crossing) times, but not periodic and not protected by a
   hierarchical-stability argument — just chaotic transient behavior that
   happens not to have resolved into a collision or an ejection yet within the
   window of interest. This is the *generic* outcome for a randomly chosen,
   non-hierarchical, bound three-body configuration, and it is the right home
   for the project's category 2 ("stable over a finite timeframe"):
   `escape_statistics.py`'s survival-fraction-vs-`T` curve quantifies exactly
   this.

## Mapping onto the project's two research questions

- **Q1** ("without a known seed, can multiple solutions of type 1 be found?")
  is really asking about class 1 (periodic — searchable by symmetry + return
  proximity, no seed required) with class 2 (linear stability) as a secondary,
  separately-checkable property once a periodic orbit is in hand. See
  `notes.md` for what was actually found.

- **Q2** ("given a timeframe, can a solution be determined?") is really
  asking about classes 3 and 4: class 3 has a closed-form constructive answer
  (Mardling–Aarseth), class 4 has a statistical answer (escape-time survival
  curve). Neither requires finding an exactly periodic orbit, which is why Q2
  is tractable by direct construction where Q1 is not.
