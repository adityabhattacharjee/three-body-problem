# Periodic Orbits and Finite-Time Stability in the Gravitational Three-Body Problem

Computational investigation into two questions about the Newtonian three-body problem:

1. **Without a known periodic orbit as a seed**, can a numerical search discover a genuinely new one from scratch?
2. **Given a target survival time**, can an initial condition be constructed or found that is guaranteed, or at least likely, to stay bound for at least that long?

Full writeup: [`paper/main.pdf`](paper/main.pdf) (technical) and [`paper/companion.pdf`](paper/companion.pdf) (plain-language, no equations).

## Videos

- [The Figure-Eight Three-Body Orbit](https://youtu.be/uMNLTKFUXI4)
- [Hierarchical Three-Body Stability: Stable vs. Disrupted](https://youtu.be/VID2Qx_FtoE)
- [Three-Body Survival Statistics](https://youtu.be/Zm_S9v5XTLs)

## Repository layout

```
research/           Numerical experiments (see taxonomy.md and notes.md for the full writeup of methods and results)
  integrator.py            Validated dual-integrator harness (REBOUND/IAS15 + scipy/DOP853)
  validate_integrator.py   Cross-validation against the figure-eight orbit
  seedless_scan.py         Q1: collinear velocity-angle scan
  free_fall_scan.py        Q1: isosceles free-fall ("brake orbit") scan
  broucke_henon_scan.py    Q1: perpendicular-syzygy scan
  shape_sphere_scan.py     Q1: 2-D shape-space scan
  hierarchical_stability.py Q2 route 1: Mardling-Aarseth hierarchical construction
  escape_statistics.py     Q2 route 2: random-scattering survival curve

data/               Saved results (.npz) referenced by the paper and the figure-generation script

manim/              Manim scenes for the three videos above (precompute.py regenerates each
                    scene's trajectory data from research/; animation.py renders the scene)

paper/              LaTeX sources for both documents, references.bib, and the figure-generation script
```

## Reproducing

All numerical results use `G=1` and equal unit masses. Random-seed and tolerance choices are
recorded in `research/notes.md`. Requires `numpy`, `scipy`, `rebound` (see notes.md for a
packaging caveat on non-AVX-512 hardware) and, for the videos, `manim`.

Every literature citation in the paper (Moore 1993, Chenciner-Montgomery 2000,
Šuvakov-Dmitrašinović 2013, Li-Liao 2017/2018, Mardling-Aarseth 2001, Stone-Leigh 2019,
Breen et al. 2020) was verified against the original source rather than recalled from memory --
see `research/notes.md` for the verification notes and one formula correction it caught.
