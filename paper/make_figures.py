"""
Generate the static PDF figures used in the LaTeX paper, from the project's
own saved/reproducible numerical results. Run from three_body/paper/.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from integrator import integrate_scipy
from hierarchical_stability import hierarchical_ic, mardling_aarseth_ratio

FIGDIR = os.path.dirname(__file__)
DATADIR = os.path.join(os.path.dirname(__file__), "..", "data")
MASSES = np.array([1.0, 1.0, 1.0])

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLORS = ["#d62728", "#1f77b4", "#2ca02c"]  # body 1, 2, 3


def fig_figure_eight():
    d = np.load(os.path.join(DATADIR, "figure8_validated.npz"))
    ic, T = d["ic"], float(d["T"])
    res = integrate_scipy(ic, MASSES, (0.0, T), n_eval=2000, rtol=1e-12, atol=1e-12)
    y = res["y"]
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(3):
        ax.plot(y[:, 2 * i], y[:, 2 * i + 1], color=COLORS[i], lw=1.6,
                label=f"body {i+1}")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Validated figure-eight orbit (one period)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_figure_eight.pdf"))
    plt.close(fig)


def fig_hierarchical():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, ratio, tmax, label in [
        (axes[0], 6.0, 60.0, "stable (a_out/a_in = 6.0)"),
        (axes[1], 1.8, 90.0, "disrupted (a_out/a_in = 1.8)"),
    ]:
        ic, T_in = hierarchical_ic(1.0, ratio)
        res = integrate_scipy(ic, MASSES, (0.0, tmax), n_eval=3000,
                               rtol=1e-11, atol=1e-11, collision_radius=1e-4)
        y, t = res["y"], res["t"]
        pos = y[:, 0:6].reshape(-1, 3, 2)
        d01 = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)  # inner separation
        d_out = np.linalg.norm(pos[:, 2] - (pos[:, 0] + pos[:, 1]) / 2, axis=1)
        ax.plot(t / T_in, d01, color="#1f77b4", lw=1.3, label="inner separation")
        ax.plot(t / T_in, d_out, color="#2ca02c", lw=1.3, label="outer distance")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("time (inner periods)")
    axes[0].set_ylabel("separation")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Hierarchical triple: known stability rule of thumb "
                 f"$\\approx${mardling_aarseth_ratio(0.5):.2f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_hierarchical.pdf"))
    plt.close(fig)


def fig_survival_curve():
    d = np.load(os.path.join(DATADIR, "escape_statistics.npz"))
    t_grid, curve = d["t_grid"], d["survival_curve"]
    n_trials = int(d["n_trials"])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(t_grid, curve, "o-", color="#9467bd", lw=1.6)
    ax.set_xlabel("T (time units)")
    ax.set_ylabel("S(T) = P(disruption time $\\geq$ T)")
    ax.set_title(f"Escape/collision survival curve (n={n_trials} random trials)")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_survival_curve.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_figure_eight()
    print("wrote fig_figure_eight.pdf")
    fig_hierarchical()
    print("wrote fig_hierarchical.pdf")
    fig_survival_curve()
    print("wrote fig_survival_curve.pdf")
