"""
Overlay figures for the report: P vs PID (point motion) and good vs bad PID (sigmoid).
Reuses load_log / reference_path from plot_pid.py. Run from the repo root:
    python3 report_plots/compare_plots.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plot_pid import load_log, reference_path  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "comparisons")
STYLES = [("#2a78d6", "o", "-"), ("#eb6834", "^", "--")]


def st(i, n):
    """Line style for the i-th run (colour, marker, line style), markers spaced out over n samples."""
    c, m, ls = STYLES[i]
    return dict(color=c, marker=m, linestyle=ls, linewidth=1.5, markersize=5, markevery=max(1, n // 15))


def finish(ax, title, xlabel, ylabel):
    """Rubric formatting: title, axis labels, zero line, grid, legend."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.legend()


def save(fig, name):
    """Save into report_plots/comparisons/ and free the figure."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def point_comparison(runs):
    """P vs PID overlays for the point goal (1, -1): errors, states + path, and phase portrait.
    runs: list of (label, run directory)."""
    logs = {lab: {k: load_log(os.path.join(d, f"{k}.csv")) for k in ("linear", "angular", "robot_pose")}
            for lab, d in runs}

    # {e-t, edot-t}: rows = linear/angular, cols = e / edot
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Point motion to (1, -1): P vs PID errors vs time")
    for r, (ctrl, u) in enumerate([("linear", "m"), ("angular", "rad")]):
        for i, lab in enumerate(logs):
            t, e, ed, _ = logs[lab][ctrl]
            axes[r, 0].plot(t, e, label=f"{lab}", **st(i, len(t)))
            axes[r, 1].plot(t, ed, label=f"{lab}", **st(i, len(t)))
        finish(axes[r, 0], f"{ctrl} error e", "time [s]", f"e [{u}]")
        finish(axes[r, 1], f"{ctrl} error rate ė", "time [s]", f"ė [{u}/s]")
    save(fig, "point_P_vs_PID_errors.png")

    # {x-t, y-t, th-t}: all states on one axes. Colour + marker = state, line style = controller.
    state_style = {"x": ("#2a78d6", "o"), "y": ("#eb6834", "^"), "theta": ("#1baf7a", "s")}
    ctrl_ls = ["-", "--"]
    short = [lab.split(" ")[0] for lab in logs]       # "P", "PID"
    # One figure, two panels: states vs time (left) and x-y path (right)
    fig, (ax, ax_xy) = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle("Point motion to (1, -1): P vs PID robot states and path")
    for i, lab in enumerate(logs):
        t, x, y, th = logs[lab]["robot_pose"]
        for name, v, u in (("x", x, "m"), ("y", y, "m"), ("theta", th, "rad")):
            c, m = state_style[name]
            ax.plot(t, v, color=c, marker=m, linestyle=ctrl_ls[i], linewidth=1.5, markersize=5,
                    markevery=max(1, len(t) // 15), markerfacecolor=c if i == 0 else "white",
                    label=f"{short[i]}: {name} [{u}]")
    for name, g in (("x", 1.0), ("y", -1.0)):
        ax.axhline(g, color=state_style[name][0], linestyle=":", linewidth=1.2, label=f"goal {name} = {g:g} m")
    finish(ax, "states vs time: P (solid, filled) vs PID (dashed, hollow)", "time [s]", "x, y [m]  /  θ [rad]")
    ax.legend(ncol=3, fontsize=9)

    # right: x-y path
    ax = ax_xy
    ax.plot(1, -1, "*", color="0.3", markersize=14, label="goal (1, -1)")
    for i, lab in enumerate(logs):
        t, x, y, _ = logs[lab]["robot_pose"]
        ax.plot(x, y, label=f"{short[i]} path", **st(i, len(t)))
    ax.plot(0, 0, "o", color="0.3", markersize=8, label="start")
    ax.set_aspect("equal", adjustable="datalim")
    finish(ax, "robot path x vs y", "x [m]", "y [m]")
    save(fig, "point_P_vs_PID_states_path.png")

    # {e-edot}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Point motion: P vs PID phase portrait e vs ė")
    for ax, (ctrl, u) in zip(axes, [("linear", "m"), ("angular", "rad")]):
        for i, lab in enumerate(logs):
            t, e, ed, _ = logs[lab][ctrl]
            ax.plot(e, ed, label=lab, **st(i, len(t)))
        ax.axvline(0, color="0.6", linewidth=0.8)
        finish(ax, f"{ctrl}: e vs ė", f"e [{u}]", f"ė [{u}/s]")
    save(fig, "point_P_vs_PID_phase.png")


def sigmoid_comparison(runs):
    """x-y paths of several sigmoid runs over the planned sigmoid (tuning comparison)."""
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    pts = reference_path("sigmoid")
    ax.plot(pts[:, 0], pts[:, 1], ":", color="0.3", linewidth=2, label="planned sigmoid")
    for i, (lab, d) in enumerate(runs):
        t, x, y, _ = load_log(os.path.join(d, "robot_pose.csv"))
        ax.plot(x, y, label=lab, **st(i, len(t)))
    ax.set_aspect("equal", adjustable="datalim")
    finish(ax, "Sigmoid trajectory: tuned vs poorly tuned PID", "x [m]", "y [m]")
    save(fig, "sigmoid_good_vs_bad_path.png")


if __name__ == "__main__":
    point_comparison([("P (klp=0.1)", "data_P_1"),
                      ("PID (klp=0.1, klv=0.15, kli=0.15 / kap=0.5, kav=0.15, kai=0.05)", "data_PID_6")])
    sigmoid_comparison([("tuned PID (klp=0.025, kap=1.0, kav=0.125)", "data_traj_6"),
                        ("poor PID (klp=0.05, kap=0.25, kav=0.15)", "data_traj_2")])
