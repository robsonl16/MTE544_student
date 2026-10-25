"""
Report plots for P / PID runs. For each run directory this makes four figures:
    {e-t, edot-t}, {x-t, y-t, th-t}, {x-y} and {e-edot}
(linear and angular errors), and prints step-response metrics for tuning.

Each run directory must contain linear.csv and angular.csv as written by
utilities.Logger (columns: e, e_dot, e_int, stamp[ns]), and optionally
robot_pose.csv (columns: x, y, theta, stamp[ns]).

Examples:
    python3 plot_pid.py                                  # plots ./data
    python3 plot_pid.py --runs data_P data_PID --labels P PID --goal 1 1
    python3 plot_pid.py --runs data_P --labels P --save figures   # figures/P_errors.png, ...
    python3 plot_pid.py --runs data_sigmoid --ref sigmoid          # adds path-tracking overshoot
    python3 plot_pid.py --ref parabola                   # overlay the planned trajectory
    python3 plot_pid.py --goal -2 -2                     # overlay a point-planner goal
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from planner import planner, TRAJECTORY_PLANNER


# (color, marker, linestyle) per series so each is distinguishable without color
SERIES_STYLE = {
    "linear":  ("#2a78d6", "o", "-"),
    "angular": ("#eb6834", "^", "--"),
    "x":       ("#2a78d6", "o", "-"),
    "y":       ("#eb6834", "^", "--"),
    "theta":   ("#1baf7a", "s", ":"),
    "path":    ("#2a78d6", None, "-"),
}

CONTROLLERS = ["linear", "angular"]
UNITS = {"linear": "m", "angular": "rad"}

SETTLING_BAND = 0.02    # settle when |e - e_ss| stays within 2% of the step ...
SETTLING_FLOOR = 0.01   # ... or this absolute tolerance, whichever is larger
FINAL_WINDOW = 0.05     # fraction of samples at the end averaged for e_ss


REF_TRAJECTORIES = {"parabola": 0, "sigmoid": 1}


def load_log(path):
    """Return (t [s], col0, col1, col2) from a Logger csv with 3 values + stamp."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3), ndmin=2)
    t = (data[:, 3] - data[0, 3]) / 1e9
    return t, data[:, 0], data[:, 1], data[:, 2]


def reference_path(ref):
    return np.array(planner(TRAJECTORY_PLANNER).trajectory_planner(REF_TRAJECTORIES[ref]))


def path_error(x, y, points):
    """
    Signed perpendicular distance from each pose to the planned path
    (+ = left of the direction of travel, - = right). Zero means on the path,
    so crossing zero is overshoot of the reference.
    """
    a, b = points[:-1], points[1:]
    seg = b - a
    seg_len2 = np.sum(seg ** 2, axis=1)
    out = np.empty(len(x))
    for i, p in enumerate(np.column_stack((x, y))):
        # project onto every segment, keep the closest
        u = np.clip(np.sum((p - a) * seg, axis=1) / seg_len2, 0.0, 1.0)
        closest = a + u[:, None] * seg
        d = np.hypot(*(p - closest).T)
        k = np.argmin(d)
        side = np.sign(seg[k, 0] * (p[1] - a[k, 1]) - seg[k, 1] * (p[0] - a[k, 0]))
        out[i] = side * d[k]
    return out


def step_metrics(t, e):
    """
    Standard 2nd-order step-response metrics computed on the error signal.

    The "step" starts where the error is largest (|e - e_ss| max): t=0 for a
    point goal, or later when a trajectory's look-ahead heading jumps.
        %OS = 100 * |e_max past e_ss| / |e_peak - e_ss|
        zeta = -ln(%OS/100) / sqrt(pi^2 + ln^2(%OS/100))
        wn   = pi / (Tp * sqrt(1 - zeta^2))
        Ts   = 4 / (zeta * wn)                        (2% criterion)
    e_ss is the mean of the last FINAL_WINDOW of samples. In trajectory mode the
    run stops when the look-ahead point reaches the last waypoint (~3 waypoints
    early), so a non-zero final linear error is expected there.
    """
    n_final = max(1, int(FINAL_WINDOW * len(e)))
    e_ss = np.mean(e[-n_final:])

    k0 = np.argmax(np.abs(e - e_ss))
    step = e[k0] - e_ss

    # overshoot: furthest the error crosses past e_ss after the step
    past = -np.sign(step) * (e[k0:] - e_ss)
    k_os = k0 + np.argmax(past)
    os_frac = max(0.0, past.max()) / abs(step) if step != 0 else 0.0
    if k_os >= len(e) - n_final:
        os_frac = 0.0   # "overshoot" inside the averaging window is just a still-moving tail

    if os_frac > 0:
        tp = t[k_os] - t[k0]
        zeta = -np.log(os_frac) / np.sqrt(np.pi ** 2 + np.log(os_frac) ** 2)
        wn = np.pi / (tp * np.sqrt(1 - zeta ** 2))
        ts_pred = 4 / (zeta * wn)
    else:
        tp = zeta = wn = ts_pred = float("nan")   # no overshoot: overdamped / too slow

    # measured settling time: last time outside the 2% band around e_ss
    band = max(SETTLING_BAND * abs(step), SETTLING_FLOOR)
    outside = k0 + np.where(np.abs(e[k0:] - e_ss) > band)[0]
    if len(outside) == 0:
        settling = 0.0
    elif outside[-1] == len(e) - 1:
        settling = float("nan")   # never settled
    else:
        settling = t[outside[-1] + 1] - t[k0]

    dt = np.diff(t, prepend=t[0])
    return {
        "e_peak": e[k0],
        "e_final": e_ss,
        "OS_%": os_frac * 100,
        "Tp_s": tp,
        "zeta": zeta,
        "wn": wn,
        "Ts_meas": settling,
        "Ts_pred": ts_pred,
        "IAE": np.sum(np.abs(e) * dt),
    }


def print_metrics(results):
    cols = ["e_peak", "e_final", "OS_%", "Tp_s", "zeta", "wn", "Ts_meas", "Ts_pred", "IAE"]
    print(f"\n{'run':<16}{'ctrl':<9}" + "".join(f"{c:>10}" for c in cols))
    for label, ctrl, m in results:
        print(f"{label:<16}{ctrl:<9}" + "".join(f"{m[c]:>10.3f}" for c in cols))
    print("\npath = signed perpendicular distance [m] from the planned --ref trajectory (+ left, - right)")
    print(f"OS/Tp/Ts measured from the error peak; settling band max({SETTLING_BAND*100:.0f}% of step, "
          f"{SETTLING_FLOOR}); e_final = mean of last {FINAL_WINDOW*100:.0f}% of samples; nan = n/a or never settled\n")


def style(name, n):
    """Line kwargs for a series: its own color, marker and line style, markers spaced out."""
    color, marker, ls = SERIES_STYLE[name]
    return dict(color=color, marker=marker, linestyle=ls, linewidth=1.5,
                markersize=6, markevery=max(1, n // 15))


def finish(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.legend()


def save_fig(fig, save_dir, label, name):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{label}_{name}.png")
        fig.savefig(path, dpi=150)
        print(f"saved figure to {path}")


def plot_run(run_dir, label, ref=None, goal=None, save_dir=None):
    """The four report figures for one run: {e-t, edot-t}, {x-t, y-t, th-t}, {x-y}, {e-edot}."""
    errors = {}
    for ctrl in CONTROLLERS:
        path = os.path.join(run_dir, f"{ctrl}.csv")
        if os.path.isfile(path):
            errors[ctrl] = load_log(path)
        else:
            print(f"skipping missing file {path}")

    # {e-t, edot-t}
    fig, (ax_e, ax_ed) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(f"{label}: errors vs time")
    for ctrl, (t, e, e_dot, _) in errors.items():
        ax_e.plot(t, e, label=f"{ctrl} e [{UNITS[ctrl]}]", **style(ctrl, len(t)))
        ax_ed.plot(t, e_dot, label=f"{ctrl} ė [{UNITS[ctrl]}/s]", **style(ctrl, len(t)))
    finish(ax_e, "error e vs time", "time [s]", "e [m] / [rad]")
    finish(ax_ed, "error rate ė vs time", "time [s]", "ė [m/s] / [rad/s]")
    save_fig(fig, save_dir, label, "errors")

    # {e-edot}: one panel per controller since their scales differ
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(f"{label}: e vs ė  (o start, X end)")
    for ax, (ctrl, (t, e, e_dot, _)) in zip(axes, errors.items()):
        u = UNITS[ctrl]
        st = style(ctrl, len(t))
        ax.plot(e, e_dot, label=f"{ctrl}", **st)
        ax.plot(e[0], e_dot[0], "o", color=st["color"], markersize=9, label="start")
        ax.plot(e[-1], e_dot[-1], "X", color=st["color"], markersize=10, label="end")
        ax.axvline(0, color="0.6", linewidth=0.8)
        finish(ax, f"{ctrl}: e vs ė", f"e [{u}]", f"ė [{u}/s]")
    save_fig(fig, save_dir, label, "phase")

    path = os.path.join(run_dir, "robot_pose.csv")
    if not os.path.isfile(path):
        print(f"skipping missing file {path}")
        return
    t, x, y, theta = load_log(path)
    print(f"{label}: start ({x[0]:.3f}, {y[0]:.3f})  end ({x[-1]:.3f}, {y[-1]:.3f}) "
          f"theta_end {theta[-1]:.3f} rad  duration {t[-1]:.2f} s")

    # {x-t, y-t, th-t}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    fig.suptitle(f"{label}: robot states vs time")
    for ax, (name, v, unit) in zip(axes, [("x", x, "m"), ("y", y, "m"), ("theta", theta, "rad")]):
        ax.plot(t, v, label=f"{name} [{unit}]", **style(name, len(t)))
        finish(ax, f"{name} vs time", "time [s]", f"{name} [{unit}]")
    save_fig(fig, save_dir, label, "states")

    # {x-y}
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if ref is not None:
        points = reference_path(ref)
        ax.plot(points[:, 0], points[:, 1], "--", color="0.5", linewidth=1.5, label=f"planned {ref}")
    if goal is not None:
        ax.plot(goal[0], goal[1], "*", color="0.3", markersize=14, label="goal")
    st = style("path", len(t))
    ax.plot(x, y, label="robot path", **st)
    ax.plot(x[0], y[0], "o", color=st["color"], markersize=9, label="start")
    ax.plot(x[-1], y[-1], "X", color=st["color"], markersize=10, label="end")
    ax.set_aspect("equal", adjustable="datalim")
    finish(ax, f"{label}: robot path x vs y", "x [m]", "y [m]")
    save_fig(fig, save_dir, label, "path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot linear/angular PID error logs and robot motion.")
    parser.add_argument("--runs", nargs="+", default=["data"],
                        help="directories containing linear.csv, angular.csv, robot_pose.csv (default: data)")
    parser.add_argument("--labels", nargs="+", help="title label per run, e.g. P PID (default: directory name)")
    parser.add_argument("--save", metavar="DIR", help="save every figure as DIR/<label>_<figure>.png")
    parser.add_argument("--ref", choices=REF_TRAJECTORIES, help="overlay the planned trajectory from planner.py")
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), help="overlay a point-planner goal")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    if len(labels) != len(args.runs):
        parser.error("--labels must have one entry per --runs directory")

    results = []
    for run_dir, label in zip(args.runs, labels):
        for ctrl in CONTROLLERS:
            path = os.path.join(run_dir, f"{ctrl}.csv")
            if os.path.isfile(path):
                t, e, _, _ = load_log(path)
                results.append((label, ctrl, step_metrics(t, e)))
        pose_path = os.path.join(run_dir, "robot_pose.csv")
        if args.ref and os.path.isfile(pose_path):
            t, x, y, _ = load_log(pose_path)
            results.append((label, "path", step_metrics(t, path_error(x, y, reference_path(args.ref)))))
        plot_run(run_dir, label, args.ref, args.goal, args.save)
    print_metrics(results)
    plt.show()
