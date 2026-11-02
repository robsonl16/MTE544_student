"""
Report plots and metrics for P / PID runs. For each run directory this makes:
    <label>_errors.png       {e-t, edot-t}      linear and angular error and error rate
    <label>_phase.png        {e-edot}           phase portrait, one panel per loop
    <label>_states_path.png  {x-t, y-t, th-t} and {x-y} side by side
    <label>_velocities.png   requested vs saturated velocity (if velocities.csv exists)
and prints two tables:
    - step-response metrics per run (overshoot, zeta, wn, settling, IAE; used for tuning)
    - REPORT TABLE (point runs only): agility / accuracy / overshoot vs the setpoint e = 0

Each run directory must contain linear.csv and angular.csv as written by
utilities.Logger (columns: e, e_dot, e_int, stamp[ns]), and optionally
robot_pose.csv (columns: x, y, theta, stamp[ns]).

Examples:
    python3 plot_pid.py --runs data_P_1 data_PID_6 --labels P PID --goal 1 -1 --no-show   # report table only
    python3 plot_pid.py --runs data_P_1 --labels P --goal 1 -1 --save figures            # save PNGs
    python3 plot_pid.py --runs data_traj_6 --ref sigmoid      # overlay planned sigmoid + path overshoot
    python3 plot_pid.py --runs data_traj_8 --ref parabola     # overlay planned parabola
"""

import argparse
import csv
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
    """Planned waypoints (N x 2 array) for 'parabola' or 'sigmoid', taken from planner.py."""
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
        %OS = 100 * |e_max past 0 (setpoint)| / |e_peak|
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

    # overshoot: furthest the error crosses past the SETPOINT (e = 0) after the step.
    # Measured against 0, not e_ss: the run stops once inside the goal tolerance,
    # so the last samples are not a steady state and averaging them would absorb
    # (cancel) the overshoot itself.
    past = -np.sign(e[k0]) * e[k0:]
    k_os = k0 + np.argmax(past)
    os_frac = max(0.0, past.max()) / abs(e[k0]) if e[k0] != 0 else 0.0

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


def report_metrics(t, e):
    """
    Point-controller metrics for the report table (rubric: agility, accuracy, overshoot).
    The error should go from e[0] to the setpoint 0, so everything is measured against 0:
      normalised response r(t) = 1 - e(t)/e(0)   (0 at start, 1 at the setpoint)
      Agility  : rise time  = t(r >= 0.9) - t(r >= 0.1)          (10-90% rise time)
                 t90        = t(r >= 0.9)
                 Ts5 / Ts2  = first time after which |e| stays within 5% / 2% of |e(0)|
                              (standard settling-time criteria; nan = never settled
                               before the run stopped)
      Accuracy : e_final    = last logged error (where the robot stopped; approximates
                              steady-state error, since the run ends at the goal tolerance)
                 IAE        = integral of |e| dt  (integral performance index)
      Overshoot: %OS        = 100 * (furthest e goes past 0) / |e(0)|
    """
    e0 = e[0]
    nan = float("nan")
    if e0 == 0:
        return dict(rise=nan, t90=nan, Ts5=nan, Ts2=nan, t_end=t[-1], e_final=e[-1], IAE=nan, OS=nan)
    r = 1 - e / e0

    def first_reach(level):
        idx = np.where(r >= level)[0]
        return t[idx[0]] if len(idx) else nan

    def settle(frac):
        outside = np.where(np.abs(e) > frac * abs(e0))[0]
        if len(outside) == 0:
            return 0.0
        if outside[-1] == len(e) - 1:
            return nan
        return t[outside[-1] + 1]

    t10, t90 = first_reach(0.1), first_reach(0.9)
    past = -np.sign(e0) * e
    dt = np.diff(t, prepend=t[0])
    return dict(rise=t90 - t10, t90=t90, Ts5=settle(0.05), Ts2=settle(0.02), t_end=t[-1],
                e_final=e[-1], IAE=np.sum(np.abs(e) * dt), OS=100 * max(0.0, past.max()) / abs(e0))


def position_overshoot(x, y, goal):
    """
    The linear error is a Euclidean distance (always >= 0), so it cannot show overshoot
    by changing sign. Instead project (goal - pose) onto the start->goal direction:
    positive = still short of the goal, negative = drove past it.
    Returns (%OS of the start->goal distance, closest signed approach [m]).
    """
    start, g = np.array([x[0], y[0]]), np.array(goal, dtype=float)
    d0 = np.linalg.norm(g - start)
    along = (g[0] - x) * (g - start)[0] / d0 + (g[1] - y) * (g - start)[1] / d0
    return 100 * max(0.0, -along.min()) / d0, along.min()


def print_report_table(rows, pos_rows):
    """rows: (label, ctrl, report_metrics dict); pos_rows: (label, %OS, closest approach)."""
    def sec(v):
        return f"{'nan':>9}" if np.isnan(v) else f"{v:>8.2f}s"

    print("\nREPORT TABLE - point controller (all metrics measured against the setpoint e = 0)")
    print(f"{'':<10}{'':<10}|{' Agility ':-^45}|{' Accuracy ':-^26}|{' Overshoot ':-^11}")
    print(f"{'run':<10}{'error':<10}|{'rise10-90':>9}{'t90':>9}{'Ts(5%)':>9}{'Ts(2%)':>9}{'t_end':>9}"
          f"|{'e_final':>13}{'IAE':>13}|{'%OS':>10} ")
    print("-" * 104)
    for label, ctrl, m in rows:
        e_final = f"{m['e_final']:+.4f} {UNITS[ctrl]}"
        print(f"{label:<10}{ctrl:<10}|{sec(m['rise'])}{sec(m['t90'])}{sec(m['Ts5'])}{sec(m['Ts2'])}{sec(m['t_end'])}"
              f"|{e_final:>13}{m['IAE']:>13.3f}|{m['OS']:>9.2f}% ")
    for label, os_pct, closest in pos_rows:
        closest_txt = f"closest {closest:+.4f} m"
        print(f"{label:<10}{'position':<10}|{'':45}|{closest_txt:>26}|{os_pct:>9.2f}% ")
    print("Ts: nan = |e| never stayed inside the band before the run stopped.  IAE units: m*s (linear), rad*s (angular).")
    print("position: error projected on the start->goal line; closest > 0 = stopped short of the goal, < 0 = drove past it.\n")


def print_metrics(results):
    """Print the step_metrics() table (one row per run and loop)."""
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
    """Apply the rubric formatting to an axes: title, axis labels, zero line, grid, legend."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.legend()


def save_fig(fig, save_dir, label, name):
    """Save fig as <save_dir>/<label>_<name>.png (no-op when --save is not given)."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{label}_{name}.png")
        fig.savefig(path, dpi=150)
        print(f"saved figure to {path}")


def plot_velocities(filename, label, save_dir=None):
    """Plot requested vs commanded (clipped) linear/angular velocity and mark samples hit by saturation."""
    with open(filename, newline="") as stream:
        rows = [{key.strip(): value.strip() for key, value in row.items()
                 if key and key.strip()} for row in csv.DictReader(stream, skipinitialspace=True)]
    if not rows:
        raise ValueError(f"{filename}: no velocity samples")
    # Subtract integer timestamps before converting to seconds.
    stamps = [int(row["stamp"]) for row in rows]
    time = np.array([stamp - stamps[0] for stamp in stamps]) / 1e9
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             constrained_layout=True)
    fig.suptitle(f"{label}: controller velocity saturation")
    for ax, name, unit in zip(axes, ("linear", "angular"), ("m/s", "rad/s")):
        def column(suffix):
            return np.array([float(row[f"{name}_{suffix}"]) for row in rows])

        requested, command = [column(key) for key in ("requested", "command")]
        clipped = requested != command
        spacing = max(1, len(rows) // 20)
        ax.plot(time, requested, "--o", color="tab:blue", markevery=spacing,
                markersize=4, label="requested (before clipping)")
        ax.plot(time, command, "-^", color="tab:orange", markevery=spacing,
                markersize=4, label="command (after clipping)")
        ax.scatter(time[clipped], command[clipped], color="red", marker="x",
                   label="clipped", zorder=3)
        ax.set_title(f"{name.capitalize()} velocity")
        ax.set_ylabel(f"Velocity [{unit}]")
        ax.grid(alpha=0.3)
        ax.legend()
        print(f"{name}: clipped {clipped.sum()}/{len(rows)} "
              f"({clipped.mean():.1%}); peak |request| {np.max(np.abs(requested)):.4g} {unit}")
    axes[-1].set_xlabel("Time [s]")
    save_fig(fig, save_dir, label, "velocities")
    return fig



def plot_run(run_dir, label, ref=None, goal=None, save_dir=None):
    """All figures for one run: velocities, {e-t, edot-t}, {e-edot}, and {states-t | x-y}."""
    velocity_path = os.path.join(run_dir, "velocities.csv")
    if os.path.isfile(velocity_path):
        plot_velocities(velocity_path, label, save_dir)
    else:
        print(f"skipping missing file {velocity_path}")

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

    # One figure, two panels: {x-t, y-t, th-t} on the left, {x-y} on the right
    fig, (ax, ax_xy) = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle(f"{label}: robot states vs time and path")

    # left: all three states on one axes (x, y in m and theta in rad share a similar range)
    for name, v, unit in [("x", x, "m"), ("y", y, "m"), ("theta", theta, "rad")]:
        ax.plot(t, v, label=f"{name} [{unit}]", **style(name, len(t)))
    if goal is not None:
        for name, g in (("x", goal[0]), ("y", goal[1])):
            ax.axhline(g, color=SERIES_STYLE[name][0], linestyle=":", linewidth=1.2, label=f"goal {name} = {g:g} m")
    finish(ax, "robot states vs time", "time [s]", "x, y [m]  /  θ [rad]")

    # right: x-y path
    ax = ax_xy
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
    finish(ax, "robot path x vs y", "x [m]", "y [m]")
    save_fig(fig, save_dir, label, "states_path")


if __name__ == "__main__":
    # Parse args, compute metrics for every run, draw/save figures, print both tables
    parser = argparse.ArgumentParser(description="Plot linear/angular PID error logs and robot motion.")
    parser.add_argument("--runs", nargs="+", default=["data"],
                        help="directories containing linear.csv, angular.csv, robot_pose.csv (default: data)")
    parser.add_argument("--labels", nargs="+", help="title label per run, e.g. P PID (default: directory name)")
    parser.add_argument("--save", metavar="DIR", help="save every figure as DIR/<label>_<figure>.png")
    parser.add_argument("--ref", choices=REF_TRAJECTORIES, help="overlay the planned trajectory from planner.py")
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"), help="overlay a point-planner goal")
    parser.add_argument("--no-show", action="store_true", help="only print the tables, do not open figure windows")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    if len(labels) != len(args.runs):
        parser.error("--labels must have one entry per --runs directory")

    results, report_rows, pos_rows = [], [], []
    for run_dir, label in zip(args.runs, labels):
        for ctrl in CONTROLLERS:
            path = os.path.join(run_dir, f"{ctrl}.csv")
            if os.path.isfile(path):
                t, e, _, _ = load_log(path)
                results.append((label, ctrl, step_metrics(t, e)))
                report_rows.append((label, ctrl, report_metrics(t, e)))
        pose_path = os.path.join(run_dir, "robot_pose.csv")
        if args.ref and os.path.isfile(pose_path):
            t, x, y, _ = load_log(pose_path)
            results.append((label, "path", step_metrics(t, path_error(x, y, reference_path(args.ref)))))
        if args.goal and os.path.isfile(pose_path):
            _, x, y, _ = load_log(pose_path)
            pos_rows.append((label, *position_overshoot(x, y, args.goal)))
        if not args.no_show or args.save:
            plot_run(run_dir, label, args.ref, args.goal, args.save)
    print_metrics(results)
    if not args.ref:   # rubric: agility/accuracy/overshoot table is for the point controller only
        print_report_table(report_rows, pos_rows)
    if not args.no_show:
        plt.show()
