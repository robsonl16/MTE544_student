"""Plot localization or PID CSV logs without requiring ROS.

Examples:
    python3 plot_errors.py --files sim/robotPose.csv
    python3 plot_errors.py --files sim/*.csv --no-show --output-dir plots
    python3 plot_errors.py --files sim/robotPose.csv --format pdf
    python3 plot_errors.py --files sim/robotPose.csv --ideal-spiral --no-show
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


POSE_COLUMNS = {
    "imu_ax", "imu_ay", "kf_ax", "kf_ay", "kf_vx", "kf_w", "kf_x", "kf_y"
}
LABELS = {
    "e": "Error", "e_dot": "Error derivative", "e_int": "Error integral",
}


def read_log(filename):
    """Read numeric columns, including the trailing commas used by Logger."""
    with Path(filename).open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV is empty")
        columns = [(i, name.strip()) for i, name in enumerate(header) if name.strip()]
        names = [name for _, name in columns]
        if len(names) != len(set(names)):
            raise ValueError("CSV contains duplicate column names")
        if "stamp" not in names:
            raise ValueError("CSV must contain a 'stamp' column in nanoseconds")
        rows = []
        for line, row in enumerate(reader, start=2):
            if not any(value.strip() for value in row):
                continue
            try:
                values = [float(row[i]) for i, _ in columns]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"invalid or missing numeric value on line {line}") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite value on line {line}")
            rows.append(values)
    if not rows:
        raise ValueError("CSV contains no data samples")
    data = dict(zip(names, np.asarray(rows).T))
    if np.any(np.diff(data["stamp"]) < 0):
        raise ValueError("timestamps must be nondecreasing; split logs at clock resets")
    return data


def ideal_spiral(time, start_delay=0.0, start_pose=(0.0, 0.0, 0.0)):
    """Integrate the nominal decisions.py spiral commands exactly between ticks.

    First command is v=0.01 m/s, w=1 rad/s; v increases by 0.01 every
    0.1 s, capped at 1 m/s. Start delay is seconds after the first CSV stamp.
    This is an ideal command reference, not measured ground truth. Timer
    delays, wheel slip, actuator dynamics and sensor noise are not modeled.
    """
    time = np.asarray(time, dtype=float)
    if not np.isfinite(time).all() or not np.isfinite([start_delay, *start_pose]).all():
        raise ValueError("spiral times and initial pose must be finite")
    if start_delay < 0:
        raise ValueError("spiral start delay must be nonnegative")
    x0, y0, yaw0 = start_pose
    elapsed = np.maximum(time - start_delay, 0.0)
    # At tick 99 the command reaches the cap and stays constant thereafter.
    ticks = np.arange(100) / 10.0
    speed = np.minimum((np.arange(100) + 1) * 0.01, 1.0)
    angles = yaw0 + ticks
    dx = speed[:-1] * (np.sin(angles[1:]) - np.sin(angles[:-1]))
    dy = speed[:-1] * (np.cos(angles[:-1]) - np.cos(angles[1:]))
    x_at_tick = x0 + np.r_[0.0, np.cumsum(dx)]
    y_at_tick = y0 + np.r_[0.0, np.cumsum(dy)]
    index = np.minimum(np.floor(elapsed * 10).astype(int), 99)
    yaw = yaw0 + elapsed
    x = x_at_tick[index] + speed[index] * (np.sin(yaw) - np.sin(angles[index]))
    y = y_at_tick[index] + speed[index] * (np.cos(angles[index]) - np.cos(yaw))
    moving = time >= start_delay
    return {"x": x, "y": y, "theta": yaw,
            "vx": np.where(moving, speed[index], 0.0),
            "w": moving.astype(float)}


def plot_errors(filename, *, with_ideal_spiral=False, spiral_start_delay=0.0,
                spiral_start_pose=(0.0, 0.0, 0.0)):
    """Return a figure for a robot pose log or a generic stamped numeric log."""
    import matplotlib.pyplot as plt

    data = read_log(filename)
    time = (data["stamp"] - data["stamp"][0]) / 1e9
    if POSE_COLUMNS.issubset(data):
        has_odom = {"odom_x", "odom_y"}.issubset(data)
        reference = (ideal_spiral(time, spiral_start_delay, spiral_start_pose)
                     if with_ideal_spiral else None)
        rows = 5 if has_odom or reference is not None else 3
        fig, axes = plt.subplots(rows, 2, figsize=(13, 3.6 * rows), layout="constrained")
        trajectory = axes[0, 0]
        trajectory.plot(data["kf_x"], data["kf_y"], color="tab:blue", label="EKF")
        if has_odom:
            trajectory.plot(data["odom_x"], data["odom_y"],
                            color="tab:orange", linestyle="--", label="Odometry")
        if reference is not None:
            trajectory.plot(reference["x"], reference["y"], color="black",
                            linestyle=":", label="Ideal spiral", linewidth=1.5)
        trajectory.scatter(data["kf_x"][0], data["kf_y"][0],
                           color="green", label="Start", zorder=3)
        trajectory.scatter(data["kf_x"][-1], data["kf_y"][-1],
                           color="red", marker="x", label="End", zorder=3)
        trajectory.set(title="Robot trajectory", xlabel="x (m)", ylabel="y (m)")
        trajectory.set_aspect("equal", adjustable="datalim")
        trajectory.legend()

        position = axes[0, 1]
        position.plot(time, data["kf_x"], label="x")
        position.plot(time, data["kf_y"], label="y", linestyle="--")
        position.set(title="Estimated position", ylabel="Position (m)")
        position.legend()

        for ax, key, title in [
            (axes[1, 0], "ax", "Longitudinal acceleration"),
            (axes[1, 1], "ay", "Lateral acceleration"),
        ]:
            ax.plot(time, data["imu_" + key], label="IMU measurement",
                    alpha=0.6, linewidth=0.8)
            ax.plot(time, data["kf_" + key], label="EKF estimate",
                    linewidth=1.2, linestyle="--")
            ax.set(title=title, ylabel="Acceleration (m/s²)")
            ax.legend()

        axes[2, 0].plot(time, data["kf_vx"], color="tab:green")
        axes[2, 0].set(title="Estimated forward velocity", ylabel="Velocity (m/s)")
        axes[2, 1].plot(time, data["kf_w"], color="tab:purple")
        axes[2, 1].set(title="Estimated yaw rate", ylabel="Yaw rate (rad/s)")
        if reference is not None:
            for ax, key in [(axes[2, 0], "vx"), (axes[2, 1], "w")]:
                ax.lines[0].set_label("EKF")
                ax.plot(time, reference[key], color="black", linestyle=":", label="Ideal command")
                ax.legend()
        if has_odom or reference is not None:
            for ax, coordinate in zip(axes[3], ("x", "y")):
                if has_odom:
                    ax.plot(time, data["odom_" + coordinate], label="Odometry",
                            color="tab:orange", linestyle="--")
                ax.plot(time, data["kf_" + coordinate], label="EKF", color="tab:blue")
                if reference is not None:
                    ax.plot(time, reference[coordinate], color="black",
                            linestyle=":", label="Ideal spiral")
                ax.set(title=f"{coordinate} position comparison", ylabel=f"{coordinate} (m)")
                ax.legend()

            heading = axes[4, 0]
            # Legacy logger names: odom_z is quaternion z and odom_theta is
            # quaternion w, not yaw. This conversion assumes planar rotation
            # (quaternion x=y=0); full 3D orientation needs all four components.
            if {"odom_z", "odom_theta"}.issubset(data):
                z, w = data["odom_z"], data["odom_theta"]
                norm = np.hypot(z, w)
                if np.any(norm == 0):
                    raise ValueError("odometry quaternion z/w has zero norm")
                yaw = np.unwrap(2 * np.arctan2(z / norm, w / norm))
                heading.plot(time, yaw, label="Odometry (planar quaternion)",
                             color="tab:orange", linestyle="--")
                heading.legend()
            elif "kf_theta" not in data and reference is None:
                heading.text(0.5, 0.5, "No odometry orientation columns",
                             ha="center", va="center", transform=heading.transAxes)
            if "kf_theta" in data:
                heading.plot(time, np.unwrap(data["kf_theta"]), label="EKF", color="tab:blue")
                heading.legend()
            if reference is not None:
                heading.plot(time, reference["theta"], color="black",
                             linestyle=":", label="Ideal spiral")
                heading.legend()
            heading.set(title="Heading (unwrapped)", ylabel="Yaw (rad)")

            difference = axes[4, 1]
            if reference is not None:
                for prefix, label, color in [("kf", "EKF", "tab:blue"),
                                              ("odom", "Odometry", "tab:orange")]:
                    if prefix == "odom" and not has_odom:
                        continue
                    distance = np.hypot(data[prefix + "_x"] - reference["x"],
                                        data[prefix + "_y"] - reference["y"])
                    rmse = np.sqrt(np.mean(distance ** 2))
                    difference.plot(time, distance, color=color,
                                    label=f"{label} (RMSE {rmse:.3f} m)")
                difference.set_title("Position difference from ideal spiral")
                difference.legend()
            else:
                separation = np.hypot(data["kf_x"] - data["odom_x"],
                                      data["kf_y"] - data["odom_y"])
                difference.plot(time, separation, color="tab:red")
                difference.set_title("EKF–odometry position difference (not ground-truth error)")
            difference.set_ylabel("Position difference (m)")
        for ax in axes.flat:
            if ax is not trajectory:
                ax.set_xlabel("Elapsed time (s)")
    else:
        if with_ideal_spiral:
            raise ValueError("--ideal-spiral requires a robot pose CSV")
        signals = [name for name in data if name != "stamp"]
        if not signals:
            raise ValueError("CSV contains no signal columns")
        fig, axes = plt.subplots(len(signals), 1, figsize=(11, 3 * len(signals)),
                                 squeeze=False, layout="constrained")
        for ax, name in zip(axes.flat, signals):
            ax.plot(time, data[name])
            ax.set(title=LABELS.get(name, name), ylabel=name, xlabel="Elapsed time (s)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{Path(filename).name} — {len(time):,} samples over {time[-1]:.1f} s",
                 fontsize=15)
    if with_ideal_spiral:
        fig.suptitle(fig._suptitle.get_text() + "\nIdeal reference assumes perfect nominal command timing/tracking",
                     fontsize=14)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="+", required=True, type=Path,
                        help="CSV files to plot (shell wildcards are supported)")
    parser.add_argument("--output-dir", type=Path,
                        help="Save plots here; defaults to each CSV's directory")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution (default: 180)")
    parser.add_argument("--no-show", action="store_true", help="Save without opening plot windows")
    parser.add_argument("--ideal-spiral", action="store_true",
                        help="Compare with the nominal spiral in decisions.py (not measured ground truth)")
    parser.add_argument("--spiral-start-delay", type=float, default=0.0,
                        help="First spiral command time after the first CSV sample, in seconds (default: 0)")
    parser.add_argument("--spiral-start-pose", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "YAW"), help="Ideal initial pose in m, m, radians (default: 0 0 0)")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    failures = False
    outputs = set()
    for filename in args.files:
        try:
            directory = args.output_dir if args.output_dir is not None else filename.parent
            suffix = "_ideal_spiral" if args.ideal_spiral else "_plot"
            output = directory / f"{filename.stem}{suffix}.{args.format}"
            if output.resolve() in outputs:
                raise ValueError(f"duplicate output path: {output}; use distinct CSV filenames")
            fig = plot_errors(filename, with_ideal_spiral=args.ideal_spiral,
                              spiral_start_delay=args.spiral_start_delay,
                              spiral_start_pose=args.spiral_start_pose)
            directory.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=args.dpi)
            outputs.add(output.resolve())
            print(f"Saved {output}")
            if args.no_show:
                plt.close(fig)
        except (OSError, ValueError) as exc:
            print(f"Error plotting {filename}: {exc}", file=sys.stderr)
            failures = True
    if not args.no_show:
        plt.show()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
