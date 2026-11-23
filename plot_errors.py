import argparse
import os

import matplotlib.pyplot as plt
from utilities import FileReader


# A run folder holds the logs written by decisions.py
POSE_FILES = ["robot_pose.csv", "robotPose.csv"]
ERROR_FILES = {"linear": "linear.csv", "angular": "angular.csv"}
PLANNED_PATH_FILE = "plannedPath.csv"


def read_log(path):
    # returns (headers, values) or None if the file is missing or has no data rows
    if not os.path.isfile(path):
        return None
    try:
        headers, values = FileReader(path).read_file()
    except StopIteration:
        return None
    if len(values) == 0:
        return None
    return headers, values


def time_axis(values):
    # stamps are in nanoseconds, convert to seconds since the first sample
    first_stamp = values[0][-1]
    return [(val[-1] - first_stamp) / 1e9 for val in values]


def column(headers, values, name):
    return [row[headers.index(name)] for row in values]


def plot_run(folder):

    pose = None
    for name in POSE_FILES:
        pose = read_log(os.path.join(folder, name))
        if pose is not None:
            break

    errors = {key: read_log(os.path.join(folder, name)) for key, name in ERROR_FILES.items()}
    planned = read_log(os.path.join(folder, PLANNED_PATH_FILE))

    if pose is None and planned is None and all(err is None for err in errors.values()):
        print(f"no logs found in {folder}")
        return

    # the kalman filter log (robotPose.csv) has imu and filter columns, the plain one has x, y, theta
    kalman = pose is not None and "kf_x" in pose[0]

    if kalman:
        layout = [["traj", "traj", "pos", "pos", "vel", "vel"],
                  ["x", "x", "x", "ax", "ax", "ax"],
                  ["y", "y", "y", "ay", "ay", "ay"],
                  ["linear", "linear", "linear", "angular", "angular", "angular"]]
        figsize = (16, 16)
    else:
        layout = [["traj", "pos"],
                  ["x", "y"],
                  ["linear", "angular"]]
        figsize = (12, 12)

    fig, axes = plt.subplot_mosaic(layout, figsize=figsize, layout="constrained")
    fig.suptitle(os.path.abspath(folder))
    ax_traj = axes["traj"]

    # trajectory and pose over time
    if pose is not None:
        headers, values = pose
        x_name, y_name = ("kf_x", "kf_y") if kalman else ("x", "y")
        x, y = column(headers, values, x_name), column(headers, values, y_name)

        ax_traj.plot(x, y, label="trajectory")
        ax_traj.plot(x[0], y[0], "go", label="start")
        ax_traj.plot(x[-1], y[-1], "rx", label="end")
        ax_traj.set_xlabel("x [m]")
        ax_traj.set_ylabel("y [m]")
        ax_traj.set_aspect("equal", adjustable="datalim")

        t = time_axis(values)

        def plot_columns(ax, names, title, ylabel):
            for name in names:
                ax.plot(t, column(headers, values, name), label=name)
            ax.set_title(title)
            ax.set_xlabel("time [s]")
            ax.set_ylabel(ylabel)

        if kalman:
            plot_columns(axes["pos"], ["kf_x", "kf_y"], "position (KF)", "[m]")
            plot_columns(axes["vel"], ["kf_vx", "kf_w"], "velocity (KF)", "[m/s], [rad/s]")
            plot_columns(axes["ax"], ["imu_ax", "kf_ax"], "x acceleration: IMU vs KF", "[m/s^2]")
            plot_columns(axes["ay"], ["imu_ay", "kf_ay"], "y acceleration: IMU vs KF", "[m/s^2]")
        else:
            plot_columns(axes["pos"], headers[:-1], "pose vs time", "[m], [rad]")

        plot_columns(axes["x"], [x_name], "x vs time", "x [m]")
        plot_columns(axes["y"], [y_name], "y vs time", "y [m]")
    else:
        axes["pos"].set_title("pose vs time")

    if planned is not None:
        headers, values = planned
        planned_x, planned_y = column(headers, values, "x"), column(headers, values, "y")
        ax_traj.plot(planned_x, planned_y, "k--", label="planned path")
        # the planned path has no timestamps, so only its goal is shown against time
        axes["x"].axhline(planned_x[-1], color="k", linestyle="--", label="goal x")
        axes["y"].axhline(planned_y[-1], color="k", linestyle="--", label="goal y")
        ax_traj.set_xlabel("x [m]")
        ax_traj.set_ylabel("y [m]")
        ax_traj.set_aspect("equal", adjustable="datalim")

    ax_traj.set_title("trajectory")

    # controller errors over time
    for key in ["linear", "angular"]:
        ax = axes[key]
        if errors[key] is not None:
            headers, values = errors[key]
            t = time_axis(values)
            for i in range(len(headers) - 1):
                ax.plot(t, [row[i] for row in values], label=headers[i])
            ax.set_xlabel("time [s]")
        ax.set_title(f"{key} error")

    for ax in axes.values():
        if ax.has_data():
            ax.legend()
        ax.grid()


    image_path = os.path.join(folder, "plots.png")
    fig.savefig(image_path)
    print(f"saved {image_path}")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="plot trajectory, pose and controller errors from logged runs")
    argParser.add_argument("folders", nargs="*", default=["data"],
                           help="folders containing robot_pose.csv, linear.csv, angular.csv (default: data)")
    args = argParser.parse_args()

    for folder in args.folders:
        plot_run(folder)

    plt.show()
