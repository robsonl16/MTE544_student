# You can use this file to plot the loged sensor data
# Note that you need to modify/adapt it to your own files
# Feel free to make any modifications/additions here

import math
import matplotlib.pyplot as plt
from pathlib import Path
from utilities import FileReader

def plot_errors(directory):
    for filepath in sorted(Path(directory).glob("*.csv")):
        plot_file(filepath)
        # Save next to the data, e.g. results/line/odom_content_line.png
        plt.gcf().savefig(filepath.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.show()


SENSOR_NAMES = {"odom": "Odometry", "imu": "IMU", "laser": "LIDAR"}


def format_title(filename):
    # e.g. odom_content_spiral.csv -> "Odometry Data: Spiral Motion"
    sensor, _, motion = Path(filename).stem.split("_", 2)
    return f"{SENSOR_NAMES.get(sensor, sensor)} Data: {motion.capitalize()} Motion"


def plot_laser(filename, headers, values, num_scans=3):
    # Convert ranges to cartesian points in the laser frame: theta_i = i * angle_increment
    # (angle_min isn't logged, so 0 is assumed; for a full 360 deg scan this only rotates the plot)
    plt.title(format_title(filename) + " (scans in LIDAR frame)")
    ranges_idx = headers.index("ranges")
    inc_idx = headers.index("angle_increment")
    first_stamp = values[0][-1]

    # Plot a few scans spread across the run (first, middle, last, ...)
    step = max(1, (len(values) - 1) // max(1, num_scans - 1))
    markers = ['o', 's', '^', 'x', 'd']
    for n, row in enumerate(values[::step][:num_scans]):
        xs, ys = [], []
        for i, r in enumerate(row[ranges_idx]):
            if math.isfinite(r):
                theta = i * row[inc_idx]
                xs.append(r * math.cos(theta))
                ys.append(r * math.sin(theta))
        t = (row[-1] - first_stamp) / 1e9
        plt.scatter(xs, ys, s=4, marker=markers[n % len(markers)], label=f"scan at t = {t:.1f} s")

    plt.scatter([0], [0], c='k', marker='*', s=80, label="robot (lidar origin)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")


def plot_odom(filename, time_list, values):
    # Large x vs y path on the left; x, y and th vs time stacked on the right
    plt.close(plt.gcf())
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(format_title(filename), fontsize=14)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.4, 1])

    xs = [lin[0] for lin in values]
    ys = [lin[1] for lin in values]
    ax_path = fig.add_subplot(grid[:, 0])
    ax_path.plot(xs, ys, color="tab:purple", label="path")
    ax_path.scatter(xs[0], ys[0], color="tab:green", marker="o", s=60, zorder=3, label="start")
    ax_path.scatter(xs[-1], ys[-1], color="tab:red", marker="X", s=60, zorder=3, label="end")
    ax_path.set_title("x vs y")
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.axis("equal")
    ax_path.legend()
    ax_path.grid()

    axes = [fig.add_subplot(grid[0, 1])]
    axes += [fig.add_subplot(grid[row, 1], sharex=axes[0]) for row in (1, 2)]
    plot_time_series(axes, time_list, values, [("x", "x [m]", "tab:blue"),
                                                ("y", "y [m]", "tab:orange"),
                                                ("th", "θ [rad]", "tab:green")])
    fig.tight_layout()


def plot_time_series(axes, time_list, values, styles):
    # One column of the logged data per subplot, all sharing the time axis
    for i, (ax, (name, ylabel, color)) in enumerate(zip(axes, styles)):
        if i < len(axes) - 1:
            ax.tick_params(labelbottom=False)
        ax.plot(time_list, [lin[i] for lin in values], color=color, label=name)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.grid()

    axes[-1].set_xlabel("time since start [s]")


def plot_imu(filename, time_list, values):
    # a_x, a_y and w_z each get their own subplot, sharing the time axis
    plt.close(plt.gcf())
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    fig.suptitle(format_title(filename), fontsize=14)
    plot_time_series(axes, time_list, values, [("a_x", "a_x [m/s²]", "tab:blue"),
                                               ("a_y", "a_y [m/s²]", "tab:orange"),
                                               ("ω_z", "ω_z [rad/s]", "tab:green")])
    fig.tight_layout()


def plot_file(filename):
    plt.figure()
    plt.title(format_title(filename))
    headers, values = FileReader(filename).read_file()

    if "ranges" in headers:
        plt.gcf().set_size_inches(10, 7)
        plot_laser(filename, headers, values)
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.grid()
        plt.tight_layout()
        return

    time_list=[]
    first_stamp=values[0][-1]

    for val in values:
        # stamps are logged in ns; plot seconds since the first sample
        time_list.append((val[-1] - first_stamp) / 1e9)

    if headers[:3] == ["x", "y", "th"]:
        plot_odom(filename, time_list, values)
        return

    if headers[:3] == ["acc_x", "acc_y", "angular_z"]:
        plot_imu(filename, time_list, values)
        return

    for i in range(0, len(headers) - 1):
        plt.plot(time_list, [lin[i] for lin in values], label=headers[i])

    plt.xlabel("time since start [s]")

    #plt.plot([lin[0] for lin in values], [lin[1] for lin in values])
    plt.legend()
    plt.grid()
    
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--directory', required=True, help='Directory containing files to process')
    
    args = parser.parse_args()
    
    print("plotting the files in", args.directory)

    plot_errors(args.directory)
