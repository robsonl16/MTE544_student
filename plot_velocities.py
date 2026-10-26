"""Plot requested/limited velocities: python3 plot_velocities.py velocities.csv."""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def plot_velocities(filename, save=None):
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
    fig.suptitle(f"Controller velocity saturation: {filename}")
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
    if save:
        fig.savefig(save, dpi=150)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", default="velocities.csv")
    parser.add_argument("--save", help="output image path")
    args = parser.parse_args()
    plot_velocities(args.filename, args.save)
    plt.show()
