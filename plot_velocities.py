"""Plot requested/limited velocities: python3 plot_velocities.py velocities.csv."""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def plot_velocities(filename, save=None):
    with open(filename, newline="") as stream:
        rows = list(csv.DictReader(stream))
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

        requested, command, limit = [column(key) for key in ("requested", "command", "limit")]
        at_limit = column("at_limit").astype(bool)
        clipped = column("clipped").astype(bool)
        spacing = max(1, len(rows) // 20)
        ax.plot(time, requested, "--o", color="tab:blue", markevery=spacing,
                markersize=4, label="requested (before clipping)")
        ax.plot(time, command, "-^", color="tab:orange", markevery=spacing,
                markersize=4, label="command (after clipping)")
        ax.plot(time, limit, ":", color="black", label="± velocity limit")
        ax.plot(time, -limit, ":", color="black")
        ax.scatter(time[at_limit], command[at_limit], color="red", marker="x",
                   label="at limit", zorder=3)
        ax.set_title(f"{name.capitalize()} velocity")
        ax.set_ylabel(f"Velocity [{unit}]")
        ax.grid(alpha=0.3)
        ax.legend()
        print(f"{name}: at limit {at_limit.sum()}/{len(rows)} samples "
              f"({at_limit.mean():.1%}); clipped {clipped.sum()}/{len(rows)} "
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
