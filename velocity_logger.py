"""Log controller requests and commands at the velocity limiter."""

import csv


class VelocityLogger:
    def __init__(self, filename="velocities.csv"):
        self.filename = filename
        with open(filename, "w", newline="") as stream:
            csv.writer(stream).writerow([
                "stamp", "linear_requested", "linear_command", "linear_limit",
                "linear_at_limit", "linear_clipped", "angular_requested",
                "angular_command", "angular_limit", "angular_at_limit",
                "angular_clipped",
            ])

    def log(self, stamp, linear_requested, linear_command, linear_limit,
            angular_requested, angular_command, angular_limit):
        # Pose timestamps are ROS builtin_interfaces/Time messages.
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        row = [stamp_ns]
        for requested, command, limit in (
                (linear_requested, linear_command, linear_limit),
                (angular_requested, angular_command, angular_limit)):
            row.extend([requested, command, limit,
                        int(abs(requested) >= limit),
                        int(abs(requested) > limit)])
        with open(self.filename, "a", newline="") as stream:
            csv.writer(stream).writerow(row)
