import sys

from utilities import Logger

from rclpy.time import Time

from utilities import (
    euler_from_quaternion,
    calculate_angular_error,
    calculate_linear_error,
)
from rclpy.node import Node
from geometry_msgs.msg import Twist

from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry as odom
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Imu
from kalman_filter import kalman_filter

from rclpy import init, spin, spin_once

import numpy as np
import message_filters

rawSensors = 0
kalmanFilter = 1
# odom_qos = QoSProfile(reliability=2, durability=2, history=1, depth=10)
odom_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
imu_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
# imu_qos = QoSProfile(reliability=2, durability=2, history=1, depth=10)_

class localization(Node):
    def __init__(
        self,
        type,
        dt,
        loggerName="robotPose.csv",
        loggerHeaders=[
            "imu_ax",
            "imu_ay",
            "kf_ax",
            "kf_ay",
            "kf_vx",
            "kf_w",
            "kf_x",
            "kf_y",
            "odom_x",
            "odom_y",
            "odom_z",
            "odom_theta",
            "stamp",
        ],
    ):

        super().__init__("localizer")

        self.loc_logger = Logger(loggerName, loggerHeaders)
        self.pose = None

        if type == rawSensors:
            self.initRawSensors()
        elif type == kalmanFilter:
            self.initKalmanfilter(dt)
        else:
            print("We don't have this type for localization", sys.stderr)
            return

    def initRawSensors(self):
        self.create_subscription(
            odom, "/odom", self.odom_callback, qos_profile=odom_qos
        )

    def initKalmanfilter(self, dt):

        # Part 3: Set up the quantities for the EKF (hint: you will need the functions for the states and measurements)

        x = np.zeros(6)  # initial state vector [x, y, th, w, v, vdot]

        # Update and tune these in lab
        Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 1 # process noise covariance

        R = np.diag([0.1, 0.1, 0.1, 0.1]) * 5  # measurement noise covariance

        P = np.eye(6) * 0.1  # initial covariance

        self.kf = kalman_filter(P, Q, R, x, dt)

        # Part 3: Use the odometry and IMU data for the EKF
        self.odom_sub = message_filters.Subscriber(
            self, odom, "/odom", qos_profile=odom_qos
        )
        # TODO: Have to check if we need different qos for imu.
        self.imu_sub = message_filters.Subscriber(self, Imu, "/imu", qos_profile=imu_qos)

        # Sync odom and imu messages to arrive at the same time
        # TODO: Worst case is set to 0.1 seconds for slop. DT is set to 0.1. May have to adjust
        time_syncher = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.imu_sub], queue_size=10, slop=0.1
        )
        time_syncher.registerCallback(self.fusion_callback)

    def fusion_callback(self, odom_msg: odom, imu_msg: Imu):

        # Part 3: Use the EKF to perform state estimation
        # Take the measurements
        # your measurements are the linear velocity and angular velocity from odom msg
        # and linear acceleration in x and y from the imu msg
        # the kalman filter should do a proper integration to provide x,y and filter ax,ay
        z = np.array(
            [
                odom_msg.twist.twist.linear.x,  # v
                odom_msg.twist.twist.angular.z,  # w
                imu_msg.linear_acceleration.x,  # ax
                imu_msg.linear_acceleration.y,  # ay
            ]
        )

        # Implement the two steps for estimation
        self.kf.predict()
        self.kf.update(z)

        # Get the estimate
        xhat = self.kf.get_states()

        # Update the pose estimate to be returned by getPose
        self.pose = np.array([xhat[0], xhat[1], xhat[2], odom_msg.header.stamp])

        # Part 4: log your data
        self.loc_logger.log_values(
            [
                z[2],
                z[3],
                xhat[5],
                xhat[3]*xhat[4],
                xhat[4],
                xhat[3],
                xhat[0],
                xhat[1],
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
                Time.from_msg(self.pose[3]).nanoseconds,
            ]
        )

    def odom_callback(self, pose_msg):

        self.pose = [
            pose_msg.pose.pose.position.x,
            pose_msg.pose.pose.position.y,
            euler_from_quaternion(pose_msg.pose.pose.orientation),
            pose_msg.header.stamp,
        ]

    # Return the estimated pose
    def getPose(self):
        return self.pose


if __name__ == "__main__":
    init()

    LOCALIZER = localization()

    spin(LOCALIZER)
