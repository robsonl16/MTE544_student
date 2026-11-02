# Imports


import sys

from utilities import euler_from_quaternion, calculate_angular_error, calculate_linear_error
from pid import PID_ctrl

from rclpy import init, spin, spin_once, shutdown
from rclpy.node import Node
from geometry_msgs.msg import Twist

from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry as odom
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from localization import localization, rawSensor

from planner import TRAJECTORY_PLANNER, POINT_PLANNER, planner
from controller import controller, trajectoryController

# You may add any other imports you may need/want to use below
# import ...

P=0; PD=1; PI=2; PID=3

class decision_maker(Node):
    """
    Top-level ROS 2 node. At a fixed rate it reads the pose from the localizer, checks whether
    the goal is reached, asks the controller for (v, w) and publishes them on cmd_vel.
    Wires together the planner (where to go), localization (where we are) and controller.
    """

    def __init__(self, goalPoint, publisher_msg=Twist, publishing_topic='cmd_vel', qos_publisher=10, rate=10, motion_type=POINT_PLANNER):

        super().__init__("decision_maker")
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST,
                        depth=10)
        # Part 4: Create a publisher for the topic responsible for robot's motion
        self.publisher=self.create_publisher(msg_type=publisher_msg, topic=publishing_topic, qos_profile=qos)

        publishing_period=1/rate
        
        # Instantiate the controller and planner for the chosen motion
        # Part 5: tuned gains (kl* = linear, ka* = angular; p, v = derivative, i)

        if motion_type == POINT_PLANNER:
            self.controller=controller(PID, klp=0.1, klv=0.15, kli=0.15, kap=0.5, kav=0.15, kai=0.05)
            self.planner=planner(POINT_PLANNER)    
    
        elif motion_type==TRAJECTORY_PLANNER:
            self.controller=trajectoryController(PID, klp=0.025, klv=0.15, kli=0.15, kap=0.5, kav=0.125, kai=0.05)
            self.planner=planner(TRAJECTORY_PLANNER)

        else:
            print("Error! you don't have this planner", file=sys.stderr)


        # Instantiate the localization, use rawSensor for now  
        self.localizer=localization(rawSensor)

        # Instantiate the planner
        # NOTE: goalPoint is used only for the pointPlanner
        self.goal=self.planner.plan(goalPoint)

        # Run the control loop every publishing_period seconds (rate Hz)
        self.create_timer(publishing_period, self.timerCallback)
        print('Decision maker node is initialized')


    def timerCallback(self):
        # One control step: update pose -> check goal -> compute and publish velocities

        # Part 3: Run the localization node
        spin_once(self.localizer, timeout_sec=0)
        # Remember that this file is already running the decision_maker node.

        if self.localizer.getPose()  is  None:
            print("waiting for odom msgs ....")
            return

        current_pose = self.localizer.getPose()

        # take current pose, find closest point in trajectory to find where you are in the path
        # self.goal will be a list of all the point you have to still reach
        # if you are at the end of the trajectory, self.goal will be null

        vel_msg=Twist()
        
        # Part 3: Check if you reached the goal
        if type(self.goal) == list:
            # Trajectory mode: done once the look-ahead point is the last waypoint
            reached_goal=self.controller.lookFarFor(current_pose, self.goal) == self.goal[-1]

        else:
            # Point mode: done when within 1 cm of the goal and ang_error < 0.04 rad (~2.3 deg)
            lin_error = calculate_linear_error(current_pose, self.goal)
            ang_error = calculate_angular_error(current_pose, self.goal)
            reached_goal = (lin_error < 0.01) and (ang_error < 0.04)
        

        if reached_goal:
            # Publish a zero Twist to stop the robot, then shut down the node
            print("reached goal")
            self.publisher.publish(vel_msg)
            
            self.controller.PID_angular.logger.save_log()
            self.controller.PID_linear.logger.save_log()
            
            #Part 3: exit the spin
            print("exiting the spin")
            shutdown()
            return
        
        # Controller computes saturated linear (m/s) and angular (rad/s) velocity from the current pose
        velocity, yaw_rate = self.controller.vel_request(self.localizer.getPose(), self.goal, True)

        # Part 4: Publish the velocity to move the robot
        vel_msg=Twist()
        vel_msg.linear.x = velocity
        vel_msg.angular.z = yaw_rate
        self.publisher.publish(vel_msg)

import argparse


def main(args=None):
    # Entry point: start ROS 2, build the decision maker for --motion point|trajectory, spin until goal reached

    init()

    # Part 3: QoS profile (reliability=2 BEST_EFFORT, durability=2 VOLATILE, history=1 KEEP_LAST, depth 10)
    odom_qos=QoSProfile(reliability=2, durability=2, history=1, depth=10)
    # odom_qos=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

    # Part 4: instantiate the decision_maker with the proper parameters for moving the robot
    if args.motion.lower() == "point":
        DM=decision_maker(goalPoint=[1.0, -1.0], publisher_msg=Twist, publishing_topic='cmd_vel', qos_publisher=odom_qos, rate=10, motion_type=POINT_PLANNER)
    elif args.motion.lower() == "trajectory":
        DM=decision_maker(goalPoint=[1.0, -1.0], publisher_msg=Twist, publishing_topic='cmd_vel', qos_publisher=odom_qos, rate=10, motion_type=TRAJECTORY_PLANNER)
    else:
        print("invalid motion type", file=sys.stderr)        
    
    
    
    # Spin until timerCallback calls shutdown() on reaching the goal
    try:
        spin(DM)
    except SystemExit:
        print(f"reached there successfully {DM.localizer.pose}")


if __name__=="__main__":

    argParser=argparse.ArgumentParser(description="point or trajectory") 
    argParser.add_argument("--motion", type=str, default="point")
    args = argParser.parse_args()

    main(args)
