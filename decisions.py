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
    
    def __init__(self, goalPoint, publisher_msg=Twist, publishing_topic='cmd_vel', qos_publisher=10, rate=10, motion_type=POINT_PLANNER):

        super().__init__("decision_maker")
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST,
                        depth=10)
        # Part 4: Create a publisher for the topic responsible for robot's motion
        self.publisher=self.create_publisher(msg_type=publisher_msg, topic=publishing_topic, qos_profile=qos)

        publishing_period=1/rate
        
        # Instantiate the controller
        # TODO Part 5: Tune your parameters here
    
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

        self.create_timer(publishing_period, self.timerCallback)
        print('Decision maker node is initialized')


    def timerCallback(self):
        
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
        
        # TODO Part 3: Check if you reached the goal
        if type(self.goal) == list:
            # if goal is a list, we are following a trajectory of points, determine if we reached last point
            reached_goal=self.controller.lookFarFor(current_pose, self.goal) == self.goal[-1]

        else: 
            # else, point_planner mode, no list of points
            # considered goal reached if less than 1cm and 2.5deg error
            lin_error = calculate_linear_error(current_pose, self.goal)
            ang_error = calculate_angular_error(current_pose, self.goal)
            reached_goal = (lin_error < 0.01) and (ang_error < 0.04)
        

        if reached_goal:
            print("reached goal")
            self.publisher.publish(vel_msg)
            
            self.controller.PID_angular.logger.save_log()
            self.controller.PID_linear.logger.save_log()
            
            #Part 3: exit the spin
            print("exiting the spin")
            shutdown()
            return
        
        velocity, yaw_rate = self.controller.vel_request(self.localizer.getPose(), self.goal, True)

        # Part 4: Publish the velocity to move the robot
        vel_msg=Twist()
        vel_msg.linear.x = velocity
        vel_msg.angular.z = yaw_rate
        self.publisher.publish(vel_msg)

import argparse


def main(args=None):
    
    init()

    # TODO Part 3: You might need to change the QoS profile based on whether you're using the real robot or in simulation.
    # Remember to define your QoS profile based on the information available in "ros2 topic info /odom --verbose" as explained in Tutorial 3
    
    odom_qos=QoSProfile(reliability=2, durability=2, history=1, depth=10)
    # odom_qos=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

    # Part 4: instantiate the decision_maker with the proper parameters for moving the robot
    if args.motion.lower() == "point":
        DM=decision_maker(goalPoint=[1.0, -1.0], publisher_msg=Twist, publishing_topic='cmd_vel', qos_publisher=odom_qos, rate=10, motion_type=POINT_PLANNER)
    elif args.motion.lower() == "trajectory":
        DM=decision_maker(goalPoint=[1.0, -1.0], publisher_msg=Twist, publishing_topic='cmd_vel', qos_publisher=odom_qos, rate=10, motion_type=TRAJECTORY_PLANNER)
    else:
        print("invalid motion type", file=sys.stderr)        
    
    
    
    try:
        spin(DM)
    except SystemExit:
        print(f"reached there successfully {DM.localizer.pose}")


if __name__=="__main__":

    argParser=argparse.ArgumentParser(description="point or trajectory") 
    argParser.add_argument("--motion", type=str, default="point")
    args = argParser.parse_args()

    main(args)
