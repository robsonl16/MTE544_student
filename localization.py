import sys

from utilities import Logger, euler_from_quaternion
from rclpy.time import Time
from rclpy.node import Node

from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry as odom

from rclpy import init, shutdown, spin

rawSensor = 0
class localization(Node):
    
    def __init__(self, localizationType=rawSensor):

        super().__init__("localizer")
        
        # Part 3: Define the QoS profile variable based on whether you are using the simulation (Turtlebot 3 Burger) or the real robot (Turtlebot 4)
        # Remember to define your QoS profile based on the information available in "ros2 topic info /odom --verbose" as explained in Tutorial 3
        odom_qos=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.loc_logger=Logger("robot_pose.csv", ["x", "y", "theta", "stamp"])
        self.pose=None
        
        if localizationType == rawSensor:
        # Part 3: subscribe to the position sensor topic (Odometry)
            self.odom_sub=self.create_subscription(msg_type=odom, topic="/odom", callback=self.odom_callback, qos_profile=odom_qos)
        else:
            print("This type doesn't exist", sys.stderr)
    
    
    def odom_callback(self, pose_msg):
        
        # Read x,y, theta, and record the stamp
        x, y = pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y
        quat = pose_msg.pose.pose.orientation
        theta = euler_from_quaternion(quat)
        self.pose=[x, y, theta, pose_msg.header.stamp]
        
        # Log the data
        self.loc_logger.log_values([self.pose[0], self.pose[1], self.pose[2], Time.from_msg(self.pose[3]).nanoseconds])
    
    def getPose(self):
        return self.pose

# Part 3
# Here put a guard that makes the node run, ONLY when run as a main thread!
# This is to make sure this node functions right before using it in decision.py

if __name__ == "__main__":
    init()
    localizer = localization()

    try:
        spin(localizer)
    finally:
        localizer.destroy_node()
        shutdown()
    
