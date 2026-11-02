import sys

from utilities import Logger, euler_from_quaternion
from rclpy.time import Time
from rclpy.node import Node

from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry as odom

from rclpy import init, shutdown, spin

rawSensor = 0   # localization source: raw wheel odometry (no filtering yet)
class localization(Node):
    """
    ROS 2 node that subscribes to /odom, stores the latest pose [x, y, theta, stamp]
    and logs every pose to robot_pose.csv.
    """

    def __init__(self, localizationType=rawSensor):

        super().__init__("localizer")

        # Part 3: QoS matching the robot's /odom publisher (best effort, volatile, keep last 10)
        odom_qos=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.loc_logger=Logger("robot_pose.csv", ["x", "y", "theta", "stamp"])
        self.pose=None
        
        if localizationType == rawSensor:
        # Part 3: subscribe to the position sensor topic (Odometry)
            self.odom_sub=self.create_subscription(msg_type=odom, topic="/odom", callback=self.odom_callback, qos_profile=odom_qos)
        else:
            print("This type doesn't exist", sys.stderr)
    
    
    def odom_callback(self, pose_msg):
        # Called on every /odom message: extract the planar pose and log it

        # Read x, y, theta (yaw from the orientation quaternion) and record the stamp
        x, y = pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y
        quat = pose_msg.pose.pose.orientation
        theta = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.pose=[x, y, theta, pose_msg.header.stamp]
        
        # Log the data
        self.loc_logger.log_values([self.pose[0], self.pose[1], self.pose[2], Time.from_msg(self.pose[3]).nanoseconds])
    
    def getPose(self):
        # Latest pose [x, y, theta, stamp], or None until the first /odom message arrives
        return self.pose

# Part 3: run the localizer standalone only when this file is executed directly
# (lets it be tested on its own before being used inside decisions.py)

if __name__ == "__main__":
    init()
    localizer = localization()

    try:
        spin(localizer)
    finally:
        localizer.destroy_node()
        shutdown()
    
