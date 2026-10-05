# Imports
import rclpy

from rclpy.node import Node

from utilities import Logger, euler_from_quaternion
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from rclpy.time import Time

# You may add any other imports you may need/want to use below
# import ...


CIRCLE=0; SPIRAL=1; ACC_LINE=2
motion_types=['circle', 'spiral', 'line']

class motion_executioner(Node):
    
    def __init__(self, motion_type=0):
        
        super().__init__("motion_types")
        
        self.type=motion_type
        
        self.radius_=0.0
        
        self.successful_init=False
        self.imu_initialized=False
        self.odom_initialized=False
        self.laser_initialized=False

        self.spiral_speed = 0.0
        self.linear_acc_speed = 0.0
        
        # Create a publisher to send velocity commands. Set queue size to 10
        self.vel_publisher=self.create_publisher(msg_type=Twist, topic='cmd_vel', qos_profile=10)
                
        # loggers
        self.imu_logger=Logger('imu_content_'+str(motion_types[motion_type])+'.csv', headers=["acc_x", "acc_y", "angular_z", "stamp"])
        self.odom_logger=Logger('odom_content_'+str(motion_types[motion_type])+'.csv', headers=["x","y","th", "stamp"]) #can get twist from this message. do we not need?
        self.laser_logger=Logger('laser_content_'+str(motion_types[motion_type])+'.csv', headers=["ranges", "angle_increment", "stamp"])
        
        # TODO In Lab: Setup the QoS profile for the actual robot 
        # These are settings from simulation running `ros2 topic info /odom --verbose``
        qos=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        qos_lab_imu=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        qos_lab_odom=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        qos_lab_laser=QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        # TODO Part 5: Create below the subscription to the topics corresponding to the respective sensors
        # IMU subscription
        self.imu_sub=self.create_subscription(msg_type=Imu, topic='imu', callback=self.imu_callback, qos_profile=qos_lab_imu)
        
        # ENCODER subscription
        self.odom_sub=self.create_subscription(msg_type=Odometry, topic='odom', callback=self.odom_callback, qos_profile=qos_lab_odom)
        
        # LaserScan subscription 
        self.laser_sub=self.create_subscription(msg_type=LaserScan, topic='scan', callback=self.laser_callback, qos_profile=qos_lab_laser)
        
        self.create_timer(0.1, self.timer_callback)


    # TODO Part 5: Callback functions: complete the callback functions of the three sensors to log the proper data.
    # To also log the time you need to use the rclpy Time class, each ros msg will come with a header, and then
    # inside the header you have a stamp that has the time in seconds and nanoseconds, you should log it in nanoseconds as 
    # such: Time.from_msg(imu_msg.header.stamp).nanoseconds
    # You can save the needed fields into a list, and pass the list to the log_values function in utilities.py

    def imu_callback(self, imu_msg: Imu):
        # log imu msgs
        self.imu_initialized=True
        log_entry = {
            "acc_x": imu_msg.linear_acceleration.x,
            "acc_y": imu_msg.linear_acceleration.y,
            "angular_z": imu_msg.angular_velocity.z,
            "stamp": Time.from_msg(imu_msg.header.stamp).nanoseconds
        }
        self.imu_logger.log_values(log_entry)


    def odom_callback(self, odom_msg: Odometry):
        # log odom msgs
        self.odom_initialized=True
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        log_entry = {
            "x": position.x,
            "y": position.y,
            "th": yaw,
            "stamp": Time.from_msg(odom_msg.header.stamp).nanoseconds
        }
        self.odom_logger.log_values(log_entry)
                
    def laser_callback(self, laser_msg: LaserScan):
        # log laser msgs with position msg at that time
        self.laser_initialized=True
        timestamp = Time.from_msg(laser_msg.header.stamp).nanoseconds
        ranges = list(laser_msg.ranges)
        angle_increment = laser_msg.angle_increment
        log_entry = {
            "ranges": ranges,
            "angle_increment": angle_increment,
            "stamp": timestamp
        }
        self.laser_logger.log_values(log_entry)

    def timer_callback(self):
        
        if self.odom_initialized and self.laser_initialized and self.imu_initialized:
            self.successful_init=True
            
        if not self.successful_init:
            return
        
        cmd_vel_msg=Twist()
        
        if self.type==CIRCLE:
            cmd_vel_msg=self.make_circular_twist()
        
        elif self.type==SPIRAL:
            cmd_vel_msg=self.make_spiral_twist()
                        
        elif self.type==ACC_LINE:
            cmd_vel_msg=self.make_acc_line_twist()
            
        else:
            print("type not set successfully, 0: CIRCLE 1: SPIRAL and 2: ACCELERATED LINE")
            raise SystemExit 

        self.vel_publisher.publish(cmd_vel_msg)
        
    
    # Motion functions

    def make_circular_twist(self):
        
        msg=Twist()
        msg.linear.x=0.2
        msg.angular.z=0.8
        return msg

    def make_spiral_twist(self):
        msg=Twist()
        self.spiral_speed=min(self.spiral_speed+0.001, 0.2)
        msg.linear.x=self.spiral_speed
        msg.angular.z=0.4
        return msg
    
    def make_acc_line_twist(self):
        msg=Twist()
        self.linear_acc_speed=min(self.linear_acc_speed+0.01, 0.8)
        msg.linear.x=self.linear_acc_speed
        msg.angular.z=0.0
        return msg

import argparse

if __name__=="__main__":
    

    argParser=argparse.ArgumentParser(description="input the motion type")


    argParser.add_argument("--motion", type=str, default="circle")



    rclpy.init()

    args = argParser.parse_args()

    if args.motion.lower() == "circle":
        ME=motion_executioner(motion_type=CIRCLE)

    elif args.motion.lower() == "line":
        ME=motion_executioner(motion_type=ACC_LINE)

    elif args.motion.lower() =="spiral":
        ME=motion_executioner(motion_type=SPIRAL)

    else:
        print(f"we don't have {args.motion.lower()} motion type")


    
    try:
        rclpy.spin(ME)
    except KeyboardInterrupt:
        print("Exiting")
