"""Gazebo + TurtleBot3 in room.sdf, like turtlebot3_house.launch.py but with our map.

usage: ros2 launch room_world.launch.py [x_pose:=0.0] [y_pose:=0.0]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    tb3_launch = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    gazebo_launch = os.path.join(get_package_share_directory('gazebo_ros'), 'launch')
    world = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'room.sdf')

    def include(path, **args):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=args.items())

    return LaunchDescription([
        SetEnvironmentVariable('TURTLEBOT3_MODEL', os.environ.get('TURTLEBOT3_MODEL', 'burger')),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        include(os.path.join(gazebo_launch, 'gzserver.launch.py'), world=world),
        include(os.path.join(gazebo_launch, 'gzclient.launch.py')),
        include(os.path.join(tb3_launch, 'robot_state_publisher.launch.py'), use_sim_time='true'),
        include(os.path.join(tb3_launch, 'spawn_turtlebot3.launch.py'),
                x_pose=LaunchConfiguration('x_pose'), y_pose=LaunchConfiguration('y_pose')),
    ])
