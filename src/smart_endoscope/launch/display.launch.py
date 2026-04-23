import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command

def generate_launch_description():
    pkg_share = get_package_share_directory('smart_endoscope')
    urdf_file = os.path.join(pkg_share, 'urdf', 'smart_broncho.urdf')

    # Reads the URDF and publishes the /robot_description topic
    robot_desc = Command(['xacro ', urdf_file])

    # 1. Robot State Publisher (Calculates TF tree)
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )

    # 2. Joint State Publisher GUI (Provides the sliders!)
    jsp_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui'
    )

    # 3. RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen'
    )

    return LaunchDescription([rsp_node, jsp_gui_node, rviz_node])
