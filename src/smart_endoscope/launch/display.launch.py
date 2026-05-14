import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue  # <--- ADD THIS

def generate_launch_description():
    pkg_share = get_package_share_directory('smart_endoscope')
    urdf_file = os.path.join(pkg_share, 'urdf', 'smart_broncho.urdf')

    # Wrap in ParameterValue so ROS 2 knows it's a string, not YAML
    robot_desc = ParameterValue(Command(['xacro ', urdf_file]), value_type=str)  # <--- FIX

    # 1. Robot State Publisher (calculates TF tree)
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )

    # 2. Joint State Publisher GUI (provides the sliders)
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