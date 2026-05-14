import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue 

def generate_launch_description():
    pkg_share = get_package_share_directory('smart_endoscope')
    world_file = os.path.join(pkg_share, 'worlds', 'airway_world.sdf')
    urdf_file = os.path.join(pkg_share, 'urdf', 'smart_broncho.urdf')

    # Wrap in ParameterValue so ROS 2 knows it's a string, not YAML
    robot_desc = ParameterValue(Command(['xacro ', urdf_file]), value_type=str)

    # --- Tell Gazebo where to find 'package://' URIs ---
    set_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(pkg_share, '..')
    )

    # 1. Start Gazebo
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items()
    )

    # 2. Publish Robot Description
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]
    )

    # 3. Spawn Robot
    #    Note: 'create' expects a plain substitution (not ParameterValue), so use Command directly
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-string', Command(['xacro ', urdf_file]), '-name', 'bronchoscope'],
        output='screen'
    )

    # 4. Bridge the Camera and Clock to ROS 2
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/airway_world/model/bronchoscope/link/distal_tip/sensor/cmos_camera/image'
            '@sensor_msgs/msg/Image@gz.msgs.Image',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        output='screen'
    )

    # 5. Spawn Controllers
    load_jsb = Node(package='controller_manager', executable='spawner', arguments=['joint_state_broadcaster'])
    load_pc  = Node(package='controller_manager', executable='spawner', arguments=['position_controller'])

    return LaunchDescription([
        set_resource_path, gz_sim, robot_state_publisher, spawn_robot, bridge, load_jsb, load_pc
    ])