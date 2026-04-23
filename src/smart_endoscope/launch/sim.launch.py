import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command

def generate_launch_description():
    pkg_share = get_package_share_directory('smart_endoscope')
    world_file = os.path.join(pkg_share, 'worlds', 'airway_world.sdf')
    urdf_file = os.path.join(pkg_share, 'urdf', 'smart_broncho.urdf')

    robot_desc = Command(['xacro ', urdf_file])

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
        parameters=[{'robot_description': robot_desc}]
    )

    # 3. Spawn Robot (Z elevation removed, handled natively in URDF fixed joint)
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-string', robot_desc, '-name', 'bronchoscope'],
        output='screen'
    )

    # 4. Bridge the Camera to ROS 2
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/world/airway_world/model/bronchoscope/link/distal_tip/sensor/cmos_camera/image@sensor_msgs/msg/Image@gz.msgs.Image'],
        output='screen'
    )

    # 5. Spawn Controllers
    load_jsb = Node(package='controller_manager', executable='spawner', arguments=['joint_state_broadcaster'])
    load_pc = Node(package='controller_manager', executable='spawner', arguments=['position_controller'])

    return LaunchDescription([gz_sim, robot_state_publisher, spawn_robot, bridge, load_jsb, load_pc])
