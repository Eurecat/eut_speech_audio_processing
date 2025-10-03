from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    audio_capturing_node = Node(
        package="audio_stream_manager",
        executable="audio_capturing_node",
        name="audio_capturing_node",
        output="screen",
        parameters=[{"device_name": "Jabra SPEAK 510 USB: Audio (hw:1,0)"}],
    )

    audio_device_monitor_node = Node(
        package="audio_stream_manager",
        executable="audio_device_monitor_node",
        name="audio_device_monitor_node",
        output="screen",
    )

    return LaunchDescription(
        [
            audio_capturing_node,
            audio_device_monitor_node,
        ]
    )
