from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bind_host",
                default_value="0.0.0.0",
                description="Bind host for Android audio TCP receiver",
            ),
            DeclareLaunchArgument(
                "bind_port",
                default_value="17000",
                description="Bind port for Android audio TCP receiver",
            ),
            DeclareLaunchArgument(
                "default_samplerate",
                default_value="16000.0",
                description="Default sample rate used when payload omits sample_rate",
            ),
            Node(
                package="audio_stream_manager",
                executable="android_audio_bridge",
                name="android_audio_bridge",
                output="screen",
                parameters=[
                    {
                        "bind_host": LaunchConfiguration("bind_host"),
                        "bind_port": LaunchConfiguration("bind_port"),
                        "default_samplerate": LaunchConfiguration("default_samplerate"),
                        "topic_name": "audio_and_device_info",
                    }
                ],
            ),
        ]
    )
