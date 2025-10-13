from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    vad_node = Node(
        package="speech_recognition",
        executable="vad_node",
        name="vad_node",
        output="screen",
    )

    diarization_node = Node(
        package="speech_recognition",
        executable="diarization_node",
        name="diarization_node",
        output="screen",
    )

    asr_node = Node(
        package="speech_recognition",
        executable="asr_node",
        name="asr_node",
        output="screen",
    )

    return LaunchDescription(
        [
            vad_node,
            diarization_node,
            asr_node,
        ]
    )
