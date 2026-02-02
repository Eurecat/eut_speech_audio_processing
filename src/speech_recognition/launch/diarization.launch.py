import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VENV_PATH = os.environ.get(
    "AI_VENV", "/opt/ros_python_diarization_env"
)  # set AI_VENV or uses default


def _venv_site_packages(venv_path: str) -> str:
    py = os.path.join(venv_path, "bin", "python")
    return subprocess.check_output(
        [py, "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()


def _setup(context, *args, **kwargs):
    site_pkgs = _venv_site_packages(VENV_PATH)
    existing = os.environ.get("PYTHONPATH", "")
    new_py_path = site_pkgs if not existing else f"{site_pkgs}{os.pathsep}{existing}"

    # Get the path to the diarization_params.yaml file
    config_dir = get_package_share_directory("speech_recognition")
    config_file = os.path.join(config_dir, "config", "diarization_params.yaml")

    # Get debug parameter
    enable_debug = LaunchConfiguration("enable_debug_output").perform(context).lower() == "true"

    # Get ros4hri_with_id parameter
    ros4hri_with_id = LaunchConfiguration("ros4hri_with_id").perform(context).lower() == "true"
    cleanup_inactive_topics = (
        LaunchConfiguration("cleanup_inactive_topics").perform(context).lower() == "true"
    )
    inactive_topic_timeout = float(LaunchConfiguration("inactive_topic_timeout").perform(context))

    # Prepare arguments - add debug log level if debug output is enabled
    node_arguments = []
    if enable_debug:
        # Set debug level only for this specific node, not globally
        node_arguments = ["--ros-args", "--log-level", "diarization_node:=debug"]

    return [
        LogInfo(msg=f"[speech_recognition] Using AI venv: {VENV_PATH}"),
        LogInfo(msg=f"[speech_recognition] Injecting site-packages: {site_pkgs}"),
        LogInfo(msg=f"[speech_recognition] Loading Diarization config from: {config_file}"),
        LogInfo(
            msg=f"[speech_recognition] Debug logging: {'enabled' if enable_debug else 'disabled'}"
        ),
        LogInfo(
            msg=f"[speech_recognition] ROS4HRI with ID: {'enabled' if ros4hri_with_id else 'disabled'}"
        ),
        SetEnvironmentVariable("PYTHONPATH", new_py_path),
        Node(
            package="speech_recognition",
            executable="diarization_node",
            name="diarization_node",
            output="screen",
            parameters=[
                config_file,
                {
                    "ros4hri_with_id": ros4hri_with_id,
                    "cleanup_inactive_topics": cleanup_inactive_topics,
                    "inactive_topic_timeout": inactive_topic_timeout,
                },
            ],
            arguments=node_arguments,
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_debug_output",
                default_value="false",
                description="Enable debug logging output for diarization node",
            ),
            DeclareLaunchArgument(
                "ros4hri_with_id",
                default_value="false",
                description="Enable ROS4HRI standard publishing with ID approach",
            ),
            DeclareLaunchArgument(
                "cleanup_inactive_topics",
                default_value="false",
                description="Destroy topics for inactive speakers/voices",
            ),
            DeclareLaunchArgument(
                "inactive_topic_timeout",
                default_value="10.0",
                description="Timeout in seconds before destroying inactive topics",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
