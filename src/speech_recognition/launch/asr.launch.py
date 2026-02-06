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

VENV_PATH = os.environ.get("AI_VENV", "/opt/ros_python_env")  # set AI_VENV or uses default


def _venv_site_packages(venv_path: str) -> str:
    py = os.path.join(venv_path, "bin", "python")
    return subprocess.check_output(
        [py, "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()


def _setup(context, *args, **kwargs):
    site_pkgs = _venv_site_packages(VENV_PATH)
    existing = os.environ.get("PYTHONPATH", "")
    new_py_path = site_pkgs if not existing else f"{site_pkgs}{os.pathsep}{existing}"

    # Get the path to the asr_params.yaml file
    config_dir = get_package_share_directory("speech_recognition")
    config_file = os.path.join(config_dir, "config", "asr_params.yaml")

    # Get ros4hri_with_id parameter
    ros4hri_with_id = LaunchConfiguration("ros4hri_with_id").perform(context).lower() == "true"
    cleanup_inactive_topics = (
        LaunchConfiguration("cleanup_inactive_topics").perform(context).lower() == "true"
    )
    inactive_topic_timeout = float(LaunchConfiguration("inactive_topic_timeout").perform(context))
    
    # Get compute_type parameter if provided
    compute_type = LaunchConfiguration("compute_type").perform(context)
    additional_params = {}
    if compute_type and compute_type != "compute_type":  # check if actually provided
        additional_params["compute_type"] = compute_type

    return [
        LogInfo(msg=f"[speech_recognition] Using AI venv: {VENV_PATH}"),
        LogInfo(msg=f"[speech_recognition] Injecting site-packages: {site_pkgs}"),
        LogInfo(msg=f"[speech_recognition] Loading ASR config from: {config_file}"),
        LogInfo(
            msg=f"[speech_recognition] ROS4HRI with ID: {'enabled' if ros4hri_with_id else 'disabled'}"
        ),
        SetEnvironmentVariable("PYTHONPATH", new_py_path),
        Node(
            package="speech_recognition",
            executable="asr_node",
            name="asr_node",
            output="screen",
            parameters=[
                config_file,
                {
                    "ros4hri_with_id": ros4hri_with_id,
                    "cleanup_inactive_topics": cleanup_inactive_topics,
                    "inactive_topic_timeout": inactive_topic_timeout,
                    **additional_params,
                },
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
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
                default_value="300.0",
                description="Timeout in seconds before destroying inactive topics",
            ),
            DeclareLaunchArgument(
                "compute_type",
                default_value="",
                description="Override compute type (float32, float16, int8_float32, int8, int16)",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
