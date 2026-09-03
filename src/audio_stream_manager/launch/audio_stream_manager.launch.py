import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VENV_PATH = os.environ.get("AI_VENV", "/opt/ros_python_env")  # set AI_VENV or uses default


def _venv_site_packages(venv_path: str) -> str:
    py = os.path.join(venv_path, "bin", "python")
    return subprocess.check_output(
        [py, "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()


def generate_launch_description():
    # Get config file
    config_dir = get_package_share_directory("audio_stream_manager")
    config_file = os.path.join(config_dir, "config", "audio_params.yaml")

    # Setup environment
    site_pkgs = _venv_site_packages(VENV_PATH)
    existing = os.environ.get("PYTHONPATH", "")
    new_py_path = site_pkgs if not existing else f"{site_pkgs}{os.pathsep}{existing}"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "device_name",
                default_value=os.environ.get("DEVICE_NAME", "jabra"),
                description="Name or partial name of the audio input device to use "
                "(defaults to the DEVICE_NAME env var, see Docker/.env)",
            ),
            LogInfo(msg=f"[audio_stream_manager] Using AI venv: {VENV_PATH}"),
            LogInfo(msg=f"[audio_stream_manager] Injecting site-packages: {site_pkgs}"),
            LogInfo(msg=f"[audio_stream_manager] Loading config from: {config_file}"),
            SetEnvironmentVariable("PYTHONPATH", new_py_path),
            Node(
                package="audio_stream_manager",
                executable="audio_capturing",
                name="audio_capturing",
                output="screen",
                parameters=[
                    config_file,
                    {"device_name": LaunchConfiguration("device_name")},
                ],
            ),
            Node(
                package="audio_stream_manager",
                executable="audio_to_mp3",
                name="audio_to_mp3",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
