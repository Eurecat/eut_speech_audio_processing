import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo, OpaqueFunction, SetEnvironmentVariable
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

    # Get the path to the vad_params.yaml file
    config_dir = get_package_share_directory("speech_recognition")
    config_file = os.path.join(config_dir, "config", "vad_params.yaml")

    return [
        LogInfo(msg=f"[speech_recognition] Using AI venv: {VENV_PATH}"),
        LogInfo(msg=f"[speech_recognition] Injecting site-packages: {site_pkgs}"),
        LogInfo(msg=f"[speech_recognition] Loading VAD config from: {config_file}"),
        SetEnvironmentVariable("PYTHONPATH", new_py_path),
        Node(
            package="speech_recognition",
            executable="vad_node",
            name="vad_node",
            output="screen",
            parameters=[config_file],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_setup)])
